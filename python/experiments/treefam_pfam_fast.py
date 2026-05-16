#!/usr/bin/env python3
"""Fast TreeFam→Pfam mapping: query 1 Ensembl protein per family.

Queries the Ensembl REST API for Pfam domain annotations, checks against
our Pfam training split, and classifies each TreeFam family as
clean/contaminated/unknown.
"""

import json
import os
import re
import time
import glob
import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.stdout.reconfigure(line_buffering=True)

TREEFAM_DIR = "/home/yam/bio-datasets/data/treefam/treefam_family_data"
PFAM_SPLITS = "/home/yam/bio-datasets/data/pfam-seed/splits/v1.json"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SCRIPT_DIR, "treefam_pfam_cache.json")
MAPPING_FILE = os.path.join(SCRIPT_DIR, "treefam_pfam_mapping.json")
CLEAN_FILE = os.path.join(SCRIPT_DIR, "treefam_clean_families.json")

REQUEST_DELAY = 0.0  # API takes ~0.5s/req, well under 15/s limit


def load_splits():
    with open(PFAM_SPLITS) as f:
        s = json.load(f)
    return set(s["train"]), set(s["test"])


def get_families():
    files = sorted(glob.glob(os.path.join(TREEFAM_DIR, "TF*.aa.fasta")))
    return [os.path.basename(f).replace(".aa.fasta", "") for f in files]


def parse_fasta(fam_id):
    """Return (first_ensembl_id, n_seqs, seq_lengths)."""
    path = os.path.join(TREEFAM_DIR, f"{fam_id}.aa.fasta")
    ens_id = None
    n_seqs = 0
    lengths = []
    cur_seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur_seq:
                    lengths.append(len("".join(cur_seq).replace("-", "")))
                    cur_seq = []
                n_seqs += 1
                pid = line[1:].split()[0]
                if ens_id is None and re.match(r"ENS\w*P\d+", pid):
                    ens_id = pid
            else:
                cur_seq.append(line)
        if cur_seq:
            lengths.append(len("".join(cur_seq).replace("-", "")))

    return ens_id, n_seqs, lengths


CONCURRENCY = 10  # Ensembl allows 15 req/s; use 10 threads

# Thread-local sessions for connection pooling
_thread_local = threading.local()


def get_session():
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503])
        adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
        session.mount("https://", adapter)
        session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        _thread_local.session = session
    return _thread_local.session


def query_pfam(protein_id):
    """Query Ensembl for Pfam domains. Returns (pfam_ids, error)."""
    session = get_session()
    url = f"https://rest.ensembl.org/overlap/translation/{protein_id}?feature=protein_feature;type=Pfam"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 2))
            time.sleep(retry_after)
            resp = session.get(url, timeout=15)
        if resp.status_code == 400:
            return [], "http_400"
        if resp.status_code == 404:
            return [], "http_404"
        resp.raise_for_status()
        data = resp.json()
        pfams = sorted(set(d["id"] for d in data if d.get("type") == "Pfam"))
        return pfams, None
    except requests.exceptions.HTTPError as e:
        return [], f"http_{e.response.status_code if e.response else 'unknown'}"
    except Exception as e:
        return [], str(type(e).__name__)


def main():
    train_pfams, test_pfams = load_splits()
    families = get_families()
    print(f"Train Pfams: {len(train_pfams)}, Test Pfams: {len(test_pfams)}")
    print(f"TreeFam families: {len(families)}")

    # Load cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
    print(f"Cached: {len(cache)}")

    # Prepare work items: families needing API queries
    need_query = []  # (fam_id, ens_id, n_seqs, lengths, has_tree)
    for fam in families:
        if fam in cache:
            continue
        ens_id, n_seqs, lengths = parse_fasta(fam)
        has_tree = os.path.exists(os.path.join(TREEFAM_DIR, f"{fam}.nh.emf"))
        if not ens_id:
            cache[fam] = {
                "ensembl_id": None,
                "pfam_domains": [],
                "status": "unknown",
                "query_status": "no_ensembl_id",
                "n_seqs": n_seqs,
                "seq_lengths": lengths,
                "has_tree": has_tree,
            }
        else:
            need_query.append((fam, ens_id, n_seqs, lengths, has_tree))

    print(f"Need API queries: {len(need_query)}")

    n_api = 0
    t0 = time.time()

    def process_family(item):
        fam, ens_id, n_seqs, lengths, has_tree = item
        pfams, error = query_pfam(ens_id)
        return fam, ens_id, pfams, error, n_seqs, lengths, has_tree

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {}
        batch_start = 0
        BATCH_SIZE = 500

        while batch_start < len(need_query):
            batch_end = min(batch_start + BATCH_SIZE, len(need_query))
            batch = need_query[batch_start:batch_end]

            for item in batch:
                fut = executor.submit(process_family, item)
                futures[fut] = item[0]

            for fut in as_completed(futures):
                fam, ens_id, pfams, error, n_seqs, lengths, has_tree = fut.result()
                n_api += 1

                if error:
                    status = "unknown"
                    qs = error
                elif not pfams:
                    status = "clean_nopfam"
                    qs = "ok_nopfam"
                elif any(p in train_pfams for p in pfams):
                    status = "contaminated"
                    qs = "ok"
                else:
                    status = "clean"
                    qs = "ok"

                cache[fam] = {
                    "ensembl_id": ens_id,
                    "pfam_domains": pfams,
                    "status": status,
                    "query_status": qs,
                    "n_seqs": n_seqs,
                    "seq_lengths": lengths,
                    "has_tree": has_tree,
                }

            futures.clear()
            batch_start = batch_end

            # Save and report progress
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)
            elapsed = time.time() - t0
            rate = n_api / elapsed if elapsed > 0 else 0
            remaining = len(need_query) - n_api
            eta_min = remaining / rate / 60 if rate > 0 else 0
            counts = {}
            for v in cache.values():
                s = v["status"]
                counts[s] = counts.get(s, 0) + 1
            print(
                f"[{len(cache)}/{len(families)}] "
                f"API: {n_api}/{len(need_query)} ({rate:.1f}/s) ETA: {eta_min:.0f}min "
                f"{counts}"
            )

    # Final save
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

    # Reclassify all entries with current splits (in case splits changed)
    results = {}
    for fam in families:
        if fam not in cache:
            continue
        r = dict(cache[fam])
        # Update metadata
        if "n_seqs" not in r:
            _, n_seqs, lengths = parse_fasta(fam)
            r["n_seqs"] = n_seqs
            r["seq_lengths"] = lengths
        if "has_tree" not in r:
            r["has_tree"] = os.path.exists(os.path.join(TREEFAM_DIR, f"{fam}.nh.emf"))
        # Reclassify
        pfams = r.get("pfam_domains", [])
        qs = r.get("query_status", "")
        if qs in ("no_ensembl_id",) or (not pfams and qs not in ("ok", "ok_nopfam")):
            r["status"] = "unknown"
        elif not pfams:
            r["status"] = "clean_nopfam"
        elif any(p in train_pfams for p in pfams):
            r["status"] = "contaminated"
        else:
            r["status"] = "clean"
        results[fam] = r

    with open(MAPPING_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    counts = {}
    for v in results.values():
        s = v["status"]
        counts[s] = counts.get(s, 0) + 1
    print(f"\nSummary ({len(results)} families):")
    for s, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}")
    print(f"  API calls this run: {n_api}")

    # Select clean test families
    clean = []
    for fam in sorted(results):
        r = results[fam]
        if r["status"] not in ("clean", "clean_nopfam"):
            continue
        if not r.get("has_tree"):
            continue
        n = r.get("n_seqs", 0)
        if n < 5 or n > 15:
            continue
        lens = r.get("seq_lengths", [])
        if not lens:
            continue
        med = sorted(lens)[len(lens) // 2]
        if med < 50 or med > 500:
            continue
        clean.append(fam)

    # Also collect unknown families meeting criteria (likely safe)
    unknown = []
    for fam in sorted(results):
        r = results[fam]
        if r["status"] != "unknown":
            continue
        if not r.get("has_tree"):
            continue
        n = r.get("n_seqs", 0)
        if n < 5 or n > 15:
            continue
        lens = r.get("seq_lengths", [])
        if not lens:
            continue
        med = sorted(lens)[len(lens) // 2]
        if med < 50 or med > 500:
            continue
        unknown.append(fam)

    print(f"\nClean families (5-15 leaves, 50-500 aa): {len(clean)}")
    print(f"Unknown families meeting criteria: {len(unknown)}")

    if len(clean) >= 1000:
        selected = clean[:1000]
    else:
        selected = clean + unknown[: 1000 - len(clean)]
        print(f"Using {len(clean)} clean + {min(1000-len(clean), len(unknown))} unknown")

    print(f"Selected: {len(selected)}")

    with open(CLEAN_FILE, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\nSaved: {MAPPING_FILE}")
    print(f"Saved: {CLEAN_FILE}")


if __name__ == "__main__":
    main()
