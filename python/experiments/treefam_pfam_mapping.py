#!/usr/bin/env python3
"""Map TreeFam families to Pfam domains and identify clean (non-training) families.

For each TreeFam family:
1. Extract Ensembl protein IDs from FASTA headers
2. Query Ensembl REST API for Pfam domain annotations
3. Check against our Pfam training split
4. Classify as clean/contaminated/unknown
"""

import json
import os
import re
import time
import glob
import sys
import urllib.request
import urllib.error

# Paths
TREEFAM_DIR = "/home/yam/bio-datasets/data/treefam/treefam_family_data"
PFAM_SPLITS = "/home/yam/bio-datasets/data/pfam-seed/splits/v1.json"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_FILE = os.path.join(OUTPUT_DIR, "treefam_pfam_mapping.json")
CLEAN_FILE = os.path.join(OUTPUT_DIR, "treefam_clean_families.json")
CACHE_FILE = os.path.join(OUTPUT_DIR, "treefam_pfam_cache.json")

# Rate limiting
REQUEST_DELAY = 0.07  # ~14 req/s, under 15/s limit


def load_pfam_splits():
    with open(PFAM_SPLITS) as f:
        splits = json.load(f)
    return set(splits["train"]), set(splits["test"])


def get_family_ids():
    """Get all TreeFam family IDs."""
    files = glob.glob(os.path.join(TREEFAM_DIR, "TF*.aa.fasta"))
    ids = sorted(set(re.match(r"(TF\d+)", os.path.basename(f)).group(1) for f in files))
    return ids


def parse_fasta_ids(family_id):
    """Extract protein IDs from FASTA headers. Return (ensembl_ids, all_ids, n_seqs, seq_lengths)."""
    fasta_path = os.path.join(TREEFAM_DIR, f"{family_id}.aa.fasta")
    if not os.path.exists(fasta_path):
        return [], [], 0, []

    ensembl_ids = []
    all_ids = []
    seq_lengths = []
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    seq = "".join(current_seq).replace("-", "")
                    seq_lengths.append(len(seq))
                    current_seq = []
                pid = line[1:].split()[0]
                all_ids.append(pid)
                # Match Ensembl protein IDs: ENS + optional species code + P + digits
                if re.match(r"ENS\w*P\d+", pid):
                    ensembl_ids.append(pid)
            else:
                current_seq.append(line)
        if current_seq:
            seq = "".join(current_seq).replace("-", "")
            seq_lengths.append(len(seq))

    return ensembl_ids, all_ids, len(all_ids), seq_lengths


def query_ensembl_pfam(protein_id, retries=3):
    """Query Ensembl REST API for Pfam annotations of a protein."""
    url = f"https://rest.ensembl.org/overlap/translation/{protein_id}?feature=protein_feature;type=Pfam"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                pfam_ids = set()
                for feature in data:
                    if "type" in feature and feature["type"] == "Pfam":
                        # The ID field contains the Pfam accession
                        if "id" in feature:
                            pfam_ids.add(feature["id"])
                return sorted(pfam_ids), None

        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited - back off
                retry_after = float(e.headers.get("Retry-After", 1))
                print(f"  Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
                continue
            elif e.code == 400:
                return [], f"bad_request_{e.code}"
            elif e.code == 404:
                return [], "not_found"
            else:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                return [], f"http_error_{e.code}"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            return [], f"error_{type(e).__name__}"

    return [], "max_retries"


def check_tree_file(family_id):
    """Check if family has a valid tree file."""
    tree_path = os.path.join(TREEFAM_DIR, f"{family_id}.nh.emf")
    return os.path.exists(tree_path)


def main():
    train_pfams, test_pfams = load_pfam_splits()
    print(f"Training Pfam families: {len(train_pfams)}")
    print(f"Test Pfam families: {len(test_pfams)}")

    family_ids = get_family_ids()
    print(f"Total TreeFam families: {len(family_ids)}")

    # Load cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached results")
    else:
        cache = {}

    results = {}
    n_queried = 0
    n_cached = 0
    n_api_calls = 0

    for i, fam_id in enumerate(family_ids):
        ensembl_ids, all_ids, n_seqs, seq_lengths = parse_fasta_ids(fam_id)
        has_tree = check_tree_file(fam_id)

        # Check cache first
        if fam_id in cache:
            result = cache[fam_id]
            # Update metadata that doesn't require API
            result["n_seqs"] = n_seqs
            result["seq_lengths"] = seq_lengths
            result["has_tree"] = has_tree
            # Re-classify with current split
            pfam_domains = result.get("pfam_domains", [])
            if not pfam_domains and result.get("query_status") in ("no_ensembl_id", "not_found", None):
                result["status"] = "unknown"
            elif any(p in train_pfams for p in pfam_domains):
                result["status"] = "contaminated"
            else:
                result["status"] = "clean"
            results[fam_id] = result
            n_cached += 1
            continue

        # Pick first Ensembl ID to query
        if not ensembl_ids:
            results[fam_id] = {
                "ensembl_id": None,
                "pfam_domains": [],
                "status": "unknown",
                "query_status": "no_ensembl_id",
                "n_seqs": n_seqs,
                "seq_lengths": seq_lengths,
                "has_tree": has_tree,
            }
            cache[fam_id] = results[fam_id]
            continue

        # Query API with first Ensembl ID
        ens_id = ensembl_ids[0]
        pfam_domains, error = query_ensembl_pfam(ens_id)
        n_api_calls += 1
        time.sleep(REQUEST_DELAY)

        # If first ID fails, try a few more
        if error and len(ensembl_ids) > 1:
            for alt_id in ensembl_ids[1:3]:  # Try up to 2 more
                pfam_domains, error2 = query_ensembl_pfam(alt_id)
                n_api_calls += 1
                time.sleep(REQUEST_DELAY)
                if not error2:
                    ens_id = alt_id
                    error = None
                    break

        # Classify
        if error:
            status = "unknown"
        elif not pfam_domains:
            status = "unknown"  # No Pfam annotations found
        elif any(p in train_pfams for p in pfam_domains):
            status = "contaminated"
        else:
            status = "clean"

        results[fam_id] = {
            "ensembl_id": ens_id,
            "pfam_domains": pfam_domains,
            "status": status,
            "query_status": error or "ok",
            "n_seqs": n_seqs,
            "seq_lengths": seq_lengths,
            "has_tree": has_tree,
        }
        cache[fam_id] = results[fam_id]
        n_queried += 1

        # Save cache periodically
        if n_api_calls % 100 == 0:
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)
            # Print progress
            n_clean = sum(1 for r in results.values() if r["status"] == "clean")
            n_contam = sum(1 for r in results.values() if r["status"] == "contaminated")
            n_unknown = sum(1 for r in results.values() if r["status"] == "unknown")
            print(
                f"[{i+1}/{len(family_ids)}] API calls: {n_api_calls}, "
                f"clean: {n_clean}, contaminated: {n_contam}, unknown: {n_unknown}"
            )

    # Final save of cache
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

    # Save full mapping
    with open(MAPPING_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    n_clean = sum(1 for r in results.values() if r["status"] == "clean")
    n_contam = sum(1 for r in results.values() if r["status"] == "contaminated")
    n_unknown = sum(1 for r in results.values() if r["status"] == "unknown")
    print(f"\nFinal summary:")
    print(f"  Clean: {n_clean}")
    print(f"  Contaminated: {n_contam}")
    print(f"  Unknown: {n_unknown}")
    print(f"  API calls: {n_api_calls}")
    print(f"  Cached: {n_cached}")

    # Select clean families meeting criteria
    clean_families = []
    for fam_id, info in sorted(results.items()):
        if info["status"] != "clean":
            continue
        if not info["has_tree"]:
            continue
        n = info["n_seqs"]
        if n < 5 or n > 15:
            continue
        lengths = info["seq_lengths"]
        if not lengths:
            continue
        # Check median sequence length is in range
        sorted_lens = sorted(lengths)
        median_len = sorted_lens[len(sorted_lens) // 2]
        if median_len < 50 or median_len > 500:
            continue
        clean_families.append(fam_id)

    # Also include "unknown" families (no Pfam annotation = likely safe)
    # that meet the criteria
    unknown_families = []
    for fam_id, info in sorted(results.items()):
        if info["status"] != "unknown":
            continue
        if not info["has_tree"]:
            continue
        n = info["n_seqs"]
        if n < 5 or n > 15:
            continue
        lengths = info["seq_lengths"]
        if not lengths:
            continue
        sorted_lens = sorted(lengths)
        median_len = sorted_lens[len(sorted_lens) // 2]
        if median_len < 50 or median_len > 500:
            continue
        unknown_families.append(fam_id)

    print(f"\nClean families meeting criteria (5-15 leaves, 50-500 aa): {len(clean_families)}")
    print(f"Unknown families meeting criteria: {len(unknown_families)}")

    # If we have enough clean families, use those; otherwise supplement with unknown
    if len(clean_families) >= 1000:
        selected = clean_families[:1000]
    else:
        # Supplement with unknown families
        needed = 1000 - len(clean_families)
        selected = clean_families + unknown_families[:needed]
        print(f"Supplementing {len(clean_families)} clean with {min(needed, len(unknown_families))} unknown")

    print(f"Total selected: {len(selected)}")

    with open(CLEAN_FILE, "w") as f:
        json.dump(selected, f, indent=2)

    print(f"\nSaved mapping to {MAPPING_FILE}")
    print(f"Saved clean families to {CLEAN_FILE}")


if __name__ == "__main__":
    main()
