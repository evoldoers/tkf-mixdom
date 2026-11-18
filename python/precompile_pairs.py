#!/usr/bin/env python3
"""
precompile_pairs.py — Pre-compile Pfam cherry pairs into fast binary format.

Scans Pfam Stockholm MSAs, selects cherry pairs, encodes them as compact
X/A/Y records (ancestor seq, RLE alignment path, descendant seq), and writes
shuffled shards of zstd-compressed JSONL files.

This eliminates Stockholm parsing and cherry selection at training time,
shifting the bottleneck from CPU overhead to GPU computation.

Output structure:
    <out>/
        manifest.json        # metadata: n_pairs, n_families, shard_files, ...
        shard_0000.jsonl.zst # zstd-compressed JSON lines
        shard_0001.jsonl.zst
        ...

Usage:
    # Quick test on 3 families:
    python precompile_pairs.py \\
        --msa-dir pfam/ --families PF00001,PF00002,PF00003 \\
        --out /tmp/test_precompiled/

    # Full Pfam with split:
    python precompile_pairs.py \\
        --msa-dir ~/bio-datasets/data/pfam-seed/ \\
        --split train --split-file ~/bio-datasets/data/pfam-seed/splits/v1.json \\
        --out pfam/precompiled/ \\
        --max-seqs-per-family 500 \\
        --workers 4
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_log_start = time.monotonic()


def _log(msg, end='\n'):
    elapsed = time.monotonic() - _log_start
    sys.stderr.write(f"[{elapsed:8.1f}s] {msg}" + end)
    sys.stderr.flush()


# ============================================================
# Import functions from train_pfam.py (duplicated to avoid
# importing JAX and its heavyweight deps)
# ============================================================

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA = len(AMINO_ACIDS)
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}


def parse_stockholm(filepath):
    """Parse Stockholm alignment. Returns list of (name, aligned_seq)."""
    import gzip
    names = []
    seqs = {}
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                name, seq = parts[0], parts[1]
                if name in seqs:
                    seqs[name] += seq
                else:
                    names.append(name)
                    seqs[name] = seq
    return names, [seqs[n] for n in names]


def ungap(seq):
    return ''.join(c for c in seq if c in AA_TO_IDX)


def p_distance(seq1, seq2):
    matches = mismatches = 0
    for a, b in zip(seq1, seq2):
        if a in AA_TO_IDX and b in AA_TO_IDX:
            if a == b:
                matches += 1
            else:
                mismatches += 1
    total = matches + mismatches
    return mismatches / total if total > 0 else 1.0


def pdist_to_evo_time(pdist):
    if pdist >= 0.95:
        return 5.0
    corrected = 1.0 - pdist * (AA / (AA - 1.0))
    if corrected <= 0.01:
        return 5.0
    return -np.log(corrected)


def _aa_composition(seq):
    comp = np.zeros(AA)
    for c in seq:
        if c in AA_TO_IDX:
            comp[AA_TO_IDX[c]] += 1
    total = comp.sum()
    if total > 0:
        comp /= total
    return comp


def select_cherries(n_seqs, dist_fn, max_pairs=None):
    """Greedy nearest-neighbor cherry pairing."""
    if n_seqs < 2:
        return []
    dists = np.ones((n_seqs, n_seqs)) * 1e10
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            d = dist_fn(i, j)
            dists[i, j] = d
            dists[j, i] = d
    paired = set()
    pairs = []
    while len(paired) < n_seqs - 1:
        best_d = 1e10
        best_i, best_j = -1, -1
        for i in range(n_seqs):
            if i in paired:
                continue
            for j in range(i + 1, n_seqs):
                if j in paired:
                    continue
                if dists[i, j] < best_d:
                    best_d = dists[i, j]
                    best_i, best_j = i, j
        if best_i < 0:
            break
        pairs.append((best_i, best_j))
        paired.add(best_i)
        paired.add(best_j)
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


def _aligned_pair_to_int_arrays(seq1, seq2):
    """Convert string pair to gapped integer arrays (-1 = gap)."""
    a = np.array([AA_TO_IDX.get(c, -1) for c in seq1], dtype=np.int32)
    b = np.array([AA_TO_IDX.get(c, -1) for c in seq2], dtype=np.int32)
    mask = (a >= 0) | (b >= 0)
    return a[mask], b[mask]


def alignment_to_states_pure(ancestor_chars, descendant_chars, gap_token=-1):
    """Convert aligned sequences to state sequence (pure Python, no JAX).

    Returns:
        states: list of state codes (1=M, 2=I, 3=D)
        anc_chars: list of ancestor characters (for M, D states)
        desc_chars: list of descendant characters (for M, I states)
    """
    M, I, D = 1, 2, 3
    states = []
    anc_chars = []
    desc_chars = []
    for pos in range(len(ancestor_chars)):
        anc = int(ancestor_chars[pos])
        desc = int(descendant_chars[pos])
        has_anc = (anc != gap_token)
        has_desc = (desc != gap_token)
        if has_anc and has_desc:
            states.append(M)
            anc_chars.append(anc)
            desc_chars.append(desc)
        elif has_desc:
            states.append(I)
            desc_chars.append(desc)
        elif has_anc:
            states.append(D)
            anc_chars.append(anc)
    return states, anc_chars, desc_chars


# ============================================================
# Per-family processing
# ============================================================

def _process_family(msa_file, max_seqs=500):
    """Process one MSA file: parse, select cherries, encode pairs.

    Returns list of record dicts.
    """
    from tkfmixdom.jax.util.pair_format import encode_pair

    fam = os.path.basename(msa_file).split('.')[0]
    names, seqs = parse_stockholm(msa_file)
    n = len(seqs)
    if n < 2:
        return []

    # Cap sequences for cherry selection
    max_for_cherries = min(n, max_seqs)
    if n > max_for_cherries:
        rng = np.random.RandomState(hash(fam) & 0x7FFFFFFF)
        subset = sorted(rng.choice(n, max_for_cherries, replace=False))
        seqs_sub = [seqs[i] for i in subset]
        idx_map = list(subset)
    else:
        seqs_sub = seqs
        idx_map = list(range(n))

    n_sub = len(seqs_sub)

    if n_sub <= 50:
        def dist_fn(i, j, _seqs=seqs_sub):
            return p_distance(_seqs[i], _seqs[j])
    else:
        comps = np.array([_aa_composition(ungap(s)) for s in seqs_sub])
        def dist_fn(i, j, _c=comps):
            return float(np.sum((_c[i] - _c[j]) ** 2))

    cherries = select_cherries(n_sub, dist_fn)
    records = []

    for idx1, idx2 in cherries:
        orig1, orig2 = idx_map[idx1], idx_map[idx2]
        aln_i, aln_j = _aligned_pair_to_int_arrays(seqs[orig1], seqs[orig2])
        if len(aln_i) == 0:
            continue

        pd = p_distance(seqs[orig1], seqs[orig2])
        t_est = pdist_to_evo_time(pd)

        states, anc_chars, desc_chars = alignment_to_states_pure(aln_i, aln_j)
        if not states:
            continue

        # Encode both directions for reversibility
        for ac, dc, row1_name, row2_name in [
            (anc_chars, desc_chars, names[orig1], names[orig2]),
            (desc_chars, anc_chars, names[orig2], names[orig1]),
        ]:
            # For reverse direction, recompute states
            if ac is desc_chars:
                # Swap: ancestor becomes descendant and vice versa
                rev_states, rev_anc, rev_desc = alignment_to_states_pure(
                    aln_j, aln_i)
                rec = encode_pair(rev_anc, rev_desc, rev_states, t_est, fam,
                                  row1_name, row2_name)
            else:
                rec = encode_pair(ac, dc, states, t_est, fam,
                                  row1_name, row2_name)
            records.append(rec)

    return records


# ============================================================
# Shard writing
# ============================================================

def _write_one_shard(records, out_dir, shard_idx):
    """Write one shard of records as zstd-compressed JSONL."""
    import zstandard as zstd

    os.makedirs(out_dir, exist_ok=True)
    shard_name = f'shard_{shard_idx:04d}.jsonl.zst'
    shard_path = os.path.join(out_dir, shard_name)

    # Shuffle within shard for diversity
    rng = np.random.RandomState(shard_idx)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    cctx = zstd.ZstdCompressor(level=3)
    lines = [json.dumps(records[i], separators=(',', ':')) for i in indices]
    data = '\n'.join(lines).encode('utf-8')
    compressed = cctx.compress(data)

    with open(shard_path, 'wb') as f:
        f.write(compressed)

    return len(compressed)


def _write_partial_manifest(out_dir, shard_files, n_pairs, n_families, n_processed):
    """Write a partial manifest so data is usable before completion."""
    manifest = {
        'n_pairs': n_pairs,
        'n_families': n_families,
        'n_shards': len(shard_files),
        'shard_files': list(shard_files),
        'complete': False,
        'families_processed': n_processed,
    }
    path = os.path.join(out_dir, 'manifest.json')
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pre-compile Pfam cherry pairs into fast binary format')
    parser.add_argument('--msa-dir', required=True,
                        help='Directory containing *.sto / *.sto.gz files')
    parser.add_argument('--out', required=True,
                        help='Output directory for shards')
    parser.add_argument('--families', default=None,
                        help='Comma-separated family accessions (default: all)')
    parser.add_argument('--split', default=None, choices=['train', 'val', 'test'],
                        help='Use only families in this split')
    parser.add_argument('--split-file', default=None,
                        help='Path to split JSON file')
    parser.add_argument('--max-seqs-per-family', type=int, default=500,
                        help='Max sequences to consider per family (default: 500)')
    parser.add_argument('--pairs-per-shard', type=int, default=3000,
                        help='Pairs per shard file (default: 3000)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')

    args = parser.parse_args()
    msa_dir = args.msa_dir

    # ---- Find MSA files ----
    import glob as glob_mod
    if args.families:
        family_list = [f.strip() for f in args.families.split(',')]
        msa_files = []
        for fam in family_list:
            for ext in ['.sto', '.sto.gz', '.stockholm', '.stockholm.gz']:
                p = os.path.join(msa_dir, fam + ext)
                if os.path.exists(p):
                    msa_files.append(p)
                    break
            else:
                _log(f"Warning: {fam} not found in {msa_dir}")
    else:
        msa_files = sorted(
            glob_mod.glob(os.path.join(msa_dir, '*.sto')) +
            glob_mod.glob(os.path.join(msa_dir, '*.sto.gz')))

    # ---- Apply split filter ----
    if args.split:
        split_file = args.split_file
        if not split_file:
            for candidate in [
                os.path.expanduser(
                    '~/bio-datasets/data/pfam-seed/splits/v1.json'),
                os.path.join(msa_dir, 'splits', 'v1.json'),
            ]:
                if os.path.exists(candidate):
                    split_file = candidate
                    break

        if split_file and os.path.exists(split_file):
            with open(split_file) as f:
                split_data = json.load(f)
            split_fams = set(split_data.get(args.split, []))
            msa_files = [f for f in msa_files
                         if os.path.basename(f).split('.')[0] in split_fams]
            _log(f"Split '{args.split}' from {split_file}: "
                 f"{len(msa_files)} families")
        else:
            _log(f"ERROR: --split requires a valid --split-file")
            sys.exit(1)

    if not msa_files:
        _log("No MSA files found")
        sys.exit(1)

    _log(f"Processing {len(msa_files)} MSA files")

    # ---- Process families (incremental shard writing) ----
    all_records = []  # buffer, flushed every pairs_per_shard
    families_processed = 0
    families_with_pairs = 0
    shard_files = []
    shard_idx = 0
    total_pairs_written = 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_process_family, f, args.max_seqs_per_family): f
                for f in msa_files
            }
            for future in as_completed(futures):
                msa_file = futures[future]
                fam = os.path.basename(msa_file).split('.')[0]
                try:
                    records = future.result()
                    if records:
                        all_records.extend(records)
                        families_with_pairs += 1
                    families_processed += 1
                    if families_processed % max(1, len(msa_files) // 20) == 0:
                        _log(f"  {families_processed}/{len(msa_files)} families, "
                             f"{len(all_records)} pairs")
                except Exception as e:
                    _log(f"  ERROR processing {fam}: {e}")
                    families_processed += 1
    else:
        for msa_file in msa_files:
            fam = os.path.basename(msa_file).split('.')[0]
            try:
                records = _process_family(msa_file, args.max_seqs_per_family)
                if records:
                    all_records.extend(records)
                    families_with_pairs += 1
                families_processed += 1

                # Flush shard incrementally when buffer is full
                if len(all_records) >= args.pairs_per_shard:
                    shard_batch = all_records[:args.pairs_per_shard]
                    all_records = all_records[args.pairs_per_shard:]
                    _write_one_shard(shard_batch, args.out, shard_idx)
                    shard_files.append(f'shard_{shard_idx:04d}.jsonl.zst')
                    shard_idx += 1
                    total_pairs_written += len(shard_batch)
                    # Update manifest so partial results are usable
                    _write_partial_manifest(
                        args.out, shard_files, total_pairs_written,
                        families_with_pairs, families_processed)

                if families_processed % max(1, len(msa_files) // 20) == 0:
                    _log(f"  {families_processed}/{len(msa_files)} families, "
                         f"{total_pairs_written + len(all_records)} pairs, "
                         f"{len(shard_files)} shards written")
            except Exception as e:
                _log(f"  ERROR processing {fam}: {e}")
                families_processed += 1

    # Flush remaining records as final shard
    if all_records:
        _write_one_shard(all_records, args.out, shard_idx)
        shard_files.append(f'shard_{shard_idx:04d}.jsonl.zst')
        total_pairs_written += len(all_records)
        all_records = []

    _log(f"Total: {total_pairs_written} pairs from "
         f"{families_with_pairs}/{families_processed} families, "
         f"{len(shard_files)} shards")

    if total_pairs_written == 0:
        _log("No pairs found, exiting")
        sys.exit(1)

    # ---- Final manifest ----
    _write_partial_manifest(args.out, shard_files, total_pairs_written,
                            families_with_pairs, families_processed)
    # Mark as complete
    manifest = {
        'n_pairs': total_pairs_written,
        'n_families': families_with_pairs,
        'n_shards': len(shard_files),
        'shard_files': shard_files,
        'pairs_per_shard': args.pairs_per_shard,
        'complete': True,
        'source': {
            'msa_dir': os.path.abspath(args.msa_dir),
            'split': args.split,
            'max_seqs_per_family': args.max_seqs_per_family,
        },
    }

    manifest_path = os.path.join(args.out, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    _log(f"Done: {total_pairs_written} pairs in {len(shard_files)} shards")
    _log(f"Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
