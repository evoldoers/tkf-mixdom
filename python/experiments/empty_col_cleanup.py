#!/usr/bin/env python3
"""Identify benchmark families with empty columns in the spec-pruned MSA,
delete their entries from existing MixDom/TKF92 result JSONs, and emit a
per-method re-run command list.

Empty columns (cols where no kept-spec leaf has a residue) are an artifact
of xhard's leaf subsampling. They produce trivial absence predictions that
contribute zero to F1 and a nominally constant amount to logp_target /
logp_true. The empirical ablation showed F1 unchanged for parsimony /
gap-aware Felsenstein methods and ~1% drift for d3f1-VBEM.

This cleanup makes the benchmarks deterministic with respect to the
empty-column count — fels21/fels40/load_pfam_family already strip empty
cols at load-time. Affected MixDom / TKF92 method results need to be
re-run on the cleaned-up MSAs.

Usage:
    cd python && uv run python experiments/empty_col_cleanup.py [--dry-run]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tkfmixdom.jax.util.io import AA_TO_INT
from experiments.ancrec_benchmark import parse_sto


SPECS = [
    ('short', 'experiments/unified_benchmark_test_spec.json'),
    ('long',  'experiments/unified_benchmark_long_test_spec.json'),
    ('hard',  'experiments/unified_benchmark_hard_test_spec.json'),
    ('xhard', 'experiments/unified_benchmark_xhard_test_spec.json'),
]

# Methods whose results need re-running on cleaned MSAs. Maps method label
# to result-JSON pattern (with {label} = the spec's short name).
METHODS_TO_CLEAN = {
    'varanc-tkf92':       'experiments/varanc_presence_unified_{label}_test.json',
    'd3f1':               'experiments/varanc_presence_mixdom_unified_{label}_test.json',
    'd3f1-vbem-run6-it4': 'experiments/varanc_presence_mixdom_vbem_run6_iter4_unified_{label}_test.json',
}


def find_families_with_empty_cols(spec_path):
    """Return list of family names that have ≥1 empty column in the
    kept-leaf submatrix."""
    with open(spec_path) as f:
        spec = json.load(f)
    pfam_dir = os.path.expanduser(spec['pfam_dir'])
    affected = []
    n_empty_per_fam = {}
    for entry in spec['families']:
        fam = entry['family']
        leaves = list(set(entry.get('remaining', [])) | {entry['held_out']})
        sto = os.path.join(pfam_dir, fam + '.sto')
        if not os.path.exists(sto):
            continue
        try:
            seqs = parse_sto(sto)
        except Exception:
            continue
        rows = [seqs[n] for n in leaves if n in seqs]
        if not rows:
            continue
        L = max(len(r) for r in rows)
        rows = [r.ljust(L, '-') for r in rows]
        n_empty = 0
        for j in range(L):
            col = [r[j] for r in rows]
            has_aa = any(ch.isalpha() and ch.upper()
                          in 'ACDEFGHIKLMNPQRSTVWY' for ch in col)
            if not has_aa:
                n_empty += 1
        if n_empty > 0:
            affected.append(fam)
            n_empty_per_fam[fam] = n_empty
    return affected, n_empty_per_fam


def strip_affected_entries(json_path, affected_fams):
    """Load result JSON, remove entries whose family is in affected_fams,
    save back. Returns (n_before, n_after, n_deleted)."""
    if not os.path.exists(json_path):
        return None, None, 0
    if os.path.getsize(json_path) == 0:
        return None, None, 0  # truncated / empty
    try:
        with open(json_path) as f:
            d = json.load(f)
    except json.JSONDecodeError:
        return None, None, 0
    if 'results' in d:
        before = len(d['results'])
        d['results'] = [r for r in d['results']
                          if r.get('family') not in affected_fams]
        after = len(d['results'])
        d['n_families_kept'] = after
    else:
        return None, None, 0
    return before, after, before - after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be deleted but do not modify files.')
    parser.add_argument('--out-affected-list',
                        default='experiments/empty_col_affected_families.json')
    args = parser.parse_args()

    repo_python = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    affected_per_spec = {}
    print('=== Identifying affected families per spec ===')
    for label, rel_path in SPECS:
        path = os.path.join(repo_python, rel_path)
        affected, n_empty = find_families_with_empty_cols(path)
        affected_per_spec[label] = {
            'spec': rel_path,
            'affected': affected,
            'n_empty_per_family': n_empty,
        }
        print(f'  {label:<6} {len(affected):>3} affected families '
              f'(median {int(np.median(list(n_empty.values()))) if n_empty else 0} '
              f'empty cols, max {max(n_empty.values()) if n_empty else 0})')

    out = {
        'description': 'Families with ≥1 empty column in the kept-leaf '
            'submatrix (xhard subsamples can leave columns whose residues '
            'lived only in dropped/subsampled leaves). Loader strips '
            'these at load time; existing result JSONs need to be re-run '
            'for the listed families on MixDom/TKF92 methods.',
        'affected_per_spec': affected_per_spec,
    }
    with open(os.path.join(repo_python, args.out_affected_list), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote affected families list to {args.out_affected_list}')

    print('\n=== Deleting affected entries from result JSONs ===')
    total_deleted = 0
    rerun_needed = []
    for spec_label, info in affected_per_spec.items():
        affected_set = set(info['affected'])
        if not affected_set:
            continue
        for method, pattern in METHODS_TO_CLEAN.items():
            rel = pattern.format(label=spec_label)
            json_path = os.path.join(repo_python, rel)
            if not os.path.exists(json_path):
                print(f'  [skip] {method:<22} on {spec_label:<6} '
                      f'— no result file at {rel}')
                continue
            before, after, deleted = strip_affected_entries(
                json_path, affected_set)
            if before is None:
                print(f'  [malformed] {method:<22} on {spec_label:<6} {rel}')
                continue
            if args.dry_run:
                print(f'  [dry-run] would delete {deleted}/{before} entries '
                      f'from {method:<22} on {spec_label}')
            else:
                # Read full content, modify in memory, write back.
                with open(json_path) as f:
                    d = json.load(f)
                d['results'] = [r for r in d.get('results', [])
                                  if r.get('family') not in affected_set]
                d['n_families'] = len(d['results'])
                with open(json_path, 'w') as f:
                    json.dump(d, f, indent=2)
                print(f'  [deleted] {deleted}/{before} entries from '
                      f'{method:<22} on {spec_label} ({rel})')
                total_deleted += deleted
            rerun_needed.append({
                'method': method,
                'spec': spec_label,
                'fams_to_rerun': sorted(affected_set),
                'result_path': rel,
            })

    if not args.dry_run:
        print(f'\nTotal entries deleted: {total_deleted}')

    if rerun_needed:
        print('\n=== Suggested re-run commands (use --skip-existing-fams) ===')
        for x in rerun_needed:
            print(f'  # {x["method"]} on {x["spec"]} '
                  f'({len(x["fams_to_rerun"])} fams)')
            print(f'  python experiments/<benchmark>.py --dataset '
                  f'unified_{x["spec"] if x["spec"] != "short" else "short_test"}'
                  f' --skip-existing-fams ...')


if __name__ == '__main__':
    main()
