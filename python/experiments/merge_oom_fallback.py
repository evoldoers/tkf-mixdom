#!/usr/bin/env python3
"""Merge OOM-fallback per-family results into the canonical K20/C20
withsps JSONs.

Drops:
  - The 3 OOM-failed entries (BB11018, BB50002, BB50006) which lack
    per_pair_post / fsa_sps fields
  - The BB11003 control entry — already present in canonical, kept

Replaces them with the AWS results (4 entries), then BB11003 is
de-duplicated (we keep the canonical one and discard the AWS control
once we've sanity-checked it).

Recomputes corpus_post / corpus_hard / corpus_opt / corpus_fsa_sps
aggregates from the full 120-family per_family list.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def aggregate(per_family: list[dict]) -> dict[str, Any]:
    """Recompute corpus_* aggregates from per-family stats."""
    aggs: dict[str, dict[str, float]] = {}
    for key in ('post', 'hard', 'opt', 'fsa_sps'):
        e_tp = sum(f.get(f'micro_{key}', {}).get('e_tp', 0.0) for f in per_family)
        total = sum(f.get(f'micro_{key}', {}).get('total_mass', 0.0) for f in per_family)
        gold = sum(f.get(f'micro_{key}', {}).get('gold', 0.0) for f in per_family)
        e_fp = max(0.0, total - e_tp)
        e_fn = max(0.0, gold - e_tp)
        p = e_tp / total if total > 0 else 0.0
        r = e_tp / gold if gold > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        aggs[f'corpus_{key}'] = dict(e_tp=e_tp, e_fp=e_fp, e_fn=e_fn,
                                     total_mass=total, gold=gold,
                                     precision=p, recall=r, f1=f1)
    return aggs


def control_diff(canonical: dict, aws: dict, family: str = 'BB11003') -> dict[str, float]:
    """Compare canonical vs AWS BB11003 entry on key fields. Returns
    max-abs-diff per field."""
    diffs = {}
    fields = ('msa_sp_g1', 'msa_tc_g1', 'msa_sp_g0', 'msa_tc_g0',
              'per_pair_post', 'per_pair_fsa_sps')
    for f in fields:
        c = canonical.get(f)
        a = aws.get(f)
        if isinstance(c, (int, float)) and isinstance(a, (int, float)):
            diffs[f] = abs(c - a)
        elif isinstance(c, dict) and isinstance(a, dict):
            # Take the e_tp field as a single summary
            ec = c.get('e_tp', 0.0)
            ea = a.get('e_tp', 0.0)
            diffs[f] = abs(ec - ea)
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical', required=True,
                    help='Existing K20 or C20 _withsps.json (117/120 families)')
    ap.add_argument('--aws', required=True,
                    help='AWS result JSON (4 families: 3 OOM + BB11003 control)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--control-threshold', type=float, default=1e-3,
                    help='Max-abs-diff threshold for BB11003 control vs canonical')
    args = ap.parse_args()

    canonical = json.loads(Path(args.canonical).read_text())
    aws_doc = json.loads(Path(args.aws).read_text())

    if canonical['method_name'] != aws_doc['method_name']:
        print(f'!! method_name mismatch: canonical={canonical["method_name"]!r} '
              f'aws={aws_doc["method_name"]!r}', file=sys.stderr)
        return 1

    aws_by_fam = {f['family']: f for f in aws_doc['per_family']}

    # 1. Validate control
    canonical_bb11003 = next((f for f in canonical['per_family']
                              if f['family'] == 'BB11003'), None)
    if canonical_bb11003 is None:
        print('!! canonical missing BB11003', file=sys.stderr)
        return 2
    aws_bb11003 = aws_by_fam.get('BB11003')
    if aws_bb11003 is None:
        print('!! AWS result missing BB11003 control', file=sys.stderr)
        return 3
    diffs = control_diff(canonical_bb11003, aws_bb11003)
    print('BB11003 control diff:')
    for k, v in diffs.items():
        flag = ' OK' if v <= args.control_threshold else ' !! ABOVE THRESHOLD'
        print(f'  {k:>20s}  diff={v:.4e}{flag}')
    if max(diffs.values()) > args.control_threshold:
        print(f'!! Control validation FAILED. Max diff '
              f'{max(diffs.values()):.4e} > {args.control_threshold:.4e}',
              file=sys.stderr)
        print('   Refusing to merge. Inspect AWS pipeline.', file=sys.stderr)
        return 4

    # 2. Slot the 3 OOM-failed entries with AWS results
    OOM_FAMS = ('BB11018', 'BB50002', 'BB50006')
    new_per_family = []
    n_replaced = 0
    for f in canonical['per_family']:
        if f['family'] in OOM_FAMS and aws_by_fam.get(f['family']) is not None:
            new_per_family.append(aws_by_fam[f['family']])
            n_replaced += 1
            print(f'  replaced {f["family"]} (OOM -> AWS)')
        else:
            new_per_family.append(f)
    if n_replaced != 3:
        print(f'!! expected to replace 3 OOM entries, replaced {n_replaced}',
              file=sys.stderr)
        return 5

    # 3. Recompute corpus aggregates
    aggs = aggregate(new_per_family)

    # 4. Assemble + write
    out = dict(canonical)
    out['per_family'] = new_per_family
    out['n_families'] = len(new_per_family)
    out.update(aggs)
    out['oom_fallback_merged_from'] = str(args.aws)

    Path(args.out).write_text(json.dumps(out, indent=2))
    n_complete = sum(1 for f in new_per_family if f.get('msa_sp_g1') is not None)
    print(f'OK — wrote {args.out} ({n_complete}/{len(new_per_family)} '
          f'complete).')
    print(f'  corpus_fsa_sps.f1 = {aggs["corpus_fsa_sps"]["f1"]:.4f}')
    print(f'  corpus_post.f1    = {aggs["corpus_post"]["f1"]:.4f}')


if __name__ == '__main__':
    sys.exit(main() or 0)
