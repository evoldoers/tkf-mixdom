#!/usr/bin/env python3
"""Backfill logp_target and logp_true on varanc / varanc-mixdom JSONs.

These JSONs already save `p_present` per method per entry and the
ground-truth `gt_present` vector at entry-level, so both the joint log
posterior of the predicted sequence (`logp_target`) and of the true
sequence (`logp_true`) can be recomputed exactly without re-running
the model. See `logp_binary_pred` / `logp_binary_true` and the math
notes in varanc_presence_benchmark.py.

Standalone fels21 / fels40 reconstruction JSONs cannot be backfilled
from saved fields alone (per-column posteriors aren't in the JSON);
those need `--recompute-missing-fields` on the launcher to re-run the
affected entries.

Usage:
    cd python && python -u experiments/backfill_logp_target_varanc.py \\
        experiments/varanc_presence_unified_short_test.json \\
        experiments/varanc_presence_mixdom_unified_short_test.json \\
        ...

Backfills both `logp_target` (if missing) and `logp_true` (if missing)
per method per entry. Pre-existing fields are left untouched unless
`--force` is passed (recomputes fresh values).
"""
import json
import sys


def backfill_one(path, force=False):
    """Backfill logp_target / logp_true for every probabilistic method
    with p_present.

    Hard-label methods (fitch, fels21-as-presence) are SKIPPED — they
    have no calibrated per-column posterior, so logp_target / logp_true
    are category errors there. Only varanc and mixdom are backfilled.
    """
    import numpy as np
    sys.path.insert(0, '/home/yam/tkf-mixdom/python')
    from experiments.varanc_presence_benchmark import (
        logp_binary_pred, logp_binary_true,
    )

    # Hard-label presence methods: no probabilistic posterior. The
    # standalone fels21/fels40 residue launchers (NOT in this file's
    # purview) DO have calibrated posteriors and should keep the fields.
    HARD_LABEL_METHODS = {'fitch', 'fels21'}

    with open(path) as f:
        d = json.load(f)
    if 'results' not in d:
        print(f"{path}: no 'results' key, skip")
        return
    n_target = 0
    n_true = 0
    n_skipped_hard = 0
    n_stripped_hard = 0
    for r in d['results']:
        gt = r.get('gt_present')
        gt_arr = np.asarray(gt, dtype=np.int32) if gt is not None else None
        for m, mres in r.get('methods', {}).items():
            if not isinstance(mres, dict):
                continue
            if 'p_present' not in mres:
                continue
            if m in HARD_LABEL_METHODS:
                # Strip any pre-existing legacy logp_* fields (these were
                # 0 / -inf; now treated as category errors).
                if mres.pop('logp_target', None) is not None:
                    n_stripped_hard += 1
                if mres.pop('logp_true', None) is not None:
                    n_stripped_hard += 1
                n_skipped_hard += 1
                continue
            p = np.asarray(mres['p_present'], dtype=np.float64)
            if force or 'logp_target' not in mres:
                mres['logp_target'] = logp_binary_pred(p)
                n_target += 1
            if gt_arr is not None and (force or 'logp_true' not in mres):
                # Length mismatch shouldn't happen but guard against it.
                if len(gt_arr) == len(p):
                    mres['logp_true'] = logp_binary_true(p, gt_arr)
                    n_true += 1
    with open(path, 'w') as f:
        json.dump(d, f)
    print(f"{path}: backfilled logp_target={n_target}, logp_true={n_true} "
          f"(method, entry) pairs; skipped {n_skipped_hard} hard-label "
          f"entries (stripped {n_stripped_hard} legacy logp_* fields)")


def main():
    args = list(sys.argv[1:])
    force = False
    if '--force' in args:
        force = True
        args.remove('--force')
    if not args:
        print(__doc__)
        sys.exit(1)
    for path in args:
        backfill_one(path, force=force)


if __name__ == '__main__':
    main()
