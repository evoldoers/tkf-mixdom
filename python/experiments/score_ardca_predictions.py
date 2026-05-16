#!/usr/bin/env python3
"""Post-process ArDCA predictions through score_prediction to get
FB-aligned accuracy, precision, recall, and log_prob — exactly
comparable to all other unified benchmark methods.

Reads ardca_benchmark_fasttree_results.json (which must contain
pred_seq and true_seq per family), scores each prediction through
the same TKF92 pair HMM used by the unified benchmark, and writes
ardca_fasttree_scored.json.

Usage:
    cd python && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \\
        uv run python -u experiments/score_ardca_predictions.py
"""
import os, sys, json, time
import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm
from unified_reconstruction_benchmark import (
    score_prediction, TKF92_INS, TKF92_DEL, TKF92_EXT,
)

ARDCA_PATH = os.path.join(os.path.dirname(__file__),
                           'ardca_benchmark_fasttree_results.json')
SPEC_PATH = os.path.join(os.path.dirname(__file__),
                          'unified_benchmark_spec.json')
OUT_PATH = os.path.join(os.path.dirname(__file__),
                         'ardca_fasttree_scored.json')


def main():
    ardca = json.load(open(ARDCA_PATH))
    spec = json.load(open(SPEC_PATH))
    spec_by_fam = {e['family']: e for e in spec['families']}

    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg_np = np.asarray(Q_lg)
    pi_lg_np = np.asarray(pi_lg)

    scored = []
    for r in ardca['families']:
        fam = r['family']
        pred_seq = r.get('pred_seq')
        true_seq = r.get('true_seq')
        if pred_seq is None or true_seq is None:
            print(f'{fam}: no pred_seq/true_seq, skipping')
            continue

        pred = np.array(pred_seq, dtype=np.int32)
        true = np.array(true_seq, dtype=np.int32)

        # Use mean_dist from spec if available, else default t=1.0
        sp = spec_by_fam.get(fam, {})
        t_score = sp.get('mean_dist', 1.0)

        log_chi, st, sub, pi_out = make_tkf92_pair_hmm(
            TKF92_INS, TKF92_DEL, t_score, TKF92_EXT,
            jnp.array(Q_lg_np), jnp.array(pi_lg_np))

        sc = score_prediction(
            pred, true,
            np.asarray(log_chi), np.asarray(st),
            np.asarray(sub), pi_lg_np)

        entry = {
            'family': fam,
            'held_out': r.get('held_out', sp.get('held_out', '')),
            **sc,
            'pred_seq': pred_seq,
            'pred_len': len(pred_seq),
            'true_len': len(true_seq),
            'ardca_col_accuracy': r.get('accuracy'),
        }
        scored.append(entry)
        print(f'{fam}: acc={sc["accuracy"]*100:.1f}% prec={sc["precision"]*100:.1f}% '
              f'rec={sc["recall"]*100:.1f}% logP={sc["log_prob"]:.1f}')

    with open(OUT_PATH, 'w') as f:
        json.dump(scored, f, indent=2)
    print(f'\nScored {len(scored)} families → {OUT_PATH}')

    accs = [s['accuracy'] for s in scored]
    precs = [s['precision'] for s in scored]
    recs = [s['recall'] for s in scored]
    lps = [s['log_prob'] for s in scored]
    print(f'mean acc={np.mean(accs)*100:.1f}% prec={np.mean(precs)*100:.1f}% '
          f'rec={np.mean(recs)*100:.1f}% logP={np.mean(lps):.1f}')


if __name__ == '__main__':
    main()
