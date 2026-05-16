#!/usr/bin/env python3
"""Benchmark driver for partition-conditioned ancestral reconstruction.

Runs the partition-conditioned algorithm (tkf/partition-recon.tex) on
the same held-out-leaf prediction setup as
`experiments/unified_reconstruction_benchmark.py`. Imports helpers
from the unified benchmark when available, without modifying any of
its code paths. When the Pfam data helpers are unavailable (as in
a test environment), the driver falls back to a self-contained
synthetic demo.

Typical usage:

    cd python && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        uv run python experiments/partition_recon_benchmark.py \
        --max-families 20 --max-col 80

Or as a smoke test:

    cd python && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        uv run python experiments/partition_recon_benchmark.py --demo
"""

import os
import sys
import json
import time
import argparse
import traceback
import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', '1')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update('jax_enable_x64', True)

# Reuse helpers from the unified benchmark without modifying it.
try:
    from experiments.unified_reconstruction_benchmark import (
        score_prediction, prune_leaf, run_felsenstein,
        TKF92_INS, TKF92_DEL, TKF92_EXT,
    )
except Exception:
    score_prediction = None
    prune_leaf = None
    run_felsenstein = None
    TKF92_INS = 0.046
    TKF92_DEL = 0.054
    TKF92_EXT = 0.68

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm
from tkfmixdom.jax.dp.hmm import forward_backward_2d, pair_hmm_emissions

from experiments.partition_recon_adapter import (
    PartitionReconConfig, default_single_domain_model,
    run_partition_reconstruction_method, partition_reconstruction_result,
)
from tkfmixdom.jax.tree.tree_varanc import (
    infer_internal_presence, name_internal_nodes,
)


def _log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _score_prediction_minimal(pred_seq, true_seq, Q_lg, pi_lg, distance=0.3):
    """Fallback scoring when score_prediction is unavailable."""
    import jax.numpy as jnp
    from tkfmixdom.jax.core.params import M as M_ST
    pred = jnp.array(pred_seq, dtype=jnp.int32)
    true = jnp.array(true_seq, dtype=jnp.int32)
    Lp, Lt = len(pred), len(true)
    if Lp == 0 or Lt == 0:
        return {'matches': 0, 'inserts': int(Lp), 'deletes': int(Lt),
                'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0,
                'log_prob': -1e30, 'pred_len': int(Lp), 'true_len': int(Lt)}
    log_chi, st, sub, pi_out = make_tkf92_pair_hmm(
        TKF92_INS, TKF92_DEL, distance, TKF92_EXT,
        jnp.asarray(Q_lg), jnp.asarray(pi_lg))
    log_prob, posts, _ = forward_backward_2d(
        log_chi, st, pred, true, jnp.asarray(sub), jnp.asarray(pi_out))
    post = np.asarray(posts)
    st_np = np.asarray(st)
    is_M = (st_np == M_ST)
    match_post = post[1:Lp + 1, 1:Lt + 1, :][:, :, is_M].sum(axis=-1)
    E_matches = float(match_post.sum())
    E_inserts = float((1.0 - match_post.sum(axis=1)).clip(0).sum())
    E_deletes = float((1.0 - match_post.sum(axis=0)).clip(0).sum())
    precision = E_matches / max(E_matches + E_inserts, 1e-10)
    recall = E_matches / max(E_matches + E_deletes, 1e-10)
    correct = 0.0
    for i in range(Lp):
        for j in range(Lt):
            if match_post[i, j] > 1e-6 and pred_seq[i] == true_seq[j]:
                correct += match_post[i, j]
    accuracy = correct / max(E_matches, 1e-10)
    return {'matches': float(E_matches),
            'inserts': float(E_inserts),
            'deletes': float(E_deletes),
            'precision': float(precision), 'recall': float(recall),
            'accuracy': float(accuracy),
            'log_prob': float(log_prob),
            'pred_len': int(Lp), 'true_len': int(Lt)}


def _score(pred_seq, true_seq, Q_lg, pi_lg):
    if score_prediction is None:
        return _score_prediction_minimal(pred_seq, true_seq, Q_lg, pi_lg)
    import jax.numpy as jnp
    log_chi, st, sub, _ = make_tkf92_pair_hmm(
        TKF92_INS, TKF92_DEL, 0.3, TKF92_EXT,
        jnp.asarray(Q_lg), jnp.asarray(pi_lg))
    return score_prediction(pred_seq, true_seq, log_chi, st, sub, pi_lg)


def run_demo(kappa_top: float = 0.95):
    """Self-contained synthetic demo."""
    newick = '((A:0.15,B:0.2):0.1,(C:0.1,D:0.25):0.12);'
    msa = {
        'A': np.array([0, 1, 2, 3, -1, 4, 5], dtype=np.int32),
        'B': np.array([0, 1, 2, 3, 6, 4, 5], dtype=np.int32),
        'C': np.array([0, 1, 2, -1, -1, 4, 5], dtype=np.int32),
        'D': np.array([0, 1, 2, 3, -1, 4, 5], dtype=np.int32),
    }
    tree = parse_newick(newick)
    name_internal_nodes(tree)
    held_out = 'C'
    C = len(next(iter(msa.values())))
    remaining = [k for k in msa if k != held_out]
    pred_seq, elapsed, result, _ = partition_reconstruction_result(
        tree, held_out, remaining, msa, C,
        model=default_single_domain_model(kappa_top=kappa_top),
        config=PartitionReconConfig(kappa_top=kappa_top, use_jax=True),
    )
    true_seq = np.array([c for c in msa[held_out] if c >= 0], dtype=np.int32)
    Q, pi = rate_matrix_lg()
    Q = np.asarray(Q); pi = np.asarray(pi)
    score = _score(pred_seq, true_seq, Q, pi)
    return {
        'pred_seq': [int(x) for x in pred_seq],
        'true_seq': [int(x) for x in true_seq],
        'score': {k: (float(v) if hasattr(v, '__float__') else v)
                  for k, v in score.items()},
        'elapsed': float(elapsed),
        'log_Z_forward': float(result.log_Z_forward),
        'log_Z_backward': float(result.log_Z_backward),
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-families', type=int, default=20)
    ap.add_argument('--max-col', type=int, default=80)
    ap.add_argument('--min-seqs', type=int, default=5)
    ap.add_argument('--max-seqs', type=int, default=30)
    ap.add_argument('--kappa-top', type=float, default=0.95)
    ap.add_argument('--output',
                    default=os.path.join(os.path.dirname(__file__),
                                         'partition_recon_benchmark.json'))
    ap.add_argument('--demo', action='store_true',
                    help='Run a self-contained synthetic demo instead of Pfam.')
    return ap.parse_args()


def main():
    args = parse_args()

    if args.demo:
        _log('Running self-contained demo...')
        result = run_demo(kappa_top=args.kappa_top)
        _log(json.dumps(result, indent=2))
        return

    try:
        from experiments.ancrec_benchmark import (
            parse_sto, PFAM_DIR, SPLITS_PATH,
        )
    except Exception as e:
        _log(f'Pfam helpers unavailable ({e}); running --demo instead.')
        result = run_demo(kappa_top=args.kappa_top)
        _log(json.dumps(result, indent=2))
        return

    TREE_DIR = os.path.expanduser('~/bio-datasets/data/pfam-seed/trees')
    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg_np = np.asarray(Q_lg)
    pi_lg_np = np.asarray(pi_lg)

    model = default_single_domain_model(kappa_top=args.kappa_top)
    config = PartitionReconConfig(kappa_top=args.kappa_top, use_jax=True)

    with open(SPLITS_PATH) as f:
        val_fams = json.load(f)['val']
    _log(f'{len(val_fams)} val families total')

    results = []
    n_done = 0
    for fam in val_fams:
        if n_done >= args.max_families:
            break
        tree_path = os.path.join(TREE_DIR, f'{fam}.nwk')
        sto_path = os.path.join(PFAM_DIR, f'{fam}.sto')
        if not (os.path.exists(tree_path) and os.path.exists(sto_path)):
            continue

        seqs = parse_sto(sto_path)
        n = len(seqs)
        C = len(next(iter(seqs.values())))
        if n < args.min_seqs or n > args.max_seqs or C > args.max_col:
            continue

        try:
            tree = parse_newick(open(tree_path).read().strip())
            name_internal_nodes(tree)
        except Exception:
            continue

        msa = {}
        for name in seqs:
            arr = np.full(C, -1, dtype=np.int32)
            for j, ch in enumerate(seqs[name]):
                if ch in AA_TO_INT:
                    arr[j] = AA_TO_INT[ch]
            msa[name] = arr

        tree_leaves = [l.name for l in tree.leaves()]
        msa_names = set(seqs.keys())
        common = [l for l in tree_leaves if l in msa_names]
        if len(common) < args.min_seqs:
            continue
        held_out = common[0]
        true_seq = np.array([c for c in msa[held_out] if c >= 0],
                            dtype=np.int32)
        if len(true_seq) < 5:
            continue
        remaining = [l for l in common if l != held_out]

        fels_pred = None
        if run_felsenstein is not None:
            try:
                fels_pred, _ = run_felsenstein(
                    tree, held_out, remaining, msa, C, Q_lg_np, pi_lg_np)
            except Exception as e:
                _log(f'  {fam}: fels error: {e}')

        try:
            part_pred, part_time = run_partition_reconstruction_method(
                tree, held_out, remaining, msa, C, model, config)
        except Exception as e:
            traceback.print_exc()
            continue

        entry = {
            'family': fam,
            'held_out': held_out,
            'true_len': int(len(true_seq)),
            'n_cols': int(C),
            'K': int(len(remaining)),
            'partition': {
                **_score(part_pred, true_seq, Q_lg_np, pi_lg_np),
                'time': float(part_time),
                'pred_len': int(len(part_pred)),
            },
        }
        if fels_pred is not None:
            entry['fels'] = {
                **_score(fels_pred, true_seq, Q_lg_np, pi_lg_np),
                'pred_len': int(len(fels_pred)),
            }
        results.append(entry)
        n_done += 1
        _log(f'{fam}: partition acc={entry["partition"]["accuracy"]:.1%}'
             + (f'  fels acc={entry["fels"]["accuracy"]:.1%}'
                if 'fels' in entry else ''))

    _log(f'\nProcessed {n_done} families')
    with open(args.output, 'w') as f:
        json.dump({'n_families': n_done, 'results': results}, f, indent=2)
    _log(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
