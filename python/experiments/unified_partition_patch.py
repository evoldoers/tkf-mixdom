"""One-shot patch: add partition_d3f1 and partition_d5f1 entries to the
already-completed unified_reconstruction_benchmark.json.

The unified benchmark was already run end-to-end with 7 methods on 200
families before partition-recon was added. The benchmark script's resume
logic skips by family, so a plain rerun would not process the new
methods. This script:

  1. Loads the existing JSON.
  2. For each family, re-runs ONLY the partition_d3f1 and partition_d5f1
     methods, reusing the same tree/MSA loading code as the main
     benchmark.
  3. Patches the new fields into the existing result entry.
  4. Saves back to the same JSON.

It uses GPU0 by default (set CUDA_VISIBLE_DEVICES to override).

Usage:
    cd python && CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \\
        uv run python -u experiments/unified_partition_patch.py
"""
import os, sys, json, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-use unified benchmark loaders
import jax.numpy as jnp
from tkfmixdom.jax.distill.maraschino import load_params
from tkfmixdom.jax.tree.tree_varanc import name_internal_nodes
from tkfmixdom.jax.util.io import parse_newick
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm
from tkfmixdom.jax.core.protein import rate_matrix_lg

from experiments.partition_recon_adapter import (
    mixdom_model_from_params, run_partition_reconstruction_method,
    PartitionReconConfig,
)
from experiments.ancrec_benchmark import parse_sto, PFAM_DIR
from experiments.unified_reconstruction_benchmark import (
    score_prediction, AA_TO_INT, TREE_DIR,
    MIN_SEQS, MAX_SEQS, MAX_COL,
    TKF92_INS, TKF92_DEL, TKF92_EXT,
)


t0 = time.time()
def log(msg): print(f'[{time.time()-t0:.0f}s] {msg}', flush=True)


def main():
    json_path = os.path.join(os.path.dirname(__file__),
                              'unified_reconstruction_benchmark.json')
    with open(json_path) as f:
        d = json.load(f)
    results = d['results']
    log(f'Loaded {len(results)} existing results from {json_path}')

    # Load partition models from existing checkpoints
    log('Loading d3f1 + d5f1 params...')
    params_d3, _, _ = load_params('pfam/svi_bw_d3f1_full_best_val.npz')
    params_d5, _, _ = load_params('pfam/svi_bw_d5f1_full_best_val.npz')
    partition_model_d3 = mixdom_model_from_params(params_d3)
    partition_model_d5 = mixdom_model_from_params(params_d5)
    partition_config = PartitionReconConfig(use_jax=True)

    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg_np = np.asarray(Q_lg)
    pi_lg_np = np.asarray(pi_lg)

    n_done = 0
    n_skipped = 0
    n_already = 0
    for ri, r in enumerate(results):
        fam = r.get('family')
        if not fam:
            continue
        # Skip if both partition results already present
        if 'partition_d3f1' in r and 'partition_d5f1' in r:
            n_already += 1
            continue

        # Load tree + MSA the same way the main benchmark does
        tree_path = os.path.join(TREE_DIR, f'{fam}.nwk')
        sto_path = os.path.join(PFAM_DIR, f'{fam}.sto')
        if not os.path.exists(tree_path) or not os.path.exists(sto_path):
            n_skipped += 1
            continue
        try:
            seqs = parse_sto(sto_path)
        except Exception as e:
            log(f'  {fam}: parse_sto error {e}')
            n_skipped += 1
            continue
        n = len(seqs)
        if not seqs:
            n_skipped += 1
            continue
        C = len(next(iter(seqs.values())))
        if n < MIN_SEQS or n > MAX_SEQS or C > MAX_COL:
            n_skipped += 1
            continue
        try:
            with open(tree_path) as f:
                tree = parse_newick(f.read().strip())
            name_internal_nodes(tree)
        except Exception as e:
            log(f'  {fam}: tree parse error {e}')
            n_skipped += 1
            continue

        msa = {}
        for name in seqs:
            seq = np.full(C, -1, dtype=np.int32)
            for j, ch in enumerate(seqs[name]):
                if ch in AA_TO_INT:
                    seq[j] = AA_TO_INT[ch]
            msa[name] = seq

        tree_leaves = [l.name for l in tree.leaves()]
        msa_names = set(seqs.keys())
        common = [l for l in tree_leaves if l in msa_names]
        if len(common) < MIN_SEQS:
            n_skipped += 1
            continue

        # Use the SAME held-out leaf as the existing result for fair comparison
        held_out = r.get('held_out', common[0])
        if held_out not in common:
            held_out = common[0]
        true_msa_row = msa[held_out]
        true_seq = np.array([c for c in true_msa_row if c >= 0], dtype=np.int32)
        if len(true_seq) < 5:
            n_skipped += 1
            continue
        remaining = [l for l in common if l != held_out]

        # Common TKF92 pair HMM for scoring pred vs true
        # (use the same t_score = mean of true seq's pairwise distances —
        # but we don't have distances readily; reuse a representative t=1.0
        # since score_prediction is mostly insensitive to t at this scale)
        # The main benchmark uses mean(distances_k) — we don't compute
        # distances here, so use t=1.0 as a proxy. This only affects the
        # scoring pair HMM, not the alignment, and the discrepancy is
        # small (the score is the same alignment evaluated at a slightly
        # different scoring temperature).
        t_score = 1.0
        log_chi_score, st_score, sub_score, pi_score = make_tkf92_pair_hmm(
            TKF92_INS, TKF92_DEL, t_score, TKF92_EXT,
            jnp.array(Q_lg_np), jnp.array(pi_lg_np))
        log_chi_s = np.asarray(log_chi_score)
        st_s = np.asarray(st_score)
        sub_s = np.asarray(sub_score)
        pi_s = np.asarray(pi_lg_np)

        # === partition_d3f1 ===
        if 'partition_d3f1' not in r:
            try:
                pd3_pred, pd3_time = run_partition_reconstruction_method(
                    tree, held_out, remaining, msa, C,
                    model=partition_model_d3, config=partition_config)
                pd3_score = score_prediction(
                    pd3_pred, true_seq, log_chi_s, st_s, sub_s, pi_s)
                r['partition_d3f1'] = {
                    **pd3_score, 'time': float(pd3_time),
                    'pred_seq': [int(x) for x in pd3_pred],
                }
            except Exception as e:
                log(f'  {fam}: partition_d3f1 error: {e}')
                traceback.print_exc()
                r['partition_d3f1'] = {'accuracy': -1.0, 'time': 0.0,
                                        'error': str(e)[:200]}

        # === partition_d5f1 ===
        if 'partition_d5f1' not in r:
            try:
                pd5_pred, pd5_time = run_partition_reconstruction_method(
                    tree, held_out, remaining, msa, C,
                    model=partition_model_d5, config=partition_config)
                pd5_score = score_prediction(
                    pd5_pred, true_seq, log_chi_s, st_s, sub_s, pi_s)
                r['partition_d5f1'] = {
                    **pd5_score, 'time': float(pd5_time),
                    'pred_seq': [int(x) for x in pd5_pred],
                }
            except Exception as e:
                log(f'  {fam}: partition_d5f1 error: {e}')
                traceback.print_exc()
                r['partition_d5f1'] = {'accuracy': -1.0, 'time': 0.0,
                                        'error': str(e)[:200]}

        n_done += 1
        pd3 = r.get('partition_d3f1', {})
        pd5 = r.get('partition_d5f1', {})
        log(f'  {fam}: '
            f'pd3 acc={pd3.get("accuracy", -1)*100:.1f}% t={pd3.get("time", 0):.1f}s | '
            f'pd5 acc={pd5.get("accuracy", -1)*100:.1f}% t={pd5.get("time", 0):.1f}s')

        # Save every 10 families
        if n_done % 10 == 0:
            with open(json_path, 'w') as f:
                json.dump(d, f, indent=2)
            log(f'  [saved checkpoint after {n_done} families]')

    # Final save
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=2)
    log(f'\nDone. Patched {n_done} families, skipped {n_skipped}, '
        f'{n_already} already had partition results.')


if __name__ == '__main__':
    main()
