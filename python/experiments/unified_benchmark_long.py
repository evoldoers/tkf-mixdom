#!/usr/bin/env python3
"""Unified reconstruction benchmark for LONG families (101-500 columns).

Uses the spec from unified_benchmark_long_spec.json (exact family list,
held-out leaves, retained leaves). Methods:
  - Felsenstein LG08 (marginal ancestor + Fitch gap parsimony)
  - partition_d3f1 (PhyloHMM partition-recon with d3f1 MixDom)
  - partition_d5f1 (PhyloHMM partition-recon with d5f1 MixDom)
  - d3f1 beam (composite beam with SVI-BW d3f1 model)
  - d5f1 beam (composite beam with SVI-BW d5f1 model)

Additional MixDom models can be added via MIXDOM_* env vars (same as
alignment_benchmark_full.py). ArDCA runs separately via Julia.

Usage:
    cd python && CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 \\
        uv run python -u experiments/unified_benchmark_long.py

    # With additional models:
    MIXDOM_3=pfam/svi_bw_d3f1_panther_best_val.npz:d3f1_panther \\
        uv run python -u experiments/unified_benchmark_long.py
"""
import os, sys, json, time, traceback
import numpy as np

# Method selection via env var: RECON_METHODS=fels,partition_d3f1,...
# Default: all methods.
_ENABLED_METHODS = set(os.environ.get('RECON_METHODS', '').split(',')) if os.environ.get('RECON_METHODS') else None

def _method_enabled(method_name):
    """Check if a method should run (True if no filter or method in filter)."""
    return _ENABLED_METHODS is None or method_name in _ENABLED_METHODS

class _SkipMethod(Exception):
    """Raised to skip a method that's disabled or already done."""
    pass

os.environ.setdefault('JAX_ENABLE_X64', '1')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp
from tkfmixdom.jax.distill.maraschino import (
    load_params, precompute_mixdom, distill_mixdom, normalize_freqs_wfst)
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm
from tkfmixdom.jax.tree.tree_varanc import name_internal_nodes
from tkfmixdom.jax.tree.composite_beam_jax import composite_beam_reconstruct_jax
from tkfmixdom.jax.tree.composite_beam import compute_unique_weights
from tkfmixdom.jax.util.io import parse_newick

from ancrec_benchmark import parse_sto, PFAM_DIR
from unified_reconstruction_benchmark import (
    score_prediction, run_felsenstein, score_felsenstein_columns,
    build_mixdom_beam_data, build_mixdom_beam_data_class,
    prune_leaf_keep_parent, _nw_metrics,
    AA_TO_INT, TREE_DIR, TKF92_INS, TKF92_DEL, TKF92_EXT,
)
from partition_recon_adapter import (
    mixdom_model_from_params, run_partition_reconstruction_method,
    PartitionReconConfig,
)

# ── MixDom model loading (same env-var convention) ───────────────────
_MIXDOM_DEFAULTS = [
    ('pfam/svi_bw_d3f1_full_best_val.npz', 'd3f1'),
    ('pfam/svi_bw_d5f1_full_best_val.npz', 'd5f1'),
]

def _parse_mixdom_env():
    entries = []
    for k, v in sorted(os.environ.items()):
        if k.startswith('MIXDOM_') and k[7:].isdigit():
            parts = v.split(':', 1)
            entries.append((parts[0], parts[1] if len(parts) == 2 else parts[0]))
    return entries if entries else _MIXDOM_DEFAULTS

MIXDOM_MODELS = _parse_mixdom_env()
PYTHON_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BEAM_WIDTH = 30
MAX_COL = 500

SPEC_PATH = os.path.join(os.path.dirname(__file__), 'unified_benchmark_long_spec.json')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'unified_benchmark_long_results.json')

t0 = time.time()
def log(msg): print(f'[{time.time()-t0:.0f}s] {msg}', flush=True)


def main():
    spec = json.load(open(SPEC_PATH))
    families = spec['families']
    log(f'Loaded spec: {len(families)} families, cols {spec["col_range"]}')

    # Load models
    log('Loading models...')
    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg_np = np.asarray(Q_lg)
    pi_lg_np = np.asarray(pi_lg)

    partition_config = PartitionReconConfig(use_jax=True)
    partition_models = {}
    raw_params = {}
    for ckpt_path, key in MIXDOM_MODELS:
        full_path = os.path.join(PYTHON_ROOT, ckpt_path)
        if os.path.exists(full_path):
            p, nd, nc = load_params(full_path)
            partition_models[key] = mixdom_model_from_params(p)
            raw_params[key] = p
            log(f'  partition_{key}: n_dom={nd}')
    log(f'  LG08 loaded')

    # --- Beam search model precomputation ---
    beam_models = {}  # key -> (params, n_dom, n_frag, s_trans, s_pi, s_end)
    for ckpt_path, key in MIXDOM_MODELS:
        full_path = os.path.join(PYTHON_ROOT, ckpt_path)
        if os.path.exists(full_path):
            p, nd, nc = load_params(full_path)
            # Infer n_frag from the loaded params instead of hardcoding 1.
            # frag_weights is (N, F); fall back to r_frags (N, F) or scalar 1.
            if 'frag_weights' in p and np.asarray(p['frag_weights']).ndim == 2:
                n_frag = int(np.asarray(p['frag_weights']).shape[1])
            elif 'r_frags' in p and np.asarray(p['r_frags']).ndim == 2:
                n_frag = int(np.asarray(p['r_frags']).shape[1])
            else:
                n_frag = 1
            precomp = precompute_mixdom(p, max(nc, 1))
            dist = distill_mixdom(p, 0.1, max(nc, 1), precomp)
            wfst = normalize_freqs_wfst(dist)
            s_trans = np.log(np.maximum(np.array(wfst['singlet_trans']), 1e-300))
            s_start = np.array(wfst['singlet_start'])
            s_start = s_start / s_start.sum()
            s_pi = np.log(np.maximum(s_start, 1e-300))
            s_end = np.log(np.maximum(np.array(wfst['singlet_end']), 1e-300))
            beam_models[key] = (p, nd, n_frag, s_trans, s_pi, s_end)
            log(f'  beam_{key}: n_dom={nd}, n_frag={n_frag}')
    log(f'  Beam models loaded: {list(beam_models.keys())}')

    # Resume
    results = []
    results_by_fam = {}  # family -> index in results list
    done_fams = set()
    if os.path.exists(OUT_PATH):
        try:
            rd = json.load(open(OUT_PATH))
            if isinstance(rd, dict) and 'results' in rd:
                results = list(rd['results'])
                for ri, r in enumerate(results):
                    results_by_fam[r['family']] = ri
                done_fams = {r['family'] for r in results}
                log(f'Resume: {len(done_fams)} families already done')
        except Exception as e:
            log(f'Resume failed: {e}')

    beam_keys = list(beam_models.keys())  # e.g. ['d3f1', 'd5f1']

    def _save():
        with open(OUT_PATH, 'w') as f:
            json.dump({'results': results, 'n_families': len(results),
                       'spec': 'unified_benchmark_long_spec.json'}, f, indent=2)

    new_count = 0
    for fi, entry in enumerate(families):
        fam = entry['family']
        held_out = entry['held_out']
        remaining = entry['remaining']
        true_seq = np.array(entry['true_seq'], dtype=np.int32)
        C = entry['n_cols']

        # Check which beam keys still need to be run for this family
        existing_result = None
        if fam in results_by_fam:
            existing_result = results[results_by_fam[fam]]
        beam_needed = [k for k in beam_keys
                       if existing_result is None or k not in existing_result]

        # Skip if ALL enabled methods already have results
        if fam in done_fams and not beam_needed:
            if existing_result is not None:
                all_methods = ['fels'] + [f'partition_{k}' for k in ['d3f1', 'd5f1']] + beam_keys
                enabled = [m for m in all_methods if _method_enabled(m)]
                if all(m in existing_result for m in enabled):
                    continue
                # Fall through — some enabled methods missing
                log(f'  {fam}: falling through (enabled={enabled}, have={[m for m in enabled if m in existing_result]})')  # DEBUG
            else:
                log(f'  {fam}: in done_fams but no existing_result??')  # DEBUG
                continue

        # Parse MSA
        sto_path = os.path.join(PFAM_DIR, f'{fam}.sto')
        if not os.path.exists(sto_path):
            continue
        seqs = parse_sto(sto_path)
        msa = {}
        for name in seqs:
            seq = np.full(C, -1, dtype=np.int32)
            for j, ch in enumerate(seqs[name]):
                if ch in AA_TO_INT:
                    seq[j] = AA_TO_INT[ch]
            msa[name] = seq

        # Parse tree
        tree_path = os.path.join(TREE_DIR, f'{fam}.nwk')
        if not os.path.exists(tree_path):
            continue
        tree = parse_newick(open(tree_path).read().strip())
        name_internal_nodes(tree)

        # Compute distances from each remaining leaf to the target's parent node.
        rerooted_tree, _ = prune_leaf_keep_parent(tree, held_out)
        def _dist_to_root(node):
            d = 0.0
            while node.parent is not None:
                d += node.branch_length if node.branch_length else 0.0
                node = node.parent
            return d
        leaf_dist = {}
        for node in rerooted_tree.preorder():
            if node.is_leaf and node.name:
                leaf_dist[node.name] = _dist_to_root(node)
        distances_k = [max(leaf_dist.get(l, 1.0), 0.01) for l in remaining]
        t_score = float(np.mean(distances_k))
        log_chi_score, st_score, sub_score, pi_score = make_tkf92_pair_hmm(
            TKF92_INS, TKF92_DEL, t_score, TKF92_EXT,
            jnp.array(Q_lg_np), jnp.array(pi_lg_np))
        log_chi_s = np.asarray(log_chi_score)
        st_s = np.asarray(st_score)
        sub_s = np.asarray(sub_score)
        pi_s = np.asarray(pi_lg_np)

        # If this family already has results, augment; otherwise create new
        if existing_result is not None:
            result = existing_result
        else:
            result = {
                'family': fam,
                'held_out': held_out,
                'true_len': int(len(true_seq)),
                'n_cols': C,
                'K': len(remaining),
                'mean_dist': float(np.mean(distances_k)),
            }

        # === Felsenstein ===
        try:
            if not _method_enabled('fels') or 'fels' in result:
                raise _SkipMethod()
            fels_seq, fels_time = run_felsenstein(
                tree, held_out, remaining, msa, C, Q_lg_np, pi_lg_np)
            fels_score = score_prediction(
                fels_seq, true_seq, log_chi_s, st_s, sub_s, pi_s)
            nw = _nw_metrics(fels_seq, true_seq)
            result['fels'] = {**fels_score, **nw, 'time': float(fels_time),
                              'pred_seq': [int(x) for x in fels_seq]}
        except _SkipMethod:
            pass
        except Exception as e:
            log(f'  {fam}: fels error: {e}')
            traceback.print_exc()
            result['fels'] = {'accuracy': -1.0, 'time': 0.0}

        # === Partition-recon for each MixDom model ===
        for key, model in partition_models.items():
            tag = f'partition_{key}'
            try:
                if not _method_enabled(tag) or tag in result:
                    raise _SkipMethod()
                pred, elapsed = run_partition_reconstruction_method(
                    tree, held_out, remaining, msa, C,
                    model=model, config=partition_config)
                sc = score_prediction(pred, true_seq, log_chi_s, st_s, sub_s, pi_s)
                nw = _nw_metrics(pred, true_seq)
                result[tag] = {**sc, **nw, 'time': float(elapsed),
                               'pred_seq': [int(x) for x in pred]}
            except _SkipMethod:
                pass
            except Exception as e:
                log(f'  {fam}: {tag} error: {e}')
                traceback.print_exc()
                result[tag] = {'accuracy': -1.0, 'time': 0.0}

        # Add to results list if new family
        if existing_result is None:
            results.append(result)
            results_by_fam[fam] = len(results) - 1
        done_fams.add(fam)

        # === Beam reconstruction for each MixDom model ===
        # Extract ungapped descendant sequences for beam
        desc_seqs_k = [np.array([c for c in msa[l] if c >= 0], dtype=np.int32)
                       for l in remaining]

        # Compute phylo-aware weights from the FULL tree (before pruning)
        weights = compute_unique_weights(tree, tree.name, remaining)

        for bkey in beam_needed:
            if bkey not in beam_models:
                continue
            b_params, b_ndom, b_nfrag, b_s_trans, b_s_pi, b_s_end = beam_models[bkey]
            try:
                if not _method_enabled(bkey) or bkey in result:
                    raise _SkipMethod()
                # Auto-detect MixDom2 per-class structure.
                has_class = all(k in b_params for k in
                                ('class_pi', 'class_S_exch', 'class_dist'))
                if has_class:
                    (lc, st, sm, pl,
                     csm, cpl, cdist) = build_mixdom_beam_data_class(
                        b_params, b_ndom, b_nfrag, distances_k)
                else:
                    lc, st, sm, pl = build_mixdom_beam_data(
                        b_params, b_ndom, b_nfrag, distances_k)
                    csm = cpl = cdist = None
                tb = time.time()
                recon, score = composite_beam_reconstruct_jax(
                    desc_seqs_k, distances_k, lc, st, sm, pl,
                    b_ndom, b_nfrag, b_s_trans, b_s_pi, b_s_end,
                    beam_width=BEAM_WIDTH,
                    max_len=int(len(true_seq) * 1.5),
                    desc_weights=weights,
                    class_sub_matrices_list=csm,
                    class_pis_list=cpl,
                    class_dist=cdist)
                beam_time = time.time() - tb
                beam_score = score_prediction(
                    recon, true_seq, log_chi_s, st_s, sub_s, pi_s)
                nw = _nw_metrics(recon, true_seq)
                result[bkey] = {**beam_score, **nw, 'time': float(beam_time),
                                'pred_seq': [int(x) for x in recon],
                                'beam_score': float(score)}
            except _SkipMethod:
                pass
            except Exception as e:
                log(f'  {fam}: beam {bkey} error: {e}')
                traceback.print_exc()
                result[bkey] = {'accuracy': -1.0, 'time': 0.0}

        new_count += 1

        # Print summary
        def _fmt(d, key='accuracy'):
            v = d.get(key, -1) if isinstance(d, dict) else -1
            return f'{v*100:.1f}%' if isinstance(v, float) and v >= 0 else 'ERR'
        def _fmt_lp(d):
            v = d.get('log_prob', None)
            return f'{v:.1f}' if isinstance(v, (int, float)) and v > -1e20 else 'N/A'

        log(f'[{fi+1}/{len(families)}] {fam} (C={C}, K={len(remaining)}):')
        all_method_keys = (['fels'] + [f'partition_{k}' for k in partition_models]
                           + beam_keys)
        for method_key in all_method_keys:
            d = result.get(method_key, {})
            if d:
                log(f'  {method_key:16s} acc={_fmt(d)} prec={_fmt(d,"precision")} '
                    f'rec={_fmt(d,"recall")} logP={_fmt_lp(d)} t={d.get("time",0):.1f}s')

        # Save every 5 families
        if new_count % 5 == 0:
            _save()

    # Final save
    _save()

    # Summary
    log(f'\n{"="*60}')
    method_keys = (['fels'] + [f'partition_{k}' for k in partition_models]
                   + beam_keys)
    log(f'{"Method":<20} {"Accuracy":>8} {"Precision":>9} {"Recall":>8} {"N":>5}')
    log('-' * 50)
    for mk in method_keys:
        accs = [r[mk]['accuracy'] for r in results
                if isinstance(r.get(mk), dict) and r[mk].get('accuracy', -1) >= 0]
        if accs:
            precs = [r[mk]['precision'] for r in results
                     if isinstance(r.get(mk), dict) and r[mk].get('precision', -1) >= 0]
            recs = [r[mk]['recall'] for r in results
                    if isinstance(r.get(mk), dict) and r[mk].get('recall', -1) >= 0]
            log(f'{mk:<20} {np.mean(accs):>7.1%} {np.mean(precs):>8.1%} '
                f'{np.mean(recs):>7.1%} {len(accs):>5}')

    log('\nDone.')


if __name__ == '__main__':
    main()
