#!/usr/bin/env python3
"""Proper Pfam reconstruction benchmark: Fels vs Fels+gamma vs TVA+MixDom.

Evaluates all methods at matched columns (Fitch-present ∩ held-out-present).
TVA uses MixDom-distilled WFSTs (real order-1 model), not TKF91.

Usage:
    cd python && JAX_PLATFORMS=cpu uv run python experiments/pfam_recon_proper.py
"""

import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
jax.config.update('jax_enable_x64', True)

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.ctmc import ensure_rate_matrix
from tkfmixdom.jax.tree.ancestor import marginal_ancestor_all_columns_jax
from tkfmixdom.jax.tree.tree_varanc import (
    tree_varanc, tree_varanc_block_diagonal,
    infer_internal_presence, name_internal_nodes,
    NEG_INF,
)
from tkfmixdom.jax.tree.guide_tree import neighbor_joining
from tkfmixdom.jax.util.io import AA_TO_INT

from experiments.ancrec_benchmark import (
    parse_sto, msa_pairwise_distances, remove_leaf,
    find_parent_of_leaf, build_mixdom_wfst_log,
    TKF92_INS_RATE, TKF92_DEL_RATE,
    PFAM_DIR, SPLITS_PATH,
)

GAMMA_DIR = "gamma_labels"
MIXDOM_PARAMS = "pfam/maraschino_d6.npz"
MIXDOM_DYNAMIC_PARAMS = "pfam/maraschino_n2_d2_fit.npz"
A = 20
G = 4

# ======================================================================
# MixDom WFST construction (from existing benchmark infrastructure)
# ======================================================================

_mixdom_state = {}
_mixdom_dyn_state = {}


def _init_mixdom():
    if 'params' in _mixdom_state:
        return
    from tkfmixdom.jax.distill.maraschino import load_params, precompute_mixdom
    params, n_domains, n_classes = load_params(MIXDOM_PARAMS)
    precomp = precompute_mixdom(params, n_classes)
    _mixdom_state['params'] = params
    _mixdom_state['n_classes'] = n_classes
    _mixdom_state['precomp'] = precomp


def _init_mixdom_dynamic():
    if 'params' in _mixdom_dyn_state:
        return
    if not os.path.exists(MIXDOM_DYNAMIC_PARAMS):
        return
    from tkfmixdom.jax.distill.maraschino import load_params, precompute_mixdom
    params, n_domains, n_classes = load_params(MIXDOM_DYNAMIC_PARAMS)
    precomp = precompute_mixdom(params, n_classes)
    _mixdom_dyn_state['params'] = params
    _mixdom_dyn_state['n_classes'] = n_classes
    _mixdom_dyn_state['precomp'] = precomp


def _add_bos_keys(wfst_log):
    d = dict(wfst_log)
    d['log_p_bos_i_m'] = np.asarray(d['log_p_im'])[0].copy()
    d['log_p_bos_i_i'] = np.asarray(d['log_p_ii'])[0].copy()
    d['log_p_bos_i_d'] = np.asarray(d['log_p_id'])[0].copy()
    d['log_p_bos_i_e'] = np.asarray(d['log_p_ie'])[0].copy()
    d['log_p_bos_d_m'] = np.asarray(d['log_p_dm'])[:, 0].copy()
    d['log_p_bos_d_i'] = np.asarray(d['log_p_di'])[:, 0].copy()
    d['log_p_bos_d_d'] = np.asarray(d['log_p_dd'])[:, 0].copy()
    d['log_p_bos_d_e'] = np.asarray(d['log_p_de'])[:, 0].copy()
    return d


def _build_wfst_from_state(state, t):
    from tkfmixdom.jax.distill.maraschino import distill_mixdom, normalize_freqs_wfst
    dist = distill_mixdom(state['params'], t, state['n_classes'], state['precomp'])
    wfst = normalize_freqs_wfst(dist)
    log_wfst = build_mixdom_wfst_log(wfst)
    return _add_bos_keys(log_wfst)


def _build_singlet_from_state(state):
    from tkfmixdom.jax.distill.maraschino import distill_mixdom
    dist = distill_mixdom(state['params'], 0.1, state['n_classes'], state['precomp'])

    f_singlet = np.asarray(dist['f_singlet'])
    f_start = np.asarray(dist['f_singlet_start'])
    f_end = np.asarray(dist['f_singlet_end'])
    AA = f_singlet.shape[0]
    sl = lambda x: np.log(np.maximum(x, 1e-300))

    total_start = np.sum(f_start)
    p_start_emit = f_start / max(total_start + (1.0 - total_start), 1e-30)
    p_start_end = 1.0 - np.sum(p_start_emit)

    log_p_si = np.broadcast_to(sl(p_start_emit)[None, :], (AA, AA)).copy()
    log_p_se = float(sl(max(p_start_end, 1e-300)))

    Z = np.sum(f_singlet, axis=1) + f_end
    Z = np.maximum(Z, 1e-300)
    p_ii = f_singlet / Z[:, None]
    p_ie = f_end / Z

    log_p_ii = np.broadcast_to(sl(p_ii)[None, :, :], (AA, AA, AA)).copy()
    log_p_ie = np.broadcast_to(sl(p_ie)[None, :], (AA, AA)).copy()

    imp4 = np.full((AA, AA, AA, AA), NEG_INF)
    imp3 = np.full((AA, AA, AA), NEG_INF)
    imp2 = np.full((AA, AA), NEG_INF)

    return {
        'log_p_mm': imp4, 'log_p_mi': imp3, 'log_p_md': imp3, 'log_p_me': imp2,
        'log_p_im': imp4, 'log_p_ii': log_p_ii, 'log_p_id': imp3, 'log_p_ie': log_p_ie,
        'log_p_dm': imp4, 'log_p_dd': imp3, 'log_p_di': imp3, 'log_p_de': imp2,
        'log_p_sm': imp2, 'log_p_si': log_p_si, 'log_p_sd': imp2, 'log_p_se': log_p_se,
        'log_p_bos_i_m': imp3.copy(), 'log_p_bos_i_i': log_p_ii[0].copy(),
        'log_p_bos_i_d': imp2.copy(), 'log_p_bos_i_e': log_p_ie[0].copy(),
    }


def build_mixdom_wfst(t):
    _init_mixdom()
    return _build_wfst_from_state(_mixdom_state, t)


def build_mixdom_dynamic_wfst(t):
    _init_mixdom_dynamic()
    return _build_wfst_from_state(_mixdom_dyn_state, t)


def build_mixdom_singlet():
    _init_mixdom()
    return _build_singlet_from_state(_mixdom_state)


def build_mixdom_dynamic_singlet():
    _init_mixdom_dynamic()
    return _build_singlet_from_state(_mixdom_dyn_state)


# ======================================================================
# Gamma helpers
# ======================================================================

def load_gamma(family_id):
    path = os.path.join(GAMMA_DIR, f"{family_id}.G{G}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ======================================================================
# Per-holdout evaluation
# ======================================================================

_wfst_cache = {}


def run_holdout(fam_id, held_out, aligned_seqs, ungapped, tree, Q, pi,
                gamma_data, verbose=True):
    gamma_labels = gamma_data['labels']
    gamma_rates = np.array(gamma_data['rates'])

    _, branch_len = find_parent_of_leaf(tree, held_out)
    if branch_len is None:
        return None

    pruned_tree, _, _ = remove_leaf(tree, held_out)
    if pruned_tree is None:
        return None

    pruned_leaves = set(n.name for n in pruned_tree.leaves())
    remaining = [n for n in aligned_seqs if n != held_out and n in pruned_leaves]
    if len(remaining) < 3:
        return None

    C = len(next(iter(aligned_seqs.values())))
    pruned_msa = {}
    for n in remaining:
        seq = np.full(C, -1, dtype=np.int32)
        for j, ch in enumerate(aligned_seqs[n]):
            if ch in AA_TO_INT:
                seq[j] = AA_TO_INT[ch]
        pruned_msa[n] = seq

    true_seq = ungapped[held_out]

    # Infer internal presence for TVA
    name_internal_nodes(pruned_tree)
    leaf_presence = {n: np.array(pruned_msa[n] >= 0, dtype=bool) for n in pruned_msa}
    msa_presence = infer_internal_presence(pruned_tree, leaf_presence)
    root_name = pruned_tree.name
    root_pres = msa_presence.get(root_name, np.zeros(C, dtype=bool))

    # Eval columns: held-out present AND root present (matched)
    held_pres = np.array([aligned_seqs[held_out][c] in 'ACDEFGHIKLMNPQRSTVWY'
                          for c in range(C)], dtype=bool)
    eval_cols = np.where(held_pres & root_pres)[0]
    held_ungap = np.cumsum(held_pres) - 1

    if len(eval_cols) < 5:
        return None

    result = {
        'family': fam_id, 'held_out': held_out,
        'tau': float(branch_len), 'true_len': len(true_seq),
        'n_eval': int(len(eval_cols)),
    }

    def _accuracy(pred_at_col):
        if pred_at_col is None:
            return None
        n_correct = n_total = 0
        for c in eval_cols:
            h_idx = int(held_ungap[c])
            if h_idx < len(true_seq) and pred_at_col[c] >= 0:
                n_total += 1
                if pred_at_col[c] == true_seq[h_idx]:
                    n_correct += 1
        return n_correct / n_total if n_total >= 5 else None

    # --- Felsenstein ---
    try:
        _, fels_post = marginal_ancestor_all_columns_jax(pruned_tree, pruned_msa, Q, pi)
        fels_pred = np.argmax(np.asarray(fels_post), axis=1)
        result['fels'] = _accuracy(fels_pred)
    except Exception as e:
        result['fels'] = None

    # --- Felsenstein + gamma ---
    try:
        fels_g_pred = np.full(C, -1, dtype=np.int32)
        col_groups = {}
        for c in range(C):
            if not any(pruned_msa[n][c] >= 0 for n in pruned_msa):
                continue
            g = gamma_labels[c] if c < len(gamma_labels) and gamma_labels[c] >= 0 else -1
            col_groups.setdefault(g, []).append(c)
        for g, cols in col_groups.items():
            Q_g = np.array(ensure_rate_matrix(gamma_rates[g] * np.array(Q))) if g >= 0 else Q
            msa_g = {n: np.array([pruned_msa[n][c] for c in cols], dtype=np.int32)
                     for n in pruned_msa}
            _, post_g = marginal_ancestor_all_columns_jax(pruned_tree, msa_g, Q_g, pi)
            for i, c in enumerate(cols):
                fels_g_pred[c] = int(np.argmax(np.asarray(post_g)[i]))
        result['fels_gamma'] = _accuracy(fels_g_pred)
    except Exception as e:
        result['fels_gamma'] = None

    def _tva_to_pred(node_post_dict):
        """Extract per-column predictions from TVA root posteriors."""
        root_post = node_post_dict.get(root_name)
        if root_post is None or len(root_post) == 0:
            return None
        argmax = np.argmax(root_post, axis=1).astype(np.int32)
        pred = np.full(C, -1, dtype=np.int32)
        ungap_idx = 0
        for c in range(C):
            if root_pres[c]:
                if ungap_idx < len(argmax):
                    pred[c] = argmax[ungap_idx]
                ungap_idx += 1
        return pred

    def _build_tree_wfsts(wfst_builder, cache):
        """Build per-edge WFSTs for pruned tree with caching."""
        wfsts = {}
        for node in pruned_tree.preorder():
            if node.is_root:
                continue
            t = max(node.branch_length, 1e-4)
            t_key = round(t, 3)
            if t_key not in cache:
                cache[t_key] = wfst_builder(t)
            wfsts[(node.parent.name, node.name)] = cache[t_key]
        return wfsts

    # --- TVA with MixDom WFSTs ---
    try:
        mixdom_wfst = _build_tree_wfsts(build_mixdom_wfst, _wfst_cache)
        singlet = build_mixdom_singlet()
        node_post, _, _, _, _ = tree_varanc(
            pruned_tree, msa_presence, pruned_msa,
            mixdom_wfst, singlet, pi, n_iter=10, verbose=False)
        result['tva_mixdom'] = _accuracy(_tva_to_pred(node_post))
    except Exception as e:
        result['tva_mixdom'] = None
        if verbose:
            print(f"    TVA ERROR: {e}")

    # --- TVA + MixDom + gamma (block-diagonal, G rate-scaled MixDom WFSTs) ---
    try:
        wfst_per_g = []
        sing_per_g = []
        pi_per_g = []
        gamma_wfst_cache = {}
        for g in range(G):
            r_g = gamma_rates[g]
            # Rate-scale: distill at effective time r_g * t
            def _builder_g(t, _r=r_g):
                return build_mixdom_wfst(_r * t)
            wg = {}
            for node in pruned_tree.preorder():
                if node.is_root:
                    continue
                t = max(node.branch_length, 1e-4)
                cache_key = (g, round(t, 3))
                if cache_key not in gamma_wfst_cache:
                    gamma_wfst_cache[cache_key] = build_mixdom_wfst(r_g * t)
                wg[(node.parent.name, node.name)] = gamma_wfst_cache[cache_key]
            wfst_per_g.append(wg)
            sing_per_g.append(build_mixdom_singlet())
            pi_per_g.append(pi)

        gamma_mask = np.zeros((C, G), dtype=np.float64)
        for c in range(C):
            if c < len(gamma_labels) and gamma_labels[c] >= 0:
                gamma_mask[c, gamma_labels[c]] = 1.0
            else:
                gamma_mask[c, :] = 1.0 / G

        tva_g_post, _, _, _, _ = tree_varanc_block_diagonal(
            pruned_tree, msa_presence, pruned_msa,
            wfst_per_g, sing_per_g, pi_per_g,
            D=1, A=A, rate_multiplier_mask=gamma_mask,
            n_iter=10, verbose=False)
        result['tva_mixdom_gamma'] = _accuracy(_tva_to_pred(tva_g_post))
    except Exception as e:
        result['tva_mixdom_gamma'] = None
        if verbose:
            print(f"    TVA+γ ERROR: {e}")

    # --- TVA + MixDom + dynamic (D=2 classes) ---
    if 'params' in _mixdom_dyn_state:
        try:
            dyn_wfst_cache = {}
            dyn_wfst = _build_tree_wfsts(build_mixdom_dynamic_wfst, dyn_wfst_cache)
            dyn_singlet = build_mixdom_dynamic_singlet()
            node_post_d, _, _, _, _ = tree_varanc(
                pruned_tree, msa_presence, pruned_msa,
                dyn_wfst, dyn_singlet, pi, n_iter=10, verbose=False)
            result['tva_mixdom_dyn'] = _accuracy(_tva_to_pred(node_post_d))
        except Exception as e:
            result['tva_mixdom_dyn'] = None
            if verbose:
                print(f"    TVA+dyn ERROR: {e}")

        # --- TVA + MixDom + dynamic + gamma ---
        try:
            wfst_dg = []
            sing_dg = []
            pi_dg = []
            dg_cache = {}
            for g in range(G):
                r_g = gamma_rates[g]
                wg = {}
                for node in pruned_tree.preorder():
                    if node.is_root:
                        continue
                    t = max(node.branch_length, 1e-4)
                    ck = (g, round(t, 3))
                    if ck not in dg_cache:
                        dg_cache[ck] = build_mixdom_dynamic_wfst(r_g * t)
                    wg[(node.parent.name, node.name)] = dg_cache[ck]
                wfst_dg.append(wg)
                sing_dg.append(build_mixdom_dynamic_singlet())
                pi_dg.append(pi)

            tva_dg_post, _, _, _, _ = tree_varanc_block_diagonal(
                pruned_tree, msa_presence, pruned_msa,
                wfst_dg, sing_dg, pi_dg,
                D=1, A=A, rate_multiplier_mask=gamma_mask,
                n_iter=10, verbose=False)
            result['tva_mixdom_dyn_gamma'] = _accuracy(_tva_to_pred(tva_dg_post))
        except Exception as e:
            result['tva_mixdom_dyn_gamma'] = None
            if verbose:
                print(f"    TVA+dyn+γ ERROR: {e}")

    if verbose:
        def _f(k, label):
            v = result.get(k)
            return f"{label}={v:.3f}" if v is not None else f"{label}=ERR"
        parts = [_f('fels','F'), _f('fels_gamma','Fγ'),
                 _f('tva_mixdom','T'), _f('tva_mixdom_gamma','Tγ')]
        if 'tva_mixdom_dyn' in result:
            parts.extend([_f('tva_mixdom_dyn','Td'), _f('tva_mixdom_dyn_gamma','Tdγ')])
        print(f"    {held_out} (tau={branch_len:.3f}, n={len(eval_cols)}): {' '.join(parts)}")

    return result


ALL_METHODS = [
    ('fels', 'Fels'),
    ('fels_gamma', 'Fels+γ'),
    ('tva_mixdom', 'TVA+MixDom'),
    ('tva_mixdom_gamma', 'TVA+MixDom+γ'),
    ('tva_mixdom_dyn', 'TVA+MixDom+dyn'),
    ('tva_mixdom_dyn_gamma', 'TVA+MixDom+dyn+γ'),
]


def _print_summary(title, all_results, elapsed):
    # Use results where at least fels and one TVA method succeeded
    ok = [r for r in all_results if r.get('fels') is not None
          and r.get('tva_mixdom') is not None]
    if not ok:
        return
    n = len(ok)
    fels_vals = [r['fels'] for r in ok]
    fels_mean = np.mean(fels_vals)
    print(f"\n{'='*60}")
    print(f"{title} ({n} holdouts, {elapsed:.0f}s)")
    print(f"{'='*60}")
    for k, label in ALL_METHODS:
        vals = [r[k] for r in ok if r.get(k) is not None]
        if not vals:
            continue
        delta = np.mean(vals) - fels_mean
        se = np.std(vals) / len(vals)**0.5
        # Sign test vs Fels
        matched_fels = [r['fels'] for r in ok if r.get(k) is not None]
        w = sum(1 for f, o in zip(matched_fels, vals) if o > f)
        l = sum(1 for f, o in zip(matched_fels, vals) if o < f)
        t = len(vals) - w - l
        print(f"  {label:>20}: {np.mean(vals):.3f} ± {se:.3f}  "
              f"Δ={delta:+.4f}  w/t/l={w}/{t}/{l}")


def main():
    Q_lg, pi_lg = rate_matrix_lg()
    Q = np.asarray(Q_lg)
    pi = np.asarray(pi_lg)

    # Initialize MixDom (one-time cost)
    print("Initializing MixDom parameters...", flush=True)
    _init_mixdom()
    _init_mixdom_dynamic()
    has_dynamic = 'params' in _mixdom_dyn_state
    print(f"  Dynamic class model: {'loaded' if has_dynamic else 'not available'}")

    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    test_fams = splits['test']

    # Filter: gamma labels, moderate size, compute mean identity
    candidates = []
    for fam in test_fams:
        gd = load_gamma(fam)
        if gd is None:
            continue
        sto_path = os.path.join(PFAM_DIR, f"{fam}.sto")
        if not os.path.exists(sto_path):
            continue
        seqs = parse_sto(sto_path)
        n = len(seqs)
        if 5 <= n <= 12 and 50 <= gd['n_cols'] <= 200:
            # Compute mean pairwise identity
            names_list = list(seqs.keys())
            n_pairs = total_id = 0
            for ii in range(len(names_list)):
                for jj in range(ii+1, len(names_list)):
                    si, sj = seqs[names_list[ii]], seqs[names_list[jj]]
                    m = a = 0
                    for ci, cj in zip(si, sj):
                        if ci not in '.-' and cj not in '.-':
                            a += 1
                            if ci == cj: m += 1
                    if a > 0:
                        total_id += m / a
                        n_pairs += 1
            mean_id = total_id / max(n_pairs, 1)
            candidates.append((fam, n, gd['n_cols'], mean_id))

    # Sort by divergence (most divergent first)
    candidates.sort(key=lambda x: x[3])
    print(f"Candidates: {len(candidates)} (sorted by divergence)")
    if candidates:
        print(f"  Most divergent: {candidates[0][0]} id={candidates[0][3]:.2f}")
        print(f"  Least divergent: {candidates[-1][0]} id={candidates[-1][3]:.2f}")

    N = 15
    rng = np.random.RandomState(42)
    families = candidates[:N]  # take most divergent

    all_results = []
    t0 = time.time()

    for i, (fam, n_seqs, n_cols, mean_id) in enumerate(families):
        print(f"\n[{i+1}/{N}] {fam}: {n_seqs} seqs, {n_cols} cols, id={mean_id:.2f}", flush=True)

        aligned_seqs = parse_sto(os.path.join(PFAM_DIR, f"{fam}.sto"))
        names = list(aligned_seqs.keys())

        ungapped = {}
        for name in names:
            seq = [AA_TO_INT[ch] for ch in aligned_seqs[name] if ch in AA_TO_INT]
            ungapped[name] = np.array(seq, dtype=np.int32)

        try:
            nj_names, dist = msa_pairwise_distances(aligned_seqs, Q, pi)
            tree = neighbor_joining(dist, nj_names)
        except Exception as e:
            print(f"  SKIP: tree error: {e}")
            continue

        gamma_data = load_gamma(fam)
        holdouts = list(names)
        rng.shuffle(holdouts)
        holdouts = holdouts[:3]

        for held_out in holdouts:
            if len(ungapped.get(held_out, [])) < 10:
                continue
            r = run_holdout(fam, held_out, aligned_seqs, ungapped, tree,
                          Q, pi, gamma_data, verbose=True)
            if r is not None:
                all_results.append(r)

        if (i + 1) % 10 == 0 and all_results:
            _print_summary("Interim", all_results, time.time() - t0)

    # Final
    if all_results:
        _print_summary("FINAL", all_results, time.time() - t0)

    with open('experiments/pfam_recon_proper.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to experiments/pfam_recon_proper.json")


if __name__ == '__main__':
    main()
