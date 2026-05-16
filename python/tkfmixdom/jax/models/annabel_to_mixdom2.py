"""Convert Annabel's hierarchical-mixture parameters to MixDom2 flat-class format.

Exact import (no marginalization). Annabel's hierarchy:

    c_dom    ~ Categorical(domain_class_probs)              # (D,)
    c_frag   ~ Categorical(frag_class_probs[c_dom])         # (D, F)
    c_site   ~ Categorical(site_class_probs[c_dom, c_frag]) # (D, F, S)
    c_subrate ~ Categorical(rate_mult_probs[d,f,s])         # (D, F, S, R)
    pi[d,f,s,a]                                             # (D, F, S, A)
    rate_mult[d,f,s,r]                                      # (D, F, S, R)
    S_GTR shared, normalized inside Q construction

is mapped to MixDom2 (flat global classes) by enumerating C = D*F*S*R global
class indices, with classdist[d, f, c] non-zero only when c corresponds to
(d, f, *, *).

For a class c = (d, f, s, r):
    class_pis[c]     = pi[d, f, s]
    class_S_exch[c]  = (rate_mult[d,f,s,r] / mean_rate(S_GTR, pi[d,f,s])) * S_GTR
    classdist[d,f,c] = site_class_probs[d,f,s] * rate_mult_probs[d,f,s,r]

Then in MixDom2 emission:
    P(a, b | match, d, f, t) = sum_c classdist[d,f,c] · class_pis[c,a]
                                      · [exp(class_S_exch[c]·class_pis[c]·t)]_{a,b}

This reproduces Annabel's joint emission exactly (verified by
`verify_import_matches_annabel` below).
"""

import os
import numpy as np

from .annabel_mixdom import load_annabel_params

AA = 20


def annabel_to_mixdom2_params(params_dir):
    """Load Annabel params and convert to MixDom2 params dict.

    Args:
        params_dir: path to Annabel params directory (e.g.
            /home/shared/mixdom_params/GTR_3dom_3frag_3site)

    Returns:
        params dict compatible with the train_pfam.py val-eval path. Keys:
        main_ins, main_del (scalars), dom_ins, dom_del (D,), dom_weights (D,),
        frag_weights (D,F), ext_rates (D,F) — auto-converted to (D,F,F) diag
        downstream — dom_pis (D,A), dom_S_exch (D,A,A), dom_Qs (D,A,A) [legacy],
        class_pis (C,A), class_S_exch (C,A,A), classdist (D,F,C), n_classes (int).
    """
    p = load_annabel_params(params_dir)
    n_dom = p['n_dom']
    n_frag = p['n_frag']
    n_site = p['n_site']
    n_subrate = p['n_subrate']
    n_classes = n_dom * n_frag * n_site * n_subrate

    # Equilibrium frequencies (D, F, S, A) — clip + renormalize for safety
    eq = np.asarray(p['equilibriums'][:, :, :, :AA], dtype=np.float64)
    eq = np.maximum(eq, 1e-30)
    eq = eq / eq.sum(axis=-1, keepdims=True)

    site_probs = np.asarray(p['site_class_probs'], dtype=np.float64)   # (D, F, S)
    rate_probs = np.asarray(p['rate_mult_probs'], dtype=np.float64)    # (D, F, S, R)
    rate_mult = np.asarray(p['rate_multipliers'], dtype=np.float64)    # (D, F, S, R)

    if p['gtr_exch'] is not None:
        S_GTR = np.asarray(p['gtr_exch'][:AA, :AA], dtype=np.float64)
        # Symmetrize for safety (already should be symmetric)
        S_GTR = 0.5 * (S_GTR + S_GTR.T)
    else:
        # F81: S_exch = ones (off-diagonal) so Q_ij = pi_j for i!=j
        S_GTR = np.ones((AA, AA), dtype=np.float64) - np.eye(AA)

    def _flat_idx(d, f, s, r):
        return ((d * n_frag + f) * n_site + s) * n_subrate + r

    class_pis = np.zeros((n_classes, AA), dtype=np.float64)
    class_S_exch = np.zeros((n_classes, AA, AA), dtype=np.float64)
    classdist = np.zeros((n_dom, n_frag, n_classes), dtype=np.float64)

    for d in range(n_dom):
        for f in range(n_frag):
            for s in range(n_site):
                pi_dfs = eq[d, f, s]
                # Mean rate of unscaled GTR Q with this pi:
                # Q[i,i] = -sum_{j != i} S_GTR[i,j] * pi_dfs[j]
                # mean_rate = -sum_i pi_dfs[i] * Q[i,i] = sum_i pi_dfs[i] sum_j!=i S[i,j] pi[j]
                Q_off = S_GTR * pi_dfs[None, :]
                np.fill_diagonal(Q_off, 0.0)
                row_sums = Q_off.sum(axis=1)  # (A,) = -Q[i,i]
                mean_rate = float((pi_dfs * row_sums).sum())
                mean_rate = max(mean_rate, 1e-30)
                for r in range(n_subrate):
                    c = _flat_idx(d, f, s, r)
                    class_pis[c] = pi_dfs
                    # class_S_exch[c]·class_pis[c]·t in MixDom2 is the per-class
                    # rate matrix exponent. We want it equal to
                    #   rate_mult[d,f,s,r] · Q_normalized(pi_dfs) · t
                    # where Q_normalized = S_GTR · pi / mean_rate. So
                    #   class_S_exch[c] = (rate_mult[d,f,s,r] / mean_rate) · S_GTR
                    scale = rate_mult[d, f, s, r] / mean_rate
                    class_S_exch[c] = scale * S_GTR
                    classdist[d, f, c] = site_probs[d, f, s] * rate_probs[d, f, s, r]

    # Sanity: classdist row sums (over c) should be 1 for each (d,f)
    row_sums = classdist.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(
            f"classdist row sums not 1: min={row_sums.min():.6f}, "
            f"max={row_sums.max():.6f}")

    # Top-level + per-domain TKF rates / weights / fragment weights
    main_ins = float(p['top_lambda'])
    main_del = float(p['top_mu'])
    dom_ins = np.asarray(p['frag_lambda'], dtype=np.float64)
    dom_del = np.asarray(p['frag_mu'], dtype=np.float64)
    dom_weights = np.asarray(p['domain_class_probs'], dtype=np.float64)
    frag_weights = np.asarray(p['frag_class_probs'], dtype=np.float64)
    ext_rates = np.asarray(p['r_extend'], dtype=np.float64)  # (D, F)

    # Legacy MixDom1 fields — populate with sensible per-domain marginals
    # (these are used only by the MixDom1 code path; under MixDom2 they are
    # ignored once classdist/class_pis/class_S_exch are present).
    marg_pi_df = np.einsum('dfs,dfsa->dfa', site_probs, eq)  # (D, F, A)
    dom_pis = np.einsum('df,dfa->da', frag_weights, marg_pi_df)  # (D, A)
    dom_pis = dom_pis / dom_pis.sum(axis=-1, keepdims=True)
    dom_S_exch = np.tile(S_GTR[None], (n_dom, 1, 1))
    dom_Qs = np.zeros((n_dom, AA, AA))
    for d in range(n_dom):
        Q = S_GTR * dom_pis[d, None, :]
        np.fill_diagonal(Q, 0.0)
        Q[np.diag_indices(AA)] = -Q.sum(axis=1)
        dom_Qs[d] = Q

    return {
        'main_ins': main_ins,
        'main_del': main_del,
        'dom_ins': dom_ins,
        'dom_del': dom_del,
        'dom_weights': dom_weights,
        'frag_weights': frag_weights,
        'ext_rates': ext_rates,
        'dom_Qs': dom_Qs,
        'dom_pis': dom_pis,
        'dom_S_exch': dom_S_exch,
        'class_S_exch': class_S_exch,
        'class_pis': class_pis,
        'classdist': classdist,
        'n_classes': n_classes,
    }


def verify_import_matches_annabel(params_dir, t=0.5, atol=1e-8, rtol=1e-6):
    """Verify the imported MixDom2 params reproduce Annabel's per-(d,f) joint
    match emission `pi · expm(Q·t)` exactly (modulo float precision).

    Compares:
      MixDom2 reconstructed: sum_c classdist[d,f,c] · pi_c[a] · [expm(Q_c t)]_{ab}
      Annabel direct:        from build_emission_tables(p, t) → log_match (D, F, A, A)

    Both should equal Annabel's hierarchical mixture joint.
    """
    import os
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import jax
    import jax.numpy as jnp
    from .annabel_mixdom import build_emission_tables, load_annabel_params
    from ..core.ctmc import transition_matrix

    p_ann = load_annabel_params(params_dir)
    p_mxd = annabel_to_mixdom2_params(params_dir)

    # Annabel direct
    log_m_ann, _, _ = build_emission_tables(p_ann, t)
    joint_ann = np.exp(np.asarray(log_m_ann[:, :, :AA, :AA]))  # (D, F, A, A)

    # MixDom2 reconstructed via our class formulation
    n_dom = p_mxd['classdist'].shape[0]
    n_frag = p_mxd['classdist'].shape[1]
    n_classes = p_mxd['n_classes']

    # Build per-class joint = pi_c[a] · [expm(Q_c t)]_{ab}
    def _per_class_joint(c):
        pi_c = jnp.asarray(p_mxd['class_pis'][c])
        S_c = jnp.asarray(p_mxd['class_S_exch'][c])
        # Build Q from S * pi (off-diag) + diag adjustment
        Q_off = S_c * pi_c[None, :]
        Q_off = Q_off - jnp.diag(jnp.diag(Q_off))
        Q = Q_off - jnp.diag(Q_off.sum(axis=1))
        Pt = transition_matrix(Q, t)
        return pi_c[:, None] * Pt  # (A, A)

    per_class = np.stack([np.asarray(_per_class_joint(c)) for c in range(n_classes)])
    cd = p_mxd['classdist']  # (D, F, C)
    joint_mxd = np.einsum('dfc,cab->dfab', cd, per_class)

    diff = np.abs(joint_mxd - joint_ann)
    rel = diff / np.maximum(np.abs(joint_ann), 1e-30)
    max_abs = float(diff.max())
    max_rel = float(rel.max())
    ok = max_abs < atol or max_rel < rtol

    return {
        'ok': bool(ok),
        'max_abs_diff': max_abs,
        'max_rel_diff': max_rel,
        'shape': joint_ann.shape,
        'n_classes': n_classes,
    }


if __name__ == '__main__':
    import sys
    pdir = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/shared/mixdom_params/GTR_3dom_3frag_3site'
    print(f"Importing {pdir}")
    res = verify_import_matches_annabel(pdir, t=0.5)
    print(res)
    print("OK" if res['ok'] else "MISMATCH")
