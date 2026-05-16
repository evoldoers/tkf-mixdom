"""Annabel-format MixDom pair HMM constructor.

Loads parameters from Annabel's trained model directories and constructs
the MixDom pair HMM transition matrix and emission tables compatible with
our FSA pipeline.

Annabel's model uses:
- Top-level TKF91 (between domains)
- Per-domain TKF92 (within fragments) with domain-specific lambda, mu
- Per-(domain, fragment) extension probability r_extend
- Hierarchical mixture: domain -> fragment -> site class -> rate multiplier
- Per-(domain, fragment, site class) equilibrium distributions
- Shared GTR exchangeability matrix (for GTR models)

The emission probabilities are a mixture over site classes and rate
multipliers:
    P(a, b | M_df, t) = sum_s sum_r P(s|d,f) * P(r|d,f,s) *
        pi[d,f,s,a] * expm(rate_mult[d,f,s,r] * Q(S_exch, pi[d,f,s]) * t)[a,b]

The transition structure is identical to our MixDom model.
"""

import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from ..core.params import (
    S, M, I, D, E,
    tkf91_trans, tkf_beta, tkf_kappa,
)
from ..core.ctmc import transition_matrix
from ..models.mixdom import (
    build_nested_trans, state_types as mixdom_state_types,
    effective_trans_per_type, _UV_U, _UV_X, _IS_M_TYPE, _MID,
    n_states, MM, MI, MD, II, DD,
)


def load_annabel_params(params_dir):
    """Load all parameter files from an Annabel model directory.

    Args:
        params_dir: path to directory containing PARAMS-* files

    Returns:
        dict with keys:
            domain_class_probs: (C_dom,)
            frag_class_probs: (C_dom, C_frag)
            site_class_probs: (C_dom, C_frag, C_site)
            rate_mult_probs: (C_dom, C_frag, C_site, C_subrate)
            equilibriums: (C_dom, C_frag, C_site, 20)
            rate_multipliers: (C_dom, C_frag, C_site, C_subrate)
            gtr_exch: (20, 20) or None
            top_lambda: float
            top_mu: float
            frag_lambda: (C_dom,)
            frag_mu: (C_dom,)
            r_extend: (C_dom, C_frag)
            n_dom: int
            n_frag: int
            n_site: int
            n_subrate: int
    """
    d = {}

    d['domain_class_probs'] = np.load(
        os.path.join(params_dir, 'PARAMS-MAT_domain_class_probs.npy'))
    d['frag_class_probs'] = np.load(
        os.path.join(params_dir, 'PARAMS-MAT_frag_class_probs.npy'))
    d['site_class_probs'] = np.load(
        os.path.join(params_dir, 'PARAMS-MAT_site_class_probs.npy'))
    d['rate_mult_probs'] = np.load(
        os.path.join(params_dir, 'PARAMS-MAT_rate_mult_probs.npy'))
    d['equilibriums'] = np.load(
        os.path.join(params_dir, 'PARAMS-MAT_equilibriums-per-site-class.npy'))
    d['rate_multipliers'] = np.load(
        os.path.join(params_dir, 'PARAMS-MAT_rate_multipliers.npy'))

    gtr_path = os.path.join(params_dir, 'PARAMS-MAT_gtr-exchangeabilities.npy')
    if os.path.exists(gtr_path):
        d['gtr_exch'] = np.load(gtr_path)
    else:
        d['gtr_exch'] = None

    with open(os.path.join(params_dir,
              'PARAMS-DICT_top_level_tkf91_indel_params.pkl'), 'rb') as f:
        tkf91 = pickle.load(f)
    d['top_lambda'] = float(tkf91['lambda'])
    d['top_mu'] = float(tkf91['mu'])

    with open(os.path.join(params_dir,
              'PARAMS-DICT_fragment_tkf92_indel_params.pkl'), 'rb') as f:
        tkf92 = pickle.load(f)
    d['frag_lambda'] = np.asarray(tkf92['lambda'])
    d['frag_mu'] = np.asarray(tkf92['mu'])
    d['r_extend'] = np.asarray(tkf92['r_extend'])

    d['n_dom'] = d['domain_class_probs'].shape[0]
    d['n_frag'] = d['frag_class_probs'].shape[1]
    d['n_site'] = d['site_class_probs'].shape[2]
    d['n_subrate'] = d['rate_mult_probs'].shape[3]

    return d


def build_emission_tables(annabel_params, t):
    """Build per-(domain, fragment) emission log-probability tables.

    Pure JAX — safe inside jax.vmap. Uses transition_matrix +
    build_rate_matrix instead of scipy.expm.

    Returns:
        log_match_emit: (n_dom, n_frag, 21, 21) log P(a,b | match)
        log_ins_emit: (n_dom, n_frag, 21) log pi(a) for inserts
        log_del_emit: (n_dom, n_frag, 21) log pi(a) for deletes
    """
    # Annabel deliberately factorises evolutionary rate as
    # P(a -> b | t) = transition_matrix(Q, rate_mult * t) where Q is
    # built from per-(domain, fragment, site) (S_exch, pi_s) and
    # unit-normalised to mean rate 1. The absolute rate scale lives
    # entirely in `rate_multipliers` (see `_single_joint` below). Hence
    # unit normalisation here is intentional, not a bug — without it
    # `rate_multipliers` and the equilibrium-mean-rate of (S, pi_s)
    # would double-count rate scale. We pass acknowledged_lossy=True to
    # silence the UserWarning that the unnormalised-by-default policy
    # raises at all call sites of build_rate_matrix_unit_normalized.
    from functools import partial as _partial
    from ..core.ctmc import build_rate_matrix_unit_normalized
    _build_Q = _partial(build_rate_matrix_unit_normalized,
                        acknowledged_lossy=True)

    p = annabel_params
    n_dom, n_frag, n_site, n_subrate = p['n_dom'], p['n_frag'], p['n_site'], p['n_subrate']
    A = 20

    site_probs = jnp.array(p['site_class_probs'], dtype=jnp.float64)   # (D, F, S)
    rate_probs = jnp.array(p['rate_mult_probs'], dtype=jnp.float64)    # (D, F, S, R)
    pis_raw = jnp.array(p['equilibriums'][:, :, :, :A], dtype=jnp.float64)
    rate_mults = jnp.array(p['rate_multipliers'], dtype=jnp.float64)   # (D, F, S, R)
    S_exch = jnp.array(p['gtr_exch'][:A, :A], dtype=jnp.float64) if p['gtr_exch'] is not None \
             else jnp.ones((A, A), dtype=jnp.float64)

    pis = jnp.maximum(pis_raw, 1e-30)
    pis = pis / pis.sum(axis=-1, keepdims=True)

    # Flatten (D, F, S, R) → (N,) for vmap over all expm calls at once
    N = n_dom * n_frag * n_site * n_subrate
    pis_flat = jnp.broadcast_to(
        pis[:, :, :, None, :], (n_dom, n_frag, n_site, n_subrate, A)
    ).reshape(N, A)
    rates_flat = rate_mults.reshape(N)
    # weights = P(s|d,f) * P(r|d,f,s)
    weights_flat = (site_probs[:, :, :, None] * rate_probs).reshape(N)
    # (d, f) indices for segment_sum
    df_flat = jnp.arange(n_dom * n_frag).repeat(n_site * n_subrate)

    def _single_joint(pi_s, rm):
        Q_s = _build_Q(S_exch, pi_s)
        Pt = transition_matrix(Q_s, rm * t)
        return pi_s[:, None] * Pt  # (A, A) joint

    all_joints = jax.vmap(_single_joint)(pis_flat, rates_flat)  # (N, A, A)
    weighted = weights_flat[:, None, None] * all_joints

    # Sum per (d, f)
    match_emit = jax.ops.segment_sum(weighted, df_flat, n_dom * n_frag)  # (D*F, A, A)
    match_emit = match_emit.reshape(n_dom, n_frag, A, A)

    # Marginal pi per (d, f): sum_s P(s|d,f) * pi(d,f,s)
    marginal_pi = jnp.einsum('dfs,dfsa->dfa', site_probs, pis)  # (D, F, A)

    # Pad to 21 for wildcard
    log_match = jnp.log(jnp.maximum(match_emit, 1e-300))
    log_match = jnp.pad(log_match, ((0, 0), (0, 0), (0, 1), (0, 1)),
                        constant_values=0.0)
    log_ins = jnp.log(jnp.maximum(marginal_pi, 1e-300))
    log_ins = jnp.pad(log_ins, ((0, 0), (0, 0), (0, 1)),
                      constant_values=0.0)
    log_del = log_ins

    return log_match, log_ins, log_del


def build_pair_hmm_emissions(annabel_params, t, x_seq, y_seq):
    """Build full emission table for pair HMM.

    Args:
        annabel_params: dict from load_annabel_params
        t: evolutionary time
        x_seq: (Lx,) ancestor integer sequence
        y_seq: (Ly,) descendant integer sequence

    Returns:
        emit: (Lx+1, Ly+1, ns) log emission table
    """
    p = annabel_params
    n_dom = p['n_dom']
    n_frag = p['n_frag']

    log_match, log_ins, log_del = build_emission_tables(p, t)

    st = mixdom_state_types(n_dom, n_frag)
    ns = len(st)
    Lx = x_seq.shape[0]
    Ly = y_seq.shape[0]

    x_pad = jnp.concatenate([jnp.array([0]), x_seq])
    y_pad = jnp.concatenate([jnp.array([0]), y_seq])

    # Domain and fragment index for each state
    dom_idx = jnp.zeros(ns, dtype=jnp.int32)
    frag_idx = jnp.zeros(ns, dtype=jnp.int32)
    body = jnp.arange(ns - 2)
    block_size = 5 * n_frag
    dom_idx = dom_idx.at[2:].set(body // block_size)
    # Within a domain block: 5 state types, each with n_frag fragments
    # body state layout: for each domain d:
    #   MM_f0, MM_f1, ..., MI_f0, MI_f1, ..., MD_f0, ..., II_f0, ..., DD_f0, ...
    # So frag index = body % n_frag (within each block of 5*n_frag)
    frag_idx = frag_idx.at[2:].set(body % n_frag)

    # Per-state emission lookup
    # Match states: log_match[dom, frag, x[i], y[j]]
    state_match = log_match[dom_idx, frag_idx]  # (ns, 21, 21)
    match_emit = state_match[:, x_pad][:, :, y_pad]  # (ns, Lx+1, Ly+1)
    match_emit = match_emit.transpose(1, 2, 0)  # (Lx+1, Ly+1, ns)

    # Insert states: log_ins[dom, frag, y[j]]
    state_ins = log_ins[dom_idx, frag_idx]  # (ns, 21)
    ins_emit = state_ins[:, y_pad].T  # (Ly+1, ns)

    # Delete states: log_del[dom, frag, x[i]]
    state_del = log_del[dom_idx, frag_idx]  # (ns, 21)
    del_emit = state_del[:, x_pad].T  # (Lx+1, ns)

    is_M = (st == M)
    is_I = (st == I)
    is_D = (st == D)

    emit = (is_M[None, None, :] * match_emit +
            is_I[None, None, :] * ins_emit[None, :, :] +
            is_D[None, None, :] * del_emit[:, None, :])

    is_emit = is_M | is_I | is_D
    emit = jnp.where(is_emit[None, None, :], emit, 0.0)

    return emit


def build_annabel_transition_matrix(annabel_params, t):
    """Build the MixDom pair HMM transition matrix from Annabel's params.

    This delegates to our existing build_nested_trans, which implements
    the full eq:mixdom_transitions from the paper.

    Args:
        annabel_params: dict from load_annabel_params
        t: evolutionary time

    Returns:
        chi: (N, N) transition matrix (linear space)
        state_map: dict mapping (uv, d, f) -> flat index
    """
    p = annabel_params

    return build_nested_trans(
        main_ins_rate=p['top_lambda'],
        main_del_rate=p['top_mu'],
        t=t,
        dom_ins_rates=jnp.array(p['frag_lambda']),
        dom_del_rates=jnp.array(p['frag_mu']),
        dom_weights=jnp.array(p['domain_class_probs']),
        frag_weights=jnp.array(p['frag_class_probs']),
        ext_rates=jnp.array(p['r_extend']),
    )


def build_annabel_pair_hmm(params_dir, t):
    """Build pair HMM from Annabel's parameter directory.

    Args:
        params_dir: path to directory with PARAMS-* files
        t: evolutionary time

    Returns:
        log_trans: (N, N) log transition matrix
        st: (N,) state types array
        annabel_params: loaded parameter dict (for emission computation)
    """
    p = load_annabel_params(params_dir)
    chi, state_map = build_annabel_transition_matrix(p, t)
    log_trans = jnp.log(jnp.maximum(chi, 1e-300))
    st = mixdom_state_types(p['n_dom'], p['n_frag'])
    return log_trans, st, p


def make_fsa_params_annabel(annabel_params):
    """Convert Annabel params to the dict format expected by our FSA pipeline.

    Per-domain pi is computed by averaging the per-(d, f, s) equilibrium
    over (f, s) within each domain and passed as (n_dom, 20) so
    `_optimize_tau_mixdom` and `build_per_domain_sub_matrices` see
    domain-specific equilibria.

    Returns:
        fsa_params: dict with keys expected by pairwise_posteriors_mixdom
        n_dom: number of domains
        n_frag: number of fragments
    """
    p = annabel_params
    n_dom = p['n_dom']
    n_frag = p['n_frag']

    # Per-domain pi: pi_d(a) = sum_f P(f|d) sum_s P(s|d,f) pi[d,f,s,a]
    dom_pi = np.zeros((n_dom, 20))
    for d in range(n_dom):
        for f in range(n_frag):
            for s in range(p['n_site']):
                dom_pi[d] += (p['frag_class_probs'][d, f] *
                              p['site_class_probs'][d, f, s] *
                              p['equilibriums'][d, f, s])

    # For per-domain emissions, we need S_exch per domain.
    # With GTR shared exchangeability, all domains share the same S_exch;
    # the per-domain difference comes from pi.
    if p['gtr_exch'] is not None:
        S_exch_per_dom = np.tile(p['gtr_exch'][None], (n_dom, 1, 1))
    else:
        S_exch_per_dom = np.tile(
            (np.ones((20, 20)) - np.eye(20))[None], (n_dom, 1, 1))

    fsa_params = {
        'main_ins': float(p['top_lambda']),
        'main_del': float(p['top_mu']),
        'dom_ins': p['frag_lambda'],
        'dom_del': p['frag_mu'],
        'dom_weights': p['domain_class_probs'],
        'frag_weights': p['frag_class_probs'],
        'ext_rates': p['r_extend'],
        'S_exch': S_exch_per_dom,
        'pi': dom_pi,
    }

    return fsa_params, n_dom, n_frag


def pairwise_posteriors_annabel(x_seq, y_seq, annabel_params, t,
                                precomputed_emissions=None,
                                precomputed_log_trans=None):
    """Compute pairwise residue alignment posteriors using Annabel's MixDom.

    Uses a numpy-based forward-backward (no JIT) to avoid the extremely
    slow JAX trace/compilation for large state spaces. The FB runs in
    eager mode with numpy arrays for O(Lx * Ly * ns^2) work without
    any compilation overhead.

    Args:
        x_seq, y_seq: (Lx,), (Ly,) integer sequences
        annabel_params: dict from load_annabel_params
        t: evolutionary time
        precomputed_emissions: optional tuple (log_match, log_ins, log_del)
            from build_emission_tables, to avoid recomputing for each pair
            when t is the same.
        precomputed_log_trans: optional (ns, ns) numpy log transition matrix
            from build_annabel_transition_matrix, to avoid recomputing.

    Returns:
        match_posteriors: (Lx, Ly) P(residue i aligned to residue j)
        log_prob: log probability
    """
    p = annabel_params
    n_dom = p['n_dom']
    n_frag = p['n_frag']

    x = np.asarray(x_seq, dtype=np.int32)
    y = np.asarray(y_seq, dtype=np.int32)
    Lx = len(x)
    Ly = len(y)

    # Build transition matrix (use JAX for this small computation)
    if precomputed_log_trans is not None:
        log_chi = precomputed_log_trans
    else:
        chi, _ = build_annabel_transition_matrix(p, t)
        log_chi = np.asarray(jnp.log(jnp.maximum(chi, 1e-300)))

    # Build emission table with site-class mixture
    if precomputed_emissions is not None:
        log_match, log_ins, log_del = precomputed_emissions
    else:
        log_match, log_ins, log_del = build_emission_tables(p, t)

    log_match = np.asarray(log_match)
    log_ins = np.asarray(log_ins)
    log_del = np.asarray(log_del)

    st = np.asarray(mixdom_state_types(n_dom, n_frag))
    ns = len(st)

    # Build emission table (Lx+1, Ly+1, ns) in numpy
    emit = _build_emissions_numpy(
        log_match, log_ins, log_del, st, x, y, n_dom, n_frag)

    # Forward-backward in numpy (no JIT)
    log_prob, match_posteriors = _forward_backward_numpy(
        log_chi, st, emit, Lx, Ly)

    return match_posteriors, float(log_prob)


def _annabel_expected_ll(log_tau, n_trans_fixed, match_counts, insert_counts,
                         delete_counts,
                         main_ins, main_del, dom_ins, dom_del, dom_weights,
                         frag_weights, ext_rates,
                         # Site-class mixture params (all JAX arrays):
                         site_weights,    # (n_dom, n_site) = P(s|d) marginalized over frag
                         rate_weights,    # (n_dom, n_site, n_subrate) = P(r|d,s) marginalized
                         site_pis,        # (n_dom, n_site, A) = pi per (dom, site)
                         rate_mults_arr,  # (n_dom, n_site, n_subrate) = rate multipliers
                         S_exch):         # (A, A) shared GTR exchangeability
    """E[LL(tau)] for Annabel's MixDom with site-class mixture emissions.

    The emission term is:
      sum_d sum_{a,b} match_counts[d,a,b] * log(
        sum_{s,r} w_{d,s} * w_{d,s,r} * pi_{d,s}[a] * P(rate_r * tau, S, pi_{d,s})[a,b])

    All terms inside the log are differentiable via JAX (transition_matrix
    + build_rate_matrix are pure JAX). The log-of-sum structure means this is a
    proper mixture, not an approximation.
    """
    from functools import partial as _partial
    from ..models.mixdom import build_nested_trans
    # See `build_emission_tables` above for the rationale on intentional
    # unit normalisation: Annabel factors evolutionary rate as
    # `rate_mults * tau * Q` where Q is unit-normalised, so the absolute
    # rate scale lives in `rate_mults_arr`. acknowledged_lossy=True
    # suppresses the runtime UserWarning for this deliberate use.
    from ..core.ctmc import build_rate_matrix_unit_normalized
    _build_Q = _partial(build_rate_matrix_unit_normalized,
                        acknowledged_lossy=True)

    tau = jnp.exp(log_tau)

    # Transition term (same as standard MixDom)
    chi, _ = build_nested_trans(
        main_ins, main_del, tau,
        dom_ins, dom_del, dom_weights, frag_weights, ext_rates)
    log_chi = jnp.log(jnp.maximum(chi, 1e-300))
    trans_term = jnp.sum(n_trans_fixed * log_chi)

    # Emission term: site-class mixture
    n_dom = site_pis.shape[0]
    n_site = site_pis.shape[1]
    n_subrate = rate_mults_arr.shape[2]
    A = site_pis.shape[2]

    # Vectorized site-class mixture emission term.
    # Flatten (n_dom, n_site, n_subrate) into one vmap axis so JAX
    # compiles ONE transition_matrix and maps it, instead of
    # unrolling D*S*R copies in the XLA graph.
    n_flat = n_dom * n_site * n_subrate

    # Broadcast pis to (D, S, R, A) then flatten to (D*S*R, A)
    pis_flat = jnp.broadcast_to(
        site_pis[:, :, None, :],
        (n_dom, n_site, n_subrate, A)).reshape(n_flat, A)
    rates_flat = rate_mults_arr.reshape(n_flat)
    weights_flat = (site_weights[:, :, None] * rate_weights).reshape(n_flat)
    dom_flat = jnp.repeat(jnp.arange(n_dom), n_site * n_subrate)

    def _single_Pt(pi_s, rm):
        Q_s = _build_Q(S_exch, pi_s)
        return transition_matrix(Q_s, rm * tau)

    all_Pt = jax.vmap(_single_Pt)(pis_flat, rates_flat)  # (D*S*R, A, A)
    all_joint = weights_flat[:, None, None] * pis_flat[:, :, None] * all_Pt

    # Sum per domain using segment_sum
    joint_per_dom = jax.ops.segment_sum(all_joint, dom_flat, n_dom)  # (D, A, A)

    # Marginal pi per domain
    marginal_pi = jnp.einsum('ds,dsa->da', site_weights, site_pis)

    log_joint = jnp.log(jnp.maximum(joint_per_dom, 1e-300))
    log_marginal = jnp.log(jnp.maximum(marginal_pi, 1e-300))

    emit_match = jnp.sum(match_counts * log_joint)
    emit_ins = jnp.sum(insert_counts * log_marginal)
    emit_del = jnp.sum(delete_counts * log_marginal)

    return trans_term + emit_match + emit_ins + emit_del


_annabel_tau_grad = jax.jit(jax.grad(_annabel_expected_ll, argnums=0))
_annabel_tau_hess = jax.jit(jax.grad(jax.grad(_annabel_expected_ll, argnums=0),
                                      argnums=0))


def _build_emissions_numpy(log_match, log_ins, log_del, st, x, y, n_dom, n_frag):
    """Build emission table in numpy (no JIT).

    Args:
        log_match: (n_dom, n_frag, 21, 21) numpy
        log_ins: (n_dom, n_frag, 21) numpy
        log_del: (n_dom, n_frag, 21) numpy
        st: (ns,) state types numpy
        x: (Lx,) ancestor sequence
        y: (Ly,) descendant sequence
        n_dom, n_frag: model dimensions

    Returns:
        emit: (Lx+1, Ly+1, ns) numpy log emission table
    """
    ns = len(st)
    Lx = len(x)
    Ly = len(y)

    x_pad = np.concatenate([[0], x])
    y_pad = np.concatenate([[0], y])

    # Domain and fragment index for each state
    dom_idx = np.zeros(ns, dtype=np.int32)
    frag_idx = np.zeros(ns, dtype=np.int32)
    body = np.arange(ns - 2)
    block_size = 5 * n_frag
    dom_idx[2:] = body // block_size
    frag_idx[2:] = body % n_frag

    state_match = log_match[dom_idx, frag_idx]  # (ns, 21, 21)
    match_emit = state_match[:, x_pad][:, :, y_pad]  # (ns, Lx+1, Ly+1)
    match_emit = match_emit.transpose(1, 2, 0)  # (Lx+1, Ly+1, ns)

    state_ins = log_ins[dom_idx, frag_idx]  # (ns, 21)
    ins_emit = state_ins[:, y_pad].T  # (Ly+1, ns)

    state_del = log_del[dom_idx, frag_idx]  # (ns, 21)
    del_emit = state_del[:, x_pad].T  # (Lx+1, ns)

    is_M = (st == M)
    is_I = (st == I)
    is_D = (st == D)

    emit = np.zeros((Lx + 1, Ly + 1, ns))
    emit += is_M[None, None, :] * match_emit
    emit += is_I[None, None, :] * ins_emit[None, :, :]
    emit += is_D[None, None, :] * del_emit[:, None, :]

    is_emit = is_M | is_I | is_D
    emit = np.where(is_emit[None, None, :], emit, 0.0)

    return emit


def _logsumexp_np(a, axis=None):
    """Numerically stable logsumexp in numpy."""
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max_squeeze = np.max(a, axis=axis)
    # Handle all-inf case
    mask = np.isfinite(a_max_squeeze)
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    result = np.log(np.sum(np.exp(a - a_max_safe), axis=axis)) + a_max_squeeze
    return np.where(mask, result, -np.inf)


def _log_matmul(log_v, log_M):
    """Log-space matrix-vector multiply: result[j] = logsumexp_i(log_v[i] + log_M[i,j]).

    Args:
        log_v: (ns,) log vector
        log_M: (ns, ns_out) log matrix

    Returns:
        (ns_out,) result
    """
    # Use the max trick for stability
    combined = log_v[:, None] + log_M  # (ns, ns_out)
    c = np.max(combined, axis=0)  # (ns_out,)
    mask = np.isfinite(c)
    c_safe = np.where(mask, c, 0.0)
    result = np.log(np.sum(np.exp(combined - c_safe[None, :]), axis=0)) + c_safe
    return np.where(mask, result, -1e30)


def _forward_backward_numpy(log_trans, st, emit, Lx, Ly):
    """Forward-backward in numpy (no JIT). Returns match posteriors directly.

    This avoids JAX JIT compilation overhead for large state spaces.
    Uses optimized log-space matrix operations with precomputed transition
    sub-matrices for each state type.

    Args:
        log_trans: (ns, ns) numpy log transition matrix
        st: (ns,) state types
        emit: (Lx+1, Ly+1, ns) numpy log emissions
        Lx, Ly: sequence lengths

    Returns:
        log_prob: float
        match_posteriors: (Lx, Ly) numpy array
    """
    NEG_INF_NP = -1e30
    ns = len(st)
    is_M = (st == M)
    is_I = (st == I)
    is_D = (st == D)

    m_idx = np.where(is_M)[0]
    i_idx = np.where(is_I)[0]
    d_idx = np.where(is_D)[0]
    e_idx = int(np.where(st == E)[0][0])

    # Precompute transition sub-matrices for each state type
    # Forward: log_trans[:, m_idx], log_trans[:, i_idx], log_trans[:, d_idx]
    trans_to_M = log_trans[:, m_idx]  # (ns, |M|)
    trans_to_I = log_trans[:, i_idx]  # (ns, |I|)
    trans_to_D = log_trans[:, d_idx]  # (ns, |D|)

    # Backward: log_trans[k, m_idx] etc - transposed view
    trans_from_M = log_trans[:, m_idx]  # (ns, |M|) - same as forward
    trans_from_I = log_trans[:, i_idx]
    trans_from_D = log_trans[:, d_idx]

    # Forward pass
    F = np.full((Lx + 1, Ly + 1, ns), NEG_INF_NP)
    F[0, 0, S] = 0.0

    n_m = len(m_idx)
    n_i = len(i_idx)
    n_d = len(d_idx)

    for i in range(Lx + 1):
        for j in range(Ly + 1):
            if i == 0 and j == 0:
                continue

            # M-states: predecessor at (i-1, j-1)
            if i >= 1 and j >= 1 and n_m > 0:
                F[i, j, m_idx] = _log_matmul(F[i-1, j-1], trans_to_M) + emit[i, j, m_idx]

            # I-states: predecessor at (i, j-1)
            if j >= 1 and n_i > 0:
                F[i, j, i_idx] = _log_matmul(F[i, j-1], trans_to_I) + emit[i, j, i_idx]

            # D-states: predecessor at (i-1, j)
            if i >= 1 and n_d > 0:
                F[i, j, d_idx] = _log_matmul(F[i-1, j], trans_to_D) + emit[i, j, d_idx]

    log_prob = _log_matmul(F[Lx, Ly], log_trans[:, e_idx:e_idx+1])[0]

    # Backward pass
    B = np.full((Lx + 1, Ly + 1, ns), NEG_INF_NP)
    B[Lx, Ly, :] = log_trans[:, e_idx]

    for i in range(Lx, -1, -1):
        for j in range(Ly, -1, -1):
            if i == Lx and j == Ly:
                continue

            terms = np.full(ns, NEG_INF_NP)

            # M-successors at (i+1, j+1)
            if i < Lx and j < Ly and n_m > 0:
                succ_m = emit[i+1, j+1, m_idx] + B[i+1, j+1, m_idx]  # (|M|,)
                # log_trans[:, m_idx] + succ_m -> logsumexp over m -> (ns,)
                combined = trans_to_M + succ_m[None, :]  # (ns, |M|)
                c = np.max(combined, axis=1)
                mask = np.isfinite(c)
                c_safe = np.where(mask, c, 0.0)
                val = np.log(np.sum(np.exp(combined - c_safe[:, None]), axis=1)) + c_safe
                terms = np.where(mask, np.logaddexp(terms, val), terms)

            # I-successors at (i, j+1)
            if j < Ly and n_i > 0:
                succ_i = emit[i, j+1, i_idx] + B[i, j+1, i_idx]
                combined = trans_to_I + succ_i[None, :]
                c = np.max(combined, axis=1)
                mask = np.isfinite(c)
                c_safe = np.where(mask, c, 0.0)
                val = np.log(np.sum(np.exp(combined - c_safe[:, None]), axis=1)) + c_safe
                terms = np.where(mask, np.logaddexp(terms, val), terms)

            # D-successors at (i+1, j)
            if i < Lx and n_d > 0:
                succ_d = emit[i+1, j, d_idx] + B[i+1, j, d_idx]
                combined = trans_to_D + succ_d[None, :]
                c = np.max(combined, axis=1)
                mask = np.isfinite(c)
                c_safe = np.where(mask, c, 0.0)
                val = np.log(np.sum(np.exp(combined - c_safe[:, None]), axis=1)) + c_safe
                terms = np.where(mask, np.logaddexp(terms, val), terms)

            B[i, j, :] = terms

    # Match posteriors
    fb = F[1:Lx+1, 1:Ly+1, :] + B[1:Lx+1, 1:Ly+1, :]  # (Lx, Ly, ns)
    fb_match = np.where(is_M[None, None, :], fb, NEG_INF_NP)
    # vectorized logsumexp over states
    c = np.max(fb_match, axis=2)
    mask = np.isfinite(c)
    c_safe = np.where(mask, c, 0.0)
    match_post_log = np.log(np.sum(np.exp(fb_match - c_safe[:, :, None]), axis=2)) + c_safe
    match_post_log = np.where(mask, match_post_log, NEG_INF_NP)
    match_posteriors = np.exp(match_post_log - log_prob)

    return log_prob, match_posteriors


def pairwise_posteriors_annabel_jax(x_seq, y_seq, annabel_params, t,
                                     precomputed_emissions=None,
                                     precomputed_log_trans=None):
    """Like pairwise_posteriors_annabel but uses JAX forward_backward_2d.

    Slower on first call (JIT compile) but fast on subsequent calls
    with the same padded shape. No maxlen restriction.
    """
    from ..dp.hmm import forward_backward_2d, safe_log
    from ..models.mixdom import state_types as mixdom_state_types
    from ..core.params import M

    p = annabel_params
    n_dom = p['n_dom']
    n_frag = p['n_frag']
    st = mixdom_state_types(n_dom, n_frag)

    x = jnp.asarray(x_seq, dtype=jnp.int32)
    y = jnp.asarray(y_seq, dtype=jnp.int32)
    Lx, Ly = x.shape[0], y.shape[0]

    # Transition matrix
    if precomputed_log_trans is not None:
        log_chi = jnp.asarray(precomputed_log_trans)
    else:
        chi, _ = build_annabel_transition_matrix(p, t)
        log_chi = safe_log(chi)

    # Emission table
    emit = build_pair_hmm_emissions(p, t, x, y)

    # JAX forward-backward (wavefront DP, JIT-cached per padded shape)
    log_prob, posteriors, _ = forward_backward_2d(
        log_chi, st, x, y, None, None, log_emit_table=emit)

    # Extract match posteriors
    is_match = (st == M)
    match_posteriors = jnp.sum(
        posteriors[1:Lx+1, 1:Ly+1, :] * is_match[None, None, :],
        axis=-1)

    return np.asarray(match_posteriors), float(log_prob)


def precompute_annabel_nr_params(annabel_params):
    """Precompute the JAX arrays needed for the NR tau optimizer.

    Call once per model (not per pair). Returns a dict of JAX arrays
    that are passed as static args to the vmapped per-pair function.
    """
    p = annabel_params
    n_dom, n_frag, n_site, n_subrate = p['n_dom'], p['n_frag'], p['n_site'], p['n_subrate']
    A = 20

    site_probs_np = np.asarray(p['site_class_probs'])    # (D, F, S)
    frag_probs_np = np.asarray(p['frag_class_probs'])    # (D, F)
    equil_np = np.asarray(p['equilibriums'])[:, :, :, :A]
    rate_probs_np = np.asarray(p['rate_mult_probs'])
    rate_mults_np = np.asarray(p['rate_multipliers'])

    site_w = np.zeros((n_dom, n_site))
    site_pi = np.zeros((n_dom, n_site, A))
    rate_w = np.zeros((n_dom, n_site, n_subrate))
    rate_m = np.zeros((n_dom, n_site, n_subrate))
    for d in range(n_dom):
        for s in range(n_site):
            w_total = 0.0
            for f in range(n_frag):
                wf = frag_probs_np[d, f] * site_probs_np[d, f, s]
                w_total += wf
                site_pi[d, s] += wf * equil_np[d, f, s]
                rate_w[d, s] += wf * rate_probs_np[d, f, s]
                rate_m[d, s] += wf * rate_mults_np[d, f, s]
            site_w[d, s] = w_total
            if w_total > 0:
                site_pi[d, s] /= w_total
                rate_w[d, s] /= w_total
                rate_m[d, s] /= w_total
    for d in range(n_dom):
        site_w[d] /= site_w[d].sum() + 1e-30
        site_pi[d] = np.maximum(site_pi[d], 1e-30)
        site_pi[d] /= site_pi[d].sum(axis=-1, keepdims=True)
        for s in range(n_site):
            rate_w[d, s] /= rate_w[d, s].sum() + 1e-30

    S_exch = p['gtr_exch'] if p['gtr_exch'] is not None else np.ones((A, A))

    # Per-domain averaged pi and S_exch for collapsed MixDom NR
    # (used instead of full site-class mixture NR to avoid 2 TB Hessian)
    pis_avg = np.einsum('ds,dsa->da', site_w, site_pi)  # (n_dom, A)
    pis_avg = np.maximum(pis_avg, 1e-30)
    pis_avg = pis_avg / pis_avg.sum(axis=-1, keepdims=True)
    S_exch_3d = np.tile(S_exch[None, :A, :A], (n_dom, 1, 1))  # (n_dom, A, A)

    return {
        'main_ins': jnp.float64(p['top_lambda']),
        'main_del': jnp.float64(p['top_mu']),
        'dom_ins': jnp.array(p['frag_lambda'], dtype=jnp.float64),
        'dom_del': jnp.array(p['frag_mu'], dtype=jnp.float64),
        'dom_weights': jnp.array(p['domain_class_probs'], dtype=jnp.float64),
        'frag_weights': jnp.array(p['frag_class_probs'], dtype=jnp.float64),
        'ext_rates': jnp.array(p['r_extend'], dtype=jnp.float64),
        'site_w': jnp.array(site_w, dtype=jnp.float64),
        'rate_w': jnp.array(rate_w, dtype=jnp.float64),
        'site_pi': jnp.array(site_pi, dtype=jnp.float64),
        'rate_m': jnp.array(rate_m, dtype=jnp.float64),
        'S_exch': jnp.array(S_exch[:A, :A], dtype=jnp.float64),
        'S_exch_3d': jnp.array(S_exch_3d, dtype=jnp.float64),
        'pis_avg': jnp.array(pis_avg, dtype=jnp.float64),
    }


def _pairwise_posteriors_annabel_jax(x, y, real_Lx, real_Ly,
                                      annabel_params, nr_params,
                                      n_dom, n_frag, n_newton=5, tau_init=1.0):
    """vmap-safe core. Mirrors _pairwise_posteriors_mixdom_jax.

    x, y: (Lx_pad,), (Ly_pad,) padded sequences.
    real_Lx, real_Ly: traced jnp scalars of actual lengths.
    nr_params: precomputed JAX arrays from precompute_annabel_nr_params.
    """
    from ..dp.hmm import forward_backward_2d, safe_log
    from ..models.mixdom import state_types as mixdom_state_types
    from ..core.params import M

    st = mixdom_state_types(n_dom, n_frag)
    ns = len(st)
    A = 20
    Lx, Ly = x.shape[0], y.shape[0]
    is_match = (st == M)

    # Step 1: FB at tau_init
    chi0, _ = build_annabel_transition_matrix(annabel_params, tau_init)
    log_chi0 = safe_log(chi0)
    emit0 = build_pair_hmm_emissions(annabel_params, tau_init, x, y)
    _, posteriors0, n_trans0 = forward_backward_2d(
        log_chi0, st, x, y, None, None, log_emit_table=emit0,
        real_Lx=real_Lx, real_Ly=real_Ly)

    n_trans_fixed = jax.lax.stop_gradient(n_trans0)
    post_fixed = jax.lax.stop_gradient(posteriors0)

    # Reduce to per-domain counts (same as _optimize_tau_mixdom)
    is_M_j = (jnp.asarray(st) == 1).astype(jnp.float64)
    is_I_j = (jnp.asarray(st) == 2).astype(jnp.float64)
    is_D_j = (jnp.asarray(st) == 3).astype(jnp.float64)
    body_idx = jnp.maximum((jnp.arange(ns) - 2) // (5 * n_frag), 0)
    state_dom = jax.nn.one_hot(body_idx, n_dom, dtype=jnp.float64)
    X_oh = jax.nn.one_hot(x, A, dtype=jnp.float64)
    Y_oh = jax.nn.one_hot(y, A, dtype=jnp.float64)

    mask_M = state_dom * is_M_j[:, None]
    W_M = jnp.einsum('ijs,sd->dij', post_fixed[1:Lx+1, 1:Ly+1, :], mask_M)
    tmp_M = jnp.einsum('dij,ia->daj', W_M, X_oh)
    match_counts = jax.lax.stop_gradient(jnp.einsum('daj,jb->dab', tmp_M, Y_oh))

    mask_I = state_dom * is_I_j[:, None]
    I_per_j = jnp.einsum('ijs,sd->dj', post_fixed[0:Lx+1, 1:Ly+1, :], mask_I)
    insert_counts = jax.lax.stop_gradient(jnp.einsum('dj,ja->da', I_per_j, Y_oh))

    mask_D = state_dom * is_D_j[:, None]
    D_per_i = jnp.einsum('ijs,sd->di', post_fixed[1:Lx+1, 0:Ly+1, :], mask_D)
    delete_counts = jax.lax.stop_gradient(jnp.einsum('di,ia->da', D_per_i, X_oh))

    # Step 2: NR tau optimization using collapsed (per-domain averaged)
    # emissions. This uses _mixdom_tau_grad/_mixdom_tau_hess which only
    # vmap n_dom expm calls (not n_dom*n_site*n_subrate), avoiding the
    # ~2 TB Hessian compilation that the full site-class mixture requires.
    # The tau estimate is slightly approximate (ignores site classes) but
    # the final FB at tau_opt (step 3) uses the full mixture emissions.
    from ..tree.fsa_anneal import _mixdom_tau_grad, _mixdom_tau_hess

    nrp = nr_params
    log_tau = jnp.log(jnp.float64(tau_init))
    for _ in range(n_newton):
        g = _mixdom_tau_grad(
            log_tau, n_trans_fixed, match_counts, insert_counts, delete_counts,
            nrp['main_ins'], nrp['main_del'], nrp['dom_ins'], nrp['dom_del'],
            nrp['dom_weights'], nrp['frag_weights'], nrp['ext_rates'],
            nrp['S_exch_3d'], nrp['pis_avg'])
        h = _mixdom_tau_hess(
            log_tau, n_trans_fixed, match_counts, insert_counts, delete_counts,
            nrp['main_ins'], nrp['main_del'], nrp['dom_ins'], nrp['dom_del'],
            nrp['dom_weights'], nrp['frag_weights'], nrp['ext_rates'],
            nrp['S_exch_3d'], nrp['pis_avg'])
        safe_neg_h = jnp.where(jnp.abs(h) > 1e-10, -h, 1.0)
        step = jnp.clip(g / safe_neg_h, -1.0, 1.0)
        log_tau = log_tau + step
    tau_opt = jnp.exp(jnp.clip(log_tau, jnp.log(1e-4), jnp.log(10.0)))

    # Step 3: FB at optimal tau
    chi_opt, _ = build_annabel_transition_matrix(annabel_params, tau_opt)
    log_chi_opt = safe_log(chi_opt)
    emit_opt = build_pair_hmm_emissions(annabel_params, tau_opt, x, y)
    log_prob, posteriors, _ = forward_backward_2d(
        log_chi_opt, st, x, y, None, None, log_emit_table=emit_opt,
        real_Lx=real_Lx, real_Ly=real_Ly)

    match_posteriors = jnp.sum(
        posteriors[1:Lx+1, 1:Ly+1, :] * is_match[None, None, :], axis=-1)
    return match_posteriors, tau_opt, log_prob


def pairwise_posteriors_annabel_batched(xs, ys, real_Lxs, real_Lys,
                                         annabel_params, nr_params,
                                         n_dom, n_frag,
                                         n_newton=5, tau_init=1.0):
    """Batched vmap'd pairwise posteriors for Annabel's MixDom.

    xs, ys: (B, Lx_pad), (B, Ly_pad). real_Lxs, real_Lys: (B,) int32.
    nr_params: from precompute_annabel_nr_params (broadcast, not vmapped).
    """
    return jax.vmap(
        lambda x, y, lx, ly: _pairwise_posteriors_annabel_jax(
            x, y, lx, ly, annabel_params, nr_params, n_dom, n_frag,
            n_newton=n_newton, tau_init=tau_init),
        in_axes=(0, 0, 0, 0),
    )(xs, ys, real_Lxs, real_Lys)



def pairwise_posteriors_annabel_jax_tauopt(x_seq, y_seq, annabel_params,
                                            n_newton=5, tau_init=1.0):
    """Scalar wrapper. Delegates to the vmap-safe core."""
    x_j = jnp.asarray(x_seq, dtype=jnp.int32)
    y_j = jnp.asarray(y_seq, dtype=jnp.int32)
    nr_params = precompute_annabel_nr_params(annabel_params)
    mp, tau, lp = _pairwise_posteriors_annabel_jax(
        x_j, y_j,
        jnp.int32(x_j.shape[0]), jnp.int32(y_j.shape[0]),
        annabel_params, nr_params,
        annabel_params['n_dom'], annabel_params['n_frag'],
        n_newton=n_newton, tau_init=tau_init)
    return np.asarray(mp), float(tau), float(lp)
