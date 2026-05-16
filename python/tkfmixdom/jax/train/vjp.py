"""Custom VJP wrappers for TKF Pair HMM log-likelihood.

Provides JAX-differentiable log P(x,y | params) where the backward pass
uses Forward-Backward expected counts + analytic score function derivatives,
avoiding costly autodiff through the full DP.

Usage:
    log_prob = tkf91_log_prob(ins_rate, del_rate, t, Q, pi, x_seq, y_seq)
    grad_fn = jax.grad(tkf91_log_prob, argnums=(0, 1))
    d_lambda, d_mu = grad_fn(ins_rate, del_rate, t, Q, pi, x_seq, y_seq)

    # MixDom: differentiable w.r.t. all 8 structural parameters
    log_prob = mixdom_log_prob(main_ins, main_del, t, dom_ins, dom_del,
                               dom_weights, frag_weights, ext_rates,
                               Q, pi, x_seq, y_seq)
"""

import jax
import jax.numpy as jnp
from ..core.params import S, M, I, D, E, tkf91_trans, tkf92_trans
from ..core.bdi import score_derivatives, transition_count_groups
from ..core.ctmc import transition_matrix, ctmc_log_prior
from ..dp.hmm import forward_2d, forward_backward_2d, safe_log
from ..models.mixdom import build_nested_trans, state_types as mixdom_state_types


def _emission_gradient(posteriors, state_types, sub_matrix, pi, x_seq, y_seq):
    """Compute d(log P)/d(sub_matrix) from FB posteriors.

    For each match at (i,j): emission = pi[a] * sub[a,b]
    d(log emission)/d(sub[a,b]) = 1 / sub[a,b]
    Weighted by posterior: d(log P)/d(sub[a,b]) = Σ w_ij / sub[a,b]

    Fully vectorized for JIT compatibility.
    """
    A = sub_matrix.shape[0]
    Lx = x_seq.shape[0]
    Ly = y_seq.shape[0]

    # Sum match-type posteriors at each (i,j)
    m_mask = (state_types == M)  # (ns,)
    # posteriors shape: (Lx+1, Ly+1, ns)
    match_weights = jnp.sum(posteriors[1:Lx+1, 1:Ly+1, :] * m_mask[None, None, :],
                             axis=2)  # (Lx, Ly)

    # Scatter match weights into A×A matrix using sequence indices
    # match_counts[a,b] = Σ_{i,j: x[i]=a, y[j]=b} match_weights[i,j]
    x_idx = x_seq[:, None].astype(jnp.int32)  # (Lx, 1)
    y_idx = y_seq[None, :].astype(jnp.int32)  # (1, Ly)
    flat_idx = x_idx * A + y_idx  # (Lx, Ly)
    match_counts = jnp.zeros(A * A).at[flat_idx.ravel()].add(
        match_weights.ravel()).reshape(A, A)

    # d(log P)/d(sub[a,b]) = match_counts[a,b] / sub[a,b]
    return match_counts / jnp.maximum(sub_matrix, 1e-30)


def _pi_emission_gradient(posteriors, state_types, pi, x_seq, y_seq):
    """Compute d(log P)/d(pi) from per-character emission posteriors.

    Each emission state contains pi[char] as a multiplicative factor:
        Match  at (i, j, s with type M): emit ∝ pi[x[i-1]] · sub[x[i-1], y[j-1]]
        Insert at (i, j, s with type I): emit ∝ pi[y[j-1]]
        Delete at (i, j, s with type D): emit ∝ pi[x[i-1]]
    For each event with character c=a: d log emit / d pi[a] = 1 / pi[a].
    Posterior-weighted aggregate:
        d log P / d pi[a] = (match_row_count_a + insert_count_a +
                             delete_count_a) / pi[a].

    Reachability ranges (i, j indexed into the (Lx+1) x (Ly+1) FB grid):
        M: i ∈ [1, Lx], j ∈ [1, Ly]
        I: i ∈ [0, Lx], j ∈ [1, Ly]   (no x consumed at i=0)
        D: i ∈ [1, Lx], j ∈ [0, Ly]   (no y consumed at j=0)

    Note: the chain through sub[a,b] = expm(t·Q)_ab (when sub_matrix is
    built as transition_matrix) is NOT captured here — that flows
    through the autograd path on sub_matrix in the wrapper.
    """
    A = pi.shape[0]
    Lx = x_seq.shape[0]
    Ly = y_seq.shape[0]

    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    # Match: pi[x[i-1]] factor at (i ∈ [1,Lx], j ∈ [1,Ly])
    m_post_ij = jnp.sum(posteriors[1:Lx+1, 1:Ly+1, :] * is_M[None, None, :],
                         axis=2)  # (Lx, Ly)
    m_per_i = jnp.sum(m_post_ij, axis=1)  # (Lx,) — posterior summed over j
    match_count_per_a = jnp.zeros(A).at[x_seq.astype(jnp.int32)].add(m_per_i)

    # Insert: pi[y[j-1]] factor at (i ∈ [0,Lx], j ∈ [1,Ly])
    ins_post_ij = jnp.sum(posteriors[0:Lx+1, 1:Ly+1, :] * is_I[None, None, :],
                           axis=2)  # (Lx+1, Ly)
    ins_per_j = jnp.sum(ins_post_ij, axis=0)  # (Ly,) — summed over i
    ins_count_per_a = jnp.zeros(A).at[y_seq.astype(jnp.int32)].add(ins_per_j)

    # Delete: pi[x[i-1]] factor at (i ∈ [1,Lx], j ∈ [0,Ly])
    del_post_ij = jnp.sum(posteriors[1:Lx+1, 0:Ly+1, :] * is_D[None, None, :],
                           axis=2)  # (Lx, Ly+1)
    del_per_i = jnp.sum(del_post_ij, axis=1)  # (Lx,) — summed over j
    del_count_per_a = jnp.zeros(A).at[x_seq.astype(jnp.int32)].add(del_per_i)

    return (match_count_per_a + ins_count_per_a + del_count_per_a) / \
           jnp.maximum(pi, 1e-30)


def tkf91_log_prob(ins_rate, del_rate, t, Q, pi, x_seq, y_seq):
    """Compute log P(x, y | lambda, mu, t, Q, pi) under TKF91.

    Differentiable w.r.t. ins_rate, del_rate (custom VJP via BDI score)
    AND Q, pi (autograd through emission computation / matrix exponential).

    The custom VJP wraps only the DP; the emission computation (which
    depends on Q via expm) is outside the VJP scope, so JAX autodiffs
    through it normally.
    """
    sub_matrix = transition_matrix(Q, t)
    return _tkf91_dp(ins_rate, del_rate, t, sub_matrix, pi, x_seq, y_seq)


@jax.custom_vjp
def _tkf91_dp(ins_rate, del_rate, t, sub_matrix, pi, x_seq, y_seq):
    """Inner DP with custom VJP for structural params only.

    sub_matrix is treated as a leaf (no custom gradient) — its gradient
    flows through JAX's standard autograd from the outer tkf91_log_prob.
    """
    log_trans = safe_log(tkf91_trans(ins_rate, del_rate, t))
    state_types = jnp.array([S, M, I, D, E])
    log_prob, _ = forward_2d(log_trans, state_types, x_seq, y_seq, sub_matrix, pi)
    return log_prob


def _tkf91_dp_fwd(ins_rate, del_rate, t, sub_matrix, pi, x_seq, y_seq):
    log_trans = safe_log(tkf91_trans(ins_rate, del_rate, t))
    state_types = jnp.array([S, M, I, D, E])
    log_prob, posteriors, n_trans = forward_backward_2d(
        log_trans, state_types, x_seq, y_seq, sub_matrix, pi)
    # Compute d(log P)/d(sub_matrix) and d(log P)/d(pi) from FB posteriors.
    # The sub_matrix gradient covers the chain through P(t) emissions; the
    # pi gradient covers the per-character pi[a] factor in M / I / D states.
    d_sub = _emission_gradient(posteriors, state_types, sub_matrix, pi, x_seq, y_seq)
    d_pi = _pi_emission_gradient(posteriors, state_types, pi, x_seq, y_seq)
    return log_prob, (n_trans, d_sub, d_pi, ins_rate, del_rate, t)


def _tkf91_dp_bwd(res, g):
    """Backward: BDI score for structural params + emission gradient for
    sub_matrix and the direct-pi insert/delete/match factor.

    The sub_matrix gradient further chains back through autograd in the
    outer wrapper to give the P(t)-mediated part of d(log P)/d(Q) and
    d(log P)/d(pi).

    Phase 5b: chi-side d_t added. Until Phase 5b, the inner custom_vjp
    returned None for t, dropping the `Σ n_ij · ∂log τ_ij/∂t` chi-side
    contribution silently. The substitution-side flow (through
    transition_matrix → sub_matrix) was captured via outer
    autograd, but the chi-side was zero. Public-API consumers like
    FSA's per-pair t-maximization needed the full t gradient.
    """
    n_trans, d_sub, d_pi, ins_rate, del_rate, t = res

    groups = transition_count_groups(n_trans)
    derivs = score_derivatives(ins_rate, del_rate, t)

    d_lambda = jnp.zeros(())
    d_mu = jnp.zeros(())
    for name in groups:
        dl, dm = derivs[name]
        d_lambda = d_lambda + groups[name] * dl
        d_mu = d_mu + groups[name] * dm

    # Chi-side d_t = Σ_ij n_ij · ∂log τ_ij/∂t. Autograd through
    # tkf91_trans handles the analytic derivative (it's a smooth
    # function of α, β, γ, κ which are explicit functions of t).
    def _chi_q_at_t(t_):
        return jnp.sum(n_trans * safe_log(tkf91_trans(ins_rate, del_rate, t_)))
    d_t = jax.grad(_chi_q_at_t)(t)

    # 7 args: ins_rate, del_rate, t, sub_matrix, pi, x_seq, y_seq
    return (g * d_lambda, g * d_mu, g * d_t, g * d_sub, g * d_pi, None, None)


_tkf91_dp.defvjp(_tkf91_dp_fwd, _tkf91_dp_bwd)


def tkf91_log_prob_cond(ins_rate, del_rate, t, Q, pi, x_seq, y_seq):
    """Compute log P(x, y | |ancestor|, params) under TKF91.

    Conditioned on ancestor length. Differentiable w.r.t. all params.
    """
    sub_matrix = transition_matrix(Q, t)
    return _tkf91_cond_dp(ins_rate, del_rate, t, sub_matrix, pi, x_seq, y_seq)


@jax.custom_vjp
def _tkf91_cond_dp(ins_rate, del_rate, t, sub_matrix, pi, x_seq, y_seq):
    from ..core.params import tkf91_trans_cond
    log_trans = safe_log(tkf91_trans_cond(ins_rate, del_rate, t))
    state_types = jnp.array([S, M, I, D, E])
    log_prob, _ = forward_2d(log_trans, state_types, x_seq, y_seq, sub_matrix, pi)
    return log_prob


def _tkf91_cond_dp_fwd(ins_rate, del_rate, t, sub_matrix, pi, x_seq, y_seq):
    from ..core.params import tkf91_trans_cond
    log_trans = safe_log(tkf91_trans_cond(ins_rate, del_rate, t))
    state_types = jnp.array([S, M, I, D, E])
    log_prob, posteriors, n_trans = forward_backward_2d(
        log_trans, state_types, x_seq, y_seq, sub_matrix, pi)
    d_sub = _emission_gradient(posteriors, state_types, sub_matrix, pi, x_seq, y_seq)
    d_pi = _pi_emission_gradient(posteriors, state_types, pi, x_seq, y_seq)
    return log_prob, (n_trans, d_sub, d_pi, ins_rate, del_rate, t)


def _tkf91_cond_dp_bwd(res, g):
    n_trans, d_sub, d_pi, ins_rate, del_rate, t = res
    groups = dict(transition_count_groups(n_trans))
    derivs = score_derivatives(ins_rate, del_rate, t)
    groups['log_kappa'] = 0.0
    groups['log_1mkappa'] = 0.0
    d_lambda = jnp.zeros(())
    d_mu = jnp.zeros(())
    for name in groups:
        dl, dm = derivs[name]
        d_lambda = d_lambda + groups[name] * dl
        d_mu = d_mu + groups[name] * dm
    # Phase 5b: chi-side d_t (autograd through tkf91_trans_cond).
    from ..core.params import tkf91_trans_cond
    def _chi_q_at_t(t_):
        return jnp.sum(n_trans * safe_log(
            tkf91_trans_cond(ins_rate, del_rate, t_)))
    d_t = jax.grad(_chi_q_at_t)(t)
    return (g * d_lambda, g * d_mu, g * d_t, g * d_sub, g * d_pi, None, None)


_tkf91_cond_dp.defvjp(_tkf91_cond_dp_fwd, _tkf91_cond_dp_bwd)


def tkf92_log_prob(ins_rate, del_rate, t, ext, Q, pi, x_seq, y_seq):
    """Compute log P(x, y | lambda, mu, t, ext, Q, pi) under TKF92.

    Differentiable w.r.t. all params: ins_rate, del_rate, ext (custom VJP),
    Q, pi (autograd through expm).
    """
    sub_matrix = transition_matrix(Q, t)
    return _tkf92_dp(ins_rate, del_rate, t, ext, sub_matrix, pi, x_seq, y_seq)


@jax.custom_vjp
def _tkf92_dp(ins_rate, del_rate, t, ext, sub_matrix, pi, x_seq, y_seq):
    log_trans = safe_log(tkf92_trans(ins_rate, del_rate, t, ext))
    state_types = jnp.array([S, M, I, D, E])
    log_prob, _ = forward_2d(log_trans, state_types, x_seq, y_seq, sub_matrix, pi)
    return log_prob


def _tkf92_dp_fwd(ins_rate, del_rate, t, ext, sub_matrix, pi, x_seq, y_seq):
    log_trans = safe_log(tkf92_trans(ins_rate, del_rate, t, ext))
    state_types = jnp.array([S, M, I, D, E])
    log_prob, posteriors, n_trans = forward_backward_2d(
        log_trans, state_types, x_seq, y_seq, sub_matrix, pi)
    d_sub = _emission_gradient(posteriors, state_types, sub_matrix, pi, x_seq, y_seq)
    d_pi = _pi_emission_gradient(posteriors, state_types, pi, x_seq, y_seq)
    return log_prob, (n_trans, d_sub, d_pi, ins_rate, del_rate, t, ext)


def _tkf92_dp_bwd(res, g):
    """Backward for TKF92: BDI score + ext gradient + emission gradients
    for sub_matrix and the per-character pi factor."""
    n_trans, d_sub, d_pi, ins_rate, del_rate, t, ext = res
    tau91 = tkf91_trans(ins_rate, del_rate, t)

    n91_trans = jnp.array(n_trans)
    for s_idx in [M, I, D]:
        p_s = tau91[s_idx, s_idx]
        tau92_ss = ext + (1.0 - ext) * p_s
        n91_self = jnp.where(tau92_ss > 1e-30,
                             n_trans[s_idx, s_idx] * (1.0 - ext) * p_s / tau92_ss,
                             0.0)
        n91_trans = n91_trans.at[s_idx, s_idx].set(n91_self)

    groups = transition_count_groups(n91_trans)
    derivs = score_derivatives(ins_rate, del_rate, t)

    d_lambda = jnp.zeros(())
    d_mu = jnp.zeros(())
    for name in groups:
        dl, dm = derivs[name]
        d_lambda = d_lambda + groups[name] * dl
        d_mu = d_mu + groups[name] * dm

    d_ext = jnp.zeros(())
    for s_idx in [M, I, D]:
        n_ss = n_trans[s_idx, s_idx]
        n_s = n_trans[s_idx].sum()
        p_s = tau91[s_idx, s_idx]
        tau92_ss = ext + (1.0 - ext) * p_s
        d_ext = d_ext + jnp.where(tau92_ss > 1e-30,
                                   n_ss * (1.0 - p_s) / tau92_ss, 0.0)
        d_ext = d_ext - jnp.where(1.0 - ext > 1e-10,
                                   (n_s - n_ss) / (1.0 - ext), 0.0)

    # Phase 5b: chi-side d_t. Autograd through tkf92_trans handles the
    # analytic ∂log τ_92/∂t (smooth function of α, β, γ, κ for each ext).
    def _chi_q_at_t(t_):
        return jnp.sum(n_trans * safe_log(
            tkf92_trans(ins_rate, del_rate, t_, ext)))
    d_t = jax.grad(_chi_q_at_t)(t)

    # 8 args: ins, del, t, ext, sub_matrix, pi, x, y
    return (g * d_lambda, g * d_mu, g * d_t, g * d_ext, g * d_sub, g * d_pi, None, None)


_tkf92_dp.defvjp(_tkf92_dp_fwd, _tkf92_dp_bwd)


# --- MixDom (nested Pair HMM) ---

def _chi_weighted_loglik(main_ins_rate, main_del_rate, t,
                         dom_ins_rates, dom_del_rates, dom_weights,
                         frag_weights, ext_rates, n_chi):
    """Compute sum_ij n_chi[i,j] * log chi[i,j](theta).

    This is the 'score function' whose gradient w.r.t. structural params
    gives d(log P)/d(theta) via the identity:
        d(log P)/d(theta) = sum_ij E[n_ij] * d(log chi_ij)/d(theta)
    """
    chi, _ = build_nested_trans(main_ins_rate, main_del_rate, t,
                                dom_ins_rates, dom_del_rates, dom_weights,
                                frag_weights, ext_rates)
    log_chi = safe_log(chi)
    return jnp.sum(n_chi * log_chi)


def mixdom_log_prob(main_ins_rate, main_del_rate, t,
                    dom_ins_rates, dom_del_rates, dom_weights,
                    frag_weights, ext_rates,
                    Q, pi, x_seq, y_seq):
    """Compute log P(x, y | params) under nested MixDom Pair HMM.

    Differentiable w.r.t. all structural parameters and the substitution
    model (Q, pi). The chi-related part uses a custom VJP via the
    forward-backward score identity. Q's contribution is via the
    substitution matrix P(t), which is constructed in this wrapper and
    then passed as a leaf into the inner DP — JAX autograd through
    transition_matrix(Q, t) handles d/d Q. pi additionally
    enters the inner DP directly (insert/delete/match emission factors)
    and that contribution is captured analytically in the inner VJP.

    Args:
        main_ins_rate, main_del_rate: top-level (domain) indel rates
        t: evolutionary time
        dom_ins_rates: (n_dom,) per-domain insertion rates
        dom_del_rates: (n_dom,) per-domain deletion rates
        dom_weights: (n_dom,) domain mixture weights
        frag_weights: (n_dom, n_frag) fragment weights per domain
        ext_rates: (n_dom, n_frag) fragment extension probabilities
        Q: (A, A) substitution rate matrix
        pi: (A,) equilibrium distribution
        x_seq, y_seq: integer sequences

    Returns:
        log_prob: scalar log-likelihood
    """
    sub_matrix = transition_matrix(Q, t)
    return _mixdom_dp(main_ins_rate, main_del_rate, t,
                      dom_ins_rates, dom_del_rates, dom_weights,
                      frag_weights, ext_rates,
                      sub_matrix, pi, x_seq, y_seq)


@jax.custom_vjp
def _mixdom_dp(main_ins_rate, main_del_rate, t,
               dom_ins_rates, dom_del_rates, dom_weights,
               frag_weights, ext_rates,
               sub_matrix, pi, x_seq, y_seq):
    """Inner MixDom1 DP with custom VJP. sub_matrix is treated as a leaf."""
    chi, _ = build_nested_trans(main_ins_rate, main_del_rate, t,
                                dom_ins_rates, dom_del_rates, dom_weights,
                                frag_weights, ext_rates)
    n_dom = dom_ins_rates.shape[0]
    n_frag = frag_weights.shape[1]
    st = mixdom_state_types(n_dom, n_frag)
    log_chi = safe_log(chi)
    log_prob, _ = forward_2d(log_chi, st, x_seq, y_seq, sub_matrix, pi)
    return log_prob


def _mixdom_dp_fwd(main_ins_rate, main_del_rate, t,
                   dom_ins_rates, dom_del_rates, dom_weights,
                   frag_weights, ext_rates,
                   sub_matrix, pi, x_seq, y_seq):
    """Forward: build chi, run FB, save n_chi + emission gradient inputs."""
    chi, _ = build_nested_trans(main_ins_rate, main_del_rate, t,
                                dom_ins_rates, dom_del_rates, dom_weights,
                                frag_weights, ext_rates)
    n_dom = dom_ins_rates.shape[0]
    n_frag = frag_weights.shape[1]
    st = mixdom_state_types(n_dom, n_frag)
    log_chi = safe_log(chi)
    log_prob, posteriors, n_chi = forward_backward_2d(
        log_chi, st, x_seq, y_seq, sub_matrix, pi)
    d_sub = _emission_gradient(posteriors, st, sub_matrix, pi, x_seq, y_seq)
    d_pi = _pi_emission_gradient(posteriors, st, pi, x_seq, y_seq)
    res = (n_chi, d_sub, d_pi,
           main_ins_rate, main_del_rate, t,
           dom_ins_rates, dom_del_rates, dom_weights,
           frag_weights, ext_rates)
    return log_prob, res


def _mixdom_dp_bwd(res, g):
    """Backward: chi-related score identity + emission gradients.

    Chi-related (8 structural params except t): autograd through
    build_nested_trans of n_chi-weighted log chi (cheap, small).
    sub_matrix: from match-emission posteriors.
    pi (direct factor): from per-character emission posteriors.
    """
    (n_chi, d_sub, d_pi,
     main_ins_rate, main_del_rate, t,
     dom_ins_rates, dom_del_rates, dom_weights,
     frag_weights, ext_rates) = res

    # Phase 5b: include argnum 2 (t) so the chi-side d_t is captured.
    # `_chi_weighted_loglik` builds chi via `build_nested_trans(...)` and
    # multiplies by n_chi; autograd through this gives the analytic
    # ∂(Σ n_ij · log τ_ij)/∂t = chi-side d_t.
    grad_fn = jax.grad(_chi_weighted_loglik, argnums=(0, 1, 2, 3, 4, 5, 6, 7))
    g0, g1, g2, g3, g4, g5, g6, g7 = grad_fn(
        main_ins_rate, main_del_rate, t,
        dom_ins_rates, dom_del_rates, dom_weights,
        frag_weights, ext_rates, n_chi)

    # 12 args: main_ins, main_del, t, dom_ins, dom_del, dom_w, frag_w, ext,
    #          sub_matrix, pi, x_seq, y_seq
    return (g * g0, g * g1, g * g2,
            g * g3, g * g4, g * g5, g * g6, g * g7,
            g * d_sub, g * d_pi, None, None)


_mixdom_dp.defvjp(_mixdom_dp_fwd, _mixdom_dp_bwd)


# --- MixDom2 (per-fragment site-class mixture) ---

def _build_class_Q(class_S_exch_c, class_pi_c):
    """Build a single per-class rate matrix from S_exch and pi.

    Q[i, j] = S_exch[i, j] · pi[j] for i ≠ j;
    Q[i, i] = -Σ_{j≠i} Q[i, j].

    NOT rate-normalised by mean rate (matches the SVI-BW MixDom2 M-step
    convention; see CLAUDE.md "not rate-normalized — the paper's M-step").
    """
    Q = class_S_exch_c * class_pi_c[None, :]
    Q = Q - jnp.diag(jnp.diag(Q))           # zero diagonal
    Q = Q - jnp.diag(Q.sum(axis=1))         # negative sum of off-diagonals
    return Q


def _build_class_subs(class_S_exch, class_pis, t):
    """Vmapped per-class P(t) builder using the fast eigh-based expm.

    Lives OUTSIDE the @jax.custom_vjp on _mixdom2_dp. The inner VJP
    treats `class_subs` as a leaf input (analogous to how the inner
    _tkf91_dp / _tkf92_dp / _mixdom_dp treat sub_matrix). Autograd
    through this builder handles the chain `(class_S_exch, class_pis)
    → Q_c → P_c` automatically; the inner bwd only needs to deliver
    `d log P / d class_subs`.

    `transition_matrix` symmetrises Q internally before eigh.
    For a reversible Q (which is what build_class_Q produces with a
    symmetric S_exch), the symmetrisation is a no-op on the forward
    value, and the symmetrised-projected directional derivative is
    consistent with the unconstrained Jacobian whenever forward and
    backward go through the SAME function (which they do, post-
    refactor). Tests pin the wrapper-level gradient to autograd
    through the same chain, so we get the speed of eigh without the
    wrong-Jacobian pitfall that motivated the earlier general-expm
    detour.

    Args:
        class_S_exch: (C, A, A) symmetric exchangeabilities.
        class_pis:    (C, A) per-class equilibria.
        t:            scalar evolutionary time.

    Returns:
        class_subs: (C, A, A) per-class transition matrices P_c(t).
    """
    Q_c = jax.vmap(_build_class_Q)(class_S_exch, class_pis)  # (C, A, A)
    sub_c = jax.vmap(lambda Q, p: transition_matrix(Q, t))(
        Q_c, class_pis)
    return sub_c


def _mixdom2_emission_counts(class_subs, class_pis, class_dist,
                               posteriors, log_emit, x_pad, y_pad,
                               real_Lx, real_Ly, n_dom, n_frag):
    """Compute the three per-class emission sufficient statistics needed
    for the inner VJP gradients on (class_subs, class_pis, class_dist).

    Returns:
        mc_c: (C, A, A) per-class match counts at character pair (a, b)
              — gradient of the Q-function w.r.t. log class_subs[c, a, b].
        V_c_linear: (C, A) per-class character-emission counts contributed
              by the LINEAR pi factor in M / I / D emissions only (does
              NOT include the chain through P_c — that part is captured
              by autograd in the outer wrapper through expm).
        classdist_counts: (D, F, C) per-(d, f, c) posterior class
              assignment counts — gradient w.r.t. log classdist[d, f, c].

    Inputs:
        class_subs:  (C, A, A) per-class P_c(t). LEAF in the inner VJP;
                     gradient through expm/Q chain is autograd's job.
        class_pis:   (C, A) per-class stationary distributions.
        class_dist:  (D, F, C).
        posteriors:  (Lx_pad+1, Ly_pad+1, ns) FB posteriors (real region
                     non-zero, padded region zero).
        log_emit:    (Lx_pad+1, Ly_pad+1, ns) log emission table from fwd.
        x_pad, y_pad: padded sequences; the FB grid index i maps to
                     character x_pad[i-1] for i ≥ 1 (dummy 0 at i=0).
        real_Lx, real_Ly: real (unpadded) sequence lengths.
        n_dom, n_frag.
    """
    n_cls, A = class_pis.shape
    ns = posteriors.shape[2]
    Lx_pad = x_pad.shape[0]
    Ly_pad = y_pad.shape[0]

    # ratio_post[i, j, s] = post / exp(log_emit). At masked (out-of-real)
    # positions, post = 0 and log_emit = NEG_INF; gate to avoid 0 * inf.
    safe_log_emit = jnp.where(jnp.isfinite(log_emit), log_emit, 0.0)
    ratio_post = jnp.where(
        posteriors > 0.0,
        posteriors * jnp.exp(-safe_log_emit),
        0.0)                                                        # (Lx+1, Ly+1, ns)

    # Per-state (d, f, type) lookup.  Body states start at index 2:
    body = jnp.arange(ns - 2)
    dom_idx = body // (5 * n_frag)
    within_dom = body % (5 * n_frag)
    frag_idx = within_dom % n_frag
    uv_idx = within_dom // n_frag       # 0..4 → MM, MI, MD, II, DD
    M_mask_body = (uv_idx == 0)
    I_mask_body = (uv_idx == 1) | (uv_idx == 3)
    D_mask_body = (uv_idx == 2) | (uv_idx == 4)

    Lx_pad_p1, Ly_pad_p1 = posteriors.shape[0], posteriors.shape[1]

    def _aggregate_by_df(mask_body):
        sel = ratio_post[:, :, 2:] * mask_body[None, None, :]
        flat_df = (dom_idx * n_frag + frag_idx).astype(jnp.int32)
        nDF = n_dom * n_frag
        one_hot = (flat_df[:, None] == jnp.arange(nDF)[None, :]).astype(sel.dtype)
        out_flat = sel @ one_hot
        return out_flat.reshape(Lx_pad_p1, Ly_pad_p1, n_dom, n_frag)

    ratio_df_M = _aggregate_by_df(M_mask_body)
    ratio_df_I = _aggregate_by_df(I_mask_body)
    ratio_df_D = _aggregate_by_df(D_mask_body)

    cd = class_dist  # (D, F, C)

    # Per-class emission factors at each grid point. x_at_i[i] is the
    # emitted character at grid row i (dummy 0 at i=0); same for y_at_j.
    x_at_i = jnp.concatenate([jnp.array([0], dtype=x_pad.dtype), x_pad])
    y_at_j = jnp.concatenate([jnp.array([0], dtype=y_pad.dtype), y_pad])

    pi_x = class_pis[:, x_at_i]                         # (C, Lx+1)
    P_xy = class_subs[:, x_at_i, :][:, :, y_at_j]       # (C, Lx+1, Ly+1)
    f_M = pi_x[:, :, None] * P_xy                       # (C, Lx+1, Ly+1)
    f_I = class_pis[:, y_at_j][:, None, :]              # (C, 1, Ly+1)
    f_D = class_pis[:, x_at_i][:, :, None]              # (C, Lx+1, 1)

    # Per-class per-(i, j) post-weighted contribution per type.
    cdc_M = jnp.einsum('ijdf,dfc->ijc', ratio_df_M, cd)
    cdc_I = jnp.einsum('ijdf,dfc->ijc', ratio_df_I, cd)
    cdc_D = jnp.einsum('ijdf,dfc->ijc', ratio_df_D, cd)
    wM = cdc_M.transpose(2, 0, 1) * f_M                 # (C, Lx+1, Ly+1)
    wI = cdc_I.transpose(2, 0, 1) * f_I
    wD = cdc_D.transpose(2, 0, 1) * f_D

    # Per-class match counts mc_c[c, a, b] via scatter-add by
    # (x_at_i[i], y_at_j[j]).
    one_hot_x = (x_at_i[:, None] == jnp.arange(A)[None, :]).astype(wM.dtype)
    one_hot_y = (y_at_j[:, None] == jnp.arange(A)[None, :]).astype(wM.dtype)
    mc_c = jnp.einsum('cij,ia,jb->cab', wM, one_hot_x, one_hot_y)

    # Per-class linear-pi character counts V_c_linear[c, a]:
    # match contributes at x_at_i, insert at y_at_j, delete at x_at_i.
    Vc_M = jnp.einsum('cij,ia->ca', wM, one_hot_x)
    Vc_I = jnp.einsum('cij,ja->ca', wI, one_hot_y)
    Vc_D = jnp.einsum('cij,ia->ca', wD, one_hot_x)
    V_c_linear = Vc_M + Vc_I + Vc_D

    # classdist counts: cd_count[d, f, c] = cd[d, f, c] · Σ_{ν, i, j}
    #     ratio_df_ν[i, j, d, f] · f_ν(c, i, j).
    sum_M_factor = jnp.einsum('ijdf,cij->dfc', ratio_df_M, f_M)
    sum_I_factor = jnp.einsum('ijdf,cij->dfc',
                               ratio_df_I,
                               jnp.broadcast_to(f_I, (n_cls, Lx_pad_p1, Ly_pad_p1)))
    sum_D_factor = jnp.einsum('ijdf,cij->dfc',
                               ratio_df_D,
                               jnp.broadcast_to(f_D, (n_cls, Lx_pad_p1, Ly_pad_p1)))
    classdist_counts = cd * (sum_M_factor + sum_I_factor + sum_D_factor)

    return mc_c, V_c_linear, classdist_counts


def mixdom2_log_prob(main_ins_rate, main_del_rate, t,
                     dom_ins_rates, dom_del_rates, dom_weights,
                     frag_weights, ext_rates,
                     class_pis, class_S_exch, class_dist,
                     x_seq, y_seq):
    """log P(x, y) under MixDom2 (per-fragment site-class mixture).

    Differentiable via custom VJP w.r.t. all structural and emission
    parameters using the score identity:

        d log P / d θ = E_q[d log P_complete / d θ]

    - Chi-related params (main_ins/main_del/dom_ins/dom_del/dom_w/frag_w/
      ext): autograd of n_chi-weighted log chi (cheap, small matrix).
    - Emission params (class_pis, class_S_exch, class_dist): autograd of
      posterior-weighted log emit (Q_emit). Autograd flows through the
      per-class expm and per-class emission scan but NOT through the FB
      recursion — the expensive 2D DP work was already done in fwd.
    - t: returned as None (t is per-pair from data, not optimised in Adam).

    Per-state emissions are class-mixtures:
        emit_M[i, j, s with (d, f)] = log Σ_c class_dist[d,f,c]
                                            · pi_c[x[i]] · P_c[x[i], y[j]]
        emit_I[i, j, s with (d, f)] = log Σ_c class_dist[d,f,c] · pi_c[y[j]]
        emit_D[i, j, s with (d, f)] = log Σ_c class_dist[d,f,c] · pi_c[x[i]]
    where P_c = expm(t · Q_c), Q_c = (S_exch_c · pi_c)_off-diag with the
    standard negative-row-sum diagonal (no rate normalisation — matches
    SVI-BW MixDom2; see CLAUDE.md "not rate-normalized").

    Args mirror train_pfam's MixDom2 layout:
        main_ins_rate, main_del_rate: top-level TKF91 rates (scalars).
        t: scalar evolutionary time.
        dom_ins_rates, dom_del_rates: (D,) per-domain TKF91 rates.
        dom_weights: (D,) domain mixture weights.
        frag_weights: (D, F) fragment mixture weights per domain.
        ext_rates: (D, F) or (D, F, F) fragment extension probs.
        class_pis: (C, A) per-class equilibrium distributions.
        class_S_exch: (C, A, A) per-class symmetric exchangeabilities.
        class_dist: (D, F, C) per-(d, f) class mixing distribution.
        x_seq, y_seq: integer sequences (no padding required).
    """
    # Outer wrapper: build per-class subs OUTSIDE the custom VJP. Autograd
    # through this builder handles d/d class_S_exch and the expm-mediated
    # part of d/d class_pis (i.e. the chain class_pis → Q_c → P_c). The
    # inner VJP only delivers d/d class_subs and the LINEAR-pi part of
    # d/d class_pis; both are closed-form posterior counts.
    class_subs = _build_class_subs(class_S_exch, class_pis, t)
    return _mixdom2_dp(main_ins_rate, main_del_rate, t,
                        dom_ins_rates, dom_del_rates, dom_weights,
                        frag_weights, ext_rates,
                        class_subs, class_pis, class_dist,
                        x_seq, y_seq)


@jax.custom_vjp
def _mixdom2_dp(main_ins_rate, main_del_rate, t,
                 dom_ins_rates, dom_del_rates, dom_weights,
                 frag_weights, ext_rates,
                 class_subs, class_pis, class_dist,
                 x_seq, y_seq):
    """Inner DP with custom VJP. class_subs is a LEAF (no custom grad);
    its gradient flows through autograd in the outer wrapper via
    _build_class_subs → expm. class_pis is also an input to this inner
    DP (it appears as a LINEAR factor in the per-class emission, in
    addition to the indirect path through class_subs); the inner bwd
    returns ONLY the linear-factor contribution for class_pis, and the
    outer wrapper's autograd through _build_class_subs adds the
    expm-chain contribution.
    """
    from ..dp.hmm import (NEG_INF, _emit_mask, _find_e_idx,
                           _forward_2d_core_diag, _pad_seq, _pad_to_bin,
                           pair_hmm_emissions_per_class)

    chi, _ = build_nested_trans(main_ins_rate, main_del_rate, t,
                                dom_ins_rates, dom_del_rates, dom_weights,
                                frag_weights, ext_rates)
    n_dom = dom_ins_rates.shape[0]
    n_frag = frag_weights.shape[1]
    st = mixdom_state_types(n_dom, n_frag)
    log_chi = safe_log(chi)

    Lx = x_seq.shape[0]
    Ly = y_seq.shape[0]
    Lx_pad = _pad_to_bin(Lx)
    Ly_pad = _pad_to_bin(Ly)
    x_pad = _pad_seq(x_seq, Lx_pad)
    y_pad = _pad_seq(y_seq, Ly_pad)

    emit = pair_hmm_emissions_per_class(
        st, x_pad, y_pad, class_subs, class_pis, class_dist, n_dom, n_frag)
    mask = _emit_mask(Lx, Ly, Lx_pad, Ly_pad, st.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)

    log_prob_pad, F_pad = _forward_2d_core_diag(log_chi, st, emit, Lx_pad, Ly_pad)
    e_idx = _find_e_idx(st)
    log_prob = jax.nn.logsumexp(F_pad[Lx, Ly, :] + log_chi[:, e_idx])
    return log_prob


def _mixdom2_dp_fwd(main_ins_rate, main_del_rate, t,
                      dom_ins_rates, dom_del_rates, dom_weights,
                      frag_weights, ext_rates,
                      class_subs, class_pis, class_dist,
                      x_seq, y_seq):
    """Forward: build chi, emit (using class_subs leaf), run FB for
    n_chi + posteriors. Save residuals for bwd.
    """
    from ..dp.hmm import (NEG_INF, _emit_mask, _pad_seq, _pad_to_bin,
                           forward_backward_2d, pair_hmm_emissions_per_class)

    chi, _ = build_nested_trans(main_ins_rate, main_del_rate, t,
                                dom_ins_rates, dom_del_rates, dom_weights,
                                frag_weights, ext_rates)
    n_dom = dom_ins_rates.shape[0]
    n_frag = frag_weights.shape[1]
    st = mixdom_state_types(n_dom, n_frag)
    log_chi = safe_log(chi)

    Lx = x_seq.shape[0]
    Ly = y_seq.shape[0]
    Lx_pad = _pad_to_bin(Lx)
    Ly_pad = _pad_to_bin(Ly)
    x_pad = _pad_seq(x_seq, Lx_pad)
    y_pad = _pad_seq(y_seq, Ly_pad)

    emit = pair_hmm_emissions_per_class(
        st, x_pad, y_pad, class_subs, class_pis, class_dist, n_dom, n_frag)
    mask = _emit_mask(Lx, Ly, Lx_pad, Ly_pad, st.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)

    log_prob, posteriors, n_chi = forward_backward_2d(
        log_chi, st, x_pad, y_pad, sub_matrix=jnp.eye(1), pi=jnp.ones(1),
        log_emit_table=emit, real_Lx=Lx, real_Ly=Ly)

    res = (n_chi, posteriors, emit, x_pad, y_pad, Lx, Ly,
           main_ins_rate, main_del_rate, t,
           dom_ins_rates, dom_del_rates, dom_weights,
           frag_weights, ext_rates,
           class_subs, class_pis, class_dist)
    return log_prob, res


def _mixdom2_dp_bwd(res, g):
    """Backward: chi-gradient via score identity on n_chi-weighted log
    chi; emission-gradient via closed-form per-class counts.

      d log P / d class_subs[c, a, b] = mc_c[c, a, b] / class_subs[c, a, b]
      d log P / d class_pis[c, a]  (linear-factor only)
                                    = V_c_linear[c, a] / class_pis[c, a]
      d log P / d class_dist[d, f, c] = classdist_counts / class_dist

    The remaining gradient pieces (the chain `class_subs → expm →
    (class_S_exch, class_pis-via-Q)`) are delivered by autograd in the
    outer wrapper through _build_class_subs.
    """
    (n_chi, posteriors, log_emit, x_pad, y_pad, real_Lx, real_Ly,
     main_ins_rate, main_del_rate, t,
     dom_ins_rates, dom_del_rates, dom_weights,
     frag_weights, ext_rates,
     class_subs, class_pis, class_dist) = res

    n_dom = dom_ins_rates.shape[0]
    n_frag = frag_weights.shape[1]

    # Chi gradient (existing score identity pattern). Phase 5b: include
    # argnum 2 (t) so chi-side ∂log τ/∂t is captured. Substitution-side
    # t flow comes via outer autograd through _build_class_subs.
    chi_grad_fn = jax.grad(_chi_weighted_loglik,
                            argnums=(0, 1, 2, 3, 4, 5, 6, 7))
    g0, g1, g2, g3, g4, g5, g6, g7 = chi_grad_fn(
        main_ins_rate, main_del_rate, t,
        dom_ins_rates, dom_del_rates, dom_weights,
        frag_weights, ext_rates, n_chi)

    # Per-class emission counts → closed-form gradients.
    mc_c, V_c_linear, cd_count = _mixdom2_emission_counts(
        class_subs, class_pis, class_dist,
        posteriors, log_emit, x_pad, y_pad, real_Lx, real_Ly,
        n_dom, n_frag)

    g_class_subs = mc_c / jnp.maximum(class_subs, 1e-30)
    g_class_pis_linear = V_c_linear / jnp.maximum(class_pis, 1e-30)
    g_class_dist = cd_count / jnp.maximum(class_dist, 1e-30)

    # 13 args: main_ins, main_del, t, dom_ins, dom_del, dom_w, frag_w,
    # ext, class_subs, class_pis, class_dist, x_seq, y_seq
    return (g * g0, g * g1, g * g2,
            g * g3, g * g4, g * g5, g * g6, g * g7,
            g * g_class_subs, g * g_class_pis_linear, g * g_class_dist,
            None, None)


_mixdom2_dp.defvjp(_mixdom2_dp_fwd, _mixdom2_dp_bwd)


# --- Log-likelihood + log-prior wrappers ---

def gamma_log_prior(x, shape=2.0, rate=10.0):
    """Log Gamma(shape, rate) prior: (shape-1)*log(x) - rate*x + const."""
    return (shape - 1.0) * jnp.log(x) - rate * x


def tkf91_log_posterior(ins_rate, del_rate, t, Q, pi, x_seq, y_seq,
                        ins_prior=(2.0, 10.0), del_prior=(2.0, 10.0),
                        subst_gamma_shape=2.0, subst_gamma_rate=1.0):
    """Log P(x,y | params) + log prior(params).

    Priors:
    - Gamma(shape, rate) on ins_rate and del_rate
    - Gamma(subst_gamma_shape, subst_gamma_rate) on each off-diagonal CTMC rate
    Differentiable via custom VJP for the likelihood term.
    """
    ll = tkf91_log_prob(ins_rate, del_rate, t, Q, pi, x_seq, y_seq)
    lp = (gamma_log_prior(ins_rate, ins_prior[0], ins_prior[1]) +
          gamma_log_prior(del_rate, del_prior[0], del_prior[1]) +
          ctmc_log_prior(Q, subst_gamma_shape, subst_gamma_rate))
    return ll + lp


def tkf91_log_posterior_cond(ins_rate, del_rate, t, Q, pi, x_seq, y_seq,
                              ins_prior=(2.0, 10.0), del_prior=(2.0, 10.0),
                              subst_gamma_shape=2.0, subst_gamma_rate=1.0):
    """Log P(x,y | |ancestor|, params) + log prior(params).

    Conditioned on ancestor length. Numerically stable at κ=1.
    """
    ll = tkf91_log_prob_cond(ins_rate, del_rate, t, Q, pi, x_seq, y_seq)
    lp = (gamma_log_prior(ins_rate, ins_prior[0], ins_prior[1]) +
          gamma_log_prior(del_rate, del_prior[0], del_prior[1]) +
          ctmc_log_prior(Q, subst_gamma_shape, subst_gamma_rate))
    return ll + lp


def tkf92_log_posterior(ins_rate, del_rate, t, ext, Q, pi, x_seq, y_seq,
                        ins_prior=(2.0, 10.0), del_prior=(2.0, 10.0),
                        subst_gamma_shape=2.0, subst_gamma_rate=1.0):
    """Log P(x,y | params) + log prior(params) for TKF92.

    Priors:
    - Gamma(shape, rate) on ins_rate and del_rate
    - Gamma(subst_gamma_shape, subst_gamma_rate) on each off-diagonal CTMC rate
    """
    ll = tkf92_log_prob(ins_rate, del_rate, t, ext, Q, pi, x_seq, y_seq)
    lp = (gamma_log_prior(ins_rate, ins_prior[0], ins_prior[1]) +
          gamma_log_prior(del_rate, del_prior[0], del_prior[1]) +
          ctmc_log_prior(Q, subst_gamma_shape, subst_gamma_rate))
    return ll + lp


# ============================================================
# Constrained 1D log-prob (alignment-aware)
# ============================================================
#
# When the alignment path is given (as in the precomputed X/A/Y pair
# format from Pfam seed), the pair-HMM DP collapses from a 2D
# (Lx+1, Ly+1, ns) lattice to a 1D (L_align, ns) scan. This eliminates
# the (Lx, Ly) compile-storm in Adam and the d3f3c27-class OOM, and is
# also a more truthful objective: when we trust the seed alignment, we
# should condition on it rather than re-marginalising over alternatives.
#
# These functions are plain JAX (no custom_vjp) — the 1D DP is cheap
# enough that autograd through forward_backward_1d_padded gives correct
# and fast gradients. The 2D path keeps its custom VJPs because
# autograd through the 2D lattice is the costly step there.
#
# The 2D log_prob counterparts above remain available; pick which to use
# at the training-loop call site (--dp-mode {constrained, full}).


def _emission_gradient_constrained(posteriors, state_types, sub_matrix,
                                     state_seq, anc_chars, desc_chars):
    """1D constrained substitution-matrix gradient.

    For shared (A, A) `sub_matrix` (the MixDom1-flavor sub used by all
    body M-states, regardless of domain), accumulate the per-(a, b)
    M-emission count from the alignment-conditioned posteriors:

        d log P / d sub_matrix[a, b]
            = sum over (M-cols ℓ with anc[ℓ]=a, desc[ℓ]=b)
                  ( sum over M-states s of post[ℓ, s] ) / sub_matrix[a, b]

    This is the analytic counterpart to the 2D `_emission_gradient`
    helper, and is what the score-identity-friendly inner VJP needs to
    avoid autograd-through-FB.
    """
    A = sub_matrix.shape[0]
    M_TYPE = 1
    is_M_state = (state_types == M_TYPE)        # (ns,)
    is_M_col = (state_seq == M_TYPE)             # (L,)

    # Per-column total M-state posterior, gated to columns whose type is M.
    M_post_per_col = jnp.where(
        is_M_col,
        jnp.sum(posteriors * is_M_state[None, :].astype(posteriors.dtype),
                axis=1),
        0.0)                                     # (L,)

    one_hot_a = jax.nn.one_hot(anc_chars, A, dtype=posteriors.dtype)   # (L, A)
    one_hot_b = jax.nn.one_hot(desc_chars, A, dtype=posteriors.dtype)  # (L, A)
    m_count = jnp.einsum('l,la,lb->ab',
                          M_post_per_col, one_hot_a, one_hot_b)       # (A, A)
    eps = jnp.asarray(1e-30, dtype=sub_matrix.dtype)
    return m_count / jnp.maximum(sub_matrix, eps)


def _pi_emission_gradient_constrained(posteriors, state_types, pi,
                                        state_seq, anc_chars, desc_chars):
    """1D constrained pi-as-linear-factor gradient.

    pi appears once linearly per emitting state-column:
      M-col with anc=a: log pi[a] (per-column M-state contribution)
      I-col with desc=b: log pi[b]
      D-col with anc=a: log pi[a]

    Each contributes posterior[ℓ, s] / pi[a or b] respectively. We sum
    these into a single per-character gradient vector.

    NOTE: this is the LINEAR-factor part. pi also enters sub_matrix via
    `transition_matrix(Q, t)` — that chain is autograd's job
    in the outer `mixdom_constrained_log_prob` wrapper.
    """
    A = pi.shape[0]
    M_TYPE, I_TYPE, D_TYPE = 1, 2, 3
    is_M_state = (state_types == M_TYPE)
    is_I_state = (state_types == I_TYPE)
    is_D_state = (state_types == D_TYPE)
    is_M_col = (state_seq == M_TYPE)
    is_I_col = (state_seq == I_TYPE)
    is_D_col = (state_seq == D_TYPE)

    M_post = jnp.where(is_M_col,
                        jnp.sum(posteriors * is_M_state[None, :], axis=1), 0.0)
    I_post = jnp.where(is_I_col,
                        jnp.sum(posteriors * is_I_state[None, :], axis=1), 0.0)
    D_post = jnp.where(is_D_col,
                        jnp.sum(posteriors * is_D_state[None, :], axis=1), 0.0)

    one_hot_a = jax.nn.one_hot(anc_chars, A, dtype=posteriors.dtype)
    one_hot_b = jax.nn.one_hot(desc_chars, A, dtype=posteriors.dtype)

    pi_count = (jnp.einsum('l,la->a', M_post, one_hot_a) +
                jnp.einsum('l,la->a', I_post, one_hot_b) +
                jnp.einsum('l,la->a', D_post, one_hot_a))             # (A,)
    eps = jnp.asarray(1e-30, dtype=pi.dtype)
    return pi_count / jnp.maximum(pi, eps)


def mixdom_constrained_log_prob(main_ins_rate, main_del_rate, t,
                                 dom_ins_rates, dom_del_rates, dom_weights,
                                 frag_weights, ext_rates,
                                 Q, pi,
                                 state_seq, anc_chars, desc_chars, real_L):
    """1D constrained log P(x, y, A | params) under nested MixDom Pair HMM.

    Differentiable w.r.t. all structural parameters and (Q, pi). The
    alignment A is given (state_seq + anc_chars + desc_chars) so the DP
    runs in 1D along the alignment columns.

    Args:
        main_ins_rate, main_del_rate: top-level (domain) indel rates
        t: evolutionary time
        dom_ins_rates: (n_dom,) per-domain insertion rates
        dom_del_rates: (n_dom,) per-domain deletion rates
        dom_weights:   (n_dom,) domain mixture weights
        frag_weights:  (n_dom, n_frag) fragment weights per domain
        ext_rates:     (n_dom, n_frag) fragment extension probabilities
        Q:             (A, A) substitution rate matrix (shared across domains)
        pi:            (A,) equilibrium distribution
        state_seq:     (L_pad,) padded per-column state codes (M=1, I=2, D=3,
                       padding=0). The real prefix [0:real_L] is used; the
                       suffix is masked by NEG_INF emissions and never
                       contributes to log_prob.
        anc_chars:     (L_pad,) ancestor chars (used at M / D columns)
        desc_chars:    (L_pad,) descendant chars (used at M / I columns)
        real_L:        scalar int — true alignment length (≤ L_pad)

    Returns:
        log_prob: scalar log-likelihood of the (x, y, A) under the model.
    """
    sub_matrix = transition_matrix(Q, t)              # (A, A)
    return _mixdom_constrained_dp(
        main_ins_rate, main_del_rate, t,
        dom_ins_rates, dom_del_rates, dom_weights,
        frag_weights, ext_rates,
        sub_matrix, pi,
        state_seq, anc_chars, desc_chars, real_L)


# Custom-VJP inner DP for the 1D constrained MixDom1 path. sub_matrix is
# treated as a leaf — autograd through transition_matrix(Q, t)
# in the outer wrapper handles the (Q → sub_matrix) chain.

@jax.custom_vjp
def _mixdom_constrained_dp(main_ins_rate, main_del_rate, t,
                            dom_ins_rates, dom_del_rates, dom_weights,
                            frag_weights, ext_rates,
                            sub_matrix, pi,
                            state_seq, anc_chars, desc_chars, real_L):
    """Inner 1D constrained MixDom1 DP with custom_vjp. The forward path
    returns just the scalar log_prob; the bwd uses HR-analytic emission
    gradients + score-identity chi gradient (via jax.grad of
    `_chi_weighted_loglik` against `expected_trans`)."""
    from ..dp.hmm import (
        forward_backward_1d_padded, NEG_INF, pair_hmm_emissions_constrained_per_domain)
    SS_INDEX, EE_INDEX = 0, 1

    n_dom = dom_ins_rates.shape[0]
    n_frag = frag_weights.shape[1]
    st = mixdom_state_types(n_dom, n_frag)
    chi, _ = build_nested_trans(main_ins_rate, main_del_rate, t,
                                dom_ins_rates, dom_del_rates, dom_weights,
                                frag_weights, ext_rates)
    log_chi = safe_log(chi)

    sub_matrices = jnp.broadcast_to(sub_matrix[None, :, :],
                                     (n_dom,) + sub_matrix.shape)
    pis = jnp.broadcast_to(pi[None, :], (n_dom,) + pi.shape)
    log_emit = pair_hmm_emissions_constrained_per_domain(
        st, state_seq, anc_chars, desc_chars,
        sub_matrices, pis, n_dom, n_frag)
    L_pad = log_emit.shape[0]
    pos = jnp.arange(L_pad)
    is_real = (pos < real_L)[:, None]
    log_emit = jnp.where(is_real, log_emit, NEG_INF)
    log_prob = forward_backward_1d_padded(
        log_chi, log_emit, real_L,
        init_state=SS_INDEX, final_state=EE_INDEX, forward_only=True)
    return log_prob


def _mixdom_constrained_dp_fwd(main_ins_rate, main_del_rate, t,
                                 dom_ins_rates, dom_del_rates, dom_weights,
                                 frag_weights, ext_rates,
                                 sub_matrix, pi,
                                 state_seq, anc_chars, desc_chars, real_L):
    """Forward: build chi + emit, run full FB (not forward-only), compute
    HR-analytic emission gradients d_sub and d_pi from posteriors."""
    from ..dp.hmm import (
        forward_backward_1d_padded, NEG_INF, pair_hmm_emissions_constrained_per_domain)
    SS_INDEX, EE_INDEX = 0, 1

    n_dom = dom_ins_rates.shape[0]
    n_frag = frag_weights.shape[1]
    st = mixdom_state_types(n_dom, n_frag)
    chi, _ = build_nested_trans(main_ins_rate, main_del_rate, t,
                                dom_ins_rates, dom_del_rates, dom_weights,
                                frag_weights, ext_rates)
    log_chi = safe_log(chi)

    sub_matrices = jnp.broadcast_to(sub_matrix[None, :, :],
                                     (n_dom,) + sub_matrix.shape)
    pis = jnp.broadcast_to(pi[None, :], (n_dom,) + pi.shape)
    log_emit = pair_hmm_emissions_constrained_per_domain(
        st, state_seq, anc_chars, desc_chars,
        sub_matrices, pis, n_dom, n_frag)
    L_pad = log_emit.shape[0]
    pos = jnp.arange(L_pad)
    is_real = (pos < real_L)[:, None]
    log_emit = jnp.where(is_real, log_emit, NEG_INF)

    # Full FB so we have posteriors + expected_trans for the analytic
    # emission and score-identity gradients.
    log_prob, posteriors, expected_trans = forward_backward_1d_padded(
        log_chi, log_emit, real_L,
        init_state=SS_INDEX, final_state=EE_INDEX, forward_only=False)

    d_sub = _emission_gradient_constrained(
        posteriors, st, sub_matrix, state_seq, anc_chars, desc_chars)
    d_pi = _pi_emission_gradient_constrained(
        posteriors, st, pi, state_seq, anc_chars, desc_chars)

    res = (expected_trans, d_sub, d_pi,
           main_ins_rate, main_del_rate, t,
           dom_ins_rates, dom_del_rates, dom_weights,
           frag_weights, ext_rates)
    return log_prob, res


def _mixdom_constrained_dp_bwd(res, g):
    """Backward: chi-related score identity + analytic emission gradients."""
    (expected_trans, d_sub, d_pi,
     main_ins_rate, main_del_rate, t,
     dom_ins_rates, dom_del_rates, dom_weights,
     frag_weights, ext_rates) = res

    # Phase 5b: include argnum 2 (t) so chi-side d_t is captured.
    grad_fn = jax.grad(_chi_weighted_loglik,
                        argnums=(0, 1, 2, 3, 4, 5, 6, 7))
    g0, g1, g2, g3, g4, g5, g6, g7 = grad_fn(
        main_ins_rate, main_del_rate, t,
        dom_ins_rates, dom_del_rates, dom_weights,
        frag_weights, ext_rates, expected_trans)

    # 14 args: main_ins, main_del, t, dom_ins, dom_del, dom_w, frag_w, ext,
    #          sub_matrix, pi, state_seq, anc_chars, desc_chars, real_L
    return (g * g0, g * g1, g * g2,
            g * g3, g * g4, g * g5, g * g6, g * g7,
            g * d_sub, g * d_pi,
            None, None, None, None)


_mixdom_constrained_dp.defvjp(
    _mixdom_constrained_dp_fwd, _mixdom_constrained_dp_bwd)


def tkf91_constrained_log_prob(ins_rate, del_rate, t, Q, pi,
                                state_seq, anc_chars, desc_chars, real_L):
    """1D constrained log-prob for TKF91 (single domain, single fragment,
    no fragment extension). Thin shim over mixdom_constrained_log_prob with
    n_dom = n_frag = 1 and zero per-domain rates."""
    main_ins = ins_rate
    main_del = del_rate
    dom_ins = jnp.array([ins_rate])
    dom_del = jnp.array([del_rate])
    dom_weights = jnp.array([1.0])
    frag_weights = jnp.array([[1.0]])
    ext_rates = jnp.array([[0.0]])  # TKF91: zero fragment extension
    return mixdom_constrained_log_prob(
        main_ins, main_del, t,
        dom_ins, dom_del, dom_weights,
        frag_weights, ext_rates,
        Q, pi, state_seq, anc_chars, desc_chars, real_L)


def tkf92_constrained_log_prob(ins_rate, del_rate, t, ext, Q, pi,
                                state_seq, anc_chars, desc_chars, real_L):
    """1D constrained log-prob for TKF92 (single domain, single fragment,
    nonzero fragment extension)."""
    dom_ins = jnp.array([ins_rate])
    dom_del = jnp.array([del_rate])
    dom_weights = jnp.array([1.0])
    frag_weights = jnp.array([[1.0]])
    ext_rates = jnp.array([[ext]])
    return mixdom_constrained_log_prob(
        ins_rate, del_rate, t,
        dom_ins, dom_del, dom_weights,
        frag_weights, ext_rates,
        Q, pi, state_seq, anc_chars, desc_chars, real_L)


def _build_class_Q_constrained(class_S_exch_c, class_pis_c):
    """Build a per-class CTMC rate matrix Q_c from a class's symmetric
    exchange matrix S_c and equilibrium π_c.

    Off-diagonal Q_c[a, b] = S_c[a, b] · π_c[b]; diagonal is set so each
    row sums to zero. No rate normalisation (per CLAUDE.md guidance).
    """
    A = class_pis_c.shape[0]
    Q = class_S_exch_c * class_pis_c[None, :]
    Q = Q - jnp.diag(jnp.diag(Q))
    Q = Q - jnp.diag(jnp.sum(Q, axis=1))
    return Q


def _build_class_subs_constrained(class_S_exch, class_pis, t):
    """vmap of transition_matrix over per-class (Q_c, π_c) at time t.

    Returns class_sub_matrices: (C, A, A).
    """
    Q_c = jax.vmap(_build_class_Q_constrained)(class_S_exch, class_pis)
    sub_c = jax.vmap(lambda Q, p: transition_matrix(Q, t))(
        Q_c, class_pis)
    return sub_c


def mixdom2_constrained_log_prob(main_ins_rate, main_del_rate, t,
                                  dom_ins_rates, dom_del_rates, dom_weights,
                                  frag_weights, ext_rates,
                                  class_S_exch, class_pis, classdist,
                                  state_seq, anc_chars, desc_chars, real_L):
    """1D constrained log P(x, y, A | params) under MixDom2 (per-fragment
    site-class mixture) along a fixed alignment.

    Differentiable w.r.t. all structural params, class_S_exch, class_pis,
    classdist. Implementation is plain JAX (no custom_vjp): the 1D DP
    via forward_backward_1d_padded plus the (C, ns, L) intermediate in
    pair_hmm_emissions_constrained_per_class is small enough that
    autograd is both correct and cheap.

    Args:
        main_ins_rate, main_del_rate, t: top-level rates and time
        dom_ins_rates, dom_del_rates, dom_weights: per-domain
        frag_weights, ext_rates: per-(domain, fragment)
        class_S_exch: (C, A, A) per-class symmetric exchange matrices
        class_pis:   (C, A) per-class equilibrium
        classdist:   (n_dom, n_frag, C) per-(domain, fragment) class dist
        state_seq, anc_chars, desc_chars, real_L: alignment + chars + length

    Returns:
        log_prob: scalar log-likelihood.
    """
    # Outer wrapper: build class_subs from (class_S_exch, class_pis, t),
    # then dispatch to the custom_vjp inner DP that treats class_subs as
    # a LEAF. Autograd through `_build_class_subs_constrained` gives the
    # additional gradients to class_S_exch and class_pis-via-Q chain.
    class_subs = _build_class_subs_constrained(class_S_exch, class_pis, t)
    return _mixdom2_constrained_dp(
        main_ins_rate, main_del_rate, t,
        dom_ins_rates, dom_del_rates, dom_weights,
        frag_weights, ext_rates,
        class_subs, class_pis, classdist,
        state_seq, anc_chars, desc_chars, real_L)


def _mixdom2_emission_counts_constrained(class_subs, class_pis, class_dist,
                                            posteriors, log_emit,
                                            state_seq, anc_chars, desc_chars,
                                            n_dom, n_frag):
    """1D constrained per-class emission sufficient stats. Same outputs
    as `_mixdom2_emission_counts` (the 2D analog above) but consumed by
    the alignment-aware FB along a fixed state path instead of a 2D
    pair-HMM lattice.

    Returns:
        mc_c: (C, A, A) per-class match-pair counts.
        V_c_linear: (C, A) linear-pi-factor character counts (M / I / D).
        classdist_counts: (D, F, C) per-(d, f, c) class assignment counts.
    """
    n_cls, A = class_pis.shape
    L, ns = posteriors.shape

    # ratio_post[ℓ, s] = posteriors[ℓ, s] / exp(log_emit[ℓ, s]).
    safe_log_emit = jnp.where(jnp.isfinite(log_emit), log_emit, 0.0)
    ratio_post = jnp.where(
        posteriors > 0.0,
        posteriors * jnp.exp(-safe_log_emit),
        0.0)                                                # (L, ns)

    # Per-state (d, f, type) lookup.
    body = jnp.arange(ns - 2)
    dom_idx = body // (5 * n_frag)
    within_dom = body % (5 * n_frag)
    frag_idx = within_dom % n_frag
    uv_idx = within_dom // n_frag           # 0..4 → MM, MI, MD, II, DD
    M_mask_body = (uv_idx == 0)
    I_mask_body = (uv_idx == 1) | (uv_idx == 3)
    D_mask_body = (uv_idx == 2) | (uv_idx == 4)

    def _aggregate_by_df(mask_body):
        sel = ratio_post[:, 2:] * mask_body[None, :].astype(ratio_post.dtype)
        flat_df = (dom_idx * n_frag + frag_idx).astype(jnp.int32)
        nDF = n_dom * n_frag
        one_hot = (flat_df[:, None] ==
                    jnp.arange(nDF)[None, :]).astype(sel.dtype)
        out_flat = sel @ one_hot                            # (L, nDF)
        return out_flat.reshape(L, n_dom, n_frag)           # (L, D, F)

    ratio_df_M = _aggregate_by_df(M_mask_body)
    ratio_df_I = _aggregate_by_df(I_mask_body)
    ratio_df_D = _aggregate_by_df(D_mask_body)

    cd = class_dist  # (D, F, C)

    pi_x = class_pis[:, anc_chars]                          # (C, L)
    pi_y = class_pis[:, desc_chars]                         # (C, L)
    P_xy = class_subs[:, anc_chars, desc_chars]             # (C, L)
    f_M = pi_x * P_xy                                       # (C, L)
    f_I = pi_y                                              # (C, L)
    f_D = pi_x                                              # (C, L)

    M_T, I_T, D_T = 1, 2, 3
    is_M_col = (state_seq == M_T).astype(ratio_df_M.dtype)
    is_I_col = (state_seq == I_T).astype(ratio_df_M.dtype)
    is_D_col = (state_seq == D_T).astype(ratio_df_M.dtype)

    # Per-class per-col weights (gated by column type).
    cdc_M = jnp.einsum('ldf,dfc->cl', ratio_df_M, cd) * is_M_col[None, :]
    cdc_I = jnp.einsum('ldf,dfc->cl', ratio_df_I, cd) * is_I_col[None, :]
    cdc_D = jnp.einsum('ldf,dfc->cl', ratio_df_D, cd) * is_D_col[None, :]

    wM = cdc_M * f_M                                        # (C, L)
    wI = cdc_I * f_I
    wD = cdc_D * f_D

    one_hot_a = jax.nn.one_hot(anc_chars, A, dtype=wM.dtype)
    one_hot_b = jax.nn.one_hot(desc_chars, A, dtype=wM.dtype)

    mc_c = jnp.einsum('cl,la,lb->cab', wM, one_hot_a, one_hot_b)

    Vc_M = jnp.einsum('cl,la->ca', wM, one_hot_a)
    Vc_I = jnp.einsum('cl,la->ca', wI, one_hot_b)
    Vc_D = jnp.einsum('cl,la->ca', wD, one_hot_a)
    V_c_linear = Vc_M + Vc_I + Vc_D

    # classdist counts:
    # classdist_counts[d, f, c] = cd[d, f, c] · Σ_ν Σ_ℓ ratio_df_ν[ℓ, d, f] · f_ν[c, ℓ]
    sum_M = jnp.einsum('ldf,cl->dfc',
                        ratio_df_M * is_M_col[:, None, None], f_M)
    sum_I = jnp.einsum('ldf,cl->dfc',
                        ratio_df_I * is_I_col[:, None, None], f_I)
    sum_D = jnp.einsum('ldf,cl->dfc',
                        ratio_df_D * is_D_col[:, None, None], f_D)
    classdist_counts = cd * (sum_M + sum_I + sum_D)

    return mc_c, V_c_linear, classdist_counts


@jax.custom_vjp
def _mixdom2_constrained_dp(main_ins_rate, main_del_rate, t,
                              dom_ins_rates, dom_del_rates, dom_weights,
                              frag_weights, ext_rates,
                              class_subs, class_pis, classdist,
                              state_seq, anc_chars, desc_chars, real_L):
    """Inner 1D constrained MixDom2 DP with custom_vjp. class_subs is a
    LEAF — autograd through `_build_class_subs_constrained` in the outer
    wrapper handles the chain (class_S_exch, class_pis-via-Q) → class_subs.
    """
    from ..dp.hmm import (
        forward_backward_1d_padded, NEG_INF,
        pair_hmm_emissions_constrained_per_class)
    SS_INDEX, EE_INDEX = 0, 1

    n_dom = dom_ins_rates.shape[0]
    n_frag = frag_weights.shape[1]
    st = mixdom_state_types(n_dom, n_frag)
    chi, _ = build_nested_trans(main_ins_rate, main_del_rate, t,
                                dom_ins_rates, dom_del_rates, dom_weights,
                                frag_weights, ext_rates)
    log_chi = safe_log(chi)
    log_emit = pair_hmm_emissions_constrained_per_class(
        st, state_seq, anc_chars, desc_chars,
        class_subs, class_pis, classdist, n_dom, n_frag)
    L_pad = log_emit.shape[0]
    pos = jnp.arange(L_pad)
    is_real = (pos < real_L)[:, None]
    log_emit = jnp.where(is_real, log_emit, NEG_INF)
    log_prob = forward_backward_1d_padded(
        log_chi, log_emit, real_L,
        init_state=SS_INDEX, final_state=EE_INDEX, forward_only=True)
    return log_prob


def _mixdom2_constrained_dp_fwd(main_ins_rate, main_del_rate, t,
                                  dom_ins_rates, dom_del_rates, dom_weights,
                                  frag_weights, ext_rates,
                                  class_subs, class_pis, classdist,
                                  state_seq, anc_chars, desc_chars, real_L):
    """Forward: full FB + per-class emission count residuals."""
    from ..dp.hmm import (
        forward_backward_1d_padded, NEG_INF,
        pair_hmm_emissions_constrained_per_class)
    SS_INDEX, EE_INDEX = 0, 1

    n_dom = dom_ins_rates.shape[0]
    n_frag = frag_weights.shape[1]
    st = mixdom_state_types(n_dom, n_frag)
    chi, _ = build_nested_trans(main_ins_rate, main_del_rate, t,
                                dom_ins_rates, dom_del_rates, dom_weights,
                                frag_weights, ext_rates)
    log_chi = safe_log(chi)
    log_emit = pair_hmm_emissions_constrained_per_class(
        st, state_seq, anc_chars, desc_chars,
        class_subs, class_pis, classdist, n_dom, n_frag)
    L_pad = log_emit.shape[0]
    pos = jnp.arange(L_pad)
    is_real = (pos < real_L)[:, None]
    log_emit = jnp.where(is_real, log_emit, NEG_INF)
    log_prob, posteriors, expected_trans = forward_backward_1d_padded(
        log_chi, log_emit, real_L,
        init_state=SS_INDEX, final_state=EE_INDEX, forward_only=False)

    mc_c, V_c_linear, cd_count = _mixdom2_emission_counts_constrained(
        class_subs, class_pis, classdist,
        posteriors, log_emit, state_seq, anc_chars, desc_chars,
        n_dom, n_frag)

    res = (expected_trans, mc_c, V_c_linear, cd_count,
           class_subs, class_pis, classdist,
           main_ins_rate, main_del_rate, t,
           dom_ins_rates, dom_del_rates, dom_weights,
           frag_weights, ext_rates)
    return log_prob, res


def _mixdom2_constrained_dp_bwd(res, g):
    """Backward: chi via score id; class_subs / class_pis (linear factor)
    / classdist via analytic counts.
    """
    (expected_trans, mc_c, V_c_linear, cd_count,
     class_subs, class_pis, classdist,
     main_ins_rate, main_del_rate, t,
     dom_ins_rates, dom_del_rates, dom_weights,
     frag_weights, ext_rates) = res

    # Phase 5b: include argnum 2 (t) so chi-side d_t is captured.
    grad_fn = jax.grad(_chi_weighted_loglik,
                        argnums=(0, 1, 2, 3, 4, 5, 6, 7))
    g0, g1, g2, g3, g4, g5, g6, g7 = grad_fn(
        main_ins_rate, main_del_rate, t,
        dom_ins_rates, dom_del_rates, dom_weights,
        frag_weights, ext_rates, expected_trans)

    eps_subs = jnp.asarray(1e-30, dtype=class_subs.dtype)
    eps_pi = jnp.asarray(1e-30, dtype=class_pis.dtype)
    eps_cd = jnp.asarray(1e-300, dtype=classdist.dtype)

    d_class_subs = mc_c / jnp.maximum(class_subs, eps_subs)
    d_class_pis = V_c_linear / jnp.maximum(class_pis, eps_pi)
    d_classdist = cd_count / jnp.maximum(classdist, eps_cd)

    # 16 args to _mixdom2_constrained_dp:
    # main_ins, main_del, t, dom_ins, dom_del, dom_w, frag_w, ext,
    # class_subs, class_pis, classdist, state_seq, anc_chars, desc_chars, real_L
    return (g * g0, g * g1, g * g2,
            g * g3, g * g4, g * g5, g * g6, g * g7,
            g * d_class_subs, g * d_class_pis, g * d_classdist,
            None, None, None, None)


_mixdom2_constrained_dp.defvjp(
    _mixdom2_constrained_dp_fwd, _mixdom2_constrained_dp_bwd)
