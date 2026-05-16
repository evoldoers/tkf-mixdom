"""Indel model: TKF91/TKF92 BDI parameters and score derivatives.

Mirrors subst.py for the indel component of Rate EM.
All functions handle the λ≈μ regime via L'Hôpital limits
(see lhopital-limits.tex for derivations).
"""

import jax
import jax.numpy as jnp
import numpy as np

# Threshold for switching to L'Hôpital limit formulas.
# When |1 - κ| < threshold, use limit formulas for β, γ, and score derivatives.
EQUAL_RATE_THRESHOLD = 1e-4

# Threshold for switching E[S] computation from direct division
# (numer/(λ-μ)) to the analytic L'Hôpital limit (eq:ES-limit).
#
# At λ=μ exactly, direct division is 0/0; the L'Hôpital limit is the
# unique finite analytic value. For λ ≠ μ but close, direct division
# is numerically well-defined in float64 down to |1-κ| ~ 1e-10 — the
# L'Hôpital limit becomes an APPROXIMATION with O(ε) bias relative to
# the true finite-ε value (NOT O(ε²); the limit is the first term of
# the Taylor series, all higher-order terms are dropped). The previous
# 0.05 threshold was sized for float32 catastrophic cancellation; in
# current usage (float64 arithmetic throughout) it silently activated
# the limit-as-approximation path for κ ∈ [0.95, 1.05] — exactly the
# regime real Pfam fits hit (κ ≈ 0.93). That is also the source of the
# downstream conservation-law warnings (audit ledger #9).
#
# See tkf/lhopital-and-conservation.md for the empirical bias table
# and conservation derivation. With threshold = 1e-10, float64 div
# cancellation error is < 1e-5, far below the BDI estimate's
# statistical SE.
ES_LHOPITAL_THRESHOLD = 1e-10


# --- BDI parameters (α, β, γ, κ) ---

def tkf_alpha(del_rate, t):
    """Survival probability: α = exp(-μt)."""
    return jnp.exp(-del_rate * t)


def tkf_beta(ins_rate, del_rate, t):
    """Offspring parameter β, with L'Hôpital limit at λ=μ.

    General: β = λ(η-α)/(μη-λα) where η=exp(-λt), α=exp(-μt).
    Limit:   β → s/(1+s) where s=μt.
    """
    s = del_rate * t
    eta = jnp.exp(-ins_rate * t)
    alpha = jnp.exp(-s)

    # General formula
    numer = ins_rate * (eta - alpha)
    denom = del_rate * eta - ins_rate * alpha
    denom_safe = jnp.where(jnp.abs(denom) < 1e-30, 1.0, denom)
    beta_general = jnp.where(jnp.abs(denom) < 1e-30, 0.0, numer / denom_safe)

    # L'Hôpital limit
    beta_limit = s / (1.0 + s)

    kappa = ins_rate / del_rate
    return jnp.where(jnp.abs(1.0 - kappa) < EQUAL_RATE_THRESHOLD,
                     beta_limit, beta_general)


def tkf_gamma(ins_rate, del_rate, t):
    """Orphan parameter γ, with L'Hôpital limit at λ=μ.

    General: γ = 1 - μβ/(λ(1-α)).
    Limit:   γ → 1 - 1/((1+s)φ) where φ=(1-e^{-s})/s.
    """
    s = del_rate * t
    alpha = tkf_alpha(del_rate, t)
    beta = tkf_beta(ins_rate, del_rate, t)
    Phi = 1.0 - alpha

    # General formula
    gamma_general = jnp.where(
        jnp.abs(Phi) < 1e-30, 0.0,
        1.0 - del_rate * beta / (ins_rate * jnp.maximum(Phi, 1e-30)))

    # L'Hôpital limit: φ = Φ/s = (1-e^{-s})/s
    phi = jnp.where(s < 1e-10, 1.0 - s / 2.0, Phi / jnp.maximum(s, 1e-30))
    gamma_limit = 1.0 - 1.0 / ((1.0 + s) * phi)

    kappa = ins_rate / del_rate
    return jnp.where(jnp.abs(1.0 - kappa) < EQUAL_RATE_THRESHOLD,
                     gamma_limit, gamma_general)


def tkf_kappa(ins_rate, del_rate):
    """Stationary geometric parameter: κ = λ/μ."""
    return ins_rate / del_rate


# --- Smooth (differentiable) formulations for autodiff through λ=μ ---

def _expm1_ratio(x):
    """Compute expm1(x)/x = (e^x - 1)/x, stable at x=0.

    Uses Taylor expansion 1 + x/2 + x²/6 + x³/24 when |x| is small.
    """
    return jnp.where(
        jnp.abs(x) < 1e-5,
        1.0 + x / 2.0 + x * x / 6.0 + x * x * x / 24.0,
        jnp.expm1(x) / jnp.where(jnp.abs(x) < 1e-30, 1.0, x))


def _tkf_beta_smooth(ins_rate, del_rate, t):
    """β computed in a form that is both stable and differentiable at λ=μ.

    Reparameterization: β = λ·t·r / (μ·t·r + 1)
    where r = expm1(εt)/(εt) with ε = μ - λ.
    At ε=0: r=1, β = λt/(μt+1) = s/(1+s). Smooth and correct.
    """
    eps = del_rate - ins_rate
    r = _expm1_ratio(eps * t)
    return ins_rate * t * r / (del_rate * t * r + 1.0)


def _tkf_gamma_smooth(ins_rate, del_rate, t):
    """γ computed in a form that is both stable and differentiable at λ=μ.

    γ = 1 - μβ/(λ(1-α)), using smooth β.
    At λ=μ: γ → 1 - 1/((1+s)φ) where φ = (1-e^{-s})/s.
    """
    alpha = jnp.exp(-del_rate * t)
    beta = _tkf_beta_smooth(ins_rate, del_rate, t)
    Phi = 1.0 - alpha
    # Use safe denominator; when Phi→0 (t→0), γ→0
    return 1.0 - del_rate * beta / (ins_rate * jnp.maximum(Phi, 1e-30))


def _logP_wfst_smooth(lam, mu, t, n_alpha, n_1malpha, n_beta, n_1mbeta,
                       n_gamma, n_1mgamma):
    """WFST log-likelihood using smooth (differentiable) β, γ formulas.

    logP = Σ_g n_g · log(ξ_g) over the 6 WFST groups.
    Differentiable at lam=mu for use with jax.grad.
    """
    alpha = jnp.exp(-mu * t)
    beta = _tkf_beta_smooth(lam, mu, t)
    gamma = _tkf_gamma_smooth(lam, mu, t)

    # Clamp for safe log
    eps_log = 1e-30
    log_alpha = jnp.log(jnp.maximum(alpha, eps_log))
    log_1malpha = jnp.log(jnp.maximum(1.0 - alpha, eps_log))
    log_beta = jnp.log(jnp.maximum(beta, eps_log))
    log_1mbeta = jnp.log(jnp.maximum(1.0 - beta, eps_log))
    log_gamma = jnp.log(jnp.maximum(gamma, eps_log))
    log_1mgamma = jnp.log(jnp.maximum(1.0 - gamma, eps_log))

    return (n_alpha * log_alpha + n_1malpha * log_1malpha +
            n_beta * log_beta + n_1mbeta * log_1mbeta +
            n_gamma * log_gamma + n_1mgamma * log_1mgamma)


def _es_limit_analytic(groups, cons, del_rate, t, T_total):
    """E[S] at λ=μ via analytic L'Hôpital limit (eq:ES-limit).

    Uses jax.grad on the smooth WFST log-likelihood to compute:
        E[S] = μ·∂²logP/∂λ∂μ − ∂logP/∂λ − μ·∂²logP/∂λ² − t

    The smooth β/γ formulas make this differentiable at λ=μ.
    """
    mu = del_rate

    # Extract count groups as plain JAX scalars
    n_a = groups['log_alpha']
    n_1a = groups['log_1malpha']
    n_b = groups['log_beta']
    n_1b = groups['log_1mbeta']
    n_g = groups['log_gamma']
    n_1g = groups['log_1mgamma']

    # f(λ) = cons + μ·∂logP/∂μ − λ·∂logP/∂λ − λ·T_total
    # E[S] = f'(μ) by L'Hôpital (eq:ES-limit)
    #
    # f'(λ) = μ·∂²logP/∂λ∂μ − ∂logP/∂λ − λ·∂²logP/∂λ² − T_total
    #
    # We compute this via jax.grad of the E[S] numerator.

    def _numerator(lam):
        """E[S] numerator f(λ) = cons + μ·∂logP/∂μ − λ·∂logP/∂λ − λ·T_total."""
        # ∂logP/∂λ
        dlogP_dlam = jax.grad(
            lambda l: _logP_wfst_smooth(l, mu, t, n_a, n_1a, n_b, n_1b, n_g, n_1g)
        )(lam)
        # ∂logP/∂μ (at this λ)
        dlogP_dmu = jax.grad(
            lambda m: _logP_wfst_smooth(lam, m, t, n_a, n_1a, n_b, n_1b, n_g, n_1g)
        )(mu)
        return cons + mu * dlogP_dmu - lam * dlogP_dlam - lam * T_total

    # E[S] = f'(μ) = d/dλ[f(λ)]|_{λ=μ}
    lam_val = jnp.array(float(mu))
    E_S = jax.grad(_numerator)(lam_val)
    return float(E_S)


# --- Score derivatives ---

def score_derivatives(ins_rate, del_rate, t):
    """Compute ∂(log ξ)/∂λ and ∂(log ξ)/∂μ for all TKF parameters.

    Automatically gates between general and L'Hôpital limit formulas
    when |1-κ| < EQUAL_RATE_THRESHOLD.

    Returns dict mapping parameter name to (d/dλ, d/dμ) tuple.
    """
    kappa = ins_rate / del_rate
    near_equal = jnp.abs(1.0 - kappa) < EQUAL_RATE_THRESHOLD

    general = _score_derivatives_general(ins_rate, del_rate, t)
    limit = _score_derivatives_limit(ins_rate, del_rate, t)

    result = {}
    for name in general:
        dl_g, dm_g = general[name]
        dl_l, dm_l = limit[name]
        result[name] = (
            jnp.where(near_equal, dl_l, dl_g),
            jnp.where(near_equal, dm_l, dm_g),
        )
    return result


def _score_derivatives_general(ins_rate, del_rate, t):
    """General-case score derivatives."""
    eta = jnp.exp(-ins_rate * t)
    alpha = jnp.exp(-del_rate * t)
    beta = tkf_beta(ins_rate, del_rate, t)
    gamma = tkf_gamma(ins_rate, del_rate, t)
    kappa = ins_rate / del_rate
    delta = del_rate * eta - ins_rate * alpha
    Phi = 1.0 - alpha

    # Safe denominators for singular terms
    eta_m_alpha = eta - alpha
    eta_m_alpha_safe = jnp.where(
        jnp.abs(eta_m_alpha) < 1e-30, 1e-30, eta_m_alpha)
    delta_safe = jnp.where(jnp.abs(delta) < 1e-30, 1e-30, delta)
    Phi_safe = jnp.maximum(Phi, 1e-30)

    # log α
    dlog_alpha = (0.0, -t)

    # log(1-α)
    dlog_1malpha = (0.0, t * alpha / Phi_safe)

    # log β = log λ + log(η-α) - log δ
    dlog_beta_dl = (1.0 / ins_rate
                    - t * eta / eta_m_alpha_safe
                    + (t * del_rate * eta + alpha) / delta_safe)
    dlog_beta_dm = (t * alpha / eta_m_alpha_safe
                    - (eta + t * ins_rate * alpha) / delta_safe)

    # log(1-β) = -β/(1-β) * d log β
    ratio_beta = beta / jnp.maximum(1.0 - beta, 1e-30)
    dlog_1mbeta = (-ratio_beta * dlog_beta_dl,
                   -ratio_beta * dlog_beta_dm)

    # log(1-γ) = log μ + log β - log λ - log(1-α)
    dlog_1mgamma_dl = dlog_beta_dl - 1.0 / ins_rate
    dlog_1mgamma_dm = 1.0 / del_rate + dlog_beta_dm - t * alpha / Phi_safe

    # log γ = -(1-γ)/γ * d log(1-γ)
    ratio_gam = (1.0 - gamma) / jnp.maximum(gamma, 1e-30)
    dlog_gamma = (-ratio_gam * dlog_1mgamma_dl,
                  -ratio_gam * dlog_1mgamma_dm)

    # log κ
    dlog_kappa = (1.0 / ins_rate, -1.0 / del_rate)

    # log(1-κ) = -κ/(1-κ) * d log κ
    r_kappa = -kappa / jnp.maximum(1.0 - kappa, 1e-30)
    dlog_1mkappa = (r_kappa / ins_rate, -r_kappa / del_rate)

    return {
        'log_alpha': dlog_alpha,
        'log_1malpha': dlog_1malpha,
        'log_beta': (dlog_beta_dl, dlog_beta_dm),
        'log_1mbeta': dlog_1mbeta,
        'log_gamma': dlog_gamma,
        'log_1mgamma': (dlog_1mgamma_dl, dlog_1mgamma_dm),
        'log_kappa': dlog_kappa,
        'log_1mkappa': dlog_1mkappa,
    }


def _score_derivatives_limit(ins_rate, del_rate, t):
    """L'Hôpital limit score derivatives at λ≈μ.

    All formulas from lhopital-limits.tex Section 2.3.
    Returns ∂/∂λ and ∂/∂μ (not log-elasticities).
    """
    mu = del_rate
    s = mu * t
    alpha = jnp.exp(-s)
    Phi = 1.0 - alpha
    Phi_safe = jnp.maximum(Phi, 1e-30)
    # φ = (1-e^{-s})/s, with Taylor limit φ(0)=1
    phi = jnp.where(s < 1e-10, 1.0 - s / 2.0, Phi / jnp.maximum(s, 1e-30))
    # R = (1-γ)/γ = 1/((1+s)φ - 1)
    R = 1.0 / jnp.maximum((1.0 + s) * phi - 1.0, 1e-30)

    # α, 1-α: same as general (no λ dependence)
    dlog_alpha = (0.0, -t)
    dlog_1malpha = (0.0, t * alpha / Phi_safe)

    # β: ∂log β/∂λ → 1/μ - t/(2(1+s)), ∂log β/∂μ → -t/(2(1+s))
    half_t_1ps = t / (2.0 * (1.0 + s))
    dlog_beta_dl = 1.0 / mu - half_t_1ps
    dlog_beta_dm = -half_t_1ps

    # 1-β: β/(1-β) → s
    dlog_1mbeta_dl = -s * dlog_beta_dl
    dlog_1mbeta_dm = -s * dlog_beta_dm

    # 1-γ: ∂log(1-γ)/∂λ → -t/(2(1+s))
    #       ∂log(1-γ)/∂μ → 1/μ - t/(2(1+s)) - tα/Φ
    dlog_1mgamma_dl = -half_t_1ps
    dlog_1mgamma_dm = 1.0 / mu - half_t_1ps - t * alpha / Phi_safe

    # γ: -(1-γ)/γ * d log(1-γ) = -R * d log(1-γ)
    dlog_gamma_dl = -R * dlog_1mgamma_dl
    dlog_gamma_dm = -R * dlog_1mgamma_dm

    # κ, 1-κ: use general formulas (not singular in κ itself)
    kappa = ins_rate / del_rate
    dlog_kappa = (1.0 / ins_rate, -1.0 / del_rate)
    r_kappa = -kappa / jnp.maximum(1.0 - kappa, 1e-30)
    dlog_1mkappa = (r_kappa / ins_rate, -r_kappa / del_rate)

    return {
        'log_alpha': dlog_alpha,
        'log_1malpha': dlog_1malpha,
        'log_beta': (dlog_beta_dl, dlog_beta_dm),
        'log_1mbeta': (dlog_1mbeta_dl, dlog_1mbeta_dm),
        'log_gamma': (dlog_gamma_dl, dlog_gamma_dm),
        'log_1mgamma': (dlog_1mgamma_dl, dlog_1mgamma_dm),
        'log_kappa': dlog_kappa,
        'log_1mkappa': dlog_1mkappa,
    }


# --- Transition count groups ---

def transition_count_groups(n_trans):
    """Map 5x5 transition count matrix to log-parameter coefficient groups.

    Each TKF91 transition probability is a product of BDI parameters.
    For rows S,M,I: τ[s,M]=(1-β)κα, τ[s,I]=β, τ[s,D]=(1-β)κ(1-α), τ[s,E]=(1-β)(1-κ)
    For row D:      τ[D,M]=(1-γ)κα, τ[D,I]=γ, τ[D,D]=(1-γ)κ(1-α), τ[D,E]=(1-γ)(1-κ)
    """
    from .params import S, M, I, D, E

    n_log_alpha = n_trans[S, M] + n_trans[M, M] + n_trans[I, M] + n_trans[D, M]
    n_log_1malpha = n_trans[S, D] + n_trans[M, D] + n_trans[I, D] + n_trans[D, D]
    n_log_beta = n_trans[S, I] + n_trans[M, I] + n_trans[I, I]
    n_log_1mbeta = (n_trans[S, M] + n_trans[S, D] + n_trans[S, E] +
                    n_trans[M, M] + n_trans[M, D] + n_trans[M, E] +
                    n_trans[I, M] + n_trans[I, D] + n_trans[I, E])
    n_log_gamma = n_trans[D, I]
    n_log_1mgamma = n_trans[D, M] + n_trans[D, D] + n_trans[D, E]
    n_log_kappa = (n_trans[S, M] + n_trans[S, D] +
                   n_trans[M, M] + n_trans[M, D] +
                   n_trans[I, M] + n_trans[I, D] +
                   n_trans[D, M] + n_trans[D, D])
    n_log_1mkappa = n_trans[S, E] + n_trans[M, E] + n_trans[I, E] + n_trans[D, E]

    return {
        'log_alpha': n_log_alpha,
        'log_1malpha': n_log_1malpha,
        'log_beta': n_log_beta,
        'log_1mbeta': n_log_1mbeta,
        'log_gamma': n_log_gamma,
        'log_1mgamma': n_log_1mgamma,
        'log_kappa': n_log_kappa,
        'log_1mkappa': n_log_1mkappa,
    }


# --- BDI sufficient statistics from FB counts (tkf.tex eqs exposure-tkf, births-tkf, deaths-tkf) ---

# The 6 WFST groups (no κ, 1-κ geometric prior)
_WFST_GROUPS = ('log_alpha', 'log_1malpha', 'log_beta', 'log_1mbeta',
                'log_gamma', 'log_1mgamma')


def _wfst_scores(groups, ins_rate, del_rate, t):
    """Compute λ·∂logP/∂λ and μ·∂logP/∂μ using WFST groups only.

    Returns (lam_score, mu_score) where:
        lam_score = Σ_g n_g · (λ · ∂log(ξ_g)/∂λ)
        mu_score  = Σ_g n_g · (μ · ∂log(ξ_g)/∂μ)
    summing only over the 6 WFST groups (α, 1-α, β, 1-β, γ, 1-γ).
    Uses L'Hôpital-gated derivatives from score_derivatives().
    """
    derivs = score_derivatives(ins_rate, del_rate, t)
    lam_score = 0.0
    mu_score = 0.0
    for g in _WFST_GROUPS:
        dl, dm = derivs[g]
        lam_score = lam_score + groups[g] * (ins_rate * dl)
        mu_score = mu_score + groups[g] * (del_rate * dm)
    return lam_score, mu_score


def _wfst_scores_general(groups, ins_rate, del_rate, t):
    """Like _wfst_scores but always uses general (non-limit) formulas.

    Stable when λ ≠ μ. Used for finite-difference L'Hôpital computation.
    """
    derivs = _score_derivatives_general(ins_rate, del_rate, t)
    lam_score = 0.0
    mu_score = 0.0
    for g in _WFST_GROUPS:
        dl, dm = derivs[g]
        lam_score = lam_score + groups[g] * (ins_rate * dl)
        mu_score = mu_score + groups[g] * (del_rate * dm)
    return lam_score, mu_score


def _wfst_scores_smooth(groups, ins_rate, del_rate, t):
    """Compute λ·∂logP/∂λ and μ·∂logP/∂μ via autodiff on smooth β/γ formulas.

    Unlike _wfst_scores (which uses analytic derivatives with gated limits),
    this uses jax.grad on _logP_wfst_smooth, which is differentiable and
    numerically stable at all (λ, μ) including λ=μ.

    Returns (lam_score, mu_score) as Python floats.
    """
    n_a = groups['log_alpha']
    n_1a = groups['log_1malpha']
    n_b = groups['log_beta']
    n_1b = groups['log_1mbeta']
    n_g = groups['log_gamma']
    n_1g = groups['log_1mgamma']

    lam = jnp.asarray(float(ins_rate))
    mu = jnp.asarray(float(del_rate))

    dlogP_dlam = jax.grad(
        lambda l: _logP_wfst_smooth(l, mu, t, n_a, n_1a, n_b, n_1b, n_g, n_1g)
    )(lam)
    dlogP_dmu = jax.grad(
        lambda m: _logP_wfst_smooth(lam, m, t, n_a, n_1a, n_b, n_1b, n_g, n_1g)
    )(mu)

    return float(ins_rate) * float(dlogP_dlam), float(del_rate) * float(dlogP_dmu)


def _es_numerator(groups, cons, ins_rate, del_rate, t, T_total=None):
    """Numerator of E[S] formula (eq:exposure-tkf): Σ n̂(C^cons - C^λ + C^μ) - λT.

    Uses general (non-limit) score derivative formulas. Stable when λ ≠ μ.
    T_total is the total observation time (defaults to t if not provided).
    """
    if T_total is None:
        T_total = t
    lam_score, mu_score = _wfst_scores_general(groups, ins_rate, del_rate, t)
    return cons + mu_score - lam_score - ins_rate * T_total


def tkf91_stats_from_counts(n_trans, ins_rate, del_rate, t, T=None):
    """Compute E[B], E[D], E[S] from a TKF91 (5,5) WFST transition-count matrix.

    Implements tkf.tex eqs (eq:exposure-tkf)--(eq:deaths-tkf):
        E[S] = (Σ n̂_ij (C^cons - C^λ + C^μ) - λT) / (λ - μ)
        E[B] = λ · Σ n̂_ij C^λ + λ·E[S] + λT
        E[D] = μ · Σ n̂_ij C^μ + μ·E[S]

    where C^λ, C^μ are WFST log-elasticities (6 groups, no κ)
    and C^cons_ij = δ(j=I) - δ(j=D).

    .. note:: PURELY TKF91. The input n_trans MUST already be in TKF91
       state semantics — i.e. self-loops `n_trans[s, s]` (s ∈ {M, I, D})
       must represent "next fragment starts with state s in TKF91"
       events. For TKF92 chi counts, self-loops mix extension events
       with new-fragment events (see tkf92_trans in
       tkfmixdom/jax/core/params.py:92-108) and you MUST decompose them
       BEFORE calling this. Use `tkf92_stats_from_counts` (below) for
       raw TKF92 chi counts.

    Uses smooth (differentiable) β/γ parameterization for computing score
    derivatives, which is numerically stable at all λ/μ ratios.
    When λ ≈ μ, E[S] uses the analytic L'Hôpital limit (eq:ES-limit from
    lhopital-limits.tex) to avoid 0/0 cancellation.

    Args:
        n_trans: (5, 5) expected transition counts from Forward-Backward.
        ins_rate: insertion rate λ.
        del_rate: deletion rate μ.
        t: per-process evolutionary time.
        T: total BDI observation time. If None, defaults to t (single process).
           For aggregated counts over multiple processes, pass T = Σ t_n × M_n
           where M_n is the number of independent BDI processes per training
           example (sec:bw-mixdom, T accumulation).

    Returns:
        (E_B, E_D, E_S) tuple of floats.
    """
    from .params import S, M, I, D, E

    # Total observation time (sec:bw-mixdom, T accumulation)
    T_total = T if T is not None else t

    groups = transition_count_groups(n_trans)

    # Compute scores using smooth (differentiable) β/γ formulas.
    # This avoids the singularities in the analytic score derivative formulas
    # when λ ≈ μ, and gives identical results to the analytic formulas
    # when λ ≠ μ (same mathematical expressions, just evaluated smoothly).
    lam_score, mu_score = _wfst_scores_smooth(groups, ins_rate, del_rate, t)

    # Conservation law: E[B] - E[D] = (I column sum) - (D column sum)
    # (eq:conservation-tkf)
    cons = float(jnp.sum(n_trans[:, I]) - jnp.sum(n_trans[:, D]))

    kappa = float(ins_rate) / float(del_rate)
    near_equal = abs(1.0 - kappa) < ES_LHOPITAL_THRESHOLD

    if not near_equal:
        # General case: direct division (eq:exposure-tkf)
        eps = float(ins_rate - del_rate)
        numer = cons + mu_score - lam_score - float(ins_rate) * T_total
        E_S = numer / eps
    else:
        # L'Hôpital limit (eq:ES-limit from lhopital-limits.tex):
        # E[S] = f'(μ) where f(λ) is the E[S] numerator.
        # Uses analytic second derivatives via JAX autodiff on smooth β/γ.
        E_S = _es_limit_analytic(groups, cons, del_rate, t, T_total)

    E_B = lam_score + float(ins_rate) * E_S + float(ins_rate) * T_total
    E_D = mu_score + float(del_rate) * E_S

    # Invariant checks (Phase 4)
    if not (E_S >= -1e-6):
        import warnings
        warnings.warn(f"bdi_stats_from_counts: E_S={E_S:.6g} < 0 "
                      f"(λ={float(ins_rate):.4g}, μ={float(del_rate):.4g}, t={t:.4g})")
    # Conservation: E_B - E_D ≈ I_col - D_col (eq:conservation-tkf)
    conservation_err = abs((E_B - E_D) - cons)
    tol = 1e-4
    if conservation_err > tol * (abs(E_B) + abs(E_D) + 1):
        import warnings
        warnings.warn(f"bdi_stats_from_counts: conservation violation "
                      f"|({E_B:.6g}-{E_D:.6g})-{cons:.6g}| = {conservation_err:.6g}")

    return E_B, E_D, E_S


# Backwards-compat alias. New code should call `tkf91_stats_from_counts`
# directly to make the input-semantics requirement (TKF91 n_trans only)
# explicit.
bdi_stats_from_counts = tkf91_stats_from_counts


def tkf92_stats_from_counts(n_trans, ins_rate, del_rate, t, ext, T=None):
    """Compute E[B], E[D], E[S] PLUS extension counts from a TKF92 (5,5)
    WFST transition-count matrix.

    Bridges raw TKF92 chi counts to the TKF91 BDI math. TKF92's chi
    self-loops mix two distinct event types:

        tkf92_trans[s, s] = ext + (1 - ext) * tkf91_trans[s, s]
                            └──────┘   └─────────────────────────┘
                            extension       new-fragment-with-s
                            event           event (TKF91-style)

    so n_trans[s, s] (for s ∈ {M, I, D}) cannot be passed directly to
    `tkf91_stats_from_counts` — the TKF91 formula expects only the
    (1-ext) part. We split each self-loop in proportion to those two
    components, hand the TKF91-only portion to the TKF91 BDI machinery,
    and accumulate the extension count separately for the ext M-step.

    The decomposition factor is

        ext_frac[s] = ext / tkf92_trans[s, s]

    so

        ext_count_s     = ext_frac[s]      * n_trans[s, s]
        tkf91_diag_s    = (1 - ext_frac[s]) * n_trans[s, s]

    The cleaned n_trans (TKF91 diagonals only) goes to
    `tkf91_stats_from_counts` and the (E_B, E_D, E_S) it returns is
    valid TKF91 stats for the underlying ins/del rates. The total
    `ext_count` (sum over s) and `notext_count` (= sum of all body
    transitions that are NOT extensions) are returned alongside, for
    use by an ext-Beta M-step:

        ext_new = ext_count / (ext_count + notext_count)

    Args:
        n_trans:    (5, 5) — TKF92 chi WFST n_trans matrix.
        ins_rate:   λ.
        del_rate:   μ.
        t:          per-process evolutionary time.
        ext:        current ext (used for the decomposition factor).
        T:          total BDI observation time. If None, defaults to t.

    Returns:
        dict with keys 'E_B', 'E_D', 'E_S', 'ext_count', 'notext_count'.
    """
    from .params import S as _S, M as _M, I as _I, D as _D, E as _E
    from .params import tkf91_trans, tkf92_trans

    n_trans = np.asarray(n_trans, dtype=np.float64).copy()
    tau91 = np.asarray(tkf91_trans(float(ins_rate), float(del_rate),
                                     float(t)))
    tau92 = np.asarray(tkf92_trans(float(ins_rate), float(del_rate),
                                     float(t), float(ext)))

    ext_count = 0.0
    for s in (_M, _I, _D):
        diag92 = float(tau92[s, s])
        if diag92 < 1e-30:
            continue
        ext_frac_s = float(ext) / diag92
        # Numerical safety: ext_frac should lie in [0, 1] by construction.
        ext_frac_s = max(0.0, min(1.0, ext_frac_s))
        loop_count = float(n_trans[s, s])
        ext_part = ext_frac_s * loop_count
        # Replace the diagonal with the TKF91-only component.
        n_trans[s, s] = (1.0 - ext_frac_s) * loop_count
        ext_count += ext_part

    # Per body-tkf92.tex sec:bw-tkf92:
    #   E = sum_{a ∈ {M,I,D}} sum_b ñ'_{ab} - F
    # i.e. total outgoing from body states (across ALL destinations
    # including E for chain ends), minus F = ext_count.  Equivalently
    # (since only the diagonals were modified above and modified
    # n̂_{a,a} = (1-ext_frac)·ñ'_{a,a}):
    #   E = sum_a sum_b n̂_{ab} = sum_a (modified row a sum).
    # Body→E transitions ARE counted in the denominator: a chain end
    # is a non-extension event (chi[s, E] = (1-ext)·tau91[s, E]).
    notext_count = float(
        n_trans[(_M, _I, _D), :].sum())

    E_B, E_D, E_S = tkf91_stats_from_counts(
        n_trans, ins_rate, del_rate, t, T=T)
    return {
        'E_B': float(E_B), 'E_D': float(E_D), 'E_S': float(E_S),
        'ext_count': float(ext_count),
        'notext_count': float(notext_count),
        # Resolved (TKF91-level after subtracting fragment extensions
        # F_a = ext_frac[s] · ñ'_{aa}) count matrix.  This is what
        # `transition_count_groups` should be applied to when computing
        # L = sum_i (n̂_{iM} + n̂_{iD}) and M = sum_i n̂_{iE} for the
        # κ-quadratic M-step (see body-tkf92.tex sec:bw-tkf92,
        # eq for n̂_{ab} = ñ'_{ab} - δ_{ab} F_a).
        'n_trans_resolved': n_trans,
    }


# ============================================================
# Batched (vmapped) BDI sufficient statistics
# ============================================================
#
# `bdi_stats_from_counts` is called once per pair (and per domain) inside
# `exact_suffstats_per_pair_batch`. Each call traces `jax.grad` twice on a
# fresh closure that captures `(t, n_alpha, ..., n_1mgamma)` — for B=2000
# pairs and 1+K domain-level calls this becomes the dominant per-iteration
# Python+JAX-dispatch overhead (≈ 540 sec/iter at d3f1).
#
# The batched version vmaps the entire computation over a (B,) leading axis
# on `(n_trans, t, T)` while sharing the scalar `(ins_rate, del_rate)`. The
# Python-level `if not near_equal:` branch can stay outside the JIT because
# `(ins_rate, del_rate)` are scalars — `near_equal` is a single boolean for
# the whole batch.
#
# Per-pair-t coherence is preserved: each batch element uses its own t_p
# and n_trans_p; no averaging or representative-t shortcut is introduced.


def _bdi_general_core(n_trans, ins_rate, del_rate, t, T_total):
    """Pure-JAX general-case BDI for a single pair (vmappable).

    Replicates `bdi_stats_from_counts` for the |1-κ| ≥ ES_LHOPITAL_THRESHOLD
    branch. All ops are JAX, so this is safe under vmap.
    """
    from .params import S, M, I, D, E
    # Inline transition_count_groups (numpy-style indexing also works under
    # vmap, where leading batch axes are just transparent)
    n_la = n_trans[S, M] + n_trans[M, M] + n_trans[I, M] + n_trans[D, M]
    n_l1ma = n_trans[S, D] + n_trans[M, D] + n_trans[I, D] + n_trans[D, D]
    n_lb = n_trans[S, I] + n_trans[M, I] + n_trans[I, I]
    n_l1mb = (n_trans[S, M] + n_trans[S, D] + n_trans[S, E] +
              n_trans[M, M] + n_trans[M, D] + n_trans[M, E] +
              n_trans[I, M] + n_trans[I, D] + n_trans[I, E])
    n_lg = n_trans[D, I]
    n_l1mg = n_trans[D, M] + n_trans[D, D] + n_trans[D, E]

    # _wfst_scores_smooth: λ·∂logP/∂λ and μ·∂logP/∂μ via autodiff on the
    # smooth β/γ formulas (numerically stable at all (λ, μ)).
    dlogP_dlam = jax.grad(
        lambda l: _logP_wfst_smooth(
            l, del_rate, t, n_la, n_l1ma, n_lb, n_l1mb, n_lg, n_l1mg)
    )(ins_rate)
    dlogP_dmu = jax.grad(
        lambda m: _logP_wfst_smooth(
            ins_rate, m, t, n_la, n_l1ma, n_lb, n_l1mb, n_lg, n_l1mg)
    )(del_rate)
    lam_score = ins_rate * dlogP_dlam
    mu_score = del_rate * dlogP_dmu

    # Conservation law
    cons = jnp.sum(n_trans[:, I]) - jnp.sum(n_trans[:, D])

    # General E[S] formula (eq:exposure-tkf)
    eps = ins_rate - del_rate
    numer = cons + mu_score - lam_score - ins_rate * T_total
    E_S = numer / eps

    E_B = lam_score + ins_rate * E_S + ins_rate * T_total
    E_D = mu_score + del_rate * E_S
    return E_B, E_D, E_S


def _bdi_limit_core(n_trans, ins_rate, del_rate, t, T_total):
    """Pure-JAX L'Hôpital-limit BDI for a single pair (vmappable).

    Replicates `bdi_stats_from_counts` for the |1-κ| < ES_LHOPITAL_THRESHOLD
    branch (eq:ES-limit). Inputs as in `_bdi_general_core`.
    """
    from .params import S, M, I, D, E
    n_la = n_trans[S, M] + n_trans[M, M] + n_trans[I, M] + n_trans[D, M]
    n_l1ma = n_trans[S, D] + n_trans[M, D] + n_trans[I, D] + n_trans[D, D]
    n_lb = n_trans[S, I] + n_trans[M, I] + n_trans[I, I]
    n_l1mb = (n_trans[S, M] + n_trans[S, D] + n_trans[S, E] +
              n_trans[M, M] + n_trans[M, D] + n_trans[M, E] +
              n_trans[I, M] + n_trans[I, D] + n_trans[I, E])
    n_lg = n_trans[D, I]
    n_l1mg = n_trans[D, M] + n_trans[D, D] + n_trans[D, E]

    dlogP_dlam = jax.grad(
        lambda l: _logP_wfst_smooth(
            l, del_rate, t, n_la, n_l1ma, n_lb, n_l1mb, n_lg, n_l1mg)
    )(ins_rate)
    dlogP_dmu = jax.grad(
        lambda m: _logP_wfst_smooth(
            ins_rate, m, t, n_la, n_l1ma, n_lb, n_l1mb, n_lg, n_l1mg)
    )(del_rate)
    lam_score = ins_rate * dlogP_dlam
    mu_score = del_rate * dlogP_dmu

    cons = jnp.sum(n_trans[:, I]) - jnp.sum(n_trans[:, D])

    # L'Hôpital limit (eq:ES-limit). f(λ) = cons + μ·∂logP/∂μ − λ·∂logP/∂λ
    # − λ·T_total; E[S] = f'(μ).
    mu = del_rate

    def _numerator(lam):
        d_lam = jax.grad(
            lambda l: _logP_wfst_smooth(
                l, mu, t, n_la, n_l1ma, n_lb, n_l1mb, n_lg, n_l1mg)
        )(lam)
        d_mu = jax.grad(
            lambda m: _logP_wfst_smooth(
                lam, m, t, n_la, n_l1ma, n_lb, n_l1mb, n_lg, n_l1mg)
        )(mu)
        return cons + mu * d_mu - lam * d_lam - lam * T_total

    E_S = jax.grad(_numerator)(mu)

    E_B = lam_score + ins_rate * E_S + ins_rate * T_total
    E_D = mu_score + del_rate * E_S
    return E_B, E_D, E_S


_bdi_general_batch_jit_cache = None
_bdi_limit_batch_jit_cache = None


def _get_bdi_general_batch_jit():
    global _bdi_general_batch_jit_cache
    if _bdi_general_batch_jit_cache is None:
        # in_axes: n_trans batched (axis 0), ins/del shared, t/T batched (axis 0)
        vmapped = jax.vmap(_bdi_general_core,
                           in_axes=(0, None, None, 0, 0))
        _bdi_general_batch_jit_cache = jax.jit(vmapped)
    return _bdi_general_batch_jit_cache


def _get_bdi_limit_batch_jit():
    global _bdi_limit_batch_jit_cache
    if _bdi_limit_batch_jit_cache is None:
        vmapped = jax.vmap(_bdi_limit_core,
                           in_axes=(0, None, None, 0, 0))
        _bdi_limit_batch_jit_cache = jax.jit(vmapped)
    return _bdi_limit_batch_jit_cache


def tkf91_stats_from_counts_batch(n_trans_batch, ins_rate, del_rate,
                                    t_batch, T_batch=None):
    """Vmapped per-pair `tkf91_stats_from_counts`.

    Same TKF91-input-semantics caveat as the scalar version: each per-pair
    n_trans must already be in TKF91 state semantics. For raw TKF92 chi
    counts, decompose self-loops via `tkf92_stats_from_counts` first.

    Single JAX call (vmapped over B) returns per-pair (E[B], E[D], E[S])
    arrays. Replaces the per-pair Python loop over `bdi_stats_from_counts`
    in `exact_suffstats_per_pair_batch` — at B=2000 with 1+K calls per pair
    this reduces 8000 jax.grad-tracing dispatches to 1+K dispatches.

    Per-pair-t coherence: each batch element uses its own (n_trans_p, t_p,
    T_p); the shared scalars are only `(ins_rate, del_rate)` — which IS the
    correct semantics, since the M-step parameters are frozen across this
    E-step iteration. No averaging across pairs, no representative-t.

    Branch on `|1 - κ| < ES_LHOPITAL_THRESHOLD` is at the Python (scalar)
    level since `(ins_rate, del_rate)` are shared — same logic as the
    scalar `bdi_stats_from_counts`.

    Args:
        n_trans_batch: (B, 5, 5) per-pair expected transition counts.
        ins_rate:      scalar λ (frozen for batch).
        del_rate:      scalar μ (frozen for batch).
        t_batch:       (B,) per-pair evolutionary times.
        T_batch:       (B,) per-pair total observation times.
                       If None, defaults to t_batch.

    Returns:
        (E_B, E_D, E_S) tuple of np.ndarray, each shape (B,).
    """
    import numpy as _np
    n_trans_batch = jnp.asarray(n_trans_batch)
    t_batch = jnp.asarray(t_batch)
    if T_batch is None:
        T_batch = t_batch
    else:
        T_batch = jnp.asarray(T_batch)
    ins_rate_j = jnp.asarray(float(ins_rate))
    del_rate_j = jnp.asarray(float(del_rate))

    kappa = float(ins_rate) / float(del_rate)
    near_equal = abs(1.0 - kappa) < ES_LHOPITAL_THRESHOLD

    if near_equal:
        fn = _get_bdi_limit_batch_jit()
    else:
        fn = _get_bdi_general_batch_jit()
    E_B, E_D, E_S = fn(n_trans_batch, ins_rate_j, del_rate_j,
                       t_batch, T_batch)
    E_B = _np.asarray(E_B)
    E_D = _np.asarray(E_D)
    E_S = _np.asarray(E_S)

    # Preserve scalar `bdi_stats_from_counts` invariant warnings, but emit
    # at most one per call (over potentially many pairs) to avoid log spam.
    es_neg = E_S < -1e-6
    if _np.any(es_neg):
        n_bad = int(es_neg.sum())
        worst_idx = int(_np.argmin(E_S))
        import warnings
        warnings.warn(
            f"bdi_stats_from_counts_batch: {n_bad}/{E_S.size} pairs have "
            f"E_S<-1e-6 (worst E_S={E_S[worst_idx]:.6g}, "
            f"λ={float(ins_rate):.4g}, μ={float(del_rate):.4g}, "
            f"t={float(t_batch[worst_idx]):.4g})")

    from .params import I as _I_idx, D as _D_idx
    n_trans_np = _np.asarray(n_trans_batch)
    cons_np = (n_trans_np[:, :, _I_idx].sum(axis=1)
               - n_trans_np[:, :, _D_idx].sum(axis=1))
    cons_err = _np.abs((E_B - E_D) - cons_np)
    cons_tol = 1e-4
    cons_bad = cons_err > cons_tol * (_np.abs(E_B) + _np.abs(E_D) + 1)
    if _np.any(cons_bad):
        n_bad = int(cons_bad.sum())
        worst_idx = int(_np.argmax(cons_err))
        import warnings
        warnings.warn(
            f"bdi_stats_from_counts_batch: {n_bad}/{E_S.size} pairs have "
            f"conservation violation (worst err={cons_err[worst_idx]:.6g} at "
            f"E_B={E_B[worst_idx]:.6g}, E_D={E_D[worst_idx]:.6g}, "
            f"cons={cons_np[worst_idx]:.6g})")

    return E_B, E_D, E_S


# Backwards-compat alias (see tkf91_stats_from_counts above).
bdi_stats_from_counts_batch = tkf91_stats_from_counts_batch


import math


def m_step_indel_quadratic(B, D, S, L, M, T,
                           prior_alpha_lam=1.0, prior_alpha_mu=1.0,
                           prior_beta=0.0):
    """Closed-form joint M-step via quadratic in κ.

    Implements tkf.tex eqs (eq:kappa-quadratic)--(eq:insrate-root):
        a = B + L,  b = D - L - M
        κ²(S+T)b - κ[S(a+b+2M) + T(b+M)] + Sa = 0
        μ = (b + M/(1-κ)) / S,  λ = κμ

    With Gamma(α_λ, β) prior on λ and Gamma(α_μ, β) on μ (shared rate β),
    pseudocounts are added before solving (sec:bw-mixdom):
        B → B + α_λ - 1,  D → D + α_μ - 1,  S → S + β.

    Args:
        B, D, S: BDI sufficient statistics (from bdi_stats_from_counts).
        L: expected ancestor length (n_kappa from count matrix).
        M: expected number of trajectory terminations (n_{1-kappa}).
        T: total BDI observation time.
        prior_alpha_lam, prior_alpha_mu: Gamma shape parameters.
        prior_beta: shared Gamma rate parameter (augments S).

    Returns:
        (ins_rate_new, del_rate_new) tuple of floats.
    """
    # Augment with prior pseudocounts (sec:bw-mixdom, Priors)
    B_aug = B + prior_alpha_lam - 1.0
    D_aug = D + prior_alpha_mu - 1.0
    S_aug = S + prior_beta

    a = B_aug + L
    b = D_aug - L - M

    # Quadratic coefficients (eq:kappa-quadratic)
    A_k = (S_aug + T) * b
    B_k = -(S_aug * (a + b + 2 * M) + T * (b + M))
    C_k = S_aug * a

    # Solve for κ ∈ (0, 1) — take the smaller root (eq:kappa-root)
    disc = B_k * B_k - 4.0 * A_k * C_k
    if disc < 0 or A_k == 0:
        # Degenerate: fall back to conditioned MLE (no geometric prior)
        ins_new = max(B_aug, 1e-10) / max(S_aug + T, 1e-30)
        del_new = max(D_aug, 1e-10) / max(S_aug, 1e-30)
        return float(ins_new), float(max(del_new, ins_new + 1e-10))

    sqrt_disc = math.sqrt(max(disc, 0.0))
    kappa = (-B_k - sqrt_disc) / (2.0 * A_k)

    # Clamp κ to valid range
    kappa = max(min(kappa, 1.0 - 1e-10), 1e-10)

    # Recover rates (eq:insrate-root)
    del_new = (b + M / (1.0 - kappa)) / max(S_aug, 1e-30)
    ins_new = kappa * del_new

    # Ensure λ > 0 and μ > λ (Phase 4 invariant)
    ins_new = max(ins_new, 1e-10)
    del_new = max(del_new, ins_new + 1e-10)

    return float(ins_new), float(del_new)


# --- Invariant check utilities (Phase 4) ---

def check_count_sanity(n_trans, label=""):
    """Warn if transition counts contain NaN or negative values."""
    import warnings
    import numpy as _np
    n = _np.asarray(n_trans)
    if _np.any(_np.isnan(n)):
        warnings.warn(f"NaN in transition counts {label}")
    if _np.any(n < -1e-10):
        warnings.warn(f"Negative transition counts {label}: min={n.min():.6g}")
