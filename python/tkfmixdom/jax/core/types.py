"""Core types for the compositional parameter & likelihood framework.

Defines typed parameters (with conjugate priors), sufficient statistics,
grammar specs, and compiled models. These form aligned pytrees: params,
priors, and sufficient stats share the same tree structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Parameter types (each with conjugate prior)
# ---------------------------------------------------------------------------

@dataclass
class RateParam:
    """Positive real parameter (λ, μ, etc.). Prior: Gamma(shape, rate)."""
    value: float
    prior_shape: float = 2.0
    prior_rate: float = 10.0

    def log_prior(self) -> float:
        """Log Gamma(shape, rate) density at self.value."""
        a, b = self.prior_shape, self.prior_rate
        return (a - 1.0) * jnp.log(self.value) - b * self.value

    def log_prior_grad(self) -> float:
        """∂ log_prior / ∂ value."""
        return (self.prior_shape - 1.0) / self.value - self.prior_rate


@dataclass
class SimplexParam:
    """Probability vector. Prior: Dirichlet(alpha)."""
    value: jax.Array        # (K,) sums to 1
    prior_alpha: jax.Array  # (K,) pseudocounts

    def log_prior(self) -> float:
        """Log Dirichlet(alpha) density at self.value (up to normalization)."""
        return jnp.sum((self.prior_alpha - 1.0) * jnp.log(self.value))

    def log_prior_grad(self) -> jax.Array:
        """∂ log_prior / ∂ value_k."""
        return (self.prior_alpha - 1.0) / self.value


@dataclass
class BernoulliParam:
    """Probability in [0, 1] (extension rate, split prob). Prior: Beta(a, b)."""
    value: float
    prior_a: float = 1.0
    prior_b: float = 1.0

    def log_prior(self) -> float:
        """Log Beta(a, b) density at self.value (up to normalization)."""
        return ((self.prior_a - 1.0) * jnp.log(self.value)
                + (self.prior_b - 1.0) * jnp.log(1.0 - self.value))

    def log_prior_grad(self) -> float:
        """∂ log_prior / ∂ value."""
        return (self.prior_a - 1.0) / self.value - (self.prior_b - 1.0) / (1.0 - self.value)


# ---------------------------------------------------------------------------
# Sufficient statistics (parallel to parameter types)
# ---------------------------------------------------------------------------

@dataclass
class BDISuffStats:
    """Sufficient statistics for a BDI process.

    Recovered from the score function identity relating expected transition
    counts to derivatives of log-likelihood w.r.t. insertion and deletion rates.

    The ancestor length counts n_kappa and n_1mkappa track the geometric(κ)
    prior contribution κ^i·(1-κ)^n. These cannot be expressed as births,
    deaths, or time-integrated population. They are included in the VJP and
    M-step only when using the joint likelihood (not conditional).

    Instance semantics depend on context: either DATA suff stats (used with
    map_update) or POSTERIOR PSEUDOCOUNTS α̃ = α_prior + data (used with
    to_params_from_pseudocount / log_J). See svb-convergence.tex
    §Pseudocount representation.
    """
    E_B: float  # expected number of births
    E_D: float  # expected number of deaths
    E_S: float  # expected time-integrated population
    n_kappa: float = 0.0     # total ancestor length (n_M_col + n_D_col)
    n_1mkappa: float = 0.0   # number of ancestor sequences (n_End transitions)
    T: float = 0.0           # total BDI observation time (accumulates across pairs)

    def __add__(self, other: "BDISuffStats") -> "BDISuffStats":
        return BDISuffStats(
            E_B=self.E_B + other.E_B, E_D=self.E_D + other.E_D,
            E_S=self.E_S + other.E_S, n_kappa=self.n_kappa + other.n_kappa,
            n_1mkappa=self.n_1mkappa + other.n_1mkappa,
            T=self.T + other.T)

    def scaled(self, factor: float) -> "BDISuffStats":
        return BDISuffStats(
            E_B=self.E_B * factor, E_D=self.E_D * factor,
            E_S=self.E_S * factor, n_kappa=self.n_kappa * factor,
            n_1mkappa=self.n_1mkappa * factor, T=self.T * factor)

    def map_update(self, ins_param: RateParam, del_param: RateParam,
                   t: float = 0.0) -> tuple[float, float]:
        """MAP update for (λ, μ) from BDI stats + Gamma priors.

        Always uses the joint quadratic M-step (eq:kappa-quadratic in tkf.tex)
        which includes L=n_kappa and M=n_1mkappa counts and enforces κ < 1.

        Uses self.T (accumulated observation time across pairs) in the
        denominator. The `t` parameter is a fallback: if self.T == 0 and
        t > 0, uses t (backward compat for single-pair case).

        Returns (λ, μ) with λ < μ guaranteed.
        """
        from ..core.bdi import m_step_indel_quadratic

        # Use accumulated T, fall back to per-pair t for backward compat
        T_total = self.T if self.T > 0 else t

        lam, mu = m_step_indel_quadratic(
            self.E_B, self.E_D, self.E_S,
            L=self.n_kappa, M=self.n_1mkappa, T=T_total,
            prior_alpha_lam=ins_param.prior_shape,
            prior_alpha_mu=del_param.prior_shape,
            prior_beta=ins_param.prior_rate)

        return max(float(lam), 1e-8), max(float(mu), float(lam) + 1e-8)

    def to_params_from_pseudocount(
            self, alpha_prior: "BDIPseudocountPrior",
            t: float = 0.0) -> tuple[float, float]:
        """Closed-form M-step (λ, μ) interpreting self as posterior
        pseudocounts α̃ and alpha_prior as α_prior.

        Internally the existing m_step_indel_quadratic takes (data, prior)
        rather than (α̃); reconstruct data = α̃ − α_prior and forward.
        """
        from ..core.bdi import m_step_indel_quadratic

        T_total = self.T if self.T > 0 else t
        B_data = self.E_B - alpha_prior.alpha_lam
        D_data = self.E_D - alpha_prior.alpha_mu
        S_data = self.E_S - alpha_prior.beta

        lam, mu = m_step_indel_quadratic(
            B_data, D_data, S_data,
            L=self.n_kappa, M=self.n_1mkappa, T=T_total,
            prior_alpha_lam=alpha_prior.alpha_lam,
            prior_alpha_mu=alpha_prior.alpha_mu,
            prior_beta=alpha_prior.beta)
        return max(float(lam), 1e-8), max(float(mu), float(lam) + 1e-8)

    def log_J(self, ins: float, mu: float, t: float = 0.0) -> float:
        """Complete-data log joint J(λ, μ | α̃) = α̃·log θ − Z(θ), up to a
        constant. The M-step yields argmax of this objective.

        With self interpreted as α̃:
            J = (α̃_B − 1) log λ + (α̃_D − 1) log μ
                − λ (α̃_S + T) − μ α̃_S
                + L log κ + M log(1 − κ)
        where κ = λ/μ, L = self.n_kappa, M = self.n_1mkappa, T = self.T (or t).
        """
        import math
        T_total = self.T if self.T > 0 else t
        ins_s = max(float(ins), 1e-300)
        mu_s = max(float(mu), 1e-300)
        kappa = ins_s / mu_s
        kappa = min(max(kappa, 1e-300), 1.0 - 1e-15)
        J = ((self.E_B - 1.0) * math.log(ins_s)
             + (self.E_D - 1.0) * math.log(mu_s)
             - ins_s * (self.E_S + T_total)
             - mu_s * self.E_S
             + self.n_kappa * math.log(kappa)
             + self.n_1mkappa * math.log(1.0 - kappa))
        return float(J)


@dataclass
class BDIPseudocountPrior:
    """Prior pseudocounts for a BDI component: Gamma(α_λ, β), Gamma(α_μ, β).

    Stored separately from BDISuffStats so the same class can carry
    either data stats or posterior pseudocounts α̃.
    """
    alpha_lam: float   # Gamma shape for λ
    alpha_mu: float    # Gamma shape for μ
    beta: float        # shared Gamma rate (scale parameter in tkf.tex)

    def as_suffstats(self) -> "BDISuffStats":
        """Prior represented as a BDISuffStats (α_B=α_λ, α_D=α_μ, α_S=β)."""
        return BDISuffStats(
            E_B=self.alpha_lam, E_D=self.alpha_mu, E_S=self.beta,
            n_kappa=0.0, n_1mkappa=0.0, T=0.0)


@dataclass
class SimplexSuffStats:
    """Posterior category counts for a mixture parameter.

    Instance semantics depend on context: either DATA suff stats (used with
    map_update) or POSTERIOR PSEUDOCOUNTS α̃ (used with log_J /
    to_params_from_pseudocount). See svb-convergence.tex
    §Pseudocount representation.
    """
    counts: jax.Array  # (K,)

    def __add__(self, other: "SimplexSuffStats") -> "SimplexSuffStats":
        return SimplexSuffStats(counts=self.counts + other.counts)

    def scaled(self, factor: float) -> "SimplexSuffStats":
        return SimplexSuffStats(counts=self.counts * factor)

    def map_update(self, param: SimplexParam) -> jax.Array:
        """MAP update: (counts + α - 1) / sum, i.e. Dirichlet MAP."""
        unnorm = jnp.maximum(self.counts + param.prior_alpha - 1.0, 0.0)
        total = jnp.sum(unnorm)
        K = self.counts.shape[0]
        return jnp.where(total > 1e-30, unnorm / total,
                         jnp.ones(K) / K)

    def to_params_from_pseudocount(self) -> jax.Array:
        """Dirichlet MAP interpreting self.counts as α̃: θ_i ∝ (α̃_i − 1)_+."""
        unnorm = jnp.maximum(self.counts - 1.0, 0.0)
        total = jnp.sum(unnorm)
        K = self.counts.shape[0]
        return jnp.where(total > 1e-30, unnorm / total,
                         jnp.ones(K) / K)

    def log_J(self, theta: jax.Array) -> float:
        """J(θ|α̃) = Σ_i (α̃_i − 1) log θ_i, Dirichlet log-density up to const."""
        import numpy as _np
        theta_safe = jnp.maximum(theta, 1e-300)
        return float(jnp.sum((self.counts - 1.0) * jnp.log(theta_safe)))


@dataclass
class BernoulliSuffStats:
    """Extension/split loop counts.

    Instance semantics depend on context: either DATA (used with map_update)
    or POSTERIOR PSEUDOCOUNTS α̃ = (α̃_succ, α̃_fail) (used with log_J /
    to_params_from_pseudocount).
    """
    n_success: float  # times the event occurred (extend / split)
    n_failure: float  # times it did not (exit / character)

    def __add__(self, other: "BernoulliSuffStats") -> "BernoulliSuffStats":
        return BernoulliSuffStats(
            n_success=self.n_success + other.n_success,
            n_failure=self.n_failure + other.n_failure)

    def scaled(self, factor: float) -> "BernoulliSuffStats":
        return BernoulliSuffStats(
            n_success=self.n_success * factor,
            n_failure=self.n_failure * factor)

    def map_update(self, param: BernoulliParam) -> float:
        """MAP update: Beta MAP = (n_success + a - 1) / (n_total + a + b - 2)."""
        num = max(self.n_success + param.prior_a - 1.0, 0.0)
        den = max(self.n_success + self.n_failure
                  + param.prior_a + param.prior_b - 2.0, 1e-30)
        return num / den

    def to_params_from_pseudocount(self) -> float:
        """Beta MAP interpreting (n_success, n_failure) as (α̃_a, α̃_b):
        θ̂ = (α̃_a − 1)_+ / (α̃_a + α̃_b − 2)_+, uniform fallback when denom ≤ 0.
        """
        num = max(self.n_success - 1.0, 0.0)
        den = max(self.n_success + self.n_failure - 2.0, 1e-30)
        return num / den if num > 0 else 0.5

    def log_J(self, theta: float) -> float:
        """J(θ|α̃) = (α̃_a − 1) log θ + (α̃_b − 1) log(1 − θ)."""
        import math
        theta_s = min(max(float(theta), 1e-300), 1.0 - 1e-300)
        return float((self.n_success - 1.0) * math.log(theta_s)
                     + (self.n_failure - 1.0) * math.log(1.0 - theta_s))


# ---------------------------------------------------------------------------
# Grammar class
# ---------------------------------------------------------------------------

class GrammarClass(Enum):
    """Classification of grammar by Chomsky hierarchy level.

    LEFT_REGULAR grammars produce HMMs (Forward-Backward for inference).
    CONTEXT_FREE grammars produce SCFGs (Inside-Outside for inference).

    Detection: a grammar is CONTEXT_FREE if it has any bifurcating productions
    (X → Y·Z) or right-emitting terminals. Otherwise it is LEFT_REGULAR.
    """
    LEFT_REGULAR = 'left_regular'
    CONTEXT_FREE = 'context_free'


# ---------------------------------------------------------------------------
# Grammar spec (pre-compilation) and compiled model
# ---------------------------------------------------------------------------

@dataclass
class NullInfo:
    """Metadata about nullable states, used for null removal and restoration.

    Populated by compile() during nullability computation and null cycle removal.
    """
    nullabilities: Any = None       # η(X) for each nullable nonterminal
    null_state_indices: Any = None  # which states are null
    upsilon: Any = None             # exploded transition matrix (before null removal)
    closure: Any = None             # (I - T_ZZ)^{-1}
    T_NN: Any = None                # non-null to non-null transitions
    T_Nnull: Any = None             # non-null to null transitions
    T_nullN: Any = None             # null to non-null transitions
    null_contrib: Any = None        # T_Nnull @ closure @ T_nullN


@dataclass
class GrammarSpec:
    """Raw grammar produced by elaboration builders, before null removal."""
    params: dict
    grammar_class: GrammarClass
    build_rules: Callable  # params -> (rules/transitions, state_types, null_info)


class CompiledModel(Protocol):
    """Grammar after null removal — ready for DP."""
    params: dict
    grammar_class: GrammarClass
    null_info: NullInfo

    def build_trans(self, params: dict) -> tuple[jax.Array, jax.Array]:
        """Build null-free transition matrix and state types.

        Returns:
            trans: (N, N) transition/rule weight matrix (linear space)
            state_types: (N,) state type indicators (M/I/D/S/E)
        """
        ...

    def extract_stats(self, n_counts: jax.Array, trans: jax.Array,
                      params: dict) -> dict:
        """Extract sufficient statistics from DP expected counts.

        Includes null count restoration (inverse of null removal).

        Returns:
            Pytree of {BDISuffStats, SimplexSuffStats, BernoulliSuffStats}
            with same structure as params.
        """
        ...

    def m_step(self, suff_stats: dict, params: dict) -> dict:
        """Update parameters from sufficient statistics.

        Returns:
            New params pytree.
        """
        ...
