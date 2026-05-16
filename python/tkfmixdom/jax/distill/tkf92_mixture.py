"""K-component mixture-of-TKF92 fit on per-family cherry-count tensors.

Per Pfam-seed family n a single hidden component k_n ~ Cat(K) is shared
by all that family's cherries:

    P(MSA_n | k) = ∏_{cherries in MSA_n} P(cherry | θ_k)

where θ_k = (λ_k, μ_k, r_k, S_k, π_k) is a TKF92 parameter set. The
cherry sufficient stats live in `pfam/cherries_tkf92/PFXXXXX.npz` as
three τ-binned tensors (match (T,A,A), singlet (T,A), transition
(T,5,5) with Start/M/I/D/End indexing).

Outer EM over the family-level mixture:

  E-step: r_{n,k} ∝ π_mix[k] · P(MSA_n | θ_k), normalised across k.

  M-step:
    π_mix[k] = (1/N) Σ_n r_{n,k}
    aggregated counts per k: weighted sum of family count tensors.
    For each k, run inner TKF92 Baum–Welch on the aggregate.

Inner TKF92 BW (on aggregate τ-binned counts):
  - Split each (M→M, I→I, D→D) cherry count into "extension" and
    "new-fragment" parts using current r and the TKF91 base trans.
    The remaining off-diagonal cherry transitions are pure TKF91.
  - HR M-step (strategy #2 in tkf.tex 5.8): coordinate ascent
    on (S, π) via `m_step_subst_option1`.
  - BDI M-step: joint κ-quadratic via `m_step_indel_quadratic`.
  - r M-step: Bernoulli MLE from extension vs new-fragment counts.

Output: a MixDom1-loadable .npz checkpoint (each component is one
domain), so it can be lifted into a `train_pfam` `--n-dom K --n-frag 1`
warm-start.

Implementation notes:
  - One eigh per inner-EM iteration per component (R_k = S_k·diag(π_k)
    is fixed inside the inner E-step); both exp(R_k·τ_t) and the
    Holmes–Rubin integrals reuse the same eigendecomposition.
  - vmap over τ-bins for HR/exp(R·τ).
  - Outer LL evaluation is dominated by einsum contractions
    over (N families, K components, T bins, A alphabet); tractable on
    CPU at the chosen sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

from tkfmixdom.jax.core.bdi import bdi_stats_from_counts, m_step_indel_quadratic
from tkfmixdom.jax.core.ctmc import (
    holmes_rubin_integrals, holmes_rubin_weighted_stats,
    m_step_subst_option1, transition_matrix,
)
from tkfmixdom.jax.core.protein import rate_matrix_lg
# tkf92_mixture distillation builds emission tensors against a unit-rate
# Q convention: `t` is in expected-substitutions-per-site units and the
# (S, π) pair contributes equilibrium chemistry only. Distillation
# tensors downstream consumers expect this calibration, so the
# unit-normalisation is intentional. acknowledged_lossy=True silences
# the UserWarning otherwise raised on every call.
from functools import partial as _partial
from tkfmixdom.jax.core.ctmc import (
    build_rate_matrix_unit_normalized as _build_rate_matrix_unnorm)
build_rate_matrix = _partial(_build_rate_matrix_unnorm, acknowledged_lossy=True)
from tkfmixdom.jax.distill.maraschino import bdi_params, tkf91_trans

AA = 20
N_STATES = 5  # Start, M, I, D, End
S_IDX, M_IDX, I_IDX, D_IDX, E_IDX = 0, 1, 2, 3, 4
EPS = 1e-30


# ============================================================
# Cherry tensor I/O
# ============================================================
@dataclass
class CherryStack:
    """Stacked per-family cherry counts.

    All arrays have a leading family axis of size N. They are stored as
    np.float64 (not int) so they can flow through JAX without dtype
    conversion overhead.

    Attributes:
        match_counts:      (N, T, A, A)
        singlet_counts:    (N, T, A)         insert + delete combined
        transition_counts: (N, T, 5, 5)      Start/M/I/D/End indexing
        tau_centers:       (T,)
        tau_edges:         (T+1,)
        family_ids:        list[str] of length N
        n_pairs:           (N,) int counts per family
    """
    match_counts: jnp.ndarray
    singlet_counts: jnp.ndarray
    transition_counts: jnp.ndarray
    tau_centers: jnp.ndarray
    tau_edges: jnp.ndarray
    family_ids: list
    n_pairs: jnp.ndarray


def load_cherry_stack(npz_dir: str | Path,
                      families: Iterable[str] | None = None) -> CherryStack:
    """Load all (or a subset of) per-family cherry .npz files into one stack."""
    npz_dir = Path(npz_dir)
    if families is None:
        files = sorted(npz_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files in {npz_dir}")
    else:
        files = [npz_dir / f"{fam}.npz" for fam in families]
        files = [p for p in files if p.exists()]
        if not files:
            raise FileNotFoundError(f"None of the requested families found in {npz_dir}")

    match_list = []
    singlet_list = []
    trans_list = []
    family_ids = []
    n_pairs_list = []
    tau_centers = None
    tau_edges = None

    for p in files:
        d = np.load(p, allow_pickle=True)
        match_list.append(np.asarray(d["match_counts"], dtype=np.float64))
        singlet_list.append(np.asarray(d["singlet_counts"], dtype=np.float64))
        trans_list.append(np.asarray(d["transition_counts"], dtype=np.float64))
        family_ids.append(str(d["family"]))
        n_pairs_list.append(int(d["n_pairs"]))
        if tau_centers is None:
            tau_centers = np.asarray(d["tau_centers"], dtype=np.float64)
            tau_edges = np.asarray(d["tau_edges"], dtype=np.float64)

    return CherryStack(
        match_counts=jnp.asarray(np.stack(match_list)),
        singlet_counts=jnp.asarray(np.stack(singlet_list)),
        transition_counts=jnp.asarray(np.stack(trans_list)),
        tau_centers=jnp.asarray(tau_centers),
        tau_edges=jnp.asarray(tau_edges),
        family_ids=family_ids,
        n_pairs=jnp.asarray(np.array(n_pairs_list, dtype=np.int64)),
    )


# ============================================================
# TKF92 5×5 Pair-HMM transition matrix
# ============================================================
def tkf92_pair_trans(lam, mu, r, t):
    """5×5 TKF92 PairHMM transition matrix, rows = source state, cols = dest.

    State indexing: Start=0, M=1, I=2, D=3, End=4.

    Each of M, I, D self-extends with prob r (within-fragment), or with
    prob 1−r terminates the fragment and follows the underlying TKF91
    transition. Start does not extend.
    """
    alpha, beta, gamma_val, kappa = bdi_params(lam, mu, t)
    tau91 = tkf91_trans(alpha, beta, gamma_val, kappa)  # (5, 5)
    is_extendable = jnp.array([0.0, 1.0, 1.0, 1.0, 0.0])  # M, I, D
    eye = jnp.eye(5)
    # Each row x: (1 - r·is_ext[x])·tau91[x] + r·is_ext[x]·eye[x]
    return (1.0 - r * is_extendable)[:, None] * tau91 + (r * is_extendable)[:, None] * eye


# ============================================================
# Per-component bin-wise model tensors
# ============================================================
def _component_eig(S, pi):
    """One reversible eigh of R = S·diag(pi).

    Returns (eigvals, eigvecs, sqrt_pi) so that for any t the bridge
    transition matrix and HR integrals can be reconstructed without
    re-decomposing.
    """
    Q = build_rate_matrix(S, pi)  # zero-diagonal-corrected rate matrix
    sqrt_pi = jnp.sqrt(jnp.maximum(pi, EPS))
    # Symmetrise R via similarity by sqrt(pi)
    Rsym = Q * (sqrt_pi[:, None] / sqrt_pi[None, :])
    Rsym = (Rsym + Rsym.T) / 2.0
    eigvals, eigvecs = jnp.linalg.eigh(Rsym)
    return Q, eigvals, eigvecs, sqrt_pi


def _per_bin_log_emit(Q, pi, t):
    """log(pi[a] · P(t)[a,b]) and log(pi[a]) at one tau."""
    P = transition_matrix(Q, t)
    log_pi = jnp.log(jnp.maximum(pi, EPS))
    log_pi_P = log_pi[:, None] + jnp.log(jnp.maximum(P, EPS))  # (A, A)
    return log_pi_P, log_pi


def _component_log_tensors(theta, tau_centers):
    """For one component, compute the log-tensors used in the LL evaluation.

    Returns:
        log_match: (T, A, A)   log(π[a]·P(τ_t)[a,b])
        log_sing:  (A,)         log π   (time-independent)
        log_trans: (T, 5, 5)   log TKF92 PairHMM transitions
    """
    lam, mu, r, S, pi = theta
    Q = build_rate_matrix(S, pi)

    def per_bin(t):
        log_match_t, log_pi_t = _per_bin_log_emit(Q, pi, t)
        trans_t = tkf92_pair_trans(lam, mu, r, t)
        log_trans_t = jnp.log(jnp.maximum(trans_t, EPS))
        return log_match_t, log_trans_t

    log_match, log_trans = jax.vmap(per_bin)(tau_centers)  # (T, A, A), (T, 5, 5)
    log_sing = jnp.log(jnp.maximum(pi, EPS))               # (A,)
    return log_match, log_sing, log_trans


# ============================================================
# Outer E-step: family LL under each component
# ============================================================
def family_log_likelihoods(stack: CherryStack, thetas) -> jnp.ndarray:
    """Compute log P(MSA_n | θ_k) for every (n, k).

    Returns:
        log_lik: (N, K) jax array of family log-likelihoods.
    """
    K = len(thetas)

    # Pre-build per-component log tensors (vmappable but K is small)
    log_match_K = []
    log_sing_K = []
    log_trans_K = []
    for k in range(K):
        lm, ls, lt = _component_log_tensors(thetas[k], stack.tau_centers)
        log_match_K.append(lm)
        log_sing_K.append(ls)
        log_trans_K.append(lt)
    log_match_K = jnp.stack(log_match_K)   # (K, T, A, A)
    log_sing_K = jnp.stack(log_sing_K)     # (K, A)
    log_trans_K = jnp.stack(log_trans_K)   # (K, T, 5, 5)

    # match LL: sum_{t, a, b} match_counts[n, t, a, b] · log_match[k, t, a, b]
    ll_match = jnp.einsum("ntab,ktab->nk", stack.match_counts, log_match_K)
    # singlet LL: sum_{t, a} singlet_counts[n, t, a] · log_sing[k, a]
    sing_total = stack.singlet_counts.sum(axis=1)  # (N, A)
    ll_sing = jnp.einsum("na,ka->nk", sing_total, log_sing_K)
    # trans LL: sum_{t, u, v} transition_counts[n, t, u, v] · log_trans[k, t, u, v]
    ll_trans = jnp.einsum("ntuv,ktuv->nk", stack.transition_counts, log_trans_K)

    return ll_match + ll_sing + ll_trans


def responsibilities(log_lik, log_mix):
    """E-step: r_{n,k} = softmax_k(log_mix[k] + log_lik[n, k]).

    Args:
        log_lik: (N, K) family log-likelihoods.
        log_mix: (K,)   log mixing weights.

    Returns:
        log_resp: (N, K)  log responsibilities (numerically-safe softmax)
        log_evidence: (N,)  log P(MSA_n) = logsumexp_k of (log_mix + log_lik)
    """
    log_joint = log_lik + log_mix[None, :]
    log_evidence = jax.nn.logsumexp(log_joint, axis=1)
    log_resp = log_joint - log_evidence[:, None]
    return log_resp, log_evidence


# ============================================================
# Outer M-step: aggregate counts per component
# ============================================================
def aggregate_counts(stack: CherryStack, log_resp):
    """Sum stack counts across families weighted by exp(log_resp[n, k]).

    Returns three arrays of shape (K, T, ...).
    """
    resp = jnp.exp(log_resp)  # (N, K) — fine to materialise; small grid
    match_K = jnp.einsum("nk,ntab->ktab", resp, stack.match_counts)
    sing_K = jnp.einsum("nk,nta->kta", resp, stack.singlet_counts)
    trans_K = jnp.einsum("nk,ntuv->ktuv", resp, stack.transition_counts)
    return match_K, sing_K, trans_K


# ============================================================
# Inner Baum–Welch: TKF92 fit on one (T, A, A)+(T, A)+(T, 5, 5) aggregate
# ============================================================
def _split_extension_counts(trans_counts_T, lam, mu, r, tau_centers):
    """Decompose τ-binned cherry transition counts into (n_91, n_ext).

    For each (X→X) self-loop with X∈{M, I, D} the cherry observation
    contains the BOTH the extension (geometric within-fragment) and the
    "new fragment after a TKF91 X→X step" contributions. Their relative
    weights, given the current model, are

        ext_resp[X→X]   = r / (r + (1 - r)·tau91[X→X])
        new91_resp[X→X] = 1 - ext_resp[X→X]

    Off-diagonal cherry transitions are pure new-fragment events.

    Args:
        trans_counts_T: (T, 5, 5) cherry transition counts.
        lam, mu, r:     scalar TKF92 params.
        tau_centers:    (T,) bin centres.

    Returns:
        n_91:        (T, 5, 5) τ-binned underlying-TKF91 transition counts.
        ext_total:   total expected extension events Σ_t Σ_X ext_resp·count[t,X,X]
        term_total:  total expected fragment-termination events
                     (Σ over t and over off-diagonal entries from M/I/D rows
                      plus diagonal new-fragment portion).
    """
    is_extendable = jnp.array([0.0, 1.0, 1.0, 1.0, 0.0])  # M=1, I=1, D=1

    def per_bin(t, c_t):
        alpha, beta, gamma_val, kappa = bdi_params(lam, mu, t)
        tau91 = tkf91_trans(alpha, beta, gamma_val, kappa)  # (5, 5)
        # ext_resp on the diagonal entries of M, I, D rows
        diag_tau91 = jnp.diag(tau91)                              # (5,)
        ext_denom = r + (1.0 - r) * diag_tau91                     # (5,)
        ext_resp = jnp.where(
            is_extendable > 0.5,
            r / jnp.maximum(ext_denom, EPS),
            jnp.zeros_like(diag_tau91))                            # (5,)
        # Diagonal new-fragment counts: c_t[X,X] · (1 - ext_resp[X])
        diag_new = jnp.diag(c_t) * (1.0 - ext_resp)
        # Diagonal extension counts
        diag_ext = jnp.diag(c_t) * ext_resp
        # n_91[t]: same as cherry except diagonals replaced by diag_new
        n_91_t = c_t.at[jnp.arange(5), jnp.arange(5)].set(diag_new)
        # Termination count contribution from this bin = ext-extendable rows
        # going to anything other than self (off-diag) + new-fragment portion
        # of diagonal (already in n_91).
        # i.e. total terminations = Σ_X∈{M,I,D} Σ_Y c_t[X,Y]  −  Σ_X diag_ext[X]
        ext_rows_total = jnp.sum(c_t * is_extendable[:, None])
        bin_term = ext_rows_total - jnp.sum(diag_ext)
        return n_91_t, jnp.sum(diag_ext), bin_term

    n_91, ext_per_bin, term_per_bin = jax.vmap(per_bin)(tau_centers, trans_counts_T)
    return n_91, jnp.sum(ext_per_bin), jnp.sum(term_per_bin)


def _aggregate_bdi_stats(n_91, lam, mu, tau_centers):
    """Sum bdi_stats_from_counts across τ-bins under current (λ, μ).

    Returns aggregated (E_B, E_D, E_S, L, M, T_total).
    """
    # bdi_stats_from_counts is implemented in plain numpy/JAX with a
    # control-flow branch for λ≈μ; simplest is to loop over bins on host.
    n_91_np = np.asarray(n_91)
    tau_np = np.asarray(tau_centers)
    EB = ED = ES = 0.0
    L_total = 0.0
    M_total = 0.0
    T_total = 0.0
    for t_idx, t in enumerate(tau_np):
        n_t = n_91_np[t_idx]
        n_term_t = float(n_t[:, E_IDX].sum())  # # BDI processes (cherries) in this bin
        if n_t.sum() < 1e-12 or n_term_t < 1e-12:
            continue
        # Each cherry is one BDI process at time t. The bin's accumulated
        # count matrix `n_t` came from `n_term_t` independent processes,
        # so the total observation time for this bin is `t × n_term_t`.
        T_bin = float(t) * n_term_t
        eb, ed, es = bdi_stats_from_counts(jnp.asarray(n_t), float(lam), float(mu),
                                           float(t), T=T_bin)
        EB += float(eb)
        ED += float(ed)
        ES += float(es)
        # L = number of κ-arcs = Σ entries into M or D
        L_total += float(n_t[:, M_IDX].sum() + n_t[:, D_IDX].sum())
        # M = number of (1-κ)-arcs = Σ entries into End
        M_total += n_term_t
        T_total += T_bin
    return EB, ED, ES, L_total, M_total, T_total


def _aggregate_hr_stats(match_T, singlet_T, Q, pi, tau_centers):
    """Aggregate Holmes–Rubin W, U, V across τ-bins under current (Q, π)."""

    def per_bin(t, mc_t):
        return holmes_rubin_weighted_stats(Q, pi, t, mc_t)

    W_T, U_T = jax.vmap(per_bin)(tau_centers, match_T)  # (T, A), (T, A, A)
    W = jnp.sum(W_T, axis=0)
    U = jnp.sum(U_T, axis=0)
    # V: ancestor character composition under joint pair HMM.
    # Match-position ancestors: match_T.sum(axis=2).sum(axis=0) over t.
    # Singlet emissions (combined insert + delete): singlet_T.sum(axis=0).
    V = match_T.sum(axis=(0, 2)) + singlet_T.sum(axis=0)
    return W, U, V


def inner_em(match_counts, singlet_counts, transition_counts, tau_centers,
             theta_init, *, n_iter_max=30, rel_tol=1e-4,
             pi_lg=None, S_lg=None, pi_pseudo=1.0, S_pseudo=0.0,
             prior_alpha_lam=2.0, prior_alpha_mu=2.0, prior_beta=10.0,
             ext_prior_alpha=2.0, ext_prior_beta=3.0,
             log_fn=None):
    """Inner TKF92 BW on aggregate τ-binned counts.

    Returns (theta_new, history).
    history is a list of per-iter LL values (computed on the SAME
    aggregate counts, useful for checking convergence).
    """
    lam, mu, r, S, pi = theta_init
    history = []

    def _log_likelihood(theta):
        lam_, mu_, r_, S_, pi_ = theta
        Q_ = build_rate_matrix(S_, pi_)

        def per_bin(t, mc_t, sc_t, ct_t):
            P = transition_matrix(Q_, t)
            log_match = jnp.log(jnp.maximum(pi_[:, None] * P, EPS))
            log_sing = jnp.log(jnp.maximum(pi_, EPS))
            trans = tkf92_pair_trans(lam_, mu_, r_, t)
            log_trans = jnp.log(jnp.maximum(trans, EPS))
            return (jnp.sum(mc_t * log_match)
                    + jnp.sum(sc_t * log_sing)
                    + jnp.sum(ct_t * log_trans))

        per_t_ll = jax.vmap(per_bin)(tau_centers, match_counts, singlet_counts,
                                     transition_counts)
        return float(jnp.sum(per_t_ll))

    prev_ll = _log_likelihood((lam, mu, r, S, pi))
    history.append(prev_ll)
    if log_fn is not None:
        log_fn(f"  inner BW iter 0: LL={prev_ll:.4f}")

    for it in range(1, n_iter_max + 1):
        # ---- E-step ----
        n_91, ext_total, term_total = _split_extension_counts(
            transition_counts, lam, mu, r, tau_centers)

        # ---- M-step (r) ----
        ext_total_f = float(ext_total) + (ext_prior_alpha - 1.0)
        term_total_f = float(term_total) + (ext_prior_beta - 1.0)
        denom_r = max(ext_total_f + term_total_f, EPS)
        r_new = max(min(ext_total_f / denom_r, 1.0 - 1e-6), 1e-6)

        # ---- M-step (λ, μ) via BDI quadratic ----
        EB, ED, ES, L, M, T_total = _aggregate_bdi_stats(
            n_91, lam, mu, tau_centers)
        if EB + ED + L + M < 1e-9:
            lam_new, mu_new = lam, mu
        else:
            lam_new, mu_new = m_step_indel_quadratic(
                EB, ED, ES, L, M, T_total,
                prior_alpha_lam=prior_alpha_lam,
                prior_alpha_mu=prior_alpha_mu,
                prior_beta=prior_beta)

        # ---- M-step (S, π) via HR strategy #2 (m_step_subst_option1) ----
        Q = build_rate_matrix(S, pi)
        W, U, V = _aggregate_hr_stats(match_counts, singlet_counts, Q, pi,
                                      tau_centers)
        S_new, pi_new, _ = m_step_subst_option1(
            np.asarray(W), np.asarray(U), np.asarray(V),
            S_prior=np.asarray(S_lg) if S_lg is not None else None,
            pi_prior=np.asarray(pi_lg) if pi_lg is not None else None,
            pi_pseudo=pi_pseudo, S_pseudo=S_pseudo,
            n_iter=50, tol=1e-10)

        lam, mu, r = float(lam_new), float(mu_new), float(r_new)
        S = jnp.asarray(S_new)
        pi = jnp.asarray(pi_new)

        ll = _log_likelihood((lam, mu, r, S, pi))
        history.append(ll)
        if log_fn is not None:
            log_fn(f"  inner BW iter {it}: LL={ll:.4f} "
                   f"(λ={lam:.4g}, μ={mu:.4g}, r={r:.4g})")

        if abs(ll - prev_ll) <= rel_tol * (abs(prev_ll) + 1.0):
            break
        prev_ll = ll

    return (lam, mu, r, S, pi), history


# ============================================================
# Outer EM driver
# ============================================================
def init_components(K, *, seed=0, perturb=0.05, base_lam=0.02, base_mu=0.025,
                    base_r=0.5):
    """Initialise K TKF92 components: uniform with small symmetry-breaking.

    All components share the LG (S, π) baseline. Each component's TKF
    rates are the base rate × (1 ± uniform[−perturb, +perturb]) so they
    are not identical (the EM is otherwise stuck on the K=1 saddle).
    """
    Q_lg, pi_lg = rate_matrix_lg()
    pi_lg_np = np.asarray(pi_lg)
    Q_lg_np = np.asarray(Q_lg)
    S_lg = Q_lg_np / np.maximum(pi_lg_np[None, :], EPS) * (1.0 - np.eye(AA))
    S_lg = (S_lg + S_lg.T) / 2.0

    rng = np.random.RandomState(seed)
    thetas = []
    for k in range(K):
        # Small per-component perturbation of the LG π (Dirichlet noise)
        # so the components have distinct emissions at iter 0.
        pi_k = pi_lg_np * (1.0 + perturb * rng.randn(AA))
        pi_k = np.maximum(pi_k, 1e-6)
        pi_k = pi_k / pi_k.sum()
        # Perturb the rates (multiplicative)
        lam_k = base_lam * (1.0 + perturb * rng.randn())
        mu_k = base_mu * (1.0 + perturb * rng.randn())
        # Ensure μ > λ
        mu_k = max(mu_k, lam_k + 1e-4)
        r_k = base_r + perturb * rng.randn()
        r_k = max(min(r_k, 0.95), 0.05)
        thetas.append((float(lam_k), float(mu_k), float(r_k),
                       jnp.asarray(S_lg.copy()), jnp.asarray(pi_k)))
    log_mix = jnp.log(jnp.ones(K) / K)
    return thetas, log_mix


def fit_mixture(stack: CherryStack, K: int, *,
                seed: int = 0,
                outer_n_iter_max: int = 50,
                outer_rel_tol: float = 1e-5,
                inner_n_iter_max: int = 30,
                inner_rel_tol: float = 1e-4,
                perturb: float = 0.05,
                base_lam: float = 0.02,
                base_mu: float = 0.025,
                base_r: float = 0.5,
                pi_pseudo: float = 1.0,
                S_pseudo: float = 0.0,
                ext_prior_alpha: float = 2.0,
                ext_prior_beta: float = 3.0,
                prior_alpha_lam: float = 2.0,
                prior_alpha_mu: float = 2.0,
                prior_beta: float = 10.0,
                log_fn=print):
    """Fit K-component mixture-of-TKF92 to a CherryStack.

    Returns:
        thetas: list[K] of TKF92 parameter tuples (λ, μ, r, S, π).
        log_mix: (K,) log mixing weights.
        history: list of dicts {iter, total_ll, mix} per outer step.
    """
    Q_lg, pi_lg = rate_matrix_lg()
    pi_lg_np = np.asarray(pi_lg)
    Q_lg_np = np.asarray(Q_lg)
    S_lg = Q_lg_np / np.maximum(pi_lg_np[None, :], EPS) * (1.0 - np.eye(AA))
    S_lg = (S_lg + S_lg.T) / 2.0

    thetas, log_mix = init_components(
        K, seed=seed, perturb=perturb,
        base_lam=base_lam, base_mu=base_mu, base_r=base_r)

    history = []
    prev_total_ll = -float("inf")

    # First outer iter uses Dirichlet-sampled responsibilities (uniform
    # mixing weights and tiny perturbation make all components nearly
    # identical at init; per-family LL differences over thousands of
    # cherries then collapse the softmax onto one component, so we
    # bypass the model on iter 1 to seed component divergence). After
    # iter 1 the components have meaningfully different θ from training
    # on different soft-partitions of the data.
    rng = np.random.RandomState(seed + 1)
    N = len(stack.family_ids)
    init_resp_np = rng.dirichlet(np.ones(K), size=N)  # (N, K), each row sums to 1
    init_log_resp = jnp.log(jnp.maximum(jnp.asarray(init_resp_np), EPS))

    for outer in range(1, outer_n_iter_max + 1):
        # ---- Outer E-step ----
        if outer == 1:
            log_resp = init_log_resp
            log_evidence = jnp.zeros(N)  # placeholder; total_ll defined below
            log_lik = jnp.zeros((N, K))  # not used at iter 1
        else:
            log_lik = family_log_likelihoods(stack, thetas)    # (N, K)
            log_resp, log_evidence = responsibilities(log_lik, log_mix)
        total_ll = float(jnp.sum(log_evidence))

        # ---- Outer M-step ----
        # mix
        resp = jnp.exp(log_resp)
        new_mix = resp.sum(axis=0)
        new_mix = new_mix / jnp.maximum(new_mix.sum(), EPS)
        log_mix = jnp.log(jnp.maximum(new_mix, EPS))

        # aggregated counts
        match_K, sing_K, trans_K = aggregate_counts(stack, log_resp)

        # per-component inner BW
        new_thetas = []
        for k in range(K):
            theta_k, hist = inner_em(
                match_K[k], sing_K[k], trans_K[k], stack.tau_centers,
                thetas[k],
                n_iter_max=inner_n_iter_max, rel_tol=inner_rel_tol,
                pi_lg=pi_lg_np, S_lg=S_lg, pi_pseudo=pi_pseudo, S_pseudo=S_pseudo,
                prior_alpha_lam=prior_alpha_lam, prior_alpha_mu=prior_alpha_mu,
                prior_beta=prior_beta,
                ext_prior_alpha=ext_prior_alpha, ext_prior_beta=ext_prior_beta,
                log_fn=None)
            new_thetas.append(theta_k)
        thetas = new_thetas

        if outer == 1:
            log_fn(f"[outer EM iter 1/{outer_n_iter_max}] "
                   f"(seeded from Dirichlet random responsibilities) "
                   f"mix={[f'{w:.3f}' for w in np.asarray(new_mix)]}")
            history.append({"iter": outer, "total_ll": None,
                            "mix": np.asarray(new_mix).tolist()})
            # No LL on iter 1 (we used random resps, not the model);
            # convergence check defers to iter 2.
            continue

        log_fn(f"[outer EM iter {outer}/{outer_n_iter_max}] "
               f"total_ll={total_ll:.4f}  mix={[f'{w:.3f}' for w in np.asarray(new_mix)]}")

        history.append({
            "iter": outer, "total_ll": total_ll,
            "mix": np.asarray(new_mix).tolist()
        })

        if not np.isinf(prev_total_ll) and \
                abs(total_ll - prev_total_ll) <= outer_rel_tol * (abs(prev_total_ll) + 1.0):
            log_fn(f"  outer EM converged at iter {outer}")
            break
        prev_total_ll = total_ll

    return thetas, log_mix, history


# ============================================================
# Output: write a MixDom1-loadable .npz checkpoint
# ============================================================
def to_mixdom1_checkpoint(thetas, log_mix, *,
                          main_ins: float = 0.014,
                          main_del: float = 0.015,
                          n_frag: int = 1,
                          em_iter: int = 0,
                          config: dict | None = None) -> dict:
    """Map the K-component fit into a train_pfam-loadable param dict.

    Each component k → one MixDom domain, with `dom_ins[k]=λ_k`,
    `dom_del[k]=μ_k`, `ext_rates[k, 0, 0]=r_k`,
    `dom_pis[k]=π_k`, `dom_S_exch[k]=S_k`. Top-level rates are taken
    from `main_ins`/`main_del` (default ~ d3f1's). frag_weights[k, :]
    is uniform.
    """
    K = len(thetas)
    A = AA
    F = n_frag
    dom_ins = np.zeros(K, dtype=np.float32)
    dom_del = np.zeros(K, dtype=np.float32)
    dom_pis = np.zeros((K, A), dtype=np.float32)
    dom_S_exch = np.zeros((K, A, A), dtype=np.float32)
    ext_rates = np.zeros((K, F, F), dtype=np.float32)
    for k, (lam, mu, r, S, pi) in enumerate(thetas):
        dom_ins[k] = float(lam)
        dom_del[k] = float(mu)
        dom_pis[k] = np.asarray(pi).astype(np.float32)
        dom_S_exch[k] = np.asarray(S).astype(np.float32)
        ext_rates[k, 0, 0] = float(r)  # F=1 so a single self-extension prob
    dom_weights = np.exp(np.asarray(log_mix)).astype(np.float32)
    dom_weights = dom_weights / dom_weights.sum()
    frag_weights = np.ones((K, F), dtype=np.float32)
    # dom_Qs: build from S, pi
    dom_Qs = np.zeros((K, A, A), dtype=np.float32)
    for k in range(K):
        Q = np.asarray(build_rate_matrix(jnp.asarray(dom_S_exch[k]),
                                          jnp.asarray(dom_pis[k])))
        dom_Qs[k] = Q.astype(np.float32)

    out: dict = {
        "main_ins": np.float32(main_ins),
        "main_del": np.float32(main_del),
        "dom_ins": dom_ins,
        "dom_del": dom_del,
        "dom_weights": dom_weights,
        "frag_weights": frag_weights,
        "ext_rates": ext_rates,
        "dom_Qs": dom_Qs,
        "dom_pis": dom_pis,
        "dom_S_exch": dom_S_exch,
        "em_iter": np.int32(em_iter),
    }
    if config is not None:
        import json
        out["_config"] = np.asarray(json.dumps(config))
    return out
