"""Algebraic distillation of MixDom model to order-1 WFST and singlet HMM.

WARNING: Distilled models are for INFERENCE ONLY (tree composition, beam
search, progressive alignment). Do NOT use distilled WFSTs for simulation.
For controlled simulation experiments, use the direct generative simulator
in ``simulate/mixdom_sim.py`` which samples from the exact nested model
with fully labeled domain/fragment/class annotations.

Uses Woodbury identity for efficient computation: O(N·AA² + 3³) instead of
O((5N)³). See algebraic-distillation.tex for mathematical derivation.

For MixDom2 (F>1 fragments with Markovian F×F transition matrix), the
within-domain blocks are 3F×3F (match) and F×F (insert/delete) instead of
3×3 and scalar. The distillation is ALGEBRAICALLY EXACT for any F — no
scalar effective extension rate approximation is needed. The Woodbury
identity still operates with a 3×3 kernel (top-level M/I/D coupling),
giving O(N·(5F)² + 3³) complexity per domain.

Key functions:
  - load_params: Load raw MixDom parameters from .npz file
  - constrain_params: Convert unconstrained → constrained parameters
  - distill_mixdom: Compute order-1 adjacency frequencies via Woodbury
  - normalize_freqs: Backward-compatible alias for normalize_freqs_hmm
  - normalize_freqs_hmm: Convert frequencies to HMM transition probabilities
  - normalize_freqs_wfst: Convert frequencies to WFST conditional probabilities
  - effective_pair_hmm: Compute emission-marginalized transition matrix
    and domain-mixture emission model for use in profile DP
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

from ..dp.hmm import safe_log


# ============================================================
# Constants
# ============================================================
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA = len(AMINO_ACIDS)  # 20

# Emitting state types in nested Pair HMM (per domain)
# MM=0, MI=1, MD=2, II=3, DD=4
EST_MM, EST_MI, EST_MD, EST_II, EST_DD = range(5)
N_EST = 5

# Top-level Pair HMM states: S=0, M=1, I=2, D=3, E=4
TL_S, TL_M, TL_I, TL_D, TL_E = range(5)

# Mapping: emitting state type -> top-level type
EST_TO_TL = jnp.array([TL_M, TL_M, TL_M, TL_I, TL_D])


# ============================================================
# LG08 substitution model (PAML order, converted to alphabetical)
# ============================================================
_PAML_ORDER = "ARNDCQEGHILKMFPSTWYV"
_PAML_TO_ALPHA = np.array([_PAML_ORDER.index(a) for a in AMINO_ACIDS])

_LG_S_LOWER = np.array([
    0.425093,
    0.276818, 0.751878,
    0.395144, 0.123954, 5.076149,
    2.489084, 0.534551, 0.528768, 0.062556,
    0.969894, 2.807908, 1.038545, 0.363970, 0.746078,
    1.038545, 0.363970, 0.746078, 5.243870, 0.084329, 5.115644,
    2.066040, 0.390894, 1.437645, 0.554236, 0.075382, 0.594093, 2.547870,
    0.358858, 2.137150, 3.038533, 0.312261, 0.006334, 1.506500, 0.528768, 0.306475,
    0.149830, 0.109261, 0.528768, 0.042610, 0.308635, 0.126991, 0.001800, 0.021543, 0.236199,
    0.395144, 0.528768, 0.100872, 0.006613, 0.320627, 0.350230, 0.058654, 0.018625, 0.468199, 3.088510,
    0.906265, 5.351420, 3.148580, 0.569265, 0.072854, 2.006569, 1.137630, 0.336355, 0.122346, 0.068674, 0.277724,
    0.893496, 0.691268, 0.245034, 0.006613, 0.691268, 0.811614, 0.095382, 0.066236, 0.304803, 3.277830, 4.257460, 0.285078,
    0.210494, 0.145482, 0.065314, 0.003218, 0.897871, 0.089525, 0.006613, 0.062556, 0.645560, 0.829175, 2.106910, 0.046730, 1.190630,
    1.438550, 0.368739, 0.164126, 0.410886, 0.393379, 0.666506, 0.367902, 0.233397, 0.483768, 0.050644, 0.312261, 0.205711, 0.050644, 0.035454,
    4.509480, 0.887753, 3.681060, 1.169970, 2.137150, 1.003450, 0.544060, 1.595430, 0.611973, 0.131528, 0.267828, 0.665585, 0.247847, 0.364434, 1.341820,
    2.000540, 0.530324, 2.000540, 0.679371, 0.739772, 0.402941, 0.252167, 0.336355, 0.428437, 1.059470, 0.196258, 0.604070, 0.515706, 0.090855, 0.564432, 4.378020,
    0.113855, 0.869489, 0.049906, 0.006613, 0.911370, 0.247103, 0.006613, 0.167042, 0.540027, 0.157001, 0.868166, 0.035454, 0.506734, 1.289460, 0.049906, 0.306905, 0.152335,
    0.195510, 0.124630, 0.324525, 0.109261, 0.649361, 0.244157, 0.028906, 0.044265, 4.813505, 0.208836, 0.332517, 0.076701, 0.320627, 6.312580, 0.148483, 0.456190, 0.171995, 2.370130,
    2.386260, 0.186979, 0.062556, 0.068674, 1.173890, 0.117132, 0.174845, 0.188182, 0.222455, 7.821300, 1.129560, 0.137505, 2.020060, 0.569265, 0.249060, 0.582457, 2.370130, 0.268491, 0.257336,
])

_LG_PI = np.array([
    0.079066, 0.055941, 0.041977, 0.053052, 0.012937,
    0.040767, 0.071586, 0.057337, 0.022355, 0.062157,
    0.099081, 0.064600, 0.022951, 0.042302, 0.044040,
    0.061197, 0.053287, 0.012066, 0.034155, 0.069147,
])


def _lower_tri_to_sym(vals, n=20):
    S = np.zeros((n, n))
    idx = 0
    for i in range(1, n):
        for j in range(i):
            S[i, j] = vals[idx]
            S[j, i] = vals[idx]
            idx += 1
    return S


def get_lg08():
    """Return LG08 exchangeability matrix S and equilibrium pi in alphabetical order."""
    S_paml = _lower_tri_to_sym(_LG_S_LOWER)
    pi_paml = _LG_PI / _LG_PI.sum()
    S_alpha = S_paml[_PAML_TO_ALPHA][:, _PAML_TO_ALPHA]
    pi_alpha = pi_paml[_PAML_TO_ALPHA]
    return jnp.array(S_alpha), jnp.array(pi_alpha)


# ============================================================
# Substitution model functions
# ============================================================
# `build_rate_matrix` moved to `tkfmixdom.jax.core.ctmc` and renamed to
# `build_rate_matrix_unit_normalized`. The new name advertises that it
# divides Q by the equilibrium mean rate — a calibration step that is
# WRONG for any rate matrix learned from data. Building rate matrices is
# not a Maraschino-specific concern; it lives in the foundational CTMC
# module.
#
# Existing external callers can still import `build_rate_matrix` from
# here as a shim. The shim deliberately does NOT pass
# `acknowledged_lossy=True`, so each external call site fires the
# UserWarning that the unit-normalisation policy attaches to its
# default form — that is how new (or older mis-routed) trained-Q
# callers get caught at first run.
#
# Internal Maraschino distillation calls below DO want the
# unit-normalisation: gamma rate classes (`rate_mults` / gamma_rates)
# absorb the absolute rate scale, exactly as Annabel's
# `rate_multipliers` do. To avoid spamming the warning on every
# distill_mixdom call from inside this module, a separate
# `_build_rate_matrix_acknowledged` partial is used at all in-module
# call sites.
#
# Migration target for external callers:
#   - `tkfmixdom.jax.core.ctmc.build_Q_from_S_pi` (no normalization;
#     correct for trained / per-domain / per-class rate matrices)
#   - `tkfmixdom.jax.core.ctmc.build_rate_matrix_unit_normalized`
#     (rate-normalized; ONLY correct for fixed published matrices like
#     LG08, WAG, JC69 calibrated to mean rate 1, or for callers that
#     deliberately separate rate scale into a multiplier)
#
# See the `build_rate_matrix_unit_normalized` docstring for the failure
# mode this rename was meant to surface.
from functools import partial as _partial
from tkfmixdom.jax.core.ctmc import (
    build_rate_matrix_unit_normalized,
)
# Public shim alias (intentionally unsilenced). External callers that
# import `build_rate_matrix` from here get the UserWarning on every call
# unless they explicitly pass `acknowledged_lossy=True`. This is the
# safety net for trained-Q misuses.
build_rate_matrix = build_rate_matrix_unit_normalized
# Internal partial used at all in-module call sites: gamma rate
# multipliers absorb the absolute rate scale, so unit normalisation is
# intentional and the warning is suppressed.
_build_rate_matrix_acknowledged = _partial(
    build_rate_matrix_unit_normalized, acknowledged_lossy=True)


def eigen_decompose(Q, pi):
    """Precompute eigendecomposition for transition probability computation."""
    sqrt_pi = jnp.sqrt(jnp.maximum(pi, 1e-30))
    Sym = Q * (sqrt_pi[:, None] / sqrt_pi[None, :])
    Sym = (Sym + Sym.T) / 2
    eigvals, eigvecs = jnp.linalg.eigh(Sym)
    return eigvals, eigvecs, sqrt_pi


def transition_probs_from_eigen(eigvals, eigvecs, sqrt_pi, t):
    """Compute P(t) from precomputed eigendecomposition."""
    P = (eigvecs * jnp.exp(eigvals * t)[None, :]) @ eigvecs.T
    P = P * (sqrt_pi[None, :] / sqrt_pi[:, None])
    return jnp.maximum(P, 0.0)


def transition_probs(Q, pi, t):
    """Compute P(t) = exp(Q*t) via eigendecomposition of symmetrized matrix."""
    eigvals, eigvecs, sqrt_pi = eigen_decompose(Q, pi)
    return transition_probs_from_eigen(eigvals, eigvecs, sqrt_pi, t)


# ============================================================
# Discretized gamma rate classes (Yang 1994)
# ============================================================
def gamma_rates(alpha_shape, n_classes):
    """Discretized gamma rate multipliers (mean 1) using quantile midpoints."""
    rates = _gamma_quantile_midpoints(alpha_shape, n_classes)
    return rates / jnp.mean(rates)


def _gamma_quantile_midpoints(alpha, n_classes):
    """Approximate midpoint rates for discretized gamma."""
    mid_probs = (jnp.arange(n_classes) + 0.5) / n_classes
    from jax.scipy.stats import norm
    z = norm.ppf(mid_probs)
    factor = 1.0 - 1.0 / (9.0 * alpha) + z * jnp.sqrt(1.0 / (9.0 * alpha))
    rates = alpha * jnp.maximum(factor, 0.01) ** 3
    return rates


# ============================================================
# BDI (Birth-Death-Immigration) parameters
# ============================================================
def bdi_params(lam, mu, t):
    """Compute (alpha, beta, gamma, kappa) from lambda, mu, t."""
    kappa = lam / mu
    alpha = jnp.exp(-mu * t)
    e_lam = jnp.exp(-lam * t)
    e_mu = jnp.exp(-mu * t)
    denom = mu * e_lam - lam * e_mu
    beta = jnp.where(
        jnp.abs(lam - mu) < 1e-10 * mu,
        mu * t / (1.0 + mu * t),
        lam * (e_lam - e_mu) / denom
    )
    beta = jnp.clip(beta, 1e-15, 1.0 - 1e-15)
    gamma_val = jnp.where(
        jnp.abs(1.0 - alpha) < 1e-15,
        beta,
        1.0 - mu * beta / (lam * (1.0 - alpha))
    )
    gamma_val = jnp.clip(gamma_val, 1e-15, 1.0 - 1e-15)
    return alpha, beta, gamma_val, kappa


# ============================================================
# TKF91 5x5 transition matrix: rows [S, M, I, D, E]
# ============================================================
def tkf91_trans(alpha, beta, gamma_val, kappa):
    """5x5 TKF91 Pair HMM transition matrix (takes BDI params directly)."""
    ob = 1.0 - beta
    ok = 1.0 - kappa
    og = 1.0 - gamma_val

    row_smi = jnp.array([0.0, ob*kappa*alpha, beta, ob*kappa*(1-alpha), ob*ok])
    row_d   = jnp.array([0.0, og*kappa*alpha, gamma_val, og*kappa*(1-alpha), og*ok])
    row_e   = jnp.zeros(5)
    return jnp.stack([row_smi, row_smi, row_smi, row_d, row_e])


# ============================================================
# Fragment-averaged effective extension rate
# ============================================================
    # NOTE: effective_ext_rate and effective_ext_rate_markov were removed.
    # These functions collapsed the F×F fragment extension matrix to a
    # scalar — a compromise that loses fragment state information.
    # The D×F partition reconstruction (partition_recon.py) now tracks
    # fragment state explicitly and does not need scalar approximations.
    # The 3F×3F Woodbury distillation (distill_mixdom) also handles
    # the full F×F matrix directly.


# ============================================================
# MixDom parameter handling
# ============================================================
def constrain_params(raw):
    """Convert raw params to constrained (positive rates, valid probs, etc.).

    Handles both nFrag=1 (logit_r shape (N,)) and nFrag>1 (logit_r shape (N,F)).
    When nFrag>1, also expects 'logit_frag_weights' of shape (N, F).
    When nSites>1, expects 'log_pi' of shape (N, C, AA) and
    'logit_site_weights' of shape (N, C).

    MixDom2 per-frag class structure (Annabel-style): when ``raw`` contains
    ``log_class_pi`` (C, AA), ``log_class_S`` (C, AA, AA), and
    ``log_class_dist`` (D, F, C), these are passed through as constrained
    ``class_pi`` / ``class_S_exch`` / ``class_dist`` (linear, normalised)
    along with ``n_classes_per_frag``. Downstream consumers (partition
    recon, composite beam) use these to build per-class P(t) tables and
    fragment-dependent emission mixtures rather than collapsing across
    classes.
    """
    softplus = lambda x: jnp.log1p(jnp.exp(x))
    sigmoid = lambda x: jax.nn.sigmoid(x)
    softmax = lambda x: jax.nn.softmax(x, axis=-1)

    lam0 = softplus(raw['log_lam0']) + 1e-6
    mu0  = softplus(raw['log_mu0']) + 1e-6
    mu0 = jnp.maximum(mu0, lam0 + 1e-4)

    lam = softplus(raw['log_lam']) + 1e-6
    mu  = softplus(raw['log_mu']) + 1e-6
    mu  = jnp.maximum(mu, lam + 1e-4)

    r_raw = sigmoid(raw['logit_r'])
    v = softmax(raw['log_v'])

    log_S = raw['log_S']  # (N, AA, AA) or (AA, AA)
    S_raw = jnp.exp(log_S)
    if S_raw.ndim == 3:
        def _sym(S):
            S_s = (S + S.T) / 2
            return S_s - jnp.diag(jnp.diag(S_s))
        S_exch = jax.vmap(_sym)(S_raw)
    else:
        S_exch = (S_raw + S_raw.T) / 2
        S_exch = S_exch - jnp.diag(jnp.diag(S_exch))

    alpha_gamma = softplus(raw['log_alpha_gamma']) + 0.01

    result = {
        'lam0': lam0, 'mu0': mu0,
        'lam': lam, 'mu': mu,
        'v': v,
        'S_exch': S_exch,
        'alpha_gamma': alpha_gamma,
    }

    # Fragment handling: r_raw is (N,) for nFrag=1, (N,F) for nFrag>1,
    # or (N,F,F) for MixDom2
    if r_raw.ndim == 1:
        # nFrag=1: synthesize (N,1) arrays so downstream code never sees None
        result['r'] = r_raw
        result['frag_weights'] = jnp.ones((r_raw.shape[0], 1))
        result['r_frags'] = r_raw[:, None]
        result['ext_matrix'] = r_raw[:, None, None]  # (N, 1, 1)
    elif r_raw.ndim == 3:
        # MixDom2: (N, F, F) fragment transition matrix
        frag_weights = softmax(raw['logit_frag_weights'])  # (N, F)
        result['r_frags'] = r_raw           # (N, F, F) transition matrices
        result['frag_weights'] = frag_weights  # (N, F) fragment weights
        result['ext_matrix'] = r_raw         # (N, F, F)
        # Scalar r is a diagnostic only (mean row sum); NOT used for inference.
        result['r'] = jnp.mean(r_raw.sum(axis=-1), axis=-1)  # (N,)
    else:
        # nFrag>1 MixDom1: (N, F) per-fragment scalar rates
        frag_weights = softmax(raw['logit_frag_weights'])  # (N, F)
        result['r_frags'] = r_raw           # (N, F) per-fragment rates
        result['frag_weights'] = frag_weights  # (N, F) fragment weights
        # Convert to diagonal ext_matrix for unified representation.
        result['ext_matrix'] = jax.vmap(jnp.diag)(r_raw)  # (N, F, F)
        result['r'] = jnp.mean(r_raw, axis=-1)  # (N,) diagnostic only

    # Site class handling: log_pi is (N, AA) for nSites=1, (N, C, AA) for nSites>1
    pi_raw = softmax(raw['log_pi'])
    if pi_raw.ndim == 2:
        # nSites=1: single pi per domain
        result['pi'] = pi_raw                # (N, AA)
        result['site_weights'] = None
        result['pi_classes'] = None
    else:
        # nSites>1: class-weighted average pi
        site_weights = softmax(raw['logit_site_weights'])  # (N, C)
        result['pi_classes'] = pi_raw         # (N, C, AA)
        result['site_weights'] = site_weights  # (N, C)
        result['pi'] = jnp.einsum('nc,nca->na', site_weights, pi_raw)  # (N, AA)

    # MixDom2 per-frag class structure (Annabel hierarchy): C-class flat
    # mixture indexed across (D, F). Pass through as linear, normalised.
    # Symmetrised exchangeability with zeroed diagonal (matches per-domain
    # S_exch convention above).
    if 'log_class_pi' in raw and 'log_class_S' in raw and 'log_class_dist' in raw:
        class_pi = softmax(raw['log_class_pi'])  # (C, AA)
        S_class_raw = jnp.exp(raw['log_class_S'])  # (C, AA, AA)
        def _sym(S):
            S_s = (S + S.T) / 2
            return S_s - jnp.diag(jnp.diag(S_s))
        class_S_exch = jax.vmap(_sym)(S_class_raw)  # (C, AA, AA)
        # classdist is normalised over the C axis at each (d, f).
        class_dist = softmax(raw['log_class_dist'])  # (D, F, C)
        result['class_pi'] = class_pi
        result['class_S_exch'] = class_S_exch
        result['class_dist'] = class_dist
        result['n_classes_per_frag'] = int(class_pi.shape[0])

    return result


def convert_bw_checkpoint(data):
    """Convert old Baum-Welch checkpoint format to maraschino raw params.

    BW checkpoints store dom_Qs (N,A,A), dom_S_exch (N,A,A), dom_pis (N,A),
    dom_weights (N,), dom_ins (N,), dom_del (N,), ext_rates (N,F),
    frag_weights (N,F), main_ins, main_del.

    MixDom2 (Annabel-style) checkpoints additionally store class_pis (C,A),
    class_S_exch (C,A,A), classdist (D,F,C), and n_classes_frag. When these
    keys are present, they are forwarded as ``log_class_pi``, ``log_class_S``,
    and ``log_class_dist`` raw params; ``constrain_params`` then exposes
    them as ``class_pi``, ``class_S_exch``, ``class_dist``, and
    ``n_classes_per_frag`` for downstream consumers (partition recon,
    composite beam). The legacy per-domain ``log_S`` / ``log_pi`` still
    carry the class-marginal averages for backward compatibility, but
    callers that want the real per-class structure should read the
    ``class_*`` keys.

    Returns:
        raw_params: dict with log_S, log_pi, log_v, etc. in maraschino format.
            When MixDom2 class data is present, also contains
            ``log_class_pi``, ``log_class_S``, ``log_class_dist``.
        n_domains: int
        n_classes: int. For MixDom2 checkpoints, this is the per-fragment
            class count ``n_classes_frag`` from the file. Otherwise 1
            (legacy: BW models used no gamma).
    """
    n_dom = int(data['dom_weights'].shape[0])
    n_frag = int(data['ext_rates'].shape[1]) if data['ext_rates'].ndim == 2 else 1

    # S_exch: per-domain (N, A, A) — use LG if not in checkpoint
    if 'dom_S_exch' in data:
        S_exch = np.array(data['dom_S_exch'])
        S_sym = (S_exch + S_exch.transpose(0, 2, 1)) / 2
    else:
        from ..core.protein import rate_matrix_lg
        Q_lg, pi_lg = rate_matrix_lg()
        Q_lg, pi_lg = np.asarray(Q_lg), np.asarray(pi_lg)
        S_lg = Q_lg / pi_lg[None, :]
        np.fill_diagonal(S_lg, 0.0)
        S_lg = (S_lg + S_lg.T) / 2
        S_sym = np.broadcast_to(S_lg[None, :, :], (n_dom, 20, 20)).copy()
    S_sym = np.maximum(S_sym, 1e-30)

    # Equilibrium distributions — use LG if not in checkpoint
    if 'dom_pis' in data:
        pis = np.array(data['dom_pis'])
    else:
        from ..core.protein import rate_matrix_lg
        _, pi_lg = rate_matrix_lg()
        pis = np.broadcast_to(np.asarray(pi_lg)[None, :], (n_dom, 20)).copy()
    pis = np.maximum(pis, 1e-30)
    pis = pis / pis.sum(axis=-1, keepdims=True)

    raw = {
        'log_S': jnp.array(np.log(S_sym)),                    # (N, A, A)
        'log_pi': jnp.array(np.log(pis)),                     # (N, A)
        'log_v': jnp.array(np.log(np.maximum(data['dom_weights'], 1e-30))),
        'log_lam': jnp.array(np.log(np.maximum(data['dom_ins'], 1e-30))),
        'log_mu': jnp.array(np.log(np.maximum(data['dom_del'], 1e-30))),
        'log_lam0': jnp.array(np.log(max(float(data['main_ins']), 1e-30))),
        'log_mu0': jnp.array(np.log(max(float(data['main_del']), 1e-30))),
        'log_alpha_gamma': jnp.array(0.0),  # no gamma — single rate class
    }

    # Extension rates / fragment weights
    ext = np.array(data['ext_rates'])
    if ext.ndim == 3:
        # MixDom2 format: (D, F, F) fragment transition matrix
        # Store the full matrix as logit_r (D, F, F)
        ext_clipped = np.clip(ext, 1e-30, 1 - 1e-30)
        raw['logit_r'] = jnp.array(np.log(ext_clipped / (1 - ext_clipped)))
        fw = np.array(data['frag_weights'])
        fw = np.maximum(fw, 1e-30)
        fw = fw / fw.sum(axis=-1, keepdims=True)
        raw['logit_frag_weights'] = jnp.array(np.log(fw))
        n_frag = ext.shape[1]
    elif ext.ndim == 2 and ext.shape[1] > 1:
        raw['logit_r'] = jnp.array(np.log(np.maximum(ext, 1e-30) /
                                           np.maximum(1 - ext, 1e-30)))
        fw = np.array(data['frag_weights'])
        fw = np.maximum(fw, 1e-30)
        fw = fw / fw.sum(axis=-1, keepdims=True)
        raw['logit_frag_weights'] = jnp.array(np.log(fw))
    else:
        r = ext.ravel()[:n_dom]
        raw['logit_r'] = jnp.array(np.log(np.maximum(r, 1e-30) /
                                           np.maximum(1 - r, 1e-30)))

    # MixDom2 per-frag class structure (Annabel hierarchy). When the
    # checkpoint carries class_pis / class_S_exch / classdist + n_classes_frag,
    # forward them to constrain_params as log-space raw params. We use
    # log-space here because constrain_params applies softmax/symmetrise to
    # turn them into normalised probabilities and a clean exchangeability.
    n_classes_per_frag = 1
    if ('class_pis' in data and 'class_S_exch' in data
            and 'classdist' in data and 'n_classes_frag' in data):
        n_classes_per_frag = int(data['n_classes_frag'])
        class_pis = np.array(data['class_pis'], dtype=np.float64)  # (C, A)
        class_S_exch = np.array(data['class_S_exch'],
                                dtype=np.float64)               # (C, A, A)
        classdist = np.array(data['classdist'], dtype=np.float64)  # (D, F, C)

        # Renormalise WITHOUT a positivity floor: Annabel's classdist is
        # genuinely sparse (e.g. 868 / 972 entries are exact zeros for the
        # GTR_3dom_3frag_3site model — only c=(d, f, *, *) are nonzero
        # for fixed (d, f)). A floor like np.maximum(classdist, 1e-30)
        # would corrupt that sparsity and silently spread ghost mass
        # across all 108 classes (feedback_no_compromises.md #8: silent
        # epsilon floor where the math wants honest zero).
        # Sparse zeros propagate through np.log → -inf → softmax in
        # constrain_params yields exact zero probability for inactive
        # classes, which is what the model expects. Pre-log we still
        # divide by the row sum to renormalise, with a 1e-300 floor only
        # on the divisor (never the entries themselves) for safety.
        classdist = (classdist /
                     np.maximum(classdist.sum(axis=-1, keepdims=True), 1e-300))
        class_pis = (class_pis /
                     np.maximum(class_pis.sum(axis=-1, keepdims=True), 1e-300))
        class_S_sym = (class_S_exch + class_S_exch.transpose(0, 2, 1)) / 2

        # log of zero → -inf, propagates correctly through softmax
        # (exp(-inf) = 0) and the downstream class_dist > 0 / class_pi > 0
        # / class_S_exch > 0 guards.
        with np.errstate(divide='ignore'):
            raw['log_class_pi'] = jnp.array(np.log(class_pis))    # (C, A)
            raw['log_class_S'] = jnp.array(np.log(class_S_sym))   # (C, A, A)
            raw['log_class_dist'] = jnp.array(np.log(classdist))  # (D, F, C)

    return raw, n_dom, n_classes_per_frag


def load_params(path):
    """Load MixDom parameters from .npz file and constrain.

    Handles maraschino format (log_S, log_pi, etc.) and old Baum-Welch
    checkpoint format (dom_Qs, dom_S_exch, dom_pis, etc.).
    Handles nFrag=1 (logit_r shape (N,)) and nFrag>1 (logit_r shape (N,F)).
    Handles nSites=1 (log_pi shape (N, AA)) and nSites>1 (log_pi shape (N, C, AA)).
    Handles MixDom2 (Annabel-style) checkpoints with per-fragment class
    mixture (``class_pis``, ``class_S_exch``, ``classdist`` keys); when
    present, the returned ``params`` dict carries ``class_pi`` (C, A),
    ``class_S_exch`` (C, A, A), ``class_dist`` (D, F, C), and
    ``n_classes_per_frag`` so downstream consumers can build per-class
    P(t) tables and fragment-dependent emission mixtures.

    Args:
        path: path to .npz file (e.g. params/allseed_params.npz)

    Returns:
        params: constrained parameter dict
        n_domains: int
        n_classes: int. For BW MixDom2 checkpoints this is the per-fragment
            flat-class count (e.g. 108 for D=3, F=3, S=3 Annabel imports).
            For maraschino-format files this is the gamma rate-class count.
            For legacy BW MixDom1 checkpoints this is 1.
    """
    data = np.load(path, allow_pickle=True)

    # Auto-detect BW checkpoint format (SVI-BW saves dom_weights, main_ins, etc.)
    if 'dom_Qs' in data or 'dom_S_exch' in data or ('dom_weights' in data and 'main_ins' in data):
        raw_params, n_domains, n_classes = convert_bw_checkpoint(data)
        params = constrain_params(raw_params)
        return params, n_domains, n_classes

    n_domains = int(data['n_domains'])
    n_classes = int(data['n_classes'])

    required = ['log_lam0', 'log_mu0', 'log_lam', 'log_mu', 'logit_r',
                'log_v', 'log_pi', 'log_S', 'log_alpha_gamma']
    optional = ['logit_frag_weights', 'logit_site_weights']

    raw_params = {}
    for key in required:
        raw_params[key] = jnp.array(data[key])
    for key in optional:
        if key in data:
            raw_params[key] = jnp.array(data[key])

    params = constrain_params(raw_params)
    return params, n_domains, n_classes


def load_constrained_params(path):
    """Load pre-constrained MixDom parameters from .npz file.

    For files that store already-constrained params (no log_/logit_ prefixes).
    Infers n_domains from shape of 'lam' and n_classes as the gamma rate
    classes from alpha_gamma.

    Args:
        path: path to .npz file with constrained params

    Returns:
        params: constrained parameter dict
        n_domains: int
        n_classes: int (gamma rate classes, from alpha_gamma)
    """
    data = np.load(path, allow_pickle=True)

    params = {}
    for key in data.keys():
        val = data[key]
        if val.ndim == 0 and key in ('n_domains', 'n_classes'):
            continue  # metadata, not params
        params[key] = jnp.array(val)

    # Infer n_domains from lam shape
    n_domains = int(params['lam'].shape[0])

    # n_classes = gamma rate classes (inferred from alpha_gamma, default 4)
    n_classes = int(data['n_classes']) if 'n_classes' in data else 4

    # Ensure frag_weights and site_weights exist as None if absent
    for k in ('frag_weights', 'r_frags', 'site_weights'):
        if k not in params:
            params[k] = None

    return params, n_domains, n_classes


# ============================================================
# Matrix inversion helpers
# ============================================================
def _batch_inv_3x3(A):
    """Closed-form inverse of batched 3x3 matrices via adjugate/det."""
    a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g, h, i = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]

    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

    adj = jnp.stack([
        jnp.stack([e*i - f*h, c*h - b*i, b*f - c*e], axis=-1),
        jnp.stack([f*g - d*i, a*i - c*g, c*d - a*f], axis=-1),
        jnp.stack([d*h - e*g, b*g - a*h, a*e - b*d], axis=-1),
    ], axis=-2)

    return adj / det[..., None, None]


def _inv_2x2(A):
    """Closed-form inverse of a 2x2 matrix."""
    det = A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]
    return jnp.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / det


# ============================================================
# Precomputation (tau-independent)
# ============================================================
def precompute_mixdom(params, n_classes):
    """Precompute tau-independent quantities for distillation."""
    S_exch = params['S_exch']
    pis = params['pi']

    if S_exch.ndim == 3:
        # Per-domain S_exch: (N, AA, AA)
        def domain_eigen(S_n, pi_n):
            Q = _build_rate_matrix_acknowledged(S_n, pi_n)
            return eigen_decompose(Q, pi_n)
        all_eigvals, all_eigvecs, all_sqrt_pi = vmap(domain_eigen)(S_exch, pis)
    else:
        # Shared S_exch: (AA, AA) — backward compatible
        def domain_eigen(pi_n):
            Q = _build_rate_matrix_acknowledged(S_exch, pi_n)
            return eigen_decompose(Q, pi_n)
        all_eigvals, all_eigvecs, all_sqrt_pi = vmap(domain_eigen)(pis)
    rate_mults = gamma_rates(params['alpha_gamma'], n_classes)

    return {
        'eigvals': all_eigvals,
        'eigvecs': all_eigvecs,
        'sqrt_pi': all_sqrt_pi,
        'rate_mults': rate_mults,
    }


# ============================================================
# Core algebraic distillation via Woodbury identity
# ============================================================
def distill_mixdom(params, tau, n_classes, precomp=None, P_domains_override=None,
                   class_exposed=False):
    """Compute order-1 Singlet HMM and Pair WFST adjacency frequencies.

    Uses Woodbury identity: O(N·AA² + 3³) instead of O((5N)³).
    All domain loops vectorized. Eigendecompositions optionally precomputed.
    Can be vmapped over tau when precomp is provided.

    Args:
        params: constrained parameter dict
        tau: evolutionary time (scalar)
        n_classes: number of gamma rate classes (gamma rate categories)
        precomp: optional precomputed eigendecompositions from precompute_mixdom
        P_domains_override: optional (N, AA, AA) substitution matrices
        class_exposed: if True, use D*AA effective alphabet (expose dynamic class);
            if False, marginalize dynamic class to AA alphabet (default)

    Returns:
        dict with adjacency frequency arrays and singlet HMM params.
        When class_exposed=True, frequency tensors have shape (D*AA, ...)
        instead of (AA, ...).
    """
    N = params['lam'].shape[0]
    v = params['v']
    r = params['r']
    pis = params['pi']

    # --- Determine fragment structure ---
    r_frags = params.get('r_frags', None)
    frag_weights_param = params.get('frag_weights', None)
    if r_frags is not None and r_frags.ndim == 3:
        # MixDom2: full F×F fragment transition matrix per domain
        F = r_frags.shape[1]
        ext_matrix = r_frags  # (N, F, F)
    elif r_frags is not None and r_frags.ndim == 2 and r_frags.shape[1] > 1:
        # MixDom1: diagonal F×F (per-fragment scalar rates)
        F = r_frags.shape[1]
        ext_matrix = jnp.zeros((N, F, F)).at[
            :, jnp.arange(F), jnp.arange(F)
        ].set(r_frags)  # (N, F, F) diagonal
    else:
        # F=1: scalar extension rate
        F = 1
        ext_matrix = r[:, None, None]  # (N, 1, 1)

    if frag_weights_param is None or F == 1:
        frag_w = jnp.ones((N, F)) / F  # uniform for F=1
    else:
        frag_w = frag_weights_param  # (N, F)
    S5F = 5 * F  # total number of within-domain states

    # --- BDI params (vectorized over domains) ---
    alphas, betas, gammas, kappas = vmap(
        lambda l, m: bdi_params(l, m, tau)
    )(params['lam'], params['mu'])
    alpha0, beta0, gamma0, kappa0 = bdi_params(params['lam0'], params['mu0'], tau)

    ob = 1.0 - betas
    og = 1.0 - gammas

    # --- Null closure (2x2 closed form) ---
    z0 = jnp.sum(v * (1.0 - kappas))
    zt = jnp.sum(v * (1.0 - kappas) * ob)

    tau_tl = tkf91_trans(alpha0, beta0, gamma0, kappa0)

    U_AB = jnp.array([
        [zt * tau_tl[1, 1] + z0 * tau_tl[1, 2], z0 * tau_tl[1, 3]],
        [zt * tau_tl[3, 1] + z0 * tau_tl[3, 2], z0 * tau_tl[3, 3]]
    ])
    null_closure = _inv_2x2(jnp.eye(2) - U_AB)

    # --- T_bullet (5x5 null-eliminated top-level matrix) ---
    z_factor = jnp.array([0.0, 1.0 - zt, 1.0 - z0, 1.0 - z0, 1.0])
    U_direct = (tau_tl * z_factor[None, :]).at[:, 0].set(0.0)

    U_to_null = jnp.stack([
        zt * tau_tl[:, 1] + z0 * tau_tl[:, 2],
        z0 * tau_tl[:, 3]
    ], axis=1)

    U_from_null = jnp.stack([
        jnp.array([0.0, (1 - zt)*tau_tl[1, 1], (1 - z0)*tau_tl[1, 2],
                    (1 - z0)*tau_tl[1, 3], tau_tl[1, 4]]),
        jnp.array([0.0, (1 - zt)*tau_tl[3, 1], (1 - z0)*tau_tl[3, 2],
                    (1 - z0)*tau_tl[3, 3], tau_tl[3, 4]]),
    ])

    T_bullet = U_direct + U_to_null @ null_closure @ U_from_null
    T_mid = T_bullet[1:4, 1:4]

    # --- Per-domain P(t) ---
    if precomp is not None:
        rate_mults = precomp['rate_mults']
        def domain_P(eigvals, eigvecs, sqrt_pi):
            Ps = vmap(lambda rho: transition_probs_from_eigen(
                eigvals, eigvecs, sqrt_pi, rho * tau))(rate_mults)
            return jnp.mean(Ps, axis=0)
        P_domains = vmap(domain_P)(
            precomp['eigvals'], precomp['eigvecs'], precomp['sqrt_pi'])
    else:
        S_exch = params['S_exch']
        rate_mults = gamma_rates(params['alpha_gamma'], n_classes)
        def domain_P_full(pi_n):
            Q = _build_rate_matrix_acknowledged(S_exch, pi_n)
            eigvals, eigvecs, sqrt_pi = eigen_decompose(Q, pi_n)
            Ps = vmap(lambda rho: transition_probs_from_eigen(
                eigvals, eigvecs, sqrt_pi, rho * tau))(rate_mults)
            return jnp.mean(Ps, axis=0)
        P_domains = vmap(domain_P_full)(pis)

    if P_domains_override is not None:
        P_domains = P_domains_override

    # --- Within-domain blocks D_n (N, 5F, 5F) ---
    # tau_3x3: (N, 3, 3) TKF transition matrix restricted to {M, I, D}
    tau_3x3 = jnp.stack([
        jnp.stack([ob*kappas*alphas, betas, ob*kappas*(1-alphas)], axis=1),
        jnp.stack([ob*kappas*alphas, betas, ob*kappas*(1-alphas)], axis=1),
        jnp.stack([og*kappas*alphas, gammas, og*kappas*(1-alphas)], axis=1),
    ], axis=1)  # (N, 3, 3)

    # notext_f = 1 - sum_g ext_{fg}: prob of not extending from fragment f
    notext = 1.0 - ext_matrix.sum(axis=-1)  # (N, F)

    # D_match (3F×3F per domain):
    #   D_match[(x,f),(y,g)] = ext_{fg} · δ(x=y) + notext_f · τ_{xy} · w_g
    # Kronecker structure: I_3 ⊗ ext + τ ⊗ (notext · w^T)
    # Build as (N, 3, F, 3, F) then reshape to (N, 3F, 3F)
    D_match = (jnp.eye(3)[:, :, None, None] * ext_matrix[:, None, None, :, :]
               + tau_3x3[:, :, :, None, None]
               * (notext[:, None, None, :, None] * frag_w[:, None, None, None, :]))
    # Shape: (N, 3, 3, F, F) → transpose to (N, 3, F, 3, F) → reshape
    D_match = D_match.transpose(0, 1, 3, 2, 4).reshape(N, 3*F, 3*F)

    # D_loop (F×F per domain): for II and DD states
    #   D_loop[f,g] = ext_{fg} + notext_f · κ · w_g
    D_loop = ext_matrix + kappas[:, None, None] * (
        notext[:, :, None] * frag_w[:, None, :])  # (N, F, F)

    D_blocks = jnp.zeros((N, S5F, S5F))
    D_blocks = D_blocks.at[:, :3*F, :3*F].set(D_match)
    D_blocks = D_blocks.at[:, 3*F:4*F, 3*F:4*F].set(D_loop)
    D_blocks = D_blocks.at[:, 4*F:5*F, 4*F:5*F].set(D_loop)

    # --- G_n = (I - D_n)^{-1} per domain ---
    G_all = vmap(lambda D: jnp.linalg.inv(jnp.eye(S5F) - D))(D_blocks)  # (N, 5F, 5F)

    # --- Exit and start vectors (N, 5F) ---
    # exit_vecs[(s,f)] = notext_f · exit_factor_s
    tau_XE_m = ob * (1 - kappas)  # (N,)
    tau_XE_d = og * (1 - kappas)  # (N,)

    exit_factor_5 = jnp.stack([
        tau_XE_m, tau_XE_m, tau_XE_d,   # MM, MI, MD
        1 - kappas, 1 - kappas,          # II, DD
    ], axis=1)  # (N, 5)
    exit_vecs = (exit_factor_5[:, :, None] * notext[:, None, :]).reshape(N, S5F)

    # start_vecs[(s,f)] = start_5_s · w_f
    zt_safe = jnp.maximum(1 - zt, 1e-30)
    z0_safe = jnp.maximum(1 - z0, 1e-30)

    start_5 = jnp.stack([
        v * ob * kappas * alphas / zt_safe,       # MM
        v * betas / zt_safe,                       # MI
        v * ob * kappas * (1 - alphas) / zt_safe,  # MD
        v * kappas / z0_safe,                      # II
        v * kappas / z0_safe,                      # DD
    ], axis=1)  # (N, 5)
    start_vecs = (start_5[:, :, None] * frag_w[:, None, :]).reshape(N, S5F)

    T_bullet_S_5 = jnp.array([T_bullet[0, 1], T_bullet[0, 1], T_bullet[0, 1],
                               T_bullet[0, 2], T_bullet[0, 3]])
    T_bullet_S_5F = jnp.repeat(T_bullet_S_5, F)  # (5F,)
    T_start_n = T_bullet_S_5F[None, :] * start_vecs  # (N, 5F)

    T_bullet_E_5 = jnp.array([T_bullet[1, 4], T_bullet[1, 4], T_bullet[1, 4],
                               T_bullet[2, 4], T_bullet[3, 4]])
    T_bullet_E_5F = jnp.repeat(T_bullet_E_5, F)  # (5F,)
    T_end_n = exit_vecs * T_bullet_E_5F[None, :]  # (N, 5F)

    # --- Projection to top-level MID types (5F → 3) ---
    est_to_tl_5F = jnp.repeat(EST_TO_TL, F)  # (5F,)
    tl_mid = jax.nn.one_hot(est_to_tl_5F, 5)[:, 1:4]  # (5F, 3)

    # --- Masks for emission-type groups (5F,) ---
    M_mask = jnp.zeros(S5F).at[:F].set(1.0)
    I_mask = jnp.zeros(S5F).at[F:2*F].set(1.0).at[3*F:4*F].set(1.0)
    D_mask = jnp.zeros(S5F).at[2*F:3*F].set(1.0).at[4*F:5*F].set(1.0)

    # --- Woodbury: path_sum = G + G·E·K·S^T·G ---
    # The inter-domain coupling still has rank 3 (via top-level M,I,D types)
    SGE_3 = jnp.einsum('sp,ns,nst,nt,tq->pq',
                        tl_mid, start_vecs, G_all, exit_vecs, tl_mid)

    inv_I_ST = _batch_inv_3x3((jnp.eye(3) - SGE_3 @ T_mid)[None])[0]
    K_3 = T_mid @ inv_I_ST

    # --- L = T_start @ path_sum via Woodbury ---
    L_G = jnp.einsum('ns,nst->nt', T_start_n, G_all)
    l_3 = jnp.einsum('ns,nst,nt,tq->q', T_start_n, G_all, exit_vecs, tl_mid)
    kl_3 = l_3 @ K_3
    sG_3 = jnp.einsum('sp,ns,nst->npt', tl_mid, start_vecs, G_all)
    L_r = L_G + jnp.einsum('npt,p->nt', sG_3, kl_3)

    # --- R = path_sum @ T_end via Woodbury ---
    R_G = jnp.einsum('nst,nt->ns', G_all, T_end_n)
    r_3 = jnp.einsum('sp,ns,nst,nt->p', tl_mid, start_vecs, G_all, T_end_n)
    kr_3 = K_3 @ r_3
    Ge_3 = jnp.einsum('nst,nt,tq->nsq', G_all, exit_vecs, tl_mid)
    R_r = R_G + jnp.einsum('nsq,q->ns', Ge_3, kr_3)

    # --- Structural weights W ---

    def compute_W(mask1, mask2):
        w_diag = jnp.einsum('ns,nst,nt->n', L_r * mask1, D_blocks, R_r * mask2)
        l_x = jnp.einsum('ns,ns,sq->nq', L_r * mask1, exit_vecs, tl_mid)
        r_y = jnp.einsum('ns,ns,sp->np', start_vecs, R_r * mask2, tl_mid)
        w_cross = jnp.einsum('nq,qp,mp->nm', l_x, T_mid, r_y)
        return jnp.diag(w_diag) + w_cross

    W = {
        'MM': compute_W(M_mask, M_mask), 'MI': compute_W(M_mask, I_mask),
        'MD': compute_W(M_mask, D_mask), 'IM': compute_W(I_mask, M_mask),
        'II': compute_W(I_mask, I_mask), 'ID': compute_W(I_mask, D_mask),
        'DM': compute_W(D_mask, M_mask), 'DD': compute_W(D_mask, D_mask),
        'DI': compute_W(D_mask, I_mask),
    }

    W_start = {
        'M': jnp.sum(T_start_n * R_r * M_mask, axis=1),
        'I': jnp.sum(T_start_n * R_r * I_mask, axis=1),
        'D': jnp.sum(T_start_n * R_r * D_mask, axis=1),
    }
    W_end = {
        'M': jnp.sum(L_r * T_end_n * M_mask, axis=1),
        'I': jnp.sum(L_r * T_end_n * I_mask, axis=1),
        'D': jnp.sum(L_r * T_end_n * D_mask, axis=1),
    }

    # --- Emission tensors ---
    # Dynamic site classes: build composite emission tensors
    if params.get('site_weights') is not None:
        # nSites>1 site-class mixture (from log_pi with nSites>1):
        # class-weighted average of per-class P(t).
        pi_classes = params['pi_classes']  # (N, C, AA)
        site_weights = params['site_weights']  # (N, C)
        e_M = jnp.einsum('nc,nca,nab->nab', site_weights, pi_classes, P_domains)
    else:
        e_M = pis[:, :, None] * P_domains  # (N, AA, AA)

    # --- Full-context Insert/Delete chain computation ---
    # With F fragments, Insert states are MI(F..2F) and II(3F..4F) → 2F per domain.
    # Delete states are MD(2F..3F) and DD(4F..5F) → 2F per domain.
    # Build the full N×2F Insert/Delete chains via Woodbury with 2×2 kernel.

    # --- Insert chain ---
    # Extract restricted Insert sub-blocks from D_blocks
    I_idx = jnp.concatenate([jnp.arange(F, 2*F), jnp.arange(3*F, 4*F)])  # (2F,)
    D_I_restricted = D_blocks[:, I_idx][:, :, I_idx]  # (N, 2F, 2F)
    g_I_mat = vmap(lambda D: jnp.linalg.inv(jnp.eye(2*F) - D))(D_I_restricted)  # (N, 2F, 2F)
    start_I = start_vecs[:, I_idx]  # (N, 2F)
    exit_I = exit_vecs[:, I_idx]    # (N, 2F)

    # TL projection for Insert: MI(F states)→M(0), II(F states)→I(1)
    tl_mid_I = jnp.zeros((2*F, 2)).at[:F, 0].set(1.0).at[F:, 1].set(1.0)
    T_mid_I = T_mid[:2, :2]  # (2, 2)

    SGE_I = jnp.einsum('sp,ns,nst,nt,tq->pq',
                        tl_mid_I, start_I, g_I_mat, exit_I, tl_mid_I)  # (2, 2)
    K_I = T_mid_I @ jnp.linalg.inv(jnp.eye(2) - SGE_I @ T_mid_I)  # (2, 2)

    # --- Delete chain ---
    D_didx = jnp.concatenate([jnp.arange(2*F, 3*F), jnp.arange(4*F, 5*F)])  # (2F,)
    D_D_restricted = D_blocks[:, D_didx][:, :, D_didx]  # (N, 2F, 2F)
    g_D_mat = vmap(lambda D: jnp.linalg.inv(jnp.eye(2*F) - D))(D_D_restricted)  # (N, 2F, 2F)
    start_D = start_vecs[:, D_didx]  # (N, 2F)
    exit_D = exit_vecs[:, D_didx]    # (N, 2F)

    # TL projection for Delete: MD(F states)→M(0), DD(F states)→D(1 in 2-dim space)
    tl_mid_D = jnp.zeros((2*F, 2)).at[:F, 0].set(1.0).at[F:, 1].set(1.0)
    T_mid_D = T_mid[jnp.array([0, 2])][:, jnp.array([0, 2])]  # (2, 2)

    SGE_D = jnp.einsum('sp,ns,nst,nt,tq->pq',
                        tl_mid_D, start_D, g_D_mat, exit_D, tl_mid_D)  # (2, 2)
    K_D = T_mid_D @ jnp.linalg.inv(jnp.eye(2) - SGE_D @ T_mid_D)  # (2, 2)

    # --- Ancestor-conditioned left vectors L̃^I (for Insert passthrough) ---
    # Lambda: total M/D left boundary weight per domain
    # Ancestor-emitting states: MM(0..F), MD(2F..3F), DD(4F..5F)
    lambda_MD = (L_r[:, :F].sum(axis=1)
                 + L_r[:, 2*F:3*F].sum(axis=1)
                 + L_r[:, 4*F:5*F].sum(axis=1))  # (N,)
    lambda_MD_safe = jnp.maximum(lambda_MD, 1e-30)

    # Within-domain M/D→I entry: transitions from MM/MD states into MI states
    # MM(f)→MI(g): D_blocks[:, f, F+g] for f∈[0,F), g∈[0,F)
    # MD(f)→MI(g): D_blocks[:, 2F+f, F+g] for f∈[0,F), g∈[0,F)
    # Weighted by L_r at source states, summed over source fragments
    entry_I_within = jnp.zeros((N, 2*F))
    entry_MI = (jnp.einsum('nf,nfg->ng', L_r[:, :F], D_blocks[:, :F, F:2*F])
                + jnp.einsum('nf,nfg->ng', L_r[:, 2*F:3*F], D_blocks[:, 2*F:3*F, F:2*F]))
    entry_I_within = entry_I_within.at[:, :F].set(entry_MI)  # MI entries; II=0

    # Cross-domain M/D→I: exit from M/D → T_mid → start at Insert
    exit_MD_3 = jnp.stack([
        jnp.sum(L_r[:, :F] * exit_vecs[:, :F], axis=1)
        + jnp.sum(L_r[:, 2*F:3*F] * exit_vecs[:, 2*F:3*F], axis=1),   # M col
        jnp.zeros(N),                                                     # I col
        jnp.sum(L_r[:, 4*F:5*F] * exit_vecs[:, 4*F:5*F], axis=1),      # D col
    ], axis=1) / lambda_MD_safe[:, None]  # (N, 3)
    c_I = exit_MD_3 @ T_mid[:, :2]  # (N, 2): cross-domain entry coeff (TL-space)

    # Map c_I to 2F-space: c_I_2F[n, f] = c_I[n, tl_type_of_f]
    c_I_2F = c_I @ tl_mid_I.T  # (N, 2F)

    # Compute H_I[n0, n1] via Woodbury on the Insert sub-chain
    within_entry_I_norm = entry_I_within / lambda_MD_safe[:, None]  # (N, 2F)

    # Within-domain contribution
    H_I_diag = jnp.einsum('nf,nfg->n', within_entry_I_norm, g_I_mat)  # (N,)

    # Cross-domain direct through g_I
    gs_I = jnp.einsum('nfg,ng->nf', g_I_mat, start_I)  # (N, 2F)
    H_I_cross = jnp.einsum('nf,mf->nm', c_I_2F, gs_I)  # (N, N)

    # Woodbury correction for Insert chain
    B_I_within = jnp.einsum('nf,nfg,ng,gp->np', within_entry_I_norm, g_I_mat, exit_I, tl_mid_I)
    B_I_cross = c_I * jnp.einsum('nf,nfg,ng,gp->p', start_I, g_I_mat, exit_I, tl_mid_I)[None, :]
    B_I = B_I_within + B_I_cross  # (N, 2)

    # Correction: B @ K_I @ (tl_mid_I^T @ gs_I)^T, contracted to (N, N)
    gs_I_tl = jnp.einsum('fp,nf->np', tl_mid_I, gs_I)  # (N, 2)
    H_I_correction = (B_I @ K_I) @ gs_I_tl.T  # (N, N)

    H_I = jnp.diag(H_I_diag) + H_I_cross + H_I_correction  # (N, N)

    # L̃^I[n1, X] = sum_{n0} lambda[n0] · pi[n0, X] · H_I[n0, n1]
    L_tilde_I = jnp.einsum('n,nx,nm->mx', lambda_MD, pis, H_I)  # (N, AA)
    L_tilde_I_sum = jnp.sum(L_tilde_I, axis=1, keepdims=True)
    p_anc_I = L_tilde_I / jnp.maximum(L_tilde_I_sum, 1e-30)  # (N, AA)

    # --- Descendant-conditioned left vectors L̃^D (for Delete passthrough) ---
    # Lambda: total M/I left boundary weight
    # Descendant-emitting states: MM(0..F), MI(F..2F), II(3F..4F)
    lambda_MI = (L_r[:, :F].sum(axis=1)
                 + L_r[:, F:2*F].sum(axis=1)
                 + L_r[:, 3*F:4*F].sum(axis=1))  # (N,)
    lambda_MI_safe = jnp.maximum(lambda_MI, 1e-30)

    # Within-domain M/I→D entry: MM(f)→MD(g) and MI(f)→MD(g) transitions
    entry_D_within = jnp.zeros((N, 2*F))
    entry_MD = (jnp.einsum('nf,nfg->ng', L_r[:, :F], D_blocks[:, :F, 2*F:3*F])
                + jnp.einsum('nf,nfg->ng', L_r[:, F:2*F], D_blocks[:, F:2*F, 2*F:3*F]))
    entry_D_within = entry_D_within.at[:, :F].set(entry_MD)  # MD entries; DD=0

    # Cross-domain M/I→D
    exit_MI_3 = jnp.stack([
        jnp.sum(L_r[:, :F] * exit_vecs[:, :F], axis=1)
        + jnp.sum(L_r[:, F:2*F] * exit_vecs[:, F:2*F], axis=1),       # M col
        jnp.sum(L_r[:, 3*F:4*F] * exit_vecs[:, 3*F:4*F], axis=1),     # I col
        jnp.zeros(N),                                                     # D col
    ], axis=1) / lambda_MI_safe[:, None]  # (N, 3)
    c_D = exit_MI_3 @ T_mid[:, jnp.array([0, 2])]  # (N, 2)
    c_D_2F = c_D @ tl_mid_D.T  # (N, 2F)

    # Compute H_D via same Woodbury pattern as Insert
    within_entry_D_norm = entry_D_within / lambda_MI_safe[:, None]  # (N, 2F)
    H_D_diag = jnp.einsum('nf,nfg->n', within_entry_D_norm, g_D_mat)  # (N,)

    gs_D = jnp.einsum('nfg,ng->nf', g_D_mat, start_D)  # (N, 2F)
    H_D_cross = jnp.einsum('nf,mf->nm', c_D_2F, gs_D)  # (N, N)

    B_D_within = jnp.einsum('nf,nfg,ng,gp->np', within_entry_D_norm, g_D_mat, exit_D, tl_mid_D)
    B_D_cross = c_D * jnp.einsum('nf,nfg,ng,gp->p', start_D, g_D_mat, exit_D, tl_mid_D)[None, :]
    B_D = B_D_within + B_D_cross  # (N, 2)

    gs_D_tl = jnp.einsum('fp,nf->np', tl_mid_D, gs_D)  # (N, 2)
    H_D_correction = (B_D @ K_D) @ gs_D_tl.T  # (N, N)

    H_D = jnp.diag(H_D_diag) + H_D_cross + H_D_correction  # (N, N)

    # L̃^D[n1, Y] = sum_{n0} lambda_MI[n0] · pi[n0, Y] · H_D[n0, n1]
    L_tilde_D = jnp.einsum('n,ny,nm->my', lambda_MI, pis, H_D)  # (N, AA)
    L_tilde_D_sum = jnp.sum(L_tilde_D, axis=1, keepdims=True)
    p_desc_D = L_tilde_D / jnp.maximum(L_tilde_D_sum, 1e-30)  # (N, AA)

    # --- Full-context emission tensors ---
    e_I_full = p_anc_I[:, :, None] * pis[:, None, :]  # (N, AA_anc, AA_desc)
    e_D_full = pis[:, :, None] * p_desc_D[:, None, :]  # (N, AA_anc, AA_desc)

    # --- Adjacency frequencies via einsum (full-context) ---
    # Match-sourced: unchanged shapes
    f_MM = jnp.einsum('ij,iab,jcd->abcd', W['MM'], e_M, e_M)     # (AA,AA,AA,AA)
    f_MI = jnp.einsum('ij,iab,jc->abc',   W['MI'], e_M, pis)     # (AA,AA,AA)
    f_MD = jnp.einsum('ij,iab,jc->abc',   W['MD'], e_M, pis)     # (AA,AA,AA)

    # Insert-sourced: gain ancestor passthrough X dimension
    f_IM = jnp.einsum('ij,ixy,jac->xyac', W['IM'], e_I_full, e_M)  # (AA,AA,AA,AA)
    f_II = jnp.einsum('ij,ixy,jz->xyz',   W['II'], e_I_full, pis)  # (AA,AA,AA)
    f_ID = jnp.einsum('ij,ixy,jz->xyz',   W['ID'], e_I_full, pis)  # (AA,AA,AA)

    # Delete-sourced: gain descendant passthrough Y dimension
    f_DM = jnp.einsum('ij,ixy,jac->xyac', W['DM'], e_D_full, e_M)  # (AA,AA,AA,AA)
    f_DD = jnp.einsum('ij,ixy,jz->xyz',   W['DD'], e_D_full, pis)  # (AA,AA,AA)
    f_DI = jnp.einsum('ij,ixy,jz->xyz',   W['DI'], e_D_full, pis)  # (AA,AA,AA)

    # Start/End transitions
    f_SM = jnp.einsum('i,iab->ab', W_start['M'], e_M)                     # (AA,AA)
    f_SI = jnp.einsum('i,ixy->xy', W_start['I'], e_I_full)               # (AA,AA)
    f_SD = jnp.einsum('i,ixy->xy', W_start['D'], e_D_full)               # (AA,AA)
    f_ME = jnp.einsum('i,iab->ab', W_end['M'], e_M)                       # (AA,AA)
    f_IE = jnp.einsum('i,ixy->xy', W_end['I'], e_I_full)                 # (AA,AA)
    f_DE = jnp.einsum('i,ixy->xy', W_end['D'], e_D_full)                 # (AA,AA)
    f_SE = T_bullet[TL_S, TL_E]

    # --- Singlet HMM ---
    # With F fragments, the singlet HMM has NF states: (domain n, fragment f).
    # Within-domain transitions: D_loop[n, f, g] (fragment extension + kappa re-entry).
    # Cross-domain: exit domain → top-level null closure → re-enter at (m, g).
    z0_sing = jnp.sum(v * (1 - kappas))
    null_closure_sing = 1.0 / jnp.maximum(1 - kappa0 * z0_sing, 1e-30)
    kappa0_eff = kappa0 * null_closure_sing     # κ₀/(1-κ₀·z₀)
    v_nonempty = v * kappas                      # raw weights, sum = 1-z₀
    end_factor = (1 - kappa0) * null_closure_sing  # (1-κ₀)/(1-κ₀·z₀)

    NF = N * F
    # D_loop: (N, F, F) within-domain fragment transition
    # Exit prob from (n, f): notext[n, f] * (1-kappa[n])
    sing_exit = notext * (1 - kappas[:, None])  # (N, F)

    # Build NF×NF singlet transition matrix
    # Block-diagonal: D_loop[n] at position (n*F:(n+1)*F, n*F:(n+1)*F)
    T_sing_blocks = jnp.zeros((N, F, N, F))
    T_sing_blocks = T_sing_blocks.at[jnp.arange(N), :, jnp.arange(N), :].set(D_loop)
    T_sing = T_sing_blocks.reshape(NF, NF)

    # Cross-domain: sing_exit[n, f] → kappa0_eff * v_nonempty[m] * frag_w[m, g]
    # T_cross[(n,f),(m,g)] = sing_exit[n,f] * kappa0_eff * v_nonempty[m] * frag_w[m, g]
    sing_exit_flat = sing_exit.reshape(NF)  # (NF,)
    sing_start_flat = (kappa0_eff * v_nonempty[:, None] * frag_w).reshape(NF)  # (NF,)
    T_sing = T_sing + sing_exit_flat[:, None] * sing_start_flat[None, :]

    G_sing = jnp.linalg.inv(jnp.eye(NF) - T_sing)

    sing_start = sing_start_flat  # (NF,)
    sing_end = sing_exit_flat * end_factor  # (NF,)

    L_sing = sing_start @ G_sing  # (NF,)
    R_sing = G_sing @ sing_end    # (NF,)

    W_sing = L_sing[:, None] * T_sing * R_sing[None, :]  # (NF, NF)
    # Emission: state (n, f) emits from pi[n] — fragments share domain emission
    pis_sing = jnp.repeat(pis, F, axis=0)  # (NF, AA)
    f_singlet = jnp.einsum('ij,ia,jb->ab', W_sing, pis_sing, pis_sing)
    f_singlet_start = jnp.einsum('i,ia->a', sing_start * R_sing, pis_sing)
    f_singlet_end = jnp.einsum('i,ia->a', L_sing * sing_end, pis_sing)

    return {
        'f_MM': f_MM, 'f_MI': f_MI, 'f_MD': f_MD,
        'f_IM': f_IM, 'f_II': f_II, 'f_ID': f_ID,
        'f_DM': f_DM, 'f_DD': f_DD, 'f_DI': f_DI,
        'f_SM': f_SM, 'f_SI': f_SI, 'f_SD': f_SD,
        'f_ME': f_ME, 'f_IE': f_IE, 'f_DE': f_DE,
        'f_SE': f_SE,
        'f_singlet': f_singlet,
        'f_singlet_start': f_singlet_start,
        'f_singlet_end': f_singlet_end,
        # Also return intermediate quantities needed for emission model
        'P_domains': P_domains,  # (N, AA, AA)
        'W': W,                  # structural weights
        'W_start': W_start,
        'W_end': W_end,
        'T_bullet': T_bullet,    # 5x5 null-eliminated transition
    }


# ============================================================
# Normalization: frequencies → probabilities
# ============================================================
def normalize_freqs_hmm(dist):
    """Normalize adjacency frequencies to HMM transition probabilities.

    Produces P(next_state, a', b' | current_state, a, b) — the joint
    probability of the next state AND both ancestor/descendant emissions.
    Correct for pairwise HMM scoring but NOT for tree composition (where
    ancestor emissions must not be double-counted across branches).

    Full-context tensor shapes:
        M context: (AA_anc, AA_desc) = (AA, AA)
        I context: (AA_anc_pass, AA_desc) = (AA, AA) — ancestor is passthrough
        D context: (AA_anc, AA_desc_pass) = (AA, AA) — descendant is passthrough
    """
    # Z_M(a,b): total outgoing weight from Match with context (a,b)
    Z_M = (dist['f_MM'].sum(axis=(2,3)) + dist['f_MI'].sum(axis=2) +
           dist['f_MD'].sum(axis=2) + dist['f_ME'])
    Z_M = jnp.maximum(Z_M, 1e-30)

    # Z_I(x,y): total outgoing weight from Insert with context (x_pass, y)
    Z_I = (dist['f_IM'].sum(axis=(2,3)) + dist['f_II'].sum(axis=2) +
           dist['f_ID'].sum(axis=2) + dist['f_IE'])
    Z_I = jnp.maximum(Z_I, 1e-30)

    # Z_D(x,y): total outgoing weight from Delete with context (x, y_pass)
    Z_D = (dist['f_DM'].sum(axis=(2,3)) + dist['f_DD'].sum(axis=2) +
           dist['f_DI'].sum(axis=2) + dist['f_DE'])
    Z_D = jnp.maximum(Z_D, 1e-30)

    Z_S = dist['f_SM'].sum() + dist['f_SI'].sum() + dist['f_SD'].sum() + dist['f_SE']
    Z_S = jnp.maximum(Z_S, 1e-30)

    f_sing = dist['f_singlet']
    f_end = dist['f_singlet_end']
    f_start = dist['f_singlet_start']
    total_out = f_sing.sum(axis=1) + f_end

    return {
        'p_mm': dist['f_MM'] / Z_M[:, :, None, None],        # (AA,AA,AA,AA)
        'p_mi': dist['f_MI'] / Z_M[:, :, None],              # (AA,AA,AA)
        'p_md': dist['f_MD'] / Z_M[:, :, None],              # (AA,AA,AA)
        'p_me': dist['f_ME'] / Z_M,                          # (AA,AA)
        'p_im': dist['f_IM'] / Z_I[:, :, None, None],        # (AA,AA,AA,AA)
        'p_ii': dist['f_II'] / Z_I[:, :, None],              # (AA,AA,AA)
        'p_id': dist['f_ID'] / Z_I[:, :, None],              # (AA,AA,AA)
        'p_ie': dist['f_IE'] / Z_I,                          # (AA,AA)
        'p_dm': dist['f_DM'] / Z_D[:, :, None, None],        # (AA,AA,AA,AA)
        'p_dd': dist['f_DD'] / Z_D[:, :, None],              # (AA,AA,AA)
        'p_di': dist['f_DI'] / Z_D[:, :, None],              # (AA,AA,AA)
        'p_de': dist['f_DE'] / Z_D,                          # (AA,AA)
        'p_sm': dist['f_SM'] / Z_S,                          # (AA,AA)
        'p_si': dist['f_SI'] / Z_S,                          # (AA,AA)
        'p_sd': dist['f_SD'] / Z_S,                          # (AA,AA)
        'p_se': float(dist['f_SE'] / Z_S),
        'singlet_trans': f_sing / jnp.maximum(total_out[:, None], 1e-30),
        'singlet_start': f_start / jnp.maximum(f_start.sum(), 1e-30),
        'singlet_end': f_end / jnp.maximum(total_out, 1e-30),
    }


# normalize_freqs_hmm exists for testing HMM row sums.
# For all production use (Machine Boss export, tree composition), use normalize_freqs_wfst.
_normalize_freqs = normalize_freqs_hmm  # internal alias for test compatibility


def normalize_freqs_wfst(dist):
    """Normalize adjacency frequencies to WFST transition probabilities.

    For sync transitions (those that consume an ancestor char a'), normalizes
    per (a, b, a') to give P(b', next_state | a', current_state, a, b).
    This is the correct normalization for tree composition, where the ancestor
    sequence is shared across branches and must not be double-counted.

    For non-sync transitions (insert, end — no parent consumed), normalizes
    per (a, b) as in the HMM case.

    Also computes the singlet transition log_singlet[a, a'] to be applied
    ONCE at each sync step in compose_intersect.

    Returns:
        dict with WFST-normalized tensors and singlet transition:
            p_mm, p_md, p_im, p_id, p_dm, p_dd: sync (per a,b,a')
            p_mi, p_me, p_ii, p_ie, p_di, p_de: non-sync (per a,b)
            p_sm, p_sd: sync start (per a')
            p_si, p_se: non-sync start
            singlet_trans, singlet_start, singlet_end: singlet model
            log_singlet: (AA, AA) log P(a' | a) from singlet model
    """
    # --- Sync normalization: per (a, b, a') ---
    # Z_M_sync[a,b,a'] = sum_{b'} f_MM[a,b,a',b'] + f_MD[a,b,a']
    Z_M_sync = dist['f_MM'].sum(axis=3) + dist['f_MD']   # (AA,AA,AA)
    Z_M_sync = jnp.maximum(Z_M_sync, 1e-30)

    # Z_I_sync[x,y,a'] = sum_{b'} f_IM[x,y,a',b'] + f_ID[x,y,a']
    Z_I_sync = dist['f_IM'].sum(axis=3) + dist['f_ID']   # (AA,AA,AA)
    Z_I_sync = jnp.maximum(Z_I_sync, 1e-30)

    # Z_D_sync[x,y,a'] = sum_{b'} f_DM[x,y,a',b'] + f_DD[x,y,a']
    Z_D_sync = dist['f_DM'].sum(axis=3) + dist['f_DD']   # (AA,AA,AA)
    Z_D_sync = jnp.maximum(Z_D_sync, 1e-30)

    # Z_S_sync[a'] = sum_{b'} f_SM[a',b'] + f_SD[a'] (sum over b' for SM)
    Z_S_sync = dist['f_SM'].sum(axis=1) + dist['f_SD'].sum(axis=1)  # (AA,) — per a'
    # Actually f_SM is (AA, AA) = (a', b'), f_SD is (AA, AA) = (a', y_pass)
    # For Start, there's no source context (a,b), just normalize per a'
    # Z_S_sync[a'] = f_SM[a', :].sum() + f_SD[a', :].sum()
    Z_S_sync = dist['f_SM'].sum(axis=1) + dist['f_SD'].sum(axis=1)  # (AA,)
    Z_S_sync = jnp.maximum(Z_S_sync, 1e-30)

    # --- Non-sync normalization: per (a, b) as in HMM ---
    Z_M = (dist['f_MM'].sum(axis=(2,3)) + dist['f_MI'].sum(axis=2) +
           dist['f_MD'].sum(axis=2) + dist['f_ME'])
    Z_M = jnp.maximum(Z_M, 1e-30)

    Z_I = (dist['f_IM'].sum(axis=(2,3)) + dist['f_II'].sum(axis=2) +
           dist['f_ID'].sum(axis=2) + dist['f_IE'])
    Z_I = jnp.maximum(Z_I, 1e-30)

    Z_D = (dist['f_DM'].sum(axis=(2,3)) + dist['f_DD'].sum(axis=2) +
           dist['f_DI'].sum(axis=2) + dist['f_DE'])
    Z_D = jnp.maximum(Z_D, 1e-30)

    Z_S = dist['f_SM'].sum() + dist['f_SI'].sum() + dist['f_SD'].sum() + dist['f_SE']
    Z_S = jnp.maximum(Z_S, 1e-30)

    # --- Singlet model ---
    f_sing = dist['f_singlet']
    f_end = dist['f_singlet_end']
    f_start = dist['f_singlet_start']
    total_out = f_sing.sum(axis=1) + f_end
    singlet_trans = f_sing / jnp.maximum(total_out[:, None], 1e-30)
    singlet_start = f_start / jnp.maximum(f_start.sum(), 1e-30)
    singlet_end = f_end / jnp.maximum(total_out, 1e-30)

    return {
        # Sync transitions: normalized per (a, b, a')
        'p_mm': dist['f_MM'] / Z_M_sync[:, :, :, None],     # (AA,AA,AA,AA)
        'p_md': dist['f_MD'] / Z_M_sync,                     # (AA,AA,AA)
        'p_im': dist['f_IM'] / Z_I_sync[:, :, :, None],     # (AA,AA,AA,AA)
        'p_id': dist['f_ID'] / Z_I_sync,                     # (AA,AA,AA)
        'p_dm': dist['f_DM'] / Z_D_sync[:, :, :, None],     # (AA,AA,AA,AA)
        'p_dd': dist['f_DD'] / Z_D_sync,                     # (AA,AA,AA)
        # Sync start: normalized per a'
        'p_sm': dist['f_SM'] / Z_S_sync[:, None],            # (AA,AA)
        'p_sd': dist['f_SD'] / Z_S_sync[:, None],            # (AA,AA)
        # Non-sync transitions: normalized per (a, b) as in HMM
        'p_mi': dist['f_MI'] / Z_M[:, :, None],              # (AA,AA,AA)
        'p_me': dist['f_ME'] / Z_M,                          # (AA,AA)
        'p_ii': dist['f_II'] / Z_I[:, :, None],              # (AA,AA,AA)
        'p_ie': dist['f_IE'] / Z_I,                          # (AA,AA)
        'p_di': dist['f_DI'] / Z_D[:, :, None],              # (AA,AA,AA)
        'p_de': dist['f_DE'] / Z_D,                          # (AA,AA)
        # Non-sync start: normalized per total S
        'p_si': dist['f_SI'] / Z_S,                          # (AA,AA)
        'p_se': float(dist['f_SE'] / Z_S),
        # Singlet model
        'singlet_trans': singlet_trans,
        'singlet_start': singlet_start,
        'singlet_end': singlet_end,
        'log_singlet': jnp.log(jnp.maximum(singlet_trans, 1e-300)),  # (AA, AA)
    }


# ============================================================
# Effective pair HMM for profile DP
# ============================================================
def effective_pair_hmm(params, n_classes, tau, precomp=None):
    """Compute emission-marginalized transition matrix and mixture emission model.

    Returns quantities needed by the profile-based progressive alignment DP:
    - A 5x5 transition matrix (marginalizing over emission contexts)
    - Per-domain substitution matrices and equilibrium frequencies for
      computing domain-mixture Felsenstein likelihoods at each column

    Args:
        params: constrained MixDom parameter dict
        n_classes: number of gamma rate classes
        tau: evolutionary time (total branch length for the pair)
        precomp: optional precomputed eigendecompositions

    Returns:
        dict with:
            log_trans: (5, 5) log transition matrix
            P_domains: (N, AA, AA) per-domain substitution matrices at tau
            pis: (N, AA) per-domain equilibrium distributions
            domain_weights: (N,) domain weights
            pi_mix: (AA,) mixture equilibrium distribution
    """
    dist = distill_mixdom(params, tau, n_classes, precomp)

    # Effective transition matrix by summing frequencies over emissions
    f_M_total = (dist['f_MM'].sum() + dist['f_MI'].sum() +
                 dist['f_MD'].sum() + dist['f_ME'].sum())
    f_I_total = (dist['f_IM'].sum() + dist['f_II'].sum() +
                 dist['f_ID'].sum() + dist['f_IE'].sum())
    f_D_total = (dist['f_DM'].sum() + dist['f_DD'].sum() +
                 dist['f_DI'].sum() + dist['f_DE'].sum())
    f_S_total = dist['f_SM'].sum() + dist['f_SI'].sum() + dist['f_SD'].sum() + dist['f_SE']

    # Build 5x5 transition matrix [S, M, I, D, E]
    trans = jnp.zeros((5, 5))
    # S row
    trans = trans.at[0, 1].set(dist['f_SM'].sum() / jnp.maximum(f_S_total, 1e-30))
    trans = trans.at[0, 2].set(dist['f_SI'].sum() / jnp.maximum(f_S_total, 1e-30))
    trans = trans.at[0, 3].set(dist['f_SD'].sum() / jnp.maximum(f_S_total, 1e-30))
    trans = trans.at[0, 4].set(dist['f_SE'] / jnp.maximum(f_S_total, 1e-30))
    # M row
    trans = trans.at[1, 1].set(dist['f_MM'].sum() / jnp.maximum(f_M_total, 1e-30))
    trans = trans.at[1, 2].set(dist['f_MI'].sum() / jnp.maximum(f_M_total, 1e-30))
    trans = trans.at[1, 3].set(dist['f_MD'].sum() / jnp.maximum(f_M_total, 1e-30))
    trans = trans.at[1, 4].set(dist['f_ME'].sum() / jnp.maximum(f_M_total, 1e-30))
    # I row
    trans = trans.at[2, 1].set(dist['f_IM'].sum() / jnp.maximum(f_I_total, 1e-30))
    trans = trans.at[2, 2].set(dist['f_II'].sum() / jnp.maximum(f_I_total, 1e-30))
    trans = trans.at[2, 3].set(dist['f_ID'].sum() / jnp.maximum(f_I_total, 1e-30))
    trans = trans.at[2, 4].set(dist['f_IE'].sum() / jnp.maximum(f_I_total, 1e-30))
    # D row
    trans = trans.at[3, 1].set(dist['f_DM'].sum() / jnp.maximum(f_D_total, 1e-30))
    trans = trans.at[3, 2].set(dist['f_DI'].sum() / jnp.maximum(f_D_total, 1e-30))
    trans = trans.at[3, 3].set(dist['f_DD'].sum() / jnp.maximum(f_D_total, 1e-30))
    trans = trans.at[3, 4].set(dist['f_DE'].sum() / jnp.maximum(f_D_total, 1e-30))

    v = params['v']
    pis = params['pi']
    pi_mix = jnp.sum(v[:, None] * pis, axis=0)

    return {
        'log_trans': safe_log(trans),
        'P_domains': dist['P_domains'],
        'pis': pis,
        'domain_weights': v,
        'pi_mix': pi_mix,
    }


def domain_substitution_matrices(params, n_classes, t, precomp=None):
    """Compute per-domain substitution matrices P_n(t) for a specific branch length.

    Unlike effective_pair_hmm (which distills at tau = total pair distance),
    this computes P_n(t) for a single branch, needed for Felsenstein emissions.

    Args:
        params: constrained MixDom parameter dict
        n_classes: number of gamma rate classes
        t: branch length (single branch, not pair total)
        precomp: optional precomputed eigendecompositions

    Returns:
        P_domains: (N, AA, AA) per-domain substitution matrices
    """
    if precomp is not None:
        rate_mults = precomp['rate_mults']
        def domain_P(eigvals, eigvecs, sqrt_pi):
            Ps = vmap(lambda rho: transition_probs_from_eigen(
                eigvals, eigvecs, sqrt_pi, rho * t))(rate_mults)
            return jnp.mean(Ps, axis=0)
        return vmap(domain_P)(
            precomp['eigvals'], precomp['eigvecs'], precomp['sqrt_pi'])
    else:
        S_exch = params['S_exch']
        rate_mults = gamma_rates(params['alpha_gamma'], n_classes)
        def domain_P_full(pi_n):
            Q = _build_rate_matrix_acknowledged(S_exch, pi_n)
            eigvals, eigvecs, sqrt_pi = eigen_decompose(Q, pi_n)
            Ps = vmap(lambda rho: transition_probs_from_eigen(
                eigvals, eigvecs, sqrt_pi, rho * t))(rate_mults)
            return jnp.mean(Ps, axis=0)
        return vmap(domain_P_full)(params['pi'])


# ============================================================
# Machine Boss JSON export
# ============================================================

def _wfst_to_machineboss_json(probs, precision=6):
    """Convert normalized WFST probabilities to Machine Boss JSON format.

    The WFST has states: S, M(a,b), I(x,y), D(x,y), E
    where a,b,x,y are amino acid indices (0..AA-1).

    State IDs:
        S: 0
        M(a,b): 1 + a*AA + b
        I(x,y): 1 + AA*AA + x*AA + y
        D(x,y): 1 + 2*AA*AA + x*AA + y
        E: 1 + 3*AA*AA

    Returns:
        dict in Machine Boss JSON transducer format.
    """
    def _w(val):
        v = float(val)
        return round(v, precision) if v > 0 else 0.0

    states = []

    # Start state S
    s_trans = []
    p_sm = np.asarray(probs['p_sm'])  # (AA, AA)
    p_si = np.asarray(probs['p_si'])  # (AA, AA)
    p_sd = np.asarray(probs['p_sd'])  # (AA, AA)
    p_se = float(probs['p_se'])

    for a in range(AA):
        for b in range(AA):
            w = _w(p_sm[a, b])
            if w > 0:
                s_trans.append({
                    'to': 1 + a * AA + b,
                    'in': AMINO_ACIDS[a],
                    'out': AMINO_ACIDS[b],
                    'weight': w,
                })
    for x in range(AA):
        for y in range(AA):
            w = _w(p_si[x, y])
            if w > 0:
                s_trans.append({
                    'to': 1 + AA * AA + x * AA + y,
                    'out': AMINO_ACIDS[y],
                    'weight': w,
                })
    for x in range(AA):
        for y in range(AA):
            w = _w(p_sd[x, y])
            if w > 0:
                s_trans.append({
                    'to': 1 + 2 * AA * AA + x * AA + y,
                    'in': AMINO_ACIDS[x],
                    'weight': w,
                })
    if p_se > 0:
        s_trans.append({
            'to': 1 + 3 * AA * AA,
            'weight': _w(p_se),
        })
    states.append({'n': 0, 'id': 'S', 'trans': s_trans})

    # Match states M(a,b)
    p_mm = np.asarray(probs['p_mm'])  # (AA,AA,AA,AA)
    p_mi = np.asarray(probs['p_mi'])  # (AA,AA,AA)
    p_md = np.asarray(probs['p_md'])  # (AA,AA,AA)
    p_me = np.asarray(probs['p_me'])  # (AA,AA)

    for a in range(AA):
        for b in range(AA):
            trans = []
            for ap in range(AA):
                for bp in range(AA):
                    w = _w(p_mm[a, b, ap, bp])
                    if w > 0:
                        trans.append({
                            'to': 1 + ap * AA + bp,
                            'in': AMINO_ACIDS[ap],
                            'out': AMINO_ACIDS[bp],
                            'weight': w,
                        })
            for yp in range(AA):
                w = _w(p_mi[a, b, yp])
                if w > 0:
                    trans.append({
                        'to': 1 + AA * AA + a * AA + yp,
                        'out': AMINO_ACIDS[yp],
                        'weight': w,
                    })
            for xp in range(AA):
                w = _w(p_md[a, b, xp])
                if w > 0:
                    trans.append({
                        'to': 1 + 2 * AA * AA + xp * AA + b,
                        'in': AMINO_ACIDS[xp],
                        'weight': w,
                    })
            w = _w(p_me[a, b])
            if w > 0:
                trans.append({
                    'to': 1 + 3 * AA * AA,
                    'weight': w,
                })
            states.append({
                'n': 1 + a * AA + b,
                'id': f'M({AMINO_ACIDS[a]},{AMINO_ACIDS[b]})',
                'trans': trans,
            })

    # Insert states I(x,y)
    p_im = np.asarray(probs['p_im'])  # (AA,AA,AA,AA)
    p_ii = np.asarray(probs['p_ii'])  # (AA,AA,AA)
    p_id = np.asarray(probs['p_id'])  # (AA,AA,AA)
    p_ie = np.asarray(probs['p_ie'])  # (AA,AA)

    for x in range(AA):
        for y in range(AA):
            trans = []
            for ap in range(AA):
                for bp in range(AA):
                    w = _w(p_im[x, y, ap, bp])
                    if w > 0:
                        trans.append({
                            'to': 1 + ap * AA + bp,
                            'in': AMINO_ACIDS[ap],
                            'out': AMINO_ACIDS[bp],
                            'weight': w,
                        })
            for yp in range(AA):
                w = _w(p_ii[x, y, yp])
                if w > 0:
                    trans.append({
                        'to': 1 + AA * AA + x * AA + yp,
                        'out': AMINO_ACIDS[yp],
                        'weight': w,
                    })
            for xp in range(AA):
                w = _w(p_id[x, y, xp])
                if w > 0:
                    trans.append({
                        'to': 1 + 2 * AA * AA + xp * AA + y,
                        'in': AMINO_ACIDS[xp],
                        'weight': w,
                    })
            w = _w(p_ie[x, y])
            if w > 0:
                trans.append({
                    'to': 1 + 3 * AA * AA,
                    'weight': w,
                })
            states.append({
                'n': 1 + AA * AA + x * AA + y,
                'id': f'I({AMINO_ACIDS[x]},{AMINO_ACIDS[y]})',
                'trans': trans,
            })

    # Delete states D(x,y)
    p_dm = np.asarray(probs['p_dm'])  # (AA,AA,AA,AA)
    p_di = np.asarray(probs['p_di'])  # (AA,AA,AA)
    p_dd = np.asarray(probs['p_dd'])  # (AA,AA,AA)
    p_de = np.asarray(probs['p_de'])  # (AA,AA)

    for x in range(AA):
        for y in range(AA):
            trans = []
            for ap in range(AA):
                for bp in range(AA):
                    w = _w(p_dm[x, y, ap, bp])
                    if w > 0:
                        trans.append({
                            'to': 1 + ap * AA + bp,
                            'in': AMINO_ACIDS[ap],
                            'out': AMINO_ACIDS[bp],
                            'weight': w,
                        })
            for yp in range(AA):
                w = _w(p_di[x, y, yp])
                if w > 0:
                    trans.append({
                        'to': 1 + AA * AA + x * AA + yp,
                        'out': AMINO_ACIDS[yp],
                        'weight': w,
                    })
            for xp in range(AA):
                w = _w(p_dd[x, y, xp])
                if w > 0:
                    trans.append({
                        'to': 1 + 2 * AA * AA + xp * AA + y,
                        'in': AMINO_ACIDS[xp],
                        'weight': w,
                    })
            w = _w(p_de[x, y])
            if w > 0:
                trans.append({
                    'to': 1 + 3 * AA * AA,
                    'weight': w,
                })
            states.append({
                'n': 1 + 2 * AA * AA + x * AA + y,
                'id': f'D({AMINO_ACIDS[x]},{AMINO_ACIDS[y]})',
                'trans': trans,
            })

    # End state E (no transitions)
    states.append({
        'n': 1 + 3 * AA * AA,
        'id': 'E',
    })

    return {'state': states}


def _singlet_to_machineboss_json(probs, precision=6):
    """Convert singlet HMM probabilities to Machine Boss JSON format.

    The singlet HMM has states: S, I(x) for x in 0..AA-1, E.
    Total: 2 + AA states.

    For N=2 domains with AA=20: 2 + 20 = 22 states.

    Returns:
        dict in Machine Boss JSON transducer format.
    """
    def _w(val):
        v = float(val)
        return round(v, precision) if v > 0 else 0.0

    sing_trans = np.asarray(probs['singlet_trans'])  # (AA, AA)
    sing_start = np.asarray(probs['singlet_start'])  # (AA,)
    sing_end = np.asarray(probs['singlet_end'])      # (AA,)

    states = []

    # Start state S
    s_trans = []
    for x in range(AA):
        w = _w(sing_start[x])
        if w > 0:
            s_trans.append({
                'to': 1 + x,
                'out': AMINO_ACIDS[x],
                'weight': w,
            })
    # Start -> End (if all start probs are very small)
    se = 1.0 - float(np.sum(sing_start))
    if se > 1e-10:
        s_trans.append({'to': 1 + AA, 'weight': _w(se)})
    states.append({'n': 0, 'id': 'S', 'trans': s_trans})

    # Insert states I(x)
    for x in range(AA):
        trans = []
        for xp in range(AA):
            w = _w(sing_trans[x, xp])
            if w > 0:
                trans.append({
                    'to': 1 + xp,
                    'out': AMINO_ACIDS[xp],
                    'weight': w,
                })
        w = _w(sing_end[x])
        if w > 0:
            trans.append({'to': 1 + AA, 'weight': w})
        states.append({
            'n': 1 + x,
            'id': f'I({AMINO_ACIDS[x]})',
            'trans': trans,
        })

    # End state
    states.append({'n': 1 + AA, 'id': 'E'})

    return {'state': states}
