"""LG08 protein exchangeabilities and equilibrium frequencies.

A copy of tkf-dp's src/tkfdp/lg08.py (2026-11-18), for the scripts of this folder.

Numeric values (190 lower-triangle exchangeabilities + 20 frequencies)
copied verbatim from ~/tkf-mixdom/python/tkfmixdom/jax/core/protein.py,
which itself sources them from the published LG matrix in PAML format.

Exposes S, pi in *alphabetical* AA order (ACDEFGHIKLMNPQRSTVWY).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

PAML_ORDER = "ARNDCQEGHILKMFPSTWYV"
ALPHA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

# LG lower-triangle exchangeabilities (190 values) in PAML order.
_LG_S_LOWER = np.array([
    # R
    0.425093,
    # N
    0.276818, 0.751878,
    # D
    0.395144, 0.123954, 5.076149,
    # C
    2.489084, 0.534551, 0.528768, 0.062556,
    # Q
    0.969894, 2.807908, 1.695752, 0.523386, 0.084808,
    # E
    1.038545, 0.363970, 0.541712, 5.243870, 0.003499, 4.128591,
    # G
    2.066040, 0.390192, 1.437645, 0.844926, 0.569265, 0.267959, 0.348847,
    # H
    0.358858, 2.426601, 4.509238, 0.927114, 0.640543, 4.813505, 0.423881, 0.311484,
    # I
    0.149830, 0.126991, 0.191503, 0.010690, 0.320627, 0.072854, 0.044265, 0.008705, 0.108882,
    # L
    0.395337, 0.301848, 0.068427, 0.015076, 0.594007, 0.582457, 0.069673, 0.044261, 0.366317, 4.145067,
    # K
    0.536518, 6.326067, 2.145078, 0.282959, 0.013266, 3.234294, 1.807177, 0.296636, 0.697264, 0.159069, 0.137500,
    # M
    1.124035, 0.484133, 0.371004, 0.025548, 0.893680, 1.672569, 0.173735, 0.139538, 0.442472, 4.273607, 6.312358, 0.656604,
    # F
    0.253701, 0.052722, 0.089525, 0.017416, 1.105251, 0.035855, 0.018811, 0.089586, 0.682139, 1.112727, 2.592692, 0.023918, 1.798853,
    # P
    1.177651, 0.332533, 0.161787, 0.394456, 0.075382, 0.624294, 0.419409, 0.196961, 0.508851, 0.078281, 0.249060, 0.390322, 0.099849, 0.094464,
    # S
    4.727182, 0.858151, 4.008358, 1.240275, 2.784478, 1.223828, 0.611973, 1.739990, 0.990012, 0.064105, 0.182287, 0.748683, 0.346960, 0.361819, 1.338132,
    # T
    2.139501, 0.578987, 2.000679, 0.425860, 1.143480, 1.080136, 0.604545, 0.129836, 0.584262, 1.033739, 0.302936, 1.136863, 2.020366, 0.165001, 0.571468, 6.472279,
    # W
    0.180717, 0.593607, 0.045376, 0.029890, 0.670128, 0.236199, 0.077852, 0.268491, 0.597054, 0.111660, 0.619632, 0.049906, 0.696175, 2.457121, 0.095131, 0.248862, 0.140825,
    # Y
    0.218959, 0.314440, 0.612025, 0.135107, 1.165532, 0.257336, 0.120037, 0.054679, 5.306834, 0.232523, 0.299648, 0.131932, 0.481306, 7.803902, 0.089613, 0.400547, 0.245841, 3.151815,
    # V
    2.547870, 0.170887, 0.083688, 0.037967, 1.959291, 0.210332, 0.245034, 0.076701, 0.119013, 10.649107, 1.702745, 0.185202, 1.898718, 0.654683, 0.296501, 0.098369, 2.188158, 0.189510, 0.249313,
])

_LG_PI = np.array([
    0.079066, 0.055941, 0.041977, 0.053052, 0.012937,
    0.040767, 0.071586, 0.057337, 0.022355, 0.062157,
    0.099081, 0.064600, 0.022951, 0.042302, 0.044040,
    0.061197, 0.053287, 0.012066, 0.034155, 0.069147,
])


def _lower_tri_to_matrix(s_values: np.ndarray, n: int = 20) -> np.ndarray:
    expected = n * (n - 1) // 2
    assert len(s_values) == expected
    S = np.zeros((n, n))
    idx = 0
    for i in range(1, n):
        for j in range(i):
            S[i, j] = s_values[idx]
            S[j, i] = s_values[idx]
            idx += 1
    return S


def _paml_to_alpha_perm() -> np.ndarray:
    return np.array([PAML_ORDER.index(aa) for aa in ALPHA_ORDER])


def get_lg08():
    """Return (S_alpha, pi_alpha) in alphabetical AA order (ACDEFGHIKLMNPQRSTVWY).

    S_alpha is the (20, 20) symmetric exchangeability matrix.
    pi_alpha sums to 1.

    The single-site rate matrix is built by `build_single_site_Q`.
    """
    S_paml = _lower_tri_to_matrix(_LG_S_LOWER)
    pi_paml = _LG_PI / _LG_PI.sum()
    perm = _paml_to_alpha_perm()
    S_alpha = S_paml[perm][:, perm]
    pi_alpha = pi_paml[perm]
    return S_alpha, pi_alpha


def build_single_site_Q(S: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Build the GTR rate matrix Q[i,j] = S[i,j] * pi[j] for i != j,
    diagonals set so rows sum to 0, then normalized so the mean rate
    (-sum_i pi_i Q_ii) equals 1.
    """
    Q = S * pi[None, :]
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    mean_rate = -float(np.sum(pi * np.diag(Q)))
    return Q / mean_rate


# Cached LG08 in alphabetical order, plus the singleton GTR rate matrix.
S_LG08, PI_LG08 = get_lg08()
Q_LG08 = build_single_site_Q(S_LG08, PI_LG08)

# F81 exchangeability: the symmetric matrix S' such that Q_LG08[x, y] =
# S'[x, y] * pi[y] off-diagonal, with mean rate 1 at PI_LG08. This is
# the form used in main.tex \S2 ('Q^s = eta_s * S_{xx'} * pi(x')').
# It differs from the published S_LG08 by the LG08 mean-rate normalizer
# (S_LG08_F81 = S_LG08 / mean_rate_at_pi_LG08 ~ S_LG08 / 0.894).
def _S_F81_from_Q(Q: np.ndarray, pi: np.ndarray) -> np.ndarray:
    Sf = Q / pi[None, :]
    np.fill_diagonal(Sf, 0.0)
    Sf = 0.5 * (Sf + Sf.T)   # numerically symmetrize
    return Sf

S_LG08_F81 = _S_F81_from_Q(Q_LG08, PI_LG08)

# JAX-typed exports for downstream JIT use.
S_LG08_J = jnp.asarray(S_LG08)             # published paper coefficients
S_LG08_F81_J = jnp.asarray(S_LG08_F81)     # F81 form (rate-1-normalized)
PI_LG08_J = jnp.asarray(PI_LG08)
Q_LG08_J = jnp.asarray(Q_LG08)
Q_LG08_J = jnp.asarray(Q_LG08)
