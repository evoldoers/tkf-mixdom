"""Empirical protein substitution models: WAG and LG.

Rate matrices are stored as lower-triangle exchangeability parameters
and equilibrium frequencies. The rate matrix is Q_ij = S_ij * pi_j
for i != j, with diagonal set so rows sum to zero, then normalized
so the expected rate equals 1.

Amino acid order: ARNDCQEGHILKMFPSTWYV (standard PAML order).

Values from:
- WAG: Whelan & Goldman (2001) Mol Biol Evol 18:691-699
- LG: Le & Gascuel (2008) Mol Biol Evol 25:1307-1320
"""

import jax.numpy as jnp
import numpy as np


AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

# WAG lower-triangle exchangeabilities (190 values, row by row)
# Row i (1..19), entries j (0..i-1)
_WAG_S_LOWER = np.array([
    # R
    0.551571,
    # N
    0.509848, 0.635346,
    # D
    0.738998, 0.147304, 5.429420,
    # C
    1.027040, 0.528191, 0.265256, 0.0302949,
    # Q
    0.908598, 3.035500, 1.543640, 0.616783, 0.0988179,
    # E
    1.582850, 0.439157, 0.947198, 6.174160, 0.021352, 5.469470,
    # G
    1.416720, 0.584665, 1.125560, 0.865584, 0.306674, 0.330052, 0.567717,
    # H
    0.316954, 2.137150, 3.956290, 0.930676, 0.248972, 4.294110, 0.570025, 0.249410,
    # I
    0.193335, 0.186979, 0.554236, 0.039437, 0.170135, 0.113917, 0.127395, 0.0304501, 0.138190,
    # L
    0.397915, 0.497671, 0.131528, 0.0848047, 0.384287, 0.869489, 0.154263, 0.0613037, 0.499462, 3.170970,
    # K
    0.906265, 5.351420, 3.012010, 0.479855, 0.0740339, 3.894900, 2.584430, 0.373558, 0.890432, 0.323832, 0.257555,
    # M
    0.893496, 0.683162, 0.198221, 0.103754, 0.390482, 1.545260, 0.315124, 0.174100, 0.404141, 4.257460, 4.854020, 0.934276,
    # F
    0.210494, 0.102711, 0.0961621, 0.0467304, 0.398020, 0.0999208, 0.0811339, 0.049931, 0.679371, 1.059470, 2.115170, 0.088836, 1.190630,
    # P
    1.438550, 0.679489, 0.195081, 0.423984, 0.109404, 0.933372, 0.682355, 0.243570, 0.696198, 0.0999288, 0.415844, 0.556896, 0.171329, 0.161444,
    # S
    3.370790, 1.224190, 3.974230, 1.071760, 1.407660, 1.028870, 0.704939, 1.341820, 0.740169, 0.319440, 0.344739, 0.967130, 0.493905, 0.545931, 1.613280,
    # T
    2.121110, 0.554413, 2.030060, 0.374866, 0.512984, 0.857928, 0.822765, 0.225833, 0.473307, 1.458160, 0.326622, 1.386980, 1.516120, 0.171903, 0.795384, 4.378020,
    # W
    0.113133, 1.163920, 0.0719167, 0.129767, 0.717070, 0.215737, 0.156557, 0.336983, 0.262569, 0.212483, 0.665309, 0.137505, 0.515706, 1.529640, 0.139405, 0.523742, 0.110864,
    # Y
    0.240735, 0.381533, 1.086000, 0.325711, 0.543833, 0.227710, 0.196303, 0.103604, 3.873440, 0.420170, 0.398618, 0.133264, 0.428437, 6.454280, 0.216046, 0.786993, 0.291148, 2.485390,
    # V
    2.006010, 0.251849, 0.196246, 0.152335, 1.002140, 0.301281, 0.588731, 0.187247, 0.118358, 7.821300, 1.800340, 0.305434, 2.058450, 0.649892, 0.314887, 0.232739, 1.388230, 0.365369, 0.314730,
])

# WAG equilibrium frequencies
_WAG_PI = np.array([
    0.0866279, 0.043972, 0.0390894, 0.0570451, 0.0193078,
    0.0367281, 0.0580589, 0.0832518, 0.0244313, 0.048466,
    0.086209, 0.0620286, 0.0195027, 0.0384319, 0.0457631,
    0.0695179, 0.0610127, 0.0143859, 0.0352742, 0.0708956,
])

# LG lower-triangle exchangeabilities (190 values)
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

# LG equilibrium frequencies
_LG_PI = np.array([
    0.079066, 0.055941, 0.041977, 0.053052, 0.012937,
    0.040767, 0.071586, 0.057337, 0.022355, 0.062157,
    0.099081, 0.064600, 0.022951, 0.042302, 0.044040,
    0.061197, 0.053287, 0.012066, 0.034155, 0.069147,
])


def _lower_tri_to_matrix(s_values, n=20):
    """Convert lower-triangle values to symmetric matrix."""
    expected = n * (n - 1) // 2
    if len(s_values) != expected:
        raise ValueError(f"Expected {expected} values, got {len(s_values)}")
    S = np.zeros((n, n))
    idx = 0
    for i in range(1, n):
        for j in range(i):
            S[i, j] = s_values[idx]
            S[j, i] = s_values[idx]
            idx += 1
    return S


def _paml_to_alphabetical_perm():
    """Permutation to reorder from PAML (AA_ORDER) to alphabetical (io.AMINO_ACIDS).

    Returns perm such that Q_alpha = Q_paml[perm][:, perm] and pi_alpha = pi_paml[perm].
    """
    from ..util.io import AMINO_ACIDS
    return np.array([AA_ORDER.index(aa) for aa in AMINO_ACIDS])


_PERM = None  # cached permutation


def _get_perm():
    global _PERM
    if _PERM is None:
        _PERM = _paml_to_alphabetical_perm()
    return _PERM


def _build_rate_matrix(s_values, pi_values):
    """Build normalized rate matrix from exchangeabilities and frequencies.

    The raw data is in PAML amino acid order (ARNDCQEGHILKMFPSTWYV).
    The returned Q and pi are permuted to alphabetical order (ACDEFGHIKLMNPQRSTVWY)
    matching io.AMINO_ACIDS, so they can be used directly with seq_to_int().
    """
    S = _lower_tri_to_matrix(s_values)
    pi = pi_values / pi_values.sum()
    Q = S * pi[None, :]
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    mean_rate = -np.sum(pi * np.diag(Q))
    Q = Q / mean_rate

    # Permute from PAML order to alphabetical order
    perm = _get_perm()
    Q = Q[perm][:, perm]
    pi = pi[perm]

    return jnp.array(Q), jnp.array(pi)


def rate_matrix_wag():
    """WAG protein substitution rate matrix.

    Returns Q and pi in alphabetical amino acid order (ACDEFGHIKLMNPQRSTVWY),
    matching io.AMINO_ACIDS and seq_to_int(). Raw data is stored in PAML order
    and permuted automatically.

    Returns:
        Q: (20, 20) rate matrix (normalized to 1 substitution per unit time)
        pi: (20,) equilibrium frequencies
    """
    return _build_rate_matrix(_WAG_S_LOWER, _WAG_PI)


def rate_matrix_lg():
    """LG protein substitution rate matrix.

    Returns Q and pi in alphabetical amino acid order (ACDEFGHIKLMNPQRSTVWY),
    matching io.AMINO_ACIDS and seq_to_int(). Raw data is stored in PAML order
    and permuted automatically.

    Returns:
        Q: (20, 20) rate matrix (normalized to 1 substitution per unit time)
        pi: (20,) equilibrium frequencies
    """
    return _build_rate_matrix(_LG_S_LOWER, _LG_PI)


def rate_matrix_lg21(ins_rate=0.03, del_rate=0.03):
    """LG08 extended to 21 states (20 AA + gap) with mean-field indel model.

    Gap is state index 20. The rate matrix is:
        Q21[a, b]   = LG08 Q[a, b]         for a, b in 0..19 (AA substitution)
        Q21[a, 20]  = del_rate              for a in 0..19    (residue → gap)
        Q21[20, a]  = ins_rate * pi_lg[a]   for a in 0..19    (gap → residue)
        Q21[20, 20] = -(sum of gap→residue rates)
        Q21[a, a]   adjusted so rows sum to 0

    The equilibrium of this 21-state model has:
        pi21[a] = pi_lg[a] * ins_rate / (ins_rate + del_rate)  for a in 0..19
        pi21[20] = del_rate / (ins_rate + del_rate)

    Normalized so mean rate = 1 over the 21-state equilibrium.

    Args:
        ins_rate: rate of gap → residue (insertion). Default 0.02.
        del_rate: rate of residue → gap (deletion). Default 0.04.

    Returns:
        Q21: (21, 21) rate matrix
        pi21: (21,) equilibrium frequencies
    """
    Q20, pi20 = rate_matrix_lg()
    Q20 = np.asarray(Q20)
    pi20 = np.asarray(pi20)

    Q21 = np.zeros((21, 21))
    # AA block: copy LG08
    Q21[:20, :20] = Q20
    # Deletion: residue a → gap
    for a in range(20):
        Q21[a, 20] = del_rate
    # Insertion: gap → residue a
    for a in range(20):
        Q21[20, a] = ins_rate * pi20[a]
    # Fix diagonals
    np.fill_diagonal(Q21, 0.0)
    np.fill_diagonal(Q21, -Q21.sum(axis=1))

    # Equilibrium
    pi21 = np.zeros(21)
    kappa = ins_rate / (ins_rate + del_rate)
    pi21[:20] = pi20 * kappa
    pi21[20] = 1.0 - kappa
    pi21 = pi21 / pi21.sum()

    # Normalize to mean rate 1
    mean_rate = -np.sum(pi21 * np.diag(Q21))
    if mean_rate > 0:
        Q21 = Q21 / mean_rate

    return jnp.array(Q21), jnp.array(pi21)


def lg_exchangeability():
    """LG exchangeability matrix S and equilibrium pi in alphabetical order.

    Returns S (symmetric, zero diagonal) and pi, both permuted to match
    io.AMINO_ACIDS ordering.  Useful as a prior for GTR M-steps.

    Returns:
        S: (20, 20) symmetric exchangeability matrix
        pi: (20,) equilibrium frequencies
    """
    S_paml = _lower_tri_to_matrix(_LG_S_LOWER)
    pi_paml = _LG_PI / _LG_PI.sum()
    perm = _get_perm()
    S = S_paml[np.ix_(perm, perm)]
    pi = pi_paml[perm]
    return S, pi
