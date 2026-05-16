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
    # Row 1 (R): A-R
    0.551571,
    # Row 2 (N): A-N, R-N
    0.509848, 0.635346,
    # Row 3 (D): A-D, R-D, N-D
    0.738998, 0.147304, 5.429420,
    # Row 4 (C): A-C, R-C, N-C, D-C
    1.027040, 0.528191, 0.265256, 0.0302949,
    # Row 5 (Q): A-Q, R-Q, N-Q, D-Q, C-Q
    0.908598, 3.035500, 1.582850, 0.439157, 0.947198,
    # Row 6 (E): A-E, R-E, N-E, D-E, C-E, Q-E
    1.582850, 0.439157, 0.947198, 6.174160, 0.021352, 5.469470,
    # Row 7 (G): A-G, R-G, N-G, D-G, C-G, Q-G, E-G
    1.416720, 0.523386, 1.736520, 0.554236, 0.084808, 0.523386, 2.058450,
    # Row 8 (H): A-H, R-H, N-H, D-H, C-H, Q-H, E-H, G-H
    0.316258, 2.137150, 3.370790, 0.352742, 0.0951682, 1.437640, 0.478005, 0.233397,
    # Row 9 (I): A-I, R-I, N-I, D-I, C-I, Q-I, E-I, G-I, H-I
    0.193335, 0.186979, 0.554236, 0.039437, 0.170135, 0.113855, 0.025965, 0.028906, 0.286027,
    # Row 10 (L): A-L, R-L, N-L, D-L, C-L, Q-L, E-L, G-L, H-L, I-L
    0.397358, 0.497671, 0.131528, 0.012937, 0.313311, 0.368739, 0.049906, 0.028906, 0.439157, 3.151300,
    # Row 11 (K): A-K, R-K, N-K, D-K, C-K, Q-K, E-K, G-K, H-K, I-K, L-K
    0.906265, 5.351420, 3.012010, 0.568853, 0.072854, 2.064840, 1.003450, 0.325711, 0.152335, 0.086209, 0.280717,
    # Row 12 (M): A-M, R-M, N-M, D-M, C-M, Q-M, E-M, G-M, H-M, I-M, L-M, K-M
    0.893496, 0.683162, 0.198221, 0.0302949, 0.619951, 0.748683, 0.089525, 0.087791, 0.313311, 3.170970, 4.257460, 0.303676,
    # Row 13 (F): A-F, R-F, N-F, D-F, C-F, Q-F, E-F, G-F, H-F, I-F, L-F, K-F, M-F
    0.210494, 0.102711, 0.096457, 0.008068, 0.953164, 0.077852, 0.009860, 0.084808, 0.579672, 0.789745, 2.115170, 0.042610, 1.190630,
    # Row 14 (P): A-P, R-P, N-P, D-P, C-P, Q-P, E-P, G-P, H-P, I-P, L-P, K-P, M-P, F-P
    1.438550, 0.312261, 0.138293, 0.397358, 0.364434, 0.612025, 0.340156, 0.243768, 0.534551, 0.063274, 0.283464, 0.172206, 0.044265, 0.044265,
    # Row 15 (S): A-S, R-S, N-S, D-S, C-S, Q-S, E-S, G-S, H-S, I-S, L-S, K-S, M-S, F-S, P-S
    4.509480, 0.934276, 3.881900, 1.240750, 2.006010, 0.925860, 0.601972, 1.437010, 0.714830, 0.111570, 0.232523, 0.625272, 0.236199, 0.413600, 1.355610,
    # Row 16 (T): A-T, R-T, N-T, D-T, C-T, Q-T, E-T, G-T, H-T, I-T, L-T, K-T, M-T, F-T, P-T, S-T
    2.000540, 0.656604, 2.000540, 0.778142, 0.601972, 0.484018, 0.346983, 0.383684, 0.439157, 1.117590, 0.216046, 0.655045, 0.515706, 0.090855, 0.549570, 4.378020,
    # Row 17 (W): A-W, R-W, N-W, D-W, C-W, Q-W, E-W, G-W, H-W, I-W, L-W, K-W, M-W, F-W, P-W, S-W, T-W
    0.113855, 0.717840, 0.064563, 0.023787, 0.867763, 0.239248, 0.042610, 0.152335, 0.494887, 0.136906, 0.817316, 0.035454, 0.454571, 1.177650, 0.064563, 0.361625, 0.165473,
    # Row 18 (Y): A-Y, R-Y, N-Y, D-Y, C-Y, Q-Y, E-Y, G-Y, H-Y, I-Y, L-Y, K-Y, M-Y, F-Y, P-Y, S-Y, T-Y, W-Y
    0.195510, 0.156542, 0.341068, 0.103400, 0.568853, 0.208836, 0.044265, 0.057861, 4.435700, 0.267828, 0.348131, 0.088836, 0.320627, 6.312580, 0.148483, 0.469395, 0.204141, 2.121110,
    # Row 19 (V): A-V, R-V, N-V, D-V, C-V, Q-V, E-V, G-V, H-V, I-V, L-V, K-V, M-V, F-V, P-V, S-V, T-V, W-V, Y-V
    2.386260, 0.228116, 0.062556, 0.088836, 1.164410, 0.117132, 0.157001, 0.185133, 0.247847, 7.821300, 1.049870, 0.156542, 2.020060, 0.569265, 0.214717, 0.525096, 2.370130, 0.267828, 0.234601,
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
    # Row 1 (R)
    0.425093,
    # Row 2 (N)
    0.276818, 0.751878,
    # Row 3 (D)
    0.395144, 0.123954, 5.076149,
    # Row 4 (C)
    2.489084, 0.534551, 0.528768, 0.062556,
    # Row 5 (Q)
    0.969894, 2.807908, 1.038545, 0.363970, 0.746078,
    # Row 6 (E)
    1.038545, 0.363970, 0.746078, 5.243870, 0.084329, 5.115644,
    # Row 7 (G)
    2.066040, 0.390894, 1.437645, 0.554236, 0.075382, 0.594093, 2.547870,
    # Row 8 (H)
    0.358858, 2.137150, 3.038533, 0.312261, 0.006334, 1.506500, 0.528768, 0.306475,
    # Row 9 (I)
    0.149830, 0.109261, 0.528768, 0.042610, 0.308635, 0.126991, 0.001800, 0.021543, 0.236199,
    # Row 10 (L)
    0.395144, 0.528768, 0.100872, 0.006613, 0.320627, 0.350230, 0.058654, 0.018625, 0.468199, 3.088510,
    # Row 11 (K)
    0.906265, 5.351420, 3.148580, 0.569265, 0.072854, 2.006569, 1.137630, 0.336355, 0.122346, 0.068674, 0.277724,
    # Row 12 (M)
    0.893496, 0.691268, 0.245034, 0.006613, 0.691268, 0.811614, 0.095382, 0.066236, 0.304803, 3.277830, 4.257460, 0.285078,
    # Row 13 (F)
    0.210494, 0.145482, 0.065314, 0.003218, 0.897871, 0.089525, 0.006613, 0.062556, 0.645560, 0.829175, 2.106910, 0.046730, 1.190630,
    # Row 14 (P)
    1.438550, 0.368739, 0.164126, 0.410886, 0.393379, 0.666506, 0.367902, 0.233397, 0.483768, 0.050644, 0.312261, 0.205711, 0.050644, 0.035454,
    # Row 15 (S)
    4.509480, 0.887753, 3.681060, 1.169970, 2.137150, 1.003450, 0.544060, 1.595430, 0.611973, 0.131528, 0.267828, 0.665585, 0.247847, 0.364434, 1.341820,
    # Row 16 (T)
    2.000540, 0.530324, 2.000540, 0.679371, 0.739772, 0.402941, 0.252167, 0.336355, 0.428437, 1.059470, 0.196258, 0.604070, 0.515706, 0.090855, 0.564432, 4.378020,
    # Row 17 (W)
    0.113855, 0.869489, 0.049906, 0.006613, 0.911370, 0.247103, 0.006613, 0.167042, 0.540027, 0.157001, 0.868166, 0.035454, 0.506734, 1.289460, 0.049906, 0.306905, 0.152335,
    # Row 18 (Y)
    0.195510, 0.124630, 0.324525, 0.109261, 0.649361, 0.244157, 0.028906, 0.044265, 4.813505, 0.208836, 0.332517, 0.076701, 0.320627, 6.312580, 0.148483, 0.456190, 0.171995, 2.370130,
    # Row 19 (V)
    2.386260, 0.186979, 0.062556, 0.068674, 1.173890, 0.117132, 0.174845, 0.188182, 0.222455, 7.821300, 1.129560, 0.137505, 2.020060, 0.569265, 0.249060, 0.582457, 2.370130, 0.268491, 0.257336,
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
