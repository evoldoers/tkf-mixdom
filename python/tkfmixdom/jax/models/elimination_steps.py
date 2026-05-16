"""Step-by-step null elimination chain for MixDom.

Each function takes a transition matrix at one level of elaboration,
eliminates a specific set of null states, and returns:
  - The reduced transition matrix
  - A restoration function that maps counts on the reduced model
    back to counts on the pre-elimination model

The chain of eliminations reduces the fully exploded model to the
collapsed 5NK+2 model. The chain of restorations inverts this exactly.

Each step has T_ZZ = 0 (trivial null closure) except Step 6.
"""

import numpy as np
from ..core.params import S, M, I, D, E, tkf91_trans


def _partition(T, keep_idx, elim_idx):
    """Partition transition matrix into keep/elim blocks."""
    T_KK = T[np.ix_(keep_idx, keep_idx)]
    T_KZ = T[np.ix_(keep_idx, elim_idx)]
    T_ZK = T[np.ix_(elim_idx, keep_idx)]
    T_ZZ = T[np.ix_(elim_idx, elim_idx)]
    return T_KK, T_KZ, T_ZK, T_ZZ


def _eliminate(T, keep_idx, elim_idx):
    """Null-eliminate states in elim_idx.

    Returns (T_reduced, closure_C).
    """
    T_KK, T_KZ, T_ZK, T_ZZ = _partition(T, keep_idx, elim_idx)
    n_elim = len(elim_idx)
    C = np.linalg.inv(np.eye(n_elim) - T_ZZ)
    T_reduced = T_KK + T_KZ @ C @ T_ZK
    return T_reduced, C


def _restore_counts_general(n_reduced, T, keep_idx, elim_idx, C):
    """Vectorized count restoration from reduced to pre-elimination model.

    Uses the ghost-usage formula (ghost-usage.tex, eq:ghost-hmm) to
    distribute counts from the reduced model to the pre-elimination model.

    For C=I (Steps 1-5), this is a simple proportional split.
    For general C (Step 6), uses matrix operations.

    Returns n_full (N_full × N_full) count matrix.
    """
    T_KK, T_KZ, T_ZK, T_ZZ = _partition(T, keep_idx, elim_idx)
    N_full = T.shape[0]
    N_keep = len(keep_idx)
    N_elim = len(elim_idx)

    # Reduced (eliminated) transition matrix
    T_reduced = T_KK + T_KZ @ C @ T_ZK

    # Scale matrix: n / chi (element-wise, with zero where chi=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        Scale = np.where(np.abs(T_reduced) > 1e-30,
                         n_reduced / T_reduced, 0.0)

    # Precompute h-vectors and backward h-vectors for all states
    # H[a, sj] = (C @ T_ZK)[a, sj] = expected flow from elim a to kept sj
    H = C @ T_ZK   # (N_elim, N_keep)

    # For kept→elim counts:
    # n_KZ[si, a] = Σ_sj Scale[si,sj] * T_KZ[si,a] * H[a,sj]
    #             = T_KZ[si,a] * Σ_sj Scale[si,sj] * H[a,sj]
    #             = T_KZ[si,a] * (Scale @ H.T)[si,a]
    SH = Scale @ H.T  # (N_keep, N_elim)
    n_KZ = T_KZ * SH

    # For elim→kept counts:
    # n_ZK[a, sj] = Σ_si Scale[si,sj] * (T_KZ @ C)[si,a] * T_ZK[a,sj]
    # Define Ct[si,a] = (T_KZ @ C)[si,a]
    Ct = T_KZ @ C  # (N_keep, N_elim)
    # n_ZK[a, sj] = T_ZK[a,sj] * Σ_si Scale[si,sj] * Ct[si,a]
    #             = T_ZK[a,sj] * (Ct.T @ Scale)[a,sj]
    CtS = Ct.T @ Scale  # (N_elim, N_keep)
    n_ZK = T_ZK * CtS

    # For elim→elim (ghost counts):
    # G[a,b] = Σ_{si,sj} Scale[si,sj] * Ct[si,a] * T_ZZ[a,b] * H[b,sj]
    # = Σ_a Σ_b [Σ_si Ct[si,a] * Scale_row_si] · T_ZZ[a,b] · [Σ_sj H[b,sj] * Scale_col_sj]
    #
    # Let A[a] = Σ_{si,sj} Ct[si,a] * Scale[si,sj] * H_col_sj_at_a ... this doesn't factor.
    #
    # Actually the formula is:
    # G_ZZ = Σ_{si,sj} Scale[si,sj] * outer(Ct[si,:], H[:,sj]) * T_ZZ  (element-wise)
    #      = (Σ_{si,sj} Scale[si,sj] * Ct[si,:].T @ H[:,sj].T) * T_ZZ  ... no.
    #
    # More carefully:
    # G[a,b] = T_ZZ[a,b] * Σ_{si,sj} Scale[si,sj] * Ct[si,a] * H[b,sj]
    # = T_ZZ[a,b] * (Ct.T @ Scale @ H.T)[a, b]
    # = T_ZZ[a,b] * (CtS @ H.T)[a, b]  ... wait:
    # CtS = Ct.T @ Scale is (N_elim, N_keep)
    # H.T is (N_keep, N_elim)
    # (CtS @ H.T) is ... wait no. CtS is (N_elim, N_keep), H.T is (N_keep, N_elim).
    # Hmm, CtS[a, sj] = Σ_si Ct[si, a] * Scale[si, sj]
    # We want: Σ_{si,sj} Scale[si,sj] * Ct[si,a] * H[b,sj]
    #        = Σ_sj (Σ_si Ct[si,a] * Scale[si,sj]) * H[b,sj]
    #        = Σ_sj CtS[a,sj] * H[b,sj]
    #        = (CtS @ H.T)[a, b]  ... no!
    # H is (N_elim, N_keep), so H.T is (N_keep, N_elim).
    # CtS is (N_elim, N_keep).
    # CtS @ H.T would be (N_elim, N_keep) @ (N_keep, N_elim) = (N_elim, N_elim). ✓
    # And [a,b] = Σ_sj CtS[a,sj] * H.T[sj, b] = Σ_sj CtS[a,sj] * H[b, sj]. ✓

    G_factor = CtS @ H.T  # (N_elim, N_elim)
    n_ZZ = T_ZZ * G_factor

    # Direct keep→keep
    n_KK = Scale * T_KK

    # Assemble into full matrix
    n_full = np.zeros((N_full, N_full))
    n_full[np.ix_(keep_idx, keep_idx)] = n_KK
    n_full[np.ix_(keep_idx, elim_idx)] = n_KZ
    n_full[np.ix_(elim_idx, keep_idx)] = n_ZK
    n_full[np.ix_(elim_idx, elim_idx)] = n_ZZ

    return n_full
