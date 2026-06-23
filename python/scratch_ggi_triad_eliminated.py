"""Helper module: null-elimination of the ID state from the 13x13 Triad.

The ODE integrator entry points (`dtheta_dt`, `run_flow`,
`boundary_condition`) are re-exported from
`scratch_ggi_cond_kl_quad` -- this file previously held an alternative
kl_fit-based implementation that existed as a workaround for a now-fixed
L'Hopital recursion bug in cond_kl_quad.

What remains here is the legitimate ID-null-elimination machinery, used
by `triad_counts_eliminated`:

    Q_{ij} = Q'_{ij} + Q'_{i, ID} * Q'_{ID, j} / (1 - Q'_{ID, ID})

Since B' has no nonzero ID row (z_{II}, z_{ID}, z_{DD} etc. have no dt
derivative), Q'_{ID, :} = A'_{ID, :} exactly, and Q'_{ID, ID} = A'_{ID, ID}
exactly.  So the elimination splits cleanly into:

    U_{ij} = A'_{ij} + A'_{i, ID} * A'_{ID, j} / (1 - A'_{ID, ID})
    V_{ij} = B'_{ij} + B'_{i, ID} * A'_{ID, j} / (1 - A'_{ID, ID})

with U, V the 12x12 (no ID) matrices.  `coarse_grain_12` then collapses
to the 5-state pair-HMM via the CORRECTED coarse-graining map (which
differs from `scratch_ggi_triad_algebraic.coarse_grain` for sI/mI/MD --
see the docstring there for the discrepancy).
"""
import sys
import numpy as np
sys.path.insert(0, '/Users/yam/tkf-mixdom/python')

from scratch_ggi_triad_algebraic import build_AB

# Re-export the canonical ODE entry points
from scratch_ggi_cond_kl_quad import (  # noqa: F401
    dtheta_dt,
    boundary_condition,
    run_flow,
)

S, M, I, D, E = 0, 1, 2, 3, 4
SS_, sI_, MM_, mI_, MD_, IM_, iI_, ID_, Ds_, Dm_, Di_, Dd_, EE_ = range(13)

# Coarse-graining per the user's appendix spec (see sec:comp-triad-hmm):
#   {SS, MM, EE} -> {S, M, E};
#   {sI, mI, IM, iI} -> I;
#   {MD, Ds, Dm, Di, Dd} -> D
# (MD is not explicitly listed in the user's spec but per the same logic --
#  entering MD means the second component went to D, so a D-column is
#  emitted -- it goes to D.  The OLD C_map at scratch_ggi_triad_algebraic.py
#  retained the first component, which is wrong for descendant-alignment
#  counting.)
# Indices [SS, sI, MM, mI, MD, IM, iI, Ds, Dm, Di, Dd, EE]
KEEP_13 = [i for i in range(13) if i != ID_]  # 12 indices, sorted ascending
CMAP_12 = [S, I, M, I, D, I, I, D, D, D, D, E]


def null_eliminate_ID(A_full, B_full, atol=1e-12):
    """Eliminate the ID state from the 13x13 (A', B') to get 12x12 (U, V)."""
    # Sanity: B' should have no nonzero ID row
    if not np.allclose(B_full[ID_, :], 0, atol=atol):
        raise ValueError(f"B' has nonzero ID row: {B_full[ID_, :]}")
    g = A_full[ID_, ID_]
    one_minus_g = 1.0 - g
    if abs(one_minus_g) < 1e-14:
        raise ValueError(f"1 - A'_{{ID,ID}} = {one_minus_g}, can't eliminate")
    A_iID = A_full[KEEP_13, ID_]   # column over kept indices
    A_IDj = A_full[ID_, KEEP_13]   # row over kept indices
    B_iID = B_full[KEEP_13, ID_]
    A_keep = A_full[np.ix_(KEEP_13, KEEP_13)]
    B_keep = B_full[np.ix_(KEEP_13, KEEP_13)]
    U = A_keep + np.outer(A_iID, A_IDj) / one_minus_g
    V = B_keep + np.outer(B_iID, A_IDj) / one_minus_g
    return U, V


def coarse_grain_12(n12):
    """Coarse-grain a 12x12 transition count to 5x5 (S, M, I, D, E)."""
    m = np.zeros((5, 5))
    for a in range(12):
        for b in range(12):
            m[CMAP_12[a], CMAP_12[b]] += n12[a, b]
    return m


def triad_counts_eliminated(lam_T, mu_T, r, t, lam0, mu0, x_ins, y_del):
    """Compute coarse-grained 5x5 m^(0) and Delta m from the null-eliminated Triad.

    Returns (m0, dm) where m0 is the surrogate's own TKF92 counts and dm is
    the leading-dt perturbation from the GGI step.
    """
    Afull, Bfull, _Y = build_AB(lam_T, mu_T, r, lam0, mu0, x_ins, y_del, t)
    U, V = null_eliminate_ID(Afull, Bfull)
    SS_idx_12 = 0   # SS is still index 0 in the kept ordering
    EE_idx_12 = 11  # EE is now index 11 (was 12) after removing ID (index 7)
    C = np.linalg.inv(np.eye(12) - U)
    CVC = C @ V @ C
    # n^(0)_{ab} = C[SS, a] * U[a, b] * C[b, EE]
    n0 = np.outer(C[SS_idx_12, :], C[:, EE_idx_12]) * U
    # n^(1)_{ab} = (CVC)[SS,a]*U[a,b]*C[b,EE] + C[SS,a]*V[a,b]*C[b,EE] + C[SS,a]*U[a,b]*(CVC)[b,EE]
    n1 = (np.outer(CVC[SS_idx_12, :], C[:, EE_idx_12]) * U
          + np.outer(C[SS_idx_12, :], C[:, EE_idx_12]) * V
          + np.outer(C[SS_idx_12, :], CVC[:, EE_idx_12]) * U)
    m0 = coarse_grain_12(n0)
    m1 = coarse_grain_12(n1)
    return m0, m1


if __name__ == "__main__":
    print("ODE entry points re-export from scratch_ggi_cond_kl_quad.\n"
          "Run that file's __main__ for the canonical self-test.")
