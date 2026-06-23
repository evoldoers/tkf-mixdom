"""DEPRECATED: re-export shim for `scratch_ggi_cond_kl_quad`.

This file previously contained an alternative ODE integrator that used
`kl_fit` (L-BFGS-B inside the ODE RHS) instead of the closed-form BDI map.
It existed as a workaround for a now-fixed L'Hopital recursion bug in
`scratch_ggi_cond_kl_quad._tkf91_bdi_from_m` (the perturbation was smaller
than the threshold, so the recursive fallback never escaped).

The canonical pure-numpy algebraic integrator is
`scratch_ggi_cond_kl_quad`.  Use it directly in new code.
"""
from scratch_ggi_cond_kl_quad import (  # noqa: F401
    dtheta_dt,
    boundary_condition,
    run_flow,
)
