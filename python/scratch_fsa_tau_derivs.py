"""Verify hand-derived analytic d/dtau and d2/dtau2 of the FSA tau-objective
against the repo's JAX autodiff ground truth (_tkf92_tau_grad/_tkf92_tau_hess).

The other-repo agent (TS/Rust, no autodiff) needs these analytic forms.
We derive them, implement in pure numpy, and check to ~1e-9.
"""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from tkfmixdom.jax.tree.fsa_anneal import (
    _tkf92_expected_ll, _tkf92_tau_grad, _tkf92_tau_hess)
from tkfmixdom.jax.core.ctmc import build_Q_from_S_pi, transition_matrix
from tkfmixdom.jax.core.params import tkf92_trans
from tkfmixdom.jax.core.bdi import tkf_beta, tkf_gamma

S, M, I, D, E = 0, 1, 2, 3, 4


# ---- g(x) = (e^x - 1)/x and derivatives, Taylor-stable near 0 ----
def g_g1_g2(x):
    """Return g, g', g'' of g(x)=(e^x-1)/x."""
    if abs(x) < 1e-2:
        # Taylor: g = sum x^n/(n+1)!
        g = 1 + x/2 + x**2/6 + x**3/24 + x**4/120 + x**5/720
        g1 = 0.5 + x/3 + x**2/8 + x**3/30 + x**4/144
        g2 = 1/3 + x/4 + x**2/10 + x**3/36 + x**4/168
        return g, g1, g2
    ex = np.exp(x)
    g = (ex - 1) / x
    g1 = ((x - 1) * ex + 1) / x**2
    g2 = ((x**2 - 2*x + 2) * ex - 2) / x**3
    return g, g1, g2


def analytic(lam, mu, tau, r, Qsub, pi, N, W):
    """Hand-derived Q'(tau), Q''(tau), and log-space g=tau*Q', h=tau^2*Q''+tau*Q'."""
    kappa = lam / mu
    eps = mu - lam

    # alpha, Phi
    al = np.exp(-mu * tau)
    al1 = -mu * al
    al2 = mu**2 * al
    Phi = 1 - al
    Phi1 = mu * al           # = -al1
    Phi2 = -mu**2 * al       # = -al2

    # rho = g(eps*tau)
    x = eps * tau
    g, g1, g2 = g_g1_g2(x)
    rho = g
    rho1 = eps * g1
    rho2 = eps**2 * g2

    # w = tau*rho
    w = tau * rho
    w1 = rho + tau * rho1
    w2 = 2 * rho1 + tau * rho2

    # beta = lam*w/(mu*w+1)
    Qd = mu * w + 1.0
    beta = lam * w / Qd
    beta1 = lam * w1 / Qd**2
    beta2 = lam * (w2 * Qd - 2 * mu * w1**2) / Qd**3

    # gamma = 1 - (mu/lam)*beta/Phi
    c = mu / lam
    Rr = beta / Phi
    gamma = 1 - c * Rr
    m = beta1 * Phi - beta * Phi1
    Rr1 = m / Phi**2
    gamma1 = -c * Rr1
    mp = beta2 * Phi - beta * Phi2
    Rr2 = (mp * Phi - 2 * m * Phi1) / Phi**3
    gamma2 = -c * Rr2

    # --- atoms A_* (rows S,M,I; use beta), B_* (row D; use gamma) ---
    def prod_atoms(b, b1, b2):
        # returns (M,I,D,E) value/'/'' for a row parameterized by b in {beta,gamma}
        # _M = kappa*alpha*(1-b); _I = b; _D = kappa*Phi*(1-b); _E = (1-kappa)*(1-b)
        M_ = kappa * al * (1 - b)
        M1 = kappa * (al1 * (1 - b) - al * b1)
        M2 = kappa * (al2 * (1 - b) - 2 * al1 * b1 - al * b2)
        I_ = b
        I1 = b1
        I2 = b2
        D_ = kappa * Phi * (1 - b)
        D1 = kappa * (Phi1 * (1 - b) - Phi * b1)
        D2 = kappa * (Phi2 * (1 - b) - 2 * Phi1 * b1 - Phi * b2)
        E_ = (1 - kappa) * (1 - b)
        E1 = -(1 - kappa) * b1
        E2 = -(1 - kappa) * b2
        return (np.array([M_, I_, D_, E_]),
                np.array([M1, I1, D1, E1]),
                np.array([M2, I2, D2, E2]))

    Av, A1, A2 = prod_atoms(beta, beta1, beta2)
    Bv, B1, B2 = prod_atoms(gamma, gamma1, gamma2)

    # --- assemble chi value/'/'' (5x5). cols order in atoms is [M,I,D,E] ---
    cols = [M, I, D, E]
    chi = np.zeros((5, 5)); chiP = np.zeros((5, 5)); chiPP = np.zeros((5, 5))
    # S row: factor 1, no self-loop
    for k, j in enumerate(cols):
        chi[S, j] = Av[k]; chiP[S, j] = A1[k]; chiPP[S, j] = A2[k]
    # M, I rows: (1-r)*A + r on diagonal
    for src in (M, I):
        for k, j in enumerate(cols):
            chi[src, j] = (1 - r) * Av[k] + (r if j == src else 0.0)
            chiP[src, j] = (1 - r) * A1[k]
            chiPP[src, j] = (1 - r) * A2[k]
    # D row: (1-r)*B + r on diagonal
    for k, j in enumerate(cols):
        chi[D, j] = (1 - r) * Bv[k] + (r if j == D else 0.0)
        chiP[D, j] = (1 - r) * B1[k]
        chiPP[D, j] = (1 - r) * B2[k]

    # transition term derivatives
    Tp = 0.0; Tpp = 0.0
    for i in range(5):
        for j in range(5):
            if N[i, j] != 0.0 and chi[i, j] > 0:
                ratio = chiP[i, j] / chi[i, j]
                Tp += N[i, j] * ratio
                Tpp += N[i, j] * (chiPP[i, j] / chi[i, j] - ratio**2)

    # substitution term derivatives: P=exp(Qsub tau), P'=Qsub P, P''=Qsub^2 P
    P = np.asarray(transition_matrix(jnp.asarray(Qsub), tau))
    MP = Qsub @ P
    M2P = Qsub @ Qsub @ P
    Ep = 0.0; Epp = 0.0
    for a in range(P.shape[0]):
        for b in range(P.shape[1]):
            if W[a, b] != 0.0:
                ratio = MP[a, b] / P[a, b]
                Ep += W[a, b] * ratio
                Epp += W[a, b] * (M2P[a, b] / P[a, b] - ratio**2)

    Qprime = Tp + Ep
    Qpp = Tpp + Epp
    g_log = tau * Qprime
    h_log = tau**2 * Qpp + tau * Qprime
    return Qprime, Qpp, g_log, h_log, (beta, gamma)


def make_case(rng, A=20, equal=False):
    pi = rng.dirichlet(np.ones(A))
    Sx = rng.gamma(1.0, size=(A, A)); Sx = (Sx + Sx.T) / 2; np.fill_diagonal(Sx, 0.0)
    Qsub = np.asarray(build_Q_from_S_pi(jnp.asarray(Sx), jnp.asarray(pi)))
    mu = rng.uniform(0.2, 1.5)
    lam = mu if equal else mu * rng.uniform(0.4, 0.95)
    r = rng.uniform(0.1, 0.8)
    tau = rng.uniform(0.05, 3.0)
    # counts only on structurally non-zero chi entries: rows S,M,I,D x cols M,I,D,E
    N = np.zeros((5, 5))
    for i in (S, M, I, D):
        for j in (M, I, D, E):
            N[i, j] = rng.gamma(2.0)
    W = rng.gamma(1.0, size=(A, A)) * (rng.random((A, A)) < 0.5)
    return lam, mu, r, tau, Qsub, pi, N, W


def main():
    rng = np.random.default_rng(0)
    print(f"{'case':>6} {'lam':>6} {'mu':>6} {'tau':>6} "
          f"{'beta_err':>10} {'gam_err':>10} {'g_relerr':>11} {'h_relerr':>11}")
    worst_g = worst_h = 0.0
    for c in range(12):
        equal = (c >= 10)  # last two: lambda == mu (L'Hopital regime)
        lam, mu, r, tau, Qsub, pi, N, W = make_case(rng, equal=equal)
        Qp, Qpp, g_log, h_log, (bz, gz) = analytic(lam, mu, tau, r, Qsub, pi, N, W)

        # ground truth from repo autodiff (log-tau space)
        log_tau = jnp.log(jnp.float64(tau))
        g_ref = float(_tkf92_tau_grad(
            log_tau, jnp.asarray(N), jnp.asarray(W),
            jnp.float64(lam), jnp.float64(mu), jnp.float64(r),
            jnp.asarray(Qsub), jnp.asarray(pi)))
        h_ref = float(_tkf92_tau_hess(
            log_tau, jnp.asarray(N), jnp.asarray(W),
            jnp.float64(lam), jnp.float64(mu), jnp.float64(r),
            jnp.asarray(Qsub), jnp.asarray(pi)))

        # check beta/gamma values vs repo
        b_ref = float(tkf_beta(lam, mu, tau)); g2_ref = float(tkf_gamma(lam, mu, tau))
        be = abs(bz - b_ref); ge = abs(gz - g2_ref)

        # Independent ground truth: finite-difference the objective VALUE
        # (nan-free even at lam=mu, since the forward pass uses the limit form)
        def val(u):
            return float(_tkf92_expected_ll(
                jnp.float64(u), jnp.asarray(N), jnp.asarray(W),
                jnp.float64(lam), jnp.float64(mu), jnp.float64(r),
                jnp.asarray(Qsub), jnp.asarray(pi)))
        eps_fd = 1e-5
        u0 = float(log_tau)
        g_fd = (val(u0 + eps_fd) - val(u0 - eps_fd)) / (2 * eps_fd)
        # Hessian ground truth at lam=mu: difference the (finite) repo GRADIENT,
        # avoiding the nan second-order autodiff entirely.
        def gref(u):
            return float(_tkf92_tau_grad(
                jnp.float64(u), jnp.asarray(N), jnp.asarray(W),
                jnp.float64(lam), jnp.float64(mu), jnp.float64(r),
                jnp.asarray(Qsub), jnp.asarray(pi)))
        h_fd = (gref(u0 + eps_fd) - gref(u0 - eps_fd)) / (2 * eps_fd)
        gre_fd = abs(g_log - g_fd) / (abs(g_fd) + 1e-9)
        hre_fd = abs(h_log - h_fd) / (abs(h_fd) + 1e-9)

        gre = abs(g_log - g_ref) / (abs(g_ref) + 1e-12)
        hre = hre_fd if (np.isnan(h_ref)) else abs(h_log - h_ref) / (abs(h_ref) + 1e-12)
        worst_g = max(worst_g, gre); worst_h = max(worst_h, hre)
        tag = f"  (lam=mu) fd_g={gre_fd:.1e} fd_h={hre_fd:.1e}" if equal else ""
        print(f"{c:>6} {lam:>6.3f} {mu:>6.3f} {tau:>6.3f} "
              f"{be:>10.2e} {ge:>10.2e} {gre:>11.2e} {hre:>11.2e}{tag}")

    print(f"\nworst g rel-err: {worst_g:.2e}   worst h rel-err: {worst_h:.2e}")
    assert worst_g < 1e-7 and worst_h < 1e-6, "MISMATCH"
    print("PASS: analytic grad/hess match autodiff")


if __name__ == "__main__":
    main()
