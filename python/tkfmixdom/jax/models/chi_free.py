"""Build collapsed MixDom transition matrix chi (transnest) from free parameters.

Implements the transnest formula from tkf.tex directly, with ALL parameters
as explicit free inputs. No derived quantities, no constraints.
Suitable for autograd verification of eqs {eq:main-counts}--{eq:var-counts}.

The formula: chi_ij = domexit × T_UV × domenter
                    + delta(U=V) delta(l=m) (pSameDom + delta(X=Y) delta(f=g) pSameFrag)

domenter includes (1-z_T)^{-1} or (1-z_0)^{-1} normalization factors
to condition on the domain being nonempty.
"""

import jax
import jax.numpy as jnp
from ..dp.hmm import safe_log


def build_transnest(tau_0, dom_taus, kappas, notkappas,
                    dom_weights, frag_weights,
                    ext_rates, notext_rates):
    """Build the collapsed Pair HMM transition matrix chi directly.

    Implements the transnest formula from tkf.tex §3.1 (the big table).
    All inputs are free parameters for autograd.

    Args:
        tau_0: (5,5) top-level TKF91 Pair HMM (includes top-level kappa)
        dom_taus: list of K (5,5) per-domain TKF91 Pair HMMs
        kappas: (K,) kappa_k (free, not lambda_k/mu_k)
        notkappas: (K,) 1-kappa_k (free, not 1-kappas)
        dom_weights: (K,) v_k
        frag_weights: (K,F) w_kf
        ext_rates: (K,F) or (K,F,F) fragment extension rates
        notext_rates: (K,F) rho_f = 1-sum_g ext_f,g (free)

    Returns:
        chi: (5KF+2, 5KF+2)
    """
    # Auto-detect and handle MixDom2 (K,F,F) ext_rates
    ext_rates = jnp.asarray(ext_rates)
    if ext_rates.ndim == 2:
        # MixDom1: convert to diagonal (K,F,F)
        ext_rates = jax.vmap(jnp.diag)(ext_rates)
    Si, Mi, Ii, Di, Ei = 0, 1, 2, 3, 4
    K = len(dom_taus)
    F = frag_weights.shape[1]
    N = 5 * K * F + 2

    # Compute T_eff (the 5×5 null-eliminated top-level matrix)
    z_T = jnp.sum(dom_weights * jnp.array([dom_taus[k][Si, Ei] for k in range(K)]))
    z_0 = jnp.sum(dom_weights * notkappas)

    upsilon = jnp.zeros((8, 8))
    for src in range(5):
        if src == Ei:
            continue
        upsilon = upsilon.at[src, Mi].set((1 - z_T) * tau_0[src, Mi])
        upsilon = upsilon.at[src, Ii].set((1 - z_0) * tau_0[src, Ii])
        upsilon = upsilon.at[src, Di].set((1 - z_0) * tau_0[src, Di])
        upsilon = upsilon.at[src, Ei].set(tau_0[src, Ei])
        upsilon = upsilon.at[src, 5].set(z_T * tau_0[src, Mi])
        upsilon = upsilon.at[src, 6].set(z_0 * tau_0[src, Ii])
        upsilon = upsilon.at[src, 7].set(z_0 * tau_0[src, Di])
    for ns, rs in [(5, Mi), (6, Ii), (7, Di)]:
        upsilon = upsilon.at[ns].set(upsilon[rs])

    T_ZZ = upsilon[5:8, 5:8]
    Z_null = jnp.linalg.inv(jnp.eye(3) - T_ZZ)
    T_eff = upsilon[0:5, 0:5] + upsilon[0:5, 5:8] @ Z_null @ upsilon[5:8, 0:5]

    def idx(uv, k, f):
        return 2 + 5 * (k * F + f) + uv

    # Normalization factors for domain entry conditioned on non-empty
    inv_1mzT = 1.0 / (1.0 - z_T)  # for M-type destination
    inv_1mz0 = 1.0 / (1.0 - z_0)  # for I/D-type destination

    chi = jnp.zeros((N, N))

    # --- SS row ---
    for dk in range(K):
        tau_dk = dom_taus[dk]
        for df in range(F):
            for y, y_hmm in enumerate([Mi, Ii, Di]):
                chi = chi.at[0, idx(y, dk, df)].add(
                    T_eff[Si, Mi] * inv_1mzT * dom_weights[dk] * tau_dk[Si, y_hmm] * frag_weights[dk, df])
            chi = chi.at[0, idx(3, dk, df)].add(
                T_eff[Si, Ii] * inv_1mz0 * dom_weights[dk] * kappas[dk] * frag_weights[dk, df])
            chi = chi.at[0, idx(4, dk, df)].add(
                T_eff[Si, Di] * inv_1mz0 * dom_weights[dk] * kappas[dk] * frag_weights[dk, df])
    chi = chi.at[0, 1].set(T_eff[Si, Ei])

    # --- Body rows ---
    for sk in range(K):
        tau_sk = dom_taus[sk]
        for sf in range(F):
            for suv in range(5):
                src = idx(suv, sk, sf)

                if suv <= 2:
                    top_U = Mi
                    x_hmm = [Mi, Ii, Di][suv]
                elif suv == 3:
                    top_U = Ii
                    x_hmm = Ii
                else:
                    top_U = Di
                    x_hmm = Di

                if suv <= 2:
                    dom_exit = notext_rates[sk, sf] * tau_sk[x_hmm, Ei]
                else:
                    dom_exit = notext_rates[sk, sf] * notkappas[sk]

                # Inter-domain
                for dk in range(K):
                    tau_dk = dom_taus[dk]
                    for df in range(F):
                        for y, y_hmm in enumerate([Mi, Ii, Di]):
                            dom_enter = inv_1mzT * dom_weights[dk] * tau_dk[Si, y_hmm] * frag_weights[dk, df]
                            chi = chi.at[src, idx(y, dk, df)].add(
                                dom_exit * T_eff[top_U, Mi] * dom_enter)

                        dom_enter_I = inv_1mz0 * dom_weights[dk] * kappas[dk] * frag_weights[dk, df]
                        chi = chi.at[src, idx(3, dk, df)].add(
                            dom_exit * T_eff[top_U, Ii] * dom_enter_I)
                        chi = chi.at[src, idx(4, dk, df)].add(
                            dom_exit * T_eff[top_U, Di] * dom_enter_I)

                chi = chi.at[src, 1].add(dom_exit * T_eff[top_U, Ei])

                # Same-domain
                if suv <= 2:
                    for df in range(F):
                        for y, y_hmm in enumerate([Mi, Ii, Di]):
                            p_same = notext_rates[sk, sf] * tau_sk[x_hmm, y_hmm] * frag_weights[sk, df]
                            chi = chi.at[src, idx(y, sk, df)].add(p_same)
                elif suv == 3:
                    for df in range(F):
                        p_same = notext_rates[sk, sf] * kappas[sk] * frag_weights[sk, df]
                        chi = chi.at[src, idx(3, sk, df)].add(p_same)
                else:
                    for df in range(F):
                        p_same = notext_rates[sk, sf] * kappas[sk] * frag_weights[sk, df]
                        chi = chi.at[src, idx(4, sk, df)].add(p_same)

                # Fragment extension: ext[d, sf, df] with delta(suv=duv)
                for df in range(F):
                    dst = idx(suv, sk, df)
                    chi = chi.at[src, dst].add(ext_rates[sk, sf, df])

    return chi


def q_collapsed(n_chi, chi):
    """Q = Σ n_ij × log(chi_ij)."""
    return jnp.sum(n_chi * safe_log(chi))


# --- Autograd gradient functions ---

def grad_q_tau_k(n_chi, d, tau_0, dom_taus, kappas, notkappas,
                 dom_weights, frag_weights, ext_rates, notext_rates):
    """dQ/d(tau_k[i,j]) for domain d. Returns (5,5)."""
    def q_fn(tau_k):
        taus = [dom_taus[i] if i != d else tau_k for i in range(len(dom_taus))]
        return q_collapsed(n_chi, build_transnest(tau_0, taus, kappas, notkappas,
                           dom_weights, frag_weights, ext_rates, notext_rates))
    return jax.grad(q_fn)(dom_taus[d])


def grad_q_kappa(n_chi, d, tau_0, dom_taus, kappas, notkappas,
                 dom_weights, frag_weights, ext_rates, notext_rates):
    """dQ/d(kappa_d), treating kappa as free."""
    def q_fn(k_d):
        k = kappas.at[d].set(k_d)
        return q_collapsed(n_chi, build_transnest(tau_0, dom_taus, k, notkappas,
                           dom_weights, frag_weights, ext_rates, notext_rates))
    return jax.grad(q_fn)(kappas[d])


def grad_q_notkappa(n_chi, d, tau_0, dom_taus, kappas, notkappas,
                    dom_weights, frag_weights, ext_rates, notext_rates):
    """dQ/d(notkappa_d), treating notkappa as free."""
    def q_fn(nk_d):
        nk = notkappas.at[d].set(nk_d)
        return q_collapsed(n_chi, build_transnest(tau_0, dom_taus, kappas, nk,
                           dom_weights, frag_weights, ext_rates, notext_rates))
    return jax.grad(q_fn)(notkappas[d])


def grad_q_weight(n_chi, d, tau_0, dom_taus, kappas, notkappas,
                  dom_weights, frag_weights, ext_rates, notext_rates):
    """dQ/d(v_d)."""
    def q_fn(v_d):
        dw = dom_weights.at[d].set(v_d)
        return q_collapsed(n_chi, build_transnest(tau_0, dom_taus, kappas, notkappas,
                           dw, frag_weights, ext_rates, notext_rates))
    return jax.grad(q_fn)(dom_weights[d])


def grad_q_ext(n_chi, k, f, tau_0, dom_taus, kappas, notkappas,
               dom_weights, frag_weights, ext_rates, notext_rates, g=None):
    """dQ/d(ext[k,f,g]), treating ext as free.

    If g is None, returns dQ/d(ext[k,f,:]) as a vector (MixDom2).
    If g is an int, returns dQ/d(ext[k,f,g]) as a scalar.
    """
    ext_rates = jnp.asarray(ext_rates)
    if ext_rates.ndim == 2:
        ext_rates = jax.vmap(jnp.diag)(ext_rates)
    if g is not None:
        def q_fn(r_kfg):
            er = ext_rates.at[k, f, g].set(r_kfg)
            return q_collapsed(n_chi, build_transnest(tau_0, dom_taus, kappas, notkappas,
                               dom_weights, frag_weights, er, notext_rates))
        return jax.grad(q_fn)(ext_rates[k, f, g])
    else:
        def q_fn(r_kf_row):
            er = ext_rates.at[k, f].set(r_kf_row)
            return q_collapsed(n_chi, build_transnest(tau_0, dom_taus, kappas, notkappas,
                               dom_weights, frag_weights, er, notext_rates))
        return jax.grad(q_fn)(ext_rates[k, f])


def grad_q_notext(n_chi, k, f, tau_0, dom_taus, kappas, notkappas,
                  dom_weights, frag_weights, ext_rates, notext_rates):
    """dQ/d(notext[k,f]), treating notext as free."""
    def q_fn(rho_kf):
        nr = notext_rates.at[k, f].set(rho_kf)
        return q_collapsed(n_chi, build_transnest(tau_0, dom_taus, kappas, notkappas,
                           dom_weights, frag_weights, ext_rates, nr))
    return jax.grad(q_fn)(notext_rates[k, f])


def grad_q_tau_0(n_chi, tau_0, dom_taus, kappas, notkappas,
                 dom_weights, frag_weights, ext_rates, notext_rates):
    """dQ/d(tau_0[i,j]). Returns (5,5)."""
    def q_fn(t0):
        return q_collapsed(n_chi, build_transnest(t0, dom_taus, kappas, notkappas,
                           dom_weights, frag_weights, ext_rates, notext_rates))
    return jax.grad(q_fn)(tau_0)
