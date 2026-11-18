"""Pairwise evolutionary times by one EM iteration (profile_classes.tex, Algorithm 6).

E-step: the expected transition counts n and aligned-pair counts W at t0 are the
gradients of the log Forward probability with respect to log tau and log J.
M-step: a fixed number of Newton-Raphson steps on u = log t for
  Q(t) = sum n_XY log tau_XY(t) + sum W_ab log J_ab(t).
Pairs are bucketed by lengths padded to the geometric grid of tkfmixdom and
processed with jax.vmap; the Newton steps are vectorized over all pairs.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tkfmixdom.jax.dp.hmm import _pad_to_bin

from . import wavefront as wf

T_MIN, T_MAX = 1e-3, 10.0


def _pair_logZ(logT, logJ, logpib, x, y, Lx_real, Ly_real, st, Lx, Ly):
    eM = lambda i, j: logJ[x[i - 1], y[j - 1]]
    eI = jnp.concatenate([jnp.zeros(1), logpib[y]])
    eD = jnp.concatenate([jnp.zeros(1), logpib[x]])
    return wf.forward(logT, st, eM, eI, eD, Lx, Ly, Lx_real, Ly_real)


@partial(jax.jit, static_argnums=(8, 9))
def _counts_batch(logT, logJ, logpib, xs, ys, Lxs, Lys, st, Lx, Ly):
    g = jax.grad(_pair_logZ, argnums=(0, 1))
    return jax.vmap(g, in_axes=(None, None, None, 0, 0, 0, 0, None, None, None))(
        logT, logJ, logpib, xs, ys, Lxs, Lys, st, Lx, Ly)


def _make_newton(indel, sub, n_newton, mask):
    def Q(u, n, W):
        t = jnp.exp(u)
        tau = indel.trans(t)
        logT = jnp.where(mask, jnp.log(jnp.where(mask, tau, 1.0)), 0.0)
        return jnp.sum(n * logT) + jnp.sum(W * jnp.log(sub.joint(t)))

    g = jax.grad(Q)
    h = jax.grad(g)

    def one(u0, n, W):
        def step(u, _):
            gg, hh = g(u, n, W), h(u, n, W)
            du = jnp.where(hh < 0, -gg / hh, gg)
            return jnp.clip(u + jnp.clip(du, -1.0, 1.0), np.log(T_MIN), np.log(T_MAX)), None
        u, _ = jax.lax.scan(step, u0, None, length=n_newton)
        return u

    return jax.jit(jax.vmap(one, in_axes=(None, 0, 0)))


def pairwise_times(seqs, sub, indel, t0=1.0, n_newton=5, max_cells=3e7, msub=None):
    """(N, N) matrix of estimated times for integer-encoded sequences (values 0..19).

    The E-step uses `sub` (LG in the aligner). The M-step maximizes the expected
    complete-data log-likelihood, with the alignment as the latent data, under
    `msub` if given: its aligned-pair marginal J_ab(t) is exact, so a profile
    mixture can set the time scale without changing the E-step."""
    N = len(seqs)
    st = indel.state_types
    logT0 = indel.log_trans(t0)
    logJ0 = jnp.log(sub.joint(t0))
    logpib = jnp.log(sub.pibar)
    buckets = {}
    for a in range(N):
        for b in range(a + 1, N):
            x, y = (a, b) if len(seqs[a]) <= len(seqs[b]) else (b, a)
            key = (_pad_to_bin(len(seqs[x])), _pad_to_bin(len(seqs[y])))
            buckets.setdefault(key, []).append((a, b, x, y))
    ns = int(st.shape[0])
    n_all, W_all, idx = [], [], []
    for (Lx, Ly), items in sorted(buckets.items()):
        # memory guard: checkpointed carries are ~ (Lx+Ly) * (min(Lx,Ly)+1) * ns per pair
        per = (Lx + Ly) * (min(Lx, Ly) + 1) * ns
        chunk = max(1, int(max_cells // per))
        for c in range(0, len(items), chunk):
            part = items[c:c + chunk]
            xs = np.zeros((len(part), Lx), int)
            ys = np.zeros((len(part), Ly), int)
            for r, (_, _, x, y) in enumerate(part):
                xs[r, :len(seqs[x])] = seqs[x]
                ys[r, :len(seqs[y])] = seqs[y]
            Lxs = jnp.array([len(seqs[x]) for (_, _, x, _) in part])
            Lys = jnp.array([len(seqs[y]) for (_, _, _, y) in part])
            n, W = _counts_batch(logT0, logJ0, logpib, jnp.asarray(xs), jnp.asarray(ys),
                                 Lxs, Lys, st, Lx, Ly)
            n_all.append(np.asarray(n)); W_all.append(np.asarray(W))
            idx += [(a, b) for (a, b, _, _) in part]
    if not idx:
        z = np.zeros((N, N))
        return [z for _ in msub] if isinstance(msub, (list, tuple)) else z
    mask = indel.trans(t0) > 0
    n_cat, W_cat = jnp.asarray(np.concatenate(n_all)), jnp.asarray(np.concatenate(W_all))
    msubs = msub if isinstance(msub, (list, tuple)) else [msub if msub is not None else sub]
    Ds = []
    for ms in msubs:
        u = np.asarray(_make_newton(indel, ms, n_newton, mask)(jnp.log(t0), n_cat, W_cat))
        Dm = np.zeros((N, N))
        for (a, b), uu in zip(idx, u):
            Dm[a, b] = Dm[b, a] = float(np.exp(uu))
        Ds.append(Dm)
    return Ds if isinstance(msub, (list, tuple)) else Ds[0]
