"""Sparse-class lower bound on the exact column-pair log-likelihood.

Designed and adversarially verified in the bound-design study (see the profile-class note): for each
class k, the exact term E_k(m, n) = l_A^T K^k l_B with l = pi^k o F^k and K^k = M^k(t) Pi_k^{-1}
(symmetric, nonnegative) is split into "top residue + rest" on both sides:
  E_k = v_A r_B(x*_A) + v_B r_A(y*_B) - v_A v_B K(x*_A, y*_B) + a2^T K b2,
where the first three terms are exact and a2^T K b2 >= max(|a2| min_x (K b2)(x), |b2| min_y (K a2)(y)).
The residual maximum is taken after summing over classes (sum_k max >= max sum). On the union S of
each column's top-R classes (by class posterior) the exact E_k replaces its lower bound. The result is
a genuine lower bound, exact for one-hot (leaf) columns, at 5K + |S|(A + O(1)) operations per cell
instead of KA. Single-column scores are exact.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

BLOCK = 64


@partial(jax.jit, static_argnums=(5,))
def column_stats(F, lsc, P, M, w, R):
    Kmat = M / P[:, None, :]                                     # (K, A, A) symmetric
    K = P.shape[0]
    kk = jnp.arange(K)[None, :]
    l = P[None] * F                                              # (L, K, A)
    r = jnp.einsum("kxy,lky->lkx", M, F)
    p = l.sum(2)
    xs = jnp.argmax(l, axis=2)                                   # (L, K)
    v = jnp.take_along_axis(l, xs[:, :, None], 2)[:, :, 0]
    rest = r - Kmat[kk, :, xs] * v[:, :, None]
    return dict(l=l, r=r, xs=xs, v=v, m=p - v, lo2=jnp.maximum(rest.min(2), 0.0),
                ctop=jnp.argsort(-(w[None] * p), axis=1)[:, :R], lam=lsc, p=p)   # class posterior order


@jax.jit
def _block(SAb, SB, w, Kmat):
    """Table rows for a block of A columns."""
    K = w.shape[0]
    kk = jnp.arange(K)
    xA, vA, rA, lA = SAb["xs"], SAb["v"], SAb["r"], SAb["l"]
    xB, vB, rB = SB["xs"], SB["v"], SB["r"]
    rB_at = jnp.transpose(rB[:, kk[None, :], xA], (1, 0, 2))            # (bA, LB, K): r_B(x*_A)
    rA_at = rA[:, kk[None, :], xB]                                       # (bA, LB, K): r_A(y*_B)
    kab = Kmat[kk[None, None, :], xA[:, None, :], xB[None, :, :]]         # (bA, LB, K)
    core_k = vA[:, None, :] * rB_at + vB[None] * rA_at - vA[:, None, :] * vB[None] * kab
    ra_k = SAb["m"][:, None, :] * SB["lo2"][None]
    rb_k = SB["m"][None] * SAb["lo2"][:, None, :]
    # union of top-R classes: mask of cells x classes that get the exact term
    inS = (jax.nn.one_hot(SAb["ctop"], K).sum(1)[:, None, :] + jax.nn.one_hot(SB["ctop"], K).sum(1)[None]) > 0
    E = jnp.einsum("mka,nka->mnk", lA, rB)                                # exact, used only on S
    notS = ~inS
    core = jnp.sum(jnp.where(notS, w * core_k, 0.0), axis=2)
    ra = jnp.sum(jnp.where(notS, w * ra_k, 0.0), axis=2)
    rb = jnp.sum(jnp.where(notS, w * rb_k, 0.0), axis=2)
    exactS = jnp.sum(jnp.where(inS, w * E, 0.0), axis=2)
    tot = core + jnp.maximum(jnp.maximum(ra, rb), 0.0) + exactS
    return jnp.log(tot) + SAb["lam"][:, None] + SB["lam"][None]


def tables(FA, lA, FB, lB, sub, t, R=2):
    """(tab, singA, singB): sparse-class lower bound on the match table, exact singles."""
    P, w = sub.P, sub.w
    M = sub.M(t)
    SA = column_stats(FA, lA, P, M, w, R)
    SB = column_stats(FB, lB, P, M, w, R)
    Kmat = M / P[:, None, :]
    LA = FA.shape[0]
    rows = []
    for s in range(0, LA, BLOCK):
        e = min(s + BLOCK, LA)
        pad = BLOCK - (e - s)
        SAb = {k: (jnp.concatenate([v[s:e], jnp.repeat(v[e - 1:e], pad, axis=0)]) if pad else v[s:e])
               for k, v in SA.items()}
        rows.append(_block(SAb, SB, w, Kmat)[: e - s])
    tab = jnp.concatenate(rows, axis=0)
    singA = jnp.log(SA["p"] @ w) + lA
    singB = jnp.log(SB["p"] @ w) + lB
    return tab, singA, singB
