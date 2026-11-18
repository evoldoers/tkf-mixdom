"""Column summaries, the O(K+A) pair score, and Merge / Extend / Reestimate.

A clade's MSA column c carries a summary s_c = (V'_c, U_c, T_c, C_c, q_c)
(profile_classes.tex, Sections 3-5 and 7). Per class k the bound is
  Phi_k(s) = C + V'.log pi^k + U log r_k - r_k T.(S pi^k),
and the score of a column alone is logsumexp_k(log w_k + Phi_k(s)).
All functions are vectorized over columns.
"""
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tkfmixdom.jax.dp.hmm import _pad_to_bin

A = 20
EPS_SUPPORT = 1e-3
TOPR = 2                     # residues per column considered for the sparse cross term


@dataclass
class Summ:
    Vp: jnp.ndarray   # (L, A)
    U: jnp.ndarray    # (L,)
    T: jnp.ndarray    # (L, A)
    C: jnp.ndarray    # (L,)
    q: jnp.ndarray    # (L, A)

    def take(self, idx):
        idx = jnp.asarray(idx, dtype=jnp.int32)
        return Summ(self.Vp[idx], self.U[idx], self.T[idx], self.C[idx], self.q[idx])

    @staticmethod
    def concat(parts):
        return Summ(*[jnp.concatenate([getattr(p, f) for p in parts]) for f in ("Vp", "U", "T", "C", "q")])


def leaf_summ(seq):
    """Summaries of a single sequence's residues (values 0..19)."""
    e = jax.nn.one_hot(jnp.asarray(seq), A, dtype=jnp.float64)
    L = e.shape[0]
    return Summ(e, jnp.zeros(L), jnp.zeros((L, A)), jnp.zeros(L), e)


def phi(s, sub):
    """(L, K) per-class bounds Phi_k(s)."""
    lP, lR = jnp.log(sub.P), jnp.log(sub.R)
    return s.C[:, None] + s.Vp @ lP.T + s.U[:, None] * lR[None] - sub.R[None] * (s.T @ sub.G.T)


def single_scores(s, sub):
    return jax.nn.logsumexp(jnp.log(sub.w)[None] + phi(s, sub), axis=1)


def _Lmat(sub, t):
    L = jnp.log(jnp.where(sub.S > 0, sub.S, 1.0)) + jnp.log(t)
    return L * (1.0 - jnp.eye(A))


def _topr(q):
    idx = jnp.argsort(-q, axis=1)[:, :TOPR]
    return idx, jnp.take_along_axis(q, idx, axis=1)


@jax.jit
def _match_table(aA, qA, ellA, bB, qB, lP, lR):
    """(LA, LB) match scores log sum_k u_mk v_nk e^{chi_k} + ell_m . q_n (Algorithm 2)."""
    sigma = qA @ qB.T                                        # (LA, LB)
    chi = -sigma[:, :, None] * lR[None, None, :]             # rate cross term
    iA, vA = _topr(qA)
    iB, vB = _topr(qB)
    for r1 in range(TOPR):
        for r2 in range(TOPR):
            same = (iA[:, r1][:, None] == iB[:, r2][None, :])
            prod = vA[:, r1][:, None] * vB[:, r2][None, :]
            keep = same & (vA[:, r1][:, None] > EPS_SUPPORT) & (vB[:, r2][None, :] > EPS_SUPPORT)
            lPx = lP.T[iA[:, r1]]                            # (LA, K) log pi^k at A's residue
            chi = chi - jnp.where(keep, prod, 0.0)[:, :, None] * lPx[:, None, :]
    z = aA[:, None, :] + bB[None, :, :] + chi
    return jax.nn.logsumexp(z, axis=2) + ellA @ qB.T


def match_table(sA, sB, sub, t):
    """Match scores of every column of clade A against every column of clade B at time t."""
    ph_A, ph_B = phi(sA, sub), phi(sB, sub)
    RG = sub.R[None] * (0.5 * t)
    aA = jnp.log(sub.w)[None] + ph_A - RG * (sA.q @ sub.G.T)
    bB = ph_B - RG * (sB.q @ sub.G.T) + jnp.log(sub.R)[None]
    ellA = sA.q @ _Lmat(sub, t)
    return _match_table(aA, sA.q, ellA, bB, sB.q, jnp.log(sub.P), jnp.log(sub.R))


def merge(sA, sB, tA, tB, sub):
    """Single-jump merge of paired columns (Algorithm 4, Merge)."""
    t = tA + tB
    qA, qB = sA.q, sB.q
    off = qA[:, :, None] * qB[:, None, :] * (1.0 - jnp.eye(A))[None]
    return Summ(Vp=sA.Vp + sB.Vp - qA * qB,
                U=sA.U + sB.U + 1.0 - jnp.sum(qA * qB, axis=1),
                T=sA.T + sB.T + 0.5 * t * (qA + qB),
                C=sA.C + sB.C + jnp.einsum("lab,ab->l", off, _Lmat(sub, t)),
                q=qA * qB + (tB / t) * qA * (1 - qB) + (tA / t) * qB * (1 - qA))


def extend(s, t_up):
    """Column present on one side only: no substitution on the branch up to the parent."""
    return Summ(s.Vp, s.U, s.T + t_up * s.q, s.C, s.q)


# ---------------------------------------------------------------- Reestimate

def _divdiff(lam, t):
    li, lj = lam[..., :, None], lam[..., None, :]
    d = li - lj
    close = jnp.abs(d) * t < 1e-8
    safe = jnp.where(close, 1.0, d)
    return jnp.where(close, t * jnp.exp(li * t), (jnp.exp(li * t) - jnp.exp(lj * t)) / safe)


@jax.jit
def _msg(Psi, lam, sq, Lv, t):
    """M(t) L_v per column, and the right projection Psi^T Pi^{1/2} L_v."""
    Lt = jnp.einsum("lai,la->li", Psi, sq * Lv)
    m = jnp.einsum("lai,li->la", Psi, jnp.exp(lam * t) * Lt) / sq
    return m, Lt


@jax.jit
def _branch(Psi, lam, sq, R, Lt, t, C):
    """Accumulate eigensubstitution counts for one branch; return outside message at the child."""
    Rt = jnp.einsum("lai,la->li", Psi, R / sq)
    Z = jnp.sum(Rt * jnp.exp(lam * t) * Lt, axis=1)
    C = C + Rt[:, :, None] * Lt[:, None, :] * _divdiff(lam, t) / Z[:, None, None]
    eta = sq * jnp.einsum("lai,li->la", Psi, jnp.exp(lam * t) * Rt)
    return C, eta / jnp.sum(eta, axis=1, keepdims=True)


def reestimate(res, rows, clade, sub, kstar):
    """Exact posterior statistics of class kstar[c] for each column (Algorithm 7).

    res:   (L, n_rows) residues (0..19) or -1 for gaps (missing data)
    rows:  leaf ids of the columns of res, in order
    clade: {'root': P, 'children': {node: [(child, t), ...]}} restricted to this clade
    """
    L = res.shape[0]
    Lp = _pad_to_bin(L)
    pad = Lp - L
    res_p = np.concatenate([res, np.full((pad, res.shape[1]), -1)]) if pad else res
    k = np.concatenate([np.asarray(kstar), np.zeros(pad, int)]) if pad else np.asarray(kstar)
    Psi, lam = sub.Psi[k], sub.lam[k]
    P = sub.P[k]
    sq = jnp.sqrt(P)
    row_of = {leaf: r for r, leaf in enumerate(rows)}
    ch = clade["children"]
    order, stack = [], [clade["root"]]
    while stack:
        u = stack.pop(); order.append(u); stack.extend(c for c, _ in ch[u])
    Lvec, sig, msgs, Ltil = {}, {}, {}, {}
    for u in reversed(order):                                   # inside pass
        if not ch[u]:
            r = res_p[:, row_of[u]]
            Lu = jnp.where(jnp.asarray(r)[:, None] >= 0,
                           jax.nn.one_hot(jnp.asarray(np.maximum(r, 0)), A, dtype=jnp.float64), 1.0)
            su = jnp.zeros(Lp)
        else:
            Lu, su = jnp.ones((Lp, A)), jnp.zeros(Lp)
            for v, t in ch[u]:
                m, Lt = _msg(Psi, lam, sq, Lvec[v], t)
                msgs[v], Ltil[v] = m, Lt
                Lu = Lu * m
                su = su + sig[v]
        z = jnp.max(Lu, axis=1)
        Lvec[u], sig[u] = Lu / z[:, None], su + jnp.log(z)
    root = clade["root"]
    pL = jnp.sum(P * Lvec[root], axis=1)
    logp = jnp.log(pL) + sig[root]
    q = P * Lvec[root] / pL[:, None]
    Cmat = jnp.zeros((Lp, A, A))
    eta = {root: P}
    for u in order:                                             # outside pass
        for v, t in ch[u]:
            Rv = eta[u]
            for v2, _ in ch[u]:
                if v2 != v:
                    Rv = Rv * msgs[v2]
            Cmat, eta[v] = _branch(Psi, lam, sq, Rv, Ltil[v], t, Cmat)
    Jt = jnp.einsum("lai,lij,lbj->lab", Psi, Cmat, Psi)
    T = jnp.diagonal(Jt, axis1=1, axis2=2)
    N = sub.B[k] * Jt * (1.0 - jnp.eye(A))[None]
    Vp = q + jnp.sum(N, axis=1)
    U = jnp.sum(N, axis=(1, 2))
    lPk, lRk, Rk, Gk = jnp.log(P), jnp.log(sub.R)[k], sub.R[k], sub.G[k]
    C = logp - (jnp.sum(Vp * lPk, axis=1) + U * lRk - Rk * jnp.sum(T * Gk, axis=1))
    s = Summ(Vp, U, T, C, q)
    return s.take(np.arange(L)), logp[:L]
