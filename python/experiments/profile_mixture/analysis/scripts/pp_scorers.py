"""Posterior-pair column scores (harness-compatible), from the identity
  log p(A,B) - log p(A) - log p(B) = log sum_k (gA_k gB_k / w_k) qA^k^T R^k qB^k,
with gamma the class posteriors, q^k the class-k posterior tops, R^k = M^k(t) Pi_k^{-1} (symmetric).
  pp_reest   q = the (re-estimated) summary top, class-marginal ratio J(t)/(pibar pibar^T); O(A)/cell
  pp_mix     q = mixture posterior top from the partials, class-marginal ratio;                O(A)/cell
  pp_sparse  exact identity restricted to the union of each column's top-R classes (drop the rest:
             a lower bound on the exact log-odds);                                               O(|S| A)/cell
All use exact single-column scores, so tab = LO + singA + singB.
"""
import numpy as np


def _col(F, lam, sub):
    P, w = np.asarray(sub.P), np.asarray(sub.w)
    l = P[None] * np.asarray(F)                    # (L, K, A) = pi^k o F^k
    pk = l.sum(2)                                  # (L, K)
    g = w[None] * pk
    sing = np.log(g.sum(1)) + np.asarray(lam)
    g = g / g.sum(1, keepdims=True)                # class posteriors
    qk = l / pk[:, :, None]                        # class-k posterior tops
    qmix = np.einsum("lk,lka->la", g, qk)
    return dict(g=g, qk=qk, qmix=qmix, sing=sing)


def _ratio(sub, t):
    J = np.asarray(sub.joint(t)); pb = np.asarray(sub.pibar)
    return J / (pb[:, None] * pb[None, :])


def pp_reest(sA, sB, FA, lA, FB, lB, sub, t, tA, tB):
    CA, CB = _col(FA, lA, sub), _col(FB, lB, sub)
    lo = np.log(np.asarray(sA.q) @ _ratio(sub, t) @ np.asarray(sB.q).T)
    return lo + CA["sing"][:, None] + CB["sing"][None], CA["sing"], CB["sing"]


def pp_mix(sA, sB, FA, lA, FB, lB, sub, t, tA, tB):
    CA, CB = _col(FA, lA, sub), _col(FB, lB, sub)
    lo = np.log(CA["qmix"] @ _ratio(sub, t) @ CB["qmix"].T)
    return lo + CA["sing"][:, None] + CB["sing"][None], CA["sing"], CB["sing"]


def make_pp_sparse(R=2):
    def pp_sparse(sA, sB, FA, lA, FB, lB, sub, t, tA, tB):
        CA, CB = _col(FA, lA, sub), _col(FB, lB, sub)
        P, w = np.asarray(sub.P), np.asarray(sub.w)
        Rk = np.asarray(sub.M(t)) / P[:, None, :]                     # (K, A, A)
        RqB = np.einsum("kxy,nky->nkx", Rk, CB["qk"])                # (LB, K, A)
        topA = np.argsort(-CA["g"], 1)[:, :R]; topB = np.argsort(-CB["g"], 1)[:, :R]
        K = w.shape[0]
        inA = np.zeros((CA["g"].shape[0], K), bool); np.put_along_axis(inA, topA, True, 1)
        inB = np.zeros((CB["g"].shape[0], K), bool); np.put_along_axis(inB, topB, True, 1)
        S = inA[:, None, :] | inB[None, :, :]                        # (LA, LB, K)
        Ck = np.einsum("mka,nka->mnk", CA["qk"], RqB)                # exact per class (dense here;
        wts = CA["g"][:, None, :] * CB["g"][None, :, :] / w           #  only S is used)
        lo = np.log(np.sum(np.where(S, wts * Ck, 0.0), axis=2))
        return lo + CA["sing"][:, None] + CB["sing"][None], CA["sing"], CB["sing"]
    pp_sparse.__name__ = f"pp_sparse_R{R}"
    return pp_sparse


def pp_summary(sA, sB, FA, lA, FB, lB, sub, t, tA, tB):
    """Posterior-pair log-odds from the summary tops alone (no partials, no pruning): with single-jump
    summaries the tops are the soft-Fitch distributions; with re-estimated ones, class-k* posteriors.
    Single scores are the summary bounds; they cancel in decoding (each column contributes once)."""
    from progalign import profile as pf
    sa = np.asarray(pf.single_scores(sA, sub)); sb = np.asarray(pf.single_scores(sB, sub))
    lo = np.log(np.asarray(sA.q) @ _ratio(sub, t) @ np.asarray(sB.q).T)
    return lo + sa[:, None] + sb[None], sa, sb
