"""Statistical progressive alignment (profile_classes.tex, Algorithms 1-7).

Guide tree: BioNJ on pairwise times estimated under TKF92 + LG by one EM
iteration, rooted at the midpoint. Each merge: the O(K+A) column-pair match
table, anti-diagonal Viterbi under the MixFrag Pair HMM at t = t_A + t_B,
then Merge/Extend summaries, optionally re-estimated from exact posteriors.
"""
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tkfmixdom.jax.dp.hmm import _pad_to_bin

from . import profile as pf
from . import sparse as sp
from . import wavefront as wf
from .distance import pairwise_times
from .model import load_indel, make_subst
from .tree import guide_tree

AMINO = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AMINO)}
UNKNOWN = -2          # residue present but not one of the 20 (X, B, Z, ...): missing data
GAP = -1
T_MIN = 1e-3


def encode(s):
    return np.array([AA_INDEX.get(c, UNKNOWN) for c in s.upper()], int)


@dataclass
class Profile:
    rows: list          # leaf ids
    res: np.ndarray     # (L, n_rows): residue 0..19, UNKNOWN, or GAP
    summ: pf.Summ       # bound summaries (scoring="bound")
    F: object = None    # (L, K, A) per-class partial likelihoods at the top node (scoring="exact")
    lsc: object = None  # (L,) log scale of F


# ---------------------------------------------------------------- exact scoring

@jax.jit
def _exact_tables(FA, lA, FB, lB, P, w, Mt):
    """Exact column-pair and single-column log-likelihoods under the K-class mixture."""
    left = (w[:, None] * P)[None] * FA                               # (LA, K, A)
    right = jnp.einsum("kxy,lky->lkx", Mt, FB)                       # (LB, K, A)
    tab = jnp.log(jnp.einsum("mkx,nkx->mn", left, right)) + lA[:, None] + lB[None, :]
    sA = jnp.log(jnp.einsum("k,kx,lkx->l", w, P, FA)) + lA
    sB = jnp.log(jnp.einsum("k,kx,lkx->l", w, P, FB)) + lB
    return tab, sA, sB


@jax.jit
def _pp_tables(FA, lA, FB, lB, P, w, J):
    """Posterior-pair scores: log-odds log qA^T (J / pibar pibar^T) qB with the columns' mixture
    posterior tops, plus exact single-column scores. Exact for one class and for single-residue
    columns; for K > 1 it ignores that the two columns share a class (not a bound). O(A) per cell."""
    def col(F, l):
        lk = P[None] * F                                  # (L, K, A)
        pk = lk.sum(2)
        g = w[None] * pk
        sing = jnp.log(g.sum(1)) + l
        q = jnp.einsum("lk,lka->la", g / g.sum(1, keepdims=True), lk / pk[:, :, None])
        return q, sing
    qA, sA = col(FA, lA)
    qB, sB = col(FB, lB)
    pb = w @ P
    lo = jnp.log(qA @ (J / (pb[:, None] * pb[None, :])) @ qB.T)
    return lo + sA[:, None] + sB[None, :], sA, sB


@jax.jit
def _propagate(F, Mt):
    return jnp.einsum("kxy,lky->lkx", Mt, F)


def _renorm(F, lsc):
    z = jnp.max(F, axis=(1, 2))
    return F / z[:, None, None], lsc + jnp.log(z)


def _leaf_profile(leaf, codes, sub):
    known = codes >= 0
    s = pf.leaf_summ(np.where(known, codes, 0))
    if not known.all():
        pib = sub.pibar
        ent = -jnp.sum(pib * jnp.log(pib))
        kn = jnp.asarray(known)[:, None]
        s = pf.Summ(jnp.where(kn, s.Vp, pib[None]), s.U, s.T,
                    jnp.where(jnp.asarray(known), s.C, ent), jnp.where(kn, s.q, pib[None]))
    K = sub.K
    Fl = jnp.where(jnp.asarray(known)[:, None, None],
                   jax.nn.one_hot(jnp.asarray(np.maximum(codes, 0)), 20, dtype=jnp.float64)[:, None, :],
                   1.0) * jnp.ones((1, K, 1))
    return Profile([leaf], codes[:, None].copy(), s, Fl, jnp.zeros(len(codes)))


@partial(jax.jit, static_argnums=(5, 6))
def _viterbi(logT, st, tab, eI, eD, Lx, Ly, Lxr, Lyr):
    return wf.viterbi(logT, st, lambda i, j: tab[i - 1, j - 1], eI, eD, Lx, Ly, Lxr, Lyr)


@partial(jax.jit, static_argnums=(5, 6))
def _match_posteriors(logT, st, tab, eI, eD, Lx, Ly, Lxr, Lyr):
    """Posterior probability that columns i and j are matched (summed over fragtypes):
    the gradient of the log Forward probability with respect to the match table."""
    f = lambda tb: wf.forward(logT, st, lambda i, j: tb[i - 1, j - 1], eI, eD, Lx, Ly, Lxr, Lyr)
    return jax.grad(f)(tab)


def _mea_log_trans(st):
    """Unit transitions (0 in max-plus) for the expected-accuracy Needleman-Wunsch."""
    ns = st.shape[0]
    T = jnp.zeros((ns, ns))
    T = T.at[:, 0].set(wf.NEG_INF)                       # nothing enters S
    return T.at[ns - 1, :].set(wf.NEG_INF)               # nothing leaves E


def _subtree(tree, node):
    ch = tree["children"]
    out, stack = {}, [node]
    while stack:
        u = stack.pop()
        out[u] = ch[u]
        stack.extend(c for c, _ in ch[u])
    return {"root": node, "children": out}


def _scaled(clade, c):
    return {"root": clade["root"],
            "children": {u: [(v, c * t) for v, t in cs] for u, cs in clade["children"].items()}}


def merge_profiles(PA, PB, tA, tB, clade, sub, indel, reestimate=True, decode="viterbi",
                   scoring="bound", sub_scale=1.0):
    """tA, tB are indel-clock branch lengths; substitution uses sub_scale * t (clade is pre-scaled)."""
    t = max(tA + tB, T_MIN)
    c = sub_scale
    LA, LB = PA.res.shape[0], PB.res.shape[0]
    if scoring == "bound":
        tab = pf.match_table(PA.summ, PB.summ, sub, c * t)
        sA, sB = pf.single_scores(PA.summ, sub), pf.single_scores(PB.summ, sub)
    elif scoring == "le":
        # derived log-expectation score: log qA^T (J(t) / pibar pibar^T) qB on the summaries' top
        # distributions (soft-Fitch unless re-estimated); exact log-odds for one chain with posterior
        # tops. Single scores cancel in decoding (every column contributes exactly one).
        sA, sB = pf.single_scores(PA.summ, sub), pf.single_scores(PB.summ, sub)
        pb = sub.pibar
        Rbar = sub.joint(c * t) / (pb[:, None] * pb[None, :])
        tab = jnp.log(PA.summ.q @ Rbar @ PB.summ.q.T) + sA[:, None] + sB[None, :]
    elif scoring == "exact":
        tab, sA, sB = _exact_tables(PA.F, PA.lsc, PB.F, PB.lsc, sub.P, sub.w, sub.M(c * t))
    elif scoring == "sparse":
        tab, sA, sB = sp.tables(PA.F, PA.lsc, PB.F, PB.lsc, sub, c * t)
    elif scoring == "pp":
        tab, sA, sB = _pp_tables(PA.F, PA.lsc, PB.F, PB.lsc, sub.P, sub.w, sub.joint(c * t))
    else:
        raise ValueError(scoring)
    Lx, Ly = _pad_to_bin(LA), _pad_to_bin(LB)
    tabp = jnp.zeros((Lx, Ly)).at[:LA, :LB].set(tab)
    eD = jnp.zeros(Lx + 1).at[1:LA + 1].set(sA)
    eI = jnp.zeros(Ly + 1).at[1:LB + 1].set(sB)
    st = indel.state_types
    if decode == "viterbi":
        _, last, bps = _viterbi(indel.log_trans(t), st, tabp, eI, eD, Lx, Ly, LA, LB)
    elif decode in ("esps", "mea"):
        # expected-SPS decoding ("mea" is a legacy alias): maximize the expected number of correct
        # residue pairs added by this merge, sum over matched column pairs of p_mn * n_m * n_n,
        # with p_mn the posterior that columns m and n are homologous (no gap term: SPS ignores gaps)
        post = _match_posteriors(indel.log_trans(t), st, tabp, eI, eD, Lx, Ly, LA, LB)
        wA = jnp.zeros(Lx).at[:LA].set((PA.res != GAP).sum(1))
        wB = jnp.zeros(Ly).at[:LB].set((PB.res != GAP).sum(1))
        score = post * wA[:, None] * wB[None, :]
        _, last, bps = _viterbi(_mea_log_trans(st), st, score, jnp.zeros(Ly + 1), jnp.zeros(Lx + 1),
                                Lx, Ly, LA, LB)
    else:
        raise ValueError(decode)
    types = wf.traceback(bps, st, int(last), LA, LB, Ly)
    nA, nB = PA.res.shape[1], PB.res.shape[1]
    res, pieces = [], []
    i = j = 0
    for ty in types:
        if ty == wf.M:
            res.append(np.concatenate([PA.res[i], PB.res[j]])); pieces.append(("M", i, j)); i += 1; j += 1
        elif ty == wf.D:
            res.append(np.concatenate([PA.res[i], np.full(nB, GAP)])); pieces.append(("D", i, None)); i += 1
        else:
            res.append(np.concatenate([np.full(nA, GAP), PB.res[j]])); pieces.append(("I", None, j)); j += 1
    res = np.array(res)
    # summaries (bound scoring), vectorized by column type, then restored to column order
    kind = np.array([p[0] for p in pieces])
    if scoring in ("exact", "sparse", "pp"):
        kind = np.array([])
    order, parts = [], []
    mi = np.where(kind == "M")[0]
    if len(mi):
        parts.append(pf.merge(PA.summ.take([pieces[q][1] for q in mi]),
                              PB.summ.take([pieces[q][2] for q in mi]), c * tA, c * tB, sub))
        order += list(mi)
    di = np.where(kind == "D")[0]
    if len(di):
        parts.append(pf.extend(PA.summ.take([pieces[q][1] for q in di]), c * tA)); order += list(di)
    ii = np.where(kind == "I")[0]
    if len(ii):
        parts.append(pf.extend(PB.summ.take([pieces[q][2] for q in ii]), c * tB)); order += list(ii)
    rows = PA.rows + PB.rows
    if scoring in ("exact", "sparse", "pp"):
        FA = _propagate(PA.F, sub.M(max(c * tA, T_MIN / 2)))
        FB = _propagate(PB.F, sub.M(max(c * tB, T_MIN / 2)))
        ia = np.array([p[1] if p[1] is not None else 0 for p in pieces])
        ib = np.array([p[2] if p[2] is not None else 0 for p in pieces])
        hasA = jnp.asarray([p[1] is not None for p in pieces])[:, None, None]
        hasB = jnp.asarray([p[2] is not None for p in pieces])[:, None, None]
        Fn = jnp.where(hasA, FA[ia], 1.0) * jnp.where(hasB, FB[ib], 1.0)
        ls = jnp.where(hasA[:, 0, 0], PA.lsc[ia], 0.0) + jnp.where(hasB[:, 0, 0], PB.lsc[ib], 0.0)
        Fn, ls = _renorm(Fn, ls)
        return Profile(rows, res, None, Fn, ls)
    summ = pf.Summ.concat(parts).take(np.argsort(order))
    if reestimate:
        kstar = np.asarray(jnp.argmax(jnp.log(sub.w)[None] + pf.phi(summ, sub), axis=1))
        summ, _ = pf.reestimate(np.where(res >= 0, res, GAP), rows, clade, sub, kstar)
    return Profile(rows, res, summ)


# ---------------------------------------------------------------- iterative refinement

def _adjacency(tree):
    adj = {}
    for u, cs in tree["children"].items():
        adj.setdefault(u, {})
        for v, t in cs:
            adj[u][v] = t
            adj.setdefault(v, {})[u] = t
    return adj


def _side(adj, u, excl, n_leaves):
    """Leaves reachable from u without crossing to excl."""
    out, stack, seen = [], [u], {u, excl}
    while stack:
        x = stack.pop()
        if x < n_leaves:
            out.append(x)
        for y in adj[x]:
            if y not in seen:
                seen.add(y); stack.append(y)
    return out


def _prune(res, row_of, adj, u, excl, sub, c, n_leaves):
    """Exact per-class partials (L, K, A) and log scales of the columns in res at node u, for the
    part of the tree on u's side of the edge (u, excl); gapped or unknown residues are missing data."""
    K = sub.K
    L = res.shape[0]

    def rec(x, parent):
        if x < n_leaves:
            r = res[:, row_of[x]]
            F = jnp.where(jnp.asarray(r)[:, None, None] >= 0,
                          jax.nn.one_hot(jnp.asarray(np.maximum(r, 0)), 20, dtype=jnp.float64)[:, None, :],
                          1.0) * jnp.ones((1, K, 1))
            return F, jnp.zeros(L)
        F, ls = jnp.ones((L, K, 20)), jnp.zeros(L)
        for y, t in adj[x].items():
            if y == parent:
                continue
            Fy, ly = rec(y, x)
            F = F * _propagate(Fy, sub.M(max(c * t, T_MIN / 2)))
            ls = ls + ly
        return _renorm(F, ls)

    return rec(u, excl)


def _decode_path(tab, sA, sB, t, indel, decode, wA, wB, return_score=False):
    LA, LB = tab.shape
    Lx, Ly = _pad_to_bin(LA), _pad_to_bin(LB)
    tabp = jnp.zeros((Lx, Ly)).at[:LA, :LB].set(tab)
    eD = jnp.zeros(Lx + 1).at[1:LA + 1].set(sA)
    eI = jnp.zeros(Ly + 1).at[1:LB + 1].set(sB)
    st = indel.state_types
    if decode == "viterbi":
        _, last, bps = _viterbi(indel.log_trans(t), st, tabp, eI, eD, Lx, Ly, LA, LB)
    else:
        post = _match_posteriors(indel.log_trans(t), st, tabp, eI, eD, Lx, Ly, LA, LB)
        score = post * jnp.zeros(Lx).at[:LA].set(wA)[:, None] * jnp.zeros(Ly).at[:LB].set(wB)[None, :]
        _, last, bps = _viterbi(_mea_log_trans(st), st, score, jnp.zeros(Ly + 1), jnp.zeros(Lx + 1),
                                Lx, Ly, LA, LB)
        if return_score:
            return wf.traceback(bps, st, int(last), LA, LB, Ly), np.asarray(score)[:LA, :LB]
    return wf.traceback(bps, st, int(last), LA, LB, Ly)


def refine(res, rows, tree, sub, indel, sub_scale, scoring, decode, n_rounds, n_leaves, trace=None):
    """Tree-dependent iterative refinement: for every edge, realign the two sides' sub-alignments
    under the same scoring and decoder, and keep the result. With decode="esps" each step can only
    increase the split's expected SPS (the old alignment is a feasible path)."""
    adj = _adjacency(tree)
    edges = [(u, v, t) for u, cs in tree["children"].items() for v, t in cs]
    row_of = {leaf: r for r, leaf in enumerate(rows)}
    for _ in range(n_rounds):
        changed = 0
        for u, v, tl in edges:
            left = _side(adj, v, u, n_leaves)            # the clade below the edge
            right = _side(adj, u, v, n_leaves)
            if not left or not right:
                continue
            ci = [row_of[x] for x in left]; cj = [row_of[x] for x in right]
            keepA = np.where((res[:, ci] != GAP).any(1))[0]
            keepB = np.where((res[:, cj] != GAP).any(1))[0]
            FA, lA = _prune(res[keepA], row_of, adj, v, u, sub, sub_scale, n_leaves)
            FB, lB = _prune(res[keepB], row_of, adj, u, v, sub, sub_scale, n_leaves)
            t = max(tl, T_MIN)
            if scoring == "exact":
                tab, sA, sB = _exact_tables(FA, lA, FB, lB, sub.P, sub.w, sub.M(sub_scale * t))
            elif scoring == "sparse":
                tab, sA, sB = sp.tables(FA, lA, FB, lB, sub, sub_scale * t)
            elif scoring == "pp":
                tab, sA, sB = _pp_tables(FA, lA, FB, lB, sub.P, sub.w, sub.joint(sub_scale * t))
            else:
                raise ValueError("refinement needs scoring in {exact, sparse, pp} (le: use pp)")
            wA = (res[keepA][:, ci] != GAP).sum(1); wB = (res[keepB][:, cj] != GAP).sum(1)
            if trace is not None and decode == "esps":
                types, sc = _decode_path(tab, sA, sB, t, indel, decode, jnp.asarray(wA), jnp.asarray(wB),
                                         return_score=True)
                # expected cross pairs of the old and new alignments of this split, same posteriors
                posA = {c: i for i, c in enumerate(keepA)}; posB = {c: j for j, c in enumerate(keepB)}
                old = sum(sc[posA[c], posB[c]] for c in range(res.shape[0]) if c in posA and c in posB)
                i2 = j2 = 0; newv = 0.0
                for ty in types:
                    if ty == wf.M:
                        newv += sc[i2, j2]
                    i2 += ty in (wf.M, wf.D); j2 += ty in (wf.M, wf.I)
                trace.append((old, newv))
            else:
                types = _decode_path(tab, sA, sB, t, indel, decode, jnp.asarray(wA), jnp.asarray(wB))
            new = []
            i = j = 0
            for ty in types:
                col = np.full(res.shape[1], GAP)
                if ty in (wf.M, wf.D):
                    col[ci] = res[keepA[i], ci]; i += 1
                if ty in (wf.M, wf.I):
                    col[cj] = res[keepB[j], cj]; j += 1
                new.append(col)
            new = np.array(new)
            if new.shape != res.shape or not np.array_equal(new, res):
                changed += 1
            res = new
        if changed == 0:
            break
    return res


def align(names, seqs, subst="C20", indel="mixfrag_F2", reestimate=True,
          t0=1.0, n_newton=5, tscale=1.0, decode="viterbi", scoring="bound", refine_rounds=0,
          return_tree=False):
    """Align raw sequences (strings). Returns {name: aligned string} (and the tree).

    tscale multiplies the guide-tree branch lengths (estimated under TKF92 + LG)
    before they are used by the aligner's own model, whose time scale may differ
    (each C20 class is normalized to rate 1 within its own profile)."""
    sub = make_subst(subst) if isinstance(subst, str) else subst
    ind = load_indel(indel) if isinstance(indel, str) else indel
    codes = [encode(s) for s in seqs]
    n = len(seqs)
    if n == 1:
        return {names[0]: seqs[0]}
    # distances: TKF92 + LG, whatever the aligner's own model (unknown residues -> most common)
    fill = int(np.argmax(np.asarray(make_subst("LG").pibar)))
    dist_codes = [np.where(c >= 0, c, fill) for c in codes]
    # E-step under TKF92 + LG. The M-step under LG gives the tree and the indel clock (the
    # indel parameters were trained with LG08); the M-step under the aligner's substitution
    # model gives its clock, as a family-level ratio sub_scale applied to substitution times.
    lg = make_subst("LG")
    D, Dsub = pairwise_times(dist_codes, lg, load_indel("tkf92"), t0=t0, n_newton=n_newton,
                             msub=[lg, sub])
    iu = np.triu_indices(n, 1)
    sub_scale = float(np.median(Dsub[iu] / np.maximum(D[iu], 1e-6))) if sub.name != "LG" else 1.0
    sub_scale *= tscale
    tree = guide_tree(D)
    profiles = {}
    stack = [(tree["root"], False)]
    while stack:
        u, done = stack.pop()
        ch = tree["children"][u]
        if not ch:
            profiles[u] = _leaf_profile(u, codes[u], sub)
            continue
        if not done:
            stack.append((u, True))
            stack.extend((c, False) for c, _ in ch)
            continue
        (a, ta), (b, tb) = ch
        profiles[u] = merge_profiles(profiles.pop(a), profiles.pop(b), ta, tb,
                                     _scaled(_subtree(tree, u), sub_scale), sub, ind, reestimate, decode,
                                     scoring, sub_scale)
    P = profiles[tree["root"]]
    res = P.res
    if refine_rounds:
        res = refine(res, P.rows, tree, sub, ind, sub_scale, scoring,
                     "esps" if decode in ("esps", "mea") else decode, refine_rounds, n)
    out = {}
    for r, leaf in enumerate(P.rows):
        chars, pos = [], 0
        for c in res[:, r]:
            if c == GAP:
                chars.append("-")
            else:
                chars.append(seqs[leaf][pos]); pos += 1
        out[names[leaf]] = "".join(chars)
    return (out, tree, D) if return_tree else out
