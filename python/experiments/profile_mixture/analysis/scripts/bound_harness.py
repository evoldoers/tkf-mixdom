"""Harness for comparing column-pair scoring functions against exact per-class pruning.

A scorer is a function  score(sA, sB, FA, lA, FB, lB, sub, t, tA, tB) -> (tab, singA, singB):
tab (LA, LB) log-scores for every column pair and singA (LA,), singB (LB,) the scores of each
column on its own (the Pair HMM's delete/insert emissions); a scorer may return tab alone, in
which case the single scores default to profile.single_scores of the summaries. where sA, sB are profile.Summ summaries of the two
clades' columns (top node at the clade root), FA/FB (L, K, A) and lA/lB (L,) are the exact
per-class partials and their log scales (available for building references, NOT to be used
by an O(K+A) scorer), and t = tA + tB is the time between the two clade tops.

Cases:
  leaf_<subst>_t<t>   all 20 x 20 single-residue pairs
  bali_<fam>_<mode>   the two clades at the guide-tree root of a BAliBASE family, with
                      columns taken from the reference alignment, summaries built bottom-up
                      along the clade trees (mode = single: single-jump merges only;
                      reest: re-estimated from exact posteriors at every internal node)
Metrics per case: max(tab - exact) (must be <= 0 for a valid lower bound) and the mean raw gap
(exact - tab); then, on the log-odds scale LO = tab - singA - singB that drives alignment
decisions: sd of (LO_exact - LO), Spearman correlation over cells, and top-1 agreement: the
fraction of reference-matched A columns whose best-scoring B column (by LO) is the reference
partner, under the scorer vs under exact.

Usage (python3.12, JAX_ENABLE_X64=1):
  from bound_harness import build_cases, evaluate, current_bound
  cases = build_cases(); print(evaluate(current_bound, cases))
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import progalign  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from progalign import profile as pf  # noqa: E402
from progalign.align import (GAP, _exact_tables, _match_posteriors, _mea_log_trans,  # noqa: E402
                                   _propagate, _renorm, _viterbi, encode)
from progalign import wavefront as wf  # noqa: E402
from tkfmixdom.jax.dp.hmm import _pad_to_bin  # noqa: E402
from progalign.distance import pairwise_times  # noqa: E402
from progalign.model import load_indel, make_subst  # noqa: E402
from progalign.tree import guide_tree  # noqa: E402
from tkfmixdom.util.msa_benchmark import parse_fasta  # noqa: E402

BALI = Path.home() / "bio-datasets" / "data" / "balibase" / "bali3pdbm"
DEFAULT_FAMILIES = ["BB11001", "BB11003", "BB11004", "BB11005", "BB12001", "BB12003",
                    "BB20001", "BB20002", "BB30001", "BB40001", "BB50001"]


def exact_scorer(sA, sB, FA, lA, FB, lB, sub, t, tA, tB):
    return tuple(np.asarray(x) for x in _exact_tables(FA, lA, FB, lB, sub.P, sub.w, sub.M(t)))


def current_bound(sA, sB, FA, lA, FB, lB, sub, t, tA, tB):
    return np.asarray(pf.match_table(sA, sB, sub, t))


def _leaf_case(sub, t):
    x = np.arange(20)
    s = pf.leaf_summ(x)
    F = jax.nn.one_hot(jnp.asarray(x), 20, dtype=jnp.float64)[:, None, :] * jnp.ones((1, sub.K, 1))
    return dict(name=f"leaf_{sub.name}_t{t}", sub=sub, sA=s, sB=s, FA=F, lA=jnp.zeros(20), FB=F,
                lB=jnp.zeros(20), t=t, tA=t / 2, tB=t / 2, ref_pairs=[(a, a) for a in range(20)])


def _clade_columns(res_all, leaves):
    """Columns of the reference alignment restricted to `leaves` (drop all-gap columns)."""
    sub = res_all[:, leaves]
    keep = np.where((sub != GAP).any(1))[0]
    return sub[keep], keep


def _build(node, tree, res_all, sub, reest):
    """Summaries and exact partials of the columns of node's clade, bottom-up along the tree."""
    ch = tree["children"][node]
    if not ch:
        res, keep = _clade_columns(res_all, [node])
        codes = res[:, 0]
        known = codes >= 0
        s = pf.leaf_summ(np.maximum(codes, 0))
        F = jnp.where(jnp.asarray(known)[:, None, None],
                      jax.nn.one_hot(jnp.asarray(np.maximum(codes, 0)), 20, dtype=jnp.float64)[:, None, :],
                      1.0) * jnp.ones((1, sub.K, 1))
        return dict(leaves=[node], keep=keep, s=s, F=F, l=jnp.zeros(len(codes)))
    (a, ta), (b, tb) = ch
    A = _build(a, tree, res_all, sub, reest)
    B = _build(b, tree, res_all, sub, reest)
    leaves = A["leaves"] + B["leaves"]
    res, keep = _clade_columns(res_all, leaves)
    posA = {c: i for i, c in enumerate(A["keep"])}
    posB = {c: i for i, c in enumerate(B["keep"])}
    ia = [posA.get(c) for c in keep]
    ib = [posB.get(c) for c in keep]
    parts, order = [], []
    M = [k for k in range(len(keep)) if ia[k] is not None and ib[k] is not None]
    D = [k for k in range(len(keep)) if ia[k] is not None and ib[k] is None]
    I = [k for k in range(len(keep)) if ia[k] is None and ib[k] is not None]
    if M:
        parts.append(pf.merge(A["s"].take([ia[k] for k in M]), B["s"].take([ib[k] for k in M]), ta, tb, sub)); order += M
    if D:
        parts.append(pf.extend(A["s"].take([ia[k] for k in D]), ta)); order += D
    if I:
        parts.append(pf.extend(B["s"].take([ib[k] for k in I]), tb)); order += I
    s = pf.Summ.concat(parts).take(np.argsort(order))
    clade = {"root": node, "children": {}}
    stack = [node]
    while stack:
        u = stack.pop(); clade["children"][u] = tree["children"][u]; stack.extend(c for c, _ in tree["children"][u])
    if reest:
        kstar = np.asarray(jnp.argmax(jnp.log(sub.w)[None] + pf.phi(s, sub), axis=1))
        s, _ = pf.reestimate(res, leaves, clade, sub, kstar)
    FA = _propagate(A["F"], sub.M(max(ta, 5e-4)))
    FB = _propagate(B["F"], sub.M(max(tb, 5e-4)))
    hasA = jnp.asarray([x is not None for x in ia])[:, None, None]
    hasB = jnp.asarray([x is not None for x in ib])[:, None, None]
    ia0 = np.array([x if x is not None else 0 for x in ia]); ib0 = np.array([x if x is not None else 0 for x in ib])
    F = jnp.where(hasA, FA[ia0], 1.0) * jnp.where(hasB, FB[ib0], 1.0)
    l = jnp.where(hasA[:, 0, 0], A["l"][ia0], 0.0) + jnp.where(hasB[:, 0, 0], B["l"][ib0], 0.0)
    F, l = _renorm(F, l)
    return dict(leaves=leaves, keep=keep, s=s, F=F, l=l)


def _bali_case(fam, sub, reest):
    ref = parse_fasta(str(BALI / "ref" / fam))
    names = list(ref)
    L = len(ref[names[0]])
    res_all = np.full((L, len(names)), GAP)
    for r, n in enumerate(names):
        for c, ch in enumerate(ref[n]):
            if ch not in "-.":
                code = encode(ch)[0]
                res_all[c, r] = code if code >= 0 else 0
    seqs = [np.array([encode(ch)[0] for ch in ref[n] if ch not in "-."]) for n in names]
    seqs = [np.where(s >= 0, s, 0) for s in seqs]
    D = pairwise_times(seqs, make_subst("LG"), load_indel("tkf92"), msub=sub)
    tree = guide_tree(D)
    (a, ta), (b, tb) = tree["children"][tree["root"]]
    A = _build(a, tree, res_all, sub, reest)
    B = _build(b, tree, res_all, sub, reest)
    posB = {c: j for j, c in enumerate(B["keep"])}
    ref_pairs = [(i, posB[c]) for i, c in enumerate(A["keep"]) if c in posB]
    return dict(name=f"bali_{fam}_{sub.name}_{'reest' if reest else 'single'}", sub=sub,
                sA=A["s"], sB=B["s"], FA=A["F"], lA=A["l"], FB=B["F"], lB=B["l"],
                t=max(ta + tb, 1e-3), tA=ta, tB=tb, ref_pairs=ref_pairs)


CACHE = Path(os.environ.get("BOUND_HARNESS_CACHE",
                            str(Path(tempfile.gettempdir()) / "bound_cases.pkl")))


def load_cases():
    """Cached cases (build_cases() writes the cache)."""
    import pickle
    raw = pickle.loads(CACHE.read_bytes())
    subs = {k: make_subst(k) for k in ("C20", "LG")}
    for c in raw:
        c["sub"] = subs[c.pop("sub_name")]
        c["sA"] = pf.Summ(*[jnp.asarray(x) for x in c["sA"]]); c["sB"] = pf.Summ(*[jnp.asarray(x) for x in c["sB"]])
        for k in ("FA", "lA", "FB", "lB"):
            c[k] = jnp.asarray(c[k])
    return raw


def _save_cases(cases):
    import pickle
    out = []
    for c in cases:
        d = {k: v for k, v in c.items() if k not in ("sub", "sA", "sB", "FA", "lA", "FB", "lB")}
        d["sub_name"] = c["sub"].name
        d["sA"] = [np.asarray(getattr(c["sA"], f)) for f in ("Vp", "U", "T", "C", "q")]
        d["sB"] = [np.asarray(getattr(c["sB"], f)) for f in ("Vp", "U", "T", "C", "q")]
        for k in ("FA", "lA", "FB", "lB"):
            d[k] = np.asarray(c[k])
        out.append(d)
    CACHE.write_bytes(pickle.dumps(out))


def build_cases(families=None, substs=("C20", "LG"), leaf_ts=(0.3, 1.0, 2.0, 4.0), reest_modes=(True, False)):
    cases = []
    for sk in substs:
        sub = make_subst(sk)
        for t in leaf_ts:
            cases.append(_leaf_case(sub, t))
        for fam in (families or DEFAULT_FAMILIES):
            if not (BALI / "ref" / fam).exists():
                continue
            for reest in reest_modes:
                cases.append(_bali_case(fam, sub, reest))
    _save_cases(cases)
    return cases


def _call(scorer, c):
    out = scorer(c["sA"], c["sB"], c["FA"], c["lA"], c["FB"], c["lB"], c["sub"], c["t"], c["tA"], c["tB"])
    if isinstance(out, tuple):
        return tuple(np.asarray(x) for x in out)
    return (np.asarray(out), np.asarray(pf.single_scores(c["sA"], c["sub"])),
            np.asarray(pf.single_scores(c["sB"], c["sub"])))


_INDEL = None


def dp_recall(tab, sA, sB, t, ref_pairs, decode):
    """Merge the two clades with the Pair HMM DP (MixFrag F=2) using these scores; return the
    fraction of reference column pairs recovered."""
    global _INDEL
    if _INDEL is None:
        _INDEL = load_indel("mixfrag_F2")
    ind = _INDEL
    LA, LB = tab.shape
    Lx, Ly = _pad_to_bin(LA), _pad_to_bin(LB)
    tabp = jnp.zeros((Lx, Ly)).at[:LA, :LB].set(jnp.asarray(tab))
    eD = jnp.zeros(Lx + 1).at[1:LA + 1].set(jnp.asarray(sA))
    eI = jnp.zeros(Ly + 1).at[1:LB + 1].set(jnp.asarray(sB))
    st = ind.state_types
    if decode == "viterbi":
        _, last, bps = _viterbi(ind.log_trans(t), st, tabp, eI, eD, Lx, Ly, LA, LB)
    else:
        post = _match_posteriors(ind.log_trans(t), st, tabp, eI, eD, Lx, Ly, LA, LB)
        _, last, bps = _viterbi(_mea_log_trans(st), st, post, jnp.zeros(Ly + 1), jnp.zeros(Lx + 1), Lx, Ly, LA, LB)
    types = wf.traceback(bps, st, int(last), LA, LB, Ly)
    i = j = 0
    got = set()
    for ty in types:
        if ty == wf.M:
            got.add((i, j)); i += 1; j += 1
        elif ty == wf.D:
            i += 1
        else:
            j += 1
    return len(got & set(ref_pairs)) / max(1, len(ref_pairs))


def evaluate(scorer, cases, verbose=True, dp=True):
    rows = []
    for c in cases:
        ex, exA, exB = _call(exact_scorer, c)
        sc, sA, sB = _call(scorer, c)
        lo_ex = ex - exA[:, None] - exB[None, :]
        lo = sc - sA[:, None] - sB[None, :]
        rp = c["ref_pairs"]
        top_sc = np.mean([np.argmax(lo[i]) == j for i, j in rp]) if rp else np.nan
        top_ex = np.mean([np.argmax(lo_ex[i]) == j for i, j in rp]) if rp else np.nan
        r = dict(case=c["name"], cells=int(sc.size), t=round(float(c["t"]), 3),
                 max_excess=float(np.max(sc - ex)), mean_gap=float(np.mean(ex - sc)),
                 sd_gap=float(np.std(lo_ex - lo)), lo_bias=float(np.mean(lo_ex - lo)),
                 spearman=float(spearmanr(lo_ex.ravel(), lo.ravel())[0]), top1=float(top_sc), top1_exact=float(top_ex))
        if dp and c["name"].startswith("bali"):
            for dec in ("viterbi", "mea"):
                r[f"recall_{dec}"] = dp_recall(sc, sA, sB, c["t"], rp, dec)
                r[f"recall_{dec}_exact"] = dp_recall(ex, exA, exB, c["t"], rp, dec)
        rows.append(r)
        if verbose:
            print("{case:34s} t={t:<6} cells={cells:<7d} max_excess={max_excess:+.2e} gap={mean_gap:6.2f} "
                  "LO-bias={lo_bias:5.2f} LO-sd={sd_gap:5.2f} rho={spearman:.3f}".format(**r)
                  + ("  recall vit {:.3f} (exact {:.3f}) mea {:.3f} (exact {:.3f})".format(
                      r["recall_viterbi"], r["recall_viterbi_exact"], r["recall_mea"], r["recall_mea_exact"])
                     if "recall_mea" in r else ""))
    return rows


def summarize(rows):
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        key = "leaf" if r["case"].startswith("leaf") else ("bali_reest" if r["case"].endswith("reest") else "bali_single")
        g[key].append(r)
    out = {}
    for k, rs in g.items():
        out[k] = dict(n=len(rs), valid=all(r["max_excess"] <= 1e-6 for r in rs),
                      mean_gap=float(np.mean([r["mean_gap"] for r in rs])),
                      mean_sd_gap=float(np.mean([r["sd_gap"] for r in rs])),
                      mean_lo_bias=float(np.mean([r["lo_bias"] for r in rs])),
                      mean_spearman=float(np.nanmean([r["spearman"] for r in rs])),
                      mean_top1=float(np.nanmean([r["top1"] for r in rs])),
                      mean_top1_exact=float(np.nanmean([r["top1_exact"] for r in rs])))
        for dec in ("viterbi", "mea"):
            if all(f"recall_{dec}" in r for r in rs):
                out[k][f"recall_{dec}"] = float(np.mean([r[f"recall_{dec}"] for r in rs]))
                out[k][f"recall_{dec}_exact"] = float(np.mean([r[f"recall_{dec}_exact"] for r in rs]))
    return out


if __name__ == "__main__":
    cases = load_cases() if CACHE.exists() and "--rebuild" not in sys.argv else build_cases()
    rows = evaluate(current_bound, cases)
    import json
    print(json.dumps(summarize(rows), indent=1))


# ---------------------------------------------------------------- proposal quality (MCMC / IS)

from functools import partial as _partial  # noqa: E402


@_partial(jax.jit, static_argnums=(5, 6))
def _occupancy(logT, st, tab, eI, eD, Lx, Ly, Lxr, Lyr):
    """log Z and posterior occupancies (match cells, insert columns, delete columns) of the Pair
    HMM whose emissions are tab / eI / eD: the gradients of log Forward."""
    f = lambda tb, ei, ed: wf.forward(logT, st, lambda i, j: tb[i - 1, j - 1], ei, ed, Lx, Ly, Lxr, Lyr)
    return jax.value_and_grad(f, argnums=(0, 1, 2))(tab, eI, eD)


def _pair_hmm_stats(tab, sA, sB, t):
    global _INDEL
    if _INDEL is None:
        _INDEL = load_indel("mixfrag_F2")
    LA, LB = tab.shape
    Lx, Ly = _pad_to_bin(LA), _pad_to_bin(LB)
    tabp = jnp.zeros((Lx, Ly)).at[:LA, :LB].set(jnp.asarray(tab))
    eD = jnp.zeros(Lx + 1).at[1:LA + 1].set(jnp.asarray(sA))
    eI = jnp.zeros(Ly + 1).at[1:LB + 1].set(jnp.asarray(sB))
    logZ, (pM, pI, pD) = _occupancy(_INDEL.log_trans(t), _INDEL.state_types, tabp, eI, eD, Lx, Ly, LA, LB)
    return float(logZ), np.asarray(pM)[:LA, :LB], np.asarray(pI)[1:LB + 1], np.asarray(pD)[1:LA + 1]


def proposal_kl(q_tables, ex_tables, t):
    """KL(target || proposal) and KL(proposal || target), in nats, between the Pair HMM posteriors
    over alignments of a clade pair under the exact scores (target) and a scorer's (proposal).
    The two HMMs differ only in emissions, so log p_ex(path) - log p_q(path) is linear in the path's
    cell occupancy: KL(ex||q) = sum P_ex * Delta - (logZ_ex - logZ_q), Delta = exact - proposal."""
    (tq, aq, bq), (te, ae, be) = q_tables, ex_tables
    Zq, Mq, Iq, Dq = _pair_hmm_stats(tq, aq, bq, t)
    Ze, Me, Ie, De = _pair_hmm_stats(te, ae, be, t)
    dT, dA, dB = te - tq, ae - aq, be - bq
    lin_e = float(np.sum(Me * dT) + np.sum(De * dA) + np.sum(Ie * dB))
    lin_q = float(np.sum(Mq * dT) + np.sum(Dq * dA) + np.sum(Iq * dB))
    return dict(kl_target_proposal=lin_e - (Ze - Zq), kl_proposal_target=(Ze - Zq) - lin_q,
                n_cols=int(tq.shape[0] + tq.shape[1]))


def evaluate_proposals(scorer, cases, verbose=True):
    rows = []
    for c in cases:
        if not c["name"].startswith("bali"):
            continue
        ex = _call(exact_scorer, c)
        q = _call(scorer, c)
        r = dict(case=c["name"], **proposal_kl(q, ex, c["t"]))
        rows.append(r)
        if verbose:
            print("{case:34s} KL(target||prop)={kl_target_proposal:9.3f}  KL(prop||target)={kl_proposal_target:9.3f}"
                  "  cols={n_cols}".format(**r))
    return rows
