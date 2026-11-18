#!/usr/bin/env python3
"""FSA on BAliBASE with a per-site profile-mixture substitution model.

Wraps paper 1's BAliBASE pipeline (tkf-mixdom
``experiments/expected_pairwise_balibase.py``: same pairs, same sequence
annealing, same SP/TC scoring) and changes only the pair-HMM emissions and
the per-pair time estimate.

Substitution models (``--subst``):
    LG      paper 1's "LG08" (tkf-mixdom's table, which differs from the
            published LG in 171 of 190 exchangeabilities)
    LGp     the published LG (PAML lg.dat)
    C20p    as C20 with the published LG exchangeabilities
    C20     LG exchangeabilities with the C20 profiles, each class at mean rate 1
    C20g    as C20 with one global scale (the mixture has mean rate 1)
    CML20   the CherryML 20-class mixture of paper 1 (21-state, gap state
            dropped, weights unchanged), used per site
    CML20r  as CML20 with class weights conditioned on a residue being
            present, w_k (1 - pi^k_gap)

``--gamma ALPHA`` adds four discrete Gamma rate categories (Yang 1994 mean
rates) to every class.

A match column emits (a, b) with probability J(t)_ab = sum_k w_k pi^k_a M^k(t)_ab,
insert and delete columns emit from pibar = sum_k w_k pi^k.  Since the class is
drawn independently per column this is the exact class-marginal pair HMM, at
the same cost as a single substitution model.

Per pair:
  1. E-step: Forward-Backward under the indel model + LG at tau = 1 (as paper 1).
  2. tau_LG: 5 Newton steps on the expected complete-data log-likelihood (paper 1).
  3. ``--clock one``: one time for both processes, from Newton on the same
     objective with the LG match term replaced by sum_ab W_ab log J(tau)_ab.
     ``--clock two``: indel time tau_LG; substitution time from Newton on
     sum_ab W_ab log J(tau)_ab alone (``--newton-sub`` steps, clipped to
     ``--tmax-sub``; paper 1's values 5 and 10 are the defaults).
     ``--clock anchor``: both at tau_LG (paper 1's anchor time).
     ``--em-rounds n`` repeats the E-step (under the current model and times)
     and step 3 n-1 more times; with ``--clock two`` the later rounds refit the
     indel time from the indel term alone.
  4. Forward-Backward at those times -> match posteriors -> FSA.

With ``--subst LG --clock one --em-rounds 1`` this is paper 1's pipeline
(checked against ``_pairwise_posteriors_mixfrag_jax`` in ``--selftest``).

``--indel mixfrag_s --mixfrag-s-npz CKPT`` is MixFrag-S: MixFrag whose
fragtype f carries its own substitution model (Q_f, pi_f), read from a
checkpoint in the svi-BW contract format (keys ins_rate, del_rate, exts,
weights, S_exch, pi, Q, ...).  ``--subst`` is ignored.  M_f emits (a, b) with
pi_f[a] P_f(t)_ab, I_f / D_f emit from pi_f (``pair_hmm_emissions_per_fragtype``
fed to Forward-Backward as a precomputed table).  Clock 'one' only: round 0 is
the anchor above (Forward-Backward under LG at t = 1 for every fragtype), then
Newton on  indel term + sum_f sum_ab W_f[a,b] log(pi_f[a] P_f(t)_ab)  with
W_f the expected match counts of the M_f states; the returned match posterior
is the sum over the M_f states.  ``--em-rounds`` repeats the E-step under the
per-fragtype matrices as above.  With every fragtype carrying the same
(Q, pi) this reproduces ``--indel mixfrag --subst GTR --clock one``.

Per-family results are appended to ``<out>.jsonl`` as they finish (rerunning
skips families already there); the summary JSON is written at the end.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import json
from functools import partial
import sys
import time
from pathlib import Path

import numpy as np

MIXDOM_PY = Path(__file__).resolve().parents[3]   # tkf-mixdom/python
sys.path.insert(0, str(MIXDOM_PY))
sys.path.insert(0, str(MIXDOM_PY / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax
import jax.numpy as jnp

if os.environ.get("JAX_CACHE_DIR"):
    jax.config.update("jax_compilation_cache_dir", os.environ["JAX_CACHE_DIR"])
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)

from tkfmixdom.jax.core.ctmc import build_Q_from_S_pi, transition_matrix
from tkfmixdom.jax.core.params import mixfrag_trans
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.dp.hmm import (_pad_to_bin, _pad_seq, forward_backward_2d,
                                  pair_hmm_emissions_per_fragtype)
from tkfmixdom.jax.models.left_regular import (make_mixfrag_pair_hmm, M as M_STATE,
                                               I as I_STATE, D as D_STATE)
from tkfmixdom.util.expected_pair_f1 import aggregate_corpus

import expected_pairwise_balibase as epb
from progalign import lg_paml
from progalign.model import Subst, make_subst

A = 20
# Default indel parameters: the fitted GTR-era Pfam fits that paper 1's TKF92 and
# MixFrag F=2 rows describe and that their likelihood cells use
# (~/tkf-mixdom/python/experiments/test_ll/tkf92_pfam_params.json and
# experiments/mixfrag_F/mixfrag_F2_logspace.json). Until 2026-11-13 this table
# pinned the LG-era values recorded in the original BAliBASE output JSONs
# (TKF92 0.045806/0.046799, ext 0.683455; MixFrag 0.033924/0.034690,
# exts 0.413963/0.860309, weights 0.700861/0.299139); runs made with those are
# archived as *_lgparams*. Pass --indel-params to be explicit.
INDEL = {
    "mixfrag": dict(lam=0.04645159361484088, mu=0.04747838153710662,
                    exts=[0.4307407767585113, 0.8558260528890574],
                    weights=[0.7097808771217816, 0.2902191228782184]),
    "tkf92": dict(lam=0.04428, mu=0.04521, exts=[0.677], weights=[1.0]),
}
CHERRYML_NPZ = Path(os.environ["CHERRYML_NPZ"]) if os.environ.get("CHERRYML_NPZ") else MIXDOM_PY / "pfam" / "cherryml_mixture_C20_n5000.npz"   # env override: e.g. the 20-state refit


def subst_from_Q(Q, pi, w, name):
    """Subst for arbitrary reversible per-class chains (Q^k, pi^k), weights w."""
    Q, pi, w = np.asarray(Q, float), np.asarray(pi, float), np.asarray(w, float)
    sq = np.sqrt(pi)
    B = sq[:, :, None] * Q / sq[:, None, :]
    B = 0.5 * (B + np.swapaxes(B, 1, 2))
    lam, Psi = np.linalg.eigh(B)
    K = len(w)
    return Subst(jnp.zeros((A, A)), jnp.asarray(pi), jnp.asarray(w / w.sum()),
                 jnp.ones(K), jnp.asarray(lam), jnp.asarray(Psi), name=name)


def gamma_rates(alpha, n=4):
    """Mean rates of n equiprobable discrete Gamma(alpha, 1/alpha) categories (Yang 1994)."""
    from scipy.stats import gamma as G
    from scipy.special import gammainc
    q = G.ppf(np.arange(1, n) / n, alpha, scale=1 / alpha)
    return n * np.diff(gammainc(alpha + 1, np.concatenate([[0], q * alpha, [np.inf]])))


def with_gamma(sub, alpha, n=4):
    """Each class k becomes n classes (k, g) with weight w_k/n and rates scaled by r_g."""
    r = jnp.asarray(gamma_rates(alpha, n))
    K = sub.K
    return Subst(sub.S, jnp.repeat(sub.P, n, 0), jnp.repeat(sub.w, n) / n,
                 jnp.repeat(sub.R, n) * jnp.tile(r, K),
                 (sub.lam[:, None, :] * r[None, :, None]).reshape(K * n, A),
                 jnp.repeat(sub.Psi, n, 0), name=f"{sub.name}+G{alpha}")


def load_subst(kind, gamma=None):
    if gamma:
        return with_gamma(load_subst(kind), gamma)
    if kind == "LG":
        Q, pi = rate_matrix_lg()
        return subst_from_Q(np.asarray(Q)[None], np.asarray(pi)[None], [1.0], "LG")
    if kind in ("C20", "C20g"):
        return make_subst(kind)
    if kind == "GTR":
        # a single GTR refitted to the precompiled Pfam training pairs at their own clock (fit_paircolumn_sitemix.py
        # --n-classes 1 -> pfam/gtr_refit_Qpi.npz); absolute rate as fitted, not renormalised
        d = np.load(os.environ["GTR_NPZ"], allow_pickle=True)
        Q = np.asarray(d["Q"], float)[None]; P = np.asarray(d["pi"], float)[None]; P = P / P.sum(1, keepdims=True)
        return subst_from_Q(Q, P, np.ones(1), kind)
    if kind in ("LGp", "C20p"):
        S, pi_lg = lg_paml.exchangeability()
        if kind == "LGp":
            P, w = pi_lg[None], np.ones(1)
        else:
            from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c20
            P, w, _ = le_gascuel_c20()
            P = np.asarray(P, float) / np.asarray(P, float).sum(1, keepdims=True)
            w = np.asarray(w, float) / np.sum(w)
        Q = S[None] * P[:, None, :]
        Q[:, np.arange(A), np.arange(A)] = -Q.sum(2)
        Q = Q / np.einsum("ka,kaa->k", P, -Q)[:, None, None]   # each class at mean rate 1
        return subst_from_Q(Q, P, w, kind)
    if kind in ("CML20", "CML20r"):
        d = np.load(CHERRYML_NPZ, allow_pickle=True)
        S = np.asarray(d["S"], float)[:, :A, :A]
        pi21 = np.asarray(d["pi"], float)
        pi21 = pi21 / pi21.sum(1, keepdims=True)
        pi = pi21[:, :A] / pi21[:, :A].sum(1, keepdims=True)
        w = np.asarray(d["weights"], float)
        if kind == "CML20r":
            w = w * (1.0 - pi21[:, A])
        Q = S * pi[:, None, :]
        Q[:, np.arange(A), np.arange(A)] = 0.0
        Q[:, np.arange(A), np.arange(A)] = -Q.sum(2)
        return subst_from_Q(Q, pi, w, kind)
    raise ValueError(kind)


def _sub_arrays(sub):
    return (sub.P, sub.w, sub.lam, sub.Psi)


def _joint(arrs, t):
    P, w, lam, Psi = arrs
    sq = jnp.sqrt(P)
    core = jnp.einsum("kxi,ki,kyi->kxy", Psi, jnp.exp(lam * t), Psi)
    Mt = core * (sq[:, None, :] / sq[:, :, None])
    return jnp.einsum("k,ka,kab->ab", w, P, Mt)


def _log(x):
    return jnp.log(jnp.maximum(x, 1e-300))


def _indel_ll(log_tau, n_trans, lam, mu, exts, weights):
    chi = mixfrag_trans(lam, mu, jnp.exp(log_tau), exts, weights)
    return jnp.sum(n_trans * _log(chi))


def _sub_ll(log_tau, W, arrs):
    return jnp.sum(W * _log(_joint(arrs, jnp.exp(log_tau))))


def _newton(f, log_tau0, n_newton, tmax=10.0):
    g, h = jax.grad(f), jax.grad(jax.grad(f))
    lt = log_tau0
    for _ in range(n_newton):
        gv, hv = g(lt), h(lt)
        safe_neg_h = jnp.where(jnp.abs(hv) > 1e-10, -hv, 1.0)
        lt = lt + jnp.clip(gv / safe_neg_h, -1.0, 1.0)
    return jnp.exp(jnp.clip(lt, jnp.log(1e-4), jnp.log(tmax)))


@jax.jit
def _hmm(t_ind, t_sub, lam, mu, exts, weights, arrs):
    """Pair-HMM parameters: MixFrag transitions at t_ind, class-marginal emissions at t_sub."""
    log_trans, st, _, _ = make_mixfrag_pair_hmm(
        lam, mu, t_ind, exts, weights, jnp.zeros((A, A)), jnp.ones(A) / A)
    J = _joint(arrs, t_sub)
    pibar = J.sum(1)
    return log_trans, st, J / pibar[:, None], pibar


@partial(jax.jit, static_argnames=("P", "mode"))
def _cap_gap_open(log_trans, st, P, mode):
    """Gap-opening cap / fix on the pair HMM's log transition matrix.

    In every M-type row, the mass leaving to I-type states and the mass
    leaving to D-type states are each set to ``min(mass, P/2)`` (``mode ==
    "cap"``) or to exactly ``P/2`` (``mode == "fix"``); the difference goes to
    the M-type entries of the row, so each row stays stochastic.  Extensions
    (I/D rows), S/E transitions and emissions are untouched.  With several
    states per type (MixFrag F>1) each group is rescaled proportionally.
    Done in log space with multiplicative factors, so an inactive cap (factor
    exactly 1) returns ``log_trans`` bit for bit."""
    T = jnp.exp(log_trans)
    is_M = (st == M_STATE)
    is_I = (st == I_STATE)
    is_D = (st == D_STATE)
    m_rows = is_M[:, None]
    mass_M = jnp.sum(jnp.where(is_M[None, :], T, 0.0), axis=1)
    mass_I = jnp.sum(jnp.where(is_I[None, :], T, 0.0), axis=1)
    mass_D = jnp.sum(jnp.where(is_D[None, :], T, 0.0), axis=1)
    half = jnp.float64(P / 2.0)
    if mode == "cap":
        tgt_I, tgt_D = jnp.minimum(mass_I, half), jnp.minimum(mass_D, half)
    elif mode == "fix":
        tgt_I, tgt_D = jnp.full_like(mass_I, half), jnp.full_like(mass_D, half)
    else:
        raise ValueError(mode)
    safe = lambda a, b: jnp.where(b > 0, a / jnp.where(b > 0, b, 1.0), 1.0)
    f_I = safe(tgt_I, mass_I)
    f_D = safe(tgt_D, mass_D)
    f_M = safe(mass_M + (mass_I - tgt_I) + (mass_D - tgt_D), mass_M)
    fac = jnp.where(is_I[None, :], f_I[:, None],
                    jnp.where(is_D[None, :], f_D[:, None],
                              jnp.where(is_M[None, :], f_M[:, None], 1.0)))
    fac = jnp.where(m_rows, fac, 1.0)
    return log_trans + jnp.log(fac)


@jax.jit
def _match_W(post, st, x, y):
    """Match posteriors (Lx_pad, Ly_pad) and expected match counts W_ab."""
    Lx_pad, Ly_pad = x.shape[0], y.shape[0]
    is_M = (st == M_STATE).astype(jnp.float64)
    mp = jnp.einsum("ijs,s->ij", post[1:Lx_pad + 1, 1:Ly_pad + 1, :], is_M)
    X = jax.nn.one_hot(x, A, dtype=jnp.float64)
    Y = jax.nn.one_hot(y, A, dtype=jnp.float64)
    return mp, jnp.einsum("ij,ia,jb->ab", mp, X, Y)


@partial(jax.jit, static_argnames=("mode", "n_newton", "tmax"))
def _fit_time(lt0, n_trans, W, lam, mu, exts, weights, arrs, mode, n_newton, tmax):
    """Newton on log t; mode: 'joint' (indel + substitution), 'indel', or 'sub'."""
    def f(l):
        v = 0.0
        if mode in ("joint", "indel"):
            v = v + _indel_ll(l, n_trans, lam, mu, exts, weights)
        if mode in ("joint", "sub"):
            v = v + _sub_ll(l, W, arrs)
        return v
    return _newton(f, lt0, n_newton, tmax)


def make_pair_fn(clock, em_rounds, n_newton=5, n_newton_sub=5, tmax_sub=10.0,
                 gap_open_cap=None, gap_open_fix=None):
    """Per-pair posteriors.  Only forward_backward_2d (module-level jit in
    tkf-mixdom, as in paper 1) and _match_W depend on the sequence lengths, so
    each length bin compiles one Forward-Backward shared by every variant.

    ``gap_open_cap`` / ``gap_open_fix`` (at most one; default None = off):
    after the branch length has been fitted exactly as without them, the final
    pair HMM at that length gets its M->I and M->D probabilities capped at /
    set to P/2 (see ``_cap_gap_open``) before the posteriors are computed."""
    if gap_open_cap is not None and gap_open_fix is not None:
        raise ValueError("gap_open_cap and gap_open_fix are mutually exclusive")
    gap_P = gap_open_cap if gap_open_cap is not None else gap_open_fix
    gap_mode = "cap" if gap_open_cap is not None else "fix"

    def pair_fn(x, y, Lx, Ly, lam, mu, exts, weights, arrs_lg, arrs):
        t_ind = t_sub = jnp.float64(1.0)
        e_arrs = arrs_lg
        for rnd in range(em_rounds):
            log_trans, st, sub, pibar = _hmm(t_ind, t_sub, lam, mu, exts, weights, e_arrs)
            _, post, n_trans = forward_backward_2d(
                log_trans, st, x, y, sub, pibar, real_Lx=Lx, real_Ly=Ly)
            _, W = _match_W(post, st, x, y)
            lt_ind0 = jnp.log(t_ind) if rnd else jnp.float64(0.0)
            lt_sub0 = jnp.log(t_sub) if rnd else jnp.float64(0.0)
            common = (n_trans, W, lam, mu, exts, weights)
            if clock == "anchor":
                t_ind = t_sub = _fit_time(lt_ind0, *common, arrs_lg, mode="joint",
                                          n_newton=n_newton, tmax=10.0)
            elif clock == "one":
                t_ind = t_sub = _fit_time(lt_ind0, *common, arrs, mode="joint",
                                          n_newton=n_newton, tmax=10.0)
            else:
                if rnd == 0:
                    # paper 1's tau under LG sets the indel clock
                    t_ind = _fit_time(lt_ind0, *common, arrs_lg, mode="joint",
                                      n_newton=n_newton, tmax=10.0)
                else:
                    t_ind = _fit_time(lt_ind0, *common, arrs, mode="indel",
                                      n_newton=n_newton, tmax=10.0)
                t_sub = _fit_time(lt_sub0, *common, arrs, mode="sub",
                                  n_newton=n_newton_sub, tmax=tmax_sub)
            e_arrs = arrs
        log_trans, st, sub, pibar = _hmm(t_ind, t_sub, lam, mu, exts, weights, arrs)
        if gap_P is not None:
            log_trans = _cap_gap_open(log_trans, st, P=float(gap_P), mode=gap_mode)
        log_prob, post, _ = forward_backward_2d(
            log_trans, st, x, y, sub, pibar, real_Lx=Lx, real_Ly=Ly)
        mp, _ = _match_W(post, st, x, y)
        return mp, t_ind, t_sub, log_prob
    return pair_fn


def _method_from_pair_fn(pair_fn, model_args, time_log):
    """The per-pair loop shared by every model: pad, run ``pair_fn(x, y, Lx, Ly,
    *model_args)``, trim the match posterior, log (t_ind, t_sub, log_prob),
    and record (rather than raise) per-pair failures."""
    def method(int_seqs, names, pairs, _raw):
        pp, failed = {}, []
        for i, j in pairs:
            xs, ys = int_seqs[names[i]], int_seqs[names[j]]
            Lx, Ly = len(xs), len(ys)
            try:
                x = _pad_seq(jnp.asarray(xs, jnp.int32), _pad_to_bin(Lx))
                y = _pad_seq(jnp.asarray(ys, jnp.int32), _pad_to_bin(Ly))
                mp, ti, ts, lp = pair_fn(x, y, jnp.int32(Lx), jnp.int32(Ly), *model_args)
                pp[(i, j)] = np.asarray(mp)[:Lx, :Ly]
                time_log.append((names[i], names[j], float(ti), float(ts), float(lp)))
            except Exception as e:
                jax.clear_caches()
                failed.append({"pair": [int(i), int(j)], "name_i": names[i],
                               "name_j": names[j], "Lx": Lx, "Ly": Ly,
                               "error_type": type(e).__name__,
                               "error_msg": str(e)[:500]})
        return pp, "soft", failed, {}
    return method


def make_method(indel, subst, clock, em_rounds, time_log, gamma=None,
                n_newton_sub=5, tmax_sub=10.0, gap_open_cap=None, gap_open_fix=None):
    p = INDEL[indel]
    lam, mu = jnp.float64(p["lam"]), jnp.float64(p["mu"])
    exts, weights = jnp.asarray(p["exts"]), jnp.asarray(p["weights"])
    arrs_lg = _sub_arrays(load_subst("LG"))
    arrs = _sub_arrays(load_subst(subst, gamma))
    pair_fn = make_pair_fn(clock, em_rounds, n_newton_sub=n_newton_sub,
                           tmax_sub=tmax_sub, gap_open_cap=gap_open_cap,
                           gap_open_fix=gap_open_fix)
    return _method_from_pair_fn(pair_fn, (lam, mu, exts, weights, arrs_lg, arrs), time_log)


# ---------------------------------------------------------------------------
# MixFrag-S: one substitution model (Q_f, pi_f) per fragtype.  A parallel path
# to _hmm / _match_W / _fit_time, which assume one shared matrix.
# ---------------------------------------------------------------------------

def load_mixfrag_s(path):
    """Read a MixFrag-S checkpoint (svi-BW contract format).

    Returns ``(indel, Qs, pis)``: the INDEL-table dict (lam, mu, exts, weights),
    the per-fragtype rate matrices (F, A, A) in the checkpoint's own (raw) units,
    and the per-fragtype equilibria (F, A) normalised to sum to one.  ``Q`` is
    taken from the checkpoint when present, else rebuilt from ``S_exch`` and ``pi``."""
    d = np.load(path, allow_pickle=True)
    exts = [float(v) for v in np.asarray(d["exts"], float)]
    weights = [float(v) for v in np.asarray(d["weights"], float)]
    F = int(d["n_fragtypes"]) if "n_fragtypes" in d.files else len(exts)
    indel = dict(lam=float(d["ins_rate"]), mu=float(d["del_rate"]), exts=exts, weights=weights)
    pis = np.asarray(d["pi"], float)
    pis = pis / pis.sum(1, keepdims=True)
    if "Q" in d.files:
        Qs = np.asarray(d["Q"], float)
    else:
        Qs = np.stack([np.asarray(build_Q_from_S_pi(jnp.asarray(S, jnp.float64), jnp.asarray(p)))
                       for S, p in zip(np.asarray(d["S_exch"], float), pis)])
    if Qs.shape != (F, A, A) or pis.shape != (F, A) or len(exts) != F or len(weights) != F:
        raise ValueError(f"{path}: inconsistent shapes Q {Qs.shape} pi {pis.shape} "
                         f"exts {len(exts)} weights {len(weights)} n_fragtypes {F}")
    return indel, Qs, pis


def _P_stack(Qs, t):
    """P_f(t) = expm(Q_f t) for every fragtype, (F, A, A)."""
    return jnp.stack([transition_matrix(Qs[f], t) for f in range(Qs.shape[0])])


@jax.jit
def _hmm_s(t, lam, mu, exts, weights, Qs, pis, x, y):
    """MixFrag-S pair HMM at time t: MixFrag transitions and the per-fragtype
    log-emission table (Lx+1, Ly+1, 3F+2) for the (padded) sequences x, y."""
    log_trans, st, _, _ = make_mixfrag_pair_hmm(
        lam, mu, t, exts, weights, jnp.zeros((A, A)), jnp.ones(A) / A)
    table = pair_hmm_emissions_per_fragtype(st, x, y, _P_stack(Qs, t), pis)
    return log_trans, st, table


@jax.jit
def _match_W_s(post, x, y):
    """Match posterior summed over the M_f states (Lx_pad, Ly_pad) and the
    per-fragtype expected match counts W_f[a,b] (F, A, A).  State order is
    make_mixfrag_pair_hmm's S, M_1..M_F, I_1..I_F, D_1..D_F, E, so M_f is
    state 1 + f and F = (n_states - 2) / 3."""
    Lx_pad, Ly_pad = x.shape[0], y.shape[0]
    F = (post.shape[2] - 2) // 3
    mpf = post[1:Lx_pad + 1, 1:Ly_pad + 1, 1:1 + F]
    X = jax.nn.one_hot(x, A, dtype=jnp.float64)
    Y = jax.nn.one_hot(y, A, dtype=jnp.float64)
    return mpf.sum(-1), jnp.einsum("ijf,ia,jb->fab", mpf, X, Y)


def _sub_ll_s(log_tau, Wf, Qs, pis):
    """sum_f sum_ab W_f[a,b] log(pi_f[a] P_f(t)[a,b])."""
    t = jnp.exp(log_tau)
    v = 0.0
    for f in range(Qs.shape[0]):
        v = v + jnp.sum(Wf[f] * (_log(pis[f])[:, None] + _log(transition_matrix(Qs[f], t))))
    return v


@partial(jax.jit, static_argnames=("n_newton", "tmax"))
def _fit_time_s(lt0, n_trans, Wf, lam, mu, exts, weights, Qs, pis, n_newton, tmax):
    """Clock 'one' for MixFrag-S: Newton on log t for indel term + _sub_ll_s."""
    def f(l):
        return _indel_ll(l, n_trans, lam, mu, exts, weights) + _sub_ll_s(l, Wf, Qs, pis)
    return _newton(f, lt0, n_newton, tmax)


def make_pair_fn_s(em_rounds, n_newton=5):
    """MixFrag-S per-pair posteriors, clock 'one'.  Round 0 is the anchor
    E-step of make_pair_fn (every fragtype under LG at t = 1); every round fits
    t by Newton on the MixFrag-S objective; later rounds (--em-rounds) run the
    E-step under the per-fragtype matrices at the current t."""
    def pair_fn(x, y, Lx, Ly, lam, mu, exts, weights, Qs_lg, pis_lg, Qs, pis):
        t = jnp.float64(1.0)
        eQ, epi = Qs_lg, pis_lg
        for rnd in range(em_rounds):
            log_trans, st, table = _hmm_s(t, lam, mu, exts, weights, eQ, epi, x, y)
            _, post, n_trans = forward_backward_2d(
                log_trans, st, x, y, None, None, log_emit_table=table, real_Lx=Lx, real_Ly=Ly)
            _, Wf = _match_W_s(post, x, y)
            lt0 = jnp.log(t) if rnd else jnp.float64(0.0)
            t = _fit_time_s(lt0, n_trans, Wf, lam, mu, exts, weights, Qs, pis,
                            n_newton=n_newton, tmax=10.0)
            eQ, epi = Qs, pis
        log_trans, st, table = _hmm_s(t, lam, mu, exts, weights, Qs, pis, x, y)
        log_prob, post, _ = forward_backward_2d(
            log_trans, st, x, y, None, None, log_emit_table=table, real_Lx=Lx, real_Ly=Ly)
        mp, _ = _match_W_s(post, x, y)
        return mp, t, t, log_prob
    return pair_fn


def make_method_s(npz, em_rounds, time_log):
    """MixFrag-S method from a checkpoint; also registers INDEL['mixfrag_s']."""
    indel, Qs, pis = load_mixfrag_s(npz)
    INDEL["mixfrag_s"] = indel
    F = len(indel["exts"])
    lam, mu = jnp.float64(indel["lam"]), jnp.float64(indel["mu"])
    exts, weights = jnp.asarray(indel["exts"]), jnp.asarray(indel["weights"])
    Q_lg, pi_lg = rate_matrix_lg()
    Qs_lg = jnp.tile(jnp.asarray(Q_lg, jnp.float64)[None], (F, 1, 1))
    pis_lg = jnp.tile(jnp.asarray(pi_lg, jnp.float64)[None], (F, 1))
    model_args = (lam, mu, exts, weights, Qs_lg, pis_lg,
                  jnp.asarray(Qs, jnp.float64), jnp.asarray(pis, jnp.float64))
    return _method_from_pair_fn(make_pair_fn_s(em_rounds), model_args, time_log)


def selftest():
    """--subst LG --clock one --em-rounds 1 must reproduce paper 1's posteriors."""
    from tkfmixdom.jax.tree.fsa_anneal import _pairwise_posteriors_mixfrag_jax
    rng = np.random.default_rng(0)
    Q, pi = rate_matrix_lg()
    for indel in ("mixfrag", "tkf92"):
        p = INDEL[indel]
        fn = make_method(indel, "LG", "one", 1, [])
        for Lx, Ly in ((37, 52), (90, 70)):
            xs = rng.integers(0, A, Lx)
            ys = np.concatenate([xs[:Ly // 2], rng.integers(0, A, Ly - Ly // 2)])
            pp, _, failed, _ = fn({"a": xs, "b": ys}, ["a", "b"], [(0, 1)], None)
            assert not failed, failed
            x = _pad_seq(jnp.asarray(xs, jnp.int32), _pad_to_bin(Lx))
            y = _pad_seq(jnp.asarray(ys, jnp.int32), _pad_to_bin(Ly))
            ref, tau, _ = _pairwise_posteriors_mixfrag_jax(
                x, y, jnp.int32(Lx), jnp.int32(Ly), jnp.float64(p["lam"]),
                jnp.float64(p["mu"]), jnp.asarray(p["exts"]),
                jnp.asarray(p["weights"]), jnp.asarray(Q), jnp.asarray(pi))
            d = np.abs(pp[(0, 1)] - np.asarray(ref)[:Lx, :Ly]).max()
            print(f"selftest {indel} {Lx}x{Ly}: tau={float(tau):.4f} max|diff|={d:.2e}")
            assert d < 1e-8
    for kind, g in (("C20", None), ("C20g", None), ("CML20", None), ("LG", 1.5), ("C20", 0.35)):
        arrs = _sub_arrays(load_subst(kind, g))
        for t in (0.0, 0.3, 2.0):
            J = np.asarray(_joint(arrs, t))
            assert abs(J.sum() - 1) < 1e-10 and np.abs(J - J.T).max() < 1e-10, kind
            pibar = np.asarray(arrs[1] @ arrs[0])
            assert np.abs(J.sum(1) - pibar).max() < 1e-10, kind
        J0 = np.asarray(_joint(arrs, 0.0))
        assert np.abs(J0 - np.diag(np.diag(J0))).max() < 1e-12, kind
        if g:
            r = gamma_rates(g)
            assert abs(r.mean() - 1) < 1e-10
            base = _sub_arrays(load_subst(kind))
            Jg = sum(np.asarray(_joint(base, 0.7 * ri)) for ri in r) / 4
            assert np.abs(Jg - np.asarray(_joint(arrs, 0.7))).max() < 1e-12
        print(f"selftest {kind}{'+G' + str(g) if g else ''}: joint symmetric, sums to 1, margins pibar, J(0) diagonal")
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--indel", choices=list(INDEL) + ["mixfrag_s"], default="mixfrag")
    ap.add_argument("--indel-params", default=None,
                    help="JSON with lam/mu/exts/weights overriding the INDEL table entry (any number of fragtypes)")
    ap.add_argument("--mixfrag-s-npz", default=None,
                    help="--indel mixfrag_s: MixFrag-S checkpoint (svi-BW contract npz: ins_rate, del_rate, exts, "
                         "weights, S_exch, pi, Q, ...); indel and per-fragtype substitution parameters come from "
                         "it and --subst is ignored (clock 'one' only)")
    ap.add_argument("--subst", choices=["LG", "LGp", "GTR", "C20", "C20p", "C20g", "CML20", "CML20r"],
                    default="C20")
    ap.add_argument("--clock", choices=["one", "two", "anchor"], default="two")
    ap.add_argument("--em-rounds", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=None,
                    help="Gamma shape for 4 discrete rate categories per class")
    ap.add_argument("--tmax-sub", type=float, default=10.0)
    ap.add_argument("--newton-sub", type=int, default=5)
    ap.add_argument("--out", help="summary JSON (per-family JSONL alongside)")
    ap.add_argument("--balibase-dir",
                    default=str(Path.home() / "bio-datasets/data/balibase/bali3pdbm"))
    ap.add_argument("--families", default=None,
                    help="comma-separated family names, or a file with one family name per line")
    ap.add_argument("--gap-open-cap", type=float, default=None, metavar="P",
                    help="cap the final pair HMM's M->I and M->D probabilities at P/2 each "
                         "(mass returned to M->M); e.g. 0.04 = ProbCons/MUSCLE's total opening")
    ap.add_argument("--gap-open-fix", type=float, default=None, metavar="P",
                    help="set M->I and M->D to exactly P/2 at every branch length "
                         "(divergence-independent opening, like ProbCons)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache-name", default=None,
                    help="store the pairwise posteriors in the shared posterior cache under this method name "
                         "(~/.cache/tkf-mixdom-balibase/<name>/), so decoding variants can be rerun from cache")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.gap_open_cap is not None and args.gap_open_fix is not None:
        ap.error("--gap-open-cap and --gap-open-fix are mutually exclusive")
    gap_suffix = ""
    if args.gap_open_cap is not None:
        gap_suffix = f"_cap{round(args.gap_open_cap * 1000):03d}"
    elif args.gap_open_fix is not None:
        gap_suffix = f"_fix{round(args.gap_open_fix * 1000):03d}"

    if args.indel == "mixfrag_s":
        if not args.mixfrag_s_npz:
            ap.error("--indel mixfrag_s needs --mixfrag-s-npz")
        if args.clock != "one":
            ap.error("--indel mixfrag_s supports --clock one only")
        if args.indel_params:
            ap.error("--indel mixfrag_s takes its indel parameters from the checkpoint; drop --indel-params")
        if args.gamma:
            ap.error("--gamma is not supported with --indel mixfrag_s")
        if gap_suffix:
            ap.error("--gap-open-cap/--gap-open-fix are not supported with --indel mixfrag_s")
        name = (f"mixfrag_s_{Path(args.mixfrag_s_npz).stem}_{args.clock}"
                + (f"_em{args.em_rounds}" if args.em_rounds > 1 else ""))
    else:
        name = (f"{args.indel}_{args.subst}" + (f"G{args.gamma:g}" if args.gamma else "")
                + f"_{args.clock}" + (f"_T{args.tmax_sub:g}" if args.tmax_sub != 10 else "")
                + (f"_em{args.em_rounds}" if args.em_rounds > 1 else "")
                + gap_suffix)
    if gap_suffix and args.cache_name and gap_suffix not in args.cache_name:
        # the capped posteriors must never land in the uncapped cache:
        # tkf92_GTR_one__bali3pdbm -> tkf92_GTR_one_cap040__bali3pdbm
        base, sep, ds = args.cache_name.partition("__")
        args.cache_name = base + gap_suffix + sep + ds
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = out.with_suffix(".jsonl")
    done = {}
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            r = json.loads(line)
            done[r["family"]] = r
    in_dir = os.path.join(args.balibase_dir, "in")
    ref_dir = os.path.join(args.balibase_dir, "ref")
    fams = sorted(f for f in os.listdir(in_dir) if os.path.exists(os.path.join(ref_dir, f)))
    if args.families:
        if os.path.isfile(args.families):
            want = {ln.strip() for ln in open(args.families) if ln.strip() and not ln.startswith("#")}
        else:
            want = set(args.families.split(","))
        fams = [f for f in fams if f in want]
    print(f"{name}: {len(fams)} families, {len(done)} already done", flush=True)
    times = []
    if args.indel_params:
        _ip = json.load(open(args.indel_params))
        INDEL[args.indel] = dict(lam=float(_ip["lam"]), mu=float(_ip["mu"]), exts=[float(v) for v in _ip["exts"]], weights=[float(v) for v in _ip["weights"]])
        print(f"indel params from {args.indel_params}: {INDEL[args.indel]}", flush=True)
    if args.indel == "mixfrag_s":
        method = make_method_s(args.mixfrag_s_npz, args.em_rounds, times)
        print(f"MixFrag-S from {args.mixfrag_s_npz}: F={len(INDEL['mixfrag_s']['exts'])} "
              f"indel {INDEL['mixfrag_s']}", flush=True)
    else:
        method = make_method(args.indel, args.subst, args.clock, args.em_rounds, times,
                             gamma=args.gamma, n_newton_sub=args.newton_sub,
                             tmax_sub=args.tmax_sub, gap_open_cap=args.gap_open_cap,
                             gap_open_fix=args.gap_open_fix)
        if gap_suffix:
            print(f"gap opening {'cap' if args.gap_open_cap is not None else 'fix'} "
                  f"P={args.gap_open_cap if args.gap_open_cap is not None else args.gap_open_fix} "
                  f"(M->I, M->D each {'<=' if args.gap_open_cap is not None else '='} P/2); "
                  f"cache name {args.cache_name}", flush=True)
    for fi, fam in enumerate(fams):
        if fam in done:
            continue
        times.clear()
        t0 = time.time()
        res = epb.process_family(fam, in_dir, ref_dir, method, fsa_mode="auto",
                                 fsa_anneal_iters=3, fsa_seed=42, fsa_sps=True,
                                 cache_method_name=args.cache_name, cache_params_key=args.cache_name,
                                 cache_disabled=args.cache_name is None)
        if res is None:
            continue
        res["pair_times"] = list(times)
        res["time_total"] = time.time() - t0
        with open(jsonl, "a") as f:
            f.write(json.dumps(res) + "\n")
        done[fam] = res
        print(f"[{fi+1:>3}/{len(fams)}] {fam:<10} SP {res.get('msa_sp_g1', float('nan')):.3f} "
              f"TC {res.get('msa_tc_g1', float('nan')):.3f}  "
              f"(g0 {res.get('msa_sp_g0', float('nan')):.3f})  {res['time_total']:.0f}s",
              flush=True)

    results = [done[f] for f in fams if f in done]
    summ = {k: aggregate_corpus([{"per_pair": r[f"per_pair_{k}"]} for r in results
                                 if r.get(f"per_pair_{k}") is not None])
            for k in ("post", "hard", "opt", "fsa_sps")}
    sp1 = [r["msa_sp_g1"] for r in results if r.get("msa_sp_g1") is not None]
    tc1 = [r["msa_tc_g1"] for r in results if r.get("msa_tc_g1") is not None]
    sp0 = [r["msa_sp_g0"] for r in results if r.get("msa_sp_g0") is not None]
    tc0 = [r["msa_tc_g0"] for r in results if r.get("msa_tc_g0") is not None]
    payload = {
        "method_name": name, "indel": args.indel, "indel_params": INDEL[args.indel],
        "subst": "mixfrag_s" if args.indel == "mixfrag_s" else args.subst,
        "mixfrag_s_npz": args.mixfrag_s_npz, "gamma": args.gamma, "clock": args.clock,
        "em_rounds": args.em_rounds, "tmax_sub": args.tmax_sub,
        "newton_sub": args.newton_sub,
        "gap_open_cap": args.gap_open_cap, "gap_open_fix": args.gap_open_fix,
        "cache_name": args.cache_name,
        "n_families": len(results),
        "mean_sp_g1": float(np.mean(sp1)) if sp1 else None,
        "mean_tc_g1": float(np.mean(tc1)) if tc1 else None,
        "mean_sp_g0": float(np.mean(sp0)) if sp0 else None,
        "mean_tc_g0": float(np.mean(tc0)) if tc0 else None,
        "n_scored_g1": len(sp1),
        **{f"corpus_{k}": v for k, v in summ.items()},
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"{name}: SP {payload['mean_sp_g1']:.4f} TC {payload['mean_tc_g1']:.4f} "
          f"(n={len(sp1)}); gap_factor=0: SP {payload['mean_sp_g0']:.4f} "
          f"TC {payload['mean_tc_g0']:.4f}", flush=True)


if __name__ == "__main__":
    main()
