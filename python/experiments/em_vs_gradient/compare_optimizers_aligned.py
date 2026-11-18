#!/usr/bin/env python3
"""Closed-form EM vs gradient methods, ALIGNMENT-GIVEN (summarised counts).

Companion to compare_optimizers.py (which sums over alignments with a 2D
pair HMM).  Here the alignments are given, so each cherry is summarised into
the count tensors of supplement B.6 (match-run lengths, gap (i,j) totals,
boundary counts per tau bin) and the indel log-likelihood is a closed-form
function of the parameters that needs no per-pair DP: one small
forward-backward per tau bin over the aggregated counts.  TKF92 is the F=1
case of MixFrag.  (Q, pi) is held fixed (LG08) and is an additive constant,
so only the indel part of the log-likelihood is compared.

Methods, all on the same objective from the same initial parameters:
  em     exact summarised-count EM (`em_fit`, closed-form M-steps, flat priors)
  adam   Adam on unconstrained parameters through autodiff of the per-bin
         log-partition
  lbfgs  scipy L-BFGS-B on the same parameters and gradient

Corpora:
  --corpus pfam       the same breadth-first Pfam cherries as compare_optimizers.py
                      (seed 0), with their Pfam alignments
  --corpus sim_tkf92  MixFrag F=1 cherries simulated at the sim_tkf92 truth
                      (lam 0.058, mu 0.060, r 0.75) at Pfam-like branch lengths

MixFrag-S (--model mixfrag_s --n-fragtypes F): the latent fragtype decides the
emission, so the summarised-count likelihood above does not exist for it.  As for
the paper's MixDom aligned row, the objective is instead the 1D ALIGNMENT-
CONDITIONED (path-conditioned) log-likelihood of the given alignments,

    LL(theta) = (1/N) sum_pairs log P(alignment columns, x, y | theta, t),

summed over the fragtype assignments only (at an M column only the M_f states
may fire, at an I column only I_f, at a D column only D_f; S at the start, E at
the end), computed by the 1D forward-backward along each alignment's column
sequence (tkfmixdom.jax.train.mixfrag_svi_bw, dp_mode 'path').  Methods:
  em     full-batch 1D E-step (estep_batch_mixfrag_s_path: expected transition
         counts -> BDI / fragtype statistics, per-column fragtype posteriors ->
         per-fragtype residue counts and Holmes-Rubin (W_f, U_f) at each pair's
         own t) + the closed-form M-steps of compare_optimizers.em_mstep with
         FLAT priors
  lbfgs / adam  on compare_optimizers.Model's unconstrained vector (log mu,
         logit kappa, logit r_f, w logits, per fragtype log S_f upper triangle
         and pi_f logits) through reverse-mode autodiff of the 1D forward
The score-identity gradient (the same assembly as compare_optimizers.
vjp_value_grad, fed the 1D statistics) is checked against that autodiff.
  --corpus pfam           the same 1,000 aligned Pfam cherries (as path pairs)
  --corpus sim_mixfrag_s  MixFrag-S cherries simulated at SIM_TRUTH["sim_mixfrag_s"]
                          with the per-fragtype substitution truth of
                          compare_optimizers.mixfrag_s_truth_subst
Outputs: results_aligned*/pfam_mixfrag_s{F}.json, sim_mixfrag_s_mixfrag_s{F}.json
(convergence_figure.py reads these), same trajectory layout plus the gradcheck /
final_params entries of compare_optimizers.py.

Each run writes a JSON with per-evaluation trajectories, as compare_optimizers.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
# compare_optimizers (imported below for the MixFrag-S pieces) caps the FB pad bins at 256 for its 2D
# tables; the 1D path objective pads alignment COLUMNS, which the simulated corpora do not cap, so set
# a generous cap first (the count-based rows never pad anything, so this does not touch them).
os.environ.setdefault("TKFMIXDOM_MAX_PAD", "4096")
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PY)
sys.path.insert(0, os.path.join(PY, "experiments"))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from build_tkf92_cherry_counts import AA, TAU_MIN, TAU_MAX, AMINO_ACIDS   # noqa: E402
from build_mixfrag_cherry_counts import (                     # noqa: E402
    count_cherry_mixfrag, geom_bin_edges, discretize_tau,
)
from tkfmixdom.jax.core.params import tkf91_trans            # noqa: E402
from tkfmixdom.jax.train.mixfrag_cherry_em import (          # noqa: E402
    em_fit, prep_bins, _bin_indel_logZ, _bin_args,
)
from tkfmixdom.jax.simulate.mixfrag_sim import simulate_mixfrag_cherries  # noqa: E402
from tkfmixdom.jax.core.params import M as TYPE_M, I as TYPE_I, D as TYPE_D, mixfrag_trans   # noqa: E402
from tkfmixdom.jax.train.mixfrag_svi_bw import (                              # noqa: E402
    path_pair_from_decoded, bucket_pairs_lazy_path, stack_bucket_lazy_path,
    estep_batch_mixfrag_s_path, _estep_pair_mixfrag_s_path_forward_only_core,
)
import compare_optimizers as co   # noqa: E402  (Model / M-steps / init of the MixFrag-S rows)

CANONICAL_VAL_SEED = 0xCAFE


# ---------------------------------------------------------------------------
# Corpora (aligned rows)
# ---------------------------------------------------------------------------

def rows_from_states(x_int, y_int, states):
    a, b, i, j = [], [], 0, 0
    for s in states:
        if s == 1:      # M
            a.append(AMINO_ACIDS[x_int[i]]); b.append(AMINO_ACIDS[y_int[j]]); i += 1; j += 1
        elif s == 2:    # I
            a.append('-'); b.append(AMINO_ACIDS[y_int[j]]); j += 1
        elif s == 3:    # D
            a.append(AMINO_ACIDS[x_int[i]]); b.append('-'); i += 1
        else:
            raise ValueError(s)
    assert i == len(x_int) and j == len(y_int)
    return ''.join(a), ''.join(b)


def load_pfam_aligned(n_pairs, max_len, seed, as_path=False):
    """Same sampling as run_tkf92_2dfb_pfam.load_breadth_first_pairs (train
    side), but keeping the alignment.  Returns (row_a, row_b, t) cherries, or
    with as_path=True the same pairs as 'path' tuples (states, anc_full,
    desc_full, t) for the MixFrag-S 1D objective."""
    from train_pfam import PrecompiledPairSource
    src = PrecompiledPairSource(os.path.join(PY, "pfam/precompiled"), max_alignment_len=max_len)
    by_family = defaultdict(list)
    for item in src._decode_all():
        x_int, y_int, states, ac, dc, t_est, fam = item
        by_family[fam].append((x_int, y_int, states, ac, dc, float(t_est)))
    families = sorted(by_family.keys())
    n_val_fam = int(round(len(families) * 0.05))
    train_families = families[n_val_fam:]
    rng_train = np.random.default_rng(seed)
    rng_train.shuffle(train_families)
    per_family_q = {f: list(by_family[f]) for f in train_families}
    for f in per_family_q:
        rng_train.shuffle(per_family_q[f])
    out = []
    any_left = True
    while any_left and len(out) < n_pairs:
        any_left = False
        for f in train_families:
            if not per_family_q[f]:
                continue
            out.append(per_family_q[f].pop())
            any_left = True
            if len(out) >= n_pairs:
                break
    if as_path:
        return [path_pair_from_decoded(x, y, st, ac, dc, t) for x, y, st, ac, dc, t in out]
    return [(*rows_from_states(x, y, st), t) for x, y, st, _ac, _dc, t in out]


def path_pair_from_rows(row_a, row_b, t):
    """'path' tuple (states int8, anc_full, desc_full, t) of an aligned cherry
    given as two gapped rows ('-' = gap); residues outside AMINO_ACIDS map to
    the wildcard index AA (emits 1)."""
    idx = {c: i for i, c in enumerate(AMINO_ACIDS)}
    L = len(row_a)
    assert len(row_b) == L
    states = np.zeros(L, np.int8); anc = np.zeros(L, np.int8); desc = np.zeros(L, np.int8)
    for l, (a, b) in enumerate(zip(row_a, row_b)):
        if a == '-':
            states[l] = TYPE_I; desc[l] = idx.get(b, AA)
        elif b == '-':
            states[l] = TYPE_D; anc[l] = idx.get(a, AA)
        else:
            states[l] = TYPE_M; anc[l] = idx.get(a, AA); desc[l] = idx.get(b, AA)
    return states, anc, desc, float(t)


SIM_TRUTH = {
    "sim_tkf92":   dict(lam=0.058, mu=0.060, exts=[0.75], weights=[1.0]),
    "sim_mixfrag": dict(lam=0.033924, mu=0.034690, exts=[0.41396, 0.86031], weights=[0.70086, 0.29914]),
    # MixFrag-S F=2 (same truth as compare_optimizers.SIM_TRUTH["sim_mixfrag_s"]): fitted MixFrag F=2 indel
    # parameters; fragtype 2 substitutes under pi tilted toward P, G, S, D, E by 1.6 (mixfrag_s_tilted_subst)
    "sim_mixfrag_s": dict(lam=0.04645159361484088, mu=0.04747838153710662,
                          exts=[0.4307407767585113, 0.8558260528890574],
                          weights=[0.7097808771217816, 0.2902191228782184]),
}


def simulate_aligned(n_pairs, seed, corpus="sim_tkf92", subst_npz=""):
    """Aligned cherries simulated from the named model (TKF92 = MixFrag F=1) at
    Pfam-like branch lengths, drawn from the sim_tkf92 corpus's own t values."""
    truth = SIM_TRUTH[corpus]
    lam, mu = truth["lam"], truth["mu"]
    exts, weights = np.array(truth["exts"]), np.array(truth["weights"])
    import pickle
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    if subst_npz:   # e.g. the single GTR refitted to the training pairs
        d = np.load(subst_npz, allow_pickle=True); Q = np.asarray(d["Q"], float); pi = np.asarray(d["pi"], float); pi = pi / pi.sum()
    else:
        Q, pi = rate_matrix_lg()
        Q, pi = np.asarray(Q), np.asarray(pi)
    pairs = pickle.load(open(os.path.join(PY, "experiments/2dfb_sim/sim_tkf92/train.pkl"), "rb"))
    ts = np.array([p[2] for p in pairs])
    rng = np.random.default_rng(seed)
    if corpus == "sim_mixfrag_s":
        from tkfmixdom.jax.simulate.mixfrag_sim import simulate_mixfrag_s_cherries, mixfrag_s_tilted_subst
        Qs, pis = mixfrag_s_tilted_subst(Q, pi, tilt="PGSDE", factor=1.6, F=len(exts))
        def sim(rng, k, t):
            return simulate_mixfrag_s_cherries(rng, k, lam, mu, exts, weights, t, Qs, pis)
    else:
        def sim(rng, k, t):
            return simulate_mixfrag_cherries(rng, k, lam, mu, exts, weights, t, Q, pi)
    out = []
    # simulate in small groups sharing a t so the sampler's per-call setup is amortised
    while len(out) < n_pairs:
        t = float(rng.choice(ts))
        if t <= 0:
            continue
        sims = sim(rng, 10, t)
        out += [(s["row_a"], s["row_b"], s["t"]) for s in sims]
    return out[:n_pairs]


def build_counts(cherries, n_bins):
    edges, centers = geom_bin_edges(n_bins, TAU_MIN, TAU_MAX)
    mc = np.zeros((n_bins, AA, AA), np.int64)
    sc = np.zeros((n_bins, AA), np.int64)
    scal = np.zeros((n_bins, 3), np.int64)
    mr, gm, ge = {}, {}, {}
    for sa, sb, t in cherries:
        tb = int(discretize_tau(t, edges))
        count_cherry_mixfrag(sa, sb, tb, mc, sc, scal, mr, gm, ge)
    return dict(n_tau_bins=n_bins, tau_centers=centers, scalars=scal,
                match_counts=mc, singlet_counts=sc,
                match_run=mr, gap_to_match=gm, gap_to_end=ge,
                n_pairs=len(cherries))


# ---------------------------------------------------------------------------
# Objective for the gradient methods
# ---------------------------------------------------------------------------

class Model:
    def __init__(self, F):
        self.F = F

    def pack(self, p):
        lam, mu = p["lam"], p["mu"]
        v = [np.log(mu), np.log(lam / mu / (1 - lam / mu))]
        v += [np.log(r / (1 - r)) for r in p["exts"]]
        if self.F > 1:
            w = np.asarray(p["weights"])
            v += list(np.log(w) - np.log(w[-1]))
        return jnp.asarray(np.array(v, np.float64))

    def unpack(self, v):
        mu = jnp.exp(v[0])
        kappa = jax.nn.sigmoid(v[1])
        lam = kappa * mu
        exts = jax.nn.sigmoid(v[2:2 + self.F])
        weights = jax.nn.softmax(v[2 + self.F:2 + 2 * self.F]) if self.F > 1 else jnp.ones(1)
        return lam, mu, exts, weights

    def natural(self, v):
        lam, mu, exts, weights = self.unpack(jnp.asarray(v))
        return dict(lam=float(lam), mu=float(mu),
                    exts=[float(e) for e in np.asarray(exts)],
                    weights=[float(w) for w in np.asarray(weights)])


def make_objective(model, bins):
    """Returns f(v) -> (LL, grad) summing the per-bin log-partition; one jitted
    value_and_grad per bin shape (cached by XLA on shapes)."""
    def bin_ll(v, t, Nmatch, NgapM, NgapE, Nstart, Nend, Nempty):
        lam, mu, exts, weights = model.unpack(v)
        tau = tkf91_trans(lam, mu, t)
        return _bin_indel_logZ(tau, weights, exts, 1.0 - exts,
                               Nmatch, NgapM, NgapE, Nstart, Nend, Nempty)
    vg = jax.jit(jax.value_and_grad(bin_ll))
    args = [(jnp.asarray(float(bn["t"])),) + _bin_args(bn) for bn in bins]

    def f(v):
        ll = 0.0
        g = jnp.zeros_like(v)
        for a in args:
            l, gg = vg(v, *a)
            ll += float(l)
            g = g + gg
        return ll, g
    return f


def fmt(p):
    s = f"lam={p['lam']:.5f} mu={p['mu']:.5f} r=" + ",".join(f"{e:.4f}" for e in p["exts"])
    if len(p["weights"]) > 1:
        s += " w=" + ",".join(f"{w:.4f}" for w in p["weights"])
    if "pi" in p:   # mixfrag_s: how far the per-fragtype equilibria have moved apart
        pis = np.asarray(p["pi"], np.float64)
        s += " max|pi_f-pi_1|=" + ",".join(f"{np.max(np.abs(pis[f] - pis[0])):.3f}" for f in range(1, pis.shape[0]))
    return s


def _rec(model, v):
    """Per-pass parameter record at the unconstrained point v (the MixFrag-S
    Model keeps its matrices out of the trajectory; see final_params)."""
    p = model.natural(v)
    return model.record(p) if hasattr(model, "record") else p


# ---------------------------------------------------------------------------
# MixFrag-S: 1D path-conditioned objective, E-step and score-identity gradient
# ---------------------------------------------------------------------------

def make_path_chunks(pairs, chunk):
    """Bucket 'path' pairs by padded column count and cut into fixed-size
    chunks (the last chunk of a bucket is padded by repetition and masked by
    w).  Chunk = (states, anc, desc, L_real, ts, w), all (chunk, ...) arrays."""
    buckets = bucket_pairs_lazy_path(pairs)
    chunks = []
    for L_pad in sorted(buckets):
        lst = buckets[L_pad]
        for i in range(0, len(lst), chunk):
            sub = lst[i:i + chunk]
            st, an, de, Ls, ts = stack_bucket_lazy_path(sub, L_pad, n_pad_to=chunk)
            w = np.zeros(chunk, np.float64); w[:len(sub)] = 1.0
            chunks.append(tuple(jnp.asarray(a) for a in (st, an, de, Ls, ts, w)))
    return chunks, len(buckets)


def make_objective_mixfrag_s(model, scan_mode):
    """Per-chunk jitted (LL_sum, dLL/dv) by reverse-mode autodiff of the 1D
    path-conditioned forward (expm of each Q_f, the per-column emission table,
    the forward recursion along the alignment's columns)."""
    F = model.F

    def per_pair(v, st, an, de, L, t):
        lam, mu, exts, weights = model.unpack(v)
        _Ss, pis, Qs = model.unpack_subst(v)
        lp = _estep_pair_mixfrag_s_path_forward_only_core(
            st, an, de, L, t, lam, mu, exts, weights, Qs, pis, scan_mode=scan_mode)
        # An empty alignment (L = 0 columns; the simulated corpora keep such cherries, as the count-based
        # rows do through Nempty) has P = tau[S, E].  The full forward-backward of the E-step route handles
        # seq_length == 0, but the library's forward-only kernel returns NEG_INF there, so patch it here.
        lp_empty = jnp.log(jnp.maximum(mixfrag_trans(lam, mu, t, exts, weights)[0, 3 * F + 1], 1e-30))
        return jnp.where(L == 0, lp_empty, lp)
    per_pair = jax.checkpoint(per_pair)

    def chunk_ll(v, st, an, de, Ls, ts, w):
        lp = jax.vmap(per_pair, in_axes=(None, 0, 0, 0, 0, 0))(v, st, an, de, Ls, ts)
        return jnp.sum(w * lp)
    return jax.jit(jax.value_and_grad(chunk_ll))


def full_value_grad_chunks(vg, v, chunks):
    ll = 0.0
    g = jnp.zeros_like(v)
    for c in chunks:
        l, gg = vg(v, *c)
        ll += float(l)
        g = g + gg
    return ll, g


def em_pass_mixfrag_s(chunks, p, scan_mode):
    """One full-batch 1D path-conditioned E-step at the natural parameters p.
    Returns (LL_sum, suff) with the sufficient statistics of
    compare_optimizers.em_pass for mixfrag_s: B, D, S, L, M, T, F_f, E_f, and
    the per-fragtype W_f (A,), U_f (A,A), V_f = anc + ins + del (A,)."""
    ll = 0.0
    suff = None
    for st, an, de, Ls, ts, w in chunks:
        out = estep_batch_mixfrag_s_path(st, an, de, Ls, ts, p["lam"], p["mu"],
                                         np.asarray(p["exts"]), np.asarray(p["weights"]),
                                         np.asarray(p["Q"]), np.asarray(p["pi"]), scan_mode=scan_mode)
        ll += float(jnp.sum(w * out["log_p"]))
        agg = {k: float(jnp.sum(w * out[k])) for k in ("B", "D", "S", "L", "M", "T")}
        agg["F_f"] = np.asarray(jnp.sum(w[:, None] * out["F_f"], axis=0))
        agg["E_f"] = np.asarray(jnp.sum(w[:, None] * out["E_f"], axis=0))
        agg["W"] = np.asarray(jnp.einsum("b,bfa->fa", w, out["W"]))
        agg["U"] = np.asarray(jnp.einsum("b,bfij->fij", w, out["U"]))
        agg["V"] = np.asarray(jnp.einsum("b,bfa->fa", w,
                                         jnp.sum(out["mc"], axis=-1) + out["ins"] + out["del"]))
        if suff is None:
            suff = agg
        else:
            for k in suff:
                suff[k] = suff[k] + agg[k]
    return ll, suff


def q_function_mixfrag_s(model, suff, vv):
    """Expected complete-data log-likelihood (Q-function) of MixFrag-S at vv
    given the E-step statistics: the same assembly as the q_fn inside
    compare_optimizers.vjp_value_grad (indel part shared with mixfrag, then
    sum_f [ sum_{i!=j} U_f log Q_f + sum_i W_f[i] Q_f[i,i] + sum_a V_f log pi_f ]),
    here fed the 1D path-conditioned statistics.  Its gradient at the
    current parameters is the gradient of the observed-data log-likelihood
    (Fisher's identity)."""
    B, D, S, L, M, T = (suff[k] for k in ("B", "D", "S", "L", "M", "T"))
    lam, mu, exts, weights = model.unpack(vv)
    q = (B * jnp.log(lam) - lam * (S + T) + D * jnp.log(mu) - mu * S
         + L * jnp.log(lam / mu) + M * jnp.log(1.0 - lam / mu))
    F_f = jnp.asarray(suff["F_f"]); E_f = jnp.asarray(suff["E_f"])
    q = q + jnp.sum(F_f * jnp.log(exts) + E_f * jnp.log(1.0 - exts) + E_f * jnp.log(weights))
    Ss, pis, Qs = model.unpack_subst(vv)
    eye = jnp.eye(model.A, dtype=bool)[None]
    W = jnp.asarray(suff["W"]); U = jnp.asarray(suff["U"]); V = jnp.asarray(suff["V"])
    log_Q_off = jnp.log(jnp.where(eye, 1.0, Qs))
    return (q + jnp.sum(jnp.where(eye, 0.0, U * log_Q_off))
            + jnp.sum(W * jnp.diagonal(Qs, axis1=-2, axis2=-1))
            + jnp.sum(V * jnp.log(pis)))


def vjp_value_grad_mixfrag_s(model, chunks, v, scan_mode):
    """(LL_sum, dLL/dv) from one 1D E-step and the score-identity gradient."""
    ll, suff = em_pass_mixfrag_s(chunks, model.natural(v), scan_mode)
    g = jax.grad(lambda vv: q_function_mixfrag_s(model, suff, vv))(jnp.asarray(v))
    return ll, g


def run_em_mixfrag_s(model, chunks, N, p0, max_iter, tol, log, scan_mode, final=None):
    """Full-batch EM: 1D path E-step + compare_optimizers.em_mstep (flat priors).
    Trajectory ll at pass k is the objective at the parameters BEFORE that
    pass's M-step; stops when the per-pair gain falls below tol."""
    traj = []
    p = dict(p0)
    t0 = time.time()
    prev = -np.inf
    for k in range(max_iter):
        ll, suff = em_pass_mixfrag_s(chunks, p, scan_mode)
        ll /= N
        traj.append({"pass": k + 1, "wall": time.time() - t0, "ll": ll, "params": model.record(p)})
        if k % 10 == 0 or k == max_iter - 1:
            log(f"  em   pass {k+1:4d} ll={ll:.6f} {fmt(p)} ({time.time()-t0:.1f}s)")
        if ll - prev < tol and k > 0:
            log(f"  em   converged (dLL={ll-prev:.2e} < {tol}) at pass {k+1}")
            break
        prev = ll
        p = co.em_mstep(model, suff, p)
    if final is not None:
        final.update(model.record_full(p))
    return traj


# ---------------------------------------------------------------------------
# Runners (trajectory records: pass, wall, ll per cherry at the params BEFORE
# the update, params)
# ---------------------------------------------------------------------------

def run_em(agg, F, p0, N, max_iter, tol, log, max_gap):
    t0 = time.time()
    times = []

    def log_fn(msg):
        times.append(time.time() - t0)
        log("  em   " + msg)

    # (Q, pi) only enter the constant substitution term; any proper model will do.
    pi = np.ones(AA) / AA
    Q = np.ones((AA, AA)) - AA * np.eye(AA)
    out = em_fit(agg, n_fragtypes=F, Q=Q, pi=pi, n_iter=max_iter, tol=tol * N,
                 init_lam=p0["lam"], init_mu=p0["mu"],
                 init_exts=list(p0["exts"]), init_weights=list(p0["weights"]),
                 prior_alpha_lam=1.0, prior_alpha_mu=1.0, prior_beta=0.0,
                 ext_prior_alpha=1.0, ext_prior_beta=1.0, weight_prior_alpha=1.0,
                 max_gap=max_gap, log_fn=log_fn, log_every=1)
    wall = time.time() - t0
    hist = out["history"]
    traj = []
    prev = dict(p0)
    for k, h in enumerate(hist):
        # indel_ll at iteration k is evaluated at the parameters BEFORE that
        # iteration's M-step (the init for k=0).
        traj.append({"pass": k + 1, "wall": wall * (k + 1) / len(hist),
                     "ll": float(h["indel_ll"]) / N, "params": prev})
        prev = dict(lam=float(h["lam"]), mu=float(h["mu"]),
                    exts=[float(e) for e in np.asarray(h["exts"])],
                    weights=[float(w) for w in np.asarray(h["weights"])])
    log(f"  em   {len(hist)} passes, {wall:.1f}s, final {fmt(prev)}")
    return traj, prev


def run_adam(f, model, p0, N, lr, max_iter, tol, log, final=None):
    import optax
    v = model.pack(p0)
    opt = optax.adam(lr)
    state = opt.init(v)
    traj, best, stall = [], -np.inf, 0
    t0 = time.time()
    for k in range(max_iter):
        ll, g = f(v)
        ll /= N
        traj.append({"pass": k + 1, "wall": time.time() - t0, "ll": ll, "params": _rec(model, v)})
        if k % 100 == 0:
            log(f"  adam(lr={lr}) pass {k+1:4d} ll={ll:.6f} {fmt(_rec(model, v))} ({time.time()-t0:.1f}s)")
        if ll > best + tol:
            best, stall = ll, 0
        else:
            stall += 1
            if stall >= 300:
                log(f"  adam(lr={lr}) stalled at pass {k+1}")
                break
        upd, state = opt.update(-g / N, state, v)
        v = optax.apply_updates(v, upd)
    if final is not None:
        final.update(model.record_full(model.natural(v)))
    return traj


def run_lbfgs(f, model, p0, N, max_fun, log, final=None):
    import scipy.optimize
    traj = []
    t0 = time.time()
    n = [0]

    def fun(x):
        v = jnp.asarray(x)
        ll, g = f(v)
        n[0] += 1
        traj.append({"pass": n[0], "wall": time.time() - t0, "ll": ll / N, "params": _rec(model, v)})
        return -ll / N, -np.asarray(g, np.float64) / N

    res = scipy.optimize.minimize(fun, np.asarray(model.pack(p0), np.float64), jac=True,
                                  method="L-BFGS-B",
                                  options=dict(maxfun=max_fun, maxiter=max_fun, ftol=0.0,
                                               gtol=1e-10, maxcor=20))
    log(f"  lbfgs done: {res.message} nit={res.nit} nfev={res.nfev} ll={-res.fun:.6f} "
        f"{fmt(_rec(model, jnp.asarray(res.x)))}")
    if final is not None:
        final.update(model.record_full(model.natural(jnp.asarray(res.x))))
    return traj


def main_mixfrag_s(args, log):
    """--model mixfrag_s: EM vs L-BFGS vs Adam on the 1D path-conditioned
    log-likelihood of the aligned pairs (see the module docstring)."""
    log(f"jax devices: {jax.devices()}")
    Q, pi = co.load_subst(args.subst_npz)
    model = co.Model("mixfrag_s", n_fragtypes=args.n_fragtypes, A=pi.shape[0])
    F = model.F
    t0 = time.time()
    if args.corpus == "pfam":
        pairs = load_pfam_aligned(args.n_pairs, args.max_len, args.seed, as_path=True)
    else:
        cherries = simulate_aligned(args.n_pairs, args.seed, corpus=args.corpus, subst_npz=args.subst_npz)
        pairs = [path_pair_from_rows(a, b, t) for a, b, t in cherries]
    N = len(pairs)
    chunks, n_buckets = make_path_chunks(pairs, args.chunk)
    lens = np.array([int(np.asarray(p[0]).shape[0]) for p in pairs])
    log(f"corpus {args.corpus}: N={N} aligned cherries as path pairs, mean alignment length {lens.mean():.1f} "
        f"max={lens.max()}, {n_buckets} column-count buckets, {len(chunks)} chunks of {args.chunk}; "
        f"built in {time.time()-t0:.1f}s")

    exts0 = args.init_exts if args.init_exts is not None else ([0.3, 0.6] if F == 2 else list(np.linspace(0.3, 0.8, F)))
    w0 = args.init_weights if args.init_weights is not None else [1.0 / F] * F
    assert len(exts0) == F and len(w0) == F, (exts0, w0, F)
    p0 = dict(lam=args.init_lam, mu=args.init_mu, exts=[float(e) for e in exts0],
              weights=[float(w) / float(np.sum(w0)) for w in w0])
    S0 = co.S_from_Q_pi(Q, pi)   # every fragtype starts at the fixed (Q, pi) of the other rows
    p0["S"] = np.tile(S0[None], (F, 1, 1)); p0["pi"] = np.tile(pi[None], (F, 1))
    p0["Q"] = np.asarray(jax.vmap(co.build_Q_from_S_pi)(jnp.asarray(p0["S"]), jnp.asarray(p0["pi"])), np.float64)
    log(f"mixfrag_s: F={F}, A={model.A}, {model.n_params} unconstrained parameters "
        f"({model.n_indel} indel + {F} x ({model.n_tri} log S + {model.A} pi logits)); "
        f"max|Q_f - Q| at init = {np.max(np.abs(p0['Q'] - Q[None])):.2e}; scan_mode={args.fb_scan_mode}")
    log(f"init: {fmt(p0)}")

    vg = make_objective_mixfrag_s(model, args.fb_scan_mode)

    def f(v):
        return full_value_grad_chunks(vg, v, chunks)

    # Warm every JIT shape of both routes before timing anything.
    t0 = time.time()
    v0 = model.pack(p0)
    ll_g, _ = f(v0)
    ll_e, _ = vjp_value_grad_mixfrag_s(model, chunks, v0, args.fb_scan_mode)
    log(f"warm-up {time.time()-t0:.1f}s; LL/pair at init: autodiff path {ll_g/N:.6f}, "
        f"E-step path {ll_e/N:.6f} (should agree)")

    def gradcheck(v, tag, chunks_, n_pairs_):
        _, ga = full_value_grad_chunks(vg, v, chunks_)
        _, gv = vjp_value_grad_mixfrag_s(model, chunks_, v, args.fb_scan_mode)
        ga = np.asarray(ga); gv = np.asarray(gv)
        rel = float(np.max(np.abs(ga - gv)) / max(np.max(np.abs(ga)), 1e-300))
        per = {}
        for name, a, b in model.blocks():
            per[name] = float(np.max(np.abs(ga[a:b] - gv[a:b])) / max(np.max(np.abs(ga[a:b])), 1e-300))
        log(f"gradcheck [{tag}, {n_pairs_} pairs]: {ga.shape[0]} parameters, max|g| = {np.max(np.abs(ga)):.4e}, "
            f"max|autodiff - score-identity|/max|autodiff| = {rel:.2e}")
        log("    per block: " + " ".join(f"{k}={e:.2e}" for k, e in per.items()))
        return dict(n_pairs=n_pairs_, autodiff=ga.tolist(), vjp=gv.tolist(), max_rel_diff=rel,
                    max_rel_diff_per_block=per)
    gc = {"init": gradcheck(v0, "init", chunks, N)}
    sub = pairs[:args.gradcheck_pairs]
    sub_chunks, _ = make_path_chunks(sub, args.chunk)
    rng_sub = np.random.default_rng(1)
    for tag, vv in (("init", v0), ("random", v0 + 0.5 * rng_sub.standard_normal(v0.shape[0]))):
        gc[f"subset_{tag}"] = gradcheck(jnp.asarray(vv), f"subset_{tag}", sub_chunks, len(sub))
    rng_gc = np.random.default_rng(0)
    for k in range(2):
        vk = v0 + 0.5 * rng_gc.standard_normal(v0.shape[0])
        gc[f"random{k}"] = gradcheck(jnp.asarray(vk), f"random{k}", chunks, N)
    t0 = time.time(); f(v0); t_grad = time.time() - t0
    t0 = time.time(); em_pass_mixfrag_s(chunks, p0, args.fb_scan_mode); t_em = time.time() - t0
    t0 = time.time(); vjp_value_grad_mixfrag_s(model, chunks, v0, args.fb_scan_mode); t_vjp = time.time() - t0
    log(f"timed pass: value+grad (autodiff) {t_grad:.3f}s, E-step {t_em:.3f}s, "
        f"E-step + closed-form gradient {t_vjp:.3f}s")

    results = dict(args=vars(args), N=N, n_buckets=n_buckets, model="mixfrag_s", n_fragtypes=F,
                   n_params=model.n_params, objective="path_conditioned_1d",
                   ll_init_per_pair=ll_g / N,
                   pass_seconds=dict(grad=t_grad, estep=t_em, vjp=t_vjp),
                   gradcheck=gc, methods={}, final_params={})

    def save():
        json.dump(results, open(args.out, "w"), indent=1, default=float)

    def final(key):
        return results["final_params"].setdefault(key, {})

    for m in args.methods:
        log(f"== {m} ==")
        if m == "em":
            results["methods"]["em"] = run_em_mixfrag_s(model, chunks, N, p0, args.em_max_iter, args.tol, log,
                                                        args.fb_scan_mode, final=final("em"))
        elif m == "lbfgs":
            results["methods"]["lbfgs"] = run_lbfgs(f, model, p0, N, args.lbfgs_max_fun, log, final=final("lbfgs"))
        elif m == "adam":
            for lr in args.adam_lrs:
                results["methods"][f"adam_lr{lr:g}"] = run_adam(f, model, p0, N, lr, args.adam_max_iter, args.tol,
                                                              log, final=final(f"adam_lr{lr:g}"))
                save()
        save()
    log(f"saved {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["pfam", "sim_tkf92", "sim_mixfrag", "sim_mixfrag_s"], required=True)
    ap.add_argument("--model", choices=["tkf92", "mixfrag", "mixfrag_s"], required=True)
    ap.add_argument("--n-fragtypes", type=int, default=2, help="F of --model mixfrag_s (mixfrag is F=2)")
    ap.add_argument("--methods", nargs="+", default=["em", "lbfgs", "adam"])
    ap.add_argument("--adam-lrs", type=float, nargs="+", default=[0.1, 0.01])
    ap.add_argument("--n-pairs", type=int, default=1000)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--n-bins", type=int, default=32)
    ap.add_argument("--max-gap", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-lam", type=float, default=0.02)
    ap.add_argument("--init-mu", type=float, default=0.025)
    ap.add_argument("--init-ext", type=float, default=0.4)
    ap.add_argument("--init-exts", type=float, nargs="+", default=None,
                    help="F values; default 0.3 0.6 for F=2, else linspace(0.3, 0.8, F)")
    ap.add_argument("--init-weights", type=float, nargs="+", default=None, help="F values; default uniform")
    ap.add_argument("--em-max-iter", type=int, default=2000)
    ap.add_argument("--adam-max-iter", type=int, default=3000)
    ap.add_argument("--lbfgs-max-fun", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--subst-npz", default="", help="npz with Q, pi used to simulate the corpus instead of LG08 "
                    "(for mixfrag_s also the initial (S_f, pi_f) of every fragtype)")
    ap.add_argument("--chunk", type=int, default=32, help="mixfrag_s: pairs per jitted 1D kernel call")
    ap.add_argument("--fb-scan-mode", choices=["associative", "sequential"], default="associative",
                    help="mixfrag_s: 1D forward-backward scan implementation")
    ap.add_argument("--gradcheck-pairs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def log(m):
        print(m, flush=True)

    log(f"args: {vars(args)}")
    if args.model == "mixfrag_s":
        return main_mixfrag_s(args, log)
    t0 = time.time()
    if args.corpus == "pfam":
        cherries = load_pfam_aligned(args.n_pairs, args.max_len, args.seed)
    else:
        cherries = simulate_aligned(args.n_pairs, args.seed, corpus=args.corpus, subst_npz=args.subst_npz)
    N = len(cherries)
    agg = build_counts(cherries, args.n_bins)
    bins = prep_bins(agg, max_gap=args.max_gap, log_fn=lambda *a, **k: None)
    lens = [len(a) for a, _, _ in cherries]
    log(f"corpus {args.corpus}: N={N} aligned cherries, mean alignment length "
        f"{np.mean(lens):.1f}, {len(bins)} non-empty tau bins; built in {time.time()-t0:.1f}s")

    F = 2 if args.model == "mixfrag" else 1
    model = Model(F)
    if F == 1:
        p0 = dict(lam=args.init_lam, mu=args.init_mu, exts=[args.init_ext], weights=[1.0])
    else:
        exts0 = args.init_exts if args.init_exts is not None else [0.3, 0.6]
        w0 = args.init_weights if args.init_weights is not None else [0.5, 0.5]
        p0 = dict(lam=args.init_lam, mu=args.init_mu, exts=list(exts0), weights=list(w0))
    log(f"init: {fmt(p0)}")

    f = make_objective(model, bins)
    t0 = time.time(); ll0, _ = f(model.pack(p0)); t_warm = time.time() - t0
    t0 = time.time(); f(model.pack(p0)); t_grad = time.time() - t0
    log(f"warm-up {t_warm:.1f}s; LL/cherry at init {ll0/N:.6f}; timed gradient pass {t_grad:.3f}s")

    results = dict(args=vars(args), N=N, ll_init_per_pair=ll0 / N,
                   pass_seconds=dict(grad=t_grad), methods={})

    def save():
        json.dump(results, open(args.out, "w"), indent=1, default=float)

    for m in args.methods:
        log(f"== {m} ==")
        if m == "em":
            traj, _ = run_em(agg, F, p0, N, args.em_max_iter, args.tol, log, args.max_gap)
            results["methods"]["em"] = traj
            results["pass_seconds"]["estep"] = traj[-1]["wall"] / len(traj) if traj else None
        elif m == "lbfgs":
            results["methods"]["lbfgs"] = run_lbfgs(f, model, p0, N, args.lbfgs_max_fun, log)
        elif m == "adam":
            for lr in args.adam_lrs:
                results["methods"][f"adam_lr{lr:g}"] = run_adam(f, model, p0, N, lr,
                                                              args.adam_max_iter, args.tol, log)
                save()
        save()
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
