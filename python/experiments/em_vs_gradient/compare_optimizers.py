#!/usr/bin/env python3
"""Closed-form EM vs gradient-based optimization of the TKF joint likelihood.

Full-batch (deterministic) comparison on one fixed corpus of unaligned
sequence pairs.  All methods maximise the SAME objective,

    LL(theta) = (1/N) sum_pairs log P(x, y | theta, t)   (2D pair-HMM forward),

with the substitution model (LG08) held fixed, starting from the SAME
initial parameters, and are measured by (i) the number of passes over the
corpus (an EM iteration = one forward-backward pass + closed-form M-step; an
Adam step or an L-BFGS function evaluation = one forward pass + reverse-mode
gradient) and (ii) wall-clock on a warm JIT cache.

Methods
  em     closed-form Baum-Welch: BDI sufficient statistics from the
         forward-backward chi matrix, kappa-quadratic / Beta / Dirichlet
         M-steps with FLAT priors (so the fixed point is the MLE).
  adam   Adam on unconstrained parameters (log mu, logit kappa, logit r_f,
         softmax logits w_f) through reverse-mode autodiff of the forward DP.
  lbfgs  scipy L-BFGS-B on the same unconstrained parameters and gradient.

Models: tkf91 (lam, mu), tkf92 (lam, mu, r), mixfrag F=2 (lam, mu, r_1, r_2, w),
        mixfrag_s (--n-fragtypes F, default 2): MixFrag-S, the MixFrag pair HMM in
        which fragtype f owns its own reversible substitution model (S_f, pi_f),
        Q_f = build_Q_from_S_pi(S_f, pi_f) in raw units.  Its unconstrained vector
        is (log mu, logit kappa, logit r_f, w logits, and per fragtype the log of
        the A(A-1)/2 upper-triangular exchangeabilities and A softmax logits of
        pi_f), initialised from the fixed (Q, pi) of --subst-npz tiled F times.
        Its EM = the unaligned 2D MixFrag-S E-step (indel counts + per-fragtype
        Holmes-Rubin (W_f, U_f) at each pair's own t + residue counts V_f) and
        the closed-form M-steps with flat priors (pi_pseudo 0, S_pseudo 0); its
        score-identity gradient adds, to the indel part shared with mixfrag,
        the gradient of  sum_f [ sum_{i!=j} U_f[i,j] log Q_f[i,j]
        + sum_i W_f[i] Q_f[i,i] + sum_a V_f[a] log pi_f[a] ]  taken by jax.grad
        through Q_f = S_f pi_f to the unconstrained parameters (vjp_value_grad).
Corpora: --corpus sim_tkf92 (experiments/2dfb_sim/sim_tkf92/train.pkl),
         --corpus sim_tkf91 / sim_mixfrag / sim_mixfrag_s (re-simulated under
         SIM_TRUTH at that corpus's Pfam-like branch lengths; sim_mixfrag_s is
         MixFrag-S F=2 with a per-fragtype substitution model, see
         mixfrag_s_truth_subst), or
         --corpus pfam (breadth-first sample of Pfam cherries, as in
         experiments/2dfb/benchmark.md).

Each run writes a JSON with the per-evaluation trajectory
(pass index, wall-clock, LL/pair, parameters) of every method, so
"iterations to tolerance" is computed post hoc by summarize.py.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("TKFMIXDOM_MAX_PAD", "256")
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(os.path.dirname(HERE))      # ~/tkf-mixdom/python
sys.path.insert(0, PY)
sys.path.insert(0, os.path.join(PY, "experiments"))

import jax                                                     # noqa: E402
import jax.numpy as jnp                                        # noqa: E402

from tkfmixdom.jax.core.protein import rate_matrix_lg          # noqa: E402
from tkfmixdom.jax.train.tkf92_svi_bw import (                 # noqa: E402
    _bin_bucket_pairs, _stack_bucket, estep_batch_tkf92,
    _estep_pair_tkf92_forward_only_core, m_step_lam_mu, m_step_ext,
)
from tkfmixdom.jax.train.mixfrag_svi_bw import (               # noqa: E402
    estep_batch_mixfrag, _estep_pair_mixfrag_forward_only_core,
    m_step_ext_per_fragtype, m_step_weights,
    estep_batch_mixfrag_s, _estep_pair_mixfrag_s_forward_only_core,
    m_step_subst_per_fragtype,
)
from tkfmixdom.jax.core.ctmc import build_Q_from_S_pi          # noqa: E402


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

SIM_TRUTH = {   # model-matched simulated corpora (same Pfam-like t pool, LG08)
    "sim_tkf91":   dict(lam=0.058, mu=0.060, exts=[0.0], weights=[1.0]),
    # MixFrag F=2 at its Pfam-fitted values (params/best/cem_mixfrag_F2_pfamTrain)
    "sim_mixfrag": dict(lam=0.033924, mu=0.034690, exts=[0.41396, 0.86031], weights=[0.70086, 0.29914]),
    # MixFrag-S F=2: indel parameters of the fitted MixFrag F=2 (experiments/mixfrag_F/mixfrag_F2_logspace.json);
    # fragtype 1 substitutes under the fixed (Q, pi) of --subst-npz, fragtype 2 under the same exchangeabilities
    # with pi tilted toward P, G, S, D, E by MIXFRAG_S_TILT_FACTOR (mixfrag_s_tilted_subst)
    "sim_mixfrag_s": dict(lam=0.04645159361484088, mu=0.04747838153710662,
                          exts=[0.4307407767585113, 0.8558260528890574],
                          weights=[0.7097808771217816, 0.2902191228782184]),
}
MIXFRAG_S_TILT = "PGSDE"
MIXFRAG_S_TILT_FACTOR = 1.6
# TKF92 truth of experiments/2dfb_sim/sim_tkf92 (simulate_pfam_like.py); used when the corpus is re-simulated under
# a substitution model other than LG08 (--subst-npz), instead of loading that LG08 pickle
SIM_TRUTH_TKF92 = dict(lam=0.058, mu=0.060, exts=[0.75], weights=[1.0])


def load_subst(path):
    """(Q, pi) of the fixed substitution model: LG08 when path is empty, else Q and pi from an npz (e.g. the
    single GTR refitted to the training pairs, pfam/gtr_refit_Qpi.npz)."""
    if not path:
        Q, pi = rate_matrix_lg()
        return np.asarray(Q), np.asarray(pi)
    d = np.load(path, allow_pickle=True)
    pi = np.asarray(d["pi"], float); pi = pi / pi.sum()
    return np.asarray(d["Q"], float), pi


def S_from_Q_pi(Q, pi):
    """Symmetric exchangeabilities (zero diagonal) of a reversible Q: S[a,b] = Q[a,b] / pi[b], symmetrised
    (exact for a reversible Q, so build_Q_from_S_pi(S, pi) rebuilds Q)."""
    Q = np.asarray(Q, np.float64); pi = np.asarray(pi, np.float64)
    A = pi.shape[0]
    S = Q / np.maximum(pi[None, :], 1e-300)
    S[np.arange(A), np.arange(A)] = 0.0
    return 0.5 * (S + S.T)


def mixfrag_s_truth_subst(Q, pi, F=2):
    """Per-fragtype (Qs, pis) of the sim_mixfrag_s truth: fragtype 1 = (Q, pi), fragtype 2 = pi tilted toward
    MIXFRAG_S_TILT by MIXFRAG_S_TILT_FACTOR with Q rebuilt from the same exchangeabilities."""
    from tkfmixdom.jax.simulate.mixfrag_sim import mixfrag_s_tilted_subst
    return mixfrag_s_tilted_subst(Q, pi, tilt=MIXFRAG_S_TILT, factor=MIXFRAG_S_TILT_FACTOR, F=F)


def subst_tag(path):
    return "" if not path else "_" + os.path.splitext(os.path.basename(path))[0]


def simulate_corpus(name, n_pairs, seed, max_len, Q, pi, tag=""):
    """Unaligned pairs simulated from the named model under (Q, pi), with branch lengths drawn
    from the sim_tkf92 corpus's own (Pfam-derived) t values; cached to a pickle (tag names the substitution model)."""
    cache = os.path.join(PY, f"experiments/2dfb_sim/{name}_n{n_pairs}_seed{seed}_L{max_len}{tag}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    from tkfmixdom.jax.simulate.mixfrag_sim import simulate_mixfrag_cherries, simulate_mixfrag_s_cherries
    truth = SIM_TRUTH_TKF92 if name == "sim_tkf92" else SIM_TRUTH[name]
    base = pickle.load(open(os.path.join(PY, "experiments/2dfb_sim/sim_tkf92/train.pkl"), "rb"))
    ts = np.array([p[2] for p in base]); ts = ts[ts > 0]
    rng = np.random.default_rng(seed)
    if name == "sim_mixfrag_s":
        Qs, pis = mixfrag_s_truth_subst(Q, pi, F=len(truth["exts"]))
        def sim(rng, k, t):
            return simulate_mixfrag_s_cherries(rng, k, truth["lam"], truth["mu"], np.array(truth["exts"]),
                                               np.array(truth["weights"]), t, Qs, pis)
    else:
        def sim(rng, k, t):
            return simulate_mixfrag_cherries(rng, k, truth["lam"], truth["mu"], np.array(truth["exts"]),
                                             np.array(truth["weights"]), t, Q, pi)
    out = []
    while len(out) < n_pairs:
        t = float(rng.choice(ts))
        sims = sim(rng, 8, t)
        for sm in sims:
            if 0 < len(sm["x"]) <= max_len and 0 < len(sm["y"]) <= max_len:
                out.append((np.asarray(sm["x"], np.int32), np.asarray(sm["y"], np.int32), float(sm["t"])))
    out = out[:n_pairs]
    pickle.dump(out, open(cache, "wb"))
    return out


def load_corpus(args, Q, pi):
    if args.corpus in SIM_TRUTH or (args.corpus == "sim_tkf92" and args.subst_npz):
        pairs = simulate_corpus(args.corpus, args.n_pairs, args.seed, args.max_len, Q, pi, subst_tag(args.subst_npz))
    elif args.corpus == "sim_tkf92":
        pairs = pickle.load(open(os.path.join(
            PY, "experiments/2dfb_sim/sim_tkf92/train.pkl"), "rb"))
        pairs = [p for p in pairs
                 if p[0].shape[0] <= args.max_len and p[1].shape[0] <= args.max_len]
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(pairs), args.n_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    elif args.corpus == "pfam":
        from run_tkf92_2dfb_pfam import load_breadth_first_pairs
        pairs, _val = load_breadth_first_pairs(
            os.path.join(PY, "pfam/precompiled"), args.max_len,
            args.n_pairs, 50, seed=args.seed)
    else:
        raise ValueError(args.corpus)
    pairs = [(np.asarray(x, np.int32), np.asarray(y, np.int32), float(t))
             for x, y, t in pairs]
    return pairs


def make_chunks(pairs, chunk):
    """Bucket by padded shape and cut into fixed-size chunks (last chunk of a
    bucket is padded by repetition and masked by `w`)."""
    buckets = _bin_bucket_pairs(pairs)
    chunks = []
    for key, lst in buckets.items():
        for i in range(0, len(lst), chunk):
            sub = lst[i:i + chunk]
            n_real = len(sub)
            if n_real < chunk:
                sub = sub + [sub[-1]] * (chunk - n_real)
            xs, ys, ts, Lx, Ly = _stack_bucket(sub)
            w = np.zeros(chunk, np.float64)
            w[:n_real] = 1.0
            chunks.append((jnp.asarray(xs), jnp.asarray(ys), jnp.asarray(ts),
                           jnp.asarray(Lx), jnp.asarray(Ly), jnp.asarray(w)))
    return chunks, len(buckets)


# ---------------------------------------------------------------------------
# Parameterisation shared by the gradient methods
# ---------------------------------------------------------------------------

class Model:
    """Maps between natural parameters (lam, mu, exts, weights; for mixfrag_s
    also S (F,A,A), pi (F,A), Q (F,A,A)) and the unconstrained vector used by
    Adam / L-BFGS.  mixfrag_s layout: [log mu, logit kappa, logit r_1..r_F,
    w logits (F), then for each fragtype f: log S_f[upper triangle] (A(A-1)/2),
    pi_f logits (A)]."""

    def __init__(self, name, n_fragtypes=2, A=20):
        self.name = name
        self.A = int(A)
        if name == "mixfrag":
            self.F = 2
        elif name == "mixfrag_s":
            self.F = int(n_fragtypes)
        else:
            self.F = 1
        self._iu = np.triu_indices(self.A, 1)
        self.n_tri = self.A * (self.A - 1) // 2
        self.n_indel = 2 + {"tkf91": 0, "tkf92": 1}.get(name, 2 * self.F)
        self.n_params = self.n_indel + (self.F * (self.n_tri + self.A) if name == "mixfrag_s" else 0)

    def blocks(self):
        """[(name, start, stop)] of the parameter blocks of the unconstrained vector."""
        b = [("lam_mu", 0, 2)]
        if self.name == "tkf92":
            b.append(("ext", 2, 3))
        elif self.name in ("mixfrag", "mixfrag_s"):
            b += [("exts", 2, 2 + self.F), ("weights", 2 + self.F, 2 + 2 * self.F)]
        if self.name == "mixfrag_s":
            off = self.n_indel
            for f in range(self.F):
                b.append((f"S_{f+1}", off, off + self.n_tri)); off += self.n_tri
                b.append((f"pi_{f+1}", off, off + self.A)); off += self.A
        return b

    # natural dict -> unconstrained vector
    def pack(self, p):
        lam, mu = p["lam"], p["mu"]
        v = [np.log(mu), np.log(lam / mu / (1 - lam / mu))]
        if self.name == "tkf92":
            r = p["exts"][0]
            v.append(np.log(r / (1 - r)))
        elif self.name in ("mixfrag", "mixfrag_s"):
            for r in p["exts"]:
                v.append(np.log(r / (1 - r)))
            w = np.asarray(p["weights"])
            v += list(np.log(w) - np.log(w[-1]))   # last logit is 0
        if self.name == "mixfrag_s":
            for f in range(self.F):
                v += list(np.log(np.asarray(p["S"][f], np.float64)[self._iu]))
                pf = np.asarray(p["pi"][f], np.float64)
                v += list(np.log(pf) - np.log(pf[-1]))   # last logit is 0
        v = np.array(v, np.float64)
        assert v.shape[0] == self.n_params, (v.shape, self.n_params)
        return jnp.asarray(v)

    # unconstrained vector -> per-fragtype substitution models (jax): (S (F,A,A), pi (F,A), Q (F,A,A))
    def unpack_subst(self, v):
        assert self.name == "mixfrag_s"
        A = self.A
        off = self.n_indel
        Ss, pis = [], []
        for f in range(self.F):
            ls = v[off:off + self.n_tri]; off += self.n_tri
            lp = v[off:off + A]; off += A
            S = jnp.zeros((A, A), dtype=v.dtype).at[self._iu].set(jnp.exp(ls))
            Ss.append(S + S.T)
            pis.append(jax.nn.softmax(lp))
        Ss = jnp.stack(Ss); pis = jnp.stack(pis)
        Qs = jax.vmap(build_Q_from_S_pi)(Ss, pis)
        return Ss, pis, Qs

    def record(self, p):
        """JSON-able per-pass record of the natural parameters (the matrices of
        mixfrag_s are kept out of the trajectory except pi; see final_params)."""
        r = dict(lam=float(p["lam"]), mu=float(p["mu"]),
                 exts=[float(e) for e in np.asarray(p["exts"])],
                 weights=[float(w) for w in np.asarray(p["weights"])])
        if self.name == "mixfrag_s":
            r["pi"] = np.asarray(p["pi"], np.float64).tolist()
        return r

    def record_full(self, p):
        r = self.record(p)
        if self.name == "mixfrag_s":
            for k in ("S", "pi", "Q"):
                r[k] = np.asarray(p[k], np.float64).tolist()
        return r

    # unconstrained vector -> natural (jax)
    def unpack(self, v):
        mu = jnp.exp(v[0])
        kappa = jax.nn.sigmoid(v[1])
        lam = kappa * mu
        if self.name == "tkf91":
            exts = jnp.zeros(1)
            weights = jnp.ones(1)
        elif self.name == "tkf92":
            exts = jax.nn.sigmoid(v[2:3])
            weights = jnp.ones(1)
        else:
            exts = jax.nn.sigmoid(v[2:2 + self.F])
            weights = jax.nn.softmax(v[2 + self.F:2 + 2 * self.F])
        return lam, mu, exts, weights

    def natural(self, v):
        v = jnp.asarray(v)
        lam, mu, exts, weights = self.unpack(v)
        p = dict(lam=float(lam), mu=float(mu),
                 exts=[float(e) for e in np.asarray(exts)],
                 weights=[float(w) for w in np.asarray(weights)])
        if self.name == "mixfrag_s":
            Ss, pis, Qs = self.unpack_subst(v)
            p["S"] = np.asarray(Ss, np.float64); p["pi"] = np.asarray(pis, np.float64)
            p["Q"] = np.asarray(Qs, np.float64)
        return p


def make_objective(model, Q, pi, chunk):
    """Per-chunk jitted (LL_sum, grad) of the unconstrained parameters."""
    Qj, pij = jnp.asarray(Q), jnp.asarray(pi)

    def per_pair(v, x, y, t, Lx, Ly):
        lam, mu, exts, weights = model.unpack(v)
        if model.name == "mixfrag_s":
            _Ss, pis, Qs = model.unpack_subst(v)
            return _estep_pair_mixfrag_s_forward_only_core(
                x, y, t, Lx, Ly, lam, mu, exts, weights, Qs, pis)
        if model.name == "mixfrag":
            return _estep_pair_mixfrag_forward_only_core(
                x, y, t, Lx, Ly, lam, mu, exts, weights, Qj, pij)
        return _estep_pair_tkf92_forward_only_core(
            x, y, t, Lx, Ly, lam, mu, exts[0], Qj, pij)

    per_pair = jax.checkpoint(per_pair)

    def chunk_ll(v, xs, ys, ts, Lx, Ly, w):
        lp = jax.vmap(per_pair, in_axes=(None, 0, 0, 0, 0, 0))(v, xs, ys, ts, Lx, Ly)
        return jnp.sum(w * lp)

    return jax.jit(jax.value_and_grad(chunk_ll)), jax.jit(chunk_ll)


# ---------------------------------------------------------------------------
# EM pieces (flat priors -> MLE fixed point)
# ---------------------------------------------------------------------------

def em_pass(model, chunks, p, Q, pi):
    """One full-batch E-step. Returns (LL_sum, suff dict)."""
    ll = 0.0
    suff = None
    for xs, ys, ts, Lx, Ly, w in chunks:
        if model.name == "mixfrag_s":
            out = estep_batch_mixfrag_s(xs, ys, ts, Lx, Ly, p["lam"], p["mu"],
                                        np.asarray(p["exts"]), np.asarray(p["weights"]),
                                        np.asarray(p["Q"]), np.asarray(p["pi"]))
        elif model.name == "mixfrag":
            out = estep_batch_mixfrag(xs, ys, ts, Lx, Ly, p["lam"], p["mu"],
                                      np.asarray(p["exts"]), np.asarray(p["weights"]),
                                      Q, pi)
        else:
            out = estep_batch_tkf92(xs, ys, ts, Lx, Ly, p["lam"], p["mu"],
                                    p["exts"][0], Q, pi)
        ll += float(jnp.sum(w * out["log_p"]))
        agg = {}
        for k in ("B", "D", "S", "L", "M", "T"):
            agg[k] = float(jnp.sum(w * out[k]))
        if model.name in ("mixfrag", "mixfrag_s"):
            agg["F_f"] = np.asarray(jnp.sum(w[:, None] * out["F_f"], axis=0))
            agg["E_f"] = np.asarray(jnp.sum(w[:, None] * out["E_f"], axis=0))
            if model.name == "mixfrag_s":
                # per-fragtype Holmes-Rubin dwell W_f (A,) / transition U_f (A,A) at each pair's own t,
                # and the residue counts V_f = ancestor-at-match + inserted + deleted (A,)
                agg["W"] = np.asarray(jnp.einsum("b,bfa->fa", w, out["W"]))
                agg["U"] = np.asarray(jnp.einsum("b,bfij->fij", w, out["U"]))
                agg["V"] = np.asarray(jnp.einsum("b,bfa->fa", w,
                                                 jnp.sum(out["mc"], axis=-1) + out["ins"] + out["del"]))
        else:
            agg["ext_count"] = float(jnp.sum(w * out["ext_count"]))
            agg["notext_count"] = float(jnp.sum(w * out["notext_count"]))
        if suff is None:
            suff = agg
        else:
            for k in suff:
                suff[k] = suff[k] + agg[k]
    return ll, suff


def vjp_value_grad(model, chunks, v, Q, pi):
    """(LL_sum, dLL/dv) with the gradient assembled in closed form from the
    E-step sufficient statistics (Fisher's identity: the gradient of the
    observed-data log-likelihood equals the gradient of the Q-function at the
    current parameters).  This is the "custom VJP" route of the paper: one
    forward-backward pass gives the counts, and the derivatives w.r.t. the
    natural parameters are the score-function formulas, with no reverse-mode
    autodiff through the DP.  The chain rule to the unconstrained vector v is
    taken by autodiff of the (cheap) parameter map only."""
    p = model.natural(v)
    ll, suff = em_pass(model, chunks, p, Q, pi)
    B, D, S, L, M, T = (suff[k] for k in ("B", "D", "S", "L", "M", "T"))

    def q_fn(vv):
        lam, mu, exts, weights = model.unpack(vv)
        q = (B * jnp.log(lam) - lam * (S + T) + D * jnp.log(mu) - mu * S
             + L * jnp.log(lam / mu) + M * jnp.log(1.0 - lam / mu))
        if model.name == "tkf92":
            q = q + suff["ext_count"] * jnp.log(exts[0]) + suff["notext_count"] * jnp.log(1.0 - exts[0])
        elif model.name in ("mixfrag", "mixfrag_s"):
            F_f = jnp.asarray(suff["F_f"]); E_f = jnp.asarray(suff["E_f"])
            q = q + jnp.sum(F_f * jnp.log(exts) + E_f * jnp.log(1.0 - exts) + E_f * jnp.log(weights))
        if model.name == "mixfrag_s":
            # Expected complete-data log-likelihood of the substitution side, per fragtype:
            # CTMC paths  sum_{i!=j} U_f[i,j] log Q_f[i,j] + sum_i W_f[i] Q_f[i,i]  (Q_f[i,i] = -exit rate, so the
            # dwell term is -W_f[i] * exit_i), plus the equilibrium emissions  sum_a V_f[a] log pi_f[a].
            # d/dQ_f[i,j] = U_f/Q_f off the diagonal and -W_f[i] on the diagonal; jax chains through
            # Q_f = build_Q_from_S_pi(S_f, pi_f) to the log-S / pi-logit parameters.
            Ss, pis, Qs = model.unpack_subst(vv)
            eye = jnp.eye(model.A, dtype=bool)[None]
            W = jnp.asarray(suff["W"]); U = jnp.asarray(suff["U"]); V = jnp.asarray(suff["V"])
            log_Q_off = jnp.log(jnp.where(eye, 1.0, Qs))
            q = (q + jnp.sum(jnp.where(eye, 0.0, U * log_Q_off))
                 + jnp.sum(W * jnp.diagonal(Qs, axis1=-2, axis2=-1))
                 + jnp.sum(V * jnp.log(pis)))
        return q
    g = jax.grad(q_fn)(jnp.asarray(v))
    return ll, g


def em_mstep(model, suff, p):
    lam, mu = m_step_lam_mu(suff, prior_alpha_lam=1.0, prior_alpha_mu=1.0,
                            prior_beta=0.0)
    q = dict(lam=float(lam), mu=float(mu), exts=list(p["exts"]),
             weights=list(p["weights"]))
    if model.name == "tkf92":
        q["exts"] = [float(m_step_ext(suff, prior_alpha=1.0, prior_beta=1.0))]
    elif model.name in ("mixfrag", "mixfrag_s"):
        q["exts"] = [float(e) for e in m_step_ext_per_fragtype(
            suff["F_f"], suff["E_f"], prior_alpha=1.0, prior_beta=1.0)]
        q["weights"] = [float(w) for w in m_step_weights(suff["E_f"], prior_alpha=1.0)]
    if model.name == "mixfrag_s":
        # exact per-fragtype (S_f, pi_f) M-step (coordinate ascent of m_step_subst_option1) with FLAT priors
        S_new, pi_new, Q_new = m_step_subst_per_fragtype(
            suff["W"], suff["U"], suff["V"], S_prior=None, pi_prior=None, pi_pseudo=0.0, S_pseudo=0.0)
        q["S"], q["pi"], q["Q"] = S_new, pi_new, Q_new
    return q


# ---------------------------------------------------------------------------
# Runners.  Each returns a list of trajectory records
#   {pass, wall, ll (per pair), params}
# where `ll` is the objective at `params` BEFORE the update made in that pass.
# ---------------------------------------------------------------------------

def run_em(model, chunks, N, p0, Q, pi, max_iter, tol, log, final=None):
    traj = []
    p = dict(p0)
    t0 = time.time()
    prev = -np.inf
    for k in range(max_iter):
        ll, suff = em_pass(model, chunks, p, Q, pi)
        ll /= N
        traj.append(dict(**{"pass": k + 1}, wall=time.time() - t0, ll=ll, params=model.record(p)))
        if k % 10 == 0 or k == max_iter - 1:
            log(f"  em   pass {k+1:4d} ll={ll:.6f} {fmt(p)} ({time.time()-t0:.1f}s)")
        if ll - prev < tol and k > 0:
            log(f"  em   converged (dLL={ll-prev:.2e} < {tol}) at pass {k+1}")
            break
        prev = ll
        p = em_mstep(model, suff, p)
    if final is not None:
        final.update(model.record_full(p))
    return traj


def run_adam(model, chunks, N, p0, vg, lr, max_iter, tol, log, final=None):
    import optax
    v = model.pack(p0)
    opt = optax.adam(lr)
    state = opt.init(v)
    traj = []
    t0 = time.time()
    best = -np.inf
    stall = 0
    for k in range(max_iter):
        ll, g = full_value_grad(vg, v, chunks)
        ll /= N
        traj.append(dict(**{"pass": k + 1}, wall=time.time() - t0, ll=ll,
                         params=model.record(model.natural(v))))
        if k % 50 == 0 or k == max_iter - 1:
            log(f"  adam(lr={lr}) pass {k+1:4d} ll={ll:.6f} {fmt(model.natural(v))} "
                f"|g|={float(jnp.linalg.norm(g)):.2e} ({time.time()-t0:.1f}s)")
        # Stop when the objective has stopped improving by more than tol for a
        # long window (Adam oscillates, so a one-step test is not meaningful).
        if ll > best + tol:
            best = ll
            stall = 0
        else:
            stall += 1
            if stall >= 200:
                log(f"  adam(lr={lr}) stalled (no gain > {tol} in 200 passes) at pass {k+1}")
                break
        upd, state = opt.update(-g / N, state, v)   # ascent on LL/pair
        v = optax.apply_updates(v, upd)
    if final is not None:
        final.update(model.record_full(model.natural(v)))
    return traj


def run_lbfgs(model, chunks, N, p0, vg, max_fun, log, final=None):
    import scipy.optimize
    v0 = np.asarray(model.pack(p0), np.float64)
    traj = []
    t0 = time.time()
    n_eval = [0]

    def fun(x):
        v = jnp.asarray(x)
        ll, g = full_value_grad(vg, v, chunks)
        n_eval[0] += 1
        traj.append(dict(**{"pass": n_eval[0]}, wall=time.time() - t0, ll=ll / N,
                         params=model.record(model.natural(v))))
        if n_eval[0] % 10 == 1:
            log(f"  lbfgs eval {n_eval[0]:4d} ll={ll/N:.6f} {fmt(model.natural(v))} "
                f"({time.time()-t0:.1f}s)")
        return -ll / N, -np.asarray(g, np.float64) / N

    res = scipy.optimize.minimize(fun, v0, jac=True, method="L-BFGS-B",
                                  options=dict(maxfun=max_fun, maxiter=max_fun,
                                               ftol=0.0, gtol=1e-10, maxcor=20))
    log(f"  lbfgs done: {res.message} nit={res.nit} nfev={res.nfev} "
        f"ll={-res.fun:.6f} {fmt(model.natural(jnp.asarray(res.x)))}")
    if final is not None:
        final.update(model.record_full(model.natural(jnp.asarray(res.x))))
    return traj


def full_value_grad(vg, v, chunks):
    """vg is either a per-chunk jitted autodiff value_and_grad, or a callable
    (v) -> (LL, grad) over the whole corpus (the custom-VJP route)."""
    if getattr(vg, "whole_corpus", False):
        return vg(v)
    ll = 0.0
    g = jnp.zeros_like(v)
    for xs, ys, ts, Lx, Ly, w in chunks:
        l, gg = vg(v, xs, ys, ts, Lx, Ly, w)
        ll += float(l)
        g = g + gg
    return ll, g


def fmt(p):
    s = f"lam={p['lam']:.5f} mu={p['mu']:.5f}"
    if len(p["exts"]) == 1 and p["exts"][0] > 0:
        s += f" r={p['exts'][0]:.4f}"
    elif len(p["exts"]) > 1:
        s += " r=" + ",".join(f"{e:.4f}" for e in p["exts"])
        s += " w=" + ",".join(f"{w:.4f}" for w in p["weights"])
    if "pi" in p:
        pis = np.asarray(p["pi"], np.float64)
        s += " max|pi_f-pi_1|=" + ",".join(f"{np.max(np.abs(pis[f] - pis[0])):.3f}" for f in range(1, pis.shape[0]))
    return s


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["sim_tkf92", "sim_tkf91", "sim_mixfrag", "sim_mixfrag_s", "pfam"], required=True)
    ap.add_argument("--model", choices=["tkf91", "tkf92", "mixfrag", "mixfrag_s"], required=True)
    ap.add_argument("--n-fragtypes", type=int, default=2, help="F of --model mixfrag_s (mixfrag is F=2)")
    ap.add_argument("--methods", nargs="+", default=["em", "lbfgs", "adam"])
    ap.add_argument("--adam-lrs", type=float, nargs="+", default=[0.1, 0.01])
    ap.add_argument("--n-pairs", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-lam", type=float, default=0.02)
    ap.add_argument("--init-mu", type=float, default=0.025)
    ap.add_argument("--init-ext", type=float, default=0.4)
    ap.add_argument("--init-exts", type=float, nargs="+", default=None,
                    help="F values; default 0.3 0.6 for F=2, else linspace(0.3, 0.8, F)")
    ap.add_argument("--init-weights", type=float, nargs="+", default=None, help="F values; default uniform")
    ap.add_argument("--em-max-iter", type=int, default=300)
    ap.add_argument("--adam-max-iter", type=int, default=1500)
    ap.add_argument("--lbfgs-max-fun", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-7,
                    help="per-pair LL gain below which EM/Adam stop")
    ap.add_argument("--gradcheck-pairs", type=int, default=8)
    ap.add_argument("--subst-npz", default="", help="npz with Q, pi replacing LG08 as the fixed substitution model")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    log(f"args: {vars(args)}")
    log(f"jax devices: {jax.devices()}")
    Q, pi = load_subst(args.subst_npz)
    model = Model(args.model, n_fragtypes=args.n_fragtypes, A=pi.shape[0])

    t0 = time.time()
    pairs = load_corpus(args, Q, pi)
    N = len(pairs)
    chunks, n_buckets = make_chunks(pairs, args.chunk)
    lens = np.array([max(x.shape[0], y.shape[0]) for x, y, _ in pairs])
    log(f"corpus {args.corpus}: N={N} pairs, {n_buckets} shape buckets, "
        f"{len(chunks)} chunks of {args.chunk}; max len mean={lens.mean():.1f} "
        f"max={lens.max()}; loaded in {time.time()-t0:.1f}s")

    if args.model == "tkf91":
        p0 = dict(lam=args.init_lam, mu=args.init_mu, exts=[0.0], weights=[1.0])
    elif args.model == "tkf92":
        p0 = dict(lam=args.init_lam, mu=args.init_mu, exts=[args.init_ext], weights=[1.0])
    else:
        F = model.F
        exts0 = args.init_exts if args.init_exts is not None else ([0.3, 0.6] if F == 2 else list(np.linspace(0.3, 0.8, F)))
        w0 = args.init_weights if args.init_weights is not None else [1.0 / F] * F
        assert len(exts0) == F and len(w0) == F, (exts0, w0, F)
        p0 = dict(lam=args.init_lam, mu=args.init_mu, exts=[float(e) for e in exts0],
                  weights=[float(w) / float(np.sum(w0)) for w in w0])
        if args.model == "mixfrag_s":
            # every fragtype starts at the fixed (Q, pi) of the other rows (S = Q / pi, symmetrised)
            S0 = S_from_Q_pi(Q, pi)
            p0["S"] = np.tile(S0[None], (F, 1, 1)); p0["pi"] = np.tile(pi[None], (F, 1))
            p0["Q"] = np.asarray(jax.vmap(build_Q_from_S_pi)(jnp.asarray(p0["S"]), jnp.asarray(p0["pi"])), np.float64)
            log(f"mixfrag_s: F={F}, A={model.A}, {model.n_params} unconstrained parameters "
                f"({model.n_indel} indel + {F} x ({model.n_tri} log S + {model.A} pi logits)); "
                f"max|Q_f - Q| at init = {np.max(np.abs(p0['Q'] - Q[None])):.2e}")
    log(f"init: {fmt(p0)}")

    vg, _ = make_objective(model, Q, pi, args.chunk)

    def vg_vjp(v):
        return vjp_value_grad(model, chunks, v, Q, pi)
    vg_vjp.whole_corpus = True

    # Warm every JIT shape for every code path before timing anything.
    t0 = time.time()
    v0 = model.pack(p0)
    ll_g, g_ad = full_value_grad(vg, v0, chunks)
    ll_e, g_vjp = vg_vjp(v0)
    log(f"warm-up: {time.time()-t0:.1f}s; LL/pair at init: gradient path "
        f"{ll_g/N:.6f}, E-step path {ll_e/N:.6f} (should agree)")
    # Gradient agreement: autodiff through the DP vs the score-identity
    # gradient from the E-step statistics (the custom VJP).
    def gradcheck(v, tag, chunks_=chunks, n_pairs_=N):
        _, ga = full_value_grad(vg, v, chunks_)
        _, gv = vjp_value_grad(model, chunks_, v, Q, pi)
        ga = np.asarray(ga); gv = np.asarray(gv)
        rel = float(np.max(np.abs(ga - gv)) / max(np.max(np.abs(ga)), 1e-300))
        if ga.shape[0] <= 16:
            log(f"gradcheck [{tag}, {n_pairs_} pairs]: autodiff {np.array2string(ga, precision=6)} "
                f"vjp {np.array2string(gv, precision=6)} max|diff|/max|g| = {rel:.2e}")
        else:
            log(f"gradcheck [{tag}, {n_pairs_} pairs]: {ga.shape[0]} parameters, max|g| = {np.max(np.abs(ga)):.4e}, "
                f"max|diff|/max|g| = {rel:.2e}")
        rec = dict(n_pairs=n_pairs_, autodiff=ga.tolist(), vjp=gv.tolist(), max_rel_diff=rel)
        if len(model.blocks()) > 1:
            # per parameter block: max |autodiff - score-identity| over the block / max |autodiff| over the block
            per = {}
            for name, a, b in model.blocks():
                per[name] = float(np.max(np.abs(ga[a:b] - gv[a:b])) / max(np.max(np.abs(ga[a:b])), 1e-300))
            rec["max_rel_diff_per_block"] = per
            log("    per block: " + " ".join(f"{k}={e:.2e}" for k, e in per.items()))
        return rec
    gc = {"init": gradcheck(v0, "init")}
    if model.name == "mixfrag_s":
        # the required small check: score-identity gradient vs autodiff of the forward on the first
        # --gradcheck-pairs pairs, at the initial point and at a random perturbation of every block
        sub = pairs[:args.gradcheck_pairs]
        sub_chunks, _ = make_chunks(sub, args.chunk)
        rng_sub = np.random.default_rng(1)
        for tag, vv in (("init", v0), ("random", v0 + 0.5 * rng_sub.standard_normal(v0.shape[0]))):
            gc[f"subset_{tag}"] = gradcheck(jnp.asarray(vv), f"subset_{tag}", sub_chunks, len(sub))
    # Three-way check against the repo's own custom-VJP wrappers
    # (tkfmixdom.jax.train.vjp: unpadded per-pair forward, score-identity
    # backward), on the first few pairs, so the batched score-identity
    # gradient used here is tied to the library's custom VJP.
    if model.name in ("tkf91", "tkf92"):
        from tkfmixdom.jax.train import vjp as _vjp
        sub = pairs[:args.gradcheck_pairs]
        Qj, pij = jnp.asarray(Q), jnp.asarray(pi)

        def repo_ll(vv):
            lam, mu, exts, _w = model.unpack(vv)
            tot = 0.0
            for x, y, t in sub:
                if model.name == "tkf91":
                    tot = tot + _vjp.tkf91_log_prob(lam, mu, t, Qj, pij, jnp.asarray(x), jnp.asarray(y))
                else:
                    tot = tot + _vjp.tkf92_log_prob(lam, mu, t, exts[0], Qj, pij, jnp.asarray(x), jnp.asarray(y))
            return tot
        sub_chunks, _ = make_chunks(sub, args.chunk)
        for tag, vv in (("init", v0), ("random0", v0 + 0.5 * np.random.default_rng(0).standard_normal(v0.shape[0]))):
            vv = jnp.asarray(vv)
            l_repo, g_repo = jax.value_and_grad(repo_ll)(vv)
            l_ad, g_ad2 = full_value_grad(vg, vv, sub_chunks)
            l_sc, g_sc = vjp_value_grad(model, sub_chunks, vv, Q, pi)
            g_repo, g_ad2, g_sc = (np.asarray(a) for a in (g_repo, g_ad2, g_sc))
            scale = max(np.max(np.abs(g_repo)), 1e-300)
            rec = dict(n_pairs=len(sub), ll_repo=float(l_repo), ll_autodiff=float(l_ad), ll_scoreid=float(l_sc),
                       grad_repo_vjp=g_repo.tolist(), grad_autodiff=g_ad2.tolist(), grad_scoreid=g_sc.tolist(),
                       max_rel_diff_repo_vs_autodiff=float(np.max(np.abs(g_repo - g_ad2)) / scale),
                       max_rel_diff_repo_vs_scoreid=float(np.max(np.abs(g_repo - g_sc)) / scale))
            gc[f"repo_vjp_{tag}"] = rec
            log(f"gradcheck vs repo custom VJP [{tag}, {len(sub)} pairs]: LL repo {float(l_repo):.6f} "
                f"autodiff {float(l_ad):.6f} score-id {float(l_sc):.6f}; max rel grad diff "
                f"repo-vs-autodiff {rec['max_rel_diff_repo_vs_autodiff']:.2e}, "
                f"repo-vs-score-identity {rec['max_rel_diff_repo_vs_scoreid']:.2e}")
    rng_gc = np.random.default_rng(0)
    for k in range(2):
        vk = v0 + 0.5 * rng_gc.standard_normal(v0.shape[0])
        gc[f"random{k}"] = gradcheck(jnp.asarray(vk), f"random{k}")
    # Timed single passes (warm cache).
    t0 = time.time(); full_value_grad(vg, v0, chunks); t_grad = time.time() - t0
    t0 = time.time(); em_pass(model, chunks, p0, Q, pi); t_em = time.time() - t0
    t0 = time.time(); vg_vjp(v0); t_vjp = time.time() - t0
    log(f"timed pass: value+grad (autodiff) {t_grad:.2f}s, E-step {t_em:.2f}s, "
        f"E-step + closed-form gradient (custom VJP) {t_vjp:.2f}s")

    results = dict(args=vars(args), N=N, n_buckets=n_buckets,
                   model=args.model, n_fragtypes=model.F, n_params=model.n_params,
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
            results["methods"]["em"] = run_em(model, chunks, N, p0, Q, pi,
                                              args.em_max_iter, args.tol, log, final=final("em"))
        elif m == "lbfgs":
            results["methods"]["lbfgs"] = run_lbfgs(model, chunks, N, p0, vg,
                                                    args.lbfgs_max_fun, log, final=final("lbfgs"))
        elif m == "adam":
            for lr in args.adam_lrs:
                results["methods"][f"adam_lr{lr:g}"] = run_adam(
                    model, chunks, N, p0, vg, lr, args.adam_max_iter, args.tol, log, final=final(f"adam_lr{lr:g}"))
                save()
        elif m == "lbfgs_vjp":
            results["methods"]["lbfgs_vjp"] = run_lbfgs(model, chunks, N, p0, vg_vjp,
                                                        args.lbfgs_max_fun, log, final=final("lbfgs_vjp"))
        elif m == "adam_vjp":
            for lr in args.adam_lrs:
                results["methods"][f"adam_vjp_lr{lr:g}"] = run_adam(
                    model, chunks, N, p0, vg_vjp, lr, args.adam_max_iter, args.tol, log,
                    final=final(f"adam_vjp_lr{lr:g}"))
                save()
        save()
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
