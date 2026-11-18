"""Shared pieces of the MixFrag-S gradient tails (Adam switch, L-BFGS refine).

MixFrag-S: the MixFrag pair HMM (S, M_1..M_F, I_1..I_F, D_1..D_F, E; shared
indel rates lam, mu; per-fragtype extension r_f and weight w_f) in which
fragtype f owns a reversible substitution model (S_f, pi_f), Q_f =
build_Q_from_S_pi(S_f, pi_f).

Objective (the recipe's MAP objective, the quantity svi-BW maximises):

    MAP(theta) = sum_pairs log P(alignment, x, y | theta)   [1D path-conditioned]
               + log_prior(theta)

with the log-prior whose maximiser is exactly the svi-BW M-step of
``tkfmixdom.jax.train.mixfrag_svi_bw.svi_bw_mixfrag_s`` (every term matches a
pseudocount in the corresponding M-step):
  * Gamma(alpha_lam, beta) on lam, Gamma(alpha_mu, beta) on mu (shared rate beta;
    m_step_indel_quadratic adds alpha-1 to B / D and beta to S);
  * Beta(a_ext, b_ext) on each r_f (m_step_ext_per_fragtype);
  * symmetric Dirichlet(alpha_w) on w (m_step_weights);
  * per fragtype, the LG-centred pseudo-statistics of m_step_subst_option1:
        V_f += pi_pseudo * pi_lg,
        U_f += S_pseudo * S_lg * pi_lg[None, :],
        W_f += S_pseudo * rowsum(S_lg * pi_lg[None, :]),
    i.e. log_prior_subst(S_f, pi_f) = sum_{i!=j} U_ps[i,j] log Q_f[i,j]
         + sum_i W_ps[i] Q_f[i,i] + pi_pseudo * sum_a pi_lg[a] log pi_f[a].

Gradient: the closed-form ("custom VJP") route of the paper. One 1D E-step
(``estep_bucket_chunks_mixfrag_s_path``) gives the expected sufficient
statistics (B, D, S, L, M, T, F_f, E_f, W_f, U_f, V_f); by Fisher's identity
the gradient of the observed-data log-likelihood equals the gradient of the
Q-function at the current parameters, and the chain rule to the unconstrained
vector is taken by autodiff of the (cheap) parameter map only -- no
reverse-mode pass through the forward-backward.

Unconstrained vector (the layout of ``compare_optimizers.Model('mixfrag_s')``):
    [log mu, logit kappa (lam = kappa mu), logit r_1..r_F, w logits (F),
     then per fragtype f: log S_f[upper triangle] (A(A-1)/2), pi_f logits (A)].
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(os.path.dirname(HERE))          # ~/tkf-mixdom/python
if PY not in sys.path:
    sys.path.insert(0, PY)

import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402

from tkfmixdom.jax.core.ctmc import build_Q_from_S_pi            # noqa: E402
from tkfmixdom.jax.train.mixfrag_svi_bw import (                  # noqa: E402
    lg_prior_S_pi, bucket_pairs_lazy_path, estep_bucket_chunks_mixfrag_s_path,
    _empty_suff_mfs,
)
import train_mixfrag_s as tms                                     # noqa: E402

SUFF_KEYS = ('B', 'D', 'S', 'L', 'M', 'T', 'F_f', 'E_f', 'W', 'U', 'V')

# The recipe's priors (train_mixfrag_s.py defaults).
RECIPE_PRIORS = dict(prior_alpha_lam=2.0, prior_alpha_mu=2.0, prior_beta=10.0,
                     ext_prior_alpha=2.0, ext_prior_beta=3.0,
                     weight_prior_alpha=1.5, pi_pseudo=10.0, S_pseudo=5.0)


def fmt_vec(v, prec=4):
    return np.array2string(np.asarray(v, dtype=np.float64), precision=prec,
                           separator=' ', max_line_width=10 ** 6)


# ---------------------------------------------------------------------------
# Checkpoint I/O (the train_mixfrag_s.py contract npz).
# ---------------------------------------------------------------------------

def load_contract(path):
    """-> (params dict {lam, mu, exts, weights, S, pi, Q}, F, A, raw npz dict)."""
    ck, _suff = tms.load_checkpoint(path)
    F = int(ck['n_fragtypes'])
    p = dict(lam=float(ck['ins_rate']), mu=float(ck['del_rate']),
             exts=np.asarray(ck['exts'], np.float64),
             weights=np.asarray(ck['weights'], np.float64),
             S=np.asarray(ck['S_exch'], np.float64),
             pi=np.asarray(ck['pi'], np.float64),
             Q=np.asarray(ck['Q'], np.float64))
    A = int(p['pi'].shape[-1])
    assert p['S'].shape == (F, A, A) and p['pi'].shape == (F, A)
    return p, F, A, ck


def save_contract(path, p, F, em_iter, config, history):
    """Write the contract npz (ins_rate, del_rate, exts, weights, S_exch, pi,
    Q, em_iter, n_fragtypes, _config, history) atomically."""
    state = dict(lam=p['lam'], mu=p['mu'], exts=p['exts'], weights=p['weights'],
                 S=p['S'], pi=p['pi'], Q=p['Q'], em_iter=int(em_iter),
                 suff_blend=None, history=history)
    tms.save_checkpoint_atomic(path, state, config, F)


# ---------------------------------------------------------------------------
# Unconstrained parameterisation.
# ---------------------------------------------------------------------------

class Packer:
    """Natural params <-> unconstrained vector (layout in the module doc)."""

    def __init__(self, F, A=20):
        self.F, self.A = int(F), int(A)
        self._iu = np.triu_indices(self.A, 1)
        self.n_tri = self.A * (self.A - 1) // 2
        self.n_indel = 2 + 2 * self.F
        self.n_params = self.n_indel + self.F * (self.n_tri + self.A)

    def blocks(self):
        b = [('lam_mu', 0, 2), ('exts', 2, 2 + self.F),
             ('weights', 2 + self.F, 2 + 2 * self.F)]
        off = self.n_indel
        for f in range(self.F):
            b.append((f'S_{f + 1}', off, off + self.n_tri)); off += self.n_tri
            b.append((f'pi_{f + 1}', off, off + self.A)); off += self.A
        return b

    def pack(self, p):
        lam, mu = float(p['lam']), float(p['mu'])
        kappa = lam / mu
        v = [np.log(mu), np.log(kappa / (1.0 - kappa))]
        for r in np.asarray(p['exts'], np.float64):
            v.append(np.log(r / (1.0 - r)))
        w = np.asarray(p['weights'], np.float64)
        v += list(np.log(w) - np.log(w[-1]))
        for f in range(self.F):
            v += list(np.log(np.asarray(p['S'][f], np.float64)[self._iu]))
            pf = np.asarray(p['pi'][f], np.float64)
            v += list(np.log(pf) - np.log(pf[-1]))
        v = np.asarray(v, np.float64)
        assert v.shape[0] == self.n_params, (v.shape, self.n_params)
        return v

    # jax-traceable
    def unpack_indel(self, v):
        mu = jnp.exp(v[0])
        kappa = jax.nn.sigmoid(v[1])
        lam = kappa * mu
        exts = jax.nn.sigmoid(v[2:2 + self.F])
        weights = jax.nn.softmax(v[2 + self.F:2 + 2 * self.F])
        return lam, mu, exts, weights

    def unpack_subst(self, v):
        A, off = self.A, self.n_indel
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

    def natural(self, v):
        v = jnp.asarray(v, jnp.float64)
        lam, mu, exts, weights = self.unpack_indel(v)
        Ss, pis, Qs = self.unpack_subst(v)
        return dict(lam=float(lam), mu=float(mu),
                    exts=np.asarray(exts, np.float64),
                    weights=np.asarray(weights, np.float64),
                    S=np.asarray(Ss, np.float64), pi=np.asarray(pis, np.float64),
                    Q=np.asarray(Qs, np.float64))


# ---------------------------------------------------------------------------
# Q-function, log-prior and their gradient (closed form in the natural
# parameters; autodiff only through the parameter map).
# ---------------------------------------------------------------------------

def prior_pseudo_stats(priors, A):
    """(pi_ps (A,), U_ps (A,A), W_ps (A,)) of the LG-centred substitution prior
    -- exactly the pseudocounts m_step_subst_option1 adds."""
    S_lg, pi_lg = lg_prior_S_pi()
    assert pi_lg.shape[0] == A
    pi_ps = float(priors['pi_pseudo']) * pi_lg
    U_ps = float(priors['S_pseudo']) * S_lg * pi_lg[None, :]
    W_ps = U_ps.sum(axis=1)
    return pi_ps, U_ps, W_ps


def make_value_grad(packer, priors):
    """-> jitted fn(v, suff, prior_weight) = (obj, grad) with
        obj = Q(v; suff) + prior_weight * log_prior(v),
    Q the expected complete-data log-likelihood under the (1D) E-step
    statistics ``suff`` (keys SUFF_KEYS, jnp arrays).  By Fisher's identity
    grad == d/dv [observed LL + prior_weight * log_prior] at the parameters
    the E-step was run at.  (Q itself is NOT the observed log-likelihood; the
    callers report the E-step's log_p sum for that.)"""
    F, A = packer.F, packer.A
    pi_ps, U_ps, W_ps = prior_pseudo_stats(priors, A)
    pi_ps, U_ps, W_ps = (jnp.asarray(x, jnp.float64) for x in (pi_ps, U_ps, W_ps))
    a_lam, a_mu, beta = (float(priors[k]) for k in
                         ('prior_alpha_lam', 'prior_alpha_mu', 'prior_beta'))
    a_ext, b_ext = float(priors['ext_prior_alpha']), float(priors['ext_prior_beta'])
    a_w = float(priors['weight_prior_alpha'])
    eye = jnp.eye(A, dtype=bool)[None]

    def log_prior(v):
        lam, mu, exts, weights = packer.unpack_indel(v)
        Ss, pis, Qs = packer.unpack_subst(v)
        lp = ((a_lam - 1.0) * jnp.log(lam) - beta * lam
              + (a_mu - 1.0) * jnp.log(mu) - beta * mu
              + jnp.sum((a_ext - 1.0) * jnp.log(exts) + (b_ext - 1.0) * jnp.log1p(-exts))
              + (a_w - 1.0) * jnp.sum(jnp.log(weights)))
        log_Q_off = jnp.log(jnp.where(eye, 1.0, Qs))
        lp = (lp + jnp.sum(jnp.where(eye, 0.0, U_ps[None] * log_Q_off))
              + jnp.sum(W_ps[None] * jnp.diagonal(Qs, axis1=-2, axis2=-1))
              + jnp.sum(pi_ps[None] * jnp.log(pis)))
        return lp

    def q_fn(v, suff):
        lam, mu, exts, weights = packer.unpack_indel(v)
        B, D, S_, L, M, T = (suff[k] for k in ('B', 'D', 'S', 'L', 'M', 'T'))
        q = (B * jnp.log(lam) - lam * (S_ + T) + D * jnp.log(mu) - mu * S_
             + L * jnp.log(lam / mu) + M * jnp.log1p(-lam / mu))
        q = q + jnp.sum(suff['F_f'] * jnp.log(exts) + suff['E_f'] * jnp.log1p(-exts)
                        + suff['E_f'] * jnp.log(weights))
        Ss, pis, Qs = packer.unpack_subst(v)
        log_Q_off = jnp.log(jnp.where(eye, 1.0, Qs))
        q = (q + jnp.sum(jnp.where(eye, 0.0, suff['U'] * log_Q_off))
             + jnp.sum(suff['W'] * jnp.diagonal(Qs, axis1=-2, axis2=-1))
             + jnp.sum(suff['V'] * jnp.log(pis)))
        return q

    def obj(v, suff, prior_weight):
        return q_fn(v, suff) + prior_weight * log_prior(v)

    vg = jax.jit(jax.value_and_grad(obj))
    lp_jit = jax.jit(log_prior)
    return vg, lp_jit


def suff_to_jnp(suff):
    return {k: jnp.asarray(np.asarray(suff[k], np.float64)) for k in SUFF_KEYS}


# ---------------------------------------------------------------------------
# Corpus (path format) and the 1D E-step over a list of pairs.
# ---------------------------------------------------------------------------

def load_path_pairs(precompiled_dir, log=print, with_fam=False):
    """All pairs of a precompiled corpus in the compact 'path' format
    (states int8, anc_full int8, desc_full int8, t); with_fam=True returns
    [(pair, fam)] instead (for the breadth-first subset)."""
    from train_pfam import PrecompiledPairSource
    src = PrecompiledPairSource(precompiled_dir)
    log(f'Loading {src.description} [path format] ...')
    t0 = time.time()
    out = []
    for shard_name in src.shard_files:
        for rec in src._read_shard(shard_name):
            x, y, states, t = tms.decode_record_compact(rec)
            L = states.shape[0]
            anc_full = np.zeros(L, dtype=np.int8)
            desc_full = np.zeros(L, dtype=np.int8)
            anc_full[states != 2] = x
            desc_full[states != 3] = y
            pair = (states, anc_full, desc_full, t)
            out.append((pair, rec.get('fam', 'unknown')) if with_fam else pair)
    log(f'  {len(out)} pairs loaded in {time.time() - t0:.0f}s')
    return out


def breadth_first_subset(pairs_with_fam, n_sub, seed):
    """Fixed, seeded, family-balanced subset: one pair per family first (the
    rule of mixdom_lbfgs_refine.py).  pairs_with_fam: [(pair, fam)]."""
    byfam = defaultdict(list)
    for pair, fam in pairs_with_fam:
        byfam[fam].append(pair)
    fams = sorted(byfam)
    rng = np.random.default_rng(seed)
    rng.shuffle(fams)
    for f in fams:
        rng.shuffle(byfam[f])
    sub = []
    while len(sub) < n_sub and any(byfam[f] for f in fams):
        for f in fams:
            if byfam[f]:
                sub.append(byfam[f].pop())
            if len(sub) >= n_sub:
                break
    return sub, len(fams)


def estep_pairs(pairs, p, *, chunk_max=1024, mem_bytes=2.0e9, bin_stride=1,
                scan_mode='associative'):
    """1D path-conditioned E-step over ``pairs`` (path format) at the natural
    params ``p``.  -> (obs_ll_total, suff dict (numpy), n_pairs)."""
    F, A = p['pi'].shape
    suff = _empty_suff_mfs(F, A)
    ll, n = 0.0, 0
    for L_pad, items in bucket_pairs_lazy_path(pairs, bin_stride).items():
        agg, ll_b, n_b = estep_bucket_chunks_mixfrag_s_path(
            items, L_pad, p['lam'], p['mu'], p['exts'], p['weights'], p['Q'],
            p['pi'], chunk_max=chunk_max, mem_bytes=mem_bytes,
            scan_mode=scan_mode)
        for k in suff:
            suff[k] = suff[k] + agg[k]
        ll += ll_b
        n += n_b
    return ll, suff, n


def priors_from_args(args):
    return {k: float(getattr(args, k)) for k in RECIPE_PRIORS}


def add_prior_args(ap):
    for k, dv in RECIPE_PRIORS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=float, default=dv)


def add_estep_args(ap):
    ap.add_argument('--chunk-max', type=int, default=1024,
                    help='max pairs per vmapped 1D E-step call')
    ap.add_argument('--chunk-mem-gb', type=float, default=2.0,
                    help='GPU memory budget per E-step call (sizes chunks per bin)')
    ap.add_argument('--bin-stride', type=int, default=1,
                    help='keep every k-th padding bin (fewer JIT shapes)')
    ap.add_argument('--fb-scan-mode', choices=('associative', 'sequential'),
                    default='associative')
    ap.add_argument('--jax-cache-dir', default=None)


def estep_kwargs(args):
    return dict(chunk_max=args.chunk_max, mem_bytes=args.chunk_mem_gb * 1e9,
                bin_stride=args.bin_stride, scan_mode=args.fb_scan_mode)


def setup_jax(args):
    jax.config.update('jax_enable_x64', True)
    if args.jax_cache_dir:
        d = os.path.expanduser(args.jax_cache_dir)
        os.makedirs(d, exist_ok=True)
        jax.config.update('jax_compilation_cache_dir', d)
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.5)


def make_logger(log_file, t_start):
    fh = open(log_file, 'a') if log_file else None

    def log(msg):
        line = f'[{time.time() - t_start:9.1f}s] {msg}'
        print(line, flush=True)
        if fh:
            fh.write(line + '\n')
            fh.flush()
    return log


def jsonable(rec):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rec.items()}


def dump_history(path, meta, history):
    tmp = path + f'.tmp.{os.getpid()}'
    with open(tmp, 'w') as fh:
        json.dump(dict(meta, history=[jsonable(h) for h in history]), fh, indent=1)
    os.replace(tmp, path)
