#!/usr/bin/env python3
"""Movie: MM gap probability heatmap (i, j) for GGI-steered TKF92 vs plain TKF92.

For a fixed reversible GGI parameterisation (lam0, mu0, x_del, y_ins), play
out two models side by side over evolutionary time t:

  Left  panel: GGI-steered TKF92 with (lam(t), mu(t), r(t)) following the
               closed-form matched-flow trajectory from
               composition-renormalization.tex.
  Right panel: plain TKF92 with (lam, mu, r) PINNED to the GGI t=0 boundary
               (lam_T(0), mu_T(0), r*(0)).  Same Pair HMM family, but constant
               theta.

At each frame, the MM gap probability G(i, j) is the sum of path probabilities
over all paths M -> (i deletions and j insertions in any order) -> M.  Computed
by forward DP on the (I, D) state x (i, j) wavefront -- equivalent to summing
out the I/D ordering via the hypergeometric closed form.

Convention: 1 unit of evolutionary time = 1 second of animation.

By default all five panels use the SAME underlying GGI parameterisation
(taken from --lam0 --mu0 --x --y).  Pass --config FILE.json to specify
different parameters per panel (for instance, plugging in ML estimates from
each model's individual fit; see gap_movie_example_config.json).

Under panels 2-5 we plot a smoothed graph of D_KL(sim || panel)(t) with a
moving vertical line at the current frame's t, so the visual comparison
quantifies how close each closed-form approximation is to the simulated
ground truth.

Output: ${out_mp4} (uses matplotlib.animation + ffmpeg if available, else
falls back to saving every Nth frame as a PNG).

Usage:
    python experiments/gap_heatmap_movie.py [--lam0 ...] [--mu0 ...] [--x ...]
        [--y ...] [--config FILE.json] [--Lmax 25] [--t-max 10] [--fps 24]
        [--out /tmp/gap_movie.mp4] [--sim-trials 32000] [--sim-workers 0]
"""
import argparse
import json
import os
import sys
import subprocess
import hashlib
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, FFMpegWriter

TRAJECLIKE_BIN = os.path.expanduser('~/trajectory-likelihood/trajeclike')
DEFAULT_SIM_CACHE_DIR = '/tmp/gap_movie_sim_cache'

# Make scratch_ggi_cond_kl_quad importable so we can integrate the full
# matched-flow ODE for the "flow equations" panel.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

S, M, I, D, E = 0, 1, 2, 3, 4


# -----------------------------------------------------------------
# TKF92 Pair HMM (5x5)
# -----------------------------------------------------------------

def tkf91_beta(lam, mu, t):
    """L'Hopital-safe beta at lam = mu."""
    if abs(lam - mu) < 1e-9 * max(lam, mu):
        s = mu * t
        return s / (1.0 + s)
    eta = np.exp(-lam * t)
    alpha = np.exp(-mu * t)
    return lam * (eta - alpha) / (mu * eta - lam * alpha)


def tkf91_gamma(lam, mu, t):
    """L'Hopital-safe gamma at lam = mu."""
    if abs(lam - mu) < 1e-9 * max(lam, mu):
        s = mu * t
        alpha = np.exp(-mu * t)
        phi = (1.0 - alpha) / s if s > 1e-12 else 1.0
        return 1.0 - 1.0 / ((1.0 + s) * phi)
    alpha = np.exp(-mu * t)
    return 1.0 - tkf91_beta(lam, mu, t) / (lam / mu * (1.0 - alpha))


def tkf92_trans(lam, mu, r, t):
    """5x5 TKF92 Pair HMM transition matrix (rows = S, M, I, D, E)."""
    alpha = np.exp(-mu * t)
    beta = tkf91_beta(lam, mu, t)
    gamma = tkf91_gamma(lam, mu, t)
    kappa = lam / mu

    T = np.zeros((5, 5))
    # S, M, I rows use beta
    for src in (S, M, I):
        T[src, M] = (1.0 - beta) * kappa * alpha
        T[src, I] = beta
        T[src, D] = (1.0 - beta) * kappa * (1.0 - alpha)
        T[src, E] = (1.0 - beta) * (1.0 - kappa)
    # D row uses gamma
    T[D, M] = (1.0 - gamma) * kappa * alpha
    T[D, I] = gamma
    T[D, D] = (1.0 - gamma) * kappa * (1.0 - alpha)
    T[D, E] = (1.0 - gamma) * (1.0 - kappa)
    # Fragment extension on M, I, D
    for src in (M, I, D):
        T[src] = (1.0 - r) * T[src]
        T[src, src] += r
    T[E, E] = 1.0
    return T


# -----------------------------------------------------------------
# MM gap probability DP
# -----------------------------------------------------------------

def mm_gap_probs(T, Lmax):
    """G(i, j) = P(M -> (i del + j ins, any order) -> M) for i, j in [0, Lmax].

    Forward DP on (I, D) state x (i, j) wavefront.  Equivalent to the
    hypergeometric closed form, but easier to implement and exact.

    Transition shorthand (rows = source M / I / D, cols = target M / I / D):
        a = T[M, M], b = T[M, I], c = T[M, D]
        f = T[I, M], g = T[I, I], h = T[I, D]
        p = T[D, M], q = T[D, I], r = T[D, D]

    A[I, i, j] = prob of being in state I after i del + j ins (interior path
                  starting from M, first step into I or D).
    A[D, i, j] = prob of being in state D after i del + j ins.

    Initial:    A[I, 0, 1] = b,  A[D, 1, 0] = c.
    Recurrence: A[I, i, j] = g * A[I, i, j-1] + q * A[D, i, j-1]
                A[D, i, j] = h * A[I, i-1, j] + r * A[D, i-1, j]
    Emission:   G(i, j) = A[I, i, j] * f + A[D, i, j] * p  (for i+j > 0)
                G(0, 0) = a
    """
    a = T[M, M]; b = T[M, I]; c = T[M, D]
    f = T[I, M]; g = T[I, I]; h = T[I, D]
    p = T[D, M]; q = T[D, I]; r = T[D, D]

    AI = np.zeros((Lmax + 1, Lmax + 1))
    AD = np.zeros((Lmax + 1, Lmax + 1))
    AI[0, 1] = b
    AD[1, 0] = c

    for s in range(2, 2 * Lmax + 1):
        for i in range(max(0, s - Lmax), min(s, Lmax) + 1):
            j = s - i
            if j >= 1:
                AI[i, j] = g * AI[i, j - 1] + q * AD[i, j - 1]
            if i >= 1:
                AD[i, j] = h * AI[i - 1, j] + r * AD[i - 1, j]

    G = AI * f + AD * p
    G[0, 0] += a
    return G


# -----------------------------------------------------------------
# GGI -> TKF92 closed-form trajectory
# -----------------------------------------------------------------

def integrate_full_flow(lam0, mu0, x_del, y_ins, t_grid, t_eps=5e-3):
    """Numerically integrate the matched-flow 3-d ODE for (lam(t), mu(t),
    r(t)) using scipy via scratch_ggi_cond_kl_quad.run_flow.

    NOTE on convention: scratch_ggi_cond_kl_quad.run_flow takes (x_ins, y_del)
    -- opposite of our (x_del, y_ins) convention -- so we swap when calling.

    Returns a (len(t_grid), 3) array of (lam(t), mu(t), r(t)) evaluated at
    every t in t_grid.  At t < t_eps we fall back to the t_eps value
    (effectively pinning to the boundary).
    """
    from scratch_ggi_cond_kl_quad import run_flow, boundary_condition
    # Convert convention: our (x_del, y_ins) -> scratch's (x_ins, y_del)
    x_ins, y_del = y_ins, x_del
    t_eval = np.clip(t_grid, t_eps, None)
    t_eval_sorted = np.sort(np.unique(t_eval))
    t_max = float(t_eval_sorted[-1])
    sol, (lam_T0, mu_T0, r0) = run_flow(
        lam0, mu0, x_ins, y_del,
        t_eps=t_eps, t_max=t_max, t_eval=t_eval_sorted)
    # Map back to original t_grid ordering
    out = np.zeros((len(t_grid), 3))
    for i, tt in enumerate(t_grid):
        if tt < t_eps:
            out[i] = (lam_T0, mu_T0, r0)
        else:
            j = int(np.searchsorted(t_eval_sorted, tt))
            j = min(j, len(t_eval_sorted) - 1)
            out[i] = (sol.y[0, j], sol.y[1, j], sol.y[2, j])
    return out


def run_gillespie_sim(t, lam0, mu0, x_del, y_ins, Lmax, n_trials, initlen, seed):
    """Run ~/trajectory-likelihood/trajeclike and return both the in-range
    empirical gap probability matrix AND the overflow probability at branch
    length t.

    Convention notes:
      trajeclike's --x  = INSERTION extension probability  = our y_ins
      trajeclike's --y  = DELETION  extension probability  = our x_del
      trajeclike output: row i, col j  = (i insertions, j deletions) count
      our convention:   matrix[i, j]   = (i deletions,  j insertions) prob
      So we transpose the parsed matrix before returning.

    trajeclike's output is followed by a single line giving the count of
    chop zones too large to fit in [0, Lmax]^2 (the "overflow") and a Total
    line.  We use total = matrix_sum + overflow as the normaliser, so the
    returned (probs, overflow_prob) sum to 1 across all simulated MM events.

    Returns:
        probs  : (Lmax+1, Lmax+1) probability matrix
        overflow_prob : scalar probability mass of MM zones too large
    """
    cmd = [
        TRAJECLIKE_BIN,
        '--mu', str(mu0),
        '--lambda', str(lam0),
        '--x', str(y_ins),
        '--y', str(x_del),
        '--time', str(t),
        '--maxlen', str(Lmax),
        '--simulate', '--counts',
        '-i', str(initlen),
        '-n', str(n_trials),
        '-d', str(seed),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    matrix_rows = []
    overflow_count = 0.0
    parsed_matrix = False
    for ln in lines:
        try:
            row = [float(p) for p in ln.split()]
        except ValueError:
            continue
        if not parsed_matrix and len(row) == Lmax + 1:
            matrix_rows.append(row)
            if len(matrix_rows) == Lmax + 1:
                parsed_matrix = True
        elif parsed_matrix and len(row) == 1:
            overflow_count = row[0]
            break
    counts = np.array(matrix_rows, dtype=float)
    assert counts.shape == (Lmax + 1, Lmax + 1), (
        f"trajeclike output shape {counts.shape} != ({Lmax+1}, {Lmax+1}); "
        f"raw output starts with: {res.stdout[:200]!r}")
    counts = counts.T  # convert to (deletions, insertions) indexing
    total = counts.sum() + overflow_count
    if total <= 0:
        return counts, 0.0
    return counts / total, overflow_count / total


def _sim_cache_key(lam0, mu0, x, y, t_grid, n_trials, initlen, Lmax):
    """Stable hash of all inputs that determine the simulation outputs."""
    h = hashlib.sha256()
    h.update(np.asarray([lam0, mu0, x, y, n_trials, initlen, Lmax],
                          dtype=np.float64).tobytes())
    h.update(np.asarray(t_grid, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _sim_worker(args):
    """Worker for multiprocessing.Pool.imap_unordered.  Returns (fi, frame, ov)."""
    (fi, t, lam0, mu0, x_del, y_ins, Lmax, n_trials, initlen, seed) = args
    frame, ov = run_gillespie_sim(
        t, lam0, mu0, x_del, y_ins, Lmax, n_trials, initlen, seed)
    return fi, frame, ov


def precompute_gillespie_grid(lam0, mu0, x, y, t_grid, Lmax, n_trials,
                               initlen, cache_dir=DEFAULT_SIM_CACHE_DIR,
                               n_workers=None):
    """Run trajeclike at every t in t_grid, parallelised across cores.

    Each frame uses a deterministic seed (fi+1) so the run is fully
    reproducible regardless of completion order.

    Returns (G_sim, overflow) where G_sim is (n, Lmax+1, Lmax+1) and
    overflow is a (n,) array of overflow probabilities per frame.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _sim_cache_key(lam0, mu0, x, y, t_grid, n_trials, initlen, Lmax)
    path = os.path.join(cache_dir, f'sim_{key}.npz')
    if os.path.exists(path):
        d = np.load(path)
        if 'overflow' in d.files:
            print(f"  loaded cached sim from {path}")
            return d['G_sim'], d['overflow']
        else:
            # Old-format cache without overflow; recompute.
            print(f"  cached sim at {path} lacks overflow data; recomputing")

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    n_workers = max(1, min(n_workers, len(t_grid)))
    print(f"  running {len(t_grid)} sims with {n_workers} workers...")

    work = [(fi, max(t, 1e-6), lam0, mu0, x, y, Lmax, n_trials, initlen, fi + 1)
            for fi, t in enumerate(t_grid)]
    G_sim = np.zeros((len(t_grid), Lmax + 1, Lmax + 1))
    overflow = np.zeros(len(t_grid))

    import time as _time
    t_start = _time.monotonic()
    n_done = 0
    progress_step = max(1, len(t_grid) // 10)

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            for fi, frame, ov in pool.imap_unordered(_sim_worker, work, chunksize=2):
                G_sim[fi] = frame
                overflow[fi] = ov
                n_done += 1
                if n_done % progress_step == 0:
                    el = _time.monotonic() - t_start
                    eta = el * (len(t_grid) - n_done) / max(1, n_done)
                    print(f"  sim {n_done}/{len(t_grid)}  "
                          f"({el:.1f}s elapsed, ~{eta:.1f}s remaining)")
    else:
        for w in work:
            fi, frame, ov = _sim_worker(w)
            G_sim[fi] = frame
            overflow[fi] = ov
            n_done += 1
            if n_done % progress_step == 0:
                el = _time.monotonic() - t_start
                eta = el * (len(t_grid) - n_done) / max(1, n_done)
                print(f"  sim {n_done}/{len(t_grid)}  "
                      f"({el:.1f}s elapsed, ~{eta:.1f}s remaining)")

    np.savez_compressed(
        path,
        G_sim=G_sim,
        overflow=overflow,
        lam0=lam0, mu0=mu0, x_del=x, y_ins=y,
        t_grid=np.asarray(t_grid),
        n_trials=n_trials, initlen=initlen, Lmax=Lmax)
    print(f"  cached sim -> {path}")
    return G_sim, overflow


def kl_divergence_per_frame(P_frames, Q_frames, eps=1e-30):
    """Conditional KL(P || Q) per frame, where P and Q are
    (n_frames, Lmax+1, Lmax+1) matrices.  Each frame's P and Q are
    renormalised to sum to 1 over the (i, j) grid before the KL is computed
    -- so we're comparing the conditional distribution given the gap fits
    in [0, Lmax]^2.

    Cells where P == 0 contribute 0 (by convention 0 log 0/q = 0); Q is
    clipped to eps for the log.  Returns a (n_frames,) array.
    """
    n = P_frames.shape[0]
    out = np.full(n, np.nan)
    for fi in range(n):
        P = P_frames[fi]
        Q = Q_frames[fi]
        ps = P.sum()
        qs = Q.sum()
        if ps <= 0 or qs <= 0:
            continue
        Pn = P / ps
        Qn = Q / qs
        Q_safe = np.maximum(Qn, eps)
        mask = Pn > 0
        if mask.sum() == 0:
            out[fi] = 0.0
            continue
        out[fi] = float(np.sum(Pn[mask] * (np.log(Pn[mask]) - np.log(Q_safe[mask]))))
    return out


def kl_joint_per_frame(P_frames, P_overflow, Q_frames, Q_overflow, eps=1e-30):
    """Joint KL(P || Q) per frame on (Lmax+1)^2 + 1 cells.  Each frame already
    comes with an overflow probability (sim from trajeclike, analytic via the
    wide-DP).  Both distributions sum to 1 across (i, j) cells + overflow.

    Cells where P == 0 contribute 0 (0 log 0/q = 0).  Q cells / overflow are
    clipped to eps for safety.  Returns (n_frames,) array.
    """
    n = P_frames.shape[0]
    out = np.full(n, np.nan)
    for fi in range(n):
        P = P_frames[fi]
        Q = Q_frames[fi]
        Po = P_overflow[fi]
        Qo = max(eps, Q_overflow[fi])
        if P.sum() + Po <= 0 or Q.sum() + Qo <= 0:
            continue
        Q_safe = np.maximum(Q, eps)
        mask = P > 0
        kl = float(np.sum(P[mask] * (np.log(P[mask]) - np.log(Q_safe[mask]))))
        if Po > 0:
            kl += Po * (np.log(Po) - np.log(Qo))
        out[fi] = kl
    return out


def analytic_mm_with_overflow(T, Lmax, Lmax_wide=None):
    """Return (G_in_range, overflow_prob) where the analytic MM gap probs
    G_MM(i, j) on a wider grid (Lmax_wide >= Lmax) are taken as the natural
    normalisation -- so G_in_range[i, j] = G_MM(i, j) on [0, Lmax]^2 and
    overflow_prob captures probability mass in (i, j > Lmax) within the
    wide grid.

    Probabilities are scaled to sum to 1 across (in-range cells) + overflow:
    we use sum_wide = sum over (i, j) in [0, Lmax_wide]^2 of G_MM_wide(i, j)
    as the proxy for the total MM-event probability.
    """
    if Lmax_wide is None:
        Lmax_wide = max(200, Lmax * 4)
    G_wide = mm_gap_probs(T, Lmax_wide)
    sum_wide = float(G_wide.sum())
    G_in_range = G_wide[:Lmax + 1, :Lmax + 1] / max(sum_wide, 1e-30)
    overflow = max(0.0, 1.0 - float(G_in_range.sum()))
    return G_in_range, overflow


def smooth_1d_gaussian(x, sigma):
    """Edge-padded Gaussian smoothing without scipy dependency."""
    x = np.asarray(x, dtype=float)
    if sigma <= 0:
        return x.copy()
    half = int(np.ceil(3 * sigma))
    k = np.arange(-half, half + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    # Handle NaN by interpolating
    if np.any(np.isnan(x)):
        idx = np.arange(len(x))
        good = ~np.isnan(x)
        if good.sum() < 2:
            return x.copy()
        x = np.interp(idx, idx[good], x[good])
    padded = np.pad(x, half, mode='edge')
    return np.convolve(padded, w, mode='valid')


def load_pfam_mm_frames(path, t_grid, Lmax):
    """Load a Pfam-corpus gap-counts npz (gap_counts indexed as
    (n_tau, gap_type, Lmax_file+1, Lmax_file+1)) and return MM gap
    probabilities (+ overflow) interpolated to each t in t_grid.

    Interpolation is linear in log(tau): each Pfam tau bin's MM counts
    are renormalised over [0, Lmax]^2 (in-range probability) plus a
    single overflow cell (probability mass in (i, j) > Lmax within the
    file's Lmax_file).  For t < tau_centers[0] or t > tau_centers[-1] we
    clamp to the boundary bin.

    Returns:
      G_pfam:   (n_frames, Lmax+1, Lmax+1) MM probabilities per frame
      ov_pfam:  (n_frames,)               overflow probability per frame
      meta:     dict with tau_centers and other info
    """
    d = np.load(path)
    counts = d['gap_counts'][:, 1].astype(np.float64)  # MM only: gap_type axis index 1
    pfam_tau = d['tau_centers'].astype(np.float64)
    Lmax_file = int(d['Lmax'])
    # In-range submatrix per bin + overflow probability
    in_range = counts[:, :Lmax + 1, :Lmax + 1]
    total_per_bin = counts.sum(axis=(1, 2))  # all cells in file
    in_range_sum = in_range.sum(axis=(1, 2))
    overflow_per_bin = total_per_bin - in_range_sum
    safe_total = np.maximum(total_per_bin, 1.0)
    P_bins = in_range / safe_total[:, None, None]
    P_overflow_bins = overflow_per_bin / safe_total

    # Linear interpolation in log(tau)
    log_tau = np.log(pfam_tau)
    G_pfam = np.zeros((len(t_grid), Lmax + 1, Lmax + 1))
    ov_pfam = np.zeros(len(t_grid))
    for fi, t in enumerate(t_grid):
        t_eff = max(t, 1e-9)
        log_t = np.log(t_eff)
        if log_t <= log_tau[0]:
            G_pfam[fi] = P_bins[0]
            ov_pfam[fi] = P_overflow_bins[0]
        elif log_t >= log_tau[-1]:
            G_pfam[fi] = P_bins[-1]
            ov_pfam[fi] = P_overflow_bins[-1]
        else:
            j = int(np.searchsorted(log_tau, log_t))
            j = min(max(j, 1), len(log_tau) - 1)
            w = (log_t - log_tau[j - 1]) / (log_tau[j] - log_tau[j - 1])
            G_pfam[fi] = (1.0 - w) * P_bins[j - 1] + w * P_bins[j]
            ov_pfam[fi] = (1.0 - w) * P_overflow_bins[j - 1] + w * P_overflow_bins[j]
    meta = dict(
        tau_centers=pfam_tau,
        tau_edges=d['tau_edges'] if 'tau_edges' in d.files else None,
        n_cherries_per_bin=d['n_cherries_per_bin'].astype(np.int64)
            if 'n_cherries_per_bin' in d.files else None,
        Lmax_file=Lmax_file,
        n_cherries=int(d['n_cherries_per_bin'].sum()) if 'n_cherries_per_bin' in d.files else None,
        n_families_ok=int(d['n_families_ok']) if 'n_families_ok' in d.files else None,
    )
    return G_pfam, ov_pfam, meta


def build_panel_configs(args, config_path=None):
    """Build per-panel parameter dicts.

    Each of panels 1..4 has its own GGI parameters (lam0, mu0, x_del, y_ins).
    Panel 4 additionally has a "fixed" lam, mu pair (defaults to the boundary
    of its own GGI params).  Panel 5 has its own TKF92 params (lam, mu, r).

    Precedence (lowest -> highest):
       hardcoded defaults  <  CLI --lam0 --mu0 --x --y  <  JSON config

    Returns a dict with keys 'sim', 'flow', 'closed', 'rtonly', 'plain'.
    """
    base = dict(lam0=args.lam0, mu0=args.mu0,
                x_del=args.x, y_ins=args.y)
    panel = {k: dict(base) for k in ('sim', 'flow', 'closed', 'rtonly')}
    # Panel 5: default TKF92 params = boundary of base GGI.
    base_l0, base_m0, base_r0 = ggi_to_tkf92_at_t(
        base['lam0'], base['mu0'], base['x_del'], base['y_ins'], 0.0)
    panel['plain'] = dict(lam=base_l0, mu=base_m0, r=base_r0)
    # Panel 4: default fixed rates = boundary of own GGI (filled below).

    if config_path:
        with open(config_path) as fh:
            cfg = json.load(fh)
        alias = {
            'p1_sim': 'sim',     'sim': 'sim',
            'p2_flow': 'flow',   'flow': 'flow',
            'p3_closed': 'closed', 'closed': 'closed',
            'p4_rt': 'rtonly',   'p4_rtonly': 'rtonly', 'rtonly': 'rtonly',
            'p5_plain': 'plain', 'plain': 'plain',
        }
        for k, v in cfg.items():
            if k in alias and isinstance(v, dict):
                panel[alias[k]].update(v)
            else:
                print(f"  [config] WARNING: unknown key {k!r} ignored")

    # Default fixed-(lam, mu) for the rt-only panel = its own GGI boundary
    rt_l0, rt_m0, _ = ggi_to_tkf92_at_t(
        panel['rtonly']['lam0'], panel['rtonly']['mu0'],
        panel['rtonly']['x_del'], panel['rtonly']['y_ins'], 0.0)
    panel['rtonly'].setdefault('lam_fixed', rt_l0)
    panel['rtonly'].setdefault('mu_fixed', rt_m0)
    return panel


def ggi_to_tkf92_at_t(lam0, mu0, x, y, t):
    """Closed-form (lam(t), mu(t), r(t)) under the GGI matched flow.

    r*(0) = (lam0 y(1-x) + mu0 x(1-y)) / (lam0(1-x) + mu0(1-y))
    r_inf = r*(0) / (2 - r*(0))
    k     = (lam0 + mu0)(2 - r*(0)) / (1 - r*(0))
    r(t)  = r_inf + (r*(0) - r_inf) exp(-k t)
    lam(t) = lam0 / (1 - r(t)), mu(t) = mu0 / (1 - r(t))
    """
    num = lam0 * y * (1 - x) + mu0 * x * (1 - y)
    den = lam0 * (1 - x) + mu0 * (1 - y)
    r_star = num / den
    r_inf = r_star / (2.0 - r_star)
    k = (lam0 + mu0) * (2.0 - r_star) / (1.0 - r_star)
    r_t = r_inf + (r_star - r_inf) * np.exp(-k * t)
    # Fixed-rate per wideboy_to_lambda.md 2026-06-03: lam_t, mu_t held
    # constant at boundary; only r_t evolves.
    lam_t = lam0 / max(1.0 - r_star, 1e-30)
    mu_t = mu0 / max(1.0 - r_star, 1e-30)
    return lam_t, mu_t, r_t


# -----------------------------------------------------------------
# Animation driver
# -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lam0', type=float, default=0.035,
                        help='GGI insertion rate per link')
    parser.add_argument('--mu0', type=float, default=0.040,
                        help='GGI deletion rate per residue')
    parser.add_argument('--x', type=float, default=0.7,
                        help='GGI deletion length geom (x_del)')
    parser.add_argument('--y', type=float, default=0.4,
                        help='GGI insertion length geom (y_ins)')
    parser.add_argument('--Lmax', type=int, default=20,
                        help='Max gap size shown')
    parser.add_argument('--t-max', type=float, default=10.0,
                        help='Max evolutionary time (= seconds of movie)')
    parser.add_argument('--fps', type=int, default=24,
                        help='Frames per second')
    parser.add_argument('--out', type=str, default='/tmp/gap_movie.mp4',
                        help='Output MP4 path')
    parser.add_argument('--png-dir', type=str, default=None,
                        help='If set (fallback when ffmpeg unavailable), save '
                             'PNG every ceil(fps/4) frames here.')
    parser.add_argument('--sim-trials', type=int, default=32000,
                        help='Number of trajeclike Gillespie trials per frame')
    parser.add_argument('--sim-initlen', type=int, default=200,
                        help='Initial sequence length per trajeclike run')
    parser.add_argument('--sim-workers', type=int, default=0,
                        help='Number of parallel trajeclike workers (0 = cpu_count - 1)')
    parser.add_argument('--sim-cache-dir', type=str, default=None,
                        help=f'Where to cache trajeclike simulation results '
                             f'(default {DEFAULT_SIM_CACHE_DIR})')
    parser.add_argument('--no-sim', action='store_true',
                        help='Skip the Gillespie sim panel (4-panel mode)')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON file with per-panel parameter overrides.  '
                             'Top-level keys: sim, flow, closed, rtonly, plain '
                             '(or p1_sim, p2_flow, p3_closed, p4_rt, p5_plain).  '
                             'Each value is a dict; see build_panel_configs.')
    parser.add_argument('--pfam-counts', type=str, default=None,
                        help='Path to a Pfam-corpus gap-counts npz '
                             '(from build_gap_counts.py).  When present, '
                             'an empirical Pfam panel is added as panel 0 '
                             '(leftmost), with MM gap counts linearly '
                             'interpolated in log(tau) between the file\'s '
                             'tau bins.')
    parser.add_argument('--kl-source',
                        choices=('sim', 'pfam', 'flow', 'closed', 'rtonly', 'plain'),
                        default=None,
                        help='Which panel is the KL ground-truth source.  '
                             'Default = pfam if --pfam-counts is given, else sim.  '
                             'KL plots appear under every other panel; the source '
                             'slot is annotated.')
    parser.add_argument('--kl-mode', choices=('conditional', 'joint'),
                        default='conditional',
                        help='How to compute KL(sim || panel) per frame.  '
                             'conditional (default): renormalise both matrices '
                             'to sum to 1 over (i,j) ∈ [0, Lmax]^2, then KL.  '
                             'joint: keep absolute MM probabilities; pad with a '
                             'single "overflow" cell (sim overflow from trajeclike, '
                             'analytic via wide DP at Lmax_wide=max(200, 4Lmax)); '
                             'KL over (Lmax+1)^2 + 1 cells.  The joint mode is the '
                             'closer match to what the matched-flow ODE optimises.')
    args = parser.parse_args()
    panels_cfg = build_panel_configs(args, args.config)

    print("Panel configurations:")
    for k in ('sim', 'flow', 'closed', 'rtonly', 'plain'):
        print(f"  {k:8s}: {panels_cfg[k]}")

    # Closed-form trajectory metadata (printed for the 'closed' panel's GGI)
    p_closed = panels_cfg['closed']
    lam_T0, mu_T0, r_star = ggi_to_tkf92_at_t(
        p_closed['lam0'], p_closed['mu0'],
        p_closed['x_del'], p_closed['y_ins'], 0.0)
    r_inf = r_star / (2.0 - r_star)
    k = (p_closed['lam0'] + p_closed['mu0']) * (2.0 - r_star) / (1.0 - r_star)
    print(f"\nClosed-form panel:")
    print(f"  boundary: lam_T(0)={lam_T0:.4f}, mu_T(0)={mu_T0:.4f}, "
          f"r*(0)={r_star:.4f}")
    print(f"  r_inf = {r_inf:.4f}, decay k = {k:.4f}  "
          f"(half-life {np.log(2)/k:.2f})")

    # Plain TKF92 params (panel 5)
    p_plain = panels_cfg['plain']
    PLAIN_LAM, PLAIN_MU, PLAIN_R = p_plain['lam'], p_plain['mu'], p_plain['r']

    n_frames = int(args.t_max * args.fps) + 1
    t_grid = np.linspace(0.0, args.t_max, n_frames)
    print(f"\nFrames: {n_frames} (t in [0, {args.t_max}] at fps={args.fps})")

    # Precompute heatmaps so we can establish a common color scale
    #
    # Five panels (left -> right):
    #   SIM      -- Gillespie simulation of the GGI process via trajeclike.
    #               Ground truth.
    #   FLOW     -- GGI-steered TKF92 via the FULL matched-flow ODE for
    #               (lam(t), mu(t), r(t)) integrated numerically.
    #   CLOSED   -- GGI-steered TKF92 via the CLOSED-FORM approximations
    #               for (lam*(t), mu*(t), r*(t)) (the first-order-exact
    #               heuristic from composition-renormalization.tex).
    #   RT_ONLY  -- r*(t) only:  lam = lam_T(0), mu = mu_T(0) pinned to
    #               boundary, r(t) closed-form.  Isolates the r(t)
    #               contribution.
    #   PLAIN    -- plain TKF92, (lam, mu, r) all constant at the GGI t=0
    #               boundary values.
    Lmax = args.Lmax

    # Each panel that uses GGI gets its own trajectory.
    p_flow = panels_cfg['flow']
    p_rtonly = panels_cfg['rtonly']
    p_sim = panels_cfg['sim']

    # Optional Pfam empirical panel (panel 0).
    has_pfam = args.pfam_counts is not None
    G_pfam_frames = None
    ov_pfam = None
    pfam_meta = None
    if has_pfam:
        print(f"\nLoading Pfam gap counts from {args.pfam_counts} ...")
        G_pfam_frames, ov_pfam, pfam_meta = load_pfam_mm_frames(
            args.pfam_counts, t_grid, args.Lmax)
        print(f"  Pfam tau range: [{pfam_meta['tau_centers'][0]:.4g}, "
              f"{pfam_meta['tau_centers'][-1]:.4g}]  ({len(pfam_meta['tau_centers'])} bins)")
        if pfam_meta.get('n_cherries') is not None:
            print(f"  Pfam cherries: {pfam_meta['n_cherries']}  "
                  f"families: {pfam_meta.get('n_families_ok')}")

    print(f"\nIntegrating flow ODE for FLOW panel...")
    flow_traj = integrate_full_flow(
        p_flow['lam0'], p_flow['mu0'],
        p_flow['x_del'], p_flow['y_ins'], t_grid)
    print(f"  flow_traj[0] = {flow_traj[0]}")
    print(f"  flow_traj[-1] = {flow_traj[-1]}")

    G_sim_frames = np.zeros((n_frames, Lmax + 1, Lmax + 1))
    G_flow_frames = np.zeros((n_frames, Lmax + 1, Lmax + 1))
    G_closed_frames = np.zeros((n_frames, Lmax + 1, Lmax + 1))
    G_rtonly_frames = np.zeros((n_frames, Lmax + 1, Lmax + 1))
    G_plain_frames = np.zeros((n_frames, Lmax + 1, Lmax + 1))
    # Joint-mode "absolute" probabilities and overflow (filled only if joint mode)
    Gj_flow_frames = np.zeros((n_frames, Lmax + 1, Lmax + 1))
    Gj_closed_frames = np.zeros((n_frames, Lmax + 1, Lmax + 1))
    Gj_rtonly_frames = np.zeros((n_frames, Lmax + 1, Lmax + 1))
    Gj_plain_frames = np.zeros((n_frames, Lmax + 1, Lmax + 1))
    ov_flow = np.zeros(n_frames)
    ov_closed = np.zeros(n_frames)
    ov_rtonly = np.zeros(n_frames)
    ov_plain = np.zeros(n_frames)
    sim_overflow = np.zeros(n_frames)

    closed_params = np.zeros((n_frames, 3))  # (lam*, mu*, r*) per frame
    rtonly_r = np.zeros(n_frames)             # r*(t) of the rtonly panel

    RT_LAM_FIXED = p_rtonly['lam_fixed']
    RT_MU_FIXED = p_rtonly['mu_fixed']
    Lmax_wide = max(200, Lmax * 4)  # wide DP for joint-mode normalisation

    for fi, t in enumerate(t_grid):
        t_eff = max(t, 1e-6)
        lam_c, mu_c, r_c = ggi_to_tkf92_at_t(
            p_closed['lam0'], p_closed['mu0'],
            p_closed['x_del'], p_closed['y_ins'], t_eff)
        closed_params[fi] = (lam_c, mu_c, r_c)
        _, _, r_rt = ggi_to_tkf92_at_t(
            p_rtonly['lam0'], p_rtonly['mu0'],
            p_rtonly['x_del'], p_rtonly['y_ins'], t_eff)
        rtonly_r[fi] = r_rt
        lam_f, mu_f, r_f = flow_traj[fi]
        T_flow = tkf92_trans(lam_f, mu_f, r_f, t_eff)
        T_closed = tkf92_trans(lam_c, mu_c, r_c, t_eff)
        T_rtonly = tkf92_trans(RT_LAM_FIXED, RT_MU_FIXED, r_rt, t_eff)
        T_plain = tkf92_trans(PLAIN_LAM, PLAIN_MU, PLAIN_R, t_eff)
        G_flow_frames[fi] = mm_gap_probs(T_flow, Lmax)
        G_closed_frames[fi] = mm_gap_probs(T_closed, Lmax)
        G_rtonly_frames[fi] = mm_gap_probs(T_rtonly, Lmax)
        G_plain_frames[fi] = mm_gap_probs(T_plain, Lmax)
        if args.kl_mode == 'joint':
            Gj_flow_frames[fi], ov_flow[fi] = analytic_mm_with_overflow(
                T_flow, Lmax, Lmax_wide)
            Gj_closed_frames[fi], ov_closed[fi] = analytic_mm_with_overflow(
                T_closed, Lmax, Lmax_wide)
            Gj_rtonly_frames[fi], ov_rtonly[fi] = analytic_mm_with_overflow(
                T_rtonly, Lmax, Lmax_wide)
            Gj_plain_frames[fi], ov_plain[fi] = analytic_mm_with_overflow(
                T_plain, Lmax, Lmax_wide)

    if not args.no_sim:
        print(f"\nGillespie sim: {n_frames} frames "
              f"({args.sim_trials} trials, init_len={args.sim_initlen})")
        n_workers = args.sim_workers if args.sim_workers > 0 else None
        G_sim_frames, sim_overflow = precompute_gillespie_grid(
            p_sim['lam0'], p_sim['mu0'], p_sim['x_del'], p_sim['y_ins'],
            t_grid, Lmax, args.sim_trials, args.sim_initlen,
            cache_dir=args.sim_cache_dir or DEFAULT_SIM_CACHE_DIR,
            n_workers=n_workers)

    # Common log-color scale across all frames and all panels
    show_sim = not args.no_sim
    panels = [G_flow_frames, G_closed_frames, G_rtonly_frames, G_plain_frames]
    if show_sim:
        panels.append(G_sim_frames)
    if has_pfam:
        panels.append(G_pfam_frames)
    nonzero = np.concatenate([p[p > 0].ravel() for p in panels])
    vmax = float(np.max(nonzero))
    vmin = max(float(np.min(nonzero)), vmax * 1e-8)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    print(f"  log color scale: [{vmin:.2e}, {vmax:.2e}]")

    # KL traces: pick the source panel.  Default is pfam if --pfam-counts
    # was supplied, else sim.
    kl_source = args.kl_source
    if kl_source is None:
        kl_source = 'pfam' if has_pfam else 'sim'
    if kl_source == 'pfam' and not has_pfam:
        print(f"  WARNING: --kl-source pfam requires --pfam-counts; "
              f"falling back to sim.")
        kl_source = 'sim'
    if kl_source == 'sim' and not show_sim:
        print(f"  WARNING: --kl-source sim requires the sim panel; "
              f"falling back to {'pfam' if has_pfam else 'closed'}.")
        kl_source = 'pfam' if has_pfam else 'closed'
    print(f"  KL source = {kl_source!r}")

    # Map name -> (conditional probability frames, overflow frames in joint mode)
    panel_frames_cond = {
        'flow': G_flow_frames,
        'closed': G_closed_frames,
        'rtonly': G_rtonly_frames,
        'plain': G_plain_frames,
    }
    panel_frames_joint = {
        'flow': (Gj_flow_frames, ov_flow),
        'closed': (Gj_closed_frames, ov_closed),
        'rtonly': (Gj_rtonly_frames, ov_rtonly),
        'plain': (Gj_plain_frames, ov_plain),
    }
    if show_sim:
        panel_frames_cond['sim'] = G_sim_frames
        # sim is empirical; for joint, the "absolute" version of the sim is
        # just G_sim (which sums to 1 - sim_overflow already by trajeclike's
        # normalisation).
        panel_frames_joint['sim'] = (G_sim_frames, sim_overflow)
    if has_pfam:
        panel_frames_cond['pfam'] = G_pfam_frames
        panel_frames_joint['pfam'] = (G_pfam_frames, ov_pfam)

    # Pick source frames
    if args.kl_mode == 'joint':
        P_src_frames, P_src_overflow = panel_frames_joint[kl_source]
    else:
        P_src_frames = panel_frames_cond[kl_source]
        P_src_overflow = None

    kl_smooth_sigma = max(2.0, args.fps / 8.0)  # ~1/8 second smoothing
    kl_traces = {}
    if args.kl_mode == 'joint':
        print(f"  Computing JOINT KL (Lmax+1)^2 + 1 cells, with absolute "
              f"MM probabilities; analytic normalisation via wide DP at "
              f"Lmax_wide={Lmax_wide}.")
    for name in ('pfam', 'sim', 'flow', 'closed', 'rtonly', 'plain'):
        if name == kl_source or name not in panel_frames_cond:
            continue
        if args.kl_mode == 'joint':
            Q_frames, Q_ov = panel_frames_joint[name]
            raw = kl_joint_per_frame(P_src_frames, P_src_overflow,
                                       Q_frames, Q_ov)
        else:
            raw = kl_divergence_per_frame(P_src_frames, panel_frames_cond[name])
        kl_traces[name] = smooth_1d_gaussian(raw, kl_smooth_sigma)

    if kl_traces:
        all_kl = np.concatenate(list(kl_traces.values()))
        kl_finite = all_kl[np.isfinite(all_kl)]
        kl_ymin = float(np.min(kl_finite))
        kl_ymax = float(np.max(kl_finite))
        kl_pad = 0.05 * (kl_ymax - kl_ymin + 1e-12)
        kl_ymin -= kl_pad
        kl_ymax += kl_pad
        print(f"  KL({kl_source}||panel) range ({args.kl_mode}): "
              f"[{kl_ymin:.4f}, {kl_ymax:.4f}]")
    else:
        kl_ymin, kl_ymax = 0.0, 1.0

    # ----- Figure layout -----
    # Active panel ordering (left -> right): pfam (if any), sim, flow, closed,
    # rtonly, plain.  Rows (above-the-heatmap = GGI generative params;
    # below-the-heatmap = TKF92 surrogate):
    #   0   time seek bar
    #   1   GGI lam0 / mu0 bars   (sim, flow, closed, rtonly)
    #   2   GGI x_del / y_ins bars (sim, flow, closed, rtonly)
    #   3   heatmap + colorbar
    #   4   TKF92 lambda / mu     (flow, closed, rtonly, plain)
    #   5   TKF92 r               (flow, closed, rtonly, plain)
    #   6   KL traces / Pfam histogram
    active_panels = []
    if has_pfam:
        active_panels.append('pfam')
    if show_sim:
        active_panels.append('sim')
    active_panels += ['flow', 'closed', 'rtonly', 'plain']
    n_panels = len(active_panels)
    has_kl_row = bool(kl_traces) or kl_source is not None
    height_ratios = [0.35, 0.5, 0.5, 6.0, 0.5, 0.4]  # time, ggi_rate, ggi_xy, hm, tkf_rate, tkf_r
    if has_kl_row:
        height_ratios.append(1.3)
    n_rows = len(height_ratios)
    # Match the figure dpi to the save dpi (120) so dimensions are exact,
    # then snap inches up to a 0.2 grid so pixel counts are always even.
    _DPI = 120
    import math
    fig_height_raw = sum(height_ratios) * 1.05 + 1.0
    fig_width_raw = max(18, 4.4 * n_panels + 1.0)
    fig_height = math.ceil(fig_height_raw / 0.2) * 0.2
    fig_width = math.ceil(fig_width_raw / 0.2) * 0.2
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=_DPI)
    gs = fig.add_gridspec(
        n_rows, n_panels + 1,
        height_ratios=height_ratios,
        width_ratios=[1.0] * n_panels + [0.04],
        hspace=0.55, wspace=0.18,
        left=0.035, right=0.97, top=0.94, bottom=0.05)

    ax_time = fig.add_subplot(gs[0, :n_panels])

    # Row indices (above-the-heatmap = GGI; below = TKF92)
    ROW_GGI_RATE, ROW_GGI_XY, ROW_HM, ROW_RATE, ROW_R = 1, 2, 3, 4, 5
    ROW_KL = 6 if has_kl_row else None

    # Build axes dicts keyed by panel name
    ax_ggi_rate = {name: fig.add_subplot(gs[ROW_GGI_RATE, ci])
                    for ci, name in enumerate(active_panels)}
    ax_xy = {name: fig.add_subplot(gs[ROW_GGI_XY, ci])
             for ci, name in enumerate(active_panels)}
    ax_hm = {name: fig.add_subplot(gs[ROW_HM, ci])
             for ci, name in enumerate(active_panels)}
    ax_rate = {name: fig.add_subplot(gs[ROW_RATE, ci])
               for ci, name in enumerate(active_panels)}
    ax_r = {name: fig.add_subplot(gs[ROW_R, ci])
            for ci, name in enumerate(active_panels)}
    if has_kl_row:
        ax_kl = {name: fig.add_subplot(gs[ROW_KL, ci])
                 for ci, name in enumerate(active_panels)}
    else:
        ax_kl = {}

    ax_cbar = fig.add_subplot(gs[ROW_HM, n_panels])

    # Local aliases used downstream by existing per-panel setup code.
    ax_hm_pfam = ax_hm.get('pfam')
    ax_hm_sim = ax_hm.get('sim')
    ax_hm_flow = ax_hm['flow']
    ax_hm_closed = ax_hm['closed']
    ax_hm_rtonly = ax_hm['rtonly']
    ax_hm_plain = ax_hm['plain']
    ax_rate_pfam = ax_rate.get('pfam')
    ax_rate_sim = ax_rate.get('sim')
    ax_rate_flow = ax_rate['flow']
    ax_rate_closed = ax_rate['closed']
    ax_rate_rtonly = ax_rate['rtonly']
    ax_rate_plain = ax_rate['plain']
    ax_r_pfam = ax_r.get('pfam')
    ax_r_sim = ax_r.get('sim')
    ax_r_flow = ax_r['flow']
    ax_r_closed = ax_r['closed']
    ax_r_rtonly = ax_r['rtonly']
    ax_r_plain = ax_r['plain']

    # ----- Heatmaps -----
    panels_spec = []
    if has_pfam:
        nc = pfam_meta.get('n_cherries')
        nf = pfam_meta.get('n_families_ok')
        cherries_str = f'{nc:,}' if nc else '?'
        fams_str = f'{nf:,}' if nf else '?'
        panels_spec.append((
            ax_hm_pfam,
            r'(0) Pfam empirical (linearly interp.)'
            '\n'
            fr'{cherries_str} cherries from {fams_str} families'))
    if show_sim:
        panels_spec.append((
            ax_hm_sim,
            r'(1) GGI Gillespie simulation'
            '\n'
            fr'sim: $\lambda_0$={p_sim["lam0"]:.4g}, $\mu_0$={p_sim["mu0"]:.4g}, '
            fr'$x$={p_sim["x_del"]:.3g}, $y$={p_sim["y_ins"]:.3g}'))
    panels_spec += [
        (ax_hm_flow,
         r'(2) GGI-steered TKF92, flow ODE'
         '\n'
         fr'GGI: $\lambda_0$={p_flow["lam0"]:.4g}, $\mu_0$={p_flow["mu0"]:.4g}, '
         fr'$x$={p_flow["x_del"]:.3g}, $y$={p_flow["y_ins"]:.3g}'),
        (ax_hm_closed,
         r'(3) GGI-steered TKF92, closed-form'
         '\n'
         fr'GGI: $\lambda_0$={p_closed["lam0"]:.4g}, $\mu_0$={p_closed["mu0"]:.4g}, '
         fr'$x$={p_closed["x_del"]:.3g}, $y$={p_closed["y_ins"]:.3g}'),
        (ax_hm_rtonly,
         r'(4) Closed-form $r^*(t)$, fixed $\lambda$, $\mu$'
         '\n'
         fr'GGI: $\lambda_0$={p_rtonly["lam0"]:.4g}, $\mu_0$={p_rtonly["mu0"]:.4g}, '
         fr'$x$={p_rtonly["x_del"]:.3g}, $y$={p_rtonly["y_ins"]:.3g}'),
        (ax_hm_plain,
         r'(5) Plain TKF92'
         '\n'
         fr'$\lambda$={PLAIN_LAM:.4g}, $\mu$={PLAIN_MU:.4g}, $r$={PLAIN_R:.4g}'),
    ]
    ims = []
    for ax, title in panels_spec:
        ax.set_xlabel(r'$j$  (insertions)')
        ax.set_ylabel(r'$i$  (deletions)')
        im = ax.imshow(G_closed_frames[0], norm=norm, cmap='viridis',
                       origin='upper', aspect='equal',
                       extent=(-0.5, Lmax + 0.5, Lmax + 0.5, -0.5))
        ax.set_title(title, fontsize=10.5, pad=6)
        ims.append(im)
    fig.colorbar(ims[0], cax=ax_cbar, label=r'$G_{MM}(i, j)$')

    # ----- Time seek bar -----
    ax_time.set_xlim(0, args.t_max)
    ax_time.set_ylim(0, 1)
    ax_time.set_yticks([])
    ax_time.set_xlabel(r'$t$  (evolutionary time)', fontsize=10, labelpad=2)
    ax_time.xaxis.set_label_position('top')
    ax_time.tick_params(axis='x', top=True, labeltop=True,
                         bottom=False, labelbottom=False, labelsize=9)
    # Background track
    ax_time.add_patch(Rectangle((0, 0.35), args.t_max, 0.3,
                                  facecolor='#dddddd', edgecolor='#999999',
                                  lw=0.5))
    seek_fill = Rectangle((0, 0.35), 0, 0.3,
                           facecolor='steelblue', edgecolor='none')
    ax_time.add_patch(seek_fill)
    seek_marker, = ax_time.plot([0], [0.5], 'o', color='navy',
                                  markersize=11, zorder=5)

    # Determine a clean max for the rate-bar x-axis (covers all panels)
    max_rate = max(
        np.max(closed_params[:, 0]), np.max(closed_params[:, 1]),
        np.max(flow_traj[:, 0]),     np.max(flow_traj[:, 1]),
        RT_LAM_FIXED, RT_MU_FIXED,
        PLAIN_LAM, PLAIN_MU,
        p_sim['lam0'], p_sim['mu0'])
    nice_max_rate = np.ceil(max_rate * 100) / 100  # round up to nearest 0.01

    def _setup_rate_axis(ax, title):
        ax.set_xlim(0, nice_max_rate)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([r'$\lambda$', r'$\mu$'], fontsize=11)
        ax.grid(axis='x', alpha=0.3, lw=0.5)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_title(title, fontsize=8.5, pad=2, loc='left')

    def _setup_r_axis(ax, title):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([0])
        ax.set_yticklabels(['$r$'], fontsize=11)
        ax.grid(axis='x', alpha=0.3, lw=0.5)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_title(title, fontsize=8.5, pad=2, loc='left')

    # ---- GGI generative params above each heatmap ----
    # Row 1: lam0, mu0 (rate scale)
    # Row 2: x_del, y_ins (0-1 scale)
    # Shown for sim, flow, closed, rtonly.  Blank for pfam and plain.
    GGI_X_COLOR = '#9467bd'  # x_del (deletion length geom)
    GGI_Y_COLOR = '#8c564b'  # y_ins (insertion length geom)
    GGI_LAM_COLOR = '#1f77b4'
    GGI_MU_COLOR = '#ff7f0e'
    _LABEL_BBOX = dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)

    def _setup_xy_axis(ax, panel_label):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.7, 1.7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['$x_{del}$', '$y_{ins}$'], fontsize=9)
        ax.grid(axis='x', alpha=0.3, lw=0.5)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_title(panel_label, fontsize=8.5, pad=2, loc='left')

    def _draw_xy_bars(ax, x_val, y_val):
        ax.barh(0, x_val, height=0.55, color=GGI_X_COLOR)
        ax.barh(1, y_val, height=0.55, color=GGI_Y_COLOR)
        ax.text(0.985, 0, f'{x_val:.4f}',
                va='center', ha='right', fontsize=9, bbox=_LABEL_BBOX)
        ax.text(0.985, 1, f'{y_val:.4f}',
                va='center', ha='right', fontsize=9, bbox=_LABEL_BBOX)

    def _setup_ggi_rate_axis(ax, panel_label, x_max):
        ax.set_xlim(0, x_max)
        ax.set_ylim(-0.7, 1.7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([r'$\lambda_0$', r'$\mu_0$'], fontsize=9)
        ax.grid(axis='x', alpha=0.3, lw=0.5)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_title(panel_label, fontsize=8.5, pad=2, loc='left')

    def _draw_ggi_rate_bars(ax, lam0, mu0, x_max):
        ax.barh(0, lam0, height=0.55, color=GGI_LAM_COLOR)
        ax.barh(1, mu0, height=0.55, color=GGI_MU_COLOR)
        ax.text(x_max * 0.985, 0, f'{lam0:.4f}',
                va='center', ha='right', fontsize=9, bbox=_LABEL_BBOX)
        ax.text(x_max * 0.985, 1, f'{mu0:.4f}',
                va='center', ha='right', fontsize=9, bbox=_LABEL_BBOX)

    # GGI rate-axis x-max: covers max of all GGI lam0, mu0 (small numbers).
    ggi_max_rate = max(
        p_sim['lam0'] if show_sim else 0.0,
        p_sim['mu0'] if show_sim else 0.0,
        p_flow['lam0'], p_flow['mu0'],
        p_closed['lam0'], p_closed['mu0'],
        p_rtonly['lam0'], p_rtonly['mu0'])
    nice_ggi_rate = max(0.01, np.ceil(ggi_max_rate * 100) / 100)

    # PFAM panel: blank slots for GGI rows
    if has_pfam:
        for ax in (ax_ggi_rate['pfam'], ax_xy['pfam']):
            ax.set_axis_off()
        ax_ggi_rate['pfam'].text(0.5, 0.5,
                                    '(empirical -- no GGI params)',
                                    transform=ax_ggi_rate['pfam'].transAxes,
                                    ha='center', va='center',
                                    fontsize=9, color='gray')
    # SIM panel: GGI params above
    if show_sim:
        _setup_ggi_rate_axis(ax_ggi_rate['sim'], r'GGI: $\lambda_0, \mu_0$',
                                nice_ggi_rate)
        _draw_ggi_rate_bars(ax_ggi_rate['sim'],
                               p_sim['lam0'], p_sim['mu0'], nice_ggi_rate)
        _setup_xy_axis(ax_xy['sim'], r'GGI: $x_{del}, y_{ins}$')
        _draw_xy_bars(ax_xy['sim'], p_sim['x_del'], p_sim['y_ins'])
    # FLOW / CLOSED / RTONLY panels: GGI params above
    _setup_ggi_rate_axis(ax_ggi_rate['flow'], r'GGI: $\lambda_0, \mu_0$',
                            nice_ggi_rate)
    _draw_ggi_rate_bars(ax_ggi_rate['flow'],
                           p_flow['lam0'], p_flow['mu0'], nice_ggi_rate)
    _setup_xy_axis(ax_xy['flow'], r'GGI: $x_{del}, y_{ins}$')
    _draw_xy_bars(ax_xy['flow'], p_flow['x_del'], p_flow['y_ins'])

    _setup_ggi_rate_axis(ax_ggi_rate['closed'], r'GGI: $\lambda_0, \mu_0$',
                            nice_ggi_rate)
    _draw_ggi_rate_bars(ax_ggi_rate['closed'],
                           p_closed['lam0'], p_closed['mu0'], nice_ggi_rate)
    _setup_xy_axis(ax_xy['closed'], r'GGI: $x_{del}, y_{ins}$')
    _draw_xy_bars(ax_xy['closed'], p_closed['x_del'], p_closed['y_ins'])

    _setup_ggi_rate_axis(ax_ggi_rate['rtonly'], r'GGI: $\lambda_0, \mu_0$',
                            nice_ggi_rate)
    _draw_ggi_rate_bars(ax_ggi_rate['rtonly'],
                           p_rtonly['lam0'], p_rtonly['mu0'], nice_ggi_rate)
    _setup_xy_axis(ax_xy['rtonly'], r'GGI: $x_{del}, y_{ins}$')
    _draw_xy_bars(ax_xy['rtonly'], p_rtonly['x_del'], p_rtonly['y_ins'])

    # PLAIN panel: no GGI generative params
    for ax in (ax_ggi_rate['plain'], ax_xy['plain']):
        ax.set_axis_off()
    ax_ggi_rate['plain'].text(0.5, 0.5,
                                  '(TKF92 -- no GGI generative params)',
                                  transform=ax_ggi_rate['plain'].transAxes,
                                  ha='center', va='center',
                                  fontsize=9, color='gray')

    # ---- Panel: PFAM (empirical, no rate/r bars; show metadata text) ----
    if has_pfam:
        # Blank the rate axis: show only the metadata text.
        ax_rate_pfam.set_axis_off()
        ax_rate_pfam.text(0.5, 0.5,
                            r'(empirical: no model rates)',
                            transform=ax_rate_pfam.transAxes,
                            ha='center', va='center', fontsize=9, color='gray')
        ax_r_pfam.set_axis_off()
        tau_lo, tau_hi = pfam_meta['tau_centers'][0], pfam_meta['tau_centers'][-1]
        ax_r_pfam.text(0.5, 0.5,
                        fr'$\tau$ data range: [{tau_lo:.3g}, {tau_hi:.3g}]',
                        transform=ax_r_pfam.transAxes,
                        ha='center', va='center', fontsize=9, color='gray')

    # ---- Panel: SIM (Gillespie) ----
    # GGI params go in the rows ABOVE the heatmap.  The TKF92-surrogate
    # rows (below the heatmap) get a blank annotation.
    if show_sim:
        for ax in (ax_rate_sim, ax_r_sim):
            ax.set_axis_off()
        ax_rate_sim.text(0.5, 0.5,
                          '(no TKF92 surrogate -- direct GGI sim)',
                          transform=ax_rate_sim.transAxes,
                          ha='center', va='center',
                          fontsize=9, color='gray')

    # ---- Panel: FLOW (full ODE) ----
    _setup_rate_axis(ax_rate_flow, r'flow ODE: $\lambda(t)$, $\mu(t)$')
    bar_lam_f = ax_rate_flow.barh(0, 0, height=0.6, color='#1f77b4')[0]
    bar_mu_f = ax_rate_flow.barh(1, 0, height=0.6, color='#ff7f0e')[0]
    _LABEL_BBOX = dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
    txt_lam_f = ax_rate_flow.text(nice_max_rate * 0.985, 0, '',
                                    va='center', ha='right', fontsize=9,
                                    bbox=_LABEL_BBOX)
    txt_mu_f = ax_rate_flow.text(nice_max_rate * 0.985, 1, '',
                                   va='center', ha='right', fontsize=9,
                                   bbox=_LABEL_BBOX)
    _setup_r_axis(ax_r_flow, r'flow ODE: $r(t)$')
    bar_r_f = ax_r_flow.barh(0, 0, height=0.5, color='#2ca02c')[0]
    txt_r_f = ax_r_flow.text(0.985, 0, '', va='center', ha='right',
                                fontsize=9, bbox=_LABEL_BBOX)

    # ---- Panel: CLOSED (closed-form) ----
    _setup_rate_axis(ax_rate_closed, r'closed form: $\lambda^*(t)$, $\mu^*(t)$')
    bar_lam_c = ax_rate_closed.barh(0, 0, height=0.6, color='#1f77b4')[0]
    bar_mu_c = ax_rate_closed.barh(1, 0, height=0.6, color='#ff7f0e')[0]
    txt_lam_c = ax_rate_closed.text(nice_max_rate * 0.985, 0, '',
                                       va='center', ha='right', fontsize=9,
                                       bbox=_LABEL_BBOX)
    txt_mu_c = ax_rate_closed.text(nice_max_rate * 0.985, 1, '',
                                      va='center', ha='right', fontsize=9,
                                      bbox=_LABEL_BBOX)
    _setup_r_axis(ax_r_closed, r'closed form: $r^*(t)$')
    bar_r_c = ax_r_closed.barh(0, 0, height=0.5, color='#2ca02c')[0]
    txt_r_c = ax_r_closed.text(0.985, 0, '', va='center', ha='right',
                                fontsize=9, bbox=_LABEL_BBOX)

    # ---- Panel: RT_ONLY ----
    _setup_rate_axis(ax_rate_rtonly,
                       r'fixed $\lambda$, $\mu$  (constant)')
    ax_rate_rtonly.barh(0, RT_LAM_FIXED, height=0.6, color='#1f77b4')
    ax_rate_rtonly.barh(1, RT_MU_FIXED, height=0.6, color='#ff7f0e')
    ax_rate_rtonly.text(nice_max_rate * 0.985, 0, f'{RT_LAM_FIXED:.4f}',
                         va='center', ha='right', fontsize=9, bbox=_LABEL_BBOX)
    ax_rate_rtonly.text(nice_max_rate * 0.985, 1, f'{RT_MU_FIXED:.4f}',
                         va='center', ha='right', fontsize=9, bbox=_LABEL_BBOX)
    _setup_r_axis(ax_r_rtonly, r'closed form: $r^*(t)$')
    bar_r_rt = ax_r_rtonly.barh(0, 0, height=0.5, color='#2ca02c')[0]
    txt_r_rt = ax_r_rtonly.text(0.985, 0, '', va='center', ha='right',
                                  fontsize=9, bbox=_LABEL_BBOX)

    # ---- Panel: PLAIN ----
    _setup_rate_axis(ax_rate_plain, r'$\lambda$, $\mu$  (constant)')
    ax_rate_plain.barh(0, PLAIN_LAM, height=0.6, color='#1f77b4')
    ax_rate_plain.barh(1, PLAIN_MU, height=0.6, color='#ff7f0e')
    ax_rate_plain.text(nice_max_rate * 0.985, 0, f'{PLAIN_LAM:.4f}',
                        va='center', ha='right', fontsize=9, bbox=_LABEL_BBOX)
    ax_rate_plain.text(nice_max_rate * 0.985, 1, f'{PLAIN_MU:.4f}',
                        va='center', ha='right', fontsize=9, bbox=_LABEL_BBOX)
    _setup_r_axis(ax_r_plain, r'$r$  (constant)')
    ax_r_plain.barh(0, PLAIN_R, height=0.5, color='#2ca02c')
    ax_r_plain.text(0.985, 0, f'{PLAIN_R:.4f}', va='center', ha='right',
                      fontsize=9, bbox=_LABEL_BBOX)

    # ----- KL traces -----
    kl_vlines = {}
    if has_kl_row:
        kl_palette = {
            'pfam':   '#000000',
            'sim':    '#1f77b4',
            'flow':   '#d62728',
            'closed': '#9467bd',
            'rtonly': '#8c564b',
            'plain':  '#7f7f7f',
        }
        kl_ylabel = (
            fr'$D_{{\mathrm{{KL}}}}(\mathrm{{{kl_source}}}\,\|\,\mathrm{{panel}})$  '
            + ('(joint, +overflow)' if args.kl_mode == 'joint'
               else '(conditional, renormalised)'))
        for name in active_panels:
            ax = ax_kl[name]
            if name == kl_source:
                # Pfam source gets a per-bin-counts histogram with a moving
                # vertical "current t" line.  Other sources just annotate.
                if name == 'pfam' and pfam_meta.get('n_cherries_per_bin') is not None:
                    nbin = pfam_meta['n_cherries_per_bin'].astype(np.float64)
                    tcen = pfam_meta['tau_centers']
                    tedges = pfam_meta.get('tau_edges')
                    # Linear widths from geomspace edges; bar area = count.
                    if tedges is not None:
                        widths = np.diff(tedges)
                        bar_left = tedges[:-1]
                    else:
                        widths = np.r_[np.diff(tcen), tcen[-1] - tcen[-2]]
                        bar_left = tcen - widths / 2
                    safe_widths = np.maximum(widths, 1e-12)
                    densities = nbin / safe_widths  # cherries per unit tau
                    # Use align='edge' so the bar spans [tedge[i], tedge[i+1]]
                    ax.bar(bar_left, densities, width=widths, align='edge',
                            color='lightsteelblue', edgecolor='steelblue',
                            linewidth=0.4)
                    ax.set_xlim(0, args.t_max)
                    ax.set_xlabel(r'$\tau$', fontsize=9, labelpad=1)
                    ax.set_ylabel('cherries / unit $\\tau$', fontsize=9)
                    ax.grid(axis='y', alpha=0.3, lw=0.5)
                    ax.tick_params(axis='both', labelsize=8)
                    # Vertical "current t" tracker on the linear scale, same as seek bar.
                    vline = ax.axvline(0, color='black', lw=1.2, alpha=0.8)
                    kl_vlines[name] = vline
                else:
                    ax.set_axis_off()
                    ax.text(0.5, 0.5,
                             f'(KL ground-truth source -- {name})',
                             transform=ax.transAxes,
                             ha='center', va='center', fontsize=9, color='gray')
                continue
            ax.plot(t_grid, kl_traces[name],
                     color=kl_palette.get(name, 'C0'), lw=1.5, alpha=0.9)
            ax.set_xlim(0, args.t_max)
            ax.set_ylim(kl_ymin, kl_ymax)
            ax.set_xlabel(r'$t$', fontsize=9, labelpad=1)
            ax.set_ylabel(kl_ylabel, fontsize=8)
            ax.grid(alpha=0.3, lw=0.5)
            ax.tick_params(axis='both', labelsize=8)
            vline = ax.axvline(0, color='black', lw=1.2, alpha=0.8)
            kl_vlines[name] = vline

    sup_title = fig.suptitle(
        fr'GGI sim ($\lambda_0={p_sim["lam0"]:.4g}$, $\mu_0={p_sim["mu0"]:.4g}$, '
        fr'$x={p_sim["x_del"]:.3g}$, $y={p_sim["y_ins"]:.3g}$) '
        fr' vs. four TKF92 approximations',
        fontsize=12)

    def update(fi):
        t = t_grid[fi]
        lam_c, mu_c, r_c = closed_params[fi]
        lam_f, mu_f, r_f = flow_traj[fi]
        # Heatmaps (order matches panels_spec: pfam? sim? flow closed rtonly plain)
        idx = 0
        if has_pfam:
            ims[idx].set_data(G_pfam_frames[fi]); idx += 1
        if show_sim:
            ims[idx].set_data(G_sim_frames[fi]); idx += 1
        ims[idx].set_data(G_flow_frames[fi]); idx += 1
        ims[idx].set_data(G_closed_frames[fi]); idx += 1
        ims[idx].set_data(G_rtonly_frames[fi]); idx += 1
        ims[idx].set_data(G_plain_frames[fi])
        # Seek bar
        seek_fill.set_width(t)
        seek_marker.set_xdata([t])
        # FLOW rate bars
        bar_lam_f.set_width(lam_f)
        bar_mu_f.set_width(mu_f)
        txt_lam_f.set_text(f'{lam_f:.4f}')
        txt_mu_f.set_text(f'{mu_f:.4f}')
        bar_r_f.set_width(r_f)
        txt_r_f.set_text(f'{r_f:.4f}')
        # CLOSED rate bars
        bar_lam_c.set_width(lam_c)
        bar_mu_c.set_width(mu_c)
        txt_lam_c.set_text(f'{lam_c:.4f}')
        txt_mu_c.set_text(f'{mu_c:.4f}')
        bar_r_c.set_width(r_c)
        txt_r_c.set_text(f'{r_c:.4f}')
        # RT_ONLY r bar uses ITS OWN GGI's r*(t), not the closed panel's.
        bar_r_rt.set_width(rtonly_r[fi])
        txt_r_rt.set_text(f'{rtonly_r[fi]:.4f}')
        # KL vertical lines (all on the same linear t axis as the seek bar)
        for vline in kl_vlines.values():
            vline.set_xdata([t, t])
        return (ims + [seek_fill, seek_marker,
                       bar_lam_f, bar_mu_f, txt_lam_f, txt_mu_f, bar_r_f, txt_r_f,
                       bar_lam_c, bar_mu_c, txt_lam_c, txt_mu_c, bar_r_c, txt_r_c,
                       bar_r_rt, txt_r_rt] + list(kl_vlines.values()))

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/args.fps,
                          blit=False)

    # ----- Save -----
    try:
        writer = FFMpegWriter(fps=args.fps, bitrate=2400)
        print(f"\nWriting {args.out}...")
        anim.save(args.out, writer=writer, dpi=120)
        print(f"Saved -> {args.out}")
    except Exception as ex:
        print(f"ffmpeg write failed ({ex}); falling back to PNG snapshots.")
        png_dir = args.png_dir or os.path.splitext(args.out)[0] + '_frames'
        os.makedirs(png_dir, exist_ok=True)
        stride = max(1, args.fps // 4)
        for fi in range(0, n_frames, stride):
            update(fi)
            png = os.path.join(png_dir, f'frame_{fi:04d}.png')
            fig.savefig(png, dpi=120, bbox_inches='tight')
        print(f"Saved {n_frames // stride + 1} PNGs in {png_dir}")


if __name__ == "__main__":
    main()
