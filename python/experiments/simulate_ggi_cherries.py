#!/usr/bin/env python3
"""Gillespie simulator for GGI cherries -> counts tensor in fit_ggi_cherryml format.

Per cherry:
  1. Draw an ancestor length L from a fixed distribution (geometric by default).
  2. Run GGI evolution for branch length tau:
       insertions: rate lam0 per link (L+1 links total); length k ~ Geom(1 - y_ins),
                   so P(k) = (1-y_ins) y_ins^(k-1), k = 1, 2, ...
       deletions:  rate mu0 per residue (L total); length k ~ Geom(1 - x_del),
                   truncated to remaining sequence.
  3. Record the (ancestor, descendant) column alignment in {M, I, D}.
       M = ancestor residue survived
       D = ancestor residue deleted
       I = inserted residue still present in descendant at tau
       (inserted residue then deleted before tau: not in alignment).
  4. Prepend S, append E; tally 5x5 transition counts.

Saves an .npz with `transition_counts` (n_tau, 5, 5) and `tau_centers` (n_tau,) so
that fit_ggi_cherryml.py / fit_tkf92_cherryml.py can consume it directly.

Convention (matches fit_ggi_cherryml.py):
  x_del = deletion length geometric parameter (mean del length = 1/(1-x_del))
  y_ins = insertion length geometric parameter (mean ins length = 1/(1-y_ins))

Usage:
    cd python && uv run python experiments/simulate_ggi_cherries.py \
        --lam0 0.04 --mu0 0.04 --x-del 0.6 --y-ins 0.4 \
        --n-cherries 5000 --L-mean 50 \
        --tau-centers 0.05,0.1,0.2,0.5,1.0,2.0 \
        --out /tmp/ggi_sim_counts.npz
"""

import argparse
import time
import numpy as np


# State indices (must match fit_ggi_cherryml.py / fit_tkf92_cherryml.py)
S, M, I, D, E = 0, 1, 2, 3, 4


def simulate_one_pair(L0, lam0, mu0, x_del, y_ins, t, rng):
    """Simulate GGI evolution from an ancestor of length L0 for time t.

    Returns the column list of the (ancestor, descendant) alignment, with each
    column in {M=1, I=2, D=3}.  S and E are added by the caller.

    Representation: a list of column types ('M' or 'I') plus a parallel list of
    `alive` flags. 'M' with alive=False is a D in the alignment. 'I' with
    alive=False is dropped entirely (ghost insertion: not in either sequence).
    """
    cols = ['M'] * L0
    alive = [True] * L0
    L = L0  # current live sequence length

    t_now = 0.0
    while True:
        ins_rate = lam0 * (L + 1)
        del_rate = mu0 * L
        R = ins_rate + del_rate
        if R <= 0:
            break
        dt = rng.exponential(1.0 / R)
        t_now += dt
        if t_now > t:
            break

        if rng.random() * R < ins_rate:
            # Insertion: link p in [0, L], length k ~ Geom(1 - y_ins)
            p = int(rng.integers(0, L + 1))
            k = int(rng.geometric(1.0 - y_ins))
            # Find the column position for insertion: after the p-th live token
            if p == 0:
                ins_col_pos = 0
            else:
                live_count = 0
                ins_col_pos = len(cols)  # fallback (append)
                for ci, a in enumerate(alive):
                    if a:
                        live_count += 1
                        if live_count == p:
                            ins_col_pos = ci + 1
                            break
            # Splice in k 'I' columns
            cols[ins_col_pos:ins_col_pos] = ['I'] * k
            alive[ins_col_pos:ins_col_pos] = [True] * k
            L += k
        else:
            # Deletion: start live-pos p in [0, L-1], length k ~ Geom(1 - x_del),
            # truncated to live tokens remaining.
            p = int(rng.integers(0, L))
            k = int(rng.geometric(1.0 - x_del))
            k = min(k, L - p)
            # Walk through alive tokens, mark k starting at the p-th as dead.
            # cols[ci] stays as 'M' or 'I' (origin tag); `alive=False` distinguishes
            # M-dead (-> D in alignment) from I-dead (-> dropped entirely).
            live_count = 0
            for ci in range(len(alive)):
                if alive[ci]:
                    if p <= live_count < p + k:
                        alive[ci] = False
                    live_count += 1
                    if live_count >= p + k:
                        break
            L -= k

    # Emit final alignment: drop dead I columns; M-dead -> D.
    out = []
    for c, a in zip(cols, alive):
        if c == 'M':
            out.append(M if a else D)
        else:  # c == 'I'
            if a:
                out.append(I)
            # else: skip (deleted insertion)
    return out


def tally_transitions(col_seq):
    """Return a 5x5 count matrix for one cherry alignment (list of {M,I,D})."""
    N = np.zeros((5, 5), dtype=np.float64)
    prev = S
    for c in col_seq:
        N[prev, c] += 1
        prev = c
    N[prev, E] += 1
    return N


# 6-state recoding for the TKF92 WFST conditional ML fit.
#   S=0, M=1, I0=2 (insertion before any ancestor residue), I1=3 (insertion
#   after at least one ancestor residue), D=4, E=5.
S6, M6, I0_6, I1_6, D6, E6 = 0, 1, 2, 3, 4, 5


def tally_transitions6(col_seq):
    """Return a 6x6 count matrix on {S, M, I0, I1, D, E} for one cherry.

    I columns occurring before the first M or D in the alignment become I0;
    I columns from the first M/D onwards become I1.  This matches the WFST
    state split in tkf/tkf92-wfst-derivation.tex (eq:tkf92-wfst).
    """
    first_md = None
    for i, c in enumerate(col_seq):
        if c == M or c == D:
            first_md = i
            break
    N = np.zeros((6, 6), dtype=np.float64)
    prev = S6
    for i, c in enumerate(col_seq):
        if c == M:
            cur = M6
        elif c == D:
            cur = D6
        else:  # I
            cur = I0_6 if (first_md is None or i < first_md) else I1_6
        N[prev, cur] += 1
        prev = cur
    N[prev, E6] += 1
    return N


def simulate_cherries(lam0, mu0, x_del, y_ins, tau_centers, n_cherries_per_tau,
                      L_mean, seed=0, log_every=None, with_6state=False):
    """Simulate cherries at each tau and assemble counts tensors.

    Ancestors are sampled with length L ~ Geom(1 - r_anc) where r_anc is chosen
    so that the mean length is L_mean (r_anc = 1 - 1/L_mean).  Lengths are
    bounded below by 1.

    n_cherries_per_tau may be either a scalar (same count at every tau) or a
    sequence of length len(tau_centers) (adaptive schedule).

    If with_6state=True, returns (counts5, counts6) for the joint Pair HMM and
    the conditional 6-state WFST recoding respectively.  Otherwise returns
    counts5.  The per-tau cherry counts (n_arr) are known to the caller (it
    passed them in), so we don't return them; for diagnostics we print them.
    """
    rng = np.random.default_rng(seed)
    r_anc = max(0.0, 1.0 - 1.0 / L_mean)

    n_tau = len(tau_centers)
    if np.isscalar(n_cherries_per_tau):
        n_arr = np.full(n_tau, int(n_cherries_per_tau), dtype=np.int64)
    else:
        n_arr = np.asarray(n_cherries_per_tau, dtype=np.int64)
        assert n_arr.shape == (n_tau,), \
            f"n_cherries_per_tau shape {n_arr.shape} != ({n_tau},)"

    counts = np.zeros((n_tau, 5, 5), dtype=np.float64)
    counts6 = np.zeros((n_tau, 6, 6), dtype=np.float64) if with_6state else None

    t_start = time.monotonic()
    for ti, tau in enumerate(tau_centers):
        n_this = int(n_arr[ti])
        bin_start = time.monotonic()
        for ci in range(n_this):
            L0 = int(rng.geometric(1.0 - r_anc))  # >= 1
            seq = simulate_one_pair(L0, lam0, mu0, x_del, y_ins, tau, rng)
            counts[ti] += tally_transitions(seq)
            if with_6state:
                counts6[ti] += tally_transitions6(seq)
            if log_every and ci % log_every == 0 and ci > 0:
                elapsed = time.monotonic() - bin_start
                print(f"  tau={tau:.4f}, cherry {ci}/{n_this} "
                      f"({elapsed:.1f}s, {ci/elapsed:.1f}/s)")
        bin_elapsed = time.monotonic() - bin_start
        total_n = counts[ti].sum()
        print(f"  tau={tau:.4f}: {n_this} cherries, "
              f"{int(total_n)} transitions  ({bin_elapsed:.1f}s)")
    total_elapsed = time.monotonic() - t_start
    print(f"\nTotal simulation time: {total_elapsed:.1f}s")
    if with_6state:
        return counts, counts6
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam0', type=float, default=0.04,
                        help='GGI insertion rate per link')
    parser.add_argument('--mu0', type=float, default=0.04,
                        help='GGI deletion rate per residue')
    parser.add_argument('--x-del', type=float, default=0.6,
                        help='GGI deletion length geom (mean=1/(1-x))')
    parser.add_argument('--y-ins', type=float, default=0.4,
                        help='GGI insertion length geom (mean=1/(1-y))')
    parser.add_argument('--tau-centers', type=str, default='0.05,0.1,0.2,0.5,1.0,2.0,5.0',
                        help='Comma-separated tau bin centers (branch lengths)')
    parser.add_argument('--n-cherries', type=int, default=2000,
                        help='Cherries per tau bin')
    parser.add_argument('--L-mean', type=float, default=50.0,
                        help='Mean ancestor length (geometric)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='/tmp/ggi_sim_counts.npz')
    parser.add_argument('--log-every', type=int, default=0,
                        help='Log progress every N cherries (0 = off)')
    args = parser.parse_args()

    tau_centers = np.array([float(s) for s in args.tau_centers.split(',')],
                           dtype=np.float64)

    # Reversibility check
    rev_lhs = args.lam0 * args.y_ins * (1 - args.y_ins)
    rev_rhs = args.mu0 * args.x_del * (1 - args.x_del)
    rev_err = abs(rev_lhs - rev_rhs) / max(rev_lhs, rev_rhs, 1e-30)

    # GGI stationary parameter
    rho_ggi = args.lam0 * (1 - args.x_del) / max(args.mu0 * (1 - args.y_ins), 1e-30)

    # Closed-form boundary r*(0)
    num = (args.lam0 * args.y_ins * (1 - args.x_del)
           + args.mu0 * args.x_del * (1 - args.y_ins))
    den = args.lam0 * (1 - args.y_ins) + args.mu0 * (1 - args.x_del)
    r_boundary = num / max(den, 1e-30)
    r_inf = r_boundary / (2 - r_boundary)
    k_decay = (args.lam0 + args.mu0) * (2 - r_boundary) / max(1 - r_boundary, 1e-30)

    print(f"GGI simulation parameters:")
    print(f"  lambda_0 = {args.lam0}, mu_0 = {args.mu0}")
    print(f"  x_del    = {args.x_del}, y_ins = {args.y_ins}")
    print(f"  Reversibility: lam0 y(1-y) = {rev_lhs:.6e}, mu0 x(1-x) = {rev_rhs:.6e}")
    print(f"  (relative error: {rev_err:.2e}{'  -- REVERSIBLE' if rev_err < 1e-3 else ''})")
    print(f"  GGI stationary rho = lam0(1-x)/[mu0(1-y)] = {rho_ggi:.4f}")
    print(f"    GGI mean length (if stable)            = {rho_ggi/(1-rho_ggi):.2f}"
          f"  (using L_mean={args.L_mean} for sampling)")
    print(f"  Closed-form TKF92 boundary:")
    print(f"    r*(0)   = {r_boundary:.4f}")
    print(f"    r_inf   = {r_inf:.4f}")
    print(f"    k decay = {k_decay:.4f}  (half-life {np.log(2)/max(k_decay,1e-30):.2f})")
    print(f"  Tau bin centers: {tau_centers}")
    print(f"  Cherries per bin: {args.n_cherries}, total: "
          f"{args.n_cherries * len(tau_centers)}")
    print()

    counts = simulate_cherries(
        args.lam0, args.mu0, args.x_del, args.y_ins,
        tau_centers, args.n_cherries, args.L_mean,
        seed=args.seed,
        log_every=args.log_every if args.log_every > 0 else None,
    )

    # Save in both formats: direct (for fit_ggi_cherryml) and C_xx (for fit_tkf92_cherryml).
    # C_xx style keys are 1D over tau bins.
    save_dict = dict(
        transition_counts=counts,
        tau_centers=tau_centers,
        # Metadata
        sim_lam0=args.lam0, sim_mu0=args.mu0,
        sim_x_del=args.x_del, sim_y_ins=args.y_ins,
        sim_L_mean=args.L_mean, sim_n_cherries=args.n_cherries,
        sim_seed=args.seed,
        sim_r_boundary=r_boundary, sim_r_inf=r_inf, sim_k_decay=k_decay,
    )
    _name_to_ij = {
        'C_SM': (S, M), 'C_SI': (S, I), 'C_SD': (S, D), 'C_SE': (S, E),
        'C_MM': (M, M), 'C_MI': (M, I), 'C_MD': (M, D), 'C_ME': (M, E),
        'C_IM': (I, M), 'C_II': (I, I), 'C_ID': (I, D), 'C_IE': (I, E),
        'C_DM': (D, M), 'C_DI': (D, I), 'C_DD': (D, D), 'C_DE': (D, E),
    }
    for name, (i, j) in _name_to_ij.items():
        save_dict[name] = counts[:, i, j].astype(np.float64)
    np.savez(args.out, **save_dict)
    print(f"\nSaved -> {args.out}")
    print(f"Total counts in tensor: {int(counts.sum())}")

    # Show row sums for each tau, to sanity-check
    print(f"\nPer-tau row sums (S, M, I, D, E):")
    print(f"  {'tau':>7}  {'S':>8}  {'M':>8}  {'I':>8}  {'D':>8}  {'E':>8}  {'total':>10}")
    for ti, tau in enumerate(tau_centers):
        row_sums = counts[ti].sum(axis=1)
        total = counts[ti].sum()
        print(f"  {tau:>7.3f}  " + "  ".join(f"{v:>8.0f}" for v in row_sums)
              + f"  {total:>10.0f}")


if __name__ == "__main__":
    main()
