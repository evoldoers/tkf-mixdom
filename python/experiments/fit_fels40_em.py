#!/usr/bin/env python3
"""Marginal CherryML fitter for the 40-state hidden gap model.

The 40-state model has 20 "present" states (emit their amino acid) and
20 "gapped" states (emit gap). Since the gapped states are hidden (we observe
gap but don't know which of the 20 gapped substates it's in), we can't
directly count transitions. Instead we use the marginal CherryML approach:

  Joint_obs[a, b | t] = sum_i sum_j E[i,a] * pi40[i] * expm(Q40*t)[i,j] * E[j,b]
                       = (E^T @ diag(pi40) @ expm(Q40*t) @ E)[a, b]

where E is the 40x21 emission matrix. We then optimize Q40 parameters via
gradient descent on the composite negative log-likelihood over time-binned
pairwise counts.

Parameterization (~41 free params):
  - r_del[i] for i=0..19: per-AA deletion rates (present_i -> gapped_i)
  - r_ins[i] for i=0..19: per-AA insertion rates (gapped_i -> present_i)
  - gap_subst_scale: scales LG08 for gapped-gapped substitution

Usage:
  cd /home/yam/tkf-mixdom/python
  CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 uv run python experiments/fit_fels40_em.py \
      --n-families 50 --n-iters 50
"""

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import jax.scipy.linalg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.protein_gap40 import build_Q40, build_emission_matrix
from tkfmixdom.jax.util.io import AA_TO_INT, AMINO_ACIDS

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from ancrec_benchmark import parse_sto

# Reuse data loading infrastructure from fit_fels21_cherryml
from fit_fels21_cherryml import (
    extract_family_counts_vectorized,
    tree_pairwise_distances,
    encode_seq,
)

PFAM_DIR = "/home/yam/bio-datasets/data/pfam-seed"
TREE_DIR = os.path.expanduser("~/bio-datasets/data/pfam-seed/trees")
SPLITS_PATH = os.path.join(PFAM_DIR, "splits", "v1.json")


# --- Q40 builder in JAX (differentiable) ---

def build_Q40_jax(log_r_del, log_r_ins, log_gap_subst_scale):
    """Build Q40 and pi40 from differentiable parameters.

    Args:
        log_r_del: (20,) log per-AA deletion rates
        log_r_ins: (20,) log per-AA insertion rates
        log_gap_subst_scale: scalar, log scale for gapped-gapped substitution

    Returns:
        Q40: (40, 40) rate matrix
        pi40: (40,) equilibrium distribution
    """
    r_del = jnp.exp(log_r_del)   # (20,)
    r_ins = jnp.exp(log_r_ins)   # (20,)
    gap_subst_scale = jnp.exp(log_gap_subst_scale)

    # LG08 rate matrix and equilibrium (fixed)
    Q_lg_np, pi_lg_np = rate_matrix_lg()
    Q_lg = jnp.array(Q_lg_np)
    pi_lg = jnp.array(pi_lg_np)

    Q40 = jnp.zeros((40, 40))

    # Top-left: present -> present (LG08 substitution)
    Q40 = Q40.at[:20, :20].set(Q_lg)

    # Top-right: present_i -> gapped_i (deletion), diagonal only
    Q40 = Q40.at[jnp.arange(20), jnp.arange(20) + 20].set(r_del)

    # Bottom-left: gapped_i -> present_i (insertion), diagonal only
    Q40 = Q40.at[jnp.arange(20) + 20, jnp.arange(20)].set(r_ins)

    # Bottom-right: gapped -> gapped (hidden substitution)
    Q40 = Q40.at[20:, 20:].set(Q_lg * gap_subst_scale)

    # Fix diagonals so rows sum to 0
    Q40 = Q40.at[jnp.diag_indices(40)].set(0.0)
    row_sums = Q40.sum(axis=1)
    Q40 = Q40.at[jnp.diag_indices(40)].set(-row_sums)

    # Equilibrium distribution
    # pi40[i] = pi_lg[i] * r_ins[i] / (r_ins[i] + r_del[i])   (present)
    # pi40[20+i] = pi_lg[i] * r_del[i] / (r_ins[i] + r_del[i]) (gapped)
    kappa = r_ins / (r_ins + r_del)  # (20,)
    pi40 = jnp.zeros(40)
    pi40 = pi40.at[:20].set(pi_lg * kappa)
    pi40 = pi40.at[20:].set(pi_lg * (1.0 - kappa))
    pi40 = pi40 / pi40.sum()

    # Normalize Q40 so mean rate = 1 over pi40
    mean_rate = -jnp.sum(pi40 * jnp.diag(Q40))
    Q40 = Q40 / jnp.maximum(mean_rate, 1e-30)

    return Q40, pi40


# --- Loss function ---

def make_loss_fn(stacked_counts, t_centers):
    """Create JIT-compiled loss function for marginal CherryML on the 40-state model.

    Args:
        stacked_counts: (n_bins, 21, 21) observed pairwise counts
        t_centers: (n_bins,) time bin centers

    Returns:
        neg_log_lik function: (log_r_del, log_r_ins, log_gap_subst_scale) -> scalar
    """
    E = jnp.array(build_emission_matrix())  # (40, 21)

    @jax.jit
    def neg_log_lik(params):
        log_r_del, log_r_ins, log_gap_subst_scale = params
        Q40, pi40 = build_Q40_jax(log_r_del, log_r_ins, log_gap_subst_scale)

        # For each time bin, compute observed joint and accumulate log-likelihood
        def per_bin(carry, inputs):
            t, N = inputs
            P = jax.scipy.linalg.expm(Q40 * t)  # (40, 40)
            # Joint_obs[a, b] = (E^T @ diag(pi40) @ P @ E)[a, b]
            joint = E.T @ jnp.diag(pi40) @ P @ E  # (21, 21)
            joint = jnp.maximum(joint, 1e-30)
            ll = jnp.sum(N * jnp.log(joint))
            return carry + ll, None

        total_ll, _ = jax.lax.scan(per_bin, 0.0, (t_centers, stacked_counts))
        return -total_ll

    return neg_log_lik


def main():
    parser = argparse.ArgumentParser(
        description="Marginal CherryML fitter for 40-state hidden gap model")
    parser.add_argument("--n-families", type=int, default=500,
                        help="Number of training families to use")
    parser.add_argument("--n-bins", type=int, default=50,
                        help="Number of geometric time bins")
    parser.add_argument("--t-min", type=float, default=0.01,
                        help="Minimum time bin edge")
    parser.add_argument("--t-max", type=float, default=5.0,
                        help="Maximum time bin edge")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--n-iters", type=int, default=200,
                        help="Number of optimization iterations")
    parser.add_argument("--max-pairs", type=int, default=5000,
                        help="Max pairs per family (subsample if more)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: pfam/fels40_em.npz)")
    args = parser.parse_args()

    # Output path
    if args.output is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "pfam")
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, "fels40_em.npz")

    # Load family list
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    train_fams = splits["train"][:args.n_families]
    print(f"Using {len(train_fams)} training families")

    # Set up time bins
    t_bin_edges = np.geomspace(args.t_min, args.t_max, args.n_bins + 1)
    t_centers = np.sqrt(t_bin_edges[:-1] * t_bin_edges[1:])  # geometric mean
    n_bins = args.n_bins

    # Extract counts from all families (reusing fels21 infrastructure)
    print("Extracting pairwise counts...")
    total_counts = np.zeros((n_bins, 21, 21), dtype=np.float64)
    total_char = np.zeros(21, dtype=np.float64)
    total_pairs = 0
    t0 = time.time()
    rng = np.random.default_rng(42)

    for fi, fam in enumerate(train_fams):
        counts, chars, n_pairs = extract_family_counts_vectorized(
            fam, t_bin_edges, max_pairs=args.max_pairs, rng=rng)
        total_counts += counts
        total_char += chars
        total_pairs += n_pairs
        if (fi + 1) % 50 == 0 or fi == len(train_fams) - 1:
            elapsed = time.time() - t0
            print(f"  {fi+1}/{len(train_fams)} families, "
                  f"{total_pairs} pairs, {elapsed:.1f}s")

    print(f"Total pairs: {total_pairs}")
    print(f"Total co-occurrence counts: {total_counts.sum():.0f}")

    # Report observed gap frequency
    gap_frac = total_char[20] / total_char.sum()
    print(f"Observed gap frequency: {gap_frac:.4f}")

    # --- Initialize parameters ---
    # Start with uniform del/ins rates
    init_r_del = 0.03
    init_r_ins = 0.03
    init_gap_subst_scale = 0.1

    log_r_del = jnp.full(20, jnp.log(init_r_del))
    log_r_ins = jnp.full(20, jnp.log(init_r_ins))
    log_gap_subst_scale = jnp.array(jnp.log(init_gap_subst_scale))

    params = (log_r_del, log_r_ins, log_gap_subst_scale)

    # Convert data to JAX
    stacked_counts = jnp.array(total_counts)
    t_centers_jax = jnp.array(t_centers)

    # Build loss function
    loss_fn = make_loss_fn(stacked_counts, t_centers_jax)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # Adam optimizer state
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    n_params = sum(p.size for p in jax.tree.leaves(params))
    print(f"\nOptimizing {n_params} parameters...")
    print(f"  lr={args.lr}, n_iters={args.n_iters}")

    # Optimization loop
    t_opt_start = time.time()
    best_loss = float('inf')
    best_params = params

    for it in range(args.n_iters):
        loss_val, grads = grad_fn(params)
        loss_val = float(loss_val)

        # Adam update
        m = jax.tree.map(lambda mi, gi: beta1 * mi + (1 - beta1) * gi, m, grads)
        v = jax.tree.map(lambda vi, gi: beta2 * vi + (1 - beta2) * gi ** 2, v, grads)
        m_hat = jax.tree.map(lambda mi: mi / (1 - beta1 ** (it + 1)), m)
        v_hat = jax.tree.map(lambda vi: vi / (1 - beta2 ** (it + 1)), v)
        params = jax.tree.map(
            lambda p, mh, vh: p - args.lr * mh / (jnp.sqrt(vh) + eps),
            params, m_hat, v_hat
        )

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = params

        if (it + 1) % 10 == 0 or it == 0:
            grad_norm = float(jnp.sqrt(sum(
                jnp.sum(g ** 2) for g in jax.tree.leaves(grads))))
            elapsed = time.time() - t_opt_start

            # Report current parameter summary
            r_del_cur = jnp.exp(params[0])
            r_ins_cur = jnp.exp(params[1])
            gs_cur = jnp.exp(params[2])
            gap_eq = float(jnp.mean(r_del_cur / (r_del_cur + r_ins_cur)))

            print(f"  iter {it+1:4d}: loss={loss_val:.2f}, "
                  f"|grad|={grad_norm:.4f}, "
                  f"r_del={float(jnp.mean(r_del_cur)):.4f}, "
                  f"r_ins={float(jnp.mean(r_ins_cur)):.4f}, "
                  f"gap_subst={float(gs_cur):.4f}, "
                  f"eq_gap={gap_eq:.4f}, "
                  f"time={elapsed:.1f}s")

    print(f"\nBest loss: {best_loss:.2f}")

    # Extract final parameters
    log_r_del_final, log_r_ins_final, log_gap_subst_scale_final = best_params
    r_del_final = np.array(jnp.exp(log_r_del_final))
    r_ins_final = np.array(jnp.exp(log_r_ins_final))
    gap_subst_scale_final = float(jnp.exp(log_gap_subst_scale_final))

    # Build final Q40 and pi40
    Q40_final, pi40_final = build_Q40_jax(
        log_r_del_final, log_r_ins_final, log_gap_subst_scale_final)
    Q40_final = np.array(Q40_final)
    pi40_final = np.array(pi40_final)

    # Save
    np.savez(args.output,
             Q40=Q40_final,
             pi40=pi40_final,
             r_del=r_del_final,
             r_ins=r_ins_final,
             gap_subst_scale=gap_subst_scale_final,
             training_loss=best_loss,
             n_families=len(train_fams),
             n_pairs=total_pairs,
             n_iters=args.n_iters,
             lr=args.lr)
    print(f"\nSaved to {args.output}")

    # --- Diagnostics ---
    print(f"\nFitted parameters:")
    print(f"  gap_subst_scale: {gap_subst_scale_final:.6f}")
    print(f"  mean r_del: {r_del_final.mean():.6f}")
    print(f"  mean r_ins: {r_ins_final.mean():.6f}")

    # Equilibrium gap fraction
    eq_gap = pi40_final[20:].sum()
    print(f"  equilibrium gap fraction: {eq_gap:.4f}")

    # Per-AA deletion and insertion rates
    labels = list(AMINO_ACIDS)
    print(f"\nPer-AA deletion rates (present -> gapped):")
    order = np.argsort(-r_del_final)
    for i in order[:10]:
        print(f"  {labels[i]}: r_del={r_del_final[i]:.5f}, "
              f"r_ins={r_ins_final[i]:.5f}, "
              f"eq_gap={r_del_final[i]/(r_del_final[i]+r_ins_final[i]):.4f}")

    # Q40 diagonal summary
    diag = np.diag(Q40_final)
    print(f"\nQ40 diagonal range: [{diag.min():.4f}, {diag.max():.4f}]")
    print(f"  present states: [{diag[:20].min():.4f}, {diag[:20].max():.4f}]")
    print(f"  gapped states:  [{diag[20:].min():.4f}, {diag[20:].max():.4f}]")

    # Verify detailed balance
    db_err = 0.0
    for i in range(40):
        for j in range(i + 1, 40):
            db_err += abs(pi40_final[i] * Q40_final[i, j] -
                         pi40_final[j] * Q40_final[j, i])
    print(f"\nDetailed balance error (sum |pi_i*Q_ij - pi_j*Q_ji|): {db_err:.2e}")


if __name__ == "__main__":
    main()
