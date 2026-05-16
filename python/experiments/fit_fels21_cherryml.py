#!/usr/bin/env python3
"""CherryML-style fitter for a 21-state GTR model (20 AA + gap).

Instead of Felsenstein inside-outside EM, this uses the CherryML approach:
  1. Extract pairwise co-occurrence counts from aligned MSA columns
  2. Get evolutionary distances from FastTree ML trees
  3. Pool counts into geometric time bins
  4. Maximize composite log-likelihood over symmetric exchangeabilities S

Usage:
  cd /home/yam/tkf-mixdom/python
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run python experiments/fit_fels21_cherryml.py \
      --n-families 500 --n-bins 50 --lr 0.01 --n-iters 200
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

from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT, AMINO_ACIDS

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from ancrec_benchmark import parse_sto, PFAM_DIR

TREE_DIR = os.path.expanduser("~/bio-datasets/data/pfam-seed/trees")
SPLITS_PATH = os.path.join(PFAM_DIR, "splits", "v1.json")


# --- Distance estimation from sequence identity ---

def kimura_distance(seq_a, seq_b):
    """Estimate evolutionary distance from two aligned integer sequences.

    Uses Kimura protein distance: d = -log(1 - p - 0.2*p^2) where p is the
    fraction of differing non-gap positions. Returns np.inf if correction fails.
    """
    # Only count positions where both are amino acids (0-19)
    mask = (seq_a < 20) & (seq_b < 20)
    n_sites = mask.sum()
    if n_sites < 5:
        return np.inf
    n_diff = ((seq_a[mask] != seq_b[mask])).sum()
    p = n_diff / n_sites
    # Kimura correction for protein
    arg = 1.0 - p - 0.2 * p * p
    if arg <= 0:
        return 5.0  # saturated: cap at max distance
    return -np.log(arg)


# --- LG08 exchangeabilities in alphabetical order for initialization ---

def _lg_exchangeabilities_alpha():
    """Return LG08 20x20 symmetric exchangeability matrix in alphabetical order."""
    from tkfmixdom.jax.core.protein import rate_matrix_lg, _LG_S_LOWER, _lower_tri_to_matrix, _get_perm
    S_paml = _lower_tri_to_matrix(_LG_S_LOWER, 20)
    perm = _get_perm()
    S_alpha = S_paml[perm][:, perm]
    return S_alpha


def _lg_pi_alpha():
    """Return LG08 equilibrium frequencies in alphabetical order."""
    from tkfmixdom.jax.core.protein import _LG_PI, _get_perm
    perm = _get_perm()
    pi = _LG_PI / _LG_PI.sum()
    return pi[perm]


# --- Tree pairwise distances ---

def tree_pairwise_distances(tree):
    """Compute pairwise distances between all leaves via the tree."""
    leaves = list(tree.leaves())
    leaf_names = [l.name for l in leaves]
    n = len(leaf_names)

    def path_to_root(node):
        path = []
        cur = node
        dist = 0.0
        while cur is not None:
            path.append((id(cur), dist))
            dist += cur.branch_length
            cur = cur.parent
        return path

    paths = {}
    for leaf in leaves:
        paths[leaf.name] = dict(path_to_root(leaf))

    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pi_ = paths[leaf_names[i]]
            pj = paths[leaf_names[j]]
            common = set(pi_.keys()) & set(pj.keys())
            min_dist = min(pi_[nid] + pj[nid] for nid in common)
            dist_mat[i, j] = min_dist
            dist_mat[j, i] = min_dist
    return leaf_names, dist_mat


# --- Encoding ---

def encode_seq(seq_str):
    """Encode a sequence string to integer array (0-19 = AA, 20 = gap)."""
    result = []
    for c in seq_str.upper():
        if c in ('-', '.'):
            result.append(20)
        elif c in AA_TO_INT:
            result.append(AA_TO_INT[c])
        else:
            result.append(20)  # unknown -> gap
    return np.array(result, dtype=np.int32)


# --- Data loading and count extraction ---

def extract_family_counts(fam, t_bins):
    """Extract pairwise co-occurrence counts binned by evolutionary distance.

    Returns:
        binned_counts: (n_bins, 21, 21) count arrays
        char_counts: (21,) pooled character frequencies
        n_pairs: total number of pairs processed
    """
    n_bins = len(t_bins) - 1
    binned_counts = np.zeros((n_bins, 21, 21), dtype=np.float64)
    char_counts = np.zeros(21, dtype=np.float64)

    # Load MSA
    sto_path = os.path.join(PFAM_DIR, f"{fam}.sto")
    if not os.path.exists(sto_path):
        return binned_counts, char_counts, 0

    msa = parse_sto(sto_path)
    if len(msa) < 2:
        return binned_counts, char_counts, 0

    # Encode sequences
    names = list(msa.keys())
    encoded = {name: encode_seq(msa[name]) for name in names}
    L = len(next(iter(encoded.values())))

    # Accumulate character frequencies
    for name in names:
        for c in encoded[name]:
            char_counts[c] += 1

    # Load tree and get pairwise distances
    tree_path = os.path.join(TREE_DIR, f"{fam}.nwk")
    if not os.path.exists(tree_path):
        return binned_counts, char_counts, 0

    tree = parse_newick(open(tree_path).read())
    leaf_names, dist_mat = tree_pairwise_distances(tree)

    # Map tree leaf names to MSA names
    leaf_to_idx = {name: i for i, name in enumerate(leaf_names)}

    # Process all pairs
    n_pairs = 0
    msa_names = [n for n in names if n in leaf_to_idx]
    for ii in range(len(msa_names)):
        for jj in range(ii + 1, len(msa_names)):
            na, nb = msa_names[ii], msa_names[jj]
            ti = leaf_to_idx[na]
            tj = leaf_to_idx[nb]
            t = dist_mat[ti, tj]
            if t <= 0:
                continue

            # Find bin
            bin_idx = np.searchsorted(t_bins, t, side='right') - 1
            bin_idx = max(0, min(bin_idx, n_bins - 1))

            # Count co-occurrences
            seq_a = encoded[na]
            seq_b = encoded[nb]
            for k in range(L):
                binned_counts[bin_idx, seq_a[k], seq_b[k]] += 1

            n_pairs += 1

    return binned_counts, char_counts, n_pairs


def extract_family_counts_vectorized(fam, t_bins, max_pairs=5000, rng=None):
    """Vectorized count extraction. Uses tree distances if available, else Kimura.

    For families with many sequences, randomly subsample pairs to cap at max_pairs.
    """
    n_bins = len(t_bins) - 1
    binned_counts = np.zeros((n_bins, 21, 21), dtype=np.float64)
    char_counts = np.zeros(21, dtype=np.float64)

    sto_path = os.path.join(PFAM_DIR, f"{fam}.sto")
    if not os.path.exists(sto_path):
        return binned_counts, char_counts, 0

    msa = parse_sto(sto_path)
    if len(msa) < 2:
        return binned_counts, char_counts, 0

    names = list(msa.keys())
    encoded = {name: encode_seq(msa[name]) for name in names}
    L = len(next(iter(encoded.values())))

    # Character frequencies
    all_seqs = np.stack([encoded[n] for n in names])  # (N, L)
    for i in range(21):
        char_counts[i] = np.sum(all_seqs == i)

    # Try to load tree for distances; fall back to Kimura
    tree_path = os.path.join(TREE_DIR, f"{fam}.nwk")
    use_tree = os.path.exists(tree_path)

    if use_tree:
        tree = parse_newick(open(tree_path).read())
        leaf_names, dist_mat = tree_pairwise_distances(tree)
        leaf_to_idx = {name: i for i, name in enumerate(leaf_names)}
        msa_names = [n for n in names if n in leaf_to_idx]
        if len(msa_names) < 2:
            use_tree = False

    if not use_tree:
        # Use all MSA sequences with Kimura distances
        msa_names = names

    n_msa = len(msa_names)
    if n_msa < 2:
        return binned_counts, char_counts, 0

    msa_indices = [names.index(n) for n in msa_names]

    # Build list of all pairs; subsample if too many
    all_pairs = []
    for ii in range(n_msa):
        for jj in range(ii + 1, n_msa):
            all_pairs.append((ii, jj))

    if len(all_pairs) > max_pairs:
        if rng is None:
            rng = np.random.default_rng(42)
        indices = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        all_pairs = [all_pairs[i] for i in indices]

    n_pairs = 0
    for ii, jj in all_pairs:
        if use_tree:
            t = dist_mat[leaf_to_idx[msa_names[ii]], leaf_to_idx[msa_names[jj]]]
        else:
            t = kimura_distance(all_seqs[msa_indices[ii]],
                                all_seqs[msa_indices[jj]])

        if t <= 0 or not np.isfinite(t):
            continue

        bin_idx = int(np.searchsorted(t_bins, t, side='right')) - 1
        bin_idx = max(0, min(bin_idx, n_bins - 1))

        seq_a = all_seqs[msa_indices[ii]]  # (L,)
        seq_b = all_seqs[msa_indices[jj]]  # (L,)

        # Vectorized count: use 2D histogram trick
        pair_idx = seq_a * 21 + seq_b  # (L,)
        counts_flat = np.bincount(pair_idx, minlength=21 * 21)
        binned_counts[bin_idx] += counts_flat.reshape(21, 21)
        n_pairs += 1

    return binned_counts, char_counts, n_pairs


# --- JAX optimization ---

def build_symmetric_21(upper_tri_vals):
    """Build 21x21 symmetric matrix from upper triangle values."""
    n = 21
    S = jnp.zeros((n, n))
    idx = 0
    rows, cols = jnp.triu_indices(n, k=1)
    S = S.at[rows, cols].set(upper_tri_vals)
    S = S + S.T
    return S


def upper_tri_indices_21():
    """Return indices for upper triangle of 21x21 matrix."""
    rows, cols = np.triu_indices(21, k=1)
    return rows, cols


def init_log_S_from_lg(lg_S_alpha):
    """Initialize log exchangeabilities from LG08.

    For AA-AA pairs: use LG08 values.
    For AA-gap and gap-gap pairs: initialize uniformly at ~0.05.
    """
    n = 21
    rows, cols = np.triu_indices(n, k=1)
    n_params = len(rows)  # 210
    log_S = np.zeros(n_params, dtype=np.float64)

    for idx in range(n_params):
        i, j = rows[idx], cols[idx]
        if i < 20 and j < 20:
            # AA-AA: use LG08
            val = lg_S_alpha[i, j]
            log_S[idx] = np.log(max(val, 1e-6))
        else:
            # Involves gap state (index 20)
            log_S[idx] = np.log(0.05)

    return log_S


def make_loss_fn(stacked_counts, t_centers, pi21):
    """Create JIT-compiled loss function."""

    @jax.jit
    def neg_log_lik(log_S_vec):
        S_vals = jnp.exp(log_S_vec)
        rows, cols = jnp.triu_indices(21, k=1)
        S = jnp.zeros((21, 21))
        S = S.at[rows, cols].set(S_vals)
        S = S + S.T

        # Build Q from S and pi
        Q = S * pi21[None, :]
        Q = Q.at[jnp.diag_indices(21)].set(0.0)
        row_sums = Q.sum(axis=1)
        Q = Q.at[jnp.diag_indices(21)].set(-row_sums)

        # Normalize to mean rate 1
        mean_rate = -jnp.sum(pi21 * jnp.diag(Q))
        Q = Q / jnp.maximum(mean_rate, 1e-30)

        # Composite log-likelihood over time bins
        def per_bin(t, N):
            P = jax.scipy.linalg.expm(Q * t)
            P = jnp.maximum(P, 1e-30)
            return jnp.sum(N * jnp.log(P))

        total = jax.vmap(per_bin)(t_centers, stacked_counts).sum()
        return -total

    return neg_log_lik


def main():
    parser = argparse.ArgumentParser(description="CherryML-style 21-state GTR fitter")
    parser.add_argument("--n-families", type=int, default=500,
                        help="Number of training families to use")
    parser.add_argument("--n-bins", type=int, default=50,
                        help="Number of geometric time bins")
    parser.add_argument("--t-min", type=float, default=0.01,
                        help="Minimum time bin edge")
    parser.add_argument("--t-max", type=float, default=5.0,
                        help="Maximum time bin edge")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--n-iters", type=int, default=200,
                        help="Number of optimization iterations")
    parser.add_argument("--max-pairs", type=int, default=5000,
                        help="Max pairs per family (subsample if more)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: pfam/fels21_cherryml.npz)")
    args = parser.parse_args()

    # Output path
    if args.output is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "pfam")
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, "fels21_cherryml.npz")

    # Load family list
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    train_fams = splits["train"][:args.n_families]
    print(f"Using {len(train_fams)} training families")

    # Set up time bins
    t_bin_edges = np.geomspace(args.t_min, args.t_max, args.n_bins + 1)
    t_centers = np.sqrt(t_bin_edges[:-1] * t_bin_edges[1:])  # geometric mean
    n_bins = args.n_bins

    # Extract counts from all families
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

    # Compute pi from character frequencies
    pi21 = total_char / total_char.sum()
    print(f"Equilibrium frequencies (gap={pi21[20]:.4f}):")
    for i, aa in enumerate(AMINO_ACIDS):
        print(f"  {aa}: {pi21[i]:.4f}", end="")
        if (i + 1) % 5 == 0:
            print()
    print(f"  -: {pi21[20]:.4f}")

    # Initialize from LG08
    lg_S = _lg_exchangeabilities_alpha()
    log_S_init = init_log_S_from_lg(lg_S)

    # Convert to JAX arrays
    stacked_counts = jnp.array(total_counts)
    t_centers_jax = jnp.array(t_centers)
    pi21_jax = jnp.array(pi21)

    # Build loss function
    loss_fn = make_loss_fn(stacked_counts, t_centers_jax, pi21_jax)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # Initialize optimizer (Adam)
    log_S = jnp.array(log_S_init)

    # Adam state
    m = jnp.zeros_like(log_S)
    v = jnp.zeros_like(log_S)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    print(f"\nOptimizing {len(log_S)} exchangeability parameters...")
    print(f"  lr={args.lr}, n_iters={args.n_iters}")

    # Initial loss
    t_opt_start = time.time()
    best_loss = float('inf')
    best_log_S = log_S

    for it in range(args.n_iters):
        loss_val, grad_val = grad_fn(log_S)
        loss_val = float(loss_val)

        # Adam update
        m = beta1 * m + (1 - beta1) * grad_val
        v = beta2 * v + (1 - beta2) * grad_val ** 2
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        log_S = log_S - args.lr * m_hat / (jnp.sqrt(v_hat) + eps)

        if loss_val < best_loss:
            best_loss = loss_val
            best_log_S = log_S

        if (it + 1) % 10 == 0 or it == 0:
            grad_norm = float(jnp.linalg.norm(grad_val))
            elapsed = time.time() - t_opt_start
            print(f"  iter {it+1:4d}: loss={loss_val:.2f}, "
                  f"|grad|={grad_norm:.4f}, time={elapsed:.1f}s")

    print(f"\nBest loss: {best_loss:.2f}")

    # Extract final Q matrix
    log_S_final = best_log_S
    S_vals = np.exp(np.array(log_S_final))
    rows, cols = np.triu_indices(21, k=1)
    S21 = np.zeros((21, 21))
    S21[rows, cols] = S_vals
    S21 = S21 + S21.T

    pi21_np = np.array(pi21)
    Q21 = S21 * pi21_np[None, :]
    np.fill_diagonal(Q21, 0.0)
    np.fill_diagonal(Q21, -Q21.sum(axis=1))
    mean_rate = -np.sum(pi21_np * np.diag(Q21))
    Q21 = Q21 / mean_rate

    # Save
    np.savez(args.output,
             Q21=Q21,
             pi21=pi21_np,
             S21=S21,
             training_loss=best_loss,
             n_families=len(train_fams),
             n_pairs=total_pairs,
             n_iters=args.n_iters,
             lr=args.lr)
    print(f"\nSaved to {args.output}")

    # Print some diagnostics
    print(f"\nQ21 diagonal (top 5 most negative):")
    diag = np.diag(Q21)
    order = np.argsort(diag)
    labels = list(AMINO_ACIDS) + ['-']
    for i in order[:5]:
        print(f"  {labels[i]}: {diag[i]:.4f}")

    print(f"\nLargest off-diagonal rates:")
    Q_off = Q21.copy()
    np.fill_diagonal(Q_off, 0)
    flat = Q_off.flatten()
    top_idx = np.argsort(flat)[-5:][::-1]
    for idx in top_idx:
        i, j = divmod(idx, 21)
        print(f"  {labels[i]}->{labels[j]}: {Q21[i,j]:.4f}")

    print(f"\nGap equilibrium frequency: {pi21_np[20]:.4f}")
    print(f"AA-gap mean exchangeability: {S21[:20, 20].mean():.4f}")


if __name__ == "__main__":
    main()
