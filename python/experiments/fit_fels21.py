#!/usr/bin/env python3
"""Fit a full GTR 21-state (20 AA + gap) substitution model to Pfam.

CherryML-style approach:
  1. Compute empirical pi21 from all MSA columns (gaps = character 20)
  2. Initialize Q21 from LG08 + uniform gap rates
  3. Iterate EM:
     a. E-step: for each cherry pair (pair of leaves sharing a parent)
        at distance t, compute Holmes-Rubin expected substitution counts
        E[N_{ij}] and dwell times E[T_i] using current Q21
     b. M-step: Q21[i,j] = sum E[N_{ij}] / sum E[T_i],
        then symmetrize exchangeabilities for reversibility

Uses NJ trees built from LG08 distances on the MSA.

Usage:
    cd python
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run python experiments/fit_fels21.py \\
        --n-families 100 --em-iters 5
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tkfmixdom.jax.core.protein import rate_matrix_lg, lg_exchangeability, _lower_tri_to_matrix
from tkfmixdom.jax.core.ctmc import (
    transition_matrix,
    holmes_rubin_integrals,
)
from tkfmixdom.jax.tree.guide_tree import neighbor_joining
from tkfmixdom.jax.util.io import AA_TO_INT, AMINO_ACIDS, parse_newick

# ── Paths ──────────────────────────────────────────────────────────────
PFAM_DIR = "/home/yam/bio-datasets/data/pfam-seed"
SPLITS_PATH = os.path.join(PFAM_DIR, "splits", "v1.json")

A21 = 21  # 20 amino acids + gap


# ── Stockholm parser ───────────────────────────────────────────────────

def parse_sto(path):
    """Parse Stockholm format MSA. Returns dict {name: aligned_seq_string}."""
    seqs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) == 2:
                name, seq = parts
                seqs.setdefault(name, "")
                seqs[name] += seq
    return seqs


def msa_to_int21(msa_strings):
    """Convert MSA strings to integer arrays with gap = 20.

    Returns dict {name: np.array of int32}, alignment length L.
    """
    result = {}
    for name, seq in msa_strings.items():
        arr = np.full(len(seq), 20, dtype=np.int32)  # default = gap
        for i, ch in enumerate(seq):
            if ch in "-.~":
                arr[i] = 20
            else:
                idx = AA_TO_INT.get(ch.upper(), -1)
                if 0 <= idx < 20:
                    arr[i] = idx
                else:
                    arr[i] = 20  # unknown -> gap
        result[name] = arr
    L = len(next(iter(result.values()))) if result else 0
    return result, L


# ── Empirical frequencies ──────────────────────────────────────────────

def compute_pi21(families_int):
    """Compute empirical 21-state frequencies from all MSA columns."""
    counts = np.zeros(A21, dtype=np.float64)
    for msa_int in families_int:
        for name, seq in msa_int.items():
            for c in seq:
                if 0 <= c <= 20:
                    counts[c] += 1
    # Add pseudocount
    counts += 1.0
    pi = counts / counts.sum()
    return pi


# ── NJ tree from pairwise distances ───────────────────────────────────

def build_nj_tree(msa_int, Q, pi):
    """Build NJ tree from MSA using LG08 distances (20-state, ignoring gaps)."""
    names = list(msa_int.keys())
    n = len(names)
    if n < 3:
        return None

    # Compute pairwise identity -> distance
    # Use a grid of branch lengths and find the best match
    t_values = np.concatenate([
        np.linspace(0.001, 0.1, 30),
        np.linspace(0.1, 1.0, 30),
        np.linspace(1.0, 5.0, 20),
    ])
    Q20 = np.asarray(Q)[:20, :20]
    pi20 = np.asarray(pi)[:20]
    pi20 = pi20 / pi20.sum()

    # Precompute expected identity at each t
    Q_lg, pi_lg = rate_matrix_lg()
    expected_ids = np.zeros(len(t_values))
    for ti, t in enumerate(t_values):
        P_t = np.array(transition_matrix(Q_lg, t))
        expected_ids[ti] = np.sum(np.array(pi_lg) * np.diag(P_t))

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            si = msa_int[names[i]]
            sj = msa_int[names[j]]
            matches = aligned = 0
            for ci, cj in zip(si, sj):
                if ci < 20 and cj < 20:
                    aligned += 1
                    if ci == cj:
                        matches += 1
            if aligned > 0:
                obs_id = matches / aligned
                # Find closest t
                idx = np.argmin(np.abs(expected_ids - obs_id))
                D[i, j] = D[j, i] = t_values[idx]
            else:
                D[i, j] = D[j, i] = 2.0

    tree = neighbor_joining(D, names)
    return tree


# ── Cherry extraction ──────────────────────────────────────────────────

def find_cherries(tree):
    """Find all cherry pairs: pairs of leaves sharing a direct parent.

    Returns list of (leaf1_name, leaf2_name, t1, t2) where t1, t2
    are the branch lengths from parent to each leaf.
    """
    cherries = []
    for node in tree.preorder():
        if node.is_leaf:
            continue
        leaf_children = [c for c in node.children if c.is_leaf]
        if len(leaf_children) >= 2:
            for i in range(len(leaf_children)):
                for j in range(i + 1, len(leaf_children)):
                    l1 = leaf_children[i]
                    l2 = leaf_children[j]
                    cherries.append((
                        l1.name, l2.name,
                        max(l1.branch_length, 1e-6),
                        max(l2.branch_length, 1e-6),
                    ))
    return cherries


# ── Holmes-Rubin E-step for a cherry pair ──────────────────────────────

def accumulate_cherry_counts(msa_int, cherries, Q21, pi21):
    """Accumulate expected substitution counts from cherry pairs.

    For each cherry (leaf1, leaf2) at distances (t1, t2) from their
    parent, and each MSA column, we compute:
      - Holmes-Rubin expected counts for the branch parent->leaf1 (time t1)
      - Holmes-Rubin expected counts for the branch parent->leaf2 (time t2)

    To avoid computing the parent state, we use the composite path
    leaf1 -> parent -> leaf2 at total time t1 + t2, and accumulate
    the counts for the full path. The M-step then uses these counts.

    Returns:
        u_total: (21, 21) expected transition counts
        w_total: (21,) expected dwell times
    """
    n = A21
    u_total = np.zeros((n, n), dtype=np.float64)
    w_total = np.zeros(n, dtype=np.float64)

    # Group cherries by total distance (for HR integral caching)
    from collections import defaultdict
    t_groups = defaultdict(list)
    for c in cherries:
        t_total = c[2] + c[3]
        # Round to 3 decimal places for grouping
        t_key = round(t_total, 3)
        t_groups[t_key].append(c)

    Q21_jax = jnp.asarray(Q21)
    pi21_jax = jnp.asarray(pi21)

    for t_key, group in t_groups.items():
        t = float(t_key)
        if t < 1e-6:
            continue

        # Precompute Holmes-Rubin integrals for this t
        try:
            I, M = holmes_rubin_integrals(Q21_jax, pi21_jax, t)
            I = np.asarray(I)
            M = np.asarray(M)
        except Exception:
            continue

        for leaf1_name, leaf2_name, t1, t2 in group:
            seq1 = msa_int.get(leaf1_name)
            seq2 = msa_int.get(leaf2_name)
            if seq1 is None or seq2 is None:
                continue

            for col in range(len(seq1)):
                a = int(seq1[col])
                b = int(seq2[col])
                if a < 0 or a > 20 or b < 0 or b > 20:
                    continue

                M_ab = M[a, b]
                if M_ab < 1e-30:
                    continue

                # Expected dwell times: I[a,b,i,i] / M[a,b]
                for i in range(n):
                    w_total[i] += I[a, b, i, i] / M_ab

                # Expected transition counts: Q[i,j] * I[a,b,i,j] / M[a,b]
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            u_total[i, j] += Q21[i, j] * I[a, b, i, j] / M_ab

    return u_total, w_total


def accumulate_cherry_counts_fast(msa_int, cherries, Q21, pi21):
    """Vectorized version of accumulate_cherry_counts.

    Instead of looping over states, uses numpy broadcasting.
    """
    n = A21
    u_total = np.zeros((n, n), dtype=np.float64)
    w_total = np.zeros(n, dtype=np.float64)

    from collections import defaultdict
    t_groups = defaultdict(list)
    for c in cherries:
        t_total = c[2] + c[3]
        t_key = round(t_total, 3)
        t_groups[t_key].append(c)

    Q21_jax = jnp.asarray(Q21)
    pi21_jax = jnp.asarray(pi21)

    for t_key, group in t_groups.items():
        t = float(t_key)
        if t < 1e-6:
            continue

        try:
            I, M = holmes_rubin_integrals(Q21_jax, pi21_jax, t)
            I = np.asarray(I)
            M = np.asarray(M)
        except Exception:
            continue

        # Collect all (a,b) pairs from this distance group
        pair_counts = np.zeros((n, n), dtype=np.float64)
        for leaf1_name, leaf2_name, t1, t2 in group:
            seq1 = msa_int.get(leaf1_name)
            seq2 = msa_int.get(leaf2_name)
            if seq1 is None or seq2 is None:
                continue
            for col in range(len(seq1)):
                a = int(seq1[col])
                b = int(seq2[col])
                if 0 <= a <= 20 and 0 <= b <= 20:
                    pair_counts[a, b] += 1.0

        # For each (a,b) pair with nonzero count, accumulate HR stats
        for a in range(n):
            for b in range(n):
                if pair_counts[a, b] < 0.5:
                    continue
                cnt = pair_counts[a, b]
                M_ab = M[a, b]
                if M_ab < 1e-30:
                    continue

                # w_total[i] += cnt * I[a,b,i,i] / M_ab
                diag_I = np.array([I[a, b, i, i] for i in range(n)])
                w_total += cnt * diag_I / M_ab

                # u_total[i,j] += cnt * Q21[i,j] * I[a,b,i,j] / M_ab
                u_total += cnt * Q21 * I[a, b] / M_ab

        # Zero out diagonal of u_total (accumulated)
        np.fill_diagonal(u_total, np.diag(u_total))  # no-op, will zero later

    # Zero diagonal
    np.fill_diagonal(u_total, 0.0)
    return u_total, w_total


# ── M-step: reversible Q from counts ──────────────────────────────────

def m_step_reversible(u_total, w_total, pi21):
    """Compute reversible Q21 from expected counts and dwell times.

    For reversibility: S[i,j] = (u[i,j]/pi[j] + u[j,i]/pi[i]) / (w[i] + w[j])
    Then Q[i,j] = S[i,j] * pi[j].
    """
    n = len(pi21)
    S = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            # Symmetrized exchangeability
            num = u_total[i, j] + u_total[j, i]
            denom = w_total[i] * pi21[j] + w_total[j] * pi21[i]
            if denom > 1e-30:
                S[i, j] = S[j, i] = num / denom
            else:
                S[i, j] = S[j, i] = 0.0

    # Build Q = S * pi
    Q = S * pi21[None, :]
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))

    # Normalize to mean rate 1
    mean_rate = -np.sum(pi21 * np.diag(Q))
    if mean_rate > 1e-30:
        Q /= mean_rate

    return Q, S


# ── Initialize Q21 from LG08 + gap rates ──────────────────────────────

def init_q21(pi21, r_gap=0.05):
    """Initialize Q21 from LG08 exchangeabilities + uniform gap rates.

    Returns Q21 (21x21), S21 (21x21 symmetric exchangeability).
    """
    S_lg, pi_lg = lg_exchangeability()  # (20, 20), (20,) in alphabetical order

    S21 = np.zeros((A21, A21), dtype=np.float64)
    # AA-AA block from LG
    S21[:20, :20] = S_lg

    # AA-gap exchangeabilities: uniform rate r_gap
    # (will be refined by EM)
    for i in range(20):
        S21[i, 20] = r_gap
        S21[20, i] = r_gap

    # Build Q = S * pi
    Q21 = S21 * pi21[None, :]
    np.fill_diagonal(Q21, 0.0)
    np.fill_diagonal(Q21, -Q21.sum(axis=1))

    # Normalize
    mean_rate = -np.sum(pi21 * np.diag(Q21))
    if mean_rate > 1e-30:
        Q21 /= mean_rate
        S21 /= mean_rate

    return Q21, S21


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fit 21-state GTR model to Pfam")
    parser.add_argument("--n-families", type=int, default=100,
                        help="Number of Pfam families to use")
    parser.add_argument("--em-iters", type=int, default=5,
                        help="Number of EM iterations")
    parser.add_argument("--split", default="val",
                        help="Which split to use (train/val/test)")
    parser.add_argument("--output", default=None,
                        help="Output path for fitted model (default: pfam/fels21_fitted.npz)")
    parser.add_argument("--min-seqs", type=int, default=4,
                        help="Minimum sequences per family")
    parser.add_argument("--max-seqs", type=int, default=200,
                        help="Maximum sequences per family (subsample if larger)")
    parser.add_argument("--max-len", type=int, default=500,
                        help="Maximum alignment length")
    args = parser.parse_args()

    if args.output is None:
        os.makedirs(os.path.join(os.path.dirname(__file__), "..", "pfam"),
                    exist_ok=True)
        args.output = os.path.join(os.path.dirname(__file__), "..",
                                   "pfam", "fels21_fitted.npz")

    # Load splits
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    family_ids = splits[args.split]
    print(f"Split '{args.split}': {len(family_ids)} families available")

    # Load families
    print(f"Loading up to {args.n_families} families...")
    families_int = []  # list of dict {name: int32 array}
    family_names = []
    loaded = 0
    for fam_id in family_ids:
        if loaded >= args.n_families:
            break
        sto_path = os.path.join(PFAM_DIR, f"{fam_id}.sto")
        if not os.path.exists(sto_path):
            continue
        try:
            msa = parse_sto(sto_path)
        except Exception:
            continue
        if len(msa) < args.min_seqs:
            continue

        # Check alignment length
        L = len(next(iter(msa.values())))
        if L > args.max_len:
            continue

        msa_int, L = msa_to_int21(msa)

        # Subsample if too many sequences
        if len(msa_int) > args.max_seqs:
            names = list(msa_int.keys())
            rng = np.random.default_rng(42)
            chosen = rng.choice(names, args.max_seqs, replace=False)
            msa_int = {n: msa_int[n] for n in chosen}

        families_int.append(msa_int)
        family_names.append(fam_id)
        loaded += 1

    print(f"Loaded {len(families_int)} families")

    # Step 1: Compute empirical pi21
    pi21 = compute_pi21(families_int)
    print(f"Empirical pi21: gap freq = {pi21[20]:.4f}")
    print(f"  AA freqs: {', '.join(f'{AMINO_ACIDS[i]}={pi21[i]:.4f}' for i in range(20))}")

    # Step 2: Initialize Q21
    Q21, S21 = init_q21(pi21)
    print(f"\nInitial Q21: mean gap-to-AA rate = {np.mean(Q21[20, :20]):.4f}")
    print(f"  mean AA-to-gap rate = {np.mean(Q21[:20, 20]):.4f}")

    # Step 3: Build NJ trees for all families
    print("\nBuilding NJ trees...")
    trees = []
    all_cherries = []
    for i, msa_int in enumerate(families_int):
        tree = build_nj_tree(msa_int, Q21, pi21)
        trees.append(tree)
        if tree is not None:
            ch = find_cherries(tree)
            all_cherries.append((i, ch))
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(families_int)} trees built, "
                      f"{sum(len(c) for _, c in all_cherries)} cherry pairs")

    total_cherries = sum(len(c) for _, c in all_cherries)
    print(f"Total cherry pairs: {total_cherries}")

    # Step 4: EM iterations
    for em_iter in range(args.em_iters):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"EM iteration {em_iter + 1}/{args.em_iters}")

        # E-step: accumulate Holmes-Rubin counts from all cherries
        u_total = np.zeros((A21, A21), dtype=np.float64)
        w_total = np.zeros(A21, dtype=np.float64)

        for fam_idx, cherries in all_cherries:
            msa_int = families_int[fam_idx]
            try:
                u, w = accumulate_cherry_counts_fast(
                    msa_int, cherries, Q21, pi21)
                u_total += u
                w_total += w
            except Exception as e:
                print(f"  Warning: family {family_names[fam_idx]} failed: {e}")
                continue

        print(f"  E-step: total transition counts = {u_total.sum():.0f}")
        print(f"  E-step: total dwell time = {w_total.sum():.0f}")

        # Show some stats about gap-related counts
        aa_gap_counts = u_total[:20, 20].sum()
        gap_aa_counts = u_total[20, :20].sum()
        aa_aa_counts = u_total[:20, :20].sum()
        print(f"  AA->gap counts: {aa_gap_counts:.0f}")
        print(f"  gap->AA counts: {gap_aa_counts:.0f}")
        print(f"  AA->AA counts: {aa_aa_counts:.0f}")

        # M-step
        Q21_new, S21_new = m_step_reversible(u_total, w_total, pi21)

        # Log change in Q
        dQ = np.abs(Q21_new - Q21).max()
        print(f"  M-step: max |dQ| = {dQ:.6f}")

        # Show per-AA gap exchangeabilities
        gap_S = S21_new[:20, 20]
        sorted_idx = np.argsort(-gap_S)
        print(f"  Top-5 AA->gap exchangeabilities: "
              + ", ".join(f"{AMINO_ACIDS[i]}={gap_S[i]:.4f}"
                         for i in sorted_idx[:5]))
        print(f"  Bot-5 AA->gap exchangeabilities: "
              + ", ".join(f"{AMINO_ACIDS[i]}={gap_S[i]:.4f}"
                         for i in sorted_idx[-5:]))

        Q21 = Q21_new
        S21 = S21_new

        elapsed = time.time() - t0
        print(f"  Iteration time: {elapsed:.1f}s")

    # Verify reversibility
    # pi_i * Q_ij should equal pi_j * Q_ji for all i,j
    piQ = pi21[:, None] * Q21  # piQ[i,j] = pi_i * Q_ij
    rev_err = np.max(np.abs(piQ - piQ.T))
    print(f"\nReversibility check: max |pi_i*Q_ij - pi_j*Q_ji| = {rev_err:.2e}")

    # Save
    np.savez(
        args.output,
        Q21=Q21,
        pi21=pi21,
        S21=S21,
        families_used=family_names,
        em_iters=args.em_iters,
    )
    print(f"\nSaved fitted model to {args.output}")
    print(f"  Q21 shape: {Q21.shape}")
    print(f"  pi21 shape: {pi21.shape}")


if __name__ == "__main__":
    main()
