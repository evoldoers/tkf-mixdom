#!/usr/bin/env python3
"""EM-around-CherryML for a site-class mixture of GTR substitution models.

Section sec:results-mixture-sites in tkf/frontmatter-tkf.tex: "What
components emerge when we fit a mixture-of-sites to Pfam by
EM-around-CherryML?".

Generative model
----------------
Each MSA column (SITE) is drawn from one of ``C`` latent classes.  Class
``c`` has its own GTR substitution model: symmetric exchangeabilities
``S^c`` (21x21, AA + gap) and stationary distribution ``pi^c``.  The
class label is unobserved; the per-site CherryML composite log-
likelihood under class ``c`` is

    ll(s | theta_c) = sum_b sum_{i,j} N[s, b, i, j] * log P_{theta_c}(i, j; t_b)

where ``N[s, b, i, j]`` is the count of (i, j) co-occurrences at site
``s`` for pairs in time bin ``b``.

Outer EM
--------
* E-step: r[s, c] = pi_c * exp(ll(s | theta_c)) /
      sum_c'  pi_c'  *  exp(ll(s | theta_c')).
* M-step: per class c, weighted CherryML on total_counts^c[b, i, j] =
      sum_s r[s, c] * N[s, b, i, j].  Inner optimisation: Adam over log
      exchangeabilities (same as fit_fels21_cherryml.py's loop, but
      vectorised over classes).

The class-specific stationary distributions ``pi^c`` are also fit (one
per class, tied to that class's data only).  Class weights pi_k =
mean_s r[s, k].

The data extraction is identical to fit_fels21_cherryml's vectorised
path, except per-site rather than pooled across sites.
"""

from __future__ import annotations

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
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT, AMINO_ACIDS
from ancrec_benchmark import parse_sto, PFAM_DIR
from fit_fels21_cherryml import (
    encode_seq, kimura_distance, tree_pairwise_distances,
    _lg_exchangeabilities_alpha, _lg_pi_alpha,
)

TREE_DIR = os.path.expanduser("~/bio-datasets/data/pfam-seed/trees")
SPLITS_PATH = os.path.join(PFAM_DIR, "splits", "v1.json")


# -- per-site count extraction --------------------------------------------


def extract_per_site_counts(fams, t_bin_edges, max_pairs_per_fam=100,
                              min_seqs=4, rng=None, verbose=False):
    """Extract per-site (bin, i, j) co-occurrence counts.

    Each MSA column becomes one "site" in the output.  Sites are stored
    as compact COO triples to keep memory bounded.

    Args:
        fams:               iterable of family identifiers (e.g. "PF00001").
        t_bin_edges:        (n_bins + 1,) geometric bin edges.
        max_pairs_per_fam:  hard cap on per-family pair count (subsample).
        min_seqs:           skip families with fewer than this many sequences
                            in the (MSA, tree) intersection.
        rng:                numpy.random.Generator (for subsampling).
        verbose:            print progress every 50 families.

    Returns:
        site_bins:    (N_total_nonzero,) int32 — bin index per nonzero entry.
        site_i:       (N_total_nonzero,) int8  — parent residue (0-20).
        site_j:       (N_total_nonzero,) int8  — child residue (0-20).
        site_w:       (N_total_nonzero,) int32 — count for this entry.
        site_offsets: (N_sites + 1,) int32 — start index of each site's
                      block in the four arrays above (CSR-style).
        site_meta:    (N_sites,) list of dict with 'fam', 'col', 'L'.

    The CSR-style layout means site `s`'s nonzero (bin, i, j, w) tuples
    occupy ``site_offsets[s] : site_offsets[s+1]``.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_bins = len(t_bin_edges) - 1

    bins_acc = []
    i_acc = []
    j_acc = []
    w_acc = []
    offsets = [0]
    meta = []

    t0 = time.time()
    for fi, fam in enumerate(fams):
        sto_path = os.path.join(PFAM_DIR, f"{fam}.sto")
        if not os.path.exists(sto_path):
            continue
        msa = parse_sto(sto_path)
        if len(msa) < min_seqs:
            continue
        names = list(msa.keys())
        encoded = {n: encode_seq(msa[n]) for n in names}
        all_seqs = np.stack([encoded[n] for n in names])  # (N, L)
        L = all_seqs.shape[1]

        tree_path = os.path.join(TREE_DIR, f"{fam}.nwk")
        if os.path.exists(tree_path):
            tree = parse_newick(open(tree_path).read())
            leaf_names, dist_mat = tree_pairwise_distances(tree)
            leaf_to_idx = {nm: i for i, nm in enumerate(leaf_names)}
            msa_indices = [k for k, n in enumerate(names) if n in leaf_to_idx]
            if len(msa_indices) < min_seqs:
                continue

            # Sample pairs.
            n_msa = len(msa_indices)
            all_pairs = [(a, b) for a in range(n_msa) for b in range(a + 1, n_msa)]
            if len(all_pairs) > max_pairs_per_fam:
                pick = rng.choice(len(all_pairs), max_pairs_per_fam, replace=False)
                all_pairs = [all_pairs[k] for k in pick]
            # Resolve t per pair.
            pair_t = np.empty(len(all_pairs))
            valid = np.zeros(len(all_pairs), dtype=bool)
            for k, (ii, jj) in enumerate(all_pairs):
                a_name = names[msa_indices[ii]]
                b_name = names[msa_indices[jj]]
                t = dist_mat[leaf_to_idx[a_name], leaf_to_idx[b_name]]
                if t > 0 and np.isfinite(t):
                    pair_t[k] = t
                    valid[k] = True
        else:
            # Kimura fallback.
            n_msa = len(names)
            all_pairs = [(a, b) for a in range(n_msa) for b in range(a + 1, n_msa)]
            if len(all_pairs) > max_pairs_per_fam:
                pick = rng.choice(len(all_pairs), max_pairs_per_fam, replace=False)
                all_pairs = [all_pairs[k] for k in pick]
            pair_t = np.empty(len(all_pairs))
            valid = np.zeros(len(all_pairs), dtype=bool)
            for k, (ii, jj) in enumerate(all_pairs):
                t = kimura_distance(all_seqs[ii], all_seqs[jj])
                if t > 0 and np.isfinite(t):
                    pair_t[k] = t
                    valid[k] = True
            msa_indices = list(range(n_msa))

        # Resolve bin index per pair.
        good = np.where(valid)[0]
        if len(good) == 0:
            continue
        bin_idx_per_pair = np.searchsorted(t_bin_edges, pair_t[good], side='right') - 1
        bin_idx_per_pair = np.clip(bin_idx_per_pair, 0, n_bins - 1)

        # Build per-pair (i, j) along all sites.
        i_pairs = np.array([msa_indices[all_pairs[k][0]] for k in good])
        j_pairs = np.array([msa_indices[all_pairs[k][1]] for k in good])
        # all_seqs is (N, L); pair residues across all sites:
        a_residues = all_seqs[i_pairs]  # (n_good, L)
        b_residues = all_seqs[j_pairs]  # (n_good, L)

        # For each site (column), build sparse co-occurrence count.
        for col in range(L):
            ai = a_residues[:, col]
            bj = b_residues[:, col]
            # Hash (bin, ai, bj) into a single key for aggregation.
            key = bin_idx_per_pair * (21 * 21) + ai * 21 + bj
            counts = np.bincount(key)
            keep = counts > 0
            keys = np.where(keep)[0]
            if len(keys) == 0:
                offsets.append(offsets[-1])
                meta.append({'fam': fam, 'col': col, 'L': L})
                continue
            cnts = counts[keys]
            site_bin = (keys // (21 * 21)).astype(np.int8)
            rest = keys % (21 * 21)
            site_i_local = (rest // 21).astype(np.int8)
            site_j_local = (rest % 21).astype(np.int8)
            bins_acc.append(site_bin)
            i_acc.append(site_i_local)
            j_acc.append(site_j_local)
            w_acc.append(cnts.astype(np.int32))
            offsets.append(offsets[-1] + len(keys))
            meta.append({'fam': fam, 'col': col, 'L': L})

        if verbose and ((fi + 1) % 50 == 0 or fi == len(fams) - 1):
            print(f'  [extract] {fi+1}/{len(fams)} fams, '
                  f'sites so far={len(meta)}, '
                  f'nonzeros so far={offsets[-1]}, '
                  f'elapsed={time.time()-t0:.1f}s')

    site_bins = np.concatenate(bins_acc) if bins_acc else np.zeros(0, dtype=np.int8)
    site_i = np.concatenate(i_acc) if i_acc else np.zeros(0, dtype=np.int8)
    site_j = np.concatenate(j_acc) if j_acc else np.zeros(0, dtype=np.int8)
    site_w = np.concatenate(w_acc) if w_acc else np.zeros(0, dtype=np.int32)
    site_offsets = np.array(offsets, dtype=np.int32)

    return site_bins, site_i, site_j, site_w, site_offsets, meta


# -- EM-around-CherryML ---------------------------------------------------


def _build_Q(S, pi):
    """Build a 21x21 GTR rate matrix from symmetric S and equilibrium pi.

    Q[i, j] = S[i, j] * pi[j] for i != j, then diagonal = -row sum.
    Mean rate normalised to 1.
    """
    Q = S * pi[None, :]
    Q = Q.at[jnp.diag_indices(21)].set(0.0)
    Q = Q.at[jnp.diag_indices(21)].set(-Q.sum(axis=1))
    mean_rate = -jnp.sum(pi * jnp.diag(Q))
    return Q / jnp.maximum(mean_rate, 1e-30)


def _site_log_lik_under_class(site_bins, site_i, site_j, site_w,
                                site_offsets, log_P):
    """Compute per-site composite log-lik given log P[bin, i, j] for one class.

    Returns:
        (N_sites,) log-lik values.
    """
    # Vectorised: gather log_P at every (bin, i, j) triple, weight by w,
    # then segment-sum into per-site log-lik.
    contrib = (site_w *
                 log_P[site_bins, site_i, site_j])  # (N_total_nonzero,)
    n_sites = len(site_offsets) - 1
    out = np.zeros(n_sites, dtype=np.float64)
    if len(contrib) == 0:
        return out
    # CSR segment-sum.
    np.add.at(out, np.repeat(np.arange(n_sites),
                              site_offsets[1:] - site_offsets[:-1]),
              contrib)
    return out


def _per_class_total_counts(site_bins, site_i, site_j, site_w,
                              site_offsets, resp_c, n_bins):
    """Weighted sum over sites of N[s, bin, i, j] * resp_c[s].

    Returns: (n_bins, 21, 21).
    """
    n_sites = len(site_offsets) - 1
    weights = np.repeat(resp_c.astype(np.float64),
                          site_offsets[1:] - site_offsets[:-1])
    contrib = site_w.astype(np.float64) * weights
    flat_idx = (site_bins.astype(np.int64) * (21 * 21)
                  + site_i.astype(np.int64) * 21
                  + site_j.astype(np.int64))
    out = np.zeros(n_bins * 21 * 21, dtype=np.float64)
    np.add.at(out, flat_idx, contrib)
    return out.reshape(n_bins, 21, 21)


def _make_e_step_jax(n_sites, n_classes, n_bins, K=21):
    """Build a JIT-compiled E-step that computes (n_sites, n_classes)
    log-lik in a single vectorised GPU pass.

    Replaces the per-class numpy CSR scan in the original E-step.
    Inputs are precomputed JAX arrays (passed in once and cached).
    """
    rows_i, cols_i = jnp.triu_indices(K, k=1)

    def _build_Q_vec(S_c, pi_c):
        Q = S_c * pi_c[None, :]
        Q = Q.at[jnp.diag_indices(K)].set(0.0)
        Q = Q.at[jnp.diag_indices(K)].set(-Q.sum(axis=1))
        mean_rate = -jnp.sum(pi_c * jnp.diag(Q))
        return Q / jnp.maximum(mean_rate, 1e-30)

    @jax.jit
    def e_step(log_S_classes, log_pi_classes, t_centers,
                site_bins, site_i, site_j, site_w, site_idx):
        # Build (C, K, K) S from upper-triangular log_S.
        C = log_S_classes.shape[0]
        S = jnp.zeros((C, K, K))
        S = S.at[:, rows_i, cols_i].set(jnp.exp(log_S_classes))
        S = S + jnp.transpose(S, (0, 2, 1))
        pi = jax.nn.softmax(log_pi_classes, axis=-1)  # (C, K)
        Q = jax.vmap(_build_Q_vec)(S, pi)             # (C, K, K)
        # (C, n_bins, K, K) transition matrices.
        def expm_grid_for_class(Qc):
            return jax.vmap(lambda t: jax.scipy.linalg.expm(Qc * t))(t_centers)
        P = jax.vmap(expm_grid_for_class)(Q)          # (C, n_bins, K, K)
        log_P = jnp.log(jnp.maximum(P, 1e-30))        # (C, n_bins, K, K)
        # Per-entry contribution: shape (C, N_total).
        # log_P[:, bin_n, i_n, j_n] * w_n.
        per_entry = log_P[:, site_bins, site_i, site_j] * site_w[None, :]
        # Segment-sum to (C, n_sites) via .at[].add (GPU scatter).
        out = jnp.zeros((C, n_sites))
        out = out.at[:, site_idx].add(per_entry)
        return out.T  # (n_sites, C)

    return e_step


def _make_m_step_total_counts_jax(n_classes, n_bins, K=21):
    """Build a JIT-compiled total-counts builder for the M-step,
    vectorised across classes."""
    @jax.jit
    def m_step_total_counts(resp, site_bins, site_i, site_j, site_w, site_idx):
        # resp: (n_sites, C); site_*: (N_total,)
        # Output: (C, n_bins, K, K).
        resp_per_entry = resp[site_idx]              # (N_total, C)
        weighted = resp_per_entry * site_w[:, None]  # (N_total, C)
        flat = (site_bins.astype(jnp.int32) * (K * K)
                + site_i.astype(jnp.int32) * K
                + site_j.astype(jnp.int32))           # (N_total,)
        # (n_bins*K*K, C) accumulator → reshape.
        out = jnp.zeros((n_bins * K * K, n_classes))
        out = out.at[flat].add(weighted)
        return jnp.transpose(out.reshape(n_bins, K, K, n_classes),
                                (3, 0, 1, 2))
    return m_step_total_counts


def _make_inner_loss_fn(t_centers):
    """Build a JIT-compiled neg-log-lik function for one class.

    Args:
        t_centers: (n_bins,) bin centres.

    Returns:
        Function (log_S_vec, log_pi_logits, total_counts) -> neg log lik.
    """
    n = 21
    rows, cols = np.triu_indices(n, k=1)
    rows_j = jnp.array(rows)
    cols_j = jnp.array(cols)

    @jax.jit
    def neg_log_lik(log_S_vec, log_pi_logits, total_counts):
        S_vals = jnp.exp(log_S_vec)
        S = jnp.zeros((n, n))
        S = S.at[rows_j, cols_j].set(S_vals)
        S = S + S.T
        pi = jax.nn.softmax(log_pi_logits)
        Q = _build_Q(S, pi)
        # Per-bin: sum_{ij} N[bin, i, j] * log P(i, j; t_bin).
        def per_bin(t, N):
            P = jax.scipy.linalg.expm(Q * t)
            P = jnp.maximum(P, 1e-30)
            return jnp.sum(N * jnp.log(P))
        ll = jax.vmap(per_bin)(t_centers, total_counts).sum()
        return -ll

    return neg_log_lik


def em_around_cherryml(site_bins, site_i, site_j, site_w, site_offsets,
                         t_centers, n_classes=2, n_outer_iters=10,
                         n_inner_iters=20, lr=0.05, init_seed=0,
                         verbose=False):
    """Outer EM around CherryML.  Returns class-specific (S, pi) and weights.

    Args:
        site_bins, site_i, site_j, site_w, site_offsets:
            sparse per-site count representation.
        t_centers: (n_bins,) bin centres (geometric mid-points).
        n_classes: C.
        n_outer_iters: outer EM iterations.
        n_inner_iters: inner CherryML M-step iterations per class per outer iter.
        lr: Adam learning rate for the inner CherryML M-step.
        init_seed: numpy seed for class initialisation.
        verbose: print per-iteration log-likelihood and class weights.

    Returns:
        dict with:
            'S':       (C, 21, 21) per-class symmetric exchangeabilities.
            'pi':      (C, 21)     per-class equilibrium frequencies.
            'weights': (C,)        class weights.
            'history': list of dicts (per outer iter): {'iter', 'll', 'weights'}.
            'resp':    (N_sites, C) final responsibilities.
    """
    n_sites = len(site_offsets) - 1
    n_bins = int(t_centers.shape[0])
    rng = np.random.default_rng(init_seed)

    # Initialise classes with LG perturbations.
    lg_S20 = _lg_exchangeabilities_alpha()  # (20, 20) — AA only.
    lg_pi20 = _lg_pi_alpha()                # (20,)
    # Extend to 21x21 with gap-row/col exchangeabilities at LG mean.
    lg_S = np.zeros((21, 21))
    lg_S[:20, :20] = lg_S20
    lg_mean = float(lg_S20[np.triu_indices(20, k=1)].mean())
    lg_S[:20, 20] = lg_mean
    lg_S[20, :20] = lg_mean
    rows, cols = np.triu_indices(21, k=1)
    log_S_init = np.log(np.maximum(lg_S[rows, cols], 1e-6))
    # Tile + perturb.
    log_S_classes = np.tile(log_S_init, (n_classes, 1))
    log_S_classes += 0.3 * rng.standard_normal(log_S_classes.shape)
    # Stationary: full 21-vector, init alphabet+gap with LG, perturb.
    pi_init = np.concatenate([lg_pi20, [0.05]])
    pi_init = pi_init / pi_init.sum()
    log_pi_classes = np.tile(np.log(np.maximum(pi_init, 1e-6)),
                                (n_classes, 1))
    log_pi_classes += 0.2 * rng.standard_normal(log_pi_classes.shape)
    # Class weights uniform.
    log_pi_class = np.log(np.full(n_classes, 1.0 / n_classes))

    inner_loss = _make_inner_loss_fn(jnp.array(t_centers))
    grad_fn = jax.jit(jax.value_and_grad(inner_loss, argnums=(0, 1)))

    # Precompute JAX arrays for vectorised CSR scans (pushed to device once).
    site_idx_np = np.repeat(np.arange(n_sites),
                                site_offsets[1:] - site_offsets[:-1])
    site_bins_j = jnp.asarray(site_bins, dtype=jnp.int32)
    site_i_j = jnp.asarray(site_i, dtype=jnp.int32)
    site_j_j = jnp.asarray(site_j, dtype=jnp.int32)
    site_w_j = jnp.asarray(site_w, dtype=jnp.float64)
    site_idx_j = jnp.asarray(site_idx_np, dtype=jnp.int32)
    t_centers_j = jnp.asarray(t_centers, dtype=jnp.float64)
    e_step_fn = _make_e_step_jax(n_sites, n_classes, n_bins)
    m_step_tc_fn = _make_m_step_total_counts_jax(n_classes, n_bins)

    history = []

    for outer in range(n_outer_iters):
        # ---- E-step: vectorised across all classes on device ----
        log_S_j = jnp.asarray(log_S_classes, dtype=jnp.float64)
        log_pi_j = jnp.asarray(log_pi_classes, dtype=jnp.float64)
        log_lik_per_class = np.asarray(e_step_fn(
            log_S_j, log_pi_j, t_centers_j,
            site_bins_j, site_i_j, site_j_j, site_w_j, site_idx_j))
        # log responsibility = log_pi_class + log_lik - logsumexp.
        log_unnorm = log_lik_per_class + log_pi_class[None, :]
        max_per = log_unnorm.max(axis=1, keepdims=True)
        unnorm = np.exp(log_unnorm - max_per)
        norm = unnorm.sum(axis=1, keepdims=True)
        resp = unnorm / np.maximum(norm, 1e-300)
        log_total_data = (max_per[:, 0] + np.log(np.maximum(norm[:, 0], 1e-300))).sum()

        # ---- M-step: per-class weighted CherryML on responsibility-weighted counts ----
        # Build all C total_counts in one device pass.
        resp_j = jnp.asarray(resp, dtype=jnp.float64)
        all_tc_j = m_step_tc_fn(
            resp_j, site_bins_j, site_i_j, site_j_j,
            site_w_j, site_idx_j)  # (C, n_bins, K, K)
        new_log_S = log_S_classes.copy()
        new_log_pi = log_pi_classes.copy()
        for c in range(n_classes):
            tc_jax = all_tc_j[c]
            log_S_c = jnp.array(log_S_classes[c])
            log_pi_c = jnp.array(log_pi_classes[c])
            m_S = jnp.zeros_like(log_S_c)
            v_S = jnp.zeros_like(log_S_c)
            m_p = jnp.zeros_like(log_pi_c)
            v_p = jnp.zeros_like(log_pi_c)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for it in range(n_inner_iters):
                _, (g_S, g_p) = grad_fn(log_S_c, log_pi_c, tc_jax)
                m_S = beta1 * m_S + (1 - beta1) * g_S
                v_S = beta2 * v_S + (1 - beta2) * g_S ** 2
                m_p = beta1 * m_p + (1 - beta1) * g_p
                v_p = beta2 * v_p + (1 - beta2) * g_p ** 2
                t_corr = (1 - beta1 ** (it + 1))
                v_corr = (1 - beta2 ** (it + 1))
                log_S_c = log_S_c - lr * (m_S / t_corr) / (
                    jnp.sqrt(v_S / v_corr) + eps)
                log_pi_c = log_pi_c - lr * (m_p / t_corr) / (
                    jnp.sqrt(v_p / v_corr) + eps)
            new_log_S[c] = np.array(log_S_c)
            new_log_pi[c] = np.array(log_pi_c)
        log_S_classes = new_log_S
        log_pi_classes = new_log_pi
        # Update class weights (mass).
        weights_new = resp.sum(axis=0) / n_sites
        log_pi_class = np.log(np.maximum(weights_new, 1e-300))

        if verbose:
            print(f'  EM iter {outer+1}: total log-lik = {log_total_data:.2f}, '
                  f'class weights = {weights_new.round(3)}')
        history.append({'iter': outer + 1, 'll': float(log_total_data),
                          'weights': weights_new.tolist()})

    # Build final outputs.
    S_out = np.zeros((n_classes, 21, 21))
    pi_out = np.zeros((n_classes, 21))
    for c in range(n_classes):
        Sv = np.exp(log_S_classes[c])
        S_out[c, rows, cols] = Sv
        S_out[c] = S_out[c] + S_out[c].T
        pi_out[c] = np.array(jax.nn.softmax(jnp.array(log_pi_classes[c])))
    weights_out = np.exp(log_pi_class)
    weights_out = weights_out / weights_out.sum()

    return {
        'S': S_out,
        'pi': pi_out,
        'weights': weights_out,
        'history': history,
        'resp': resp,
    }


# -- main entry ----------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description='EM-around-CherryML for site-class GTR mixture.')
    p.add_argument('--n-families', type=int, default=50)
    p.add_argument('--n-classes', type=int, default=2)
    p.add_argument('--n-bins', type=int, default=20)
    p.add_argument('--t-min', type=float, default=0.01)
    p.add_argument('--t-max', type=float, default=5.0)
    p.add_argument('--n-outer', type=int, default=8)
    p.add_argument('--n-inner', type=int, default=20)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--max-pairs-per-fam', type=int, default=100)
    p.add_argument('--init-seed', type=int, default=0)
    p.add_argument('--output', type=str, default=None)
    args = p.parse_args()

    if args.output is None:
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'pfam')
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir,
            f'cherryml_mixture_C{args.n_classes}_n{args.n_families}.npz')

    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    fams = splits['train'][:args.n_families]
    print(f'Using {len(fams)} families.')

    t_bin_edges = np.geomspace(args.t_min, args.t_max, args.n_bins + 1)
    t_centers = np.sqrt(t_bin_edges[:-1] * t_bin_edges[1:])

    print(f'Extracting per-site counts (max_pairs_per_fam={args.max_pairs_per_fam})...')
    rng = np.random.default_rng(args.init_seed)
    sb, si, sj, sw, soff, meta = extract_per_site_counts(
        fams, t_bin_edges, max_pairs_per_fam=args.max_pairs_per_fam,
        rng=rng, verbose=True)
    print(f'  N_sites={len(meta)}, total nonzeros={len(sw)}')

    print(f'Running EM (C={args.n_classes}, outer={args.n_outer}, inner={args.n_inner})...')
    out = em_around_cherryml(
        sb, si, sj, sw, soff, t_centers,
        n_classes=args.n_classes, n_outer_iters=args.n_outer,
        n_inner_iters=args.n_inner, lr=args.lr,
        init_seed=args.init_seed, verbose=True)

    print(f'Final class weights: {out["weights"].round(4)}')
    np.savez(args.output,
              S=out['S'], pi=out['pi'], weights=out['weights'],
              history=np.array(out['history'], dtype=object),
              t_centers=t_centers,
              n_classes=args.n_classes, n_families=args.n_families)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
