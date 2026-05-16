"""Simulator for the mixture-of-sites GTR substitution model (TKF §9).

Reuses ``core.ctmc.build_Q_from_S_pi`` and ``core.ctmc.transition_matrix``
(JAX) for the Q + expm pieces.  Mean-rate normalisation is applied to
match ``em_around_cherryml._build_Q``'s convention so the simulator
produces data consistent with the EM estimator.

Generative model
----------------
* C latent classes with per-class (S^c, pi^c) GTR rate matrices.
* Site (column) class labels c_s ~ Categorical(weights).
* Per cherry pair: time t_p drawn uniformly from a user grid (the same
  grid used as bin centres by the EM).  Parent residue a ~ pi[c_s].
  Child residue b ~ Categorical(P(child | a, t_p, Q[c_s])) where
  P = expm(t_p * Q[c_s]).

Output matches ``extract_per_site_counts``: a sparse CSR-style
(site_bins, site_i, site_j, site_w, site_offsets) representation, plus
fully-labeled per-site / per-pair truth so recovery tests can check
both the params and the responsibilities.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from ..core.ctmc import build_Q_from_S_pi, transition_matrix


def _build_Q_normalised(S, pi):
    """Mean-rate-normalised GTR Q matching em_around_cherryml._build_Q."""
    Q = np.asarray(build_Q_from_S_pi(jnp.asarray(S), jnp.asarray(pi)),
                     dtype=np.float64)
    mean_rate = -float((np.asarray(pi) * np.diag(Q)).sum())
    return Q / max(mean_rate, 1e-30)


def _expm_t_Q(Q, t):
    """expm(t * Q) via the existing JAX helper, returned as numpy."""
    return np.asarray(transition_matrix(jnp.asarray(Q), float(t)),
                         dtype=np.float64)


def simulate_mixture_sites_pair_data(
        S_true, pi_true, weights_true, t_grid,
        n_sites, n_pairs_per_site, *, rng=None):
    """Simulate cherry-pair data from a mixture-of-sites GTR.

    Args:
        S_true:           (C, K, K) symmetric per-class exchangeabilities
                            (K=21 for AA+gap; any K accepted).
        pi_true:          (C, K) per-class stationary distributions.
        weights_true:     (C,) site-class mixing weights.
        t_grid:           (n_bins,) candidate divergence times — bin
                            centres in the EM's time discretisation.
        n_sites:          number of independent MSA sites.
        n_pairs_per_site: number of cherry pairs per site (all pairs at a
                            site share the SAME class label, per the
                            generative model).
        rng:              numpy.random.Generator (default: fresh).

    Returns:
        dict with keys (sparse data + truth labels):
          'site_bins'/'site_i'/'site_j'/'site_w'/'site_offsets':
              CSR-style sparse representation matching
              ``extract_per_site_counts``.
          't_centers': (n_bins,) — copy of t_grid.
          'true_class':       (n_sites,) generator class per site.
          'all_pairs_class':  (n_sites, n_pairs_per_site) — per-pair label.
          'all_pairs_t' / 'all_pairs_bin' / 'all_pairs_a' / 'all_pairs_b':
              per-pair (t, bin, parent residue, child residue).
    """
    if rng is None:
        rng = np.random.default_rng()
    weights_true = np.asarray(weights_true, dtype=np.float64)
    if not np.isclose(weights_true.sum(), 1.0):
        raise ValueError(f"weights must sum to 1; got {weights_true.sum()}")
    C = weights_true.shape[0]
    K = pi_true.shape[1]
    if S_true.shape != (C, K, K):
        raise ValueError(f"S_true shape {S_true.shape} != ({C},{K},{K})")
    if pi_true.shape != (C, K):
        raise ValueError(f"pi_true shape {pi_true.shape} != ({C},{K})")
    if not np.allclose(np.transpose(S_true, (0, 2, 1)), S_true):
        raise ValueError("S_true must be symmetric per class")
    if not np.allclose(pi_true.sum(axis=1), 1.0):
        raise ValueError("pi_true rows must sum to 1")

    t_grid = np.asarray(t_grid, dtype=np.float64)
    n_bins = t_grid.shape[0]

    # Precompute (C, n_bins, K, K) transition matrices.
    P_per_class_per_bin = np.zeros((C, n_bins, K, K), dtype=np.float64)
    for c in range(C):
        Qc = _build_Q_normalised(S_true[c], pi_true[c])
        for b in range(n_bins):
            P_per_class_per_bin[c, b] = _expm_t_Q(Qc, t_grid[b])

    true_class = rng.choice(C, size=n_sites, p=weights_true).astype(np.int8)
    pair_bin = rng.integers(low=0, high=n_bins,
                              size=(n_sites, n_pairs_per_site)).astype(np.int8)
    pair_t = t_grid[pair_bin]

    pair_a = np.zeros((n_sites, n_pairs_per_site), dtype=np.int8)
    for c in range(C):
        mask = (true_class == c)
        n_in_c = int(mask.sum())
        if n_in_c == 0:
            continue
        pair_a[mask] = rng.choice(K, size=(n_in_c, n_pairs_per_site),
                                     p=pi_true[c]).astype(np.int8)

    # Vectorise child-sampling: gather P-rows by (class, bin, parent),
    # take cumsum, and inverse-cdf with uniforms.
    flat_class = np.repeat(true_class, n_pairs_per_site)
    flat_bin = pair_bin.ravel()
    flat_a = pair_a.ravel()
    P_rows = P_per_class_per_bin[flat_class, flat_bin, flat_a]  # (N, K)
    cdf = np.cumsum(P_rows, axis=1)
    cdf[:, -1] = 1.0  # fix tiny float drift
    u = rng.random(size=cdf.shape[0])
    flat_b = (cdf < u[:, None]).sum(axis=1).astype(np.int8)
    pair_b = flat_b.reshape(n_sites, n_pairs_per_site)

    # Sparse CSR aggregation per site.
    bins_acc, i_acc, j_acc, w_acc = [], [], [], []
    offsets = [0]
    KK = K * K
    for s in range(n_sites):
        key = (pair_bin[s].astype(np.int32) * KK
               + pair_a[s].astype(np.int32) * K
               + pair_b[s].astype(np.int32))
        cnt = np.bincount(key, minlength=n_bins * KK)
        keys = np.where(cnt > 0)[0]
        cnts = cnt[keys].astype(np.int32)
        b_idx = (keys // KK).astype(np.int8)
        rest = keys % KK
        bins_acc.append(b_idx)
        i_acc.append((rest // K).astype(np.int8))
        j_acc.append((rest % K).astype(np.int8))
        w_acc.append(cnts)
        offsets.append(offsets[-1] + len(keys))

    return {
        'site_bins': (np.concatenate(bins_acc) if bins_acc
                        else np.zeros(0, dtype=np.int8)),
        'site_i': (np.concatenate(i_acc) if i_acc
                     else np.zeros(0, dtype=np.int8)),
        'site_j': (np.concatenate(j_acc) if j_acc
                     else np.zeros(0, dtype=np.int8)),
        'site_w': (np.concatenate(w_acc) if w_acc
                     else np.zeros(0, dtype=np.int32)),
        'site_offsets': np.array(offsets, dtype=np.int32),
        't_centers': t_grid,
        'true_class': true_class,
        'all_pairs_class': np.broadcast_to(
            true_class[:, None], (n_sites, n_pairs_per_site)).copy(),
        'all_pairs_t': pair_t,
        'all_pairs_bin': pair_bin,
        'all_pairs_a': pair_a,
        'all_pairs_b': pair_b,
    }
