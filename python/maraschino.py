#!/usr/bin/env python3
"""
maraschino.py — distill cherries to state machines.

CherryML-like training of reversible MixDom model.

Four modes:
  A) count   — Read Stockholm MSAs, extract cherries, count adjacencies.
               Supports incremental checkpointing and resume.
  B) fit     — Fit MixDom model params to precomputed counts via Adam.
  C) distill — Generate order-1 Singlet-HMM and Pair-WFST (Machine Boss JSON).
  D) fetch   — Download Pfam Stockholm alignments (explicit or pseudorandom).

Usage:
  python maraschino.py count   --msa-dir DIR --out counts.npz [--n-tau-bins 32]
  python maraschino.py fit     --counts counts.npz --out params.npz [--n-domains 20] [--n-classes 4]
  python maraschino.py distill --params params.npz --tau 0.1,0.5,1.0 --out distilled [--precision 6]
  python maraschino.py fetch   --out-dir pfam/ --random 100 --seed 42
"""

import argparse
import hashlib
import json
import os
import sys
import re
import time
from datetime import datetime, timezone
from functools import partial

import numpy as np


# ============================================================
# Logging
# ============================================================
_log_file = None
_log_start = time.monotonic()


def _log(msg, end='\n'):
    """Write a timestamped message to stderr and optionally a log file."""
    elapsed = time.monotonic() - _log_start
    line = f"[{elapsed:8.1f}s] {msg}"
    sys.stderr.write(line + end)
    sys.stderr.flush()
    if _log_file is not None:
        _log_file.write(line + end)
        _log_file.flush()


def _log_every(step, total, interval, msg_fn):
    """Call msg_fn() and log if step is at a reporting boundary.

    interval: report every N steps (0 = auto-choose based on total).
    Always reports step 0 (first) and step total-1 (last).
    """
    if interval <= 0:
        # Auto: ~20 reports for short runs, ~50 for long, at least every 1
        interval = max(1, total // 50) if total > 100 else max(1, total // 20)
    if step == 0 or step == total - 1 or (step + 1) % interval == 0:
        _log(msg_fn())


def _setup_logging(log_path=None):
    """Open a log file (if specified). Call once at startup."""
    global _log_file, _log_start
    _log_start = time.monotonic()
    if log_path:
        _log_file = open(log_path, 'a')
        _log(f"Logging to {log_path}")

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.linalg import expm

# ============================================================
# Constants
# ============================================================
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA = len(AMINO_ACIDS)  # 20
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}
BOS_IDX = AA       # 20
EOS_IDX = AA + 1   # 21
VOCAB = AA + 2     # 22

# Emitting state types in nested Pair HMM (per domain)
# MM=0, MI=1, MD=2, II=3, DD=4
EST_MM, EST_MI, EST_MD, EST_II, EST_DD = range(5)
N_EST = 5

# Top-level Pair HMM states: S=0, M=1, I=2, D=3, E=4
TL_S, TL_M, TL_I, TL_D, TL_E = range(5)

# Mapping: emitting state type -> top-level type
EST_TO_TL = jnp.array([TL_M, TL_M, TL_M, TL_I, TL_D])

# Alignment column types for adjacency counting
COL_S, COL_M, COL_I, COL_D, COL_E = range(5)

# Geometric bin sizes for discretizing tau
TAU_MIN = 0.001
TAU_MAX = 10.0

# ============================================================
# Substitution-model helpers, BDI, TKF91 transition matrix:
#   re-exported from tkfmixdom.jax.distill.maraschino (single source
#   of truth). The duplicate definitions previously inlined here have
#   been removed; only this CLI script's unique helpers (cherry counts,
#   CherryML fitting, Pfam fetch) live below.
#
# The LG08 constants and helpers (`_PAML_ORDER`, `_PAML_TO_ALPHA`,
# `_LG_S_LOWER`, `_LG_PI`, `_lower_tri_to_sym`, `get_lg08`,
# `eigen_decompose`, `transition_probs[_from_eigen]`, `gamma_rates`,
# `_gamma_quantile_midpoints`, `bdi_params`, `tkf91_trans`) all come
# from there. `build_rate_matrix` is itself a re-export from
# `tkfmixdom.jax.core.ctmc.build_rate_matrix_unit_normalized` — see
# that function's docstring for why callers should usually NOT use it.
# ============================================================
from tkfmixdom.jax.distill.maraschino import (
    _PAML_ORDER, _PAML_TO_ALPHA, _LG_S_LOWER, _LG_PI,
    _lower_tri_to_sym, get_lg08,
    build_rate_matrix,
    eigen_decompose, transition_probs_from_eigen, transition_probs,
    gamma_rates, _gamma_quantile_midpoints,
    bdi_params,
    tkf91_trans,
)


# ============================================================
# MixDom model: parameter container and distillation
# ============================================================
def make_raw_params(n_domains, n_classes, key, per_domain_s=False,
                    n_classes_dynamic=1):
    """Initialize raw (unconstrained) parameters for MixDom model.
    Returns a dict of JAX arrays.

    Args:
        per_domain_s: if True, learn per-domain S_exch[k,a,b] (N×190 params).
            If False (default), learn shared S_exch[a,b] (190 params).
        n_classes_dynamic: number of dynamic site classes (default 1 = static).
            When > 1, adds log_rho_inter (scalar) parameter for F81-style class switching rate.
    """
    keys = jax.random.split(key, 10)
    S_lg, pi_lg = get_lg08()

    params = {
        # Top-level TKF rates (log-space for softplus)
        'log_lam0': jnp.array(-1.0),
        'log_mu0':  jnp.array(0.0),
        # Per-domain TKF rates
        'log_lam': jnp.full(n_domains, -1.0) + 0.1 * jax.random.normal(keys[0], (n_domains,)),
        'log_mu':  jnp.full(n_domains,  0.0) + 0.1 * jax.random.normal(keys[1], (n_domains,)),
        # Per-domain fragment extension (logit-space for sigmoid)
        'logit_r': jnp.full(n_domains, 1.0) + 0.1 * jax.random.normal(keys[2], (n_domains,)),
        # Domain weights (log-space for softmax)
        'log_v': jnp.zeros(n_domains) + 0.01 * jax.random.normal(keys[3], (n_domains,)),
        # Per-domain equilibrium frequencies (log-space for softmax)
        'log_pi': jnp.tile(jnp.log(pi_lg + 1e-10), (n_domains, 1))
                  + 0.05 * jax.random.normal(keys[4], (n_domains, AA)),
        # Exchangeability matrix (per-domain or shared)
        'log_S': (jnp.tile(jnp.log(S_lg + 1e-10)[None], (n_domains, 1, 1))
                  if per_domain_s else jnp.log(S_lg + 1e-10)),
        # Gamma shape parameter
        'log_alpha_gamma': jnp.array(0.0),  # alpha=1 => exponential
    }

    # Dynamic site class parameters
    if n_classes_dynamic > 1:
        D = n_classes_dynamic
        keys_dyn = jax.random.split(keys[9], 5)

        # rho_inter: scalar class switching rate (F81-style)
        params['log_rho_inter'] = jnp.array(-3.0)

        # gamma_class: (D,) per-class within-class rate multiplier
        # softplus(0.0) ≈ 0.693 each, init so gamma_class ≈ ones
        params['log_gamma_class'] = jnp.zeros(D)

        # Per-class equilibrium: (D, A) — shared across domains, symmetry-breaking
        # pi^{(d)}_a: class d has its own amino acid preferences
        params['log_pi_classes'] = (
            jnp.tile(jnp.log(pi_lg + 1e-10), (D, 1))
            + 0.1 * jax.random.normal(keys_dyn[1], (D, AA))
        )

        # Per-domain class weights: (N, D) — w_{n,d}: domain n's usage of class d
        # pi_{(d,a)|n} = w_{n,d} * pi^{(d)}_a
        params['logit_class_weights'] = (
            jnp.zeros((n_domains, D))
            + 0.05 * jax.random.normal(keys_dyn[2], (n_domains, D))
        )

    return params


def constrain_params(raw):
    """Convert raw params to constrained (positive rates, valid probs, etc.).

    Delegates to the canonical implementation in tkfmixdom.jax.distill.maraschino.
    """
    from tkfmixdom.jax.distill.maraschino import constrain_params as _constrain
    return _constrain(raw)


# ============================================================
# Algebraic distillation: compute order-1 adjacency frequencies
#
# Uses Woodbury identity to avoid 5N×5N inversion:
#   (I - T_hat)^{-1} = G + G·E·K·S^T·G
# where G = block_diag(G_1,...,G_N) with closed-form 5×5 per-domain inverses,
# and K is a single 3×3 Woodbury kernel (only {M,I,D} couple across domains).
# All domain loops are eliminated via vectorized einsum/vmap.
# ============================================================

# `_batch_inv_3x3`, `_inv_2x2`, and the distillation pipeline all
# come from tkfmixdom.jax.distill.maraschino — single source of truth.
from tkfmixdom.jax.distill.maraschino import (
    _batch_inv_3x3, _inv_2x2,
    distill_mixdom,
    normalize_freqs_wfst as _normalize_freqs,
    precompute_mixdom as _precompute_mixdom,
)



# ============================================================
# Part A: Parse Stockholm MSAs, extract cherries, count adjacencies
# ============================================================

def parse_stockholm(filepath):
    """Parse a Stockholm alignment file (plain or gzipped). Returns dict {name: aligned_sequence}."""
    import gzip
    seqs = {}
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                name, seq = parts[0], parts[1]
                if name in seqs:
                    seqs[name] += seq
                else:
                    seqs[name] = seq
    return seqs


def aa_composition(seq):
    """Compute amino acid composition vector (20-dim) from ungapped sequence."""
    comp = np.zeros(AA)
    for c in seq:
        if c in AA_TO_IDX:
            comp[AA_TO_IDX[c]] += 1
    total = comp.sum()
    if total > 0:
        comp /= total
    return comp


def ungap(seq):
    """Remove gap characters from sequence."""
    return ''.join(c for c in seq if c in AA_TO_IDX)


def p_distance(seq1, seq2):
    """Compute p-distance between two aligned sequences (fraction of mismatches at aligned positions)."""
    matches = 0
    mismatches = 0
    for a, b in zip(seq1, seq2):
        if a in AA_TO_IDX and b in AA_TO_IDX:
            if a == b:
                matches += 1
            else:
                mismatches += 1
    total = matches + mismatches
    if total == 0:
        return 1.0
    return mismatches / total


def pdist_to_evo_time(pdist, alpha_gamma=1.0, n_classes=4):
    """Convert p-distance to evolutionary time using gamma-corrected Poisson model.
    t = -alpha * log(1 - pdist/alpha_correction) approximately.
    For amino acids, use a simple correction.
    """
    if pdist >= 0.95:
        return 5.0  # cap at high distance
    # Simple Poisson correction: t = -log(1 - pdist * 20/19)
    corrected = 1.0 - pdist * (AA / (AA - 1.0))
    if corrected <= 0.01:
        return 5.0
    return -np.log(corrected)


def estimate_evo_time_lg(seq1, seq2, n_classes=4, alpha_gamma=1.0):
    """Estimate evolutionary time from pairwise alignment using LG08 + gamma.
    Uses grid search over t for simplicity.
    """
    S_exch, pi_lg = get_lg08()
    # LG08 is a published matrix calibrated to mean rate 1; gamma rates
    # below absorb cross-class rate variation. Unit normalisation is the
    # intended convention here.
    Q_lg = np.array(build_rate_matrix(S_exch, pi_lg, acknowledged_lossy=True))
    pi_np = np.array(pi_lg)

    # Compute rate multipliers
    alpha_g = max(alpha_gamma, 0.01)
    mid_probs = (np.arange(n_classes) + 0.5) / n_classes
    from scipy.stats import norm as norm_dist
    z = norm_dist.ppf(mid_probs)
    factor = 1.0 - 1.0/(9.0*alpha_g) + z * np.sqrt(1.0/(9.0*alpha_g))
    rates = alpha_g * np.maximum(factor, 0.01)**3
    rates = rates / rates.mean()

    # Collect aligned pairs
    pairs = []
    for a, b in zip(seq1, seq2):
        if a in AA_TO_IDX and b in AA_TO_IDX:
            pairs.append((AA_TO_IDX[a], AA_TO_IDX[b]))
    if not pairs:
        return 1.0

    # Grid search over t
    from scipy.linalg import expm as scipy_expm
    best_t = 0.1
    best_ll = -np.inf
    for t in np.geomspace(0.001, 10.0, 50):
        ll = 0.0
        P_avg = np.zeros((AA, AA))
        for c in range(n_classes):
            P_avg += (1.0/n_classes) * scipy_expm(rates[c] * Q_lg * t)
        P_avg = np.maximum(P_avg, 1e-30)
        for ai, bi in pairs:
            ll += np.log(pi_np[ai] * P_avg[ai, bi])
        if ll > best_ll:
            best_ll = ll
            best_t = t

    return best_t


def select_cherries(seqs_dict, max_pairs=None):
    """Select cherry pairs from an MSA using composition clustering + nearest neighbor.

    Args:
        seqs_dict: dict {name: aligned_sequence}
        max_pairs: max number of pairs to return

    Returns:
        list of (name1, name2) pairs
    """
    names = list(seqs_dict.keys())
    n = len(names)
    if n < 2:
        return []

    # Compute amino acid compositions
    comps = np.array([aa_composition(ungap(seqs_dict[name])) for name in names])

    # Simple k-means-like clustering
    n_clusters = max(1, min(n // 4, 10))

    # Use composition distance matrix for greedy pairing
    # (skip clustering for small MSAs)
    if n <= 50:
        # Direct pairwise p-distances
        dists = np.ones((n, n)) * 1e10
        for i in range(n):
            for j in range(i+1, n):
                d = p_distance(seqs_dict[names[i]], seqs_dict[names[j]])
                dists[i, j] = d
                dists[j, i] = d
    else:
        # Use composition distance as proxy
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.sum((comps[i] - comps[j])**2)
                dists[i, j] = d
                dists[j, i] = d
        np.fill_diagonal(dists, 1e10)

    # Greedy nearest-neighbor pairing
    paired = set()
    pairs = []
    while len(paired) < n - 1:
        # Find closest unpaired pair
        best_d = 1e10
        best_i, best_j = -1, -1
        for i in range(n):
            if i in paired:
                continue
            for j in range(i+1, n):
                if j in paired:
                    continue
                if dists[i, j] < best_d:
                    best_d = dists[i, j]
                    best_i, best_j = i, j
        if best_i < 0:
            break
        pairs.append((names[best_i], names[best_j]))
        paired.add(best_i)
        paired.add(best_j)
        if max_pairs and len(pairs) >= max_pairs:
            break

    return pairs


def geom_bin_edges(n_bins, tau_min=TAU_MIN, tau_max=TAU_MAX):
    """Return geometric-spaced bin edges and centers."""
    edges = np.geomspace(tau_min, tau_max, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean
    return edges, centers


def discretize_tau(tau, edges):
    """Map continuous tau to nearest bin index."""
    idx = np.searchsorted(edges, tau) - 1
    return max(0, min(idx, len(edges) - 2))


def count_adjacencies_pair(seq1, seq2, gamma_labels=None, n_gamma=None):
    """Count adjacency transitions in a pairwise alignment.

    Returns dict of count arrays for each adjacency type.
    Adjacencies between alignment columns are classified by (prev_type, curr_type)
    and annotated with emitted characters.

    Args:
        seq1: ancestor aligned sequence (string, with gaps)
        seq2: descendant aligned sequence (string, with gaps)
        gamma_labels: optional list of int (0..G-1 or -1), one per alignment column.
            When provided, count tensors gain a (G, ..., G, ...) gamma prefix:
            e.g. MM becomes (G, AA, AA, G, AA, AA) instead of (AA, AA, AA, AA).
            Columns with gamma_labels=-1 are treated as gamma=0 (uninformative).
        n_gamma: explicit number of gamma classes. If None, inferred from labels.
            Set this to ensure consistent tensor shapes across pairs.
    """
    # Determine column types, characters, and gamma labels
    columns = []  # (col_type, anc_char, desc_char, gamma_class)
    col_idx = 0
    for a, b in zip(seq1, seq2):
        is_a = a in AA_TO_IDX
        is_b = b in AA_TO_IDX
        g = 0  # default gamma class
        if gamma_labels is not None and col_idx < len(gamma_labels):
            g = max(0, gamma_labels[col_idx])  # -1 (uninformative) -> 0
        if is_a and is_b:
            columns.append((COL_M, AA_TO_IDX[a], AA_TO_IDX[b], g))
        elif is_b and not is_a:
            columns.append((COL_I, -1, AA_TO_IDX[b], g))
        elif is_a and not is_b:
            columns.append((COL_D, AA_TO_IDX[a], -1, g))
        # else: both gaps — skip (but still count col_idx)
        col_idx += 1

    if not columns:
        return {}

    G = 1
    if n_gamma is not None and n_gamma > 1:
        G = n_gamma
    elif gamma_labels is not None:
        G = max(1, max(max(0, g) for g in gamma_labels) + 1)

    # Tensor shapes: when G>1, each adjacency tensor gains (G, ..., G, ...) prefix
    # for the source and destination gamma classes.
    # Start/end tensors gain a single G prefix (one gamma at the boundary).
    if G > 1:
        counts = {
            'MM': np.zeros((G, AA, AA, G, AA, AA)),
            'MI': np.zeros((G, AA, AA, G, AA)),
            'MD': np.zeros((G, AA, AA, G, AA)),
            'IM': np.zeros((G, AA, G, AA, AA)),
            'II': np.zeros((G, AA, G, AA)),
            'ID': np.zeros((G, AA, G, AA)),
            'DM': np.zeros((G, AA, G, AA, AA)),
            'DD': np.zeros((G, AA, G, AA)),
            'DI': np.zeros((G, AA, G, AA)),
            'SM': np.zeros((G, AA, AA)),
            'SI': np.zeros((G, AA)),
            'SD': np.zeros((G, AA)),
            'ME': np.zeros((G, AA, AA)),
            'IE': np.zeros((G, AA)),
            'DE': np.zeros((G, AA)),
            'SE': np.zeros(()),
        }
    else:
        counts = {
            'MM': np.zeros((AA, AA, AA, AA)),
            'MI': np.zeros((AA, AA, AA)),
            'MD': np.zeros((AA, AA, AA)),
            'IM': np.zeros((AA, AA, AA)),
            'II': np.zeros((AA, AA)),
            'ID': np.zeros((AA, AA)),
            'DM': np.zeros((AA, AA, AA)),
            'DD': np.zeros((AA, AA)),
            'DI': np.zeros((AA, AA)),
            'SM': np.zeros((AA, AA)),
            'SI': np.zeros((AA,)),
            'SD': np.zeros((AA,)),
            'ME': np.zeros((AA, AA)),
            'IE': np.zeros((AA,)),
            'DE': np.zeros((AA,)),
            'SE': np.zeros(()),
        }

    # Start -> first column
    col_type, ca, cb, g = columns[0]
    if G > 1:
        if col_type == COL_M:
            counts['SM'][g, ca, cb] += 1
        elif col_type == COL_I:
            counts['SI'][g, cb] += 1
        elif col_type == COL_D:
            counts['SD'][g, ca] += 1
    else:
        if col_type == COL_M:
            counts['SM'][ca, cb] += 1
        elif col_type == COL_I:
            counts['SI'][cb] += 1
        elif col_type == COL_D:
            counts['SD'][ca] += 1

    # Consecutive columns
    for i in range(len(columns) - 1):
        t1, a1, b1, g1 = columns[i]
        t2, a2, b2, g2 = columns[i+1]

        if G > 1:
            if t1 == COL_M and t2 == COL_M:
                counts['MM'][g1, a1, b1, g2, a2, b2] += 1
            elif t1 == COL_M and t2 == COL_I:
                counts['MI'][g1, a1, b1, g2, b2] += 1
            elif t1 == COL_M and t2 == COL_D:
                counts['MD'][g1, a1, b1, g2, a2] += 1
            elif t1 == COL_I and t2 == COL_M:
                counts['IM'][g1, b1, g2, a2, b2] += 1
            elif t1 == COL_I and t2 == COL_I:
                counts['II'][g1, b1, g2, b2] += 1
            elif t1 == COL_I and t2 == COL_D:
                counts['ID'][g1, b1, g2, a2] += 1
            elif t1 == COL_D and t2 == COL_M:
                counts['DM'][g1, a1, g2, a2, b2] += 1
            elif t1 == COL_D and t2 == COL_D:
                counts['DD'][g1, a1, g2, a2] += 1
            elif t1 == COL_D and t2 == COL_I:
                counts['DI'][g1, a1, g2, b2] += 1
        else:
            if t1 == COL_M and t2 == COL_M:
                counts['MM'][a1, b1, a2, b2] += 1
            elif t1 == COL_M and t2 == COL_I:
                counts['MI'][a1, b1, b2] += 1
            elif t1 == COL_M and t2 == COL_D:
                counts['MD'][a1, b1, a2] += 1
            elif t1 == COL_I and t2 == COL_M:
                counts['IM'][b1, a2, b2] += 1
            elif t1 == COL_I and t2 == COL_I:
                counts['II'][b1, b2] += 1
            elif t1 == COL_I and t2 == COL_D:
                counts['ID'][b1, a2] += 1
            elif t1 == COL_D and t2 == COL_M:
                counts['DM'][a1, a2, b2] += 1
            elif t1 == COL_D and t2 == COL_D:
                counts['DD'][a1, a2] += 1
            elif t1 == COL_D and t2 == COL_I:
                counts['DI'][a1, b2] += 1

    # Last column -> end
    col_type, ca, cb, g = columns[-1]
    if G > 1:
        if col_type == COL_M:
            counts['ME'][g, ca, cb] += 1
        elif col_type == COL_I:
            counts['IE'][g, cb] += 1
        elif col_type == COL_D:
            counts['DE'][g, ca] += 1
    else:
        if col_type == COL_M:
            counts['ME'][ca, cb] += 1
        elif col_type == COL_I:
            counts['IE'][cb] += 1
        elif col_type == COL_D:
            counts['DE'][ca] += 1

    if G > 1:
        counts['n_gamma'] = G

    return counts


def count_singlet_bigrams(seq):
    """Count character bigrams in an ungapped sequence, including BOS/EOS."""
    ug = ungap(seq)
    if not ug:
        return np.zeros((VOCAB, VOCAB))

    B = np.zeros((VOCAB, VOCAB))
    # BOS -> first char
    B[BOS_IDX, AA_TO_IDX[ug[0]]] += 1
    # Char bigrams
    for i in range(len(ug) - 1):
        B[AA_TO_IDX[ug[i]], AA_TO_IDX[ug[i+1]]] += 1
    # Last char -> EOS
    B[AA_TO_IDX[ug[-1]], EOS_IDX] += 1

    return B


COUNT_KEYS = ['MM', 'MI', 'MD', 'IM', 'II', 'ID', 'DM', 'DD', 'DI',
              'SM', 'SI', 'SD', 'ME', 'IE', 'DE', 'SE']

COUNT_SHAPES = {
    'MM': (AA,AA,AA,AA), 'MI': (AA,AA,AA), 'MD': (AA,AA,AA),
    'IM': (AA,AA,AA), 'II': (AA,AA), 'ID': (AA,AA),
    'DM': (AA,AA,AA), 'DD': (AA,AA), 'DI': (AA,AA),
    'SM': (AA,AA), 'SI': (AA,), 'SD': (AA,),
    'ME': (AA,AA), 'IE': (AA,), 'DE': (AA,),
    'SE': (),
}


def _gamma_count_shapes(G):
    """Return count shapes with gamma prefix for G gamma classes.

    When G > 1, adjacency tensors gain (G, ..., G, ...) prefix for source
    and destination gamma classes. Start/end tensors gain a single G prefix.
    """
    if G <= 1:
        return COUNT_SHAPES
    return {
        'MM': (G,AA,AA,G,AA,AA), 'MI': (G,AA,AA,G,AA), 'MD': (G,AA,AA,G,AA),
        'IM': (G,AA,G,AA,AA), 'II': (G,AA,G,AA), 'ID': (G,AA,G,AA),
        'DM': (G,AA,G,AA,AA), 'DD': (G,AA,G,AA), 'DI': (G,AA,G,AA),
        'SM': (G,AA,AA), 'SI': (G,AA), 'SD': (G,AA),
        'ME': (G,AA,AA), 'IE': (G,AA), 'DE': (G,AA),
        'SE': (),
    }


def _file_hash(path):
    """Content hash of a file (SHA256 of decompressed content, truncated to 16 hex chars).

    Gzipped files are decompressed before hashing, so PF00001.sto.gz and
    PF00001.sto produce the same hash and won't be double-counted.
    """
    import gzip
    h = hashlib.sha256()
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def _make_metadata(n_tau, edges, centers, n_pairs=0,
                   msa_dir='', extra=None, n_gamma=1):
    """Build metadata dict for counts file."""
    meta = {
        'format': 'distill-mixdom-counts',
        'version': 2,
        'alphabet': AMINO_ACIDS,
        'n_tau_bins': n_tau,
        'tau_min': float(edges[0]),
        'tau_max': float(edges[-1]),
        'tau_spacing': 'geometric',
        'n_pairs': n_pairs,
        'msa_dir': msa_dir,
        'created': datetime.now(timezone.utc).isoformat(),
    }
    if n_gamma > 1:
        meta['n_gamma'] = n_gamma
    if extra:
        meta.update(extra)
    return meta


def _init_counts(n_tau, n_gamma=1):
    """Create zeroed count accumulators.

    Args:
        n_tau: number of tau bins
        n_gamma: number of gamma rate classes (1 = no gamma annotation)
    """
    shapes = _gamma_count_shapes(n_gamma) if n_gamma > 1 else COUNT_SHAPES
    C = {key: np.zeros((n_tau,) + shapes[key]) for key in COUNT_KEYS}
    B = np.zeros((n_tau, VOCAB, VOCAB))
    return C, B


def _save_counts(path, C, B, edges, centers, meta, seen_hashes):
    """Save counts with metadata and MSA provenance hashes.

    seen_hashes: dict {content_hash: filename} of MSAs already counted.
    Stored as two parallel newline-joined string arrays in the npz.
    """
    save_dict = {
        'tau_edges': edges,
        'tau_centers': centers,
        'B': B,
        '_metadata': np.array(json.dumps(meta)),
    }
    # Store provenance: parallel arrays of hashes and filenames
    if seen_hashes:
        hashes = sorted(seen_hashes.keys())
        save_dict['_msa_hashes'] = np.array('\n'.join(hashes))
        save_dict['_msa_names'] = np.array('\n'.join(seen_hashes[h] for h in hashes))
    for k, v in C.items():
        save_dict[f'C_{k}'] = v
    np.savez_compressed(path, **save_dict)


def _load_counts(path):
    """Load counts checkpoint. Returns (C, B, edges, centers, meta, seen_hashes)."""
    data = np.load(path, allow_pickle=True)
    edges = data['tau_edges']
    centers = data['tau_centers']
    B = data['B']
    C = {}
    for key in COUNT_KEYS:
        C[key] = data[f'C_{key}']
    meta = {}
    if '_metadata' in data:
        meta = json.loads(str(data['_metadata']))
    seen_hashes = {}
    if '_msa_hashes' in data and '_msa_names' in data:
        hashes = str(data['_msa_hashes']).split('\n')
        names = str(data['_msa_names']).split('\n')
        for h, n in zip(hashes, names):
            if h:
                seen_hashes[h] = n
    return C, B, edges, centers, meta, seen_hashes


def _msa_out_path(msa_file, suffix):
    """Compute per-file output path by replacing .sto/.sto.gz/etc with suffix."""
    base = msa_file
    for ext in ('.sto.gz', '.stockholm.gz', '.stk.gz', '.sto', '.stockholm', '.stk'):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    return base + suffix


def _count_one_msa(msa_file, n_tau, edges, gamma_labels=None, n_gamma=1):
    """Count adjacencies for a single MSA file. Returns (C, B, n_pairs).

    Args:
        msa_file: path to Stockholm MSA file
        n_tau: number of tau bins
        edges: tau bin edges
        gamma_labels: optional list of int (0..G-1 or -1), one per MSA column.
            Per-column MAP gamma rate category from fit_gamma_rates.py.
        n_gamma: number of gamma classes (for initializing count shapes).
            Must be >= max label + 1 when gamma_labels is provided.
    """
    C, B = _init_counts(n_tau, n_gamma=n_gamma)
    seqs = parse_stockholm(msa_file)
    n_pairs = 0
    if len(seqs) < 2:
        return C, B, n_pairs
    cherry_pairs = select_cherries(seqs)
    for name1, name2 in cherry_pairs:
        seq1, seq2 = seqs[name1], seqs[name2]
        pd = p_distance(seq1, seq2)
        tau_est = pdist_to_evo_time(pd)
        tau_bin = discretize_tau(tau_est, edges)
        pair_counts = count_adjacencies_pair(seq1, seq2, gamma_labels=gamma_labels,
                                              n_gamma=n_gamma if n_gamma > 1 else None)
        if not pair_counts:
            continue
        pair_counts_flip = count_adjacencies_pair(seq2, seq1, gamma_labels=gamma_labels,
                                                  n_gamma=n_gamma if n_gamma > 1 else None)
        for key in COUNT_KEYS:
            if key in pair_counts:
                C[key][tau_bin] += pair_counts[key]
            if pair_counts_flip and key in pair_counts_flip:
                C[key][tau_bin] += pair_counts_flip[key]
        B[tau_bin] += count_singlet_bigrams(seq1)
        B[tau_bin] += count_singlet_bigrams(seq2)
        n_pairs += 1
    return C, B, n_pairs


def _load_gamma_labels(gamma_labels_dir, family_id, n_gamma):
    """Load gamma labels for a family from a gamma_labels directory.

    Looks for gamma_labels_dir/FAMILY.G{n_gamma}.json (e.g. PF00001.G4.json).

    Returns list of int (one per MSA column) or None if file not found.
    """
    path = os.path.join(gamma_labels_dir, f'{family_id}.G{n_gamma}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get('labels', None)


def _family_id_from_msa(msa_file):
    """Extract family ID (e.g. PF00001) from MSA filename."""
    base = os.path.basename(msa_file)
    for ext in ('.sto.gz', '.stockholm.gz', '.stk.gz', '.sto', '.stockholm', '.stk'):
        if base.endswith(ext):
            return base[:-len(ext)]
    return base


def do_count(args):
    """Part A: Read MSAs, extract cherries, count adjacencies, save.

    Two output modes:
      --out FILE         Accumulate all MSAs into a single counts file (with checkpointing).
      --out-suffix SUF   Write a separate counts file per MSA (e.g. PF00001.counts.npz).

    Uses content hashes to deduplicate MSAs: each file's SHA256 is checked
    against the set of already-counted hashes. Files already in the checkpoint
    are skipped, even if renamed or reordered. New files are always processed.
    """
    import glob as glob_module

    msa_files = sorted(glob_module.glob(os.path.join(args.msa_dir, '*.sto')) +
                       glob_module.glob(os.path.join(args.msa_dir, '*.sto.gz')) +
                       glob_module.glob(os.path.join(args.msa_dir, '*.stockholm')) +
                       glob_module.glob(os.path.join(args.msa_dir, '*.stockholm.gz')))
    if not msa_files:
        msa_files = sorted(glob_module.glob(os.path.join(args.msa_dir, '*.stk')) +
                           glob_module.glob(os.path.join(args.msa_dir, '*.stk.gz')))
    if not msa_files:
        _log(f"No Stockholm files found in {args.msa_dir}")
        sys.exit(1)

    # Optional: restrict to a single split (train/val/test) from a splits JSON
    split_file = getattr(args, 'split_file', None)
    split_name = getattr(args, 'split', None)
    split_meta = None
    if split_file or split_name:
        if not (split_file and split_name):
            _log("ERROR: --split-file and --split must be used together")
            sys.exit(1)
        with open(os.path.expanduser(split_file)) as _sf:
            splits = json.load(_sf)
        if split_name not in splits or not isinstance(splits[split_name], list):
            _log(f"ERROR: '{split_name}' not in splits file or not a list")
            sys.exit(1)
        split_set = set(splits[split_name])
        before = len(msa_files)
        msa_files = [f for f in msa_files
                     if _family_id_from_msa(f) in split_set]
        _log(f"Split filter: --split={split_name} from {split_file} -> "
             f"{len(msa_files)}/{before} MSA files match")
        if not msa_files:
            _log("ERROR: no MSA files left after split filter")
            sys.exit(1)
        split_meta = {'split': split_name, 'split_file': split_file,
                      'split_n_families': len(split_set)}

    n_tau = args.n_tau_bins
    edges, centers = geom_bin_edges(n_tau)

    # Gamma label support: when --gamma-labels-dir is set, load per-column
    # MAP gamma rate labels and pass them to count_adjacencies_pair.
    gamma_labels_dir = getattr(args, 'gamma_labels_dir', None)
    n_gamma = getattr(args, 'n_gamma', 1) or 1
    if gamma_labels_dir:
        _log(f"Gamma labels: dir={gamma_labels_dir}, G={n_gamma}")
        if not os.path.isdir(gamma_labels_dir):
            _log(f"ERROR: gamma labels directory not found: {gamma_labels_dir}")
            sys.exit(1)

    def _get_gamma_labels_for_msa(msa_file):
        """Load gamma labels for a MSA file if gamma_labels_dir is set."""
        if not gamma_labels_dir:
            return None
        fam_id = _family_id_from_msa(msa_file)
        labels = _load_gamma_labels(gamma_labels_dir, fam_id, n_gamma)
        return labels

    per_file = getattr(args, 'out_suffix', None) is not None

    if per_file:
        # Per-file mode: one counts file per MSA
        n_total = len(msa_files)
        n_written = 0
        n_skipped = 0
        for fi, msa_file in enumerate(msa_files):
            out_path = _msa_out_path(msa_file, args.out_suffix)
            if os.path.exists(out_path) and not args.no_resume:
                n_skipped += 1
                _log_every(fi, n_total, args.log_every,
                           lambda: f"Count: {fi+1}/{n_total} ({n_skipped} skipped, {n_written} written)")
                continue
            gl = _get_gamma_labels_for_msa(msa_file)
            C, B, n_pairs = _count_one_msa(msa_file, n_tau, edges,
                                           gamma_labels=gl, n_gamma=n_gamma)
            meta = _make_metadata(n_tau, edges, centers, n_pairs,
                                  msa_dir=args.msa_dir, n_gamma=n_gamma,
                                  extra={'source_file': os.path.basename(msa_file)})
            seen = {_file_hash(msa_file): os.path.basename(msa_file)}
            _save_counts(out_path, C, B, edges, centers, meta, seen)
            n_written += 1
            _log_every(fi, n_total, args.log_every,
                       lambda: f"Count: {fi+1}/{n_total} ({n_skipped} skipped, {n_written} written)")
        _log(f"Total: {n_written} files written, {n_skipped} skipped (suffix: {args.out_suffix})")
        return

    # Combined mode: accumulate into single output file
    seen_hashes = {}  # {content_hash: filename}
    n_pairs_total = 0
    if os.path.exists(args.out) and not args.no_resume:
        try:
            C, B, edges_prev, centers_prev, meta, seen_hashes = _load_counts(args.out)
            if meta.get('n_tau_bins') == n_tau and len(edges_prev) == len(edges):
                n_pairs_total = meta.get('n_pairs', 0)
                edges, centers = edges_prev, centers_prev
                _log(f"Resuming from checkpoint: {len(seen_hashes)} MSAs already counted, "
                     f"{n_pairs_total} pairs")
            else:
                _log("Checkpoint has different tau config, starting fresh")
                C, B = _init_counts(n_tau, n_gamma=n_gamma)
                seen_hashes = {}
        except Exception as e:
            _log(f"Could not load checkpoint ({e}), starting fresh")
            C, B = _init_counts(n_tau, n_gamma=n_gamma)
            seen_hashes = {}
    else:
        C, B = _init_counts(n_tau, n_gamma=n_gamma)

    checkpoint_interval = args.checkpoint_every

    # Filter out already-counted files by content hash
    new_files = []
    n_skipped = 0
    n_total_files = len(msa_files)
    for fi, msa_file in enumerate(msa_files):
        h = _file_hash(msa_file)
        if h in seen_hashes:
            n_skipped += 1
        else:
            new_files.append((msa_file, h))
        _log_every(fi, n_total_files, args.log_every,
                   lambda: f"Hashing: {fi+1}/{n_total_files} ({n_skipped} skipped)")

    _log(f"Found {n_total_files} MSA files: {n_skipped} already counted, "
         f"{len(new_files)} new")

    n_new = len(new_files)
    for file_idx, (msa_file, h) in enumerate(new_files):
        gl = _get_gamma_labels_for_msa(msa_file)
        C_file, B_file, n_pairs = _count_one_msa(msa_file, n_tau, edges,
                                                  gamma_labels=gl, n_gamma=n_gamma)

        # Record this file as seen (even if <2 seqs — don't re-hash on resume)
        seen_hashes[h] = os.path.basename(msa_file)

        for key in COUNT_KEYS:
            C[key] += C_file[key]
        B += B_file
        n_pairs_total += n_pairs

        _log_every(file_idx, n_new, args.log_every,
                   lambda: f"Count: {file_idx+1}/{n_new} MSAs, {n_pairs_total} pairs")

        # Checkpoint
        if checkpoint_interval and (file_idx + 1) % checkpoint_interval == 0:
            meta = _make_metadata(n_tau, edges, centers, n_pairs_total,
                                  msa_dir=args.msa_dir, n_gamma=n_gamma,
                                  extra=split_meta)
            _save_counts(args.out, C, B, edges, centers, meta, seen_hashes)

    # Final save
    meta = _make_metadata(n_tau, edges, centers, n_pairs_total,
                          msa_dir=args.msa_dir, n_gamma=n_gamma,
                          extra=split_meta)
    _save_counts(args.out, C, B, edges, centers, meta, seen_hashes)
    _log(f"Total: {len(seen_hashes)} MSAs counted, {n_pairs_total} cherry pairs")
    _log(f"Saved counts to {args.out}")


# ============================================================
# Part B: Fit MixDom model to counts
# ============================================================

def _pair_ll_single_tau(dist, c_mm, c_mi, c_md, c_im, c_ii, c_id,
                        c_dm, c_dd, c_di, c_sm, c_si, c_sd,
                        c_me, c_ie, c_de, c_se):
    """Compute pair log-likelihood for one tau bin. All args are JAX arrays."""
    # --- Post-match: normalize per (a, b) context ---
    f_mm = jnp.maximum(dist['f_MM'], 1e-30)
    f_mi = jnp.maximum(dist['f_MI'], 1e-30)
    f_md = jnp.maximum(dist['f_MD'], 1e-30)
    f_me = jnp.maximum(dist['f_ME'], 1e-30)
    Z_M = jnp.maximum(f_mm.sum(axis=(2,3)) + f_mi.sum(axis=2) +
                       f_md.sum(axis=2) + f_me, 1e-30)

    ll = (jnp.sum(c_mm * jnp.log(f_mm / Z_M[:, :, None, None])) +
          jnp.sum(c_mi * jnp.log(f_mi / Z_M[:, :, None])) +
          jnp.sum(c_md * jnp.log(f_md / Z_M[:, :, None])) +
          jnp.sum(c_me * jnp.log(f_me / Z_M)))

    # --- Post-insert ---
    # Model has full context (AA,AA,...), counts have reduced context (AA,...).
    # Marginalize model over ancestor passthrough (axis 0) to match counts.
    f_im = jnp.maximum(dist['f_IM'].sum(axis=0), 1e-30)   # (AA,AA,AA)
    f_ii = jnp.maximum(dist['f_II'].sum(axis=0), 1e-30)   # (AA,AA)
    f_id = jnp.maximum(dist['f_ID'].sum(axis=0), 1e-30)   # (AA,AA)
    f_ie = jnp.maximum(dist['f_IE'].sum(axis=0), 1e-30)   # (AA,)
    Z_I = jnp.maximum(f_im.sum(axis=(1,2)) + f_ii.sum(axis=1) +
                       f_id.sum(axis=1) + f_ie, 1e-30)    # (AA,)

    ll += (jnp.sum(c_im * jnp.log(f_im / Z_I[:, None, None])) +
           jnp.sum(c_ii * jnp.log(f_ii / Z_I[:, None])) +
           jnp.sum(c_id * jnp.log(f_id / Z_I[:, None])) +
           jnp.sum(c_ie * jnp.log(f_ie / Z_I)))

    # --- Post-delete ---
    # Marginalize model over descendant passthrough (axis 1) to match counts.
    f_dm = jnp.maximum(dist['f_DM'].sum(axis=1), 1e-30)   # (AA,AA,AA)
    f_dd = jnp.maximum(dist['f_DD'].sum(axis=1), 1e-30)   # (AA,AA)
    f_di = jnp.maximum(dist['f_DI'].sum(axis=1), 1e-30)   # (AA,AA)
    f_de = jnp.maximum(dist['f_DE'].sum(axis=1), 1e-30)   # (AA,)
    Z_D = jnp.maximum(f_dm.sum(axis=(1,2)) + f_dd.sum(axis=1) +
                       f_di.sum(axis=1) + f_de, 1e-30)    # (AA,)

    ll += (jnp.sum(c_dm * jnp.log(f_dm / Z_D[:, None, None])) +
           jnp.sum(c_dd * jnp.log(f_dd / Z_D[:, None])) +
           jnp.sum(c_di * jnp.log(f_di / Z_D[:, None])) +
           jnp.sum(c_de * jnp.log(f_de / Z_D)))

    # --- Start ---
    # f_SI, f_SD are (AA,AA) but counts C_SI, C_SD are (AA,) — marginalize
    f_sm = jnp.maximum(dist['f_SM'], 1e-30)              # (AA,AA) — matches c_sm
    f_si = jnp.maximum(dist['f_SI'].sum(axis=0), 1e-30)  # (AA,) — marginalize anc
    f_sd = jnp.maximum(dist['f_SD'].sum(axis=1), 1e-30)  # (AA,) — marginalize desc
    f_se = jnp.maximum(dist['f_SE'], 1e-30)
    Z_S = jnp.maximum(f_sm.sum() + f_si.sum() + f_sd.sum() + f_se, 1e-30)

    ll += (jnp.sum(c_sm * jnp.log(f_sm / Z_S)) +
           jnp.sum(c_si * jnp.log(f_si / Z_S)) +
           jnp.sum(c_sd * jnp.log(f_sd / Z_S)) +
           jnp.sum(c_se * (jnp.log(f_se) - jnp.log(Z_S))))

    return ll


@partial(jax.jit, static_argnums=(3, 4, 5))
def log_likelihood(raw_params, counts, tau_centers, n_domains, n_classes,
                   n_classes_dynamic=1,
                   entropy_reg=0.0, dom_dirichlet=1.0,
                   dwell_pseudo=0.0):
    """Compute log-likelihood of counts under MixDom model.

    Vectorized over tau bins: precomputes eigendecompositions once,
    then vmaps distill_mixdom over all tau values.

    All data (counts, tau_centers) passed as explicit args (not closed over)
    so that JAX's persistent compilation cache can reuse compiled XLA across runs.

    If entropy_reg > 0, adds entropy_reg * H(v) to the objective where
    H(v) = -sum(v_k * log(v_k)) is the Shannon entropy of domain weights.
    This prevents domain collapse by penalizing weight concentration.

    Args:
        n_classes_dynamic: number of dynamic site classes (default 1 = static).
            When > 1, rho_inter is used in the distillation (class-marginalized
            for training on cherry counts).
    """
    params = constrain_params(raw_params)
    if n_classes_dynamic > 1:
        params['n_classes_dynamic'] = n_classes_dynamic
    n_tau = tau_centers.shape[0]

    total_ll = 0.0

    # --- Singlet log-likelihood (time-independent) ---
    B_total = counts['B'].sum(axis=0)  # (VOCAB, VOCAB)
    precomp = _precompute_mixdom(params, n_classes)
    singlet_dist = distill_mixdom(params, 0.01, n_classes, precomp=precomp)

    f_sing = jnp.maximum(singlet_dist['f_singlet'], 1e-30)
    f_sing_end = singlet_dist['f_singlet_end']
    total_out = f_sing.sum(axis=1) + f_sing_end

    P_singlet = jnp.zeros((VOCAB, VOCAB))
    P_singlet = P_singlet.at[:AA, :AA].set(f_sing / jnp.maximum(total_out[:, None], 1e-30))
    P_singlet = P_singlet.at[:AA, EOS_IDX].set(f_sing_end / jnp.maximum(total_out, 1e-30))
    f_start = singlet_dist['f_singlet_start']
    P_singlet = P_singlet.at[BOS_IDX, :AA].set(f_start / jnp.maximum(f_start.sum(), 1e-30))

    singlet_ll = jnp.sum(B_total * jnp.log(jnp.maximum(P_singlet, 1e-30)))
    total_ll = total_ll + singlet_ll

    # --- Pair log-likelihood: vmap distill_mixdom over all tau bins ---
    distill_tau = partial(distill_mixdom, params, n_classes=n_classes, precomp=precomp)
    all_dists_full = vmap(distill_tau)(tau_centers)
    # Strip non-array entries (W, W_start, W_end are nested dicts that can't be indexed)
    freq_keys = [k for k in all_dists_full if k.startswith('f_') or k == 'P_domains'
                 or k == 'T_bullet']
    all_dists = {k: all_dists_full[k] for k in freq_keys}

    # Compute per-tau LL using vmap
    def tau_ll(t_idx):
        # Slice the t_idx-th element from each dist array
        dist_t = {k: v[t_idx] for k, v in all_dists.items()}
        return _pair_ll_single_tau(
            dist_t,
            counts['C_MM'][t_idx], counts['C_MI'][t_idx], counts['C_MD'][t_idx],
            counts['C_IM'][t_idx], counts['C_II'][t_idx], counts['C_ID'][t_idx],
            counts['C_DM'][t_idx], counts['C_DD'][t_idx], counts['C_DI'][t_idx],
            counts['C_SM'][t_idx], counts['C_SI'][t_idx], counts['C_SD'][t_idx],
            counts['C_ME'][t_idx], counts['C_IE'][t_idx], counts['C_DE'][t_idx],
            counts['C_SE'][t_idx])

    # Use lax.map for sequential but traced execution (vmap would parallelize
    # but memory scales linearly with n_tau; for 32 bins this is fine)
    pair_lls = vmap(tau_ll)(jnp.arange(n_tau))
    total_ll = total_ll + jnp.sum(pair_lls)

    # Regularization on domain weights
    v = params['v']

    # Entropy regularization: lambda * H(v) (always computed; zero when reg=0)
    H_v = -jnp.sum(v * jnp.log(v + 1e-30))
    total_ll = total_ll + entropy_reg * H_v

    # Dirichlet prior: (alpha-1) * sum(log v_k) — same form as BW's --dom-dirichlet
    # alpha=1.0 is flat (no effect), alpha>1 resists collapse
    total_ll = total_ll + (dom_dirichlet - 1.0) * jnp.sum(jnp.log(v + 1e-30))

    # Dynamic class regularization
    if params.get('rho_inter') is not None:
        rho_inter = params['rho_inter']
        # Dwell-time pseudocount: penalize class switching rate
        # Total leaving rate for class d is rho_inter * (1 - w_d),
        # so sum over d gives rho_inter * (D - 1)
        if params.get('class_weights') is not None:
            D_cls = params['class_weights'].shape[-1]
        else:
            D_cls = 2
        total_ll = total_ll - dwell_pseudo * rho_inter * (D_cls - 1)

    return total_ll


def _flatten_params(raw_params):
    """Flatten param dict to a single numpy vector + metadata for unflattening."""
    keys = sorted(raw_params.keys())
    parts = []
    shapes = {}
    for k in keys:
        arr = np.asarray(raw_params[k]).ravel()
        shapes[k] = np.asarray(raw_params[k]).shape
        parts.append(arr)
    return np.concatenate(parts), keys, shapes


def _unflatten_params(vec, keys, shapes):
    """Unflatten a numpy vector back to a JAX param dict."""
    params = {}
    offset = 0
    for k in keys:
        size = int(np.prod(shapes[k])) if shapes[k] else 1
        arr = vec[offset:offset + size]
        params[k] = jnp.array(arr.reshape(shapes[k]) if shapes[k] else arr.item())
        offset += size
    return params


@partial(jax.jit, static_argnums=(2, 3))
def singlet_log_likelihood(raw_params, B_total, n_domains, n_classes):
    """Compute singlet-only log-likelihood from B (bigram) counts.

    Only depends on: v, pi, lam/mu ratios (kappas), lam0/mu0 ratio (kappa0), r.
    Does NOT depend on absolute rates, exchangeabilities, or alpha_gamma.
    """
    params = constrain_params(raw_params)
    N = params['lam'].shape[0]
    v = params['v']
    r = params['r']
    pis = params['pi']
    kappas = params['lam'] / params['mu']
    kappa0 = params['lam0'] / params['mu0']

    # Singlet null closure
    z0_sing = jnp.sum(v * (1 - kappas))
    null_closure_sing = 1.0 / jnp.maximum(1 - kappa0 * z0_sing, 1e-30)
    kappa0_eff = kappa0 * null_closure_sing
    v_nonempty = v * kappas
    end_factor = (1 - kappa0) * null_closure_sing

    p_sing = r + (1 - r) * kappas
    T_sing = jnp.diag(p_sing) + (1 - p_sing)[:, None] * kappa0_eff * v_nonempty[None, :]
    G_sing = jnp.linalg.inv(jnp.eye(N) - T_sing)

    sing_start = kappa0_eff * v_nonempty
    sing_end = (1 - p_sing) * end_factor

    L_sing = sing_start @ G_sing
    R_sing = G_sing @ sing_end

    W_sing = L_sing[:, None] * T_sing * R_sing[None, :]
    f_singlet = jnp.einsum('ij,ia,jb->ab', W_sing, pis, pis)
    f_singlet_start = jnp.einsum('i,ia->a', sing_start * R_sing, pis)
    f_singlet_end = jnp.einsum('i,ia->a', L_sing * sing_end, pis)

    # Build singlet transition matrix P_singlet (VOCAB x VOCAB)
    total_out = f_singlet.sum(axis=1) + f_singlet_end

    P_singlet = jnp.zeros((VOCAB, VOCAB))
    P_singlet = P_singlet.at[:AA, :AA].set(
        f_singlet / jnp.maximum(total_out[:, None], 1e-30))
    P_singlet = P_singlet.at[:AA, EOS_IDX].set(
        f_singlet_end / jnp.maximum(total_out, 1e-30))
    P_singlet = P_singlet.at[BOS_IDX, :AA].set(
        f_singlet_start / jnp.maximum(f_singlet_start.sum(), 1e-30))

    return jnp.sum(B_total * jnp.log(jnp.maximum(P_singlet, 1e-30)))


# Parameters that affect the singlet stationary distribution
_SINGLET_PARAM_KEYS = {'log_v', 'log_pi', 'log_lam', 'log_mu', 'logit_r',
                       'log_lam0', 'log_mu0'}


def _do_singlet_init(raw_params, B_total, n_domains, n_classes,
                     n_steps, lr, args):
    """Stage 1: Fit singlet-only parameters using B counts.

    Optimizes: log_v, log_pi, log_lam, log_mu, logit_r, log_lam0, log_mu0
    Freezes: log_S, log_alpha_gamma
    """
    _log("=== Stage 1: Singlet-only initialization ===")

    # Value-and-gradient w.r.t. singlet params only
    _val_grad_singlet = jax.value_and_grad(singlet_log_likelihood, argnums=0)

    def val_grad_fn(p):
        return _val_grad_singlet(p, B_total, n_domains, n_classes)

    # JIT compile
    _log("JIT-compiling singlet value_and_grad...")
    t_jit_start = time.monotonic()
    ll, grads = val_grad_fn(raw_params)
    jax.block_until_ready(ll)
    t_jit = time.monotonic() - t_jit_start
    _log(f"Singlet JIT done in {t_jit:.1f}s (initial singlet LL={float(ll):.4f})")

    # Adam optimizer (only update singlet params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    singlet_keys = _SINGLET_PARAM_KEYS & set(raw_params.keys())
    adam_state = {k: {'m': jnp.zeros_like(raw_params[k]),
                      'v_hat': jnp.zeros_like(raw_params[k])}
                 for k in singlet_keys}

    best_ll = ll
    best_params = {k: v.copy() for k, v in raw_params.items()}
    current_lr = lr

    # Apply initial gradients (step 0)
    for k in singlet_keys:
        g = grads[k]
        if jnp.any(jnp.isnan(g)):
            continue
        adam_state[k]['m'] = (1 - beta1) * g
        adam_state[k]['v_hat'] = (1 - beta2) * g**2
        m_hat = adam_state[k]['m'] / (1 - beta1)
        v_hat = adam_state[k]['v_hat'] / (1 - beta2)
        raw_params[k] = raw_params[k] + current_lr * m_hat / (jnp.sqrt(v_hat) + eps)

    log_every = args.log_every
    if log_every <= 0:
        log_every = max(1, n_steps // 50)

    for step in range(1, n_steps):
        try:
            t_step = time.monotonic()
            ll, grads = val_grad_fn(raw_params)

            if jnp.isnan(ll) or jnp.isinf(ll):
                current_lr *= 0.5
                raw_params = {k: v.copy() for k, v in best_params.items()}
                _log(f"Singlet step {step}/{n_steps}: LL=invalid, "
                     f"halving lr={current_lr:.1e}")
                continue

            if ll > best_ll:
                best_ll = ll
                best_params = {k: v.copy() for k, v in raw_params.items()}

            dt = time.monotonic() - t_step
            if step == 1 or step == n_steps - 1 or (step + 1) % log_every == 0:
                _log(f"Singlet step {step}/{n_steps}: LL={float(ll):.4f}  "
                     f"best={float(best_ll):.4f}  lr={current_lr:.1e}  "
                     f"({dt:.2f}s/step)")

            for k in singlet_keys:
                g = grads[k]
                if jnp.any(jnp.isnan(g)):
                    continue
                adam_state[k]['m'] = beta1 * adam_state[k]['m'] + (1 - beta1) * g
                adam_state[k]['v_hat'] = (beta2 * adam_state[k]['v_hat']
                                         + (1 - beta2) * g**2)
                m_hat = adam_state[k]['m'] / (1 - beta1**(step+1))
                v_hat = adam_state[k]['v_hat'] / (1 - beta2**(step+1))
                raw_params[k] = (raw_params[k]
                                 + current_lr * m_hat / (jnp.sqrt(v_hat) + eps))

        except Exception as e:
            current_lr *= 0.5
            raw_params = {k: v.copy() for k, v in best_params.items()}
            _log(f"Singlet step {step}/{n_steps}: Error: {e}, "
                 f"halving lr={current_lr:.1e}")
            continue

    _log(f"=== Stage 1 complete: best singlet LL={float(best_ll):.4f} ===")
    return best_params


def do_fit(args):
    """Part B: Fit MixDom2 to cherry counts.

    Parameters are initialised via the shared
    :func:`tkfmixdom.jax.models.mixdom_init.init_mixdom2_params_from_args`
    helper (so a seed-N fresh maraschino fit and a seed-N train_pfam
    fresh init produce bit-identical starting params).

    The cherry-count log-likelihood is computed by
    :func:`tkfmixdom.jax.distill.maraschino_fit.cherry_log_likelihood`,
    which builds the collapsed Pair HMM transition matrix via
    :func:`tkfmixdom.jax.models.mixdom.build_nested_trans` and scores
    each tau-binned adjacency tensor under the per-(d, f) class-mixture
    emissions defined in ``tkf/maraschino.tex``.

    Saves a train_pfam-compatible .npz containing the linear-space
    MixDom2 params (so downstream ``train_pfam.py --eval-only --checkpoint``
    works on the output unchanged).
    """
    import optax

    from tkfmixdom.jax.core.protein import rate_matrix_lg
    from tkfmixdom.jax.distill.maraschino_fit import (
        cherry_log_likelihood, linear_to_raw, raw_to_linear,
        save_checkpoint, load_checkpoint_linear, COUNT_KEYS)
    from tkfmixdom.jax.models.mixdom_init import init_mixdom2_params_from_args

    _log(f"Loading counts from {args.counts}")
    data = np.load(args.counts, allow_pickle=True)
    tau_centers = jnp.array(data['tau_centers'])
    n_tau = int(tau_centers.shape[0])

    counts = {f"C_{k}": jnp.array(data[f"C_{k}"]) for k in COUNT_KEYS}
    counts['B'] = jnp.array(data['B'])

    # ---- --banded-frag-init umbrella: pre-resolve dependent flags ----
    # See tkf/substitution-mstep.tex sec:mstep-tied-pi for the FragStart/
    # FragMid/FragEnd parameterisation. The umbrella locks down everything
    # the structural constraints require so users don't have to assemble
    # five flags by hand.
    if getattr(args, 'banded_frag_init', False):
        if args.n_frag != 3:
            raise ValueError(
                f"--banded-frag-init requires --n-frag 3 (got {args.n_frag})")
        if not (0.0 < args.p_ext < 1.0):
            raise ValueError(
                f"--banded-frag-init requires --p-ext in (0, 1) (got {args.p_ext})")
        if args.classdist_init in ('auto',):
            args.classdist_init = 'fragchar'
            _log("  --banded-frag-init: auto-set --classdist-init=fragchar")
        elif args.classdist_init != 'fragchar':
            raise ValueError(
                "--banded-frag-init expects --classdist-init=fragchar "
                f"(got {args.classdist_init}); fragchar tying is the whole "
                "point of the FragStart/FragMid/FragEnd parameterisation.")
        if args.n_classes == 0:
            args.n_classes = 3
            _log("  --banded-frag-init: auto-set --n-classes=3")
        elif args.n_classes != 3:
            raise ValueError(
                "--banded-frag-init requires --n-classes 3 to match the "
                f"3 fragchars (got {args.n_classes})")
        if not args.freeze_fragdist:
            args.freeze_fragdist = True
            _log("  --banded-frag-init: auto-set --freeze-fragdist=True "
                 "(fragdist[d,0]=1 is structurally pinned)")
        if args.freeze_offdiag_ext:
            raise ValueError(
                "--banded-frag-init and --freeze-offdiag-ext are mutually "
                "exclusive: banded mode imposes its own ext mask.")

    # ---- --rescale-class-S-only umbrella: imply constituent flags ----
    if getattr(args, 'rescale_class_S_only', False):
        if not args.freeze_class_S_shape:
            args.freeze_class_S_shape = True
            _log("  --rescale-class-S-only: auto-set --freeze-class-S-shape")
        if not args.freeze_class_pi:
            args.freeze_class_pi = True
            _log("  --rescale-class-S-only: auto-set --freeze-class-pi")

    n_dom = args.n_domains
    n_frag = args.n_frag
    n_classes = args.n_classes if args.n_classes > 0 else max(n_dom, n_frag)
    args.n_classes = n_classes  # canonicalise for downstream
    Q_lg, pi_lg = rate_matrix_lg()

    _log(f"Fitting MixDom2: D={n_dom} domains, F={n_frag} fragments, "
         f"C={n_classes} site classes, seed={args.seed}")
    _log(f"  {n_tau} tau bins, optimizer=adam, lr={args.lr}, steps={args.n_steps}")

    # ---- Initialise linear-space params ----
    if args.init:
        _log(f"Warm-starting from {args.init}")
        linear_params = load_checkpoint_linear(args.init)
        # Sanity-check shapes against requested model
        D_init = int(linear_params['dom_ins'].shape[0])
        F_init = int(linear_params['frag_weights'].shape[1])
        C_init = int(linear_params.get('class_pis', np.zeros((1, AA))).shape[0]) \
            if 'class_pis' in linear_params else 1
        if D_init != n_dom or F_init != n_frag:
            raise ValueError(
                f"Warm-start checkpoint has D={D_init}, F={F_init}; "
                f"requested D={n_dom}, F={n_frag}")
        if C_init != n_classes and 'class_pis' in linear_params:
            raise ValueError(
                f"Warm-start checkpoint has C={C_init} site classes; "
                f"requested C={n_classes}")
        if 'class_pis' not in linear_params and n_classes > 1:
            raise ValueError(
                "--init checkpoint lacks class_pis/class_S_exch/classdist; "
                "cannot warm-start a MixDom2 fit. Use a train_pfam --n-classes>1 "
                "checkpoint or run a fresh fit.")
    else:
        linear_params = init_mixdom2_params_from_args(
            args, n_dom, n_frag, Q_lg, pi_lg, log_fn=_log)

    # Snapshot the freeze-init values (for stop_gradient routing)
    freeze_init: dict = {}
    if args.freeze_fragdist:
        freeze_init['frag_weights'] = jnp.asarray(linear_params['frag_weights'],
                                                  dtype=jnp.float32)
        _log("  Freeze: frag_weights (held at init)")
    if getattr(args, 'freeze_class_S_shape', False):
        if n_classes <= 1:
            raise ValueError(
                "--freeze-class-S-shape requires --n-classes > 1")
        freeze_init['class_S_exch_shape'] = jnp.asarray(
            linear_params['class_S_exch'], dtype=jnp.float32)
        _log("  Freeze: class_S_exch shape (only per-class log_class_sigma "
             "varies; S_c = exp(σ_c) · S_init_c)")
    if getattr(args, 'freeze_class_pi', False):
        if n_classes <= 1:
            raise ValueError(
                "--freeze-class-pi requires --n-classes > 1")
        freeze_init['class_pis'] = jnp.asarray(
            linear_params['class_pis'], dtype=jnp.float32)
        _log("  Freeze: class_pis (held at init)")
    if args.freeze_classdist:
        if n_classes <= 1:
            raise ValueError("--freeze-classdist requires n_classes > 1")
        freeze_init['classdist'] = jnp.asarray(linear_params['classdist'],
                                               dtype=jnp.float32)
        _log("  Freeze: classdist (held at init)")
    if args.freeze_offdiag_ext:
        # We freeze ext_rates by saving the init full matrix; raw_to_linear
        # will substitute it. To allow diagonal entries to vary while
        # off-diagonal stays put, we instead handle this by mixing the
        # softmax output with a mask before assembling ext_rates.
        # (Implemented below in the per-step constraint.)
        if n_frag <= 1:
            _log("  Freeze: --freeze-offdiag-ext is a no-op when n_frag=1")
        else:
            _log("  Freeze: ext_rates off-diagonal (intra-fragment Markov coupling) "
                 "held at init; diagonal + termination remain free")

    raw_params = linear_to_raw(linear_params, n_dom, n_frag, n_classes)

    # When freeze_class_S_shape is set, raw_to_linear consumes a new key
    # `log_class_sigma` of shape (C,). Initialise σ_c = 1 ⇔ log σ = 0 so
    # the starting Q matches the linear init exactly.
    if 'class_S_exch_shape' in freeze_init:
        raw_params['log_class_sigma'] = jnp.zeros(n_classes, dtype=jnp.float32)

    # If freeze-offdiag-ext, store the init off-diagonal pattern:
    # ext_init[d, f, g] for f != g is locked to its initial value; diagonal
    # entry and termination column are free. We fold this into raw_to_linear
    # by passing through a mixin function.
    ext_init_full = None
    if args.freeze_offdiag_ext and n_frag > 1:
        # Reconstruct the F+1-column "ext_full" softmax target so the
        # off-diagonal entries match the init exactly. The diagonal and
        # termination column are taken from the live softmax output.
        ext_init = jnp.asarray(linear_params['ext_rates'], dtype=jnp.float32)
        notext_init = 1.0 - ext_init.sum(axis=-1)
        ext_init_full = jnp.concatenate([ext_init, notext_init[:, :, None]], axis=-1)

    # If banded-frag-init, build the structural-zero mask: an (F, F+1)
    # array with 1.0 on entries that are STRUCTURALLY FREE and 0.0 on
    # entries that are pinned at zero. The free pattern (n_frag=3) is:
    #   row 0 (FragStart): [_, free, free, term-free]
    #   row 1 (FragMid):   [_, free, free, _      ]   (no termination)
    #   row 2 (FragEnd):   [_, _,    _,    term-free]   (no extension)
    # where '_' means pinned at zero. After softmax, structural zeros
    # are dropped and the surviving entries are renormalised per row so
    # each row sums to 1 (proper transition distribution).
    banded_free_mask_full = None
    if getattr(args, 'banded_frag_init', False):
        from tkfmixdom.jax.train.restricted_mstep import banded_3fc_ext_mask
        ext_mask, term_mask = banded_3fc_ext_mask()  # (3, 3) bool, (3,) bool
        # Free mask: True iff this entry is allowed to take a nonzero value.
        free_2d = ext_mask.astype(jnp.float32)  # (F, F)
        free_term = term_mask.astype(jnp.float32)[:, None]  # (F, 1)
        banded_free_mask_full = jnp.concatenate(
            [free_2d, free_term], axis=-1)  # (F, F+1)
        banded_free_mask_full = jnp.broadcast_to(
            banded_free_mask_full[None, :, :], (n_dom, n_frag, n_frag + 1))

    # ---- Loss closure ----
    def _materialise_linear(raw_p):
        """Constrain raw -> linear, applying any freeze masks."""
        out = raw_to_linear(raw_p, n_dom, n_frag, n_classes,
                            freeze_init=freeze_init)
        if ext_init_full is not None:
            # Recompute ext_rates while pinning off-diagonal entries to init.
            ext_full_live = jax.nn.softmax(raw_p['logit_ext_rates'], axis=-1)
            # Build a mask: 1 for off-diagonal among the first F columns,
            # 0 for diagonal and the termination column.
            offdiag_mask = (1.0 - jnp.eye(n_frag))
            offdiag_mask_full = jnp.concatenate(
                [offdiag_mask, jnp.zeros((n_frag, 1))], axis=-1)
            offdiag_mask_full = jnp.broadcast_to(
                offdiag_mask_full[None, :, :], (n_dom, n_frag, n_frag + 1))
            # Mix: where mask=1 use init (frozen), else use live
            ext_full = (offdiag_mask_full * ext_init_full
                        + (1.0 - offdiag_mask_full) * ext_full_live)
            # Renormalise so rows sum to 1 again (necessary because the
            # diagonal+termination columns alone sum to less than 1)
            ext_full = ext_full / jnp.maximum(ext_full.sum(axis=-1, keepdims=True),
                                              1e-30)
            out['ext_rates'] = ext_full[..., :n_frag]
        elif banded_free_mask_full is not None:
            # Banded mode: zero structural-zero entries from the live
            # softmax output, then renormalise so each row sums to 1.
            # The surviving entries (3 in row 0, 2 in row 1, 1 in row 2)
            # carry all the gradient signal; structural zeros remain at
            # exactly 0 across all Adam steps.
            ext_full_live = jax.nn.softmax(raw_p['logit_ext_rates'], axis=-1)
            ext_full = ext_full_live * banded_free_mask_full
            ext_full = ext_full / jnp.maximum(
                ext_full.sum(axis=-1, keepdims=True), 1e-30)
            out['ext_rates'] = ext_full[..., :n_frag]
        return out

    def loss_fn(raw_p):
        linp = _materialise_linear(raw_p)
        return -cherry_log_likelihood(linp, counts, tau_centers,
                                      n_dom, n_frag, n_classes)

    val_grad = jax.jit(jax.value_and_grad(loss_fn))

    # ---- Optimiser ----
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(raw_params)

    _log("JIT-compiling value_and_grad(cherry_log_likelihood)...")
    t_jit = time.monotonic()
    loss, grads = val_grad(raw_params)
    loss.block_until_ready()
    _log(f"JIT compilation done in {time.monotonic() - t_jit:.1f}s "
         f"(initial LL={-float(loss):.4f})")

    best_ll = -float(loss)
    best_raw = jax.tree_util.tree_map(lambda x: jnp.asarray(x), raw_params)

    log_every = max(1, int(getattr(args, 'log_every_step', 10)))
    for step in range(args.n_steps):
        t_step = time.monotonic()
        loss, grads = val_grad(raw_params)
        ll = -float(loss)
        if not (np.isnan(ll) or np.isinf(ll)) and ll > best_ll:
            best_ll = ll
            best_raw = jax.tree_util.tree_map(lambda x: jnp.asarray(x), raw_params)
        updates, opt_state = optimizer.update(grads, opt_state, raw_params)
        raw_params = optax.apply_updates(raw_params, updates)
        if step == 0 or step == args.n_steps - 1 or (step + 1) % log_every == 0:
            dt = time.monotonic() - t_step
            _log(f"Step {step + 1}/{args.n_steps}: LL={ll:.4f}  "
                 f"best={best_ll:.4f}  ({dt:.2f}s/step)")

    # ---- Save best params in train_pfam .npz layout ----
    final_linear = _materialise_linear(best_raw)
    final_t = float(np.median(np.asarray(tau_centers)))
    config = {
        'trainer': 'maraschino fit (MixDom2 cherry-count)',
        'n_dom': int(n_dom), 'n_frag': int(n_frag), 'n_classes': int(n_classes),
        'seed': int(args.seed),
        'lr': float(args.lr), 'n_steps': int(args.n_steps),
        'init_ins': float(args.init_ins), 'init_del': float(args.init_del),
        'class_pi_init': args.class_pi_init,
        'classdist_init': args.classdist_init,
        'classdist_noise_frac': float(args.classdist_noise_frac),
        'freeze_fragdist': bool(args.freeze_fragdist),
        'freeze_classdist': bool(args.freeze_classdist),
        'freeze_offdiag_ext': bool(args.freeze_offdiag_ext),
        'banded_frag_init': bool(getattr(args, 'banded_frag_init', False)),
        'p_ext': float(getattr(args, 'p_ext', 0.6)),
        'rescale_class_S_only': bool(
            getattr(args, 'rescale_class_S_only', False)),
        'freeze_class_S_shape': bool(
            getattr(args, 'freeze_class_S_shape', False)),
        'freeze_class_pi': bool(getattr(args, 'freeze_class_pi', False)),
        'best_ll': float(best_ll),
    }
    save_checkpoint(args.out, final_linear, n_dom=n_dom, n_frag=n_frag,
                    n_classes=n_classes, t=final_t, em_iter=int(args.n_steps),
                    config=config)
    _log(f"Best LL: {best_ll:.4f}")
    _log(f"Saved parameters to {args.out}")


# ============================================================
# Part C: Generate distillations
# ============================================================

# _normalize_freqs is imported from tkfmixdom.jax.distill.maraschino above.


def _fmt(x, precision):
    """Format a float to given decimal precision, stripping trailing zeros."""
    s = f"{float(x):.{precision}f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s

# `_wfst_to_machineboss_json` and `_singlet_to_machineboss_json` were
# previously inlined here as duplicates of the implementations in
# tkfmixdom.jax.distill.maraschino. Both share the same WFST/HMM state
# numbering scheme (S=0; M(a,b)=1+a*AA+b; I(x,y)=1+AA*AA+x*AA+y;
# D(x,y)=1+2*AA*AA+x*AA+y; E=1+3*AA*AA), so dedup is a straight
# import — only cosmetic differences (state-id labels "M(A,B)" vs
# "M_AB", `round(...)` vs the local `_fmt(...)` trailing-zero strip,
# JSON indenting) survived in the duplicates and they are not
# observably different to consumers of the produced WFST.
from tkfmixdom.jax.distill.maraschino import (
    _wfst_to_machineboss_json,
    _singlet_to_machineboss_json,
)

def _write_machineboss_json(machine, path, precision=6):
    """Write Machine Boss JSON with one transition per line for compactness."""
    # Custom serializer: compact but readable
    with open(path, 'w') as f:
        f.write('{"state": [\n')
        for si, state in enumerate(machine['state']):
            parts = []
            if 'n' in state:
                parts.append(f'"n": {state["n"]}')
            if 'id' in state:
                parts.append(f'"id": {json.dumps(state["id"])}')
            if 'trans' in state:
                trans_lines = []
                for t in state['trans']:
                    tparts = [f'"to": {t["to"]}']
                    if 'in' in t:
                        tparts.append(f'"in": {json.dumps(t["in"])}')
                    if 'out' in t:
                        tparts.append(f'"out": {json.dumps(t["out"])}')
                    if 'weight' in t and t['weight'] != 1:
                        tparts.append(f'"weight": {_fmt(t["weight"], precision)}')
                    trans_lines.append('{' + ', '.join(tparts) + '}')
                trans_str = ',\n    '.join(trans_lines)
                parts.append(f'"trans": [\n    {trans_str}]')
            comma = ',' if si < len(machine['state']) - 1 else ''
            f.write('  {' + ', '.join(parts) + '}' + comma + '\n')
        f.write(']}\n')


def _is_mixdom2_ckpt(data) -> bool:
    """Detect a train_pfam-style MixDom2 checkpoint by the presence of
    `main_ins` (cross-cutting key with maraschino fit output) and the
    absence of the legacy `log_lam0` raw param."""
    return ("main_ins" in data.files and "log_lam0" not in data.files)


def _do_distill_mixdom2(args):
    """MixDom2 distill path: build chi via build_nested_trans, compute
    full-context adjacency frequencies under the per-(d, f) class-mixture
    emissions, normalise to order-1 Pair WFST + Singlet HMM transitions,
    and write Machine-Boss JSON.

    Reads a train_pfam-style checkpoint (also produced by `maraschino fit`):
        main_ins, main_del, dom_ins[D], dom_del[D],
        dom_weights[D], frag_weights[D, F], ext_rates[D, F, F],
        class_pis[C, A], class_S_exch[C, A, A], classdist[D, F, C].
    """
    from tkfmixdom.jax.distill.maraschino_fit import (
        load_checkpoint_linear, distill_mixdom2_probs,
    )

    _log(f"Loading MixDom2 parameters from {args.params}")
    data = np.load(args.params, allow_pickle=True)
    linear_params = load_checkpoint_linear(args.params)
    n_dom = int(linear_params["dom_ins"].shape[0])
    n_frag = int(linear_params["frag_weights"].shape[1])
    n_classes = (int(linear_params["class_pis"].shape[0])
                 if "class_pis" in linear_params else 1)
    if n_classes <= 1:
        raise NotImplementedError(
            "MixDom2 distill requires n_classes >= 2 (per-class GTRs in the "
            "ckpt). For MixDom1-only checkpoints use the legacy distill path "
            "with a maraschino-format (log_lam0/...) ckpt.")

    # Move arrays to JAX
    linear_params = {k: jnp.asarray(v) for k, v in linear_params.items()
                     if not isinstance(v, int)}
    linear_params["n_classes"] = n_classes

    precision = args.precision
    tau_values = [float(t.strip()) for t in args.tau.split(',')]
    out_base = args.out
    if out_base.endswith('.json'):
        out_base = out_base[:-5]

    # Save model params once
    npz_path = out_base + '.params.npz'
    np.savez(npz_path,
             tau_values=np.array(tau_values),
             n_dom=n_dom, n_frag=n_frag, n_classes=n_classes,
             main_ins=np.array(linear_params['main_ins']),
             main_del=np.array(linear_params['main_del']),
             dom_ins=np.array(linear_params['dom_ins']),
             dom_del=np.array(linear_params['dom_del']),
             dom_weights=np.array(linear_params['dom_weights']),
             frag_weights=np.array(linear_params['frag_weights']),
             ext_rates=np.array(linear_params['ext_rates']),
             class_pis=np.array(linear_params['class_pis']),
             class_S_exch=np.array(linear_params['class_S_exch']),
             classdist=np.array(linear_params['classdist']),
             amino_acids=AMINO_ACIDS)
    _log(f"  Model params -> {npz_path}")

    for ti, tau in enumerate(tau_values):
        _log(f"Computing MixDom2 distillation at tau={tau} "
             f"(D={n_dom}, F={n_frag}, C={n_classes}, precision={precision})")
        probs = distill_mixdom2_probs(linear_params, float(tau),
                                      n_dom, n_frag, n_classes)
        # Convert JAX arrays to numpy for downstream MB writers
        probs = {k: (np.asarray(v) if hasattr(v, "shape") else float(v))
                 for k, v in probs.items()}

        tag = '' if len(tau_values) == 1 else f'.t{_fmt(tau, 4)}'

        wfst_path = out_base + tag + '.wfst.json'
        wfst_machine = _wfst_to_machineboss_json(probs, precision)
        _write_machineboss_json(wfst_machine, wfst_path, precision)
        n_wfst_trans = sum(len(s.get('trans', [])) for s in wfst_machine['state'])
        _log(f"  Pair WFST: {len(wfst_machine['state'])} states, "
             f"{n_wfst_trans} transitions -> {wfst_path}")

        hmm_path = out_base + tag + '.hmm.json'
        hmm_machine = _singlet_to_machineboss_json(probs, precision)
        _write_machineboss_json(hmm_machine, hmm_path, precision)
        n_hmm_trans = sum(len(s.get('trans', [])) for s in hmm_machine['state'])
        _log(f"  Singlet HMM: {len(hmm_machine['state'])} states, "
             f"{n_hmm_trans} transitions -> {hmm_path}")


def do_distill(args):
    """Part C: order-1 distillation (Machine-Boss-format Pair WFST + Singlet HMM).

    Auto-detects checkpoint format:
      - Legacy maraschino raw params (log_lam0 / log_mu0 / ...) → MixDom1 path
        via constrain_params + distill_mixdom from distill/maraschino.py.
      - train_pfam-style MixDom2 ckpt (main_ins / dom_ins / class_pis / ...) →
        MixDom2 path via build_nested_trans + class-mixture emissions
        (tkfmixdom.jax.distill.maraschino_fit.distill_mixdom2_probs).
    """
    data = np.load(args.params, allow_pickle=True)
    if _is_mixdom2_ckpt(data):
        _do_distill_mixdom2(args)
        return

    # ---- Legacy MixDom1 path (unchanged from pre-MixDom2 maraschino) ----
    _log(f"Loading parameters from {args.params}")
    n_domains = int(data['n_domains'])
    n_classes = int(data['n_classes'])
    n_classes_dynamic = int(data['n_classes_dynamic']) if 'n_classes_dynamic' in data else 1

    raw_params = {}
    for key in ['log_lam0', 'log_mu0', 'log_lam', 'log_mu', 'logit_r',
                'log_v', 'log_pi', 'log_S', 'log_alpha_gamma']:
        raw_params[key] = jnp.array(data[key])
    # Optional dynamic class switching rate
    if 'log_rho_inter' in data:
        raw_params['log_rho_inter'] = jnp.array(data['log_rho_inter'])
    elif 'log_S_star' in data:
        # Legacy: convert S_star matrix to scalar rho_inter
        raw_params['log_S_star'] = jnp.array(data['log_S_star'])
    # Optional gamma_class and class weights
    for opt_key in ['log_gamma_class',
                    'logit_class_weights', 'log_pi_classes']:
        if opt_key in data:
            raw_params[opt_key] = jnp.array(data[opt_key])

    params = constrain_params(raw_params)
    if n_classes_dynamic > 1:
        params['n_classes_dynamic'] = n_classes_dynamic
    precision = args.precision

    # Parse tau values (comma-separated)
    tau_values = [float(t.strip()) for t in args.tau.split(',')]

    # Determine output base
    out_base = args.out
    if out_base.endswith('.json'):
        out_base = out_base[:-5]

    # Save model params once
    npz_path = out_base + '.params.npz'
    np.savez(npz_path,
             tau_values=np.array(tau_values),
             n_domains=n_domains, n_classes=n_classes,
             lam0=np.array(params['lam0']), mu0=np.array(params['mu0']),
             lam=np.array(params['lam']), mu=np.array(params['mu']),
             r=np.array(params['r']), v=np.array(params['v']),
             alpha_gamma=np.array(params['alpha_gamma']),
             amino_acids=AMINO_ACIDS)
    _log(f"  Model params -> {npz_path}")

    for ti, tau in enumerate(tau_values):
        _log(f"Computing distillation at tau={tau} "
             f"(N={n_domains} domains, M={n_classes} classes, precision={precision})")

        dist = distill_mixdom(params, tau, n_classes)
        probs = _normalize_freqs(dist)

        # Use tau in filename when multiple values
        if len(tau_values) == 1:
            tag = ''
        else:
            tag = f'.t{_fmt(tau, 4)}'

        # Write pair WFST
        wfst_path = out_base + tag + '.wfst.json'
        wfst_machine = _wfst_to_machineboss_json(probs, precision)
        _write_machineboss_json(wfst_machine, wfst_path, precision)
        n_wfst_trans = sum(len(s.get('trans', [])) for s in wfst_machine['state'])
        _log(f"  Pair WFST: {len(wfst_machine['state'])} states, "
             f"{n_wfst_trans} transitions -> {wfst_path}")

        # Write singlet HMM
        hmm_path = out_base + tag + '.hmm.json'
        hmm_machine = _singlet_to_machineboss_json(probs, precision)
        _write_machineboss_json(hmm_machine, hmm_path, precision)
        n_hmm_trans = sum(len(s.get('trans', [])) for s in hmm_machine['state'])
        _log(f"  Singlet HMM: {len(hmm_machine['state'])} states, "
             f"{n_hmm_trans} transitions -> {hmm_path}")


# ============================================================
# Part D: Fetch Pfam alignments
# ============================================================

def _pfam_family_list(min_seqs=10, max_seqs=5000, min_cols=50):
    """Fetch list of Pfam families from InterPro API.
    Returns list of dicts with 'accession', 'name'.

    Note: InterPro API no longer includes sequence counts in the list
    endpoint, so all families of type 'domain' or 'family' are returned.
    Size filtering happens implicitly when alignments are downloaded.
    """
    import urllib.request
    import urllib.parse

    families = []
    url = "https://www.ebi.ac.uk/interpro/api/entry/pfam/?page_size=200&format=json"

    _log("Fetching Pfam family index...")
    pages_fetched = 0
    while url:
        try:
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            _log(f"  API error: {e}")
            break

        for entry in data.get('results', []):
            acc = entry.get('metadata', {}).get('accession', '')
            name = entry.get('metadata', {}).get('name', '')
            if acc:
                families.append({
                    'accession': acc,
                    'name': name,
                })

        pages_fetched += 1
        url = data.get('next')
        if pages_fetched % 10 == 0:
            _log(f"  Scanned {pages_fetched} pages, {len(families)} families so far")

    _log(f"Found {len(families)} Pfam families")
    return families


def _download_pfam_alignment(accession, out_dir, aln_type='seed'):
    """Download a Pfam Stockholm alignment.
    aln_type: 'seed' (curated) or 'full' (all sequences).
    Tries InterPro wwwapi first (gzipped), falls back to api endpoint.
    """
    import urllib.request
    import urllib.error
    import gzip

    out_path = os.path.join(out_dir, f"{accession}.sto")
    if os.path.exists(out_path):
        return out_path  # already downloaded

    # Primary: InterPro wwwapi (returns gzipped Stockholm)
    url = (f"https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/{accession}/"
           f"?annotation=alignment:{aln_type}&download")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
        # Try to decompress (response is usually gzipped)
        try:
            content = gzip.decompress(raw)
        except Exception:
            content = raw
        with open(out_path, 'wb') as f:
            f.write(content)
        return out_path
    except Exception:
        pass

    # Fallback: older api endpoint
    url2 = (f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{accession}/"
            f"?annotation=alignment:{aln_type}&download")
    try:
        req = urllib.request.Request(url2)
        req.add_header('Accept', 'text/plain')
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
        with open(out_path, 'wb') as f:
            f.write(content)
        return out_path
    except Exception:
        return None


def _select_random_families(families, n, seed=42):
    """Deterministic pseudorandom selection of n families.
    Uses SHA256(seed || accession) for deterministic ordering.
    """
    def sort_key(fam):
        h = hashlib.sha256(f"{seed}:{fam['accession']}".encode()).hexdigest()
        return h
    ranked = sorted(families, key=sort_key)
    return ranked[:n]


def do_sum(args):
    """Sum multiple counts .npz files into one."""
    if not args.inputs:
        _log("No input files specified")
        sys.exit(1)

    C_total = None
    B_total = None
    edges = centers = None
    n_tau = None
    n_pairs_total = 0
    all_hashes = {}

    for fi, path in enumerate(args.inputs):
        C, B, e, c, meta, seen = _load_counts(path)
        file_n_tau = meta.get('n_tau_bins', len(c))
        if C_total is None:
            n_tau = file_n_tau
            edges, centers = e, c
            C_total = C
            B_total = B
        else:
            if file_n_tau != n_tau or len(e) != len(edges):
                _log(f"Skipping {path}: incompatible tau config "
                     f"(n_tau={file_n_tau}, expected {n_tau})")
                continue
            for key in COUNT_KEYS:
                C_total[key] += C[key]
            B_total += B
        n_pairs_total += meta.get('n_pairs', 0)
        all_hashes.update(seen)
        _log_every(fi, len(args.inputs), args.log_every,
                   lambda: f"Sum: {fi+1}/{len(args.inputs)} files, {n_pairs_total} pairs")

    if C_total is None:
        _log("No valid counts files loaded")
        sys.exit(1)

    meta = _make_metadata(n_tau, edges, centers, n_pairs_total,
                          extra={'summed_from': len(args.inputs)})
    _save_counts(args.out, C_total, B_total, edges, centers, meta, all_hashes)
    _log(f"Summed {len(args.inputs)} files ({n_pairs_total} pairs) -> {args.out}")


def do_fetch(args):
    """Part D: Fetch Pfam alignments."""
    from tkfmixdom.jax.util.bio_datasets import resolve_data_dir
    resolved = str(resolve_data_dir("pfam", local_fallback=args.out_dir))
    if resolved != args.out_dir:
        _log(f"bio-datasets: {args.out_dir} → {resolved}")
        args.out_dir = resolved
    os.makedirs(args.out_dir, exist_ok=True)

    if args.families:
        # Explicit family list
        accs = [a.strip() for a in args.families.split(',')]
        _log(f"Downloading {len(accs)} specified families")
        downloaded = 0
        for ai, acc in enumerate(accs):
            path = _download_pfam_alignment(acc, args.out_dir, args.aln_type)
            if path:
                downloaded += 1
            else:
                _log(f"  Failed: {acc}")
            if (ai + 1) % max(1, len(accs) // 20) == 0 or ai == len(accs) - 1:
                _log(f"Download: {ai+1}/{len(accs)} ({downloaded} ok)")
            time.sleep(0.5)  # rate limit
        _log(f"Downloaded {downloaded}/{len(accs)} alignments to {args.out_dir}")
    elif args.random:
        # Pseudorandom selection
        families = _pfam_family_list(args.min_seqs, args.max_seqs)
        selected = _select_random_families(families, args.random, args.seed)
        _log(f"Selected {len(selected)} families (seed={args.seed})")
        downloaded = 0
        for fi, fam in enumerate(selected):
            path = _download_pfam_alignment(fam['accession'], args.out_dir, args.aln_type)
            if path:
                downloaded += 1
            else:
                _log(f"  Failed: {fam['accession']} ({fam['name']})")
            if (fi + 1) % max(1, len(selected) // 20) == 0 or fi == len(selected) - 1:
                _log(f"Download: {fi+1}/{len(selected)} ({downloaded} ok)")
            time.sleep(0.5)  # rate limit
        _log(f"Downloaded {downloaded}/{len(selected)} alignments to {args.out_dir}")
    else:
        _log("Specify --families or --random N")
        sys.exit(1)


# ============================================================
# Main
# ============================================================

def main():
    # Shared logging arguments via parent parser
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--log-file', type=str, default=None,
                        help='Append log messages to this file (in addition to stderr)')
    common.add_argument('--log-every', type=int, default=0,
                        help='Log progress every N items (0 = auto-choose)')
    common.add_argument('--jax-cache-dir', type=str, default=None,
                        help='Directory for persistent JAX compilation cache')
    common.add_argument('--bio-datasets', type=str, default=None, metavar='DIR',
                        help='Path to bio-datasets repo (default: $BIO_DATASETS_HOME or ~/bio-datasets)')

    parser = argparse.ArgumentParser(
        description='CherryML-like training of reversible MixDom model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest='mode', required=True)

    # Mode A: count
    p_count = sub.add_parser('count', help='Extract cherries and count adjacencies',
                             parents=[common])
    p_count.add_argument('--msa-dir', required=True, help='Directory of Stockholm MSA files')
    count_out = p_count.add_mutually_exclusive_group(required=True)
    count_out.add_argument('--out', help='Output .npz file for combined counts')
    count_out.add_argument('--out-suffix', help='Per-file output: replace .sto/.sto.gz with this suffix (e.g. .counts.npz)')
    p_count.add_argument('--n-tau-bins', type=int, default=32, help='Number of tau bins')
    p_count.add_argument('--checkpoint-every', type=int, default=100,
                         help='Save checkpoint every N MSA files (0 to disable)')
    p_count.add_argument('--no-resume', action='store_true',
                         help='Do not resume from existing checkpoint')
    p_count.add_argument('--gamma-labels-dir', type=str, default=None,
                         help='Directory of per-family gamma label JSON files '
                         '(from fit_gamma_rates.py). When set, count tensors gain '
                         'gamma class prefixes (G, ..., G, ...).')
    p_count.add_argument('--n-gamma', type=int, default=1,
                         help='Number of gamma rate classes for gamma-annotated counts '
                         '(must match G in gamma label filenames, e.g. 4 for *.G4.json). '
                         'Default: 1 (no gamma annotation).')
    p_count.add_argument('--split-file', type=str, default=None,
                         help='Path to a JSON file with train/val/test split lists '
                         '(e.g. ~/bio-datasets/data/pfam/seed/splits/v1.json). '
                         'Use with --split to count only the listed families.')
    p_count.add_argument('--split', type=str, default=None,
                         choices=['train', 'val', 'test'],
                         help='Which split to count (requires --split-file).')

    # Mode B: fit
    # Mode B: fit (MixDom2 cherry-count fitting)
    p_fit = sub.add_parser('fit', help='Fit MixDom2 model to cherry counts',
                           parents=[common])
    p_fit.add_argument('--counts', required=True, help='Input counts .npz file')
    p_fit.add_argument('--out', required=True, help='Output .npz file for parameters '
                       '(train_pfam-compatible layout)')
    # Model shape (matches train_pfam.py flags so both trainers can be
    # initialised from the same fresh init via mixdom_init.py)
    p_fit.add_argument('--n-domains', '--n-dom', dest='n_domains', type=int, default=3,
                       help='Number of MixDom2 domain types')
    p_fit.add_argument('--n-frag', type=int, default=1,
                       help='Number of fragment types per domain (MixDom2 nfrag)')
    p_fit.add_argument('--n-classes', type=int, default=0,
                       help='Number of MixDom2 site classes (each with its own '
                       'GTR (S_exch, pi)). 0 -> max(n_dom, n_frag).')
    p_fit.add_argument('--init-ins', type=float, default=0.01,
                       help='Initial main_ins/dom_ins TKF rate (matches train_pfam)')
    p_fit.add_argument('--init-del', type=float, default=0.01,
                       help='Initial main_del/dom_del TKF rate (matches train_pfam)')
    # Mixdom_init flags (delegated to init_mixdom2_params_from_args)
    p_fit.add_argument('--estimate-subst', action='store_true',
                       help='Populate dom_pis/dom_S_exch (used only for the '
                       '--class-pis-from-dom-pis override and for legacy ckpts)')
    p_fit.add_argument('--class-pi-init', type=str, default='lg_noisy',
                       choices=['lg_noisy', 'c10', 'c10_topN', 'c20'],
                       help='Per-class equilibrium initialisation strategy '
                       '(matches train_pfam --class-pi-init)')
    p_fit.add_argument('--pi-init-noise-frac', type=float, default=0.2,
                       help='Dirichlet noise fraction for class_pi_init=lg_noisy')
    p_fit.add_argument('--classdist-init', type=str, default='auto',
                       choices=['auto', 'identity'],
                       help='Initial classdist scheme (auto = uniform/c10-weighted; '
                       'identity = c=d, requires n_classes==n_dom)')
    p_fit.add_argument('--class-pis-from-dom-pis', action='store_true',
                       help='Mirror dom_pis into class_pis (requires --estimate-subst '
                       'and n_classes==n_dom)')
    p_fit.add_argument('--classdist-noise-frac', type=float, default=0.0,
                       help='Symmetry-breaking Dirichlet perturbation on classdist init')
    # Optimisation
    p_fit.add_argument('--lr', type=float, default=1e-2,
                       help='Adam learning rate (default 1e-2 — note larger than the '
                       'pre-MixDom2 default 1e-3)')
    p_fit.add_argument('--n-steps', type=int, default=2000, help='Number of Adam steps')
    p_fit.add_argument('--seed', type=int, default=42, help='Random seed')
    p_fit.add_argument('--init', type=str, default=None,
                       help='Initialise from existing train_pfam-style params .npz '
                       '(warm start; takes precedence over --seed init)')
    # Parameter freezing (per maraschino_to_mixdom2_brief.md §4)
    p_fit.add_argument('--freeze-fragdist', action='store_true',
                       help='Hold frag_weights[d, f] at its initial value '
                       '(stop_gradient on the underlying logits)')
    p_fit.add_argument('--freeze-classdist', action='store_true',
                       help='Hold classdist[d, f, c] at its initial value '
                       '(stop_gradient on the underlying logits). Mirrors '
                       'train_pfam --freeze-classdist.')
    p_fit.add_argument('--freeze-offdiag-ext', action='store_true',
                       help='Constrain ext_rates[d, f1, f2] for f1 != f2 to its '
                       'initial value (zeroes the off-diagonal gradient); '
                       'leaves the diagonal self-extension and termination free. '
                       'Recovers the diagonal-ext (no intra-fragment Markov coupling) '
                       'baseline. Cannot be combined with --freeze-fragdist on the same '
                       'fragment matrix entries.')
    # ----- Banded 3-fragchar mode (FragStart / FragMid / FragEnd) -----
    p_fit.add_argument('--banded-frag-init', action='store_true',
                       help='Umbrella mode for the 3-fragchar FragStart/FragMid/FragEnd '
                       'parameterisation (see tkf/substitution-mstep.tex). Implies: '
                       '--n-frag 3, --classdist-init fragchar, --n-classes 3, '
                       '--freeze-fragdist (fragdist[d,0]=1 pinned), and a structural '
                       'mask on ext_rates so only the banded-allowed entries are free '
                       '(row 0: x_01, x_02, term=1-p; row 1: x_11, x_12; row 2: term=1). '
                       'Initial values are set by --p-ext.')
    p_fit.add_argument('--p-ext', type=float, default=0.6,
                       help='Initial extension probability for --banded-frag-init '
                       '(default: 0.6). Drives the banded init via banded_3fc_init().')
    # ----- Rate-rescale-only mode (per-class scalar σ_c on (S^c, π^c)) -----
    p_fit.add_argument('--rescale-class-S-only', action='store_true',
                       help='Umbrella: hold (S^c, π^c) shape FIXED at init; only '
                       'a per-class scalar σ_c is free. Q^c = σ_c · S_init^c · '
                       'diag(π_init^c). Implies --freeze-class-S-shape and '
                       '--freeze-class-pi. Mirrors `train_pfam --subst-mode '
                       'rescaling-rates` (closed-form there; gradient on log σ '
                       'here). Useful for warm-starts where you trust the '
                       'substitution chemistry but want only rate calibration.')
    p_fit.add_argument('--freeze-class-S-shape', action='store_true',
                       help='Constituent of --rescale-class-S-only: hold '
                       'class_S_exch shape at init; let only a per-class log '
                       'scalar log_class_sigma vary (so S_c = e^σ · S_init_c).')
    p_fit.add_argument('--freeze-class-pi', action='store_true',
                       help='Constituent of --rescale-class-S-only: hold '
                       'class_pis at init (no gradient on log_class_pis).')
    p_fit.add_argument('--log-every-step', type=int, default=10,
                       help='Print loss every K Adam steps')

    # Mode C: distill
    p_distill = sub.add_parser('distill', help='Generate order-1 distillation (Machine Boss JSON)',
                               parents=[common])
    p_distill.add_argument('--params', required=True, help='Input parameters .npz file')
    p_distill.add_argument('--tau', type=str, required=True,
                           help='Evolutionary time(s), comma-separated (e.g. 0.1,0.5,1.0)')
    p_distill.add_argument('--out', required=True,
                           help='Output base path (writes .wfst.json, .hmm.json, .params.npz)')
    p_distill.add_argument('--precision', type=int, default=6,
                           help='Decimal digits for weights (default: 6)')

    # Mode D: sum
    p_sum = sub.add_parser('sum', help='Sum multiple counts .npz files into one',
                           parents=[common])
    p_sum.add_argument('--out', required=True, help='Output .npz file for summed counts')
    p_sum.add_argument('inputs', nargs='+', help='Input counts .npz files')

    # Mode E: fetch
    p_fetch = sub.add_parser('fetch', help='Fetch Pfam Stockholm alignments',
                             parents=[common])
    p_fetch.add_argument('--out-dir', required=True, help='Output directory for .sto files')
    p_fetch.add_argument('--families', type=str, default=None,
                         help='Comma-separated Pfam accessions (e.g. PF00001,PF00002)')
    p_fetch.add_argument('--random', type=int, default=None,
                         help='Pseudorandomly select N families')
    p_fetch.add_argument('--seed', type=int, default=42, help='Random seed for selection')
    p_fetch.add_argument('--min-seqs', type=int, default=10,
                         help='Minimum sequences per family (default: 10)')
    p_fetch.add_argument('--max-seqs', type=int, default=5000,
                         help='Maximum sequences per family (default: 5000)')
    p_fetch.add_argument('--aln-type', choices=['seed', 'full'], default='seed',
                         help='Alignment type: seed (curated) or full (default: seed)')

    args = parser.parse_args()
    if args.jax_cache_dir:
        jax.config.update("jax_compilation_cache_dir", args.jax_cache_dir)
        os.makedirs(args.jax_cache_dir, exist_ok=True)
    _setup_logging(args.log_file)

    # Apply bio-datasets override if provided
    if args.bio_datasets:
        from tkfmixdom.jax.util.bio_datasets import set_bio_datasets_home
        set_bio_datasets_home(args.bio_datasets)

    if args.mode == 'count':
        do_count(args)
    elif args.mode == 'fit':
        do_fit(args)
    elif args.mode == 'distill':
        do_distill(args)
    elif args.mode == 'sum':
        do_sum(args)
    elif args.mode == 'fetch':
        do_fetch(args)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
