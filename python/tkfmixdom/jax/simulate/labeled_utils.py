"""Shared utilities for the MixDom2 / MixDom1 fully-labeled simulators.

Provides helpers used by both Simulator A (collapsed pair-HMM categorical
sampler) and Simulator B (hierarchical Gillespie). The two simulators
return (states, anc_chars, desc_chars, t) trajectories; this module
converts those into the precompiled-pair tuple format consumed by
``train_pfam._process_pairs_batched`` and friends, and exposes a few
small helpers used to compute the per-class CTMC machinery.

Precompiled-pair format (matches ``train_pfam.py``):
    (x_int, y_int, states, anc_chars, desc_chars, t_est)

where:
  - ``x_int``  : (Lx,)   ancestor as int array (M and D positions)
  - ``y_int``  : (Ly,)   descendant as int array (M and I positions)
  - ``states`` : list[int] of M=1, I=2, D=3 codes (alignment columns)
  - ``anc_chars`` : list[int] residues for M and D states (in column order)
  - ``desc_chars``: list[int] residues for M and I states (in column order)
  - ``t_est``  : float branch length (per-pair; the same t the simulator used)
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from ..core.params import S, M, I, D, E
from ..core.ctmc import transition_matrix


def trajectory_to_pair_tuple(states, anc_chars, desc_chars, t):
    """Convert a (states, anc_chars, desc_chars, t) trajectory to the
    precompiled-pair tuple format expected by ``_process_pairs_batched``.

    Args:
        states:     1D array-like of M/I/D state codes (one per alignment column)
        anc_chars:  1D array-like of residues at M and D positions, in order
        desc_chars: 1D array-like of residues at M and I positions, in order
        t:          branch length used to generate this pair

    Returns:
        (x_int, y_int, states_list, anc_list, desc_list, t_float)
    """
    states_arr = np.asarray(states, dtype=np.int32)
    anc_chars = np.asarray(anc_chars, dtype=np.int32)
    desc_chars = np.asarray(desc_chars, dtype=np.int32)

    # Build the ancestor sequence x: residues at M+D positions.
    # The alignment-order anc_chars *is* x (in column order), since x positions
    # advance only on M/D. Same logic for y on M/I.
    # x_int and y_int are simply anc_chars and desc_chars cast appropriately.
    x_int = anc_chars.astype(np.int32)
    y_int = desc_chars.astype(np.int32)

    return (x_int, y_int,
            list(int(s) for s in states_arr),
            list(int(c) for c in anc_chars),
            list(int(c) for c in desc_chars),
            float(t))


def class_Q_from_pi_and_S(pi_c, S_c):
    """Build CTMC rate matrix Q for a single class from (pi, S_exch).

    Reversible parameterisation:
        Q[i, j] = S[i, j] * pi[j]   (i != j)
        Q[i, i] = -sum_{j != i} Q[i, j]

    Args:
        pi_c: (A,) equilibrium distribution
        S_c:  (A, A) symmetric exchangeability matrix (zero diagonal)

    Returns:
        Q_c: (A, A) rate matrix
    """
    pi_c = np.asarray(pi_c, dtype=np.float64)
    S_c = np.asarray(S_c, dtype=np.float64)
    A = pi_c.shape[0]
    Q = S_c * pi_c[None, :]
    Q = Q - np.diag(np.diag(Q))           # zero diagonal
    Q = Q - np.diag(Q.sum(axis=1))        # row-zero
    return Q


def per_class_transition_matrices(class_pis, class_S_exch, t):
    """Compute exp(Q_c · t) for every class.

    Args:
        class_pis:   (C, A)  equilibrium distributions
        class_S_exch:(C, A, A) exchangeability matrices
        t:           scalar branch length

    Returns:
        Pmats: (C, A, A) transition probability matrices
        Qmats: (C, A, A) rate matrices
    """
    class_pis = np.asarray(class_pis, dtype=np.float64)
    class_S_exch = np.asarray(class_S_exch, dtype=np.float64)
    C, A = class_pis.shape

    Qmats = np.zeros((C, A, A), dtype=np.float64)
    Pmats = np.zeros((C, A, A), dtype=np.float64)
    for c in range(C):
        Qmats[c] = class_Q_from_pi_and_S(class_pis[c], class_S_exch[c])
        Pmats[c] = np.array(
            transition_matrix(jnp.array(Qmats[c]), t))
    return Pmats, Qmats


def normalise_classdist(classdist):
    """Ensure classdist sums to 1 across the class axis (axis=-1) per (d, f)."""
    classdist = np.asarray(classdist, dtype=np.float64)
    sums = classdist.sum(axis=-1, keepdims=True)
    return classdist / np.maximum(sums, 1e-30)


def default_mixdom2_params(n_dom=2, n_frag=1, n_classes=1, A=20,
                            seed=0,
                            main_ins=0.06, main_del=0.08,
                            ext_diag=0.3,
                            class_S_base=None, class_pi_base=None,
                            class_rate_mults=None,
                            classdist_skew=None):
    """Build a self-consistent ground-truth MixDom2 params dict for testing.

    The defaults yield a small, well-behaved model with non-trivial
    structure (different per-domain rates, slightly skewed dom_weights,
    Dirichlet-like classdist) suitable for parameter recovery tests.

    Args:
        n_dom, n_frag, n_classes: model size
        A: alphabet size (default 20 for protein)
        seed: PRNG seed for parameter draws
        main_ins, main_del: top-level indel rates
        ext_diag: diagonal extension probability for the (D, F, F) ext_rates
        class_S_base: (A, A) base exchangeability matrix (defaults to LG if None)
        class_pi_base: (A,) base equilibrium (defaults to LG if None)
        class_rate_mults: optional (n_classes,) array of per-class rate multipliers
        classdist_skew: optional (n_dom, n_frag, n_classes) skewing weights;
            if None, a uniform Dirichlet sample is drawn

    Returns:
        params: dict ready to pass to ``build_nested_trans`` /
                _process_pairs_batched_*.
    """
    rng = np.random.RandomState(seed)

    if class_S_base is None or class_pi_base is None:
        from ..core.protein import rate_matrix_lg
        Q_lg, pi_lg = rate_matrix_lg()
        Q_lg = np.asarray(Q_lg, dtype=np.float64)
        pi_lg = np.asarray(pi_lg, dtype=np.float64)
        # Recover S from Q = S * pi (off-diag): S = Q / pi[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            S_base = Q_lg / np.maximum(pi_lg[None, :], 1e-30)
        S_base = (S_base + S_base.T) / 2.0
        np.fill_diagonal(S_base, 0.0)
        if class_S_base is None:
            class_S_base = S_base
        if class_pi_base is None:
            class_pi_base = pi_lg

    # Per-domain rates: small jitter around (main_ins, main_del). Enforce
    # λ_d < μ_d (sub-critical) so the per-domain stationary length is finite
    # — the TKF91 model assumption used by build_nested_trans. The cap is
    # set to κ_d ≤ 0.95 (real-Pfam fits sit ≈ 0.93, so this leaves the
    # simulator one realistic notch above empirical κ); the previous
    # κ ≤ 0.8 was a hidden bias that made the simulator easier than
    # reality and weakened the recovery-test coverage.
    KAPPA_MAX = 0.95
    dom_ins = np.zeros(n_dom, dtype=np.float64)
    dom_del = np.zeros(n_dom, dtype=np.float64)
    for d in range(n_dom):
        ins_d = main_ins * (0.7 + 0.6 * rng.rand())
        del_d = max(main_del * (0.7 + 0.6 * rng.rand()), ins_d / KAPPA_MAX)
        dom_ins[d] = ins_d
        dom_del[d] = del_d

    # dom_weights: skewed (Dirichlet samples)
    dw_raw = rng.gamma(2.0, 1.0, size=n_dom)
    dom_weights = dw_raw / dw_raw.sum()

    # frag_weights per domain
    frag_weights = np.zeros((n_dom, n_frag), dtype=np.float64)
    for d in range(n_dom):
        fw_raw = rng.gamma(2.0, 1.0, size=n_frag)
        frag_weights[d] = fw_raw / fw_raw.sum()

    # ext_rates: (D, F, F). Default = ext_diag * I + small off-diagonal noise.
    ext_rates = np.zeros((n_dom, n_frag, n_frag), dtype=np.float64)
    for d in range(n_dom):
        for f in range(n_frag):
            row = np.full(n_frag, 0.0)
            row[f] = ext_diag
            ext_rates[d, f] = row

    params = {
        'main_ins': float(main_ins),
        'main_del': float(main_del),
        'dom_ins': dom_ins,
        'dom_del': dom_del,
        'dom_weights': dom_weights,
        'frag_weights': frag_weights,
        'ext_rates': ext_rates,
    }

    # Classdist machinery (only when n_classes > 1; MixDom1 omits these).
    if n_classes > 1:
        if class_rate_mults is None:
            # Mild rate spread (e.g. 0.5x to 2x) — distinguishable but realistic
            class_rate_mults = np.exp(rng.uniform(-0.5, 0.7, size=n_classes))
        class_pis = np.tile(class_pi_base[None, :], (n_classes, 1)).astype(np.float64)
        class_S_exch = np.zeros((n_classes, A, A), dtype=np.float64)
        for c in range(n_classes):
            S_c = class_S_base.copy() * float(class_rate_mults[c])
            np.fill_diagonal(S_c, 0.0)
            class_S_exch[c] = S_c

        if classdist_skew is None:
            classdist = np.zeros((n_dom, n_frag, n_classes), dtype=np.float64)
            for d in range(n_dom):
                for f in range(n_frag):
                    cd_raw = rng.gamma(1.5, 1.0, size=n_classes)
                    classdist[d, f] = cd_raw / cd_raw.sum()
        else:
            classdist = normalise_classdist(classdist_skew)

        params['n_classes'] = int(n_classes)
        params['classdist'] = classdist
        params['class_pis'] = class_pis
        params['class_S_exch'] = class_S_exch
    else:
        params['n_classes'] = 1
    return params
