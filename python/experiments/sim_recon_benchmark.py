#!/usr/bin/env python3
"""Simulation-reconstruction benchmark for TreeVarAnc.

Implements the benchmark specified in misc/simulation-design.md and
misc/simulation_reconstruction_test_plan.tex.

Simulates full ancestral MSAs under TKF91/TKF92 + LG08, masks ancestral
characters, reconstructs with Felsenstein and TreeVarAnc, reports accuracy,
log score, and detailed wall-clock timings.

Usage:
    cd python && uv run python experiments/sim_recon_benchmark.py [--budget 5m|10m|1h]
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Force CPU for reproducible timing
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from tkfmixdom.jax.core.params import S, M, I, D, E, tkf91_trans, tkf92_trans
from tkfmixdom.jax.core.bdi import tkf_kappa
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.util.io import TreeNode, parse_newick, AMINO_ACIDS
from tkfmixdom.jax.tree.tree_varanc import (
    tree_varanc,
    build_tkf91_branch_wfst,
    build_tkf91_root_wfst,
    infer_internal_presence,
    name_internal_nodes,
    _felsenstein_column_likelihood,
    NEG_INF, BOS_I, BOS_D,
)
from tkfmixdom.jax.tree.ancestor import marginal_ancestor_all_columns_jax

import jax.random as jr


# ======================================================================
# Configuration
# ======================================================================

PARAMS_PATH = Path(__file__).parent / "tkf92_fitted_params.json"

A = 20  # amino acid alphabet


@dataclass
class BenchmarkConfig:
    """Configuration for one benchmark cell."""
    sim_model: str         # 'tkf91' or 'tkf92'
    rec_method: str        # 'felsenstein' or 'tree_varanc'
    n_leaves: int          # tree size
    n_cols_target: int     # target alignment width (actual varies by sim)
    branch_regime: str     # 'short', 'moderate', 'long'
    n_replicates: int = 5
    seed: int = 42
    varanc_max_iter: int = 20
    varanc_tol: float = 1e-6


@dataclass
class ReplicateResult:
    """Results from one replicate."""
    root_accuracy: float = 0.0
    log_score: float = 0.0
    root_length: int = 0
    n_cols: int = 0
    n_sweeps: int = 0
    converged: bool = False
    final_residual: float = 0.0
    elbo_start: float = 0.0
    elbo_end: float = 0.0
    # Timings (seconds)
    t_sim: float = 0.0
    t_preprocess: float = 0.0
    t_factor_build: float = 0.0
    t_optimize: float = 0.0
    t_total_recon: float = 0.0
    t_posterior_extract: float = 0.0
    n_factors: int = 0
    n_latent_vars: int = 0


@dataclass
class CellResult:
    """Aggregated results for one benchmark cell."""
    config: dict = field(default_factory=dict)
    replicates: list = field(default_factory=list)
    feasibility_class: str = ''

    def summary(self):
        """Compute summary statistics."""
        if not self.replicates:
            return {}
        reps = self.replicates
        n = len(reps)

        def _stats(values):
            arr = np.array(values, dtype=np.float64)
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr, ddof=1)) if n > 1 else 0.0,
                'se': float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr)),
            }

        return {
            'accuracy': _stats([r.root_accuracy for r in reps]),
            'log_score': _stats([r.log_score for r in reps]),
            't_total_recon': _stats([r.t_total_recon for r in reps]),
            't_optimize': _stats([r.t_optimize for r in reps]),
            't_sim': _stats([r.t_sim for r in reps]),
            'n_sweeps': _stats([r.n_sweeps for r in reps]),
            'n_cols': _stats([r.n_cols for r in reps]),
            'root_length': _stats([r.root_length for r in reps]),
            'n_replicates': n,
        }


# ======================================================================
# Tree generation
# ======================================================================

def make_random_tree(n_leaves, branch_regime, rng):
    """Generate a random binary tree with n_leaves leaves.

    branch_regime: 'short', 'moderate', 'long' controls branch lengths.
    """
    scale = {'short': 0.05, 'moderate': 0.15, 'long': 0.4}[branch_regime]

    # Build by random joining
    nodes = [TreeNode(name=f'L{i}') for i in range(n_leaves)]
    counter = 0

    while len(nodes) > 1:
        # Pick two random nodes to join
        idx = rng.choice(len(nodes), size=2, replace=False)
        left, right = nodes[idx[0]], nodes[idx[1]]

        parent = TreeNode(name=f'int_{counter}')
        bl_left = max(0.001, rng.exponential(scale))
        bl_right = max(0.001, rng.exponential(scale))
        left.branch_length = bl_left
        right.branch_length = bl_right
        parent.children = [left, right]
        left.parent = parent
        right.parent = parent
        counter += 1

        # Remove joined nodes, add parent
        remaining = [n for i, n in enumerate(nodes) if i not in idx]
        remaining.append(parent)
        nodes = remaining

    root = nodes[0]
    root.name = 'root'
    return root


# ======================================================================
# TKF92 WFST factories (conditional form, for reconstruction)
# ======================================================================

def build_tkf92_branch_wfst_cond(ins_rate, del_rate, ext, Q, pi, t):
    """Build TKF92 branch WFST in conditional form (no kappa in branch).

    Uses the formula from simulation_reconstruction_test_plan.tex:
      T_TKF92(S, j) = T_TKF91_cond(S, j)
      T_TKF92(i, j) = (r * delta_ij + (1-r) * T_TKF91_cond(i,j)) / p_ext  for i,j in {M,I,D}
      T_TKF92(i, E) = (1-r) * T_TKF91_cond(i, E) / (1 - p_ext)            for i in {M,I,D}

    where r = ext, kappa = ins_rate/del_rate, p_ext = r + (1-r)*kappa.
    """
    from tkfmixdom.jax.core.params import tkf91_trans_cond

    if ins_rate >= del_rate:
        raise ValueError(f"Requires ins_rate < del_rate, got {ins_rate} >= {del_rate}")

    kappa = float(ins_rate / del_rate)
    p_ext = ext + (1.0 - ext) * kappa
    r = ext

    tau91_cond = np.asarray(tkf91_trans_cond(ins_rate, del_rate, t))

    # Build TKF92 conditional matrix
    tau92_cond = np.zeros((5, 5), dtype=np.float64)

    # Start row: same as TKF91
    tau92_cond[S, :] = tau91_cond[S, :]

    # Interior rows: mixture of self-loop and TKF91
    for i in [M, I, D]:
        for j in [M, I, D]:
            delta = 1.0 if i == j else 0.0
            tau92_cond[i, j] = (r * delta + (1.0 - r) * tau91_cond[i, j]) / p_ext
        tau92_cond[i, E] = (1.0 - r) * tau91_cond[i, E] / (1.0 - p_ext)

    # Clamp near-zero numerical artifacts (gamma can go slightly negative
    # when lambda ≈ mu at short branch lengths)
    tau92_cond = np.maximum(tau92_cond, 0.0)

    # Convert to WFST tensors (same structure as build_tkf91_branch_wfst)
    P = np.asarray(transition_matrix(Q, t))
    sl = lambda x: np.log(np.maximum(x, 1e-300))

    log_P = sl(P)
    tau = tau92_cond

    log_p_mm = np.broadcast_to(sl(tau[M, M]) + log_P[None, None, :, :], (A, A, A, A)).copy()
    log_p_mi = np.broadcast_to(sl(tau[M, I]) + sl(pi)[None, None, :], (A, A, A)).copy()
    log_p_md = np.full((A, A, A), sl(tau[M, D]))
    log_p_me = np.full((A, A), sl(tau[M, E]))

    log_p_im = np.broadcast_to(sl(tau[I, M]) + log_P[None, None, :, :], (A, A, A, A)).copy()
    log_p_ii = np.broadcast_to(sl(tau[I, I]) + sl(pi)[None, None, :], (A, A, A)).copy()
    log_p_id = np.full((A, A, A), sl(tau[I, D]))
    log_p_ie = np.full((A, A), sl(tau[I, E]))

    log_p_dm = np.broadcast_to(sl(tau[D, M]) + log_P[None, None, :, :], (A, A, A, A)).copy()
    log_p_dd = np.full((A, A, A), sl(tau[D, D]))
    log_p_di = np.broadcast_to(sl(tau[D, I]) + sl(pi)[None, None, :], (A, A, A)).copy()
    log_p_de = np.full((A, A), sl(tau[D, E]))

    log_p_sm = (sl(tau[S, M]) + log_P).copy()
    log_p_si = np.broadcast_to(sl(tau[S, I]) + sl(pi)[None, :], (A, A)).copy()
    log_p_sd = np.full((A, A), sl(tau[S, D]))
    log_p_se = float(sl(tau[S, E]))

    # BOS_I tensors (order-0: slices at index 0 of I tensors)
    log_p_bos_i_m = log_p_im[0].copy()
    log_p_bos_i_i = log_p_ii[0].copy()
    log_p_bos_i_d = log_p_id[0].copy()
    log_p_bos_i_e = log_p_ie[0].copy()

    # BOS_D tensors (slices at index 0 along desc axis of D tensors)
    log_p_bos_d_m = log_p_dm[:, 0].copy()
    log_p_bos_d_i = log_p_di[:, 0].copy()
    log_p_bos_d_d = log_p_dd[:, 0].copy()
    log_p_bos_d_e = log_p_de[:, 0].copy()

    return {
        'log_p_mm': log_p_mm, 'log_p_mi': log_p_mi, 'log_p_md': log_p_md, 'log_p_me': log_p_me,
        'log_p_im': log_p_im, 'log_p_ii': log_p_ii, 'log_p_id': log_p_id, 'log_p_ie': log_p_ie,
        'log_p_dm': log_p_dm, 'log_p_dd': log_p_dd, 'log_p_di': log_p_di, 'log_p_de': log_p_de,
        'log_p_sm': log_p_sm, 'log_p_si': log_p_si, 'log_p_sd': log_p_sd, 'log_p_se': log_p_se,
        'log_p_bos_i_m': log_p_bos_i_m, 'log_p_bos_i_i': log_p_bos_i_i,
        'log_p_bos_i_d': log_p_bos_i_d, 'log_p_bos_i_e': log_p_bos_i_e,
        'log_p_bos_d_m': log_p_bos_d_m, 'log_p_bos_d_i': log_p_bos_d_i,
        'log_p_bos_d_d': log_p_bos_d_d, 'log_p_bos_d_e': log_p_bos_d_e,
    }


def build_tkf92_root_wfst(ins_rate, del_rate, ext, pi):
    """Build TKF92 root (singlet) WFST.

    Per simulation_reconstruction_test_plan.tex:
      S -> I: kappa * pi[b]
      S -> E: 1 - kappa
      I -> I: p_ext * pi[b]
      I -> E: 1 - p_ext

    where kappa = ins_rate/del_rate, p_ext = r + (1-r)*kappa.
    """
    if ins_rate >= del_rate:
        raise ValueError(f"Requires ins_rate < del_rate, got {ins_rate} >= {del_rate}")

    kappa = float(ins_rate / del_rate)
    p_ext = ext + (1.0 - ext) * kappa

    sl = lambda x: np.log(np.maximum(x, 1e-300))

    # S -> I: kappa * pi[b]
    log_kappa_pi = sl(kappa) + sl(pi)
    log_p_si = np.broadcast_to(log_kappa_pi[None, :], (A, A)).copy()

    # S -> E: 1 - kappa
    log_p_se = float(sl(1.0 - kappa))

    # I -> I: p_ext * pi[b]
    log_pext_pi = sl(p_ext) + sl(pi)
    log_p_ii = np.broadcast_to(log_pext_pi[None, None, :], (A, A, A)).copy()

    # I -> E: 1 - p_ext
    log_p_ie = np.full((A, A), sl(1.0 - p_ext))

    # All other transitions impossible
    impossible_4d = np.full((A, A, A, A), NEG_INF)
    impossible_3d = np.full((A, A, A), NEG_INF)
    impossible_2d = np.full((A, A), NEG_INF)

    # BOS_I keys for root (all root inserts are BOS_I)
    log_p_bos_i_m = np.full((A, A, A), NEG_INF)
    log_p_bos_i_i = log_p_ii[0].copy()    # (A, A) = [b-, b+]
    log_p_bos_i_d = np.full((A, A), NEG_INF)
    log_p_bos_i_e = log_p_ie[0].copy()    # (A,) = [b-]

    return {
        'log_p_mm': impossible_4d, 'log_p_mi': impossible_3d,
        'log_p_md': impossible_3d, 'log_p_me': impossible_2d,
        'log_p_im': impossible_4d, 'log_p_ii': log_p_ii,
        'log_p_id': impossible_3d, 'log_p_ie': log_p_ie,
        'log_p_dm': impossible_4d, 'log_p_dd': impossible_3d,
        'log_p_di': impossible_3d, 'log_p_de': impossible_2d,
        'log_p_sm': impossible_2d, 'log_p_si': log_p_si,
        'log_p_sd': impossible_2d, 'log_p_se': log_p_se,
        'log_p_bos_i_m': log_p_bos_i_m, 'log_p_bos_i_i': log_p_bos_i_i,
        'log_p_bos_i_d': log_p_bos_i_d, 'log_p_bos_i_e': log_p_bos_i_e,
    }


# ======================================================================
# Generic order-1 WFST simulator
# ======================================================================

def simulate_wfst_root(singlet_wfst, alphabet_size, np_rng, max_len=2000):
    """Sample a root sequence from a singlet WFST (order-1).

    Walks S → BOS_I → BOS_I → ... → E, sampling characters from the
    context-dependent outgoing distribution at each step.

    Args:
        singlet_wfst: dict of log tensors (same format as tree_varanc WFSTs)
        alphabet_size: A
        np_rng: numpy RandomState
        max_len: truncation limit

    Returns:
        root_seq: (L,) int32 array of character indices
    """
    AA = alphabet_size
    chars = []
    # State tracking: start at S, contexts (BOS, BOS)
    # For root singlet, all transitions are S→BOS_I or BOS_I→BOS_I or →E
    # desc_ctx tracks the last emitted character (or BOS initially)

    # S → BOS_I(b) or S → E
    # Build outgoing log-probs from S
    log_si = np.asarray(singlet_wfst['log_p_si'])  # (A, A) = [dummy, b]
    log_se = float(singlet_wfst['log_p_se'])

    # S outgoing: A inserts + 1 end = A+1 options
    log_probs = np.empty(AA + 1)
    log_probs[:AA] = log_si[0, :]  # S→I for each desc char
    log_probs[AA] = log_se

    probs = _softmax_vec(log_probs)
    choice = np_rng.choice(AA + 1, p=probs)
    if choice == AA:
        return np.array([], dtype=np.int32)  # empty sequence

    desc_ctx = choice
    chars.append(choice)

    # Now in BOS_I state. Use BOS_I→BOS_I or BOS_I→E
    for _ in range(max_len - 1):
        # Get BOS_I outgoing tensors
        if 'log_p_bos_i_i' in singlet_wfst:
            log_ii = np.asarray(singlet_wfst['log_p_bos_i_i'])  # (A, A) = [b-, b+]
            log_ie = np.asarray(singlet_wfst['log_p_bos_i_e'])  # (A,) = [b-]
            log_probs = np.empty(AA + 1)
            log_probs[:AA] = log_ii[desc_ctx, :]
            log_probs[AA] = log_ie[desc_ctx]
        else:
            # Fallback: regular I tensors, index anc ctx at 0
            log_ii = np.asarray(singlet_wfst['log_p_ii'])  # (A,A,A)
            log_ie = np.asarray(singlet_wfst['log_p_ie'])  # (A,A)
            log_probs = np.empty(AA + 1)
            log_probs[:AA] = log_ii[0, desc_ctx, :]
            log_probs[AA] = log_ie[0, desc_ctx]

        probs = _softmax_vec(log_probs)
        choice = np_rng.choice(AA + 1, p=probs)
        if choice == AA:
            break
        desc_ctx = choice
        chars.append(choice)

    return np.array(chars, dtype=np.int32)


def simulate_wfst_branch(branch_wfst, ancestor, alphabet_size, np_rng, max_steps=None):
    """Simulate a descendant sequence from an ancestor using an order-1 branch WFST.

    Tracks (ancestor_context, descendant_context) through the simulation.
    Uses BOS_I/BOS_D state types when contexts are BOS.

    The simulation walks through ancestor positions, sampling M/I/D/E
    transitions from the context-dependent outgoing distribution.

    Args:
        branch_wfst: dict of log tensors with BOS_I/BOS_D keys
        ancestor: (L_anc,) int32 array
        alphabet_size: A
        np_rng: numpy RandomState
        max_steps: safety limit (default: 10 * L_anc + 100)

    Returns:
        descendant: (L_desc,) int32 array
        alignment: list of (anc_idx_or_None, desc_idx_or_None) tuples
    """
    AA = alphabet_size
    L_anc = len(ancestor)
    if max_steps is None:
        max_steps = 10 * L_anc + 100

    descendant = []
    alignment = []
    anc_pos = 0  # next ancestor position to consume

    # Context: (anc_ctx, desc_ctx). Initially both BOS.
    anc_ctx = None   # None = BOS
    desc_ctx = None  # None = BOS

    # Current effective state type
    state = S

    for step in range(max_steps):
        # Determine legal next states based on ancestor consumption
        can_consume = anc_pos < L_anc  # can do M or D
        must_end = anc_pos >= L_anc    # must do I or E (all ancestor consumed)

        # Build outgoing log-probability vector over: M(a,b), I(b), D(a), E
        # For M: AA*AA options (anc_char, desc_char)
        # For I: AA options (desc_char)
        # For D: 1 option per anc_char (but anc_char is determined by ancestor[anc_pos])
        # For E: 1 option
        # Total options: AA*AA + AA + 1 + 1 at most, but M and D are constrained

        log_opts = []  # (log_prob, action_type, anc_char, desc_char)

        if can_consume:
            a_next = int(ancestor[anc_pos])

            # M transitions: emit (a_next, b) for each b
            for b in range(AA):
                lp = _get_wfst_transition(branch_wfst, state, anc_ctx, desc_ctx,
                                          M, a_next, b, AA)
                log_opts.append((lp, M, a_next, b))

            # D transitions: emit (a_next, _)
            lp = _get_wfst_transition(branch_wfst, state, anc_ctx, desc_ctx,
                                      D, a_next, None, AA)
            log_opts.append((lp, D, a_next, None))

        if not must_end:
            # I transitions: emit (_, b) for each b
            for b in range(AA):
                lp = _get_wfst_transition(branch_wfst, state, anc_ctx, desc_ctx,
                                          I, None, b, AA)
                log_opts.append((lp, I, None, b))

        if must_end:
            # E transition
            lp = _get_wfst_transition(branch_wfst, state, anc_ctx, desc_ctx,
                                      E, None, None, AA)
            log_opts.append((lp, E, None, None))
        else:
            # I also possible even when can_consume (inserts between ancestor positions)
            pass  # already added above when not must_end

        if must_end and can_consume:
            raise ValueError("Logic error: both must_end and can_consume")

        if not must_end:
            # E is also possible if there happen to be no more ancestor chars
            # Only when all consumed — handled above
            pass

        # When can_consume and not must_end: M, D, I are all possible, E is not
        # When must_end: only E is possible (I would be wrong — all ancestor consumed,
        # and in TKF the HMM constrains that we can't insert after consuming all ancestor)
        # Actually in TKF, inserts CAN happen after last M/D before E.
        # Let me reconsider: the constraint is all ancestor must be consumed for E,
        # but I is always possible (inserts can happen anywhere).

        # Correction: I transitions don't consume ancestor, so they're allowed
        # even after all ancestor is consumed. Only E requires all consumed.
        # But also M and D require ancestor remaining.
        if anc_pos >= L_anc:
            # Can only do I or E
            # Re-build: I transitions
            i_opts = []
            for b in range(AA):
                lp = _get_wfst_transition(branch_wfst, state, anc_ctx, desc_ctx,
                                          I, None, b, AA)
                i_opts.append((lp, I, None, b))
            # E
            lp_e = _get_wfst_transition(branch_wfst, state, anc_ctx, desc_ctx,
                                        E, None, None, AA)
            i_opts.append((lp_e, E, None, None))
            log_opts = i_opts

        # Normalize and sample
        log_vals = np.array([o[0] for o in log_opts])
        probs = _softmax_vec(log_vals)
        idx = np_rng.choice(len(probs), p=probs)
        _, action, a_char, b_char = log_opts[idx]

        if action == E:
            break
        elif action == M:
            desc_idx = len(descendant)
            descendant.append(b_char)
            alignment.append((anc_pos, desc_idx))
            anc_ctx = a_char
            desc_ctx = b_char
            anc_pos += 1
            state = M
        elif action == I:
            desc_idx = len(descendant)
            descendant.append(b_char)
            alignment.append((None, desc_idx))
            # anc_ctx carries (unchanged)
            desc_ctx = b_char
            # Effective state: BOS_I if anc_ctx is still BOS, else regular I
            state = BOS_I if anc_ctx is None else I
        elif action == D:
            alignment.append((anc_pos, None))
            anc_ctx = a_char
            # desc_ctx carries (unchanged)
            anc_pos += 1
            state = BOS_D if desc_ctx is None else D

    return np.array(descendant, dtype=np.int32), alignment


def _get_wfst_transition(wfst, X, anc_ctx, desc_ctx, Y, a_plus, b_plus, AA):
    """Look up a single transition log-weight from WFST tensors.

    X: current state type (S, M, I, D, BOS_I, BOS_D)
    anc_ctx, desc_ctx: int or None (None = BOS)
    Y: destination state type (M, I, D, E)
    a_plus, b_plus: emitted chars (int or None)
    """
    # Determine WFST key based on source state
    state_prefix = {S: 's', M: 'm', I: 'i', D: 'd'}
    state_suffix = {M: 'm', I: 'i', D: 'd', E: 'e'}
    y_suf = state_suffix[Y]

    if X == S:
        key = f'log_p_s{y_suf}'
        tensor = np.asarray(wfst[key])
        if Y == E:
            return float(tensor)
        elif Y == M:
            return float(tensor[a_plus, b_plus])
        elif Y == I:
            return float(tensor[0, b_plus])  # S has BOS anc ctx, index 0
        elif Y == D:
            return float(tensor[a_plus, 0])  # S has BOS desc ctx, index 0
    elif X == BOS_I:
        key_bos = f'log_p_bos_i_{y_suf}'
        if key_bos in wfst:
            tensor = np.asarray(wfst[key_bos])
            dc = desc_ctx if desc_ctx is not None else 0
            if Y == E:
                return float(tensor[dc])
            elif Y == M:
                return float(tensor[dc, a_plus, b_plus])
            elif Y == I:
                return float(tensor[dc, b_plus])
            elif Y == D:
                return float(tensor[dc, a_plus])
        else:
            # Fallback
            key = f'log_p_i{y_suf}'
            tensor = np.asarray(wfst[key])
            dc = desc_ctx if desc_ctx is not None else 0
            if Y == E: return float(tensor[0, dc])
            elif Y == M: return float(tensor[0, dc, a_plus, b_plus])
            elif Y == I: return float(tensor[0, dc, b_plus])
            elif Y == D: return float(tensor[0, dc, a_plus])
    elif X == BOS_D:
        key_bos = f'log_p_bos_d_{y_suf}'
        if key_bos in wfst:
            tensor = np.asarray(wfst[key_bos])
            ac = anc_ctx if anc_ctx is not None else 0
            if Y == E:
                return float(tensor[ac])
            elif Y == M:
                return float(tensor[ac, a_plus, b_plus])
            elif Y == I:
                return float(tensor[ac, b_plus])
            elif Y == D:
                return float(tensor[ac, a_plus])
        else:
            key = f'log_p_d{y_suf}'
            tensor = np.asarray(wfst[key])
            ac = anc_ctx if anc_ctx is not None else 0
            if Y == E: return float(tensor[ac, 0])
            elif Y == M: return float(tensor[ac, 0, a_plus, b_plus])
            elif Y == I: return float(tensor[ac, 0, b_plus])
            elif Y == D: return float(tensor[ac, 0, a_plus])
    else:
        # Regular M, I, D
        key = f'log_p_{state_prefix[X]}{y_suf}'
        tensor = np.asarray(wfst[key])
        ac = anc_ctx if anc_ctx is not None else 0
        dc = desc_ctx if desc_ctx is not None else 0
        if Y == E:
            return float(tensor[ac, dc])
        elif Y == M:
            return float(tensor[ac, dc, a_plus, b_plus])
        elif Y == I:
            return float(tensor[ac, dc, b_plus])
        elif Y == D:
            return float(tensor[ac, dc, a_plus])

    return NEG_INF


def _softmax_vec(log_probs):
    """Softmax with -inf handling."""
    v = np.array(log_probs, dtype=np.float64)
    m = np.max(v)
    if not np.isfinite(m):
        raise ValueError("All transitions have -inf weight — structural error")
    e = np.exp(v - m)
    s = np.sum(e)
    if s < 1e-300:
        raise ValueError("Zero total probability after softmax — structural error")
    return e / s


# ======================================================================
# Tree simulation
# ======================================================================

def simulate_tree_msa(tree, wfst_per_edge, singlet_wfst, pi, rng_key):
    """Simulate a full ancestral MSA on a tree using order-1 WFSTs.

    Uses the generic WFST simulator for both root and branches.

    Args:
        tree: TreeNode root
        wfst_per_edge: dict {(parent, child): wfst_log_dict}
        singlet_wfst: root singlet WFST
        pi: (A,) equilibrium distribution
        rng_key: JAX PRNG key

    Returns:
        msa_presence: dict {node_name: (L_msa,) bool}
        leaf_seqs_aligned: dict {leaf_name: (L_msa,) int32, -1=gap}
        true_root_seq: (L_root,) int32
        node_seqs: dict {node_name: (L_node,) int32}
        sim_time: float
    """
    t0 = time.time()
    AA = pi.shape[0]

    # Simulate root sequence using singlet WFST
    rng_seed = int(jr.fold_in(rng_key, 0)[0]) % (2**31)
    np_rng = np.random.RandomState(rng_seed)
    root_seq = simulate_wfst_root(singlet_wfst, AA, np_rng)

    node_seqs = {tree.name: root_seq}
    branch_alignments = {}

    for node in tree.preorder():
        if node.is_root:
            continue
        parent = node.parent
        anc_seq = node_seqs[parent.name]
        key = (parent.name, node.name)

        rng_seed = int(jr.fold_in(rng_key, hash(key) % (2**30))[0]) % (2**31)
        branch_rng = np.random.RandomState(rng_seed)

        desc_seq, alignment = simulate_wfst_branch(
            wfst_per_edge[key], anc_seq, AA, branch_rng)

        node_seqs[node.name] = desc_seq
        branch_alignments[key] = alignment

    # Build MSA
    msa_presence, aligned_seqs = _build_msa_from_alignments(
        tree, node_seqs, branch_alignments)

    sim_time = time.time() - t0

    leaf_seqs_aligned = {name: seq for name, seq in aligned_seqs.items()
                         if name in {n.name for n in tree.preorder() if n.is_leaf}}

    return msa_presence, leaf_seqs_aligned, np.asarray(root_seq), node_seqs, sim_time


def _build_msa_from_alignments(tree, node_seqs, branch_alignments):
    """Build a full MSA from per-branch pairwise alignments.

    Uses the standard progressive approach: root positions first,
    then insert descendant-only positions between their flanking
    ancestor positions.
    """
    # Start from root: each root char gets a column
    root_name = tree.name
    root_seq = node_seqs[root_name]
    L_root = len(root_seq)

    # Build MSA column list incrementally
    # Each column tracks which nodes are present and their character
    # We process the tree top-down, inserting columns for insertions

    # Column representation: list of dicts {node_name: char_idx or -1}
    # Start with root columns
    all_nodes = [n.name for n in tree.preorder()]

    # For each node, track its MSA column assignments
    node_col_map = {}  # {node_name: list of MSA column indices for ungapped positions}
    node_col_map[root_name] = list(range(L_root))

    # Total columns so far
    columns = []  # list of sets: which nodes are present
    col_chars = []  # list of dicts: {node_name: char_idx}

    for c in range(L_root):
        columns.append({root_name})
        col_chars.append({root_name: int(root_seq[c])})

    # Process tree top-down
    for node in tree.preorder():
        if node.is_root:
            continue
        parent = node.parent
        alignment = branch_alignments[(parent.name, node.name)]
        desc_seq = node_seqs[node.name]

        parent_cols = node_col_map[parent.name]
        child_cols = []

        # Track insertions that need new columns
        # alignment is list of (anc_idx_or_None, desc_idx_or_None)
        # We need to place each desc position in the MSA

        # Group alignment by ancestor position
        # Insertions before the first M/D go at the beginning
        # Insertions between two M/D events go between their columns
        last_parent_col_idx = -1  # index into parent_cols (-1 = before first)

        for anc_idx, desc_idx in alignment:
            if anc_idx is not None and desc_idx is not None:
                # Match: child goes in same column as parent
                msa_col = parent_cols[anc_idx]
                columns[msa_col].add(node.name)
                col_chars[msa_col][node.name] = int(desc_seq[desc_idx])
                child_cols.append(msa_col)
                last_parent_col_idx = anc_idx
            elif anc_idx is not None:
                # Delete: parent has char, child doesn't
                # Child is absent at this column (already handled by default)
                last_parent_col_idx = anc_idx
            elif desc_idx is not None:
                # Insert: child has char, parent doesn't
                # Need a new column. Place it after last_parent_col_idx
                new_col_idx = len(columns)
                columns.append({node.name})
                col_chars.append({node.name: int(desc_seq[desc_idx])})

                # Find insertion point (after last_parent_col_idx's column)
                # For simplicity, just append to end. We'll sort later.
                child_cols.append(new_col_idx)

        node_col_map[node.name] = child_cols

    # Build final MSA arrays
    L_msa = len(columns)
    msa_presence = {}
    aligned_seqs = {}

    for name in all_nodes:
        pres = np.zeros(L_msa, dtype=bool)
        seq = np.full(L_msa, -1, dtype=np.int32)
        for c in range(L_msa):
            if name in columns[c]:
                pres[c] = True
                seq[c] = col_chars[c].get(name, -1)
        msa_presence[name] = pres
        aligned_seqs[name] = seq

    return msa_presence, aligned_seqs


# ======================================================================
# Reconstruction methods
# ======================================================================

def reconstruct_felsenstein(tree, msa_presence, leaf_seqs_aligned, Q, pi):
    """Felsenstein pruning reconstruction at root."""
    t0 = time.time()

    root_name = tree.name
    _, posteriors = marginal_ancestor_all_columns_jax(tree, leaf_seqs_aligned, Q, pi)
    posteriors = np.asarray(posteriors)

    root_pres = msa_presence[root_name]
    root_cols = np.where(root_pres)[0]

    root_post = posteriors[root_cols]  # (K, A)
    root_map = np.argmax(root_post, axis=1).astype(np.int32)

    t_total = time.time() - t0

    return root_map, root_post, {'t_total_recon': t_total, 't_preprocess': 0.0,
                                   't_factor_build': 0.0, 't_optimize': 0.0,
                                   't_posterior_extract': 0.0,
                                   'n_sweeps': 0, 'converged': True,
                                   'final_residual': 0.0, 'elbo_start': 0.0,
                                   'elbo_end': 0.0, 'n_factors': 0,
                                   'n_latent_vars': 0}


def reconstruct_tree_varanc(tree, msa_presence, leaf_seqs_aligned, Q, pi,
                             wfst_per_edge, singlet_wfst,
                             max_iter=20, tol=1e-6):
    """TreeVarAnc reconstruction at root with detailed timing."""
    from tkfmixdom.jax.tree.tree_varanc import (
        PrecomputedStructure, VariationalState, build_factor_table,
        compute_elbo, update_column,
    )

    t_start = time.time()

    # Phase 1: preprocessing
    t0 = time.time()
    precomp = PrecomputedStructure(
        tree, msa_presence, leaf_seqs_aligned, wfst_per_edge, singlet_wfst)
    t_preprocess = time.time() - t0

    # Phase 2: factor construction
    t0 = time.time()
    factor_tables = [build_factor_table(fi, precomp) for fi in precomp.factors]
    t_factor_build = time.time() - t0

    # Phase 3: optimization
    t0 = time.time()
    state = VariationalState(precomp)

    elbo_trace = []
    converged = False
    max_residual = float('inf')

    for iteration in range(max_iter):
        sweep_residual = 0.0
        for c in range(precomp.L_msa):
            residual = update_column(c, precomp, state, factor_tables)
            sweep_residual = max(sweep_residual, residual)
        max_residual = sweep_residual

        elbo = compute_elbo(precomp, state, factor_tables)
        elbo_trace.append(elbo)

        if len(elbo_trace) >= 2:
            delta = abs(elbo_trace[-1] - elbo_trace[-2])
            if delta < tol and sweep_residual < tol:
                converged = True
                break

    t_optimize = time.time() - t0

    # Phase 4: posterior extraction
    t0 = time.time()
    root_name = tree.name
    root_cols = [c for c in range(precomp.L_msa) if msa_presence[root_name][c]]
    root_post = []
    for c in root_cols:
        b = state.node_beliefs.get((root_name, c))
        if b is not None:
            root_post.append(b)
        else:
            root_post.append(np.ones(precomp.A) / precomp.A)
    root_post = np.array(root_post)
    root_map = np.argmax(root_post, axis=1).astype(np.int32)
    t_posterior_extract = time.time() - t0

    t_total = time.time() - t_start

    return root_map, root_post, {
        't_total_recon': t_total,
        't_preprocess': t_preprocess,
        't_factor_build': t_factor_build,
        't_optimize': t_optimize,
        't_posterior_extract': t_posterior_extract,
        'n_sweeps': len(elbo_trace),
        'converged': converged,
        'final_residual': max_residual,
        'elbo_start': elbo_trace[0] if elbo_trace else 0.0,
        'elbo_end': elbo_trace[-1] if elbo_trace else 0.0,
        'n_factors': len(precomp.factors),
        'n_latent_vars': len(precomp.is_latent),
    }


# ======================================================================
# Metrics
# ======================================================================

def compute_metrics(root_map, root_post, true_root_seq):
    """Compute accuracy and log score."""
    K = len(true_root_seq)
    assert len(root_map) == K, f"Length mismatch: {len(root_map)} vs {K}"

    accuracy = float(np.mean(root_map == true_root_seq))

    log_scores = []
    for i in range(K):
        true_char = true_root_seq[i]
        p = root_post[i, true_char]
        log_scores.append(float(np.log(max(p, 1e-300))))
    log_score = float(np.mean(log_scores))

    return accuracy, log_score


# ======================================================================
# Single replicate runner
# ======================================================================

# ======================================================================
# WFST factory helpers
# ======================================================================

MIXDOM_PARAMS_PATH = Path(__file__).parent.parent / "pfam" / "maraschino_d3_trainsplit_entreg.npz"

_mixdom_cache = {}  # cache precomputed MixDom objects


def build_wfst_for_model(sim_model, tree, ins_rate, del_rate, ext, Q, pi):
    """Build per-edge and root WFSTs for a given simulation model.

    Returns:
        wfst_per_edge: dict {(parent, child): wfst_log_dict}
        singlet_wfst: root singlet WFST
    """
    if sim_model == 'tkf91':
        wfst_per_edge = {}
        for node in tree.preorder():
            if node.is_root:
                continue
            key = (node.parent.name, node.name)
            wfst_per_edge[key] = build_tkf91_branch_wfst(
                ins_rate, del_rate, Q, pi, node.branch_length)
        singlet_wfst = build_tkf91_root_wfst(ins_rate, del_rate, pi)
        return wfst_per_edge, singlet_wfst

    elif sim_model == 'tkf92':
        wfst_per_edge = {}
        for node in tree.preorder():
            if node.is_root:
                continue
            key = (node.parent.name, node.name)
            wfst_per_edge[key] = build_tkf92_branch_wfst_cond(
                ins_rate, del_rate, ext, Q, pi, node.branch_length)
        singlet_wfst = build_tkf92_root_wfst(ins_rate, del_rate, ext, pi)
        return wfst_per_edge, singlet_wfst

    elif sim_model == 'mixdom':
        return build_mixdom_wfsts(tree)

    else:
        raise ValueError(f"Unknown model: {sim_model}")


def build_mixdom_wfsts(tree):
    """Build MixDom order-1 WFSTs via algebraic distillation.

    Uses the trained MixDom parameters and distills to per-branch
    order-1 WFSTs. These are context-dependent (not broadcast).

    Returns:
        wfst_per_edge: dict with BOS_I/BOS_D keys
        singlet_wfst: root singlet with BOS_I keys
    """
    from tkfmixdom.jax.distill.maraschino import (
        load_params, precompute_mixdom, distill_mixdom,
        normalize_freqs_wfst,
    )
    # Import from sibling file in experiments/
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "ancrec_benchmark", Path(__file__).parent / "ancrec_benchmark.py")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    build_mixdom_wfst_log = _mod.build_mixdom_wfst_log

    # Load and cache MixDom params
    if 'params' not in _mixdom_cache:
        params, n_domains, n_classes = load_params(str(MIXDOM_PARAMS_PATH))
        precomp = precompute_mixdom(params, n_classes)
        _mixdom_cache['params'] = params
        _mixdom_cache['n_classes'] = n_classes
        _mixdom_cache['precomp'] = precomp

    params = _mixdom_cache['params']
    n_classes = _mixdom_cache['n_classes']
    precomp = _mixdom_cache['precomp']

    wfst_per_edge = {}
    bl_cache = {}  # cache by rounded branch length

    for node in tree.preorder():
        if node.is_root:
            continue
        t = max(node.branch_length, 1e-6)
        t_key = round(t, 6)

        if t_key not in bl_cache:
            dist = distill_mixdom(params, t, n_classes, precomp)
            wfst = normalize_freqs_wfst(dist)
            log_wfst = build_mixdom_wfst_log(wfst)
            # Add BOS_I/BOS_D keys
            log_wfst = _add_bos_keys(log_wfst)
            bl_cache[t_key] = log_wfst

        key = (node.parent.name, node.name)
        wfst_per_edge[key] = bl_cache[t_key]

    # Build root singlet from MixDom stationary parameters
    singlet_wfst = _build_mixdom_root_singlet(params)

    return wfst_per_edge, singlet_wfst


def _add_bos_keys(wfst_log):
    """Add BOS_I and BOS_D tensor keys to an existing WFST log dict.

    For order-1 WFSTs, the BOS_I tensors are slices at index 0 of the
    ancestor context axis of the I tensors. For order-1 models where
    BOS has distinct semantics from char 0, the WFST factory should
    produce these separately. For now, index-0 slices.
    """
    d = dict(wfst_log)
    # BOS_I: ancestor context = BOS → slice I tensors at a=0
    d['log_p_bos_i_m'] = np.asarray(d['log_p_im'])[0].copy()
    d['log_p_bos_i_i'] = np.asarray(d['log_p_ii'])[0].copy()
    d['log_p_bos_i_d'] = np.asarray(d['log_p_id'])[0].copy()
    d['log_p_bos_i_e'] = np.asarray(d['log_p_ie'])[0].copy()
    # BOS_D: descendant context = BOS → slice D tensors at b=0
    d['log_p_bos_d_m'] = np.asarray(d['log_p_dm'])[:, 0].copy()
    d['log_p_bos_d_i'] = np.asarray(d['log_p_di'])[:, 0].copy()
    d['log_p_bos_d_d'] = np.asarray(d['log_p_dd'])[:, 0].copy()
    d['log_p_bos_d_e'] = np.asarray(d['log_p_de'])[:, 0].copy()
    return d


def _build_mixdom_root_singlet(params):
    """Build root singlet WFST from MixDom distilled singlet model.

    Uses the distilled singlet (f_singlet, f_singlet_start, f_singlet_end)
    which captures the full hierarchical MixDom stationary distribution.

    NOTE: A previous version used weighted_kappa = sum(v_d * kappa_d) = 0.66,
    giving E[root_length] ~ 2. The correct distilled singlet has P(continue)
    ~ 0.99, giving E[root_length] ~ 150. The hierarchical structure (top-level
    geometric over domains, per-domain geometric over fragments) must be
    distilled, not averaged. This discrepancy merits future study: the
    trained MixDom parameters produce very different root length distributions
    depending on how the singlet is constructed.
    """
    from tkfmixdom.jax.distill.maraschino import (
        precompute_mixdom, distill_mixdom,
    )

    n_classes = _mixdom_cache.get('n_classes', 4)
    precomp = _mixdom_cache.get('precomp')
    if precomp is None:
        precomp = precompute_mixdom(params, n_classes)

    # Distill at t=0 (arbitrary, singlet is t-independent)
    dist = distill_mixdom(params, 0.1, n_classes, precomp)

    f_singlet = np.asarray(dist['f_singlet'])       # (AA, AA)
    f_start = np.asarray(dist['f_singlet_start'])    # (AA,)
    f_end = np.asarray(dist['f_singlet_end'])        # (AA,)

    AA = f_singlet.shape[0]
    sl = lambda x: np.log(np.maximum(x, 1e-300))

    # Normalize singlet: P(b | a) = f_singlet[a,b] / (sum_b f_singlet[a,b] + f_end[a])
    # S → I: P(start, b) = f_start[b] / (sum_b f_start[b] + (1-sum_b f_start[b]))
    total_start = np.sum(f_start)
    p_start_emit = f_start / (total_start + (1.0 - total_start)) if total_start < 1.0 else f_start / np.sum(f_start)
    p_start_end = 1.0 - np.sum(p_start_emit)

    log_p_si = np.broadcast_to(sl(p_start_emit)[None, :], (AA, AA)).copy()
    log_p_se = float(sl(max(p_start_end, 1e-300)))

    # I → I: context-dependent transition P(b' | b) from singlet
    # I → E: P(end | b) from singlet
    # Normalize per source char b
    Z = np.sum(f_singlet, axis=1) + f_end  # (AA,)
    Z = np.maximum(Z, 1e-300)
    p_ii = f_singlet / Z[:, None]          # (AA, AA)
    p_ie = f_end / Z                       # (AA,)

    # Shape for WFST: log_p_ii[anc_ctx, desc_ctx, desc_emit] = (AA, AA, AA)
    # For root, anc_ctx is always BOS. Broadcast over anc_ctx axis.
    log_p_ii = np.broadcast_to(sl(p_ii)[None, :, :], (AA, AA, AA)).copy()
    log_p_ie = np.broadcast_to(sl(p_ie)[None, :], (AA, AA)).copy()

    impossible_4d = np.full((AA, AA, AA, AA), NEG_INF)
    impossible_3d = np.full((AA, AA, AA), NEG_INF)
    impossible_2d = np.full((AA, AA), NEG_INF)

    # BOS_I versions (slice at anc_ctx=0, which is broadcast anyway for root)
    log_p_bos_i_i = log_p_ii[0].copy()   # (AA, AA)
    log_p_bos_i_e = log_p_ie[0].copy()   # (AA,)

    return {
        'log_p_mm': impossible_4d, 'log_p_mi': impossible_3d,
        'log_p_md': impossible_3d, 'log_p_me': impossible_2d,
        'log_p_im': impossible_4d, 'log_p_ii': log_p_ii,
        'log_p_id': impossible_3d, 'log_p_ie': log_p_ie,
        'log_p_dm': impossible_4d, 'log_p_dd': impossible_3d,
        'log_p_di': impossible_3d, 'log_p_de': impossible_2d,
        'log_p_sm': impossible_2d, 'log_p_si': log_p_si,
        'log_p_sd': impossible_2d, 'log_p_se': log_p_se,
        'log_p_bos_i_m': np.full((AA, AA, AA), NEG_INF),
        'log_p_bos_i_i': log_p_bos_i_i,
        'log_p_bos_i_d': np.full((AA, AA), NEG_INF),
        'log_p_bos_i_e': log_p_bos_i_e,
    }


# ======================================================================
# Single replicate runner
# ======================================================================

def run_replicate(config, ins_rate, del_rate, ext, Q, pi, rng_key):
    """Run one simulation + reconstruction replicate."""
    result = ReplicateResult()

    # Generate tree
    np_rng = np.random.RandomState(int(jr.fold_in(rng_key, 0)[0]) % 2**31)
    tree = make_random_tree(config.n_leaves, config.branch_regime, np_rng)

    # Build WFSTs for simulation model
    sim_wfst_per_edge, sim_singlet_wfst = build_wfst_for_model(
        config.sim_model, tree, ins_rate, del_rate, ext, Q, pi)

    # Simulate MSA using generic order-1 WFST simulator
    rng_key, subkey = jr.split(rng_key)
    msa_presence, leaf_seqs, true_root_seq, node_seqs, sim_time = simulate_tree_msa(
        tree, sim_wfst_per_edge, sim_singlet_wfst, pi, subkey)
    result.t_sim = sim_time
    result.root_length = len(true_root_seq)
    result.n_cols = next(iter(msa_presence.values())).shape[0]

    if result.root_length == 0:
        return result  # degenerate case

    # Reconstruct
    if config.rec_method == 'felsenstein':
        root_map, root_post, timing = reconstruct_felsenstein(
            tree, msa_presence, leaf_seqs, Q, pi)
    elif config.rec_method == 'tree_varanc':
        # For reconstruction, use the SAME model's WFSTs
        # (or a different model for model-mismatch experiments)
        rec_wfst_per_edge, rec_singlet_wfst = build_wfst_for_model(
            config.sim_model, tree, ins_rate, del_rate, ext, Q, pi)

        root_map, root_post, timing = reconstruct_tree_varanc(
            tree, msa_presence, leaf_seqs, Q, pi,
            rec_wfst_per_edge, rec_singlet_wfst,
            max_iter=config.varanc_max_iter, tol=config.varanc_tol)
    else:
        raise ValueError(f"Unknown rec_method: {config.rec_method}")

    # Metrics
    accuracy, log_score = compute_metrics(root_map, root_post, true_root_seq)
    result.root_accuracy = accuracy
    result.log_score = log_score

    # Timings
    result.t_total_recon = timing['t_total_recon']
    result.t_preprocess = timing['t_preprocess']
    result.t_factor_build = timing['t_factor_build']
    result.t_optimize = timing['t_optimize']
    result.t_posterior_extract = timing['t_posterior_extract']
    result.n_sweeps = timing['n_sweeps']
    result.converged = timing['converged']
    result.final_residual = timing['final_residual']
    result.elbo_start = timing['elbo_start']
    result.elbo_end = timing['elbo_end']
    result.n_factors = timing['n_factors']
    result.n_latent_vars = timing['n_latent_vars']

    return result


# ======================================================================
# Benchmark grid
# ======================================================================

def build_pilot_grid():
    """Build a small pilot grid for initial scaling estimates.

    Ordered: all Felsenstein cells first (cheap), then TreeVarAnc (expensive).
    """
    configs = []
    # Felsenstein first — fast, covers all sim models
    for sim_model in ['tkf91', 'tkf92', 'mixdom']:
        for n_leaves in [4, 8]:
            configs.append(BenchmarkConfig(
                sim_model=sim_model, rec_method='felsenstein',
                n_leaves=n_leaves, n_cols_target=0,
                branch_regime='moderate', n_replicates=2))
    # TreeVarAnc — expensive, smallest sizes first
    for sim_model in ['tkf91', 'tkf92', 'mixdom']:
        for n_leaves in [4, 8]:
            configs.append(BenchmarkConfig(
                sim_model=sim_model, rec_method='tree_varanc',
                n_leaves=n_leaves, n_cols_target=0,
                branch_regime='moderate', n_replicates=2))
    return configs


def build_5min_grid():
    """Build grid for 5-minute budget.

    Ordered: Felsenstein first, then TreeVarAnc smallest-first.
    """
    configs = []
    # Felsenstein — cheap, full coverage
    for sim_model in ['tkf91', 'tkf92', 'mixdom']:
        for n_leaves in [4, 8, 16]:
            for branch_regime in ['short', 'moderate', 'long']:
                configs.append(BenchmarkConfig(
                    sim_model=sim_model, rec_method='felsenstein',
                    n_leaves=n_leaves, n_cols_target=0,
                    branch_regime=branch_regime, n_replicates=3))
    # TreeVarAnc — expensive, smallest first
    for sim_model in ['tkf91', 'tkf92', 'mixdom']:
        for n_leaves in [4, 8, 16]:
            for branch_regime in ['short', 'moderate', 'long']:
                configs.append(BenchmarkConfig(
                    sim_model=sim_model, rec_method='tree_varanc',
                    n_leaves=n_leaves, n_cols_target=0,
                    branch_regime=branch_regime, n_replicates=2))
    return configs


def build_10min_grid():
    """Build grid for 10-minute budget."""
    configs = []
    for sim_model in ['tkf91', 'tkf92', 'mixdom']:
        for rec_method in ['felsenstein', 'tree_varanc']:
            for n_leaves in [4, 8, 16, 32]:
                for branch_regime in ['short', 'moderate', 'long']:
                    reps = 3 if n_leaves >= 32 else 5
                    configs.append(BenchmarkConfig(
                        sim_model=sim_model,
                        rec_method=rec_method,
                        n_leaves=n_leaves,
                        n_cols_target=0,
                        branch_regime=branch_regime,
                        n_replicates=reps,
                    ))
    return configs


# ======================================================================
# Machine metadata
# ======================================================================

def collect_machine_info():
    """Collect machine metadata for reproducibility."""
    info = {
        'hostname': platform.node(),
        'cpu': platform.processor() or 'unknown',
        'platform': platform.platform(),
        'python_version': sys.version,
        'numpy_version': np.__version__,
    }
    try:
        import jax
        info['jax_version'] = jax.__version__
        info['jax_platform'] = str(jax.devices()[0].platform)
    except Exception:
        pass
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        info['git_commit'] = result.stdout.strip()
    except Exception:
        pass
    return info


# ======================================================================
# Main benchmark runner
# ======================================================================

def run_benchmark(configs, budget_seconds=None, verbose=True):
    """Run the full benchmark grid.

    Args:
        configs: list of BenchmarkConfig
        budget_seconds: optional wall-clock budget in seconds
        verbose: print progress
    """
    # Load parameters
    Q_lg, pi_lg = rate_matrix_lg()
    Q = np.asarray(Q_lg)
    pi = np.asarray(pi_lg)

    # Load fitted TKF92 params
    with open(PARAMS_PATH) as f:
        fitted = json.load(f)
    ins_rate = fitted['ins_rate']
    del_rate = fitted['del_rate']
    ext = fitted['ext_rate']

    if verbose:
        print(f"Parameters: ins={ins_rate:.4f}, del={del_rate:.4f}, ext={ext:.4f}")
        print(f"Grid: {len(configs)} cells")
        if budget_seconds:
            print(f"Budget: {budget_seconds}s")
        print()

    results = []
    t_start = time.time()
    base_rng = jr.PRNGKey(42)

    for ci, config in enumerate(configs):
        elapsed = time.time() - t_start
        if budget_seconds and elapsed > budget_seconds:
            if verbose:
                print(f"Budget exhausted at cell {ci}/{len(configs)} ({elapsed:.0f}s)")
            break

        cell_result = CellResult(config=asdict(config))

        if verbose:
            remaining = budget_seconds - elapsed if budget_seconds else float('inf')
            print(f"[{ci+1}/{len(configs)}] {config.sim_model}/{config.rec_method} "
                  f"leaves={config.n_leaves} branch={config.branch_regime} "
                  f"reps={config.n_replicates} (budget remaining: {remaining:.0f}s)")

        for rep in range(config.n_replicates):
            if budget_seconds and (time.time() - t_start) > budget_seconds:
                break

            rng_key = jr.fold_in(base_rng, ci * 1000 + rep)

            try:
                result = run_replicate(config, ins_rate, del_rate, ext, Q, pi, rng_key)
                cell_result.replicates.append(result)

                if verbose:
                    print(f"  rep {rep}: acc={result.root_accuracy:.3f} "
                          f"logscore={result.log_score:.3f} "
                          f"t_recon={result.t_total_recon:.3f}s "
                          f"cols={result.n_cols} root_len={result.root_length}"
                          + (f" sweeps={result.n_sweeps}" if config.rec_method == 'tree_varanc' else ''))
            except Exception as e:
                if verbose:
                    print(f"  rep {rep}: FAILED - {e}")

        # Classify feasibility
        if cell_result.replicates:
            mean_t = np.mean([r.t_total_recon for r in cell_result.replicates])
            if mean_t < 5:
                cell_result.feasibility_class = 'P1'
            elif mean_t < 60:
                cell_result.feasibility_class = 'P2'
            elif mean_t < 600:
                cell_result.feasibility_class = 'J1'
            else:
                cell_result.feasibility_class = 'J2'

        results.append(cell_result)

    total_time = time.time() - t_start
    if verbose:
        print(f"\nTotal benchmark time: {total_time:.1f}s")

    return results, total_time


# ======================================================================
# Reporting
# ======================================================================

def print_summary_table(results):
    """Print summary tables as specified in simulation-design.md."""
    print("\n" + "=" * 90)
    print("TABLE A: Runtime Decomposition")
    print("=" * 90)
    print(f"{'Sim':>6} {'Rec':>12} {'Leaves':>6} {'Branch':>8} {'N':>3} "
          f"{'t_sim':>7} {'t_recon':>8} {'t_opt':>7} {'sweeps':>6} {'class':>5}")
    print("-" * 90)

    for cell in results:
        if not cell.replicates:
            continue
        s = cell.summary()
        c = cell.config
        print(f"{c['sim_model']:>6} {c['rec_method']:>12} {c['n_leaves']:>6} "
              f"{c['branch_regime']:>8} {s['n_replicates']:>3} "
              f"{s['t_sim']['mean']:>7.3f} {s['t_total_recon']['mean']:>8.3f} "
              f"{s['t_optimize']['mean']:>7.3f} {s['n_sweeps']['mean']:>6.1f} "
              f"{cell.feasibility_class:>5}")

    print("\n" + "=" * 90)
    print("TABLE B: Reconstruction Quality")
    print("=" * 90)
    print(f"{'Sim':>6} {'Rec':>12} {'Leaves':>6} {'Branch':>8} {'N':>3} "
          f"{'Acc±SE':>12} {'LogScore±SE':>14} {'Cols':>6} {'RootLen':>7}")
    print("-" * 90)

    for cell in results:
        if not cell.replicates:
            continue
        s = cell.summary()
        c = cell.config
        acc = s['accuracy']
        ls = s['log_score']
        print(f"{c['sim_model']:>6} {c['rec_method']:>12} {c['n_leaves']:>6} "
              f"{c['branch_regime']:>8} {s['n_replicates']:>3} "
              f"{acc['mean']:>6.3f}±{acc['se']:>4.3f} "
              f"{ls['mean']:>8.3f}±{ls['se']:>4.3f} "
              f"{s['n_cols']['mean']:>6.0f} {s['root_length']['mean']:>7.0f}")

    print("\n" + "=" * 90)
    print("TABLE C: Speed Comparison")
    print("=" * 90)
    # Group by (sim_model, n_leaves, branch_regime) and compare methods
    grouped = defaultdict(dict)
    for cell in results:
        if not cell.replicates:
            continue
        c = cell.config
        key = (c['sim_model'], c['n_leaves'], c['branch_regime'])
        grouped[key][c['rec_method']] = cell

    print(f"{'Sim':>6} {'Leaves':>6} {'Branch':>8} {'Fels_t':>8} {'TVA_t':>8} "
          f"{'Ratio':>7} {'TVA_class':>9}")
    print("-" * 70)

    for key in sorted(grouped.keys()):
        sim, nl, br = key
        fels = grouped[key].get('felsenstein')
        tva = grouped[key].get('tree_varanc')
        fels_t = fels.summary()['t_total_recon']['mean'] if fels else float('nan')
        tva_t = tva.summary()['t_total_recon']['mean'] if tva else float('nan')
        ratio = tva_t / fels_t if fels_t > 0 else float('nan')
        tva_class = tva.feasibility_class if tva else '?'
        print(f"{sim:>6} {nl:>6} {br:>8} {fels_t:>8.4f} {tva_t:>8.4f} "
              f"{ratio:>7.1f}x {tva_class:>9}")


def save_results(results, machine_info, total_time, output_path):
    """Save results to JSON."""
    data = {
        'machine_info': machine_info,
        'total_time': total_time,
        'cells': [],
    }
    for cell in results:
        cell_data = {
            'config': cell.config,
            'feasibility_class': cell.feasibility_class,
            'summary': cell.summary(),
            'replicates': [asdict(r) for r in cell.replicates],
        }
        data['cells'].append(cell_data)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Simulation-reconstruction benchmark')
    parser.add_argument('--budget', default='5m',
                        help='Time budget: pilot, 5m, 10m, 1h (default: 5m)')
    parser.add_argument('--output', default=None,
                        help='Output JSON path (default: auto)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    # Parse budget
    budget_map = {
        'pilot': 120,
        '5m': 300,
        '10m': 600,
        '1h': 3600,
    }
    if args.budget in budget_map:
        budget = budget_map[args.budget]
    else:
        budget = int(args.budget.rstrip('s'))

    # Build grid
    grid_map = {
        'pilot': build_pilot_grid,
        '5m': build_5min_grid,
        '10m': build_10min_grid,
        '1h': build_10min_grid,  # same grid, more time
    }
    configs = grid_map.get(args.budget, build_pilot_grid)()

    verbose = not args.quiet

    # Collect machine info
    machine_info = collect_machine_info()
    if verbose:
        print("Machine info:")
        for k, v in machine_info.items():
            print(f"  {k}: {v}")
        print()

    # Run benchmark
    results, total_time = run_benchmark(configs, budget_seconds=budget, verbose=verbose)

    # Report
    print_summary_table(results)

    # Save
    output_path = args.output or f"experiments/sim_recon_{args.budget}_{int(time.time())}.json"
    save_results(results, machine_info, total_time, output_path)


if __name__ == '__main__':
    main()
