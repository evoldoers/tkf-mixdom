"""Adapter for partition-conditioned ancestral reconstruction in the
unified reconstruction benchmark.

This module exposes `run_partition_reconstruction_method`, whose
signature matches the conventions used by the existing methods in
`experiments/unified_reconstruction_benchmark.py` (Felsenstein,
TKF92 beam, MixDom beam). It does NOT modify the unified benchmark
itself: the unified benchmark can import and call this helper as an
optional additional method, or a separate driver
(`partition_recon_benchmark.py`) can use it directly.

The adapter supports a single-domain configuration (TKF92 on LG08,
analogous to the TKF92 beam baseline) and a multi-domain configuration
using per-domain rate matrices supplied by the caller.
"""

import time
import copy
import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence

from tkfmixdom.jax.tree.partition_recon import (
    PartitionReconModel, build_inputs, partition_recon_forward_backward,
)
from tkfmixdom.jax.tree.partition_recon_jax import (
    partition_recon_forward_backward_jax,
)
from tkfmixdom.jax.tree.tree_varanc import (
    infer_internal_presence, name_internal_nodes,
)
from tkfmixdom.jax.core.protein import rate_matrix_lg
from experiments.reconstruct_util import reroot_at_node


@dataclass
class PartitionReconConfig:
    """Small bundle of knobs exposed to the benchmark driver."""
    kappa_top: float = 0.95
    use_jax: bool = True


def _prune_leaf_keep_parent(tree, leaf_name):
    """Remove a leaf but keep its parent node, re-rooting the tree at that parent.

    Returns the re-rooted tree with the target leaf removed. The root of
    the returned tree is the target's former parent node.
    """
    new_tree = copy.deepcopy(tree)
    name_internal_nodes(new_tree)
    for node in new_tree.preorder():
        if node.is_leaf and node.name == leaf_name:
            parent = node.parent
            if parent is None:
                raise ValueError(f"Target '{leaf_name}' is the root")
            rerooted = reroot_at_node(new_tree, parent)
            rerooted.children = [c for c in rerooted.children
                                 if not (c.is_leaf and c.name == leaf_name)]
            return rerooted
    return tree  # leaf not found, return unchanged


def default_single_domain_model(kappa_top: float = 0.95,
                                 ins_rate: float = 0.046,
                                 del_rate: float = 0.054,
                                 ext: float = 0.68) -> PartitionReconModel:
    """Default single-domain TKF92 + LG08 model matching the benchmark's
    TKF92 baseline parameters.
    """
    Q, pi = rate_matrix_lg()
    Q = np.asarray(Q)
    pi = np.asarray(pi)
    return PartitionReconModel(
        kappa_top=float(kappa_top),
        dom_weights=np.array([1.0]),
        dom_ins_rates=np.array([ins_rate]),
        dom_del_rates=np.array([del_rate]),
        ext_rates=np.array([ext]),
        Q=np.stack([Q]),
        pi=np.stack([pi]),
    )


def load_mixdom_model(checkpoint_path: str,
                      kappa_top: Optional[float] = None) -> PartitionReconModel:
    """Load a trained MixDom checkpoint (e.g. `pfam/svi_bw_d3f1_full_best_val.npz`)
    as a `PartitionReconModel`, extracting $\\kappa_0 = \\lambda_0/\\mu_0$
    automatically.

    Thin wrapper around `PartitionReconModel.from_mixdom_checkpoint`.
    See that method's docstring for details on how the vanishing
    top-level-rate limit is applied.
    """
    return PartitionReconModel.from_mixdom_checkpoint(
        checkpoint_path, kappa_top=kappa_top)


def mixdom_model_from_params(params: dict,
                             kappa_top: Optional[float] = None
                             ) -> PartitionReconModel:
    """Build a PartitionReconModel from an already-loaded MixDom params
    dict (the output of `tkfmixdom.jax.distill.maraschino.load_params`).

    Use this from the unified reconstruction benchmark to avoid loading
    the same checkpoint twice: the benchmark already calls
    `load_params(...)` to build the MixDom beam data, and this helper
    reuses that exact dict to build the partition-recon model.
    """
    return PartitionReconModel.from_mixdom_params(params, kappa_top=kappa_top)


# ---------------------------------------------------------------------------
# Drop-in integration with unified_reconstruction_benchmark.py
# ---------------------------------------------------------------------------
#
# The partition-recon method can be added to
# `experiments/unified_reconstruction_benchmark.py` with three small
# additions, all inside `main()`, none of which modify existing code
# paths:
#
# 1. At model loading time, reuse the already-loaded `params_d3` dict
#    to build a partition-recon model:
#
#        from experiments.partition_recon_adapter import (
#            mixdom_model_from_params, run_partition_reconstruction_method,
#            PartitionReconConfig,
#        )
#        partition_model_d3 = mixdom_model_from_params(params_d3)
#        partition_config = PartitionReconConfig(use_jax=True)
#
# 2. Per family, after the existing `=== 1. Felsenstein LG08 ===` block
#    (and before `=== 2. MixDom d3f1 beam ===`), add:
#
#        # === 1b. Partition-conditioned reconstruction (d3f1) ===
#        try:
#            part_pred, part_time = run_partition_reconstruction_method(
#                tree, held_out, remaining, msa, C,
#                model=partition_model_d3, config=partition_config)
#            part_score = score_prediction(
#                part_pred, true_seq, log_chi_s, st_s, sub_s, pi_s)
#            result['partition_d3f1'] = {
#                **part_score, 'time': float(part_time),
#                'pred_seq': [int(x) for x in part_pred],
#            }
#        except Exception as e:
#            log(f'  {fam}: partition_d3f1 error: {e}')
#            result['partition_d3f1'] = {'accuracy': -1.0, 'time': 0.0}
#
# 3. Extend `method_keys` and `method_labels` in the summary section:
#
#        method_keys   = [..., 'partition_d3f1']
#        method_labels = [..., 'Partition d3f1']
#
# That is the entire diff. The adapter reuses the same
# `score_prediction`, Q_lg, pi_lg, and TKF92 scoring pair HMM, and the
# checkpoint is loaded exactly once via the existing `load_params`
# call.


def run_partition_reconstruction_method(tree,
                                        held_out: str,
                                        remaining: Sequence[str],
                                        msa: dict,
                                        C: int,
                                        model: Optional[PartitionReconModel] = None,
                                        config: Optional[PartitionReconConfig] = None,
                                        ):
    """Run partition-conditioned reconstruction for one held-out leaf.

    This function matches the shape of `run_felsenstein` in
    `unified_reconstruction_benchmark.py`: given the full tree, the
    held-out leaf name, the list of remaining leaves, the aligned MSA
    (dict {name: (C,) int array with -1 for gaps}), and the number of
    MSA columns C, it prunes the held-out leaf and runs the
    partition-recon algorithm on the pruned tree + remaining MSA.

    Returns:
        pred_seq: (K,) int array of predicted residues (gap-stripped).
        elapsed: wall time in seconds.
    """
    if config is None:
        config = PartitionReconConfig()
    if model is None:
        model = default_single_domain_model(kappa_top=config.kappa_top)

    t0 = time.time()
    pruned = _prune_leaf_keep_parent(tree, held_out)
    name_internal_nodes(pruned)

    pruned_leaf_names = {l.name for l in pruned.leaves()}
    pruned_msa = {k: v for k, v in msa.items()
                  if k in pruned_leaf_names and k in remaining}

    inputs = build_inputs(pruned, pruned_msa)

    if config.use_jax:
        result = partition_recon_forward_backward_jax(inputs, model)
    else:
        result = partition_recon_forward_backward(inputs, model)

    L = result.root_residue_map.shape[0]
    pred_seq = np.array(
        [int(result.root_residue_map[c])
         for c in range(L) if result.root_residue_map[c] >= 0],
        dtype=np.int32,
    )
    elapsed = time.time() - t0
    return pred_seq, elapsed


def partition_reconstruction_result(tree,
                                    held_out: str,
                                    remaining: Sequence[str],
                                    msa: dict,
                                    C: int,
                                    model: Optional[PartitionReconModel] = None,
                                    config: Optional[PartitionReconConfig] = None,
                                    ):
    """Richer variant that also returns the full `PartitionReconResult`
    (with posteriors, log_Z, etc.) and the pruned tree. Useful when the
    caller wants to inspect per-column class posteriors alongside the
    residue predictions.
    """
    if config is None:
        config = PartitionReconConfig()
    if model is None:
        model = default_single_domain_model(kappa_top=config.kappa_top)

    t0 = time.time()
    pruned = _prune_leaf_keep_parent(tree, held_out)
    name_internal_nodes(pruned)
    pruned_leaf_names = {l.name for l in pruned.leaves()}
    pruned_msa = {k: v for k, v in msa.items()
                  if k in pruned_leaf_names and k in remaining}
    inputs = build_inputs(pruned, pruned_msa)
    if config.use_jax:
        result = partition_recon_forward_backward_jax(inputs, model)
    else:
        result = partition_recon_forward_backward(inputs, model)
    L = result.root_residue_map.shape[0]
    pred_seq = np.array(
        [int(result.root_residue_map[c])
         for c in range(L) if result.root_residue_map[c] >= 0],
        dtype=np.int32,
    )
    elapsed = time.time() - t0
    return pred_seq, elapsed, result, pruned
