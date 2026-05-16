"""Grammar compilation: null removal and count restoration.

The compile step is model-independent post-processing that converts a raw
grammar (with nullable nonterminals and null cycles) into a DP-ready form,
and provides the inverse operation to restore counts after DP.

Two forms of null removal, applied in order:
1. Nullability removal: eliminate ε-producing nonterminals
2. Null cycle removal: eliminate cycles among null states via (I - T_ZZ)^{-1}

Count restoration is the inverse, applied in reverse order:
1. Restore null cycle counts (un-eliminate phantom BDI events)
2. Restore ε-nonterminal counts (redistribute counts by nullability weights)
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from ..core.types import NullInfo


# ---------------------------------------------------------------------------
# Null cycle removal and restoration (HMM case)
# ---------------------------------------------------------------------------

def build_null_info_hmm(upsilon: np.ndarray,
                        null_indices: list[int],
                        visible_map: dict[int, int]) -> NullInfo:
    """Build NullInfo from an exploded transition matrix with null states.

    Args:
        upsilon: (K, K) exploded transition matrix including null states
        null_indices: indices of null states in upsilon (e.g. [5, 6] for MixDom)
        visible_map: maps null state index -> visible state it folds to
                     (e.g. {5: 1, 6: 3} for M_null->M, D_null->D)

    Returns:
        NullInfo with closure and decomposition matrices.
    """
    K = upsilon.shape[0]
    n_null = len(null_indices)
    non_null = [i for i in range(K) if i not in null_indices]
    n_vis = len(non_null)

    # Extract sub-matrices
    T_NN = upsilon[np.ix_(non_null, non_null)]
    T_Nnull = upsilon[np.ix_(non_null, null_indices)]
    T_nullN = upsilon[np.ix_(null_indices, non_null)]
    T_nullnull = upsilon[np.ix_(null_indices, null_indices)]

    # Null closure
    I_null = np.eye(n_null)
    closure = np.linalg.inv(I_null - T_nullnull)

    # Null contribution to effective transitions
    null_contrib = T_Nnull @ closure @ T_nullN

    return NullInfo(
        null_state_indices=null_indices,
        upsilon=upsilon,
        closure=closure,
        T_NN=T_NN,
        T_Nnull=T_Nnull,
        T_nullN=T_nullN,
        null_contrib=null_contrib,
    )


def effective_trans_from_null_info(null_info: NullInfo) -> np.ndarray:
    """Compute effective transition matrix with nulls removed.

    T_eff = T_NN + T_Nnull @ (I - T_ZZ)^{-1} @ T_ZN
    """
    return null_info.T_NN + null_info.null_contrib


def restore_null_counts(top_counts: np.ndarray,
                        null_info: NullInfo,
                        visible_map: dict[int, int] | None = None,
                        return_full: bool = False) -> np.ndarray:
    """Restore null cycle counts from effective transition counts.

    Given expected counts on the effective T matrix (nulls removed),
    distributes counts back through the null closure to recover the
    "phantom" BDI events from null cycles.

    This is the inverse of null cycle removal: for each effective transition
    T_eff[u,v], a fraction came from direct paths (u->v) and a fraction from
    null-mediated paths (u->null->...->null->v). The null-mediated fraction
    represents phantom events that must be counted for correct BDI stats.

    Args:
        top_counts: (N_vis, N_vis) expected counts on effective T matrix
        null_info: NullInfo from build_null_info_hmm
        visible_map: maps null state index -> visible state index it folds to.
                     If None, defaults to MixDom convention: {5: M, 6: D}

    Returns:
        If return_full=False: adjusted_counts (N_vis, N_vis) with null states folded.
        If return_full=True: n_full (K, K) counts on the exploded matrix (K = N_vis + N_null).
    """
    T_NN = null_info.T_NN
    T_Nnull = null_info.T_Nnull
    T_nullN = null_info.T_nullN
    closure = null_info.closure
    null_contrib = null_info.null_contrib
    T_nullnull = null_info.upsilon[
        np.ix_(null_info.null_state_indices, null_info.null_state_indices)]

    T_eff = T_NN + null_contrib
    n_vis = T_NN.shape[0]
    n_null = closure.shape[0]

    # Default visible map for MixDom: null states 5,6 map to M=1, D=3
    if visible_map is None:
        from ..core.params import M, D
        visible_map = {0: M, 1: D}  # indices within null sub-array

    # Build full counts on exploded matrix
    K = n_vis + n_null
    n_full = np.zeros((K, K))

    for u in range(n_vis):
        for v in range(n_vis):
            if T_eff[u, v] < 1e-30 or top_counts[u, v] < 1e-15:
                continue

            n_uv = top_counts[u, v]

            # Direct path fraction
            n_full[u, v] += n_uv * T_NN[u, v] / T_eff[u, v]

            # Null-mediated fraction
            n_null_total = n_uv * null_contrib[u, v] / T_eff[u, v]
            if n_null_total < 1e-15:
                continue

            # Which null state was entered first?
            closure_T_nullN = closure @ T_nullN
            entry_unnorm = np.array([
                T_Nnull[u, k] * closure_T_nullN[k, v] for k in range(n_null)
            ])
            entry_sum = entry_unnorm.sum()
            if entry_sum < 1e-30:
                continue
            p_entry = entry_unnorm / entry_sum

            # h_v[s] = P(eventually exit to v | at null state s)
            h_v = closure_T_nullN[:, v]

            for k in range(n_null):
                if p_entry[k] < 1e-15:
                    continue
                n_enter_k = n_null_total * p_entry[k]

                # Entry transition: visible u -> null k
                n_full[u, n_vis + k] += n_enter_k

                if h_v[k] < 1e-30:
                    continue

                # Null-to-null transitions (Doob h-transform: condition on exit to v)
                for l in range(n_null):
                    for lp in range(n_null):
                        n_full[n_vis + l, n_vis + lp] += (
                            n_enter_k * closure[k, l] * T_nullnull[l, lp]
                            * h_v[lp] / h_v[k])

                # Exit transition: null -> visible v
                for l in range(n_null):
                    exit_weight = closure[k, l] * T_nullN[l, v]
                    n_full[n_vis + l, v] += n_enter_k * exit_weight / h_v[k]

    if return_full:
        return n_full

    # Fold null states back to visible
    adjusted = np.zeros((n_vis, n_vis))
    for i in range(K):
        for j in range(K):
            if n_full[i, j] < 1e-15:
                continue
            vi = visible_map.get(i - n_vis, i) if i >= n_vis else i
            vj = visible_map.get(j - n_vis, j) if j >= n_vis else j
            adjusted[vi, vj] += n_full[i, j]

    return adjusted
