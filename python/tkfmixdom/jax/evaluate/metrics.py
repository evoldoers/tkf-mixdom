"""Centralized scoring functions for reconstruction and alignment evaluation.

All benchmarks and experiment scripts should import scoring functions from
this module to ensure consistent metrics across methods and datasets.

Metrics:
  - needleman_wunsch_identity: NW alignment-based sequence identity
  - nw_metrics: NW accuracy/precision/recall/F1 (CARABS-compatible)
  - root_accuracy: Per-column accuracy of reconstructed root vs true root
  - sp_score: Sum-of-pairs alignment accuracy
  - tc_score: Total-column alignment accuracy
  - domain_accuracy: Domain assignment accuracy
  - frag_accuracy: Fragment state accuracy
  - class_accuracy: Site class accuracy
"""

import numpy as np
from typing import Dict, Tuple, Optional


def needleman_wunsch_identity(seq1, seq2,
                               match_score=1, mismatch=-1, gap=-2):
    """NW global alignment, return (identity, n_aligned, n_matches).

    Args:
        seq1, seq2: sequences (lists/arrays of comparable elements).
        match_score: score for a match.
        mismatch: score for a mismatch.
        gap: gap penalty.

    Returns:
        identity: matches / aligned (0 if no aligned positions).
        n_aligned: number of aligned (non-gap) position pairs.
        n_matches: number of matching positions.
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return 0.0, 0, 0
    dp = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match_score if seq1[i - 1] == seq2[j - 1] else mismatch
            dp[i, j] = max(dp[i - 1, j - 1] + s,
                           dp[i - 1, j] + gap,
                           dp[i, j - 1] + gap)
    # Traceback
    i, j = n, m
    matches = aligned = 0
    while i > 0 and j > 0:
        s = match_score if seq1[i - 1] == seq2[j - 1] else mismatch
        if dp[i, j] == dp[i - 1, j - 1] + s:
            aligned += 1
            if seq1[i - 1] == seq2[j - 1]:
                matches += 1
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i - 1, j] + gap:
            i -= 1
        else:
            j -= 1
    return matches / max(aligned, 1), aligned, matches


def nw_metrics(pred_seq, true_seq) -> Dict[str, float]:
    """Compute NW-based accuracy/precision/recall/F1.

    Compatible with CARABS scoring conventions.

    Args:
        pred_seq: predicted sequence (list/array of residue indices).
        true_seq: true sequence (list/array of residue indices).

    Returns:
        dict with nw_accuracy, nw_precision, nw_recall, nw_f1,
        nw_matches, nw_aligned.
    """
    nw_id, nw_aligned, nw_matches = needleman_wunsch_identity(
        pred_seq, true_seq)
    pred_len = len(pred_seq)
    true_len = len(true_seq)
    prec = nw_matches / max(pred_len, 1)
    rec = nw_matches / max(true_len, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    return {
        'nw_accuracy': float(nw_id),
        'nw_precision': float(prec),
        'nw_recall': float(rec),
        'nw_f1': float(f1),
        'nw_matches': int(nw_matches),
        'nw_aligned': int(nw_aligned),
    }


def root_accuracy(pred_root: np.ndarray, true_root: np.ndarray,
                  mask: Optional[np.ndarray] = None) -> float:
    """Per-column accuracy of reconstructed root vs true root.

    Args:
        pred_root: (L,) int array of predicted root residues.
        true_root: (L,) int array of true root residues.
        mask: optional (L,) bool array — only score where True.

    Returns:
        Fraction of (masked) columns where pred == true.
    """
    if mask is not None:
        pred_root = pred_root[mask]
        true_root = true_root[mask]
    if len(pred_root) == 0:
        return 0.0
    return float(np.mean(pred_root == true_root))


def sp_score(pred_msa: Dict[str, np.ndarray],
             true_msa: Dict[str, np.ndarray],
             names: Optional[list] = None) -> float:
    """Sum-of-pairs alignment accuracy.

    For each pair of sequences, count the fraction of homologous residue
    pairs that are correctly identified in the predicted alignment.

    Args:
        pred_msa: {name: (L_pred,) int array, -1=gap} predicted alignment.
        true_msa: {name: (L_true,) int array, -1=gap} true alignment.
        names: sequence names to compare (default: intersection of keys).

    Returns:
        SP score in [0, 1].
    """
    if names is None:
        names = sorted(set(pred_msa) & set(true_msa))
    if len(names) < 2:
        return 0.0

    total_true_pairs = 0
    correct_pairs = 0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            # Extract pairwise alignment from true MSA
            true_pairs = set()
            pi = pj = 0
            for col in range(len(true_msa[ni])):
                gi = true_msa[ni][col] >= 0
                gj = true_msa[nj][col] >= 0
                if gi and gj:
                    true_pairs.add((pi, pj))
                if gi:
                    pi += 1
                if gj:
                    pj += 1

            # Extract pairwise alignment from pred MSA
            pred_pairs = set()
            pi = pj = 0
            for col in range(len(pred_msa[ni])):
                gi = pred_msa[ni][col] >= 0
                gj = pred_msa[nj][col] >= 0
                if gi and gj:
                    pred_pairs.add((pi, pj))
                if gi:
                    pi += 1
                if gj:
                    pj += 1

            total_true_pairs += len(true_pairs)
            correct_pairs += len(true_pairs & pred_pairs)

    return correct_pairs / max(total_true_pairs, 1)


def tc_score(pred_msa: Dict[str, np.ndarray],
             true_msa: Dict[str, np.ndarray],
             names: Optional[list] = None) -> float:
    """Total-column alignment accuracy.

    Fraction of true alignment columns that are exactly reproduced in
    the predicted alignment.

    Args:
        pred_msa: {name: (L_pred,) int array} predicted alignment.
        true_msa: {name: (L_true,) int array} true alignment.
        names: sequence names to compare.

    Returns:
        TC score in [0, 1].
    """
    if names is None:
        names = sorted(set(pred_msa) & set(true_msa))
    if len(names) < 2:
        return 0.0

    # Build column signatures for true and predicted MSAs.
    def column_sigs(msa, names):
        L = len(next(iter(msa.values())))
        pos = {n: 0 for n in names}
        sigs = []
        for col in range(L):
            sig = []
            for n in names:
                if msa[n][col] >= 0:
                    sig.append(pos[n])
                    pos[n] += 1
                else:
                    sig.append(-1)
            sigs.append(tuple(sig))
        return set(sigs)

    true_sigs = column_sigs(true_msa, names)
    pred_sigs = column_sigs(pred_msa, names)

    # Only count true columns where all sequences are present.
    true_full = {s for s in true_sigs if all(x >= 0 for x in s)}
    if len(true_full) == 0:
        return 0.0
    return len(true_full & pred_sigs) / len(true_full)


def domain_accuracy(pred_domains: np.ndarray,
                    true_domains: np.ndarray) -> float:
    """Domain assignment accuracy (per-column)."""
    if len(pred_domains) != len(true_domains):
        return 0.0
    if len(pred_domains) == 0:
        return 0.0
    return float(np.mean(pred_domains == true_domains))


def frag_accuracy(pred_frags: np.ndarray,
                  true_frags: np.ndarray) -> float:
    """Fragment state accuracy (per-column)."""
    if len(pred_frags) != len(true_frags):
        return 0.0
    if len(pred_frags) == 0:
        return 0.0
    return float(np.mean(pred_frags == true_frags))


def class_accuracy(pred_classes: np.ndarray,
                   true_classes: np.ndarray) -> float:
    """Site class accuracy (per-column)."""
    if len(pred_classes) != len(true_classes):
        return 0.0
    if len(pred_classes) == 0:
        return 0.0
    return float(np.mean(pred_classes == true_classes))
