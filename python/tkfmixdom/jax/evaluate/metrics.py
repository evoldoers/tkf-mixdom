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



def _int_msa_to_str(msa, names):
    """-1 = gap, anything else = residue; all residues upper-case so that
    every column is scored (core_only=False)."""
    return {n: ''.join('-' if int(x) < 0 else 'A' for x in np.asarray(msa[n])) for n in names}


def sp_score(pred_msa: Dict[str, np.ndarray],
             true_msa: Dict[str, np.ndarray],
             names: Optional[list] = None) -> float:
    """Sum-of-pairs score of ``pred_msa`` against ``true_msa``.

    Thin wrapper over the single authoritative scorer,
    :func:`tkfmixdom.util.msa_benchmark.sp_tc_score` (qscore / BAliBASE
    semantics; every column of the reference is scored, there is no core
    annotation in an int-array MSA).

    Args:
        pred_msa: {name: (L_pred,) int array, -1=gap} predicted alignment.
        true_msa: {name: (L_true,) int array, -1=gap} true alignment.
        names: sequence names to compare (default: intersection of keys).
    """
    from tkfmixdom.util.msa_benchmark import sp_tc_score
    if names is None:
        names = sorted(set(pred_msa) & set(true_msa))
    if len(names) < 2:
        return 0.0
    sp, _ = sp_tc_score(_int_msa_to_str(pred_msa, names), _int_msa_to_str(true_msa, names), core_only=False)
    return sp



def tc_score(pred_msa: Dict[str, np.ndarray],
             true_msa: Dict[str, np.ndarray],
             names: Optional[list] = None) -> float:
    """Total-column score of ``pred_msa`` against ``true_msa``.

    Thin wrapper over :func:`tkfmixdom.util.msa_benchmark.sp_tc_score`
    (qscore / BAliBASE semantics: a reference column with >= 2 residues is
    correct when all of its residues fall in one predicted column; predicted
    residues from sequences gapped in that reference column are not examined).
    """
    from tkfmixdom.util.msa_benchmark import sp_tc_score
    if names is None:
        names = sorted(set(pred_msa) & set(true_msa))
    if len(names) < 2:
        return 0.0
    _, tc = sp_tc_score(_int_msa_to_str(pred_msa, names), _int_msa_to_str(true_msa, names), core_only=False)
    return tc


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
