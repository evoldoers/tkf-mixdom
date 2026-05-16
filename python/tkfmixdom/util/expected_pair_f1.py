"""Expected pairwise sufficient statistics from posterior match probs.

Given pairwise match posteriors ``P[ri, rj]`` for each pair ``(i, j)``
of sequences, and a reference MSA defining true matches, compute the
minimal sufficient statistics for the soft (cell-level) confusion
matrix -- deterministically and exactly, without ever materializing
an MSA.

Per pair (and pooled across pairs / families), the module reports:

    e_tp        Expected true positives   Σ_{(ri,rj)∈T} P[ri, rj].
    total_mass  Σ P[ri, rj]               (= E[TP] + E[FP]).
    gold        |T|                        (= E[TP] + E[FN]).
    n_cells     Li · Lj                    (= sum of all four cells).

Every other quantity is derivable from these four:

    E[FP]          = total_mass - e_tp
    E[FN]          = gold - e_tp
    E[TN]          = n_cells - total_mass - gold + e_tp
    soft precision = e_tp / total_mass
    soft recall    = e_tp / gold
    presence F1    = 2 e_tp / (total_mass + gold)

All four quantities are *additive* across pairs (and across families),
so pooled stats are simple sums -- exposed as ``micro`` in
:func:`expected_family_f1` and :func:`aggregate_corpus`.

Positives at the cell level are the ``(ri, rj)`` pairs that the
reference MSA places in the same aligned column (typically core-only,
i.e. uppercase residues in both sequences).

This module is intentionally pure-numpy so it can be vendored into
``tkf-dp`` or any other caller without pulling in JAX.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple, Iterable, Any

import numpy as np


GAP_CHARS = '-.'


def ref_to_pair_truth(
    ref_aln: Mapping[str, str],
    name_i: str,
    name_j: str,
    core_only: bool = True,
) -> set[tuple[int, int]]:
    """Project a reference MSA onto pairwise truth for sequences (i, j).

    Returns the set of ``(ri, rj)`` 0-based index pairs that are aligned
    in the reference (i.e. share an alignment column where both residues
    are present and, if ``core_only``, both residues are uppercase).

    Indices are positions in the ungapped sequences. Gap characters are
    ``'-'`` and ``'.'``. With ``core_only=True``, the same residue-index
    convention is used (i.e. lowercase residues are counted in the
    ungapped position numbering, just not in the truth set), matching
    the standard BAliBASE convention used by
    :func:`tkfmixdom.util.msa_benchmark.sp_tc_score`.
    """
    s_i = ref_aln[name_i]
    s_j = ref_aln[name_j]
    if len(s_i) != len(s_j):
        raise ValueError(
            f'Reference alignment length mismatch: {name_i} '
            f'({len(s_i)}) vs {name_j} ({len(s_j)})')
    ri = -1
    rj = -1
    matches: set[tuple[int, int]] = set()
    for c_i, c_j in zip(s_i, s_j):
        is_res_i = c_i not in GAP_CHARS
        is_res_j = c_j not in GAP_CHARS
        if is_res_i:
            ri += 1
        if is_res_j:
            rj += 1
        if not (is_res_i and is_res_j):
            continue
        if core_only and not (c_i.isupper() and c_j.isupper()):
            continue
        matches.add((ri, rj))
    return matches


def expected_pair_f1(
    posterior: np.ndarray,
    true_matches: Iterable[Tuple[int, int]],
) -> dict[str, Any]:
    """Sufficient stats for one pair.

    Args:
        posterior: ``(Li, Lj)`` array, ``P[ri, rj]`` = posterior
            probability that residue ``ri`` of seq i is aligned to
            residue ``rj`` of seq j.
        true_matches: iterable of ``(ri, rj)`` tuples from the reference
            MSA (typically produced by :func:`ref_to_pair_truth`).
            Out-of-bounds indices are silently ignored for ``e_tp``
            but still increment ``gold``.

    Returns:
        ``{'e_tp', 'total_mass', 'gold', 'n_cells'}``. See module
        docstring for the derived-quantity identities.
    """
    p = np.asarray(posterior, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f'posterior must be 2D, got shape {p.shape}')
    Li, Lj = p.shape
    truth = set(true_matches)
    gold = len(truth)
    total_mass = float(p.sum())
    e_tp = 0.0
    for (ri, rj) in truth:
        if 0 <= ri < Li and 0 <= rj < Lj:
            e_tp += float(p[ri, rj])
    return {
        'e_tp': float(e_tp),
        'total_mass': float(total_mass),
        'gold': int(gold),
        'n_cells': int(Li) * int(Lj),
    }


def _aggregate_micro(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool sufficient stats across rows by summation."""
    return {
        'e_tp': float(sum(r['e_tp'] for r in rows)),
        'total_mass': float(sum(r['total_mass'] for r in rows)),
        'gold': int(sum(r['gold'] for r in rows)),
        'n_cells': int(sum(r.get('n_cells', 0) for r in rows)),
    }


def expected_family_f1(
    pair_posteriors: Mapping[Tuple[int, int], np.ndarray],
    ref_aln: Mapping[str, str],
    names: Sequence[str],
    core_only: bool = True,
) -> dict[str, Any]:
    """Aggregate expected sufficient stats over all pairs in one family.

    Args:
        pair_posteriors: ``{(i, j): P[ri, rj]}``, indexed by 0-based
            positions into ``names``.
        ref_aln: reference MSA, ``{name: gapped_sequence}``.
        names: list of sequence names; pair indices index this list.
            Pairs whose names are missing from ``ref_aln`` are skipped.
        core_only: restrict truth to uppercase reference residues.

    Returns:
        dict with:
            * ``per_pair``: per-pair sparse dicts (with ``'pair'``,
              ``'name_i'``, ``'name_j'`` added).
            * ``micro``: pooled sufficient stats across all pairs.
            * ``n_pairs``, ``n_pairs_skipped``.
    """
    per_pair = []
    skipped = 0
    for (i, j), p in pair_posteriors.items():
        name_i = names[i]
        name_j = names[j]
        if name_i not in ref_aln or name_j not in ref_aln:
            skipped += 1
            continue
        truth = ref_to_pair_truth(
            ref_aln, name_i, name_j, core_only=core_only)
        row = expected_pair_f1(p, truth)
        row['pair'] = (int(i), int(j))
        row['name_i'] = name_i
        row['name_j'] = name_j
        per_pair.append(row)
    return {
        'per_pair': per_pair,
        'micro': _aggregate_micro(per_pair),
        'n_pairs': int(len(per_pair)),
        'n_pairs_skipped': int(skipped),
    }


def aggregate_corpus(
    family_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine per-family sufficient stats into corpus-level pools.

    ``micro`` pools every per-pair sufficient-stat across all families
    by summation; any downstream derived quantity (F1, precision,
    recall, etc.) can be computed from these pooled sums via the
    identities in the module docstring.
    """
    all_pairs: list[Mapping[str, Any]] = []
    for fr in family_results:
        all_pairs.extend(fr.get('per_pair', []))
    return {
        'micro': _aggregate_micro(all_pairs),
        'n_families': int(len(family_results)),
        'n_pairs_total': int(len(all_pairs)),
    }
