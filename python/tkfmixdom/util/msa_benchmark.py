"""Minimal MSA-benchmark utilities (parse_fasta, sp_tc_score).

Extracted from `maraschino-paper/benchmarks/run_msa_benchmark.py` (the
abandoned-paper directory) into the main codebase so we can delete
`maraschino-paper/`. These two functions are everything that 10
benchmark scripts in `python/experiments/` actually use from there.
"""
from __future__ import annotations


def parse_fasta(filepath: str) -> dict[str, str]:
    """Parse FASTA file, return dict of {name: sequence}."""
    seqs: dict[str, str] = {}
    name: str | None = None
    seq_parts: list[str] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    seqs[name] = ''.join(seq_parts)
                name = line[1:].split()[0]
                seq_parts = []
            elif name is not None:
                seq_parts.append(line)
    if name is not None:
        seqs[name] = ''.join(seq_parts)
    return seqs


def sp_tc_score(test_aln: dict[str, str], ref_aln: dict[str, str],
                core_only: bool = True) -> tuple[float, float]:
    """Compute SP and TC scores of test alignment against reference.

    SP score: fraction of correctly aligned pairs in the reference that
              are also aligned in the test.
    TC score: fraction of reference columns that are exactly reproduced
              in the test.

    For BAliBASE: only uppercase (core) positions are scored.

    Args:
        test_aln: dict of {name: aligned_seq_with_gaps}
        ref_aln: dict of {name: aligned_seq_with_gaps}
        core_only: if True, only score uppercase positions in reference

    Returns:
        sp: SP score (0-1)
        tc: TC score (0-1)
    """
    common_names = sorted(set(test_aln.keys()) & set(ref_aln.keys()))
    if len(common_names) < 2:
        return 0.0, 0.0

    def _pos_to_col(aligned_seq: str) -> dict[int, int]:
        """Map ungapped position index to alignment column."""
        mapping: dict[int, int] = {}
        pos = 0
        for col, c in enumerate(aligned_seq):
            if c not in '.-':
                mapping[pos] = col
                pos += 1
        return mapping

    # Identify reference core positions (uppercase in original ref).
    ref_core: dict[str, set[int]] = {}
    for name in common_names:
        core_positions: set[int] = set()
        pos = 0
        for c in ref_aln[name]:
            if c not in '.-':
                if c.isupper() or not core_only:
                    core_positions.add(pos)
                pos += 1
        ref_core[name] = core_positions

    ref_pairs = 0
    test_correct_pairs = 0
    test_pos_to_col = {n: _pos_to_col(test_aln[n]) for n in common_names}
    ref_pos_to_col = {n: _pos_to_col(ref_aln[n]) for n in common_names}

    for i, name_a in enumerate(common_names):
        for name_b in common_names[i + 1:]:
            ref_col_a = ref_pos_to_col[name_a]
            ref_col_b = ref_pos_to_col[name_b]

            ref_cols_a: dict[int, list[int]] = {}
            for pos, col in ref_col_a.items():
                if pos in ref_core[name_a]:
                    ref_cols_a.setdefault(col, []).append(pos)

            ref_cols_b: dict[int, list[int]] = {}
            for pos, col in ref_col_b.items():
                if pos in ref_core[name_b]:
                    ref_cols_b.setdefault(col, []).append(pos)

            for col in set(ref_cols_a.keys()) & set(ref_cols_b.keys()):
                for pos_a in ref_cols_a[col]:
                    for pos_b in ref_cols_b[col]:
                        ref_pairs += 1
                        test_col_a = test_pos_to_col[name_a].get(pos_a)
                        test_col_b = test_pos_to_col[name_b].get(pos_b)
                        if test_col_a is not None and test_col_b is not None:
                            if test_col_a == test_col_b:
                                test_correct_pairs += 1

    sp = test_correct_pairs / max(ref_pairs, 1)

    # TC score: fraction of reference columns exactly reproduced.
    ref_len = len(next(iter(ref_aln.values())))
    total_core_cols = 0
    correct_cols = 0

    for col in range(ref_len):
        ref_col_chars: dict[str, str] = {}
        is_core = False
        for name in common_names:
            if col < len(ref_aln[name]):
                c = ref_aln[name][col]
                if c not in '.-':
                    if c.isupper() or not core_only:
                        is_core = True
                        ref_col_chars[name] = c.upper()

        if not is_core or len(ref_col_chars) < 2:
            continue
        total_core_cols += 1

        ref_positions: dict[str, int] = {}
        for name in ref_col_chars:
            pos = 0
            for k in range(col):
                if ref_aln[name][k] not in '.-':
                    pos += 1
            ref_positions[name] = pos

        test_cols: set[int] = set()
        all_found = True
        for name, pos in ref_positions.items():
            tc = test_pos_to_col[name].get(pos)
            if tc is None:
                all_found = False
                break
            test_cols.add(tc)

        if all_found and len(test_cols) == 1:
            correct_cols += 1

    tc = correct_cols / max(total_core_cols, 1)
    return sp, tc
