"""Compact X/A/Y encoding for pre-compiled MixDom training pairs.

Each training pair is encoded as:
  X: ancestor amino acid sequence (e.g. "ACDEFG")
  A: run-length encoded alignment path (e.g. "M5I1M3D2M7")
  Y: descendant amino acid sequence

The RLE alignment path compactly represents the M/I/D state sequence.
Constraint: sum of M+D run lengths = len(X), sum of M+I run lengths = len(Y).

Usage:
    from tkfmixdom.jax.util.pair_format import encode_pair, decode_pair

    record = encode_pair(anc_seq, desc_seq, alignment_states, t_est, "PF00001")
    x_int, y_int, states, anc_chars, desc_chars, t_est, gamma, gamma_lp = decode_pair(record)
"""

import hashlib
import re

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA = len(AMINO_ACIDS)
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: a for i, a in enumerate(AMINO_ACIDS)}

# State codes matching tkfmixdom.jax.core.params
M_STATE = 1  # Match
I_STATE = 2  # Insert
D_STATE = 3  # Delete

_STATE_TO_CHAR = {M_STATE: 'M', I_STATE: 'I', D_STATE: 'D'}
_CHAR_TO_STATE = {'M': M_STATE, 'I': I_STATE, 'D': D_STATE}


def rle_encode(states):
    """Run-length encode M/I/D state list.

    Args:
        states: list of state codes (1=M, 2=I, 3=D)

    Returns:
        RLE string, e.g. "M3I1M1D2"

    Examples:
        >>> rle_encode([1, 1, 1, 2, 1, 3, 3])
        'M3I1M1D2'
        >>> rle_encode([1])
        'M1'
    """
    if not states:
        return ''
    parts = []
    current = states[0]
    count = 1
    for s in states[1:]:
        if s == current:
            count += 1
        else:
            parts.append(f'{_STATE_TO_CHAR[current]}{count}')
            current = s
            count = 1
    parts.append(f'{_STATE_TO_CHAR[current]}{count}')
    return ''.join(parts)


def rle_decode(rle_string):
    """Decode RLE string back to state list.

    Args:
        rle_string: e.g. "M3I1M1D2"

    Returns:
        list of state codes (1=M, 2=I, 3=D)

    Examples:
        >>> rle_decode('M3I1M1D2')
        [1, 1, 1, 2, 1, 3, 3]
    """
    if not rle_string:
        return []
    states = []
    for match in re.finditer(r'([MID])(\d+)', rle_string):
        char, count = match.group(1), int(match.group(2))
        states.extend([_CHAR_TO_STATE[char]] * count)
    return states


def encode_pair(anc_seq, desc_seq, alignment_states, t_est, family_id,
                row1_name='', row2_name='',
                gamma_labels=None, gamma_log_posteriors=None):
    """Encode an aligned pair as compact X/A/Y string + metadata.

    Args:
        anc_seq: ancestor amino acid characters (list of ints, 0..19)
        desc_seq: descendant amino acid characters (list of ints, 0..19)
        alignment_states: list of M/I/D state codes (from alignment_to_states)
        t_est: estimated evolutionary time (float)
        family_id: Pfam family accession (e.g. "PF00001")
        row1_name: optional name of ancestor row
        row2_name: optional name of descendant row
        gamma_labels: optional list of int (0..G-1 or -1 for uninformative),
            one per alignment column. Per-column MAP gamma rate category.
        gamma_log_posteriors: optional list of list of float (G x n_cols),
            normalized log-posteriors over gamma categories per column.

    Returns:
        dict with keys: 'x', 'a', 'y', 't', 'fam', 'id'
        and optionally 'gamma' and 'gamma_lp'
    """
    # Build ancestor string from anc_chars
    x = ''.join(IDX_TO_AA[int(c)] for c in anc_seq)
    # Build descendant string from desc_chars
    y = ''.join(IDX_TO_AA[int(c)] for c in desc_seq)
    # RLE encode alignment
    a = rle_encode(alignment_states)

    # Deterministic pair ID
    id_str = f'{family_id}:{row1_name}:{row2_name}'
    pair_id = hashlib.sha256(id_str.encode()).hexdigest()[:16]

    record = {
        'x': x,
        'a': a,
        'y': y,
        't': round(float(t_est), 6),
        'fam': family_id,
        'id': pair_id,
    }

    if gamma_labels is not None:
        record['gamma'] = [int(g) for g in gamma_labels]
    if gamma_log_posteriors is not None:
        # Store as list of lists, rounded for compactness
        record['gamma_lp'] = [[round(float(v), 4) for v in row]
                               for row in gamma_log_posteriors]

    return record


def decode_pair(record):
    """Decode a record back to arrays ready for mixdom_constrained_emissions.

    Args:
        record: dict with keys 'x', 'a', 'y', 't', 'fam', 'id'
            and optionally 'gamma' (MAP labels) and 'gamma_lp' (log posteriors)

    Returns:
        (x_int, y_int, states, anc_chars, desc_chars, t_est, gamma_labels, gamma_lp)
        where:
          x_int: np.array of ancestor character indices (0..19)
          y_int: np.array of descendant character indices (0..19)
          states: list of state codes (M=1, I=2, D=3)
          anc_chars: list of ancestor chars (for M and D positions)
          desc_chars: list of descendant chars (for M and I positions)
          t_est: float evolutionary time
          gamma_labels: list of int or None (per-column MAP gamma category)
          gamma_lp: list of list of float or None (G x n_cols log posteriors)
    """
    x_str = record['x']
    y_str = record['y']
    a_str = record['a']
    t_est = float(record['t'])

    x_int = np.array([AA_TO_IDX[c] for c in x_str], dtype=np.int32)
    y_int = np.array([AA_TO_IDX[c] for c in y_str], dtype=np.int32)
    states = rle_decode(a_str)

    # Reconstruct anc_chars and desc_chars from states + sequences
    anc_chars = []
    desc_chars = []
    xi, yi = 0, 0
    for s in states:
        if s == M_STATE:
            anc_chars.append(int(x_int[xi]))
            desc_chars.append(int(y_int[yi]))
            xi += 1
            yi += 1
        elif s == I_STATE:
            desc_chars.append(int(y_int[yi]))
            yi += 1
        elif s == D_STATE:
            anc_chars.append(int(x_int[xi]))
            xi += 1

    # Gamma rate annotations (optional, backward compatible)
    gamma_labels = record.get('gamma', None)
    gamma_lp = record.get('gamma_lp', None)

    return x_int, y_int, states, anc_chars, desc_chars, t_est, gamma_labels, gamma_lp


def validate_record(record):
    """Validate a pre-compiled record for consistency.

    Returns (ok, error_message).
    """
    try:
        x_str = record['x']
        y_str = record['y']
        a_str = record['a']

        states = rle_decode(a_str)
        n_m = sum(1 for s in states if s == M_STATE)
        n_i = sum(1 for s in states if s == I_STATE)
        n_d = sum(1 for s in states if s == D_STATE)

        if n_m + n_d != len(x_str):
            return False, f"M+D={n_m+n_d} != len(X)={len(x_str)}"
        if n_m + n_i != len(y_str):
            return False, f"M+I={n_m+n_i} != len(Y)={len(y_str)}"

        for c in x_str:
            if c not in AA_TO_IDX:
                return False, f"Invalid AA in X: {c}"
        for c in y_str:
            if c not in AA_TO_IDX:
                return False, f"Invalid AA in Y: {c}"

        return True, ''
    except Exception as e:
        return False, str(e)
