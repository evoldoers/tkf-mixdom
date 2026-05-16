"""Geometric bin padding for JIT cache reuse.

All JIT-compiled DP functions pad inputs to geometric bin sizes so JAX
reuses compiled functions instead of recompiling for every distinct
sequence length.

Usage:
    from tkfmixdom.jax.util.padding import pad_to_bin, pad_sequence, GEOM_BINS
"""

import jax.numpy as jnp
import numpy as np

# Geometric bin sizes: 4, 6, 8, 12, 16, 24, 32, ...
GEOM_BINS = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
             384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]


def pad_to_bin(L):
    """Round up length L to next geometric bin for JIT cache reuse."""
    for b in GEOM_BINS:
        if b >= L:
            return b
    return L


def pad_sequence(seq, fill_value=0):
    """Pad a 1D integer sequence to the next geometric bin size.

    Args:
        seq: 1D array (numpy or jax)
        fill_value: padding value (default 0)

    Returns:
        (padded_seq, original_length)
    """
    L = len(seq)
    padded_L = pad_to_bin(L)
    if padded_L == L:
        return jnp.array(seq), L
    padded = jnp.full(padded_L, fill_value, dtype=jnp.int32)
    padded = padded.at[:L].set(jnp.array(seq))
    return padded, L


def group_by_bin(sequences):
    """Group sequences by their padded bin size.

    Args:
        sequences: list of (x_seq, y_seq) pairs

    Returns:
        dict mapping bin_size -> list of (x_padded, y_padded, x_len, y_len) tuples
    """
    bins = {}
    for x, y in sequences:
        # Use max of the two lengths for binning (pair HMM processes both)
        max_len = max(len(x), len(y))
        bin_size = pad_to_bin(max_len)
        bins.setdefault(bin_size, []).append((x, y))
    return bins
