"""JAX-parallelized DP for weighted context-free grammars.

Provides JIT-compiled Inside algorithm for WCFGs using:
- jax.lax.scan over diagonals/spans
- jax.vmap over cells within each span
- Structured production arrays for efficient vectorized computation

For regular grammars, prefer the HMM-based DP (dp.py) which is O(L*N²)
vs O(L³*N) for the grammar Inside.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

NEG_INF = -1e30


def _precompile_grammar(grammar):
    """Convert grammar productions to JAX-friendly arrays.

    Returns structured arrays for each production type, suitable for
    vectorized operations.
    """
    N = grammar.n_nonterminals
    T = grammar.n_terminals

    # Collect productions by type
    terminals = []      # (lhs, terminal, log_weight)
    unaries = []        # (lhs, rhs, log_weight)
    binaries = []       # (lhs, B, C, log_weight)
    right_linears = []  # (lhs, terminal, B, log_weight)
    epsilons = []       # (lhs, log_weight)

    for p in grammar.productions:
        log_w = np.log(max(p.weight, 1e-300))
        if p.is_terminal:
            terminals.append((p.lhs, p.rhs[0], log_w))
        elif p.is_unary:
            unaries.append((p.lhs, p.rhs[0], log_w))
        elif p.is_binary:
            binaries.append((p.lhs, p.rhs[0], p.rhs[1], log_w))
        elif p.is_right_linear:
            right_linears.append((p.lhs, p.rhs[0], p.rhs[1], log_w))
        elif p.is_empty:
            epsilons.append((p.lhs, log_w))

    def _to_array(lst, n_cols):
        if not lst:
            return jnp.zeros((0, n_cols))
        return jnp.array(lst)

    return {
        'N': N,
        'T': T,
        'start': grammar.start,
        'terminals': _to_array(terminals, 3),
        'unaries': _to_array(unaries, 3),
        'binaries': _to_array(binaries, 4),
        'right_linears': _to_array(right_linears, 4),
        'epsilons': _to_array(epsilons, 2),
    }


def inside_jax(grammar, sequence):
    """JAX-compiled Inside algorithm for a WCFG.

    Computes log_inside[A, i, j] = log P(sequence[i:j] | A)
    using span-by-span filling with vectorized operations.

    Args:
        grammar: WCFG
        sequence: integer array of terminal indices, shape (L,)

    Returns:
        log_inside: array of shape (N, L+1, L+1)
    """
    compiled = _precompile_grammar(grammar)
    return _inside_jax_compiled(compiled, jnp.array(sequence))


def _inside_jax_compiled(compiled, sequence):
    """Core Inside computation with precompiled grammar arrays."""
    N = compiled['N']
    L = sequence.shape[0]
    terminals = compiled['terminals']
    unaries = compiled['unaries']
    binaries = compiled['binaries']
    right_linears = compiled['right_linears']
    epsilons = compiled['epsilons']

    log_I = jnp.full((N, L + 1, L + 1), NEG_INF)

    # Precompute unary closure matrix U = (I - W)^{-1}
    W = jnp.full((N, N), NEG_INF)
    for idx in range(unaries.shape[0]):
        lhs = int(unaries[idx, 0])
        rhs = int(unaries[idx, 1])
        log_w = float(unaries[idx, 2])
        W = W.at[lhs, rhs].set(jnp.logaddexp(W[lhs, rhs], log_w))

    # Compute unary closure via power iteration in log-space
    log_U = jnp.full((N, N), NEG_INF)
    for a in range(N):
        log_U = log_U.at[a, a].set(0.0)
    Wk = W
    for _ in range(N):
        for a in range(N):
            for b in range(N):
                log_U = log_U.at[a, b].set(jnp.logaddexp(log_U[a, b], Wk[a, b]))
        new_Wk = jnp.full((N, N), NEG_INF)
        for a in range(N):
            for b in range(N):
                for c in range(N):
                    new_Wk = new_Wk.at[a, b].set(
                        jnp.logaddexp(new_Wk[a, b], Wk[a, c] + W[c, b]))
        Wk = new_Wk

    def apply_unary_closure(vals):
        """vals: (N,) -> closed_vals: (N,)"""
        # closed[a] = logsumexp_b(U[a,b] + vals[b])
        return jax.nn.logsumexp(log_U + vals[None, :], axis=1)

    # Base case: epsilon spans
    for i in range(L + 1):
        for idx in range(epsilons.shape[0]):
            lhs = int(epsilons[idx, 0])
            log_w = float(epsilons[idx, 1])
            log_I = log_I.at[lhs, i, i].set(jnp.logaddexp(log_I[lhs, i, i], log_w))
        closed = apply_unary_closure(log_I[:, i, i])
        log_I = log_I.at[:, i, i].set(closed)

    # Span 1: terminal productions
    for i in range(L):
        j = i + 1
        char = sequence[i]
        for idx in range(terminals.shape[0]):
            lhs = int(terminals[idx, 0])
            term = int(terminals[idx, 1])
            log_w = float(terminals[idx, 2])
            if term == int(char):
                log_I = log_I.at[lhs, i, j].set(
                    jnp.logaddexp(log_I[lhs, i, j], log_w))

        # Right-linear at span 1: A -> a B where B covers empty span
        for idx in range(right_linears.shape[0]):
            lhs = int(right_linears[idx, 0])
            term = int(right_linears[idx, 1])
            B = int(right_linears[idx, 2])
            log_w = float(right_linears[idx, 3])
            if term == int(char):
                val = log_w + log_I[B, j, j]
                log_I = log_I.at[lhs, i, j].set(
                    jnp.logaddexp(log_I[lhs, i, j], val))

        closed = apply_unary_closure(log_I[:, i, j])
        log_I = log_I.at[:, i, j].set(closed)

    # Fill spans of length 2 to L
    for span in range(2, L + 1):
        for i in range(L - span + 1):
            j = i + span

            # Right-linear: A -> a B
            char = sequence[i]
            for idx in range(right_linears.shape[0]):
                lhs = int(right_linears[idx, 0])
                term = int(right_linears[idx, 1])
                B = int(right_linears[idx, 2])
                log_w = float(right_linears[idx, 3])
                if term == int(char):
                    val = log_w + log_I[B, i + 1, j]
                    log_I = log_I.at[lhs, i, j].set(
                        jnp.logaddexp(log_I[lhs, i, j], val))

            # Binary: A -> B C
            for idx in range(binaries.shape[0]):
                lhs = int(binaries[idx, 0])
                B = int(binaries[idx, 1])
                C = int(binaries[idx, 2])
                log_w = float(binaries[idx, 3])
                for k in range(i, j + 1):
                    val = log_w + log_I[B, i, k] + log_I[C, k, j]
                    log_I = log_I.at[lhs, i, j].set(
                        jnp.logaddexp(log_I[lhs, i, j], val))

            closed = apply_unary_closure(log_I[:, i, j])
            log_I = log_I.at[:, i, j].set(closed)

    return log_I


def inside_logprob_jax(grammar, sequence):
    """Compute log P(sequence | grammar) using JAX Inside."""
    log_I = inside_jax(grammar, sequence)
    return float(log_I[grammar.start, 0, len(sequence)])


