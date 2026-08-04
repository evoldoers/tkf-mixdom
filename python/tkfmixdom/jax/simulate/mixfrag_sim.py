"""Forward-generative MixFrag cherry simulator.

Samples cherry alignments directly from the MixFrag joint Pair HMM: walk the
(3F+2)-state machine from S, drawing each transition by its row of the joint
transition matrix tau (so the latent per-fragment fragtype is sampled along the
way), and emit residues under the shared substitution model -- a match column
draws an ancestor residue a ~ pi and a descendant b ~ exp(Qt)[a, :], an insertion
draws b ~ pi, a deletion draws a ~ pi.  This is the exact generative process
P(x, y, alignment) of the model, so it yields the TRUE alignment alongside the
sequences -- which is what lets the two trainers be cross-checked on one dataset:

  * the alignment-marginalised path (``svi_bw_mixfrag``) consumes the unaligned
    integer sequences (x, y, t);
  * the alignment-given summarised-count EM (``mixfrag_cherry_em``) consumes the
    aligned rows (row_a, row_b) via ``build_mixfrag_cherry_counts``.

Residues use the 20-letter amino-acid order of ``build_tkf92_cherry_counts``
(index i <-> AMINO_ACIDS[i]) so the integer sequences and the aligned-row strings
denote the same residues.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from ..core.params import mixfrag_trans, mixfrag_pair_index
from ..core.ctmc import transition_matrix

# Same amino-acid order as build_tkf92_cherry_counts.AMINO_ACIDS.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def simulate_mixfrag_cherries(rng, n, lam, mu, exts, weights, t, Q, pi,
                              max_len=2000):
    """Sample ``n`` MixFrag cherries at branch length ``t``.

    Args:
        rng:      numpy Generator.
        n:        number of cherries.
        lam, mu:  TKF91 insertion / deletion rates (kappa = lam/mu must be < 1).
        exts:     (F,) per-fragtype extension probabilities r_f.
        weights:  (F,) fragtype weights w_f (sum to 1).
        t:        branch length.
        Q, pi:    shared substitution generator (A, A) and equilibrium (A,).
        max_len:  guard on runaway sequences (resampled if exceeded).

    Returns:
        list of dicts, each with
            'x'     (Lx,) int32 ancestor residue indices,
            'y'     (Ly,) int32 descendant residue indices,
            't'     branch length,
            'cols'  list of 'M'/'I'/'D' column types (the true alignment),
            'row_a' aligned ancestor row (str, '-' for gaps),
            'row_b' aligned descendant row (str).
    """
    F = len(exts)
    tau = np.asarray(mixfrag_trans(lam, mu, t, jnp.asarray(exts, float),
                                   jnp.asarray(weights, float)))
    sub = np.asarray(transition_matrix(jnp.asarray(Q), t))     # exp(Qt) (A,A)
    pi = np.asarray(pi, np.float64)
    pi = pi / pi.sum()
    sub = np.clip(sub, 0.0, None)
    sub = sub / sub.sum(axis=1, keepdims=True)
    Sx, M0, I0, D0, Ex = mixfrag_pair_index(F)
    n_states = 3 * F + 2
    A = pi.shape[0]
    # Precompute normalised transition rows; the absorbing E row sums to 0 (never
    # sampled from), so guard the denominator to avoid 0/0.
    rows = np.clip(tau, 0.0, None)
    rsum = rows.sum(axis=1, keepdims=True)
    rows = rows / np.where(rsum > 0, rsum, 1.0)

    def _state_type(idx):
        if M0 <= idx < I0:
            return 'M'
        if I0 <= idx < D0:
            return 'I'
        if D0 <= idx < Ex:
            return 'D'
        return None

    out = []
    for _ in range(n):
        while True:
            cur = Sx
            cols, x, y, ra, rb = [], [], [], [], []
            ok = True
            while True:
                nxt = int(rng.choice(n_states, p=rows[cur]))
                if nxt == Ex:
                    break
                ty = _state_type(nxt)
                cols.append(ty)
                if ty == 'M':
                    a = int(rng.choice(A, p=pi))
                    b = int(rng.choice(A, p=sub[a]))
                    x.append(a); y.append(b)
                    ra.append(AMINO_ACIDS[a]); rb.append(AMINO_ACIDS[b])
                elif ty == 'I':
                    b = int(rng.choice(A, p=pi))
                    y.append(b)
                    ra.append('-'); rb.append(AMINO_ACIDS[b])
                else:                                          # 'D'
                    a = int(rng.choice(A, p=pi))
                    x.append(a)
                    ra.append(AMINO_ACIDS[a]); rb.append('-')
                cur = nxt
                if len(cols) > max_len:
                    ok = False
                    break
            if ok:
                break
        out.append(dict(
            x=np.array(x, np.int32), y=np.array(y, np.int32), t=float(t),
            cols=cols, row_a="".join(ra), row_b="".join(rb)))
    return out
