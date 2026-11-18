"""Forward-generative MixFrag cherry simulator.

Samples cherry alignments directly from the MixFrag joint Pair HMM: walk the
(3F+2)-state machine from S, drawing each transition by its row of the joint
transition matrix tau (so the latent per-fragment fragtype is sampled along the
way), and emit residues under the shared substitution model -- a match column
draws an ancestor residue a ~ pi and a descendant b ~ exp(Qt)[a, :], an insertion
draws b ~ pi, a deletion draws a ~ pi.  This is the exact generative process
P(x, y, alignment) of the model, so it yields the TRUE alignment alongside the
sequences -- which is what lets the two trainers be cross-checked on one dataset:
(``simulate_mixfrag_s_cherries`` is the same walk for MixFrag-S, where each
fragtype f carries its own substitution model (Q_f, pi_f): a column emitted by
M_f / I_f / D_f uses pi_f and exp(Q_f t), matching
``tkfmixdom.jax.dp.hmm.pair_hmm_emissions_per_fragtype``.)

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

    Equivalent to ``simulate_mixfrag_s_cherries`` with every fragtype given the
    same (Q, pi); kept as the shared-model entry point (and the two draw the
    same random stream, so identical seeds give identical cherries).

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
    Qs = np.repeat(np.asarray(Q, np.float64)[None], F, axis=0)
    pis = np.repeat(np.asarray(pi, np.float64)[None], F, axis=0)
    out = simulate_mixfrag_s_cherries(rng, n, lam, mu, exts, weights, t, Qs, pis,
                                      max_len=max_len)
    for sm in out:
        del sm['frag']
    return out


def simulate_mixfrag_s_cherries(rng, n, lam, mu, exts, weights, t, Qs, pis,
                                max_len=2000):
    """Sample ``n`` MixFrag-S cherries at branch length ``t``.

    MixFrag-S is MixFrag with one substitution model per fragtype: a column of
    a fragment of type f is emitted as in ``pair_hmm_emissions_per_fragtype``,
    i.e. a match column draws the ancestor residue a ~ pi_f and the descendant
    b ~ exp(Q_f t)[a, :], an insertion draws b ~ pi_f and a deletion draws
    a ~ pi_f.  (The ancestral residues of a fragment are therefore drawn from
    that fragment's own pi_f -- the joint Pair HMM, not a stationary ancestor
    under a common equilibrium.)

    Args:
        rng:      numpy Generator.
        n:        number of cherries.
        lam, mu:  TKF91 insertion / deletion rates (kappa = lam/mu must be < 1).
        exts:     (F,) per-fragtype extension probabilities r_f.
        weights:  (F,) fragtype weights w_f (sum to 1).
        t:        branch length.
        Qs, pis:  per-fragtype substitution generators (F, A, A), in the raw
                  units the branch lengths are expressed in, and equilibria
                  (F, A).
        max_len:  guard on runaway sequences (resampled if exceeded).

    Returns:
        list of dicts, each with
            'x'     (Lx,) int32 ancestor residue indices,
            'y'     (Ly,) int32 descendant residue indices,
            't'     branch length,
            'cols'  list of 'M'/'I'/'D' column types (the true alignment),
            'frag'  (n_cols,) int32 fragtype (0..F-1) of each column,
            'row_a' aligned ancestor row (str, '-' for gaps),
            'row_b' aligned descendant row (str).
    """
    F = len(exts)
    Qs = np.asarray(Qs, np.float64)
    pis = np.asarray(pis, np.float64)
    if Qs.shape[0] != F or pis.shape[0] != F:
        raise ValueError(f"need one (Q_f, pi_f) per fragtype: F={F}, Qs {Qs.shape}, pis {pis.shape}")
    tau = np.asarray(mixfrag_trans(lam, mu, t, jnp.asarray(exts, float),
                                   jnp.asarray(weights, float)))
    subs = np.stack([np.asarray(transition_matrix(jnp.asarray(Qf), t)) for Qf in Qs])  # (F,A,A)
    pis = pis / pis.sum(axis=1, keepdims=True)
    subs = np.clip(subs, 0.0, None)
    subs = subs / subs.sum(axis=2, keepdims=True)
    Sx, M0, I0, D0, Ex = mixfrag_pair_index(F)
    n_states = 3 * F + 2
    A = pis.shape[1]
    # Precompute normalised transition rows; the absorbing E row sums to 0 (never
    # sampled from), so guard the denominator to avoid 0/0.
    rows = np.clip(tau, 0.0, None)
    rsum = rows.sum(axis=1, keepdims=True)
    rows = rows / np.where(rsum > 0, rsum, 1.0)

    def _state_type(idx):
        # (column type, fragtype) of an emitting state; state order S, M_1..M_F, I_1..I_F, D_1..D_F, E
        if M0 <= idx < I0:
            return 'M', idx - M0
        if I0 <= idx < D0:
            return 'I', idx - I0
        if D0 <= idx < Ex:
            return 'D', idx - D0
        return None, None

    out = []
    for _ in range(n):
        while True:
            cur = Sx
            cols, frag, x, y, ra, rb = [], [], [], [], [], []
            ok = True
            while True:
                nxt = int(rng.choice(n_states, p=rows[cur]))
                if nxt == Ex:
                    break
                ty, f = _state_type(nxt)
                cols.append(ty)
                frag.append(f)
                pi, sub = pis[f], subs[f]
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
            cols=cols, frag=np.array(frag, np.int32),
            row_a="".join(ra), row_b="".join(rb)))
    return out


def mixfrag_s_tilted_subst(Q, pi, tilt="PGSDE", factor=1.6, F=2):
    """Per-fragtype (Q_f, pi_f) for a MixFrag-S simulation truth built from one GTR model.

    Fragtype 1 keeps (Q, pi).  Fragtype 2 keeps the exchangeabilities
    S[a, b] = Q[a, b] / pi[b] but tilts pi toward the residues in ``tilt`` by
    ``factor`` (then renormalises) and rebuilds Q = build_Q_from_S_pi(S, pi')
    in the same raw units.  Any further fragtypes (F > 2) repeat fragtype 1.

    Returns:
        Qs (F, A, A), pis (F, A).
    """
    from ..core.ctmc import build_Q_from_S_pi
    Q = np.asarray(Q, np.float64)
    pi = np.asarray(pi, np.float64)
    pi = pi / pi.sum()
    A = pi.shape[0]
    S = Q / pi[None, :]
    S[np.arange(A), np.arange(A)] = 0.0
    S = 0.5 * (S + S.T)                                   # exact for a reversible Q
    pi2 = pi.copy()
    for aa in tilt:
        pi2[AMINO_ACIDS.index(aa)] *= factor
    pi2 = pi2 / pi2.sum()
    Q2 = np.asarray(build_Q_from_S_pi(jnp.asarray(S), jnp.asarray(pi2)), np.float64)
    Qs = [Q, Q2] + [Q] * (F - 2)
    pis = [pi, pi2] + [pi] * (F - 2)
    return np.stack(Qs[:F]), np.stack(pis[:F])
