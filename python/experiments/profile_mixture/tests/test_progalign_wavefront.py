"""Wavefront Forward/Viterbi against tkfmixdom and a NumPy reference."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import progalign  # noqa: E402,F401  (sets x64 and the tkfmixdom path)
import jax.numpy as jnp  # noqa: E402
from tkfmixdom.jax.core.protein import rate_matrix_lg  # noqa: E402
from tkfmixdom.jax.core.ctmc import transition_matrix  # noqa: E402
from tkfmixdom.jax.dp.hmm import forward_2d  # noqa: E402
from tkfmixdom.jax.models.left_regular import make_mixfrag_pair_hmm  # noqa: E402
from progalign import wavefront as wf  # noqa: E402


def _model(t=0.7):
    Q, pi = rate_matrix_lg()
    lt, st, sub, p = make_mixfrag_pair_hmm(0.034, 0.0347, t, jnp.array([0.41, 0.86]),
                                           jnp.array([0.7, 0.3]), Q, pi)
    return lt, st, np.log(np.asarray(pi)[:, None] * np.asarray(sub)), np.log(np.asarray(pi))


def _emits(logJ, logpi, x, y, Lx, Ly):
    xp = np.zeros(Lx, int); xp[:len(x)] = x
    yp = np.zeros(Ly, int); yp[:len(y)] = y
    xj, yj, lJ = jnp.array(xp), jnp.array(yp), jnp.array(logJ)
    eM = lambda i, j: lJ[xj[i - 1], yj[j - 1]]
    eI = jnp.concatenate([jnp.zeros(1), jnp.array(logpi)[yj]])
    eD = jnp.concatenate([jnp.zeros(1), jnp.array(logpi)[xj]])
    return eM, eI, eD


def _numpy_viterbi(logT, st, logJ, logpi, x, y):
    ns, Lx, Ly = logT.shape[0], len(x), len(y)
    V = np.full((Lx + 1, Ly + 1, ns), -np.inf); V[0, 0, 0] = 0.0
    for i in range(Lx + 1):
        for j in range(Ly + 1):
            for s in range(ns):
                if st[s] == wf.M and i and j:
                    V[i, j, s] = np.max(V[i - 1, j - 1] + logT[:, s]) + logJ[x[i - 1], y[j - 1]]
                elif st[s] == wf.I and j:
                    V[i, j, s] = np.max(V[i, j - 1] + logT[:, s]) + logpi[y[j - 1]]
                elif st[s] == wf.D and i:
                    V[i, j, s] = np.max(V[i - 1, j] + logT[:, s]) + logpi[x[i - 1]]
    e = int(np.argmax(st == wf.E))
    return np.max(V[Lx, Ly] + logT[:, e])


@pytest.mark.parametrize("lens", [(7, 5), (12, 12), (3, 9)])
def test_forward_matches_tkfmixdom(lens):
    lt, st, logJ, logpi = _model()
    rng = np.random.default_rng(sum(lens))
    x, y = rng.integers(0, 20, lens[0]), rng.integers(0, 20, lens[1])
    ref, _ = forward_2d(lt, st, jnp.array(x), jnp.array(y),
                        transition_matrix(rate_matrix_lg()[0], 0.7), rate_matrix_lg()[1])
    Lx, Ly = lens[0] + 3, lens[1] + 2       # padded sizes, true lengths traced
    eM, eI, eD = _emits(logJ, logpi, x, y, Lx, Ly)
    got = wf.forward(jnp.asarray(lt), jnp.asarray(st), eM, eI, eD, Lx, Ly, lens[0], lens[1])
    assert abs(float(got) - float(ref)) < 1e-8


@pytest.mark.parametrize("lens", [(7, 5), (10, 13), (1, 6)])
def test_viterbi_matches_numpy_and_traceback_rescores(lens):
    lt, st, logJ, logpi = _model()
    lt, st = np.asarray(lt), np.asarray(st)
    rng = np.random.default_rng(10 + sum(lens))
    x, y = rng.integers(0, 20, lens[0]), rng.integers(0, 20, lens[1])
    Lx, Ly = lens[0] + 4, lens[1] + 1
    eM, eI, eD = _emits(logJ, logpi, x, y, Lx, Ly)
    score, last, bps = wf.viterbi(jnp.asarray(lt), jnp.asarray(st), eM, eI, eD, Lx, Ly, lens[0], lens[1])
    ref = _numpy_viterbi(lt, st, logJ, logpi, x, y)
    assert abs(float(score) - ref) < 1e-8
    cols = wf.traceback(bps, st, int(last), lens[0], lens[1], Ly)
    assert sum(c != wf.I for c in cols) == lens[0] and sum(c != wf.D for c in cols) == lens[1]


def test_distance_matches_tkfmixdom_fsa_estimator():
    from tkfmixdom.jax.tree.fsa_anneal import _pairwise_posteriors_tkf92_jax
    from progalign.distance import pairwise_times
    from progalign.model import load_indel, make_subst
    rng = np.random.default_rng(3)
    base = rng.integers(0, 20, 40)
    seqs = [base, np.where(rng.random(40) < 0.3, rng.integers(0, 20, 40), base)[:35]]
    tk = load_indel("tkf92")
    D = pairwise_times(seqs, make_subst("LG"), tk)
    Q, pi = rate_matrix_lg()
    _, tau, _ = _pairwise_posteriors_tkf92_jax(jnp.array(seqs[0]), jnp.array(seqs[1]), 40, 35,
                                               tk.lam, tk.mu, tk.r[0], Q, pi)
    assert abs(D[0, 1] - float(tau)) < 1e-6
