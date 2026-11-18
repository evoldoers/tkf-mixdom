"""Column summaries: Reestimate vs a NumPy reference, and the separable match table."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "scripts"))
import progalign  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from progalign import profile as pf  # noqa: E402
from progalign.model import make_subst  # noqa: E402
from progalign.tree import bionj, midpoint_root  # noqa: E402


def _tree(n, seed):
    rng = np.random.default_rng(seed)
    D = rng.uniform(0.3, 1.5, (n, n)); D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    return bionj(D), midpoint_root(bionj(D), n)


def test_reestimate_matches_numpy_reference():
    import profile_classes_pfam as ref
    S, P, w, R = ref.load_model()
    C = ref.Classes(S, P, w, R)
    sub = make_subst("C20")
    adj, rooted = _tree(6, 1)
    rng = np.random.default_rng(2)
    cols = rng.integers(0, 20, (5, 6))
    kstar = np.array([0, 3, 7, 12, 19])
    s, logp = pf.reestimate(cols, list(range(6)), rooted, sub, kstar)
    # reference: unrooted adjacency rooted at the same node
    g = {u: {} for u in rooted["children"]}
    for u, cs in rooted["children"].items():
        for v, t in cs:
            g[u][v] = t; g[v][u] = t
    for c in range(5):
        col = {i: int(cols[c, i]) for i in range(6)}
        r = ref.posterior_summary(g, rooted["root"], None, col, C, int(kstar[c]))
        assert np.allclose(np.asarray(s.Vp[c]), r["Vp"], atol=1e-8)
        assert np.allclose(np.asarray(s.T[c]), r["T"], atol=1e-8)
        assert abs(float(s.U[c]) - r["U"]) < 1e-8
        assert abs(float(s.C[c]) - r["C"]) < 1e-7
    # Phi at the reference class equals the exact class log-likelihood
    ph = np.asarray(pf.phi(s, sub))
    assert np.allclose(ph[np.arange(5), kstar], np.asarray(logp), atol=1e-8)


def test_match_table_equals_direct_merge_for_leaves_and_bounds_it_otherwise():
    sub = make_subst("C20")
    rng = np.random.default_rng(4)
    tA, tB = 0.3, 0.45
    sA, sB = pf.leaf_summ(rng.integers(0, 20, 6)), pf.leaf_summ(rng.integers(0, 20, 5))
    tab = np.asarray(pf.match_table(sA, sB, sub, tA + tB))
    for m in range(6):
        for n in range(5):
            s = pf.merge(sA.take([m]), sB.take([n]), tA, tB, sub)
            direct = float(jax.nn.logsumexp(jnp.log(sub.w) + pf.phi(s, sub)[0]))
            assert abs(tab[m, n] - direct) < 1e-9
    # fractional tops (re-estimated two-leaf clades): table <= direct, and close
    _, rooted = _tree(2, 5)
    resA = rng.integers(0, 20, (4, 2)); resB = rng.integers(0, 20, (3, 2))
    k0 = np.zeros(4, int); k1 = np.zeros(3, int)
    sA2, _ = pf.reestimate(resA, [0, 1], rooted, sub, k0)
    sB2, _ = pf.reestimate(resB, [0, 1], rooted, sub, k1)
    tab = np.asarray(pf.match_table(sA2, sB2, sub, tA + tB))
    for m in range(4):
        for n in range(3):
            s = pf.merge(sA2.take([m]), sB2.take([n]), tA, tB, sub)
            direct = float(jax.nn.logsumexp(jnp.log(sub.w) + pf.phi(s, sub)[0]))
            assert tab[m, n] <= direct + 1e-9
            assert direct - tab[m, n] < 0.5


def test_sparse_bound_is_valid_and_exact_on_leaves():
    from progalign import sparse
    from progalign.align import _exact_tables
    sub = make_subst("C20")
    rng = np.random.default_rng(8)
    # leaf (one-hot) columns: the sparse bound is exact
    FA = jax.nn.one_hot(jnp.asarray(rng.integers(0, 20, 9)), 20, dtype=jnp.float64)[:, None, :] * jnp.ones((1, 20, 1))
    FB = jax.nn.one_hot(jnp.asarray(rng.integers(0, 20, 7)), 20, dtype=jnp.float64)[:, None, :] * jnp.ones((1, 20, 1))
    z9, z7 = jnp.zeros(9), jnp.zeros(7)
    tab, sa, sb = sparse.tables(FA, z9, FB, z7, sub, 1.3)
    ex, ea, eb = _exact_tables(FA, z9, FB, z7, sub.P, sub.w, sub.M(1.3))
    assert np.allclose(np.asarray(tab), np.asarray(ex), atol=1e-10)
    assert np.allclose(np.asarray(sa), np.asarray(ea)) and np.allclose(np.asarray(sb), np.asarray(eb))
    # fractional partials: a lower bound
    FA = jnp.asarray(rng.gamma(0.3, 1.0, (9, 20, 20))); FB = jnp.asarray(rng.gamma(0.3, 1.0, (7, 20, 20)))
    tab, _, _ = sparse.tables(FA, z9, FB, z7, sub, 2.5)
    ex, _, _ = _exact_tables(FA, z9, FB, z7, sub.P, sub.w, sub.M(2.5))
    assert np.max(np.asarray(tab) - np.asarray(ex)) <= 1e-10


def test_pp_scores_exact_for_one_class():
    from progalign.align import _exact_tables, _pp_tables
    sub = make_subst("LG")
    rng = np.random.default_rng(9)
    FA = jnp.asarray(rng.gamma(0.5, 1.0, (6, 1, 20))); FB = jnp.asarray(rng.gamma(0.5, 1.0, (5, 1, 20)))
    lA, lB = jnp.asarray(rng.normal(size=6)), jnp.asarray(rng.normal(size=5))
    pp, pa, pb = _pp_tables(FA, lA, FB, lB, sub.P, sub.w, sub.joint(1.7))
    ex, ea, eb = _exact_tables(FA, lA, FB, lB, sub.P, sub.w, sub.M(1.7))
    assert np.allclose(np.asarray(pp), np.asarray(ex), atol=1e-10)
    assert np.allclose(np.asarray(pa), np.asarray(ea)) and np.allclose(np.asarray(pb), np.asarray(eb))
