"""K-component banded MixDom2 mixture fit (Phase 2 of banded fragchar pipeline).

This module fits K independent banded MixDom2 components to per-family
Maraschino cherry counts via an outer EM with HARD family-to-component
assignment (avoids mode collapse that plagues soft-assignment Maraschino
on weak order-1 correlations) and an inner Adam loop per component on
the aggregated cherry counts in banded mode.

Per-component model: 1 domain x 3 fragchars (FragStart/Mid/End) x
3 site classes (one per fragchar via classdist tying). Free params:
    lam_k, mu_k                  (top-level rates)
    dom_ins_k, dom_del_k         (domain rates; same as top-level since 1 dom)
    ext_rates_k                  (3x3 banded structural mask: 3 DOFs)
    class_pis_k[c=0..2, A]       (per-fragchar equilibrium)
    class_S_exch_k[c=0..2, A, A] (per-fragchar exchangeability)

Output: stacked K-domain banded MixDom2 checkpoint suitable for
``train_pfam.py --init`` warm start. Each (k, fragchar) maps to its own
class slot, so the global C = 3K and ``classdist[k, f, c] = 1 iff
c == 3*k + f`` (block-diagonal over the K-component partition).

See `tkf/substitution-mstep.tex` sec:mstep-tied-pi and the FragStart/
FragMid/FragEnd parameterisation; see `python/fit_tkf92_mixture.py` for
the analogous TKF92 mixture pattern this mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.distill.maraschino_fit import (
    COUNT_KEYS, cherry_log_likelihood, linear_to_raw, raw_to_linear,
)
from tkfmixdom.jax.models.mixdom_init import init_mixdom2_params_from_args
from tkfmixdom.jax.train.restricted_mstep import (
    banded_3fc_ext_mask, banded_3fc_init,
)


AA = 20
N_FRAG = 3   # banded mode is hard-wired to 3 fragchars
N_CLASS_PER_COMPONENT = 3  # fragchar-tied classdist


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class BandedFamily:
    """Per-family Maraschino cherry counts for the banded mixture fit.

    Attributes:
        family_id: family name (e.g. 'PF00001').
        counts: dict with keys ``C_<XY>`` mapping to (T, ...) ndarrays
            (the same axis layout as `_init_counts` in maraschino.py),
            plus ``B`` of shape (T,).
        n_pairs: total cherry pairs (for diagnostic).
    """
    family_id: str
    counts: dict
    n_pairs: int


def load_per_family_marcounts(
        npz_dir: str | Path,
        families: Iterable[str] | None = None,
        suffix: str = '.marcounts.npz') -> tuple[list[BandedFamily], jnp.ndarray]:
    """Load per-family Maraschino counts files (one per Pfam family).

    Args:
        npz_dir: directory of per-family .marcounts.npz files.
        families: optional whitelist of family IDs.
        suffix: filename suffix (matches `--out-suffix` of `maraschino count`).

    Returns:
        (families, tau_centers): list of BandedFamily and shared tau_centers.
    """
    npz_dir = Path(npz_dir)
    if not npz_dir.is_dir():
        raise FileNotFoundError(f"npz_dir not found: {npz_dir}")
    if families is None:
        files = sorted(npz_dir.glob(f"*{suffix}"))
        if not files:
            files = sorted(npz_dir.glob('*.npz'))
        if not files:
            raise FileNotFoundError(f"No {suffix} files in {npz_dir}")
    else:
        candidates = []
        for fam in families:
            for cand in (npz_dir / f"{fam}{suffix}",
                         npz_dir / f"{fam}.npz"):
                if cand.exists():
                    candidates.append(cand)
                    break
        files = candidates
        if not files:
            raise FileNotFoundError(
                f"None of the requested families found in {npz_dir}")

    out: list[BandedFamily] = []
    tau_centers = None
    for p in files:
        data = np.load(p, allow_pickle=True)
        # Keep counts as numpy on the HOST. With ~150 GB of total per-family
        # tensor mass for the full Pfam train split, lifting to JAX device
        # arrays at load time fills RAM. Convert to jnp at call sites.
        c = {f"C_{k}": np.asarray(data[f"C_{k}"], dtype=np.float32)
             for k in COUNT_KEYS}
        c['B'] = np.asarray(data['B'], dtype=np.float32)
        # Family ID: prefer the source-file metadata; fall back to filename.
        fam_id = str(p.name).removesuffix(suffix).removesuffix('.npz')
        if 'meta' in data.files:
            try:
                meta = data['meta'].item()
                if isinstance(meta, dict) and 'source_file' in meta:
                    fam_id = str(meta['source_file']).split('.')[0]
            except Exception:
                pass
        # Total cherry count (B sums across tau bins for a sanity diagnostic;
        # n_pairs is recorded in the metadata when written by `maraschino count`).
        n_pairs = int(np.asarray(data['B']).sum())
        if 'meta' in data.files:
            try:
                meta = data['meta'].item()
                if isinstance(meta, dict) and 'n_pairs' in meta:
                    n_pairs = int(meta['n_pairs'])
            except Exception:
                pass
        out.append(BandedFamily(family_id=fam_id, counts=c, n_pairs=n_pairs))
        if tau_centers is None:
            tau_centers = jnp.asarray(data['tau_centers'], dtype=jnp.float32)
    if tau_centers is None:
        raise RuntimeError(
            "No tau_centers found in any input .npz")
    return out, tau_centers


# ---------------------------------------------------------------------------
# Per-component init + banded mask
# ---------------------------------------------------------------------------


def _ones_S_lg():
    Q_lg, pi_lg = rate_matrix_lg()
    return Q_lg, pi_lg


class _ArgsForInit:
    """Minimal args namespace consumed by `init_mixdom2_params_from_args`."""
    def __init__(self, *, seed: int, p_ext: float, n_classes: int = 3,
                 init_ins: float = 0.01, init_del: float = 0.01,
                 noise_frac: float = 0.05):
        self.init_ins = init_ins
        self.init_del = init_del
        self.seed = seed
        self.estimate_subst = True
        self.n_classes = n_classes
        self.class_pi_init = 'lg_noisy'
        self.pi_init_noise_frac = noise_frac
        self.classdist_init = 'fragchar'
        self.class_pis_from_dom_pis = False
        self.classdist_noise_frac = 0.0
        self.banded_frag_init = True
        self.p_ext = p_ext


def init_banded_component(seed: int, *, p_ext: float = 0.6,
                          init_ins: float = 0.01, init_del: float = 0.01,
                          noise_frac: float = 0.05) -> dict:
    """Build initial linear-space params for a single banded component.

    Single domain (D=1), 3 fragchars (banded), 3 classes (one per fragchar).
    Returns a dict with keys matching `init_mixdom2_params_from_args` output.
    """
    args = _ArgsForInit(seed=seed, p_ext=p_ext,
                        init_ins=init_ins, init_del=init_del,
                        noise_frac=noise_frac)
    Q_lg, pi_lg = _ones_S_lg()
    linear = init_mixdom2_params_from_args(
        args, n_dom=1, n_frag=N_FRAG, Q_lg=Q_lg, pi_lg=pi_lg,
        log_fn=lambda *_: None)
    # Shape sanity (cheap defensive check).
    assert linear['frag_weights'].shape == (1, N_FRAG)
    assert linear['ext_rates'].shape == (1, N_FRAG, N_FRAG)
    assert linear['class_pis'].shape == (N_CLASS_PER_COMPONENT, AA)
    assert linear['class_S_exch'].shape == (
        N_CLASS_PER_COMPONENT, AA, AA)
    return linear


def _banded_mask_full(n_dom: int = 1, n_frag: int = N_FRAG) -> jnp.ndarray:
    """(n_dom, n_frag, n_frag+1) free-mask used in `_materialise_banded`.

    True iff the entry is structurally allowed to be nonzero.
    """
    ext_mask, term_mask = banded_3fc_ext_mask()
    free_2d = ext_mask.astype(jnp.float32)
    free_term = term_mask.astype(jnp.float32)[:, None]
    mask = jnp.concatenate([free_2d, free_term], axis=-1)
    return jnp.broadcast_to(mask[None, :, :], (n_dom, n_frag, n_frag + 1))


def _materialise_banded(raw_params: dict, n_dom: int = 1,
                        n_frag: int = N_FRAG,
                        n_classes: int = N_CLASS_PER_COMPONENT,
                        mask_full: jnp.ndarray | None = None,
                        freeze_init: dict | None = None,
                        freeze_main_rates: bool = False) -> dict:
    """Constrain raw -> linear, applying the banded ext mask + freeze_fragdist."""
    out = raw_to_linear(raw_params, n_dom, n_frag, n_classes,
                        freeze_init=freeze_init,
                        freeze_main_rates=freeze_main_rates)
    if mask_full is None:
        mask_full = _banded_mask_full(n_dom, n_frag)
    ext_full_live = jax.nn.softmax(raw_params['logit_ext_rates'], axis=-1)
    ext_full = ext_full_live * mask_full
    ext_full = ext_full / jnp.maximum(
        ext_full.sum(axis=-1, keepdims=True), 1e-30)
    out['ext_rates'] = ext_full[..., :n_frag]
    return out


# ---------------------------------------------------------------------------
# Per-family LL under one component
# ---------------------------------------------------------------------------


def _component_ll_one_family(linear_params: dict, fam_counts: dict,
                              tau_centers: jnp.ndarray,
                              n_dom: int = 1,
                              n_frag: int = N_FRAG,
                              n_classes: int = N_CLASS_PER_COMPONENT) -> jnp.ndarray:
    """Cherry log-likelihood for one family under one component."""
    return cherry_log_likelihood(
        linear_params, fam_counts, tau_centers, n_dom, n_frag, n_classes)


# JIT-compiled per-(family, component) LL.
_component_ll_jit = jax.jit(_component_ll_one_family,
                            static_argnames=('n_dom', 'n_frag', 'n_classes'))


def _adjacency_freqs_per_component(linear_params: dict, tau_centers: jnp.ndarray,
                                    n_dom: int = 1,
                                    n_frag: int = N_FRAG,
                                    n_classes: int = N_CLASS_PER_COMPONENT) -> dict:
    """Precompute the per-tau-bin adjacency-frequency tensors F^uv(t) for
    one component. Family-independent — share these across all per-family
    LL computations on the same component.

    Returns:
        Dict mapping COUNT_KEY -> jnp.ndarray of shape (T, ...) where the
        leading axis is over tau bins.
    """
    from tkfmixdom.jax.distill.maraschino_fit import _adjacency_freqs

    def per_t(t):
        return _adjacency_freqs(linear_params, t, n_dom, n_frag, n_classes)

    return jax.vmap(per_t)(tau_centers)


def _family_ll_from_F(F_per_tau: dict, fam_counts: dict) -> jnp.ndarray:
    """Per-family LL given the precomputed per-component F tensors.

    The math mirrors `_pair_ll_one_tau` (post-Match / post-Insert /
    post-Delete / Start blocks) summed across tau bins. Only the
    counts × log(F / Z) contractions are family-dependent; F itself is
    cached per-component.
    """
    eps = 1e-30
    f_mm = jnp.maximum(F_per_tau["MM"], eps)
    f_mi = jnp.maximum(F_per_tau["MI"], eps)
    f_md = jnp.maximum(F_per_tau["MD"], eps)
    f_me = jnp.maximum(F_per_tau["ME"], eps)
    Z_M = (f_mm.sum(axis=(3, 4)) + f_mi.sum(axis=3) + f_md.sum(axis=3) + f_me)
    Z_M = jnp.maximum(Z_M, eps)
    log_ZM = jnp.log(Z_M)
    ll = (jnp.sum(fam_counts["C_MM"] * (jnp.log(f_mm) - log_ZM[:, :, :, None, None]))
          + jnp.sum(fam_counts["C_MI"] * (jnp.log(f_mi) - log_ZM[:, :, :, None]))
          + jnp.sum(fam_counts["C_MD"] * (jnp.log(f_md) - log_ZM[:, :, :, None]))
          + jnp.sum(fam_counts["C_ME"] * (jnp.log(f_me) - log_ZM)))

    f_im = jnp.maximum(F_per_tau["IM"], eps)
    f_ii = jnp.maximum(F_per_tau["II"], eps)
    f_id = jnp.maximum(F_per_tau["ID"], eps)
    f_ie = jnp.maximum(F_per_tau["IE"], eps)
    Z_I = (f_im.sum(axis=(2, 3)) + f_ii.sum(axis=2) + f_id.sum(axis=2) + f_ie)
    Z_I = jnp.maximum(Z_I, eps)
    log_ZI = jnp.log(Z_I)
    ll += (jnp.sum(fam_counts["C_IM"] * (jnp.log(f_im) - log_ZI[:, :, None, None]))
           + jnp.sum(fam_counts["C_II"] * (jnp.log(f_ii) - log_ZI[:, :, None]))
           + jnp.sum(fam_counts["C_ID"] * (jnp.log(f_id) - log_ZI[:, :, None]))
           + jnp.sum(fam_counts["C_IE"] * (jnp.log(f_ie) - log_ZI)))

    f_dm = jnp.maximum(F_per_tau["DM"], eps)
    f_dd = jnp.maximum(F_per_tau["DD"], eps)
    f_di = jnp.maximum(F_per_tau["DI"], eps)
    f_de = jnp.maximum(F_per_tau["DE"], eps)
    Z_D = (f_dm.sum(axis=(2, 3)) + f_dd.sum(axis=2) + f_di.sum(axis=2) + f_de)
    Z_D = jnp.maximum(Z_D, eps)
    log_ZD = jnp.log(Z_D)
    ll += (jnp.sum(fam_counts["C_DM"] * (jnp.log(f_dm) - log_ZD[:, :, None, None]))
           + jnp.sum(fam_counts["C_DD"] * (jnp.log(f_dd) - log_ZD[:, :, None]))
           + jnp.sum(fam_counts["C_DI"] * (jnp.log(f_di) - log_ZD[:, :, None]))
           + jnp.sum(fam_counts["C_DE"] * (jnp.log(f_de) - log_ZD)))

    f_sm = jnp.maximum(F_per_tau["SM"], eps)  # (T, A, A)
    f_si = jnp.maximum(F_per_tau["SI"], eps)
    f_sd = jnp.maximum(F_per_tau["SD"], eps)
    f_se = jnp.maximum(F_per_tau["SE"], eps)
    Z_S = jnp.maximum(f_sm.sum(axis=(1, 2)) + f_si.sum(axis=1)
                       + f_sd.sum(axis=1) + f_se, eps)
    log_ZS = jnp.log(Z_S)
    ll += (jnp.sum(fam_counts["C_SM"] * (jnp.log(f_sm) - log_ZS[:, None, None]))
           + jnp.sum(fam_counts["C_SI"] * (jnp.log(f_si) - log_ZS[:, None]))
           + jnp.sum(fam_counts["C_SD"] * (jnp.log(f_sd) - log_ZS[:, None]))
           + jnp.sum(fam_counts["C_SE"] * (jnp.log(f_se) - log_ZS)))
    return ll


_family_ll_from_F_jit = jax.jit(_family_ll_from_F)


def _family_ll_from_F_vmapped(F_per_tau: dict, batch_counts: dict) -> jnp.ndarray:
    """Vmapped LL: F_per_tau is a single component's per-tau-bin dict;
    batch_counts has a leading family-batch axis on every entry. Returns
    (B,) ll vector for the batch."""
    return jax.vmap(_family_ll_from_F, in_axes=(None, 0))(F_per_tau, batch_counts)


_family_ll_from_F_vmapped_jit = jax.jit(_family_ll_from_F_vmapped)


def family_log_likelihoods(
        families: list[BandedFamily], components: list[dict],
        tau_centers: jnp.ndarray,
        family_batch_size: int | None = None) -> jnp.ndarray:
    """Compute (N_families, K) log-likelihood matrix.

    Strategy: precompute F^uv per component (model-only, family-
    independent) once, then vmap the (cheap) counts × log(F/Z)
    contraction over a batch of families. Vmapping the contraction
    only — NOT the F precomputation — keeps memory bounded because the
    contraction's intermediates are (T, AA^4)-shaped at most, and the
    vmap leading axis multiplies that by `family_batch_size` rather
    than by N.

    Default batch size is 32, picked to keep per-batch peak memory
    below ~2 GB on the F^MM tensor.
    """
    N = len(families)
    K = len(components)
    if family_batch_size is None or family_batch_size < 1:
        family_batch_size = 32

    # Pre-stack family counts into a single dict of (N, *shape) numpy arrays.
    # Slicing the batch indices is then a cheap numpy view per call (rather
    # than per-call dict construction from N families).
    keys = list(families[0].counts.keys())
    stacked: dict[str, np.ndarray] = {}
    for kk in keys:
        stacked[kk] = np.stack([np.asarray(f.counts[kk]) for f in families],
                                axis=0)

    ll = np.zeros((N, K), dtype=np.float64)
    for k, theta_k in enumerate(components):
        params_k = {kk: jnp.asarray(v) for kk, v in theta_k.items()}
        F_cache = _adjacency_freqs_per_component(params_k, tau_centers)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), F_cache)
        for start in range(0, N, family_batch_size):
            end = min(start + family_batch_size, N)
            batch = {kk: jnp.asarray(stacked[kk][start:end]) for kk in keys}
            batch_ll = _family_ll_from_F_vmapped_jit(F_cache, batch)
            ll[start:end, k] = np.asarray(batch_ll, dtype=np.float64)
    return ll


# ---------------------------------------------------------------------------
# Hard-assignment E-step + count aggregation
# ---------------------------------------------------------------------------


def hard_responsibilities(ll_matrix: np.ndarray) -> np.ndarray:
    """Hard family-to-component assignment.

    Returns a (N, K) one-hot matrix where entry [n, argmax_k ll[n,k]] = 1.
    """
    N, K = ll_matrix.shape
    assignments = np.argmax(ll_matrix, axis=1)
    one_hot = np.zeros((N, K), dtype=np.float32)
    one_hot[np.arange(N), assignments] = 1.0
    return one_hot


def aggregate_counts(families: list[BandedFamily],
                      responsibilities: np.ndarray) -> list[dict]:
    """Sum per-family count tensors weighted by responsibilities, per component.

    Args:
        families: list of N BandedFamily.
        responsibilities: (N, K) weights (float32). For hard EM these are
            one-hot rows.

    Returns:
        list of K count-tensor dicts, each with keys C_<XY> + B.
    """
    N, K = responsibilities.shape
    if N != len(families):
        raise ValueError(
            f"responsibilities first axis {N} does not match #families "
            f"{len(families)}")
    if N == 0:
        raise ValueError("no families to aggregate over")

    # Initialise per-component zeros from the first family's tensor shapes.
    # Keep as host numpy — converted to jnp inside cherry_log_likelihood
    # when the inner Adam loop calls value_and_grad.
    template = families[0].counts
    agg: list[dict] = []
    for k in range(K):
        comp = {kk: np.zeros_like(np.asarray(v)) for kk, v in template.items()}
        agg.append(comp)

    for n, fam in enumerate(families):
        for k in range(K):
            w = float(responsibilities[n, k])
            if w == 0.0:
                continue
            for kk, v in fam.counts.items():
                agg[k][kk] = agg[k][kk] + w * np.asarray(v)
    return agg


# ---------------------------------------------------------------------------
# Inner per-component fit: Adam on aggregated counts in banded mode
# ---------------------------------------------------------------------------


def inner_fit_component(
        agg_counts: dict,
        component_init: dict,
        tau_centers: jnp.ndarray, *,
        n_steps: int = 200,
        lr: float = 1e-2,
        n_dom: int = 1,
        n_frag: int = N_FRAG,
        n_classes: int = N_CLASS_PER_COMPONENT,
        freeze_class_S_shape: bool = False,
        freeze_class_pi: bool = False,
        freeze_main_rates: bool = False,
        log_fn=None) -> tuple[dict, float]:
    """Run an Adam loop on aggregated counts for one banded component.

    Args:
        freeze_class_S_shape: when True, hold per-class S^c shape at init
            (taken from ``component_init``); only ``log_class_sigma`` (per
            class) is free. Mirrors ``maraschino fit --freeze-class-S-shape``.
        freeze_class_pi: when True, hold class_pis at init (no gradient on
            ``log_class_pis``). Mirrors ``maraschino fit --freeze-class-pi``.
        freeze_main_rates: when True, hold (main_ins, main_del) at the
            init values supplied via ``component_init``. Useful when each
            component is conceptually a single-domain model and the
            top-level TKF91 should not absorb indel signal that belongs
            to the within-domain TKF92.

    Returns the best (linear_params, log_lik) over the Adam trajectory.
    """
    raw = linear_to_raw(component_init, n_dom, n_frag, n_classes)
    mask_full = _banded_mask_full(n_dom, n_frag)
    freeze_init = {
        'frag_weights': jnp.asarray(component_init['frag_weights'],
                                    dtype=jnp.float32),
        'classdist': jnp.asarray(component_init['classdist'],
                                  dtype=jnp.float32),
    }
    if freeze_class_S_shape:
        freeze_init['class_S_exch_shape'] = jnp.asarray(
            component_init['class_S_exch'], dtype=jnp.float32)
        # New raw param used by raw_to_linear: σ_c = exp(log_class_sigma_c).
        raw['log_class_sigma'] = jnp.zeros(n_classes, dtype=jnp.float32)
    if freeze_class_pi:
        freeze_init['class_pis'] = jnp.asarray(
            component_init['class_pis'], dtype=jnp.float32)
    if freeze_main_rates:
        freeze_init['main_ins'] = jnp.asarray(
            component_init['main_ins'], dtype=jnp.float32)
        freeze_init['main_del'] = jnp.asarray(
            component_init['main_del'], dtype=jnp.float32)

    # cherry_log_likelihood uses jax.vmap over tau bins, which inserts a
    # tracer into the t_idx position; numpy arrays from aggregate_counts
    # would crash on `counts[key][t_idx]`. Lift to jnp once here.
    agg_counts_jax = {kk: jnp.asarray(v) for kk, v in agg_counts.items()}

    def loss_fn(raw_p):
        linp = _materialise_banded(raw_p, n_dom, n_frag, n_classes,
                                    mask_full, freeze_init,
                                    freeze_main_rates=freeze_main_rates)
        return -cherry_log_likelihood(linp, agg_counts_jax, tau_centers,
                                       n_dom, n_frag, n_classes)

    val_grad = jax.jit(jax.value_and_grad(loss_fn))

    opt = optax.adam(lr)
    state = opt.init(raw)

    best_ll = -float('inf')
    best_raw = raw
    for step in range(n_steps):
        loss, grad = val_grad(raw)
        ll = -float(loss)
        if not (np.isnan(ll) or np.isinf(ll)) and ll > best_ll:
            best_ll = ll
            best_raw = raw
        upd, state = opt.update(grad, state, raw)
        raw = optax.apply_updates(raw, upd)
    if log_fn is not None and n_steps > 0:
        log_fn(f"      inner Adam: {n_steps} steps, best LL={best_ll:.4f}")
    final_linear = _materialise_banded(best_raw, n_dom, n_frag, n_classes,
                                        mask_full, freeze_init)
    return final_linear, best_ll


# ---------------------------------------------------------------------------
# Outer EM driver
# ---------------------------------------------------------------------------


def fit_banded_mixture(
        families: list[BandedFamily],
        tau_centers: jnp.ndarray, *,
        K: int,
        seed: int = 0,
        outer_n_iter: int = 30,
        outer_rel_tol: float = 1e-5,
        inner_n_steps: int = 200,
        inner_lr: float = 1e-2,
        p_ext: float = 0.6,
        init_ins: float = 0.01,
        init_del: float = 0.01,
        init_perturb: float = 0.05,
        freeze_class_S_shape: bool = False,
        freeze_class_pi: bool = False,
        freeze_main_rates: bool = False,
        family_batch_size: int = 256,
        log_fn=print) -> tuple[list[dict], np.ndarray, list[dict]]:
    """Fit a K-component banded MixDom2 mixture with hard outer EM.

    Args:
        families: list of N BandedFamily (per-family Maraschino counts).
        tau_centers: (T,) shared tau bin centres.
        K: number of mixture components.
        seed: RNG seed for symmetry breaking.
        outer_n_iter: max outer EM iterations.
        outer_rel_tol: relative LL tolerance for outer convergence.
        inner_n_steps: Adam steps per inner component fit per outer iter.
        inner_lr: Adam learning rate.
        p_ext: initial extension probability for banded init.
        init_ins / init_del: top-level rate inits.
        init_perturb: not currently used (each component gets a different
            seed already; kept for API parity with fit_tkf92_mixture).
        log_fn: progress-printer (default `print`).

    Returns:
        (components, mix_weights, history) tuple.
        components: list of K linear-space param dicts (banded MixDom2,
            1 dom × 3 fragchars × 3 classes each).
        mix_weights: (K,) component weights = #(families assigned)/N.
        history: list of dicts with per-outer-iteration stats.
    """
    N = len(families)
    if N == 0:
        raise ValueError("no families to fit on")
    if K < 1:
        raise ValueError(f"K={K} must be >= 1")

    # ---- Initialise K components with distinct seeds ----
    rng = np.random.RandomState(seed)
    component_seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(K)]
    components: list[dict] = [
        init_banded_component(s, p_ext=p_ext,
                              init_ins=init_ins, init_del=init_del)
        for s in component_seeds]
    log_fn(f"[banded-mix] init: K={K} components, seeds={component_seeds}, "
           f"p_ext={p_ext}, N={N} families")

    history: list[dict] = []
    prev_total_ll = -float('inf')

    for it in range(outer_n_iter):
        t_start = time.monotonic()

        # ---- E-step: per-(family, component) LL → hard assignment ----
        ll_matrix = family_log_likelihoods(
            families, components, tau_centers,
            family_batch_size=family_batch_size)
        resp = hard_responsibilities(ll_matrix)
        assigned_per_k = resp.sum(axis=0)  # (K,)
        total_ll = float(np.sum(np.max(ll_matrix, axis=1)))

        # Diagnostic: how many families per component?
        log_fn(f"[banded-mix] iter {it+1}/{outer_n_iter}: "
               f"total LL={total_ll:.4f}, assigned per k = "
               f"{assigned_per_k.astype(int).tolist()}")

        # ---- Convergence check ----
        if it > 0:
            rel_change = abs(total_ll - prev_total_ll) / max(
                abs(prev_total_ll), 1e-9)
            if rel_change < outer_rel_tol:
                log_fn(f"[banded-mix] converged: rel LL change "
                       f"{rel_change:.2e} < {outer_rel_tol:.2e}")
                history.append({
                    'iter': it + 1,
                    'total_ll': total_ll,
                    'assigned_per_k': assigned_per_k.astype(int).tolist(),
                    'wall_time_s': time.monotonic() - t_start,
                    'converged': True,
                })
                break
        prev_total_ll = total_ll

        # ---- M-step: per-component Adam on responsibility-weighted counts ----
        agg = aggregate_counts(families, resp)
        for k in range(K):
            if assigned_per_k[k] < 1.0:
                # Component k orphaned this iteration: re-seed it from the
                # best-LL family of any other component (helps avoid
                # permanent collapse).
                log_fn(f"  [warn] component k={k} orphaned; "
                       f"re-seeding from worst-fit family")
                # Pick the family with the lowest LL under any component
                # → seed a fresh component aimed at it.
                worst_fam_idx = int(np.argmin(np.max(ll_matrix, axis=1)))
                # Re-init with a different seed
                new_seed = int(rng.randint(0, 2**31 - 1))
                components[k] = init_banded_component(
                    new_seed, p_ext=p_ext,
                    init_ins=init_ins, init_del=init_del)
                # And do a quick Adam pass on just that one family.
                fam_counts = {kk: jnp.asarray(v)
                              for kk, v in families[worst_fam_idx].counts.items()}
                components[k], _ = inner_fit_component(
                    fam_counts, components[k], tau_centers,
                    n_steps=inner_n_steps, lr=inner_lr,
                    freeze_class_S_shape=freeze_class_S_shape,
                    freeze_class_pi=freeze_class_pi,
                    freeze_main_rates=freeze_main_rates,
                    log_fn=log_fn if it == 0 else None)
                continue
            components[k], _ll_k = inner_fit_component(
                agg[k], components[k], tau_centers,
                n_steps=inner_n_steps, lr=inner_lr,
                freeze_class_S_shape=freeze_class_S_shape,
                freeze_class_pi=freeze_class_pi,
                freeze_main_rates=freeze_main_rates,
                log_fn=log_fn if it == 0 else None)

        history.append({
            'iter': it + 1,
            'total_ll': total_ll,
            'assigned_per_k': assigned_per_k.astype(int).tolist(),
            'wall_time_s': time.monotonic() - t_start,
            'converged': False,
        })

    # Final mixture weights from final assignments.
    if history:
        last = history[-1]
        mix = np.array(last['assigned_per_k'], dtype=np.float64)
        mix = mix / max(mix.sum(), 1.0)
    else:
        mix = np.ones(K) / K

    return components, mix, history


# ---------------------------------------------------------------------------
# Output: stack K components into K-domain MixDom2 checkpoint
# ---------------------------------------------------------------------------


def to_mixdom2_checkpoint(components: list[dict], mix_weights: np.ndarray,
                           *,
                           main_ins: float = 0.014,
                           main_del: float = 0.015,
                           t: float = 1.0,
                           em_iter: int = 0,
                           config: dict | None = None) -> dict:
    """Stack K banded components into a K-domain MixDom2 checkpoint.

    The output uses 3K classes total (one per (k, fragchar) pair) with a
    block-diagonal classdist so each domain k uses fragchars
    {3k, 3k+1, 3k+2} as its three site classes.

    Compatible with `train_pfam.py --init <path>`.
    """
    K = len(components)
    AA_loc = AA
    F = N_FRAG
    C = N_CLASS_PER_COMPONENT * K

    # Per-domain rates / weights
    dom_ins = np.asarray([float(np.asarray(c['dom_ins'])[0])
                           for c in components], dtype=np.float64)
    dom_del = np.asarray([float(np.asarray(c['dom_del'])[0])
                           for c in components], dtype=np.float64)
    dom_weights = np.asarray(mix_weights, dtype=np.float64)
    dom_weights = dom_weights / max(dom_weights.sum(), 1e-30)

    # Banded fragdist + ext per component
    frag_weights = np.zeros((K, F), dtype=np.float64)
    frag_weights[:, 0] = 1.0
    ext_rates = np.zeros((K, F, F), dtype=np.float64)
    for k, comp in enumerate(components):
        ext_rates[k] = np.asarray(comp['ext_rates'])[0]

    # Stacked per-class equilibria + exchangeabilities
    class_pis = np.zeros((C, AA_loc), dtype=np.float64)
    class_S_exch = np.zeros((C, AA_loc, AA_loc), dtype=np.float64)
    for k, comp in enumerate(components):
        for c in range(N_CLASS_PER_COMPONENT):
            class_pis[N_CLASS_PER_COMPONENT * k + c] = np.asarray(
                comp['class_pis'])[c]
            class_S_exch[N_CLASS_PER_COMPONENT * k + c] = np.asarray(
                comp['class_S_exch'])[c]

    # Block-diagonal classdist: classdist[k, f, c] = 1 iff c == 3*k + f.
    classdist = np.zeros((K, F, C), dtype=np.float64)
    for k in range(K):
        for f in range(F):
            classdist[k, f, N_CLASS_PER_COMPONENT * k + f] = 1.0

    out = {
        'main_ins': float(main_ins),
        'main_del': float(main_del),
        'dom_ins': dom_ins,
        'dom_del': dom_del,
        'dom_weights': dom_weights,
        'frag_weights': frag_weights,
        'ext_rates': ext_rates,
        'class_pis': class_pis,
        'class_S_exch': class_S_exch,
        'classdist': classdist,
        # Loaders check `n_classes_frag` (Annabel BW convention) AND
        # `n_classes` (train_pfam runtime convention); save both so the
        # warm-start path through `_load_maraschino_as_train_params` →
        # `convert_bw_checkpoint` recognises the mixture as MixDom2.
        'n_classes': int(C),
        'n_classes_frag': int(C),
        'n_dom': int(K),
        'n_frag': int(F),
        't_rep': float(t),
        'em_iter': int(em_iter),
    }
    if config is not None:
        # Stash config under '_config' key (np.savez treats it as object).
        out['_config'] = config
    return out
