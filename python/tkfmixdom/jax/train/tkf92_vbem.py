"""VBEM training for plain TKF92 ancestral reconstruction.

Tests the variational rate-recovery loop in isolation from any MixDom-
specific accumulation. Uses the existing TKF92 variational ELBO
(`varanc_presence.elbo`) as the per-family E-step; adds a closed-form
M-step on (ins_rate, del_rate) via `m_step_indel_quadratic` plus a
Beta posterior on the extension probability `ext`.

Why this exists: the d3f1 → SVI-VBEM warm-start inflated indel rates
~85x relative to baseline. To localize the bug, we need a clean
parameter-recovery test. The simplest such test is on plain TKF92
(no MixDom hierarchy): simulate from known rates, fit with this
pipeline, check whether rates recover.

Public API:
    fit_family_estep_tkf92(binary_tree, leaf_present, ins, del_, ext,
                             n_iter, lr, seed) -> TKF92EStepStats
    extract_tkf92_suff_stats(stats_list, ins, del_, ext) -> dict
    m_step_tkf92(suff_stats, prior=None) -> (ins, del_, ext)
    vbem_train_tkf92(family_provider, init_ins, init_del, init_ext,
                       n_total_families, n_iter, ...) -> (params, history)
"""

from __future__ import annotations

import dataclasses
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..core.bdi import (
    tkf92_stats_from_counts, transition_count_groups, m_step_indel_quadratic,
)
from ..core.params import S as TYPE_S, M as TYPE_M, I as TYPE_I, D as TYPE_D, E as TYPE_E
from ..tree.varanc_presence import (
    edge_lookup, BinaryTree,
    NYI, PRESENT, DELETED,
    elbo as varanc_elbo,
    bp_pair_marginals, make_q_conditionals, make_root_dist, leaf_clamp_to_beta,
)


@dataclasses.dataclass
class TKF92EStepStats:
    elbo: float
    pair_marg: np.ndarray   # (E, L, 3, 3)
    edge_lengths: np.ndarray  # (E,)
    n_edges: int
    n_cols: int


def fit_family_estep_tkf92(binary_tree, leaf_present, ins_rate, del_rate,
                              ext, n_iter=100, lr=0.05, seed=0):
    """Per-family variational E-step: Adam on the TKF92 ELBO.

    The variational q is factorised by edge with TIED logits across
    columns (matches the project convention from varanc-presence): each
    edge has 2 free logits (q_cond entries from NYI parent and from
    PRESENT parent rows), broadcast across columns. The root has a
    single scalar logit. This is the same parameterisation Tree-VBEM
    uses; we keep it identical so any rate-recovery discrepancy is
    NOT attributable to a different q-class.
    """
    L = leaf_present.shape[1]
    le, re = edge_lookup(binary_tree)
    edge_lengths = np.maximum(np.asarray(binary_tree.edge_length), 1e-3)
    binary_tree = binary_tree._replace(edge_length=edge_lengths)

    rng = np.random.default_rng(seed)
    # Per-(edge, col, 2) variational logits — full per-column flexibility.
    # Tying across columns introduces variational bias because the
    # data-conditioned posterior P(z_p, z_c | leaves) varies per column.
    edge_logits = jnp.asarray(
        rng.standard_normal((binary_tree.num_edges, L, 2)) * 0.1,
        dtype=jnp.float64)
    root_logit = jnp.asarray(rng.standard_normal(()) * 0.1, dtype=jnp.float64)

    def neg_elbo(edge_logits, root_logit):
        # edge_logits: (E, L, 2) — already per-(edge, col).
        root_logits = jnp.broadcast_to(root_logit, (L,))
        elbo_total, _ = varanc_elbo(
            edge_logits, root_logits, leaf_present, binary_tree,
            ins_rate, del_rate, ext, le, re)
        return -elbo_total

    grad_fn = jax.jit(jax.grad(neg_elbo, argnums=(0, 1)))
    elbo_fn = jax.jit(neg_elbo)

    opt = optax.adam(lr)
    state = opt.init((edge_logits, root_logit))
    params = (edge_logits, root_logit)

    for _ in range(n_iter):
        grads = grad_fn(*params)
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)

    final_neg_elbo = float(elbo_fn(*params))
    edge_logits_f, root_logit_f = params

    # Final BP for per-edge pair_marg.  edge_logits_f shape (E, L, 2).
    root_logits = jnp.broadcast_to(root_logit_f, (L,))
    q_cond = make_q_conditionals(edge_logits_f)
    root_dist = make_root_dist(root_logits)
    leaf_clamp = leaf_clamp_to_beta(leaf_present)
    pair_marg, _ = bp_pair_marginals(
        q_cond, root_dist, leaf_clamp, binary_tree, le, re)

    return TKF92EStepStats(
        elbo=-final_neg_elbo,
        pair_marg=np.asarray(pair_marg),
        edge_lengths=np.asarray(edge_lengths),
        n_edges=int(binary_tree.num_edges),
        n_cols=int(L),
    )


def fit_family_estep_tkf92_padded(binary_tree, leaf_present, ins_rate,
                                       del_rate, ext, n_iter=100, lr=0.05,
                                       seed=0, target_leaves=None,
                                       target_L=None, raise_on_nan=False):
    """Padded variant of fit_family_estep_tkf92 using JIT-cacheable padded
    ELBO from train.tkf92_padded_elbo.

    The padded version pads (n_leaves, n_cols) to geometric bins and
    passes tree topology as JIT-traced ARGUMENTS, so the JIT cache is
    keyed on shape only — same compiled function reused across all
    families in the same (leaf_bin, col_bin) bucket.  Avoids the GPU
    OOM that the unpadded version hits after ~20 iterations on a 200-
    family minibatch from JIT cache pressure.

    Returns the same TKF92EStepStats type, with pair_marg trimmed back
    to (n_real_edges, L_real, 3, 3) so downstream
    extract_tkf92_suff_stats sees identical inputs.
    """
    from .tkf92_padded_elbo import fit_family_padded_tkf92

    out = fit_family_padded_tkf92(
        binary_tree, leaf_present, ins_rate, del_rate, ext,
        n_iter=n_iter, lr=lr,
        target_leaves=target_leaves, target_L=target_L)

    inputs = out['inputs']
    n_real_edges = inputs['n_real_edges']
    L_real = inputs['L_real']
    pair_marg_padded = out['pair_marg']  # (E_pad, L_pad, 3, 3)
    pair_marg_real = pair_marg_padded[:n_real_edges, :L_real]

    edge_lengths = np.maximum(np.asarray(binary_tree.edge_length), 1e-3)

    return TKF92EStepStats(
        elbo=-out['neg_elbo'],
        pair_marg=np.asarray(pair_marg_real),
        edge_lengths=np.asarray(edge_lengths),
        n_edges=int(binary_tree.num_edges),
        n_cols=int(L_real),
    )


def _compute_n_trans_per_branch(pair_marg_branch):
    """Compute the expected (5, 5) WFST transition count matrix on one branch.

    Reuses the JAX cumulant-trick implementation from
    tree_vbem._compute_W_tensor with τ=1 (single trivial tuple).

    pair_marg_branch: (L, 3, 3) — pair_marg[col, parent_state, child_state]
    Returns: (5, 5) np.ndarray — expected counts of (s_prev → s_next)
        with rows/cols indexed by (S=0, M=1, I=2, D=3, E=4).
    """
    from .tree_vbem import _compute_W_tensor
    L = pair_marg_branch.shape[0]
    q_tau = jnp.ones((L, 1), dtype=jnp.float64)
    W = np.asarray(
        _compute_W_tensor(jnp.asarray(pair_marg_branch), q_tau, 1))
    return W[:, 0, :, 0]


def extract_tkf92_suff_stats(stats_list, ins_rate, del_rate, ext):
    """Aggregate per-family E-step stats into BDI suff stats summed across
    all (family, branch) pairs.

    Returns:
        suff: dict with keys
            'B', 'D', 'S', 'L', 'M', 'T'   — TKF91 BDI stats
            'ext_count'                     — expected fragment-extension events
            'notext_count'                  — expected fragment-non-extension events
    """
    suff = {'B': 0.0, 'D': 0.0, 'S': 0.0, 'L': 0.0, 'M': 0.0, 'T': 0.0,
            'ext_count': 0.0, 'notext_count': 0.0}
    for fs in stats_list:
        for e in range(fs.n_edges):
            t = float(fs.edge_lengths[e])
            n_trans = _compute_n_trans_per_branch(fs.pair_marg[e])
            if n_trans.sum() < 1e-9:
                continue
            # Paper-correct T = t per branch (one BDI process per branch
            # of length t).  body-tkf91.tex eq:exposure-tkf has λ·t in
            # the E[S] formula; the previous T = t · n_trans.sum() was a
            # mis-port from MixDom's intra-domain T += t·n_hat_notkappa
            # (which DOES apply when each domain entry is its own BDI).
            #
            # Note: with paper-correct T=t, the +25% bias of the prior
            # convention is replaced by ~+110% bias because the BP-
            # derived n_trans has spurious counts from the factorised q
            # (e.g. spurious I→I, D→D) that previously got partially
            # masked by the inflated T.  This is a valid empirical
            # demonstration of the variational gap: the column-
            # factorised q cannot represent inter-column correlations
            # induced by chain extension, producing inflated indel rate
            # estimates.  Reported as a known limitation.
            r = tkf92_stats_from_counts(
                n_trans, ins_rate, del_rate, t, ext, T=t)
            groups = transition_count_groups(r['n_trans_resolved'])
            suff['B'] += r['E_B']
            suff['D'] += r['E_D']
            suff['S'] += r['E_S']
            suff['L'] += float(groups['log_kappa'])
            suff['M'] += float(groups['log_1mkappa'])
            suff['T'] += t
            suff['ext_count'] += r['ext_count']
            suff['notext_count'] += r['notext_count']
    return suff


def m_step_tkf92(suff_stats, prior=None, ext_dirichlet=2.0):
    """Closed-form M-step from aggregated suff stats.

    Args:
        suff_stats: dict from extract_tkf92_suff_stats.
        prior: BDI Gamma prior dict {alpha_lam, alpha_mu, beta}, or None
            (defaults to {2, 2, 10}).
        ext_dirichlet: Dirichlet pseudocount on (ext, 1-ext) — keeps ext
            away from 0 / 1 numerically.

    Returns:
        (ins_rate, del_rate, ext) tuple of floats.
    """
    if prior is None:
        prior = {'alpha_lam': 2.0, 'alpha_mu': 2.0, 'beta': 10.0}
    if suff_stats['T'] < 1e-9 or suff_stats['S'] <= 0:
        # Degenerate — return tiny rates.
        return 1e-6, 1e-6, 0.0
    ins_new, del_new = m_step_indel_quadratic(
        suff_stats['B'], suff_stats['D'], suff_stats['S'],
        suff_stats['L'], suff_stats['M'], suff_stats['T'],
        prior_alpha_lam=prior['alpha_lam'],
        prior_alpha_mu=prior['alpha_mu'],
        prior_beta=prior['beta'])
    # ext from Beta posterior with Dirichlet pseudocount.
    ext_n = suff_stats['ext_count'] + ext_dirichlet - 1.0
    notext_n = suff_stats['notext_count'] + ext_dirichlet - 1.0
    ext_n = max(ext_n, 1e-6)
    notext_n = max(notext_n, 1e-6)
    ext_new = ext_n / (ext_n + notext_n)
    return float(ins_new), float(del_new), float(ext_new)


def vbem_train_tkf92(family_provider, init_ins, init_del, init_ext,
                       n_total_families, n_iter=50, n_inner=100, lr=0.05,
                       prior=None, verbose=True, max_families_per_iter=None,
                       use_padding=False):
    """Outer VBEM loop for plain TKF92.

    Args:
        family_provider: callable(idx) -> (binary_tree, leaf_present)
            (NB: no sub_LL — TKF92 VBEM uses indel signal only.)
        init_ins, init_del, init_ext: initial rates.
        n_total_families: corpus size.
        n_iter: outer VBEM iterations.
        n_inner: Adam steps per per-family E-step.
        lr: Adam learning rate.
        prior: BDI Gamma prior (default Gamma(2, 2)/(beta=10) on rates).
        verbose: print per-iter progress.
        max_families_per_iter: optional cap on families processed per iter
            (for full-batch training, set to None = process all).

    Returns:
        (final_params, history): final_params = dict with keys
            'ins_rate', 'del_rate', 'ext'; history = list of dicts.
    """
    ins, del_, ext = float(init_ins), float(init_del), float(init_ext)
    history = []

    n_step = (max_families_per_iter or n_total_families)
    for k in range(n_iter):
        t0 = time.time()
        stats_list = []
        skipped = 0
        for fi in range(min(n_step, n_total_families)):
            try:
                bt, lp = family_provider(fi)
            except Exception as exc:
                skipped += 1
                if skipped <= 3 and verbose:
                    warnings.warn(f"[iter {k}] skipping family idx {fi}: {exc}",
                                   RuntimeWarning, stacklevel=2)
                continue
            try:
                if use_padding:
                    stats = fit_family_estep_tkf92_padded(
                        bt, lp, ins, del_, ext, n_iter=n_inner, lr=lr,
                        seed=int(k) * 100000 + int(fi))
                else:
                    stats = fit_family_estep_tkf92(
                        bt, lp, ins, del_, ext, n_iter=n_inner, lr=lr,
                        seed=int(k) * 100000 + int(fi))
            except Exception as exc:
                skipped += 1
                if skipped <= 3 and verbose:
                    warnings.warn(f"[iter {k}] E-step failed for family {fi}: {exc}",
                                   RuntimeWarning, stacklevel=2)
                continue
            stats_list.append(stats)
        if not stats_list:
            warnings.warn(f"[iter {k}] no families succeeded — abort",
                           RuntimeWarning, stacklevel=2)
            break
        e_time = time.time() - t0

        suff = extract_tkf92_suff_stats(stats_list, ins, del_, ext)
        ins_new, del_new, ext_new = m_step_tkf92(suff, prior=prior)
        mean_elbo = float(np.mean([s.elbo for s in stats_list]))
        if verbose:
            print(f'[iter {k}] mean_elbo={mean_elbo:.2f}  '
                  f'ins={ins:.4f}→{ins_new:.4f}  '
                  f'del={del_:.4f}→{del_new:.4f}  '
                  f'ext={ext:.4f}→{ext_new:.4f}  '
                  f'(n_fams={len(stats_list)}, E-step {e_time:.1f}s)')
        history.append({
            'iter': k,
            'mean_elbo': mean_elbo,
            'ins_pre': ins, 'del_pre': del_, 'ext_pre': ext,
            'ins_post': ins_new, 'del_post': del_new, 'ext_post': ext_new,
            'suff_B': suff['B'], 'suff_D': suff['D'],
            'suff_S': suff['S'], 'suff_T': suff['T'],
            'suff_L': suff['L'], 'suff_M': suff['M'],
            'suff_ext': suff['ext_count'],
            'suff_notext': suff['notext_count'],
            'n_families': len(stats_list),
            'e_time': e_time,
        })
        ins, del_, ext = ins_new, del_new, ext_new

    return {'ins_rate': ins, 'del_rate': del_, 'ext': ext}, history


def _empty_suff_tkf92():
    return {'B': 0.0, 'D': 0.0, 'S': 0.0, 'L': 0.0, 'M': 0.0, 'T': 0.0,
            'ext_count': 0.0, 'notext_count': 0.0}


def _ema_blend(suff_blend, suff_minibatch, eta, scale):
    """SVI EMA: suff_blend ← (1 - eta) * suff_blend + eta * scale * suff_mb."""
    out = {}
    for k in suff_blend:
        out[k] = (1.0 - eta) * suff_blend[k] + eta * scale * suff_minibatch[k]
    return out


def svi_vbem_train_tkf92(family_provider, init_ins, init_del, init_ext, *,
                            n_total_families, family_indices=None,
                            n_iter=200, batch_size=10, n_inner=30, lr=0.05,
                            svi_tau=10.0, svi_kappa=0.7,
                            prior=None,
                            seed=0, verbose=True, log_fn=None,
                            val_fn=None, val_every_k=10,
                            use_padding=False,
                            iter_callback=None):
    """Stochastic VBEM (SVI) for plain TKF92, paper-aligned.

    Per body-tkf92.tex sec:bw-tkf92, every iteration:
      1. sample a minibatch of family indices (breadth-first if
         ``family_indices`` is given, else random uniform);
      2. run ``fit_family_estep_tkf92`` per family, accumulate suff stats
         via ``extract_tkf92_suff_stats`` (with the paper-aligned
         T = t per branch and L/M on the resolved n̂ matrix);
      3. EMA blend the minibatch suff stats into the global accumulator
         with step size η_k = (svi_tau + k)^(−svi_kappa);
      4. M-step: ``m_step_tkf92`` on the blended accumulator.

    Args:
        family_provider:    callable(idx) -> (binary_tree, leaf_present).
        init_ins, init_del, init_ext: initial rates.
        n_total_families:   corpus size (used to scale per-minibatch
                            suff stats so the M-step sees full-corpus
                            magnitude).
        family_indices:     optional list of family indices; if given,
                            samples breadth-first (cycles through with
                            shuffling per epoch).  Else uniform random.
        n_iter:             outer SVI iterations.
        batch_size:         families per iter.
        n_inner:            Adam inner-iters per per-family E-step.
        lr:                 Adam lr for E-step.
        svi_tau, svi_kappa: SVI step-size schedule.
        prior:              passed to m_step_tkf92.
        seed:               RNG seed for sampler.
        verbose:            print per-iter progress.
        log_fn:             optional logger; defaults to print.
        val_fn:             optional callable((ins, del_, ext)) -> float
                            for held-out log-lik tracking.
        val_every_k:        run val_fn every this many iters.

    Returns:
        dict with 'ins_rate', 'del_rate', 'ext', 'history'.
    """
    if log_fn is None:
        log_fn = print if verbose else (lambda *a, **kw: None)
    rng = np.random.default_rng(seed)

    ins, del_, ext = float(init_ins), float(init_del), float(init_ext)
    suff_blend = _empty_suff_tkf92()
    history = []
    val_history = []

    if family_indices is None:
        family_indices = list(range(n_total_families))
    n_in_pool = len(family_indices)
    scale = float(n_total_families) / float(batch_size)

    # Breadth-first sampler: shuffle once per epoch, draw without replacement.
    epoch_perm = rng.permutation(n_in_pool).tolist()
    epoch_pos = 0

    t0 = time.time()
    for k in range(n_iter):
        eta_k = (svi_tau + k) ** (-svi_kappa)
        # Refill epoch buffer if exhausted.
        if epoch_pos + batch_size > n_in_pool:
            epoch_perm = rng.permutation(n_in_pool).tolist()
            epoch_pos = 0
        batch_idx_in_pool = epoch_perm[epoch_pos:epoch_pos + batch_size]
        epoch_pos += batch_size
        batch_idx = [family_indices[i] for i in batch_idx_in_pool]

        # Per-family E-step.
        stats_list = []
        skipped = 0
        for fi in batch_idx:
            try:
                bt, lp = family_provider(fi)
            except Exception as exc:
                skipped += 1
                if skipped <= 3:
                    warnings.warn(
                        f"[iter {k}] skip fam idx {fi}: {exc}",
                        RuntimeWarning, stacklevel=2)
                continue
            try:
                if use_padding:
                    stats = fit_family_estep_tkf92_padded(
                        bt, lp, ins, del_, ext, n_iter=n_inner, lr=lr,
                        seed=int(k) * 100000 + int(fi))
                else:
                    stats = fit_family_estep_tkf92(
                        bt, lp, ins, del_, ext, n_iter=n_inner, lr=lr,
                        seed=int(k) * 100000 + int(fi))
                stats_list.append(stats)
            except Exception as exc:
                skipped += 1
                if skipped <= 3:
                    warnings.warn(
                        f"[iter {k}] E-step fail fam {fi}: {exc}",
                        RuntimeWarning, stacklevel=2)
        if not stats_list:
            log_fn(f"[iter {k}] no fams succeeded; eta={eta_k:.4f}")
            continue

        suff_mb = extract_tkf92_suff_stats(stats_list, ins, del_, ext)
        # SVI EMA: at iter 0, fully replace.
        if k == 0:
            suff_blend = {kk: scale * suff_mb[kk] for kk in suff_mb}
        else:
            suff_blend = _ema_blend(suff_blend, suff_mb, eta_k, scale)

        ins_new, del_new, ext_new = m_step_tkf92(suff_blend, prior=prior)

        mean_elbo = float(np.mean([s.elbo for s in stats_list]))
        elapsed = time.time() - t0
        if verbose and ((k + 1) % 5 == 0 or k == 0 or k == n_iter - 1):
            log_fn(f'[iter {k+1:>4}/{n_iter}] '
                    f'eta={eta_k:.4f} mean_elbo/fam={mean_elbo:.2f} '
                    f'ins={ins_new:.5f} del={del_new:.5f} ext={ext_new:.4f} '
                    f'(n_fams={len(stats_list)}, {elapsed:.1f}s)')
        history.append({
            'iter': k + 1, 'eta': eta_k,
            'mean_elbo': mean_elbo,
            'ins': ins_new, 'del_': del_new, 'ext': ext_new,
            'n_fams_used': len(stats_list),
        })
        ins, del_, ext = ins_new, del_new, ext_new
        if iter_callback is not None:
            iter_callback(k + 1, ins, del_, ext, list(history))

        if val_fn is not None and ((k + 1) % val_every_k == 0):
            val_score = val_fn((ins, del_, ext))
            val_history.append({'iter': k + 1, 'val': val_score})
            log_fn(f'  [val] iter {k+1}: val_score={val_score:.4f}')

    return {
        'ins_rate': ins, 'del_rate': del_, 'ext': ext,
        'history': history, 'val_history': val_history,
    }
