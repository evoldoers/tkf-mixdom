"""Adam optimizer helpers for MixDom parameters (Phase 5
(e_step, expected_ll) split — see python/ADAM_REFACTOR_PLAN.md).

Uses standard JAX autograd through the closed-form Q-function
`adam_loss_from_delta`, which consumes per-pair-t-correct sufficient
statistics produced by the SVI-BW E-step
(`train_pfam._process_pairs_batched(_2d)`). The Q-function is
parameter-shape only; the bwd graph contains no FB and no per-pair
custom VJP. The legacy per-pair compile-storm path
(`pair_log_likelihood`, `adam_train`, `adam_train_constrained`,
`minibatch_objective`) was deleted in Phase 5 step 5 (commit
458971606).

Public API in this module:
  - `to_unconstrained` / `to_constrained` / `_logit` / `log_prior`
        — parameter transforms and prior, used by both Adam and
          (potentially) any other optimiser of the MixDom params.
  - `adam_loss_from_delta(uc, delta, n_dom, n_frag, Q=None, pi=None,
        ...)` — the Q-function loss; consumes a δ dict from
        `_process_pairs_batched(_2d)`.
  - `make_adam_step(optimizer, n_dom, n_frag, Q=None, pi=None, ...)`
        — returns a JIT'd step function `(uc, opt_state, delta) ->
        (new_uc, new_opt_state, loss)`.

The training entry point is `train_pfam.train_adam(args)` (in
`python/train_pfam.py`), which orchestrates the minibatch sampling,
E-step call, and step_fn invocation. This module does not own the
training loop.

Per-pair-t correctness: the Adam loss is t-parameter-free; t enters
only through the BDI sufficient statistics
(B, D, L, M, S, T = `delta['exact_ss']`'s top_E_B/top_E_D/top_E_S/
top_T_obs/top_5x5 columns) computed per pair at each pair's own
t_p inside `exact_suffstats_per_pair_batch`, then summed across the
minibatch. Substitution / π / classdist axes use per-pair-HR
aggregates (`dom_W` / `dom_U` / `class_W` / `class_U` / ...). NO
`t_rep`. NO consumption of `delta['agg_n_chi']` (a diagnostic
aggregate; see `.claude/examples/per_pair_t_chi_axis_recidivism.md`).
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial


# ============================================================
# Unconstrained ↔ constrained parameter transforms
# ============================================================

def to_unconstrained(params, n_dom, n_frag):
    """Convert constrained MixDom params to unconstrained space for optimization.

    All MixDom variants carry per-domain (S_exch, π) — the substitution
    model is per-domain; vanilla shared-Q-across-all-domains MixDom1
    was deleted in Phase 5.6 Sub-phase B.

      - log_dom_S_upper: (D, A*(A-1)//2) — exp → upper triangle of
        per-domain S_exch, then symmetrised. Diagonal stays zero.
      - log_dom_pi_logits: (D, A) — softmax → per-domain dom_pis.

    MixDom2 (n_classes>1, classdist/class_pis/class_S_exch present):
    additionally carry per-class
      - log_class_pis_logits: (C, A) — softmax → class_pis
      - log_class_S_exch_upper: (C, A*(A-1)//2) — exp → upper triangle of
        class_S_exch, then symmetrised. Diagonal stays zero.
      - log_classdist_logits: (D, F, C) — softmax over the C axis →
        per-(d, f) classdist.

    Required keys in `params`: dom_S_exch (D, A, A) and dom_pis (D, A).
    Init paths (mixdom_init.py / train_pfam.py init) populate these
    unconditionally; legacy single-Q checkpoints lacking them fail
    fast.
    """
    uc = {
        'log_main_ins': jnp.log(jnp.array(params['main_ins'])),
        'log_main_del': jnp.log(jnp.array(params['main_del'])),
        'log_dom_ins': jnp.log(jnp.array(params['dom_ins'])),
        'log_dom_del': jnp.log(jnp.array(params['dom_del'])),
        'log_dom_weights': jnp.log(jnp.array(params['dom_weights'])),
        'log_frag_weights': jnp.log(jnp.array(params['frag_weights'])),
        'logit_ext_rates': _logit(jnp.array(params['ext_rates'])),
    }
    if 'dom_S_exch' not in params or 'dom_pis' not in params:
        raise KeyError(
            "to_unconstrained requires params['dom_S_exch'] (D, A, A) and "
            "params['dom_pis'] (D, A). Vanilla shared-Q MixDom1 was "
            "removed in Phase 5.6 Sub-phase B; rebuild from a checkpoint "
            "or init helper that populates per-domain (S, π).")
    dom_S = jnp.asarray(params['dom_S_exch'])             # (D, A, A)
    dom_pis = jnp.asarray(params['dom_pis'])              # (D, A)
    A = dom_S.shape[-1]
    iu_i, iu_j = jnp.triu_indices(A, k=1)
    uc['log_dom_S_upper'] = jnp.log(
        jnp.maximum(dom_S[:, iu_i, iu_j], 1e-30))         # (D, A*(A-1)//2)
    uc['log_dom_pi_logits'] = jnp.log(jnp.maximum(dom_pis, 1e-30))  # (D, A)

    n_classes = int(params.get('n_classes', 1))
    if n_classes > 1 and 'class_pis' in params:
        cp = jnp.asarray(params['class_pis'])             # (C, A)
        uc['log_class_pis_logits'] = jnp.log(jnp.maximum(cp, 1e-30))
        S = jnp.asarray(params['class_S_exch'])           # (C, A, A)
        uc['log_class_S_exch_upper'] = jnp.log(
            jnp.maximum(S[:, iu_i, iu_j], 1e-30))         # (C, A*(A-1)//2)
        cd = jnp.asarray(params['classdist'])             # (D, F, C)
        uc['log_classdist_logits'] = jnp.log(jnp.maximum(cd, 1e-30))
    return uc


def to_constrained(uc, n_dom, n_frag):
    """Convert unconstrained params back to constrained space."""
    main_ins = jnp.exp(uc['log_main_ins'])
    main_del = jnp.exp(uc['log_main_del'])
    # Ensure mu > lambda
    main_del = jnp.maximum(main_del, main_ins + 1e-6)
    dom_ins = jnp.exp(uc['log_dom_ins'])
    dom_del = jnp.exp(uc['log_dom_del'])
    dom_del = jnp.maximum(dom_del, dom_ins + 1e-6)
    dom_weights = jax.nn.softmax(uc['log_dom_weights'])
    frag_weights = jax.nn.softmax(uc['log_frag_weights'], axis=-1)
    ext_rates = jax.nn.sigmoid(uc['logit_ext_rates'])

    # Per-domain (S_exch, π) — always present after Phase 5.6 Sub-phase B.
    dom_pi_logits = uc['log_dom_pi_logits']               # (D, A)
    dom_pis = jax.nn.softmax(dom_pi_logits, axis=-1)
    dom_S_upper = jnp.exp(uc['log_dom_S_upper'])          # (D, n_pairs)
    n_dom_uc, n_pairs = dom_S_upper.shape
    A = int((1 + (1 + 8 * n_pairs) ** 0.5) / 2)
    iu_i, iu_j = jnp.triu_indices(A, k=1)
    dom_S_exch = jnp.zeros((n_dom_uc, A, A))
    dom_S_exch = dom_S_exch.at[:, iu_i, iu_j].set(dom_S_upper)
    dom_S_exch = dom_S_exch + dom_S_exch.swapaxes(-1, -2)

    # Build per-domain Q[d] = S[d] · π[d] (off-diag), row-sum-zero diag.
    from ..core.ctmc import build_Q_from_S_pi
    dom_Qs = jax.vmap(build_Q_from_S_pi, in_axes=(0, 0))(dom_S_exch, dom_pis)

    result = {
        'main_ins': main_ins,
        'main_del': main_del,
        'dom_ins': dom_ins,
        'dom_del': dom_del,
        'dom_weights': dom_weights,
        'frag_weights': frag_weights,
        'ext_rates': ext_rates,
        'dom_S_exch': dom_S_exch,
        'dom_pis': dom_pis,
        'dom_Qs': dom_Qs,
    }
    if 'log_class_pis_logits' in uc:
        cp_logits = uc['log_class_pis_logits']            # (C, A)
        class_pis = jax.nn.softmax(cp_logits, axis=-1)    # (C, A)
        S_upper = jnp.exp(uc['log_class_S_exch_upper'])   # (C, n_pairs)
        n_classes, n_pairs_c = S_upper.shape
        A_c = int((1 + (1 + 8 * n_pairs_c) ** 0.5) / 2)
        iu_i_c, iu_j_c = jnp.triu_indices(A_c, k=1)
        class_S_exch = jnp.zeros((n_classes, A_c, A_c))
        class_S_exch = class_S_exch.at[:, iu_i_c, iu_j_c].set(S_upper)
        class_S_exch = class_S_exch + class_S_exch.swapaxes(-1, -2)
        cd_logits = uc['log_classdist_logits']            # (D, F, C)
        classdist = jax.nn.softmax(cd_logits, axis=-1)
        result.update({
            'n_classes': n_classes,
            'class_pis': class_pis,
            'class_S_exch': class_S_exch,
            'classdist': classdist,
        })
    return result


def _logit(x):
    x = jnp.clip(x, 1e-6, 1 - 1e-6)
    return jnp.log(x / (1 - x))


# ============================================================
# Log-prior (Gamma on rates, Dirichlet on weights, Beta on ext)
# ============================================================

def log_prior(uc, ins_prior=(2.0, 10.0), del_prior=(2.0, 10.0),
              dom_dirichlet=1.5, frag_dirichlet=1.5,
              ext_alpha=2.0, ext_beta=3.0,
              classdist_dirichlet=1.0,
              class_pi_dirichlet=1.0,
              class_S_gamma=(1.0, 0.0),
              dom_pi_dirichlet=1.0,
              dom_S_gamma=(1.0, 0.0)):
    """Log-prior in unconstrained space.

    Gamma(alpha, beta) on each rate, Dirichlet(alpha) on weights,
    Beta(alpha, beta) on extension rates.

    Per-domain (S_exch, π) priors (always present after Phase 5.6
    Sub-phase B):
      - Dirichlet(dom_pi_dirichlet) on dom_pis[d, :]. Default 1.0 = uniform.
      - Gamma(dom_S_gamma) per off-diag dom_S_exch[d, i, j] (i<j).
        Default `(1.0, 0.0)` is the improper flat density `s^{-1}`
        (non-integrable). Inherited from the pre-existing `class_S_gamma`
        default. Combined with the LG-tile init, Adam is essentially
        unregularised on the per-domain S exchangeabilities — for any
        rarely-seen amino-acid pair `(a, b)`, dom_S_exch[d, a, b] can
        drift toward 0 or ∞. Set a finite rate (e.g. `(2.0, 1.0)`) for
        a weakly-informative prior; set the same on `class_S_gamma`
        for MixDom2 consistency.

    MixDom2 (when uc has site-class keys), additionally:
      - Dirichlet(classdist_dirichlet) on classdist[d, f, :].
      - Dirichlet(class_pi_dirichlet) on class_pis[c, :].
      - Gamma(shape, rate) per off-diag class_S_exch[c, i, j] (i<j),
        defaulting to (1, 0) = improper flat.
    """
    lp = 0.0

    for key, (alpha, beta) in [('log_main_ins', ins_prior),
                                ('log_main_del', del_prior)]:
        r = jnp.exp(uc[key])
        lp += (alpha - 1) * uc[key] - beta * r

    for key, (alpha, beta) in [('log_dom_ins', ins_prior),
                                ('log_dom_del', del_prior)]:
        r = jnp.exp(uc[key])
        lp += jnp.sum((alpha - 1) * uc[key] - beta * r)

    lp += (dom_dirichlet - 1) * jnp.sum(jax.nn.log_softmax(uc['log_dom_weights']))
    lp += (frag_dirichlet - 1) * jnp.sum(jax.nn.log_softmax(uc['log_frag_weights']))

    ext = jax.nn.sigmoid(uc['logit_ext_rates'])
    lp += jnp.sum((ext_alpha - 1) * jnp.log(ext + 1e-30)
                   + (ext_beta - 1) * jnp.log(1 - ext + 1e-30))

    # Per-domain (S_exch, π) priors — always present.
    lp += (dom_pi_dirichlet - 1) * jnp.sum(
        jax.nn.log_softmax(uc['log_dom_pi_logits'], axis=-1))
    s_log_dom = uc['log_dom_S_upper']            # (D, A*(A-1)//2)
    s_val_dom = jnp.exp(s_log_dom)
    s_shape, s_rate = dom_S_gamma
    lp += jnp.sum((s_shape - 1) * s_log_dom - s_rate * s_val_dom)

    # MixDom2 site-class priors (only contribute when MixDom2 is active).
    if 'log_class_pis_logits' in uc:
        # Dirichlet on each class_pis[c, :]
        lp += (class_pi_dirichlet - 1) * jnp.sum(
            jax.nn.log_softmax(uc['log_class_pis_logits'], axis=-1))
        # Dirichlet on each classdist[d, f, :]
        lp += (classdist_dirichlet - 1) * jnp.sum(
            jax.nn.log_softmax(uc['log_classdist_logits'], axis=-1))
        # Gamma(shape, rate) on off-diag class_S_exch[c, i, j] (i<j),
        # parameterised via log of upper triangle.
        s_log = uc['log_class_S_exch_upper']  # (C, A*(A-1)//2)
        s_val = jnp.exp(s_log)
        s_shape, s_rate = class_S_gamma
        lp += jnp.sum((s_shape - 1) * s_log - s_rate * s_val)

    return lp


# ============================================================
# Phase 5 / 5.6: Adam loss with per-domain substitution model.
#
# `adam_loss_from_delta` computes -[Q(θ; δ) + log_prior(θ)] / n_pairs
# where δ is the suff-stat dict from SVI-BW's batched E-step
# (`_process_pairs_batched(_2d)`). Q has four parts:
#
#   chi:        chi_q_from_bdi(params, delta['exact_ss'])
#                 — per-pair-t-correct, t-parameter-free.
#   subst_dom:  _subst_q_from_dom_suffstats(dom_Qs, dom_W, dom_U)
#                 — per-domain HR.
#   pi_dom:     _dom_pis_linear_q_term(dom_pis, dom_match/insert/delete)
#                 — V_d log π_d (per-domain observation-only V_d).
#   subst_cls:  _subst_q_from_class_suffstats (MixDom2)
#                 — per-class HR.
#   pi_cls:     _class_pis_linear_q_term (MixDom2)
#                 — V_c log π_c.
#   classdist:  _classdist_q_term (MixDom2 only).
#
# `chi_q_from_bdi` does NOT consume `delta['agg_n_chi']` — that is the
# diagnostic aggregate (a t_rep trap; see
# `.claude/examples/per_pair_t_chi_axis_recidivism.md`).
#
# JIT signature: parameter-shape only — δ enters as a leaf input
# treated as a fixed tensor by autograd.
# ============================================================


def adam_loss_from_delta(uc, delta, n_dom, n_frag,
                          ins_prior=(2.0, 10.0), del_prior=(2.0, 10.0),
                          dom_dirichlet=1.5, frag_dirichlet=1.5,
                          ext_alpha=2.0, ext_beta=3.0,
                          classdist_dirichlet=1.0, class_pi_dirichlet=1.0,
                          class_S_gamma=(1.0, 0.0),
                          dom_pi_dirichlet=1.0, dom_S_gamma=(1.0, 0.0)):
    """Per-minibatch Adam loss = -[Q(θ; δ) + log_prior(θ)] / n_pairs.

    Args:
        uc: unconstrained params dict. After Phase 5.6 Sub-phase B always
            carries `log_dom_S_upper` (D, A·(A−1)/2) and
            `log_dom_pi_logits` (D, A) for per-domain (S, π); MixDom2
            additionally carries the per-class triple
            `log_class_pis_logits`, `log_class_S_exch_upper`,
            `log_classdist_logits`.
        delta: dict from `_process_pairs_batched(_2d)`. Required keys:
            - 'exact_ss': dict from `exact_suffstats_per_pair_batch`
              (top_5x5, dom_M_5x5, dom_w, frag_w, ext, term, top_E_B,
               top_E_D, top_E_S, top_T_obs, dom_E_B, dom_E_D, dom_E_S,
               dom_T_obs).
            - 'n_pairs': int.
            - Per-domain HR: 'dom_W' (D, A), 'dom_U' (D, A, A),
              'dom_match_counts' (D, A, A), 'dom_insert_counts' (D, A),
              'dom_delete_counts' (D, A).
            - For MixDom2: above + 'class_W' (C, A), 'class_U' (C, A, A),
              'class_match_counts' (C, A, A), 'class_insert_counts' (C, A),
              'class_delete_counts' (C, A), 'classdist_counts' (D, F, C).
        n_dom, n_frag: model dimensions.
        ins_prior, ..., dom_S_gamma: prior hyperparameters.

    Returns:
        scalar loss (negative log P + negative log prior, normalised
        per pair).

    Per-pair-t correctness:
      - chi axis: BDI form via chi_q_from_bdi — t-parameter-free.
      - substitution / pi / classdist: per-pair-t-correct via per-pair
        HR aggregates (dom_W, dom_U, class_W, class_U, ...) computed
        in the E-step at each pair's own t_p, then summed.

    JIT signature: parameter-shape only — `delta` enters as a leaf
    treated as a fixed tensor by autograd.

    Phase 5.6 Sub-phase B: `Q, pi` external kwargs are gone. The
    substitution model is per-domain — `to_constrained(uc)` builds
    `dom_S_exch`, `dom_pis`, and `dom_Qs` from `uc`, and Adam
    differentiates them via `_subst_q_from_dom_suffstats` +
    `_dom_pis_linear_q_term`.
    """
    from .likelihood import (
        chi_q_from_bdi,
        _subst_q_from_class_suffstats,
        _subst_q_from_dom_suffstats,
        _class_pis_linear_q_term,
        _classdist_q_term,
        _dom_pis_linear_q_term,
    )

    is_mixdom2 = 'log_class_pis_logits' in uc
    params = to_constrained(uc, n_dom, n_frag)

    # ---- chi (BDI form, t-parameter-free) ----
    q_chi = chi_q_from_bdi(params, delta['exact_ss'])

    # ---- per-domain substitution + π (always present) ----
    q_subst_dom = _subst_q_from_dom_suffstats(
        jnp.asarray(params['dom_Qs']),
        jnp.asarray(delta['dom_W']),
        jnp.asarray(delta['dom_U']))
    q_pi_dom = _dom_pis_linear_q_term(
        jnp.asarray(params['dom_pis']),
        jnp.asarray(delta['dom_match_counts']),
        jnp.asarray(delta['dom_insert_counts']),
        jnp.asarray(delta['dom_delete_counts']))

    # ---- per-class additions for MixDom2 ----
    if is_mixdom2:
        q_subst_class = _subst_q_from_class_suffstats(
            jnp.asarray(params['class_S_exch']),
            jnp.asarray(params['class_pis']),
            jnp.asarray(delta['class_W']),
            jnp.asarray(delta['class_U']))
        q_pi_class = _class_pis_linear_q_term(
            jnp.asarray(params['class_pis']),
            jnp.asarray(delta['class_match_counts']),
            jnp.asarray(delta['class_insert_counts']),
            jnp.asarray(delta['class_delete_counts']))
        q_classdist = _classdist_q_term(
            jnp.asarray(params['classdist']),
            jnp.asarray(delta['classdist_counts']))
        q_obs = q_chi + q_subst_dom + q_pi_dom + q_subst_class + q_pi_class + q_classdist
    else:
        q_obs = q_chi + q_subst_dom + q_pi_dom

    # ---- prior ----
    lp = log_prior(uc, ins_prior, del_prior, dom_dirichlet,
                    frag_dirichlet, ext_alpha, ext_beta,
                    classdist_dirichlet=classdist_dirichlet,
                    class_pi_dirichlet=class_pi_dirichlet,
                    class_S_gamma=class_S_gamma,
                    dom_pi_dirichlet=dom_pi_dirichlet,
                    dom_S_gamma=dom_S_gamma)

    n_pairs = jnp.maximum(jnp.asarray(delta['n_pairs'], dtype=q_obs.dtype),
                            1.0)
    return -(q_obs + lp) / n_pairs


def make_adam_step(optimizer, n_dom, n_frag,
                    ins_prior=(2.0, 10.0), del_prior=(2.0, 10.0),
                    dom_dirichlet=1.5, frag_dirichlet=1.5,
                    ext_alpha=2.0, ext_beta=3.0,
                    classdist_dirichlet=1.0, class_pi_dirichlet=1.0,
                    class_S_gamma=(1.0, 0.0),
                    dom_pi_dirichlet=1.0, dom_S_gamma=(1.0, 0.0)):
    """Return a JIT-compiled Adam step `step_fn(uc, opt_state, delta)
    -> (new_uc, new_opt_state, loss)`.

    The closure binds the optimizer + hyperparameters once. Per-step
    cost (after first compile):
      - One value_and_grad of `adam_loss_from_delta`.
      - One `optimizer.update` + `apply_updates`.
      - No FB inside the bwd graph; no per-(Lx_pad, Ly_pad) compile.
        Cache size = 1 (parameter-shape-only signature).

    All substitution-model parameters are in `uc` (per-domain
    `log_dom_S_upper`, `log_dom_pi_logits`; per-class equivalents for
    MixDom2). No `Q, pi` external kwargs — those entered the loss as
    fixed constants in the pre-Phase-5.6 design and prevented Adam
    from differentiating the substitution model.
    """
    import optax

    @jax.jit
    def _step(uc, opt_state, delta):
        loss, grads = jax.value_and_grad(adam_loss_from_delta)(
            uc, delta, n_dom, n_frag,
            ins_prior, del_prior, dom_dirichlet, frag_dirichlet,
            ext_alpha, ext_beta,
            classdist_dirichlet, class_pi_dirichlet, class_S_gamma,
            dom_pi_dirichlet, dom_S_gamma)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_uc = optax.apply_updates(uc, updates)
        return new_uc, new_opt_state, loss

    return _step
