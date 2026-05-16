"""MixDom2 fresh-init for top-level training.

Single source of truth for how `train_pfam.py` (and any other trainer
that wants seed-comparable initialisation, e.g. a maraschino fit-mode
rewrite per `maraschino_to_mixdom2_brief.md`) constructs the initial
MixDom2 parameter dict.

Honours the same CLI flags `train_pfam.py` exposes:
  --init-ins / --init-del              base TKF rates
  --seed                               RNG seed (deterministic init)
  --estimate-subst                     populate dom_Qs, dom_pis, dom_S_exch
  --n-classes                          int; 0 → max(n_dom, n_frag)
  --class-pi-init                      lg_noisy / c10 / c10_topN / c20
  --pi-init-noise-frac                 float (only used when class_pi_init=lg_noisy)
  --classdist-init                     auto / identity
  --class-pis-from-dom-pis             bool: class_pis[c] = dom_pis[c]
  --classdist-noise-frac               float Dirichlet perturbation on classdist init

Output: dict with keys matching what `train_pfam.py` writes to its
`.npz` checkpoint for an n_classes>1 run, namely:

    main_ins (scalar), main_del (scalar)
    dom_ins[D], dom_del[D]
    dom_weights[D], frag_weights[D, F], ext_rates[D, F, F]
    (only if --estimate-subst:)
        dom_Qs[D, A, A], dom_pis[D, A], dom_S_exch[D, A, A]
    (only if n_classes > 1:)
        n_classes (int)
        class_pis[C, A], class_S_exch[C, A, A]
        classdist[D, F, C]

This module is the canonical place to make any future change to fresh
init. `train_pfam.py` calls `init_mixdom2_params_from_args`. The
maraschino MixDom2 rewrite is expected to call the same function so a
seed-N fresh init produces bit-identical params on both trainers.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from tkfmixdom.jax.models.compiled import init_mixdom_params


AA = 20


def init_mixdom2_params_from_args(
    args: Any,
    n_dom: int,
    n_frag: int,
    Q_lg,
    pi_lg,
    *,
    log_fn: Callable[[str], None] = print,
) -> dict:
    """Build a fresh MixDom2 param dict from CLI args.

    Args:
        args: argparse namespace (or duck-typed equivalent) with the
            attributes documented in the module header.
        n_dom: number of top-level TKF91 domains.
        n_frag: number of TKF92 fragments per domain.
        Q_lg: (A, A) LG rate matrix (jax array OK).
        pi_lg: (A,) LG equilibrium AA distribution (jax array OK).
        log_fn: progress-line sink. Defaults to ``print``; pass
            train_pfam's ``_log`` helper to keep the logged init lines.

    Returns:
        Param dict ready to drop into the SVI-BW loop or any compatible
        downstream consumer.
    """
    log_fn("Initializing fresh parameters...")
    main_ins, main_del, dom_ins, dom_del, dom_weights, frag_weights, ext_rates = \
        init_mixdom_params(args.init_ins, args.init_del, n_dom, n_frag,
                           seed=args.seed)
    # Optional banded 3-fragchar init (overrides random frag_weights /
    # ext_rates from init_mixdom_params). See tkf/substitution-mstep.tex
    # via FragStart/FragMid/FragEnd interpretation; structurally-zero
    # entries are pinned at 0 by the M-step pseudocount mask in
    # train_pfam.py.
    if getattr(args, 'banded_frag_init', False):
        if n_frag != 3:
            raise ValueError(
                f"--banded-frag-init requires n_frag==3, got {n_frag}")
        from tkfmixdom.jax.train.restricted_mstep import banded_3fc_init
        frag_weights, ext_rates = banded_3fc_init(
            n_dom, float(getattr(args, 'p_ext', 0.6)))
        log_fn(f"  banded-frag-init: frag_weights[d,0]=1, "
               f"p_ext={float(getattr(args, 'p_ext', 0.6)):.4f} "
               f"(FragStart/FragMid/FragEnd)")
    params = {
        'main_ins': float(main_ins), 'main_del': float(main_del),
        'dom_ins': np.array(dom_ins), 'dom_del': np.array(dom_del),
        'dom_weights': np.array(dom_weights),
        'frag_weights': np.array(frag_weights),
        'ext_rates': np.array(ext_rates),
    }

    pi_lg_np = np.asarray(pi_lg)
    Q_lg_np = np.asarray(Q_lg)

    if getattr(args, 'estimate_subst', False):
        rng_init = np.random.RandomState(args.seed + 1000)
        dom_pis_init = np.tile(pi_lg_np, (n_dom, 1))
        for dd in range(n_dom):
            noise = rng_init.dirichlet(np.ones(AA) * 100.0)
            dom_pis_init[dd] = 0.95 * dom_pis_init[dd] + 0.05 * noise
            dom_pis_init[dd] /= dom_pis_init[dd].sum()
        params['dom_Qs'] = np.tile(Q_lg_np[None, :, :], (n_dom, 1, 1))
        params['dom_pis'] = dom_pis_init
        params['dom_S_exch'] = np.tile(
            (Q_lg_np / np.maximum(pi_lg_np[None, :], 1e-30)
             * (1.0 - np.eye(AA)))[None, :, :],
            (n_dom, 1, 1))
        for dd in range(n_dom):
            _s = params['dom_S_exch'][dd]
            params['dom_S_exch'][dd] = (_s + _s.T) / 2

    # MixDom2 site classes: per-fragment class distribution
    n_classes = getattr(args, 'n_classes', 0)
    if n_classes == 0:
        n_classes = max(n_dom, n_frag)
    if n_classes > 1:
        # Per-class pi initialisation strategy
        class_pi_init = getattr(args, 'class_pi_init', 'lg_noisy')
        classdist_init = np.ones((n_dom, n_frag, n_classes)) / n_classes
        if class_pi_init in ('c10', 'C10'):
            from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c10
            profiles, weights_c10, _ = le_gascuel_c10()
            if n_classes != profiles.shape[0]:
                raise ValueError(
                    f"--class-pi-init=c10 requires --n-classes 10, "
                    f"got {n_classes}")
            class_pis = profiles.astype(float)
            # Seed classdist with the LG C10 mixture weights so the
            # initial class distribution reflects typical residue
            # usage (not uniform).
            w = weights_c10 / weights_c10.sum()
            classdist_init = np.broadcast_to(
                w[None, None, :], (n_dom, n_frag, n_classes)).copy()
            log_fn(f"  Initialising class_pis from LG C10 profiles "
                   f"(10 classes, IQ-TREE C10); "
                   f"classdist seeded with C10 weights")
        elif class_pi_init in ('c10_topN', 'C10_TOPN'):
            # Use the n_classes highest-weight LG C10 profiles as
            # warm-start class_pis. Useful when n_classes < 10 and
            # we want each class to start at a distinct, biologically
            # meaningful equilibrium distribution rather than
            # near-LG (which is a near-symmetric saddle).
            from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c10
            profiles, weights_c10, _ = le_gascuel_c10()
            if n_classes > profiles.shape[0]:
                raise ValueError(
                    f"--class-pi-init=c10_topN requires "
                    f"--n-classes <= 10, got {n_classes}")
            top_idx = np.argsort(weights_c10)[::-1][:n_classes]
            class_pis = profiles[top_idx].astype(float)
            w = weights_c10[top_idx] / weights_c10[top_idx].sum()
            classdist_init = np.broadcast_to(
                w[None, None, :], (n_dom, n_frag, n_classes)).copy()
            log_fn(f"  Initialising class_pis from top-{n_classes} "
                   f"LG C10 profiles (indices={top_idx.tolist()}); "
                   f"classdist seeded with their normalised weights "
                   f"(may be overridden by --classdist-init)")
        elif class_pi_init in ('c20', 'C20'):
            from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c20
            profiles, weights_c20, _ = le_gascuel_c20()
            if n_classes != profiles.shape[0]:
                raise ValueError(
                    f"--class-pi-init=c20 requires --n-classes 20, "
                    f"got {n_classes}")
            class_pis = profiles.astype(float)
            w = weights_c20 / weights_c20.sum()
            classdist_init = np.broadcast_to(
                w[None, None, :], (n_dom, n_frag, n_classes)).copy()
            log_fn(f"  Initialising class_pis from LG C20 profiles "
                   f"(20 classes, IQ-TREE C20); "
                   f"classdist seeded with C20 weights")
        elif class_pi_init in ('c20_plus_uniform_noisy',
                                 'C20_PLUS_UNIFORM_NOISY'):
            # First 20 classes: LG C20 profiles. Remaining classes:
            # near-uniform π with small Dirichlet noise (NOT
            # near-LG — the user wants these to start far from any
            # already-fit profile, so they have head-room to specialise
            # into structurally-distinct categories).
            from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c20
            profiles, weights_c20, _ = le_gascuel_c20()
            n_c20 = profiles.shape[0]
            if n_classes < n_c20:
                raise ValueError(
                    f"--class-pi-init=c20_plus_uniform_noisy requires "
                    f"--n-classes >= {n_c20}, got {n_classes}")
            n_extra = n_classes - n_c20
            class_pis = np.zeros((n_classes, AA))
            class_pis[:n_c20] = profiles.astype(float)
            # Extra classes: uniform 1/AA + Dirichlet noise.
            rng_cls = np.random.RandomState(args.seed + 3000)
            noise_frac = float(getattr(args, 'pi_init_noise_frac', 0.05))
            uniform = np.ones(AA) / AA
            for cc in range(n_extra):
                noise = rng_cls.dirichlet(np.ones(AA) * 50.0)
                class_pis[n_c20 + cc] = (
                    (1.0 - noise_frac) * uniform + noise_frac * noise)
                class_pis[n_c20 + cc] /= class_pis[n_c20 + cc].sum()
            # classdist: C20 weights for first 20 classes, small flat
            # weight for each extra class. Choose extra-class weight so
            # that total weight on the C20 portion is e.g. 80% (the
            # extras start as "minority"). Each extra gets
            # (1 - 0.8) / n_extra = 0.2 / n_extra mass.
            extra_total_mass = 0.2
            w20 = (weights_c20 / weights_c20.sum()) * (1.0 - extra_total_mass)
            w_extra = np.full(n_extra, extra_total_mass / max(n_extra, 1))
            w_full = np.concatenate([w20, w_extra])
            classdist_init = np.broadcast_to(
                w_full[None, None, :], (n_dom, n_frag, n_classes)).copy()
            log_fn(
                f"  Initialising class_pis: {n_c20} LG C20 profiles + "
                f"{n_extra} near-uniform-noisy classes "
                f"(noise_frac={noise_frac}); classdist {n_c20}/{n_classes} "
                f"C20-weighted (mass={1-extra_total_mass:.2f}), {n_extra}/{n_classes} "
                f"flat-weighted (mass={extra_total_mass:.2f}/{n_extra})")
        else:
            # Default: LG equilibrium + Dirichlet noise per class
            rng_cls = np.random.RandomState(args.seed + 2000)
            class_pis = np.tile(pi_lg_np, (n_classes, 1))
            noise_frac = float(getattr(args, 'pi_init_noise_frac', 0.2))
            for cc in range(n_classes):
                noise = rng_cls.dirichlet(np.ones(AA) * 50.0)
                class_pis[cc] = (1.0 - noise_frac) * class_pis[cc] + noise_frac * noise
                class_pis[cc] /= class_pis[cc].sum()
            log_fn(f"  class_pis init: lg_noisy with noise_frac={noise_frac}")
        # Per-class S_exch: each class gets its own exchangeability,
        # initialised to LG. With class_gamma removed, rate scale is
        # absorbed into S itself (GTR per class).
        S_lg = (Q_lg_np / np.maximum(pi_lg_np[None, :], 1e-30)
                * (1.0 - np.eye(AA)))
        S_lg = (S_lg + S_lg.T) / 2
        class_S_exch = np.tile(S_lg[None], (n_classes, 1, 1))

        # Optional override: class_pis[c] = dom_pis[c] for c < n_dom.
        # Used with --classdist-init=identity to make V1' emissions
        # bit-identical to d3f1 at iter 1 (the only intended delta
        # being the classdist off-diagonal pseudocounts).
        if getattr(args, 'class_pis_from_dom_pis', False):
            if n_classes != n_dom:
                raise ValueError(
                    f"--class-pis-from-dom-pis requires "
                    f"n_classes == n_dom (got n_classes={n_classes}, "
                    f"n_dom={n_dom})")
            if 'dom_pis' not in params:
                raise ValueError(
                    "--class-pis-from-dom-pis requires "
                    "--estimate-subst (so dom_pis is initialised)")
            class_pis = np.array(params['dom_pis']).copy()
            log_fn(f"  class_pis: mirrored from dom_pis "
                   f"(c=d for c=0..{n_dom-1})")

        # Optional override: classdist = identity (c = d).
        # Tie class c to domain c so that each domain's subst
        # profile is decoupled from every other. Combined with
        # --freeze-classdist this makes MixDom2 (with
        # n_classes = n_dom) exactly equivalent to MixDom1.
        classdist_init_mode = getattr(args, 'classdist_init', 'auto')
        if classdist_init_mode == 'identity':
            if n_classes != n_dom:
                raise ValueError(
                    f"--classdist-init=identity requires "
                    f"--n-classes == n_dom "
                    f"(got n_classes={n_classes}, n_dom={n_dom})")
            cd_id = np.zeros((n_dom, n_frag, n_classes))
            for d in range(n_dom):
                cd_id[d, :, d] = 1.0
            classdist_init = cd_id
            log_fn(f"  classdist init: identity "
                   f"(class c tied to domain c; {n_dom} classes)")
        elif classdist_init_mode == 'fragchar':
            if n_classes != n_frag:
                raise ValueError(
                    f"--classdist-init=fragchar requires "
                    f"--n-classes == n_frag "
                    f"(got n_classes={n_classes}, n_frag={n_frag})")
            cd_fc = np.zeros((n_dom, n_frag, n_classes))
            for f in range(n_frag):
                cd_fc[:, f, f] = 1.0
            classdist_init = cd_fc
            log_fn(f"  classdist init: fragchar "
                   f"(class c tied to fragment c; {n_frag} fragments)")

        # Optional symmetry-breaking noise on classdist init.
        # The default uniform `1/n_classes` is a saddle point under
        # the M-step gradient when class params are also identical:
        # see project debug log re: d3f1c3 stuck at -670.3 val LL
        # despite c=3 being a strict generalisation of c=1. A small
        # Dirichlet perturbation per (d, f) breaks the saddle so
        # classdist can specialise during training.
        cd_noise_frac = float(getattr(args, 'classdist_noise_frac', 0.0))
        if cd_noise_frac > 0.0:
            rng_cd_noise = np.random.RandomState(args.seed + 3000)
            for d in range(n_dom):
                for f in range(n_frag):
                    noise = rng_cd_noise.dirichlet(np.ones(n_classes) * 1.0)
                    classdist_init[d, f] = (
                        (1.0 - cd_noise_frac) * classdist_init[d, f]
                        + cd_noise_frac * noise)
                    classdist_init[d, f] /= classdist_init[d, f].sum()
            log_fn(f"  classdist init: perturbed with "
                   f"noise_frac={cd_noise_frac} "
                   f"(range across (d, f, c): "
                   f"[{classdist_init.min():.4f}, {classdist_init.max():.4f}])")

        params['n_classes'] = n_classes
        params['class_pis'] = class_pis           # (C, A)
        params['class_S_exch'] = class_S_exch     # (C, A, A) per-class
        params['classdist'] = classdist_init      # (N, F, C)
        log_fn(f"  Site classes: C={n_classes}, "
               f"classdist ({n_dom},{n_frag},{n_classes})")

    return params
