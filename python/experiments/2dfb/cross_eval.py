#!/usr/bin/env python3
"""Cross-evaluate the SVI-BW and Adam-tkf92 val LLs at each other's
converged parameters.

If the loss functions ARE the same, we expect:
  SVI-BW eval at SVI params:   -624.76 (matches reported)
  Adam   eval at SVI params:   ~-624.76 (within numerical precision)
  SVI-BW eval at Adam params:  ~-665.01
  Adam   eval at Adam params:  -665.01 (matches reported)

Symmetric matrix → loss agreement, Adam just stuck in wrong basin.
Asymmetric → losses differ in some way, smoothing may help.
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["TKFMIXDOM_MAX_PAD"] = "256"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax.numpy as jnp

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.train.tkf92_svi_bw import estep_pair_tkf92
from tkfmixdom.jax.train.tkf92_adam_fb import tkf92_log_prob_fb


def load_pairs_same_seed_as_run(precompiled_dir, max_aln_len, n_train, n_val,
                                  seed=0):
    """Reproduce the EXACT val pairs the runs used (same breadth-sample logic
    as run_tkf92_2dfb_pfam.load_breadth_first_pairs).
    """
    from train_pfam import PrecompiledPairSource
    src = PrecompiledPairSource(precompiled_dir, max_alignment_len=max_aln_len)
    print(f"PrecompiledPairSource: {src.n_pairs:,} pairs, "
          f"{src.n_families:,} families", flush=True)
    all_decoded = src._decode_all()

    from collections import defaultdict
    by_family = defaultdict(list)
    for item in all_decoded:
        x_int, y_int, _states, _ac, _dc, t_est, fam = item
        by_family[fam].append((x_int, y_int, float(t_est)))

    families = list(by_family.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(families)

    n_val_fam = max(int(round(len(families) * 0.05)),
                    int(round(n_val * 1.5)))
    val_families = list(set(families[:n_val_fam]))  # to match disjoint logic
    val_families = families[:n_val_fam]

    def breadth_sample(families_in_order, by_fam, target_n):
        per_family_q = {f: list(by_fam[f]) for f in families_in_order}
        for f in per_family_q:
            rng.shuffle(per_family_q[f])
        out = []
        any_left = True
        while any_left and len(out) < target_n:
            any_left = False
            for f in families_in_order:
                if not per_family_q[f]:
                    continue
                out.append(per_family_q[f].pop())
                any_left = True
                if len(out) >= target_n:
                    break
        return out

    train_pairs = breadth_sample(families[n_val_fam:], by_family, n_train)
    val_pairs = breadth_sample(val_families, by_family, n_val)
    return val_pairs


def adam_loss_at(params_tuple, val_pairs, Q, pi):
    """Adam-tkf92's per-pair joint LL evaluator at fixed (lam, mu, ext)."""
    lam, mu, ext = params_tuple
    Qj, pij = jnp.asarray(Q), jnp.asarray(pi)
    total = 0.0
    for x, y, t in val_pairs:
        log_p = tkf92_log_prob_fb(
            jnp.asarray(lam), jnp.asarray(mu), jnp.asarray(ext),
            jnp.asarray(t), Qj, pij, jnp.asarray(x), jnp.asarray(y))
        total += float(log_p)
    return total / len(val_pairs)


def svi_loss_at(params_tuple, val_pairs, Q, pi):
    """SVI-BW's per-pair joint LL evaluator at fixed (lam, mu, ext)."""
    lam, mu, ext = params_tuple
    pi_np = np.asarray(pi)
    total = 0.0
    for x, y, t in val_pairs:
        r = estep_pair_tkf92(x, y, t, lam, mu, ext, np.asarray(Q), pi_np)
        total += r["log_p"]
    return total / len(val_pairs)


def main():
    svi = json.load(open("experiments/2dfb/svi_bw.json"))
    adam = json.load(open("experiments/2dfb/adam_tkf92.json"))
    p_svi = (svi["best_lam"], svi["best_mu"], svi["best_ext"])
    # Adam stores best_params in (log_mu, logit_kappa, logit_ext) basis
    from tkfmixdom.jax.train.tkf92_adam_fb import unpack_tkf92
    lam, mu, ext = unpack_tkf92(
        [jnp.asarray(p) for p in adam["best_params"]])
    p_adam = (float(lam), float(mu), float(ext))
    print(f"SVI-BW  params: lam={p_svi[0]:.5f} mu={p_svi[1]:.5f} "
          f"ext={p_svi[2]:.4f} kappa={p_svi[0]/p_svi[1]:.4f}")
    print(f"Adam    params: lam={p_adam[0]:.5f} mu={p_adam[1]:.5f} "
          f"ext={p_adam[2]:.4f} kappa={p_adam[0]/p_adam[1]:.4f}")
    print(f"SVI-BW reported val_ll/pair: {svi['best_val_ll_per_pair']:.4f}")
    print(f"Adam reported val_ll/pair:   {adam['best_val_ll_per_pair']:.4f}")
    print()
    Q, pi = rate_matrix_lg()

    print("Loading val pairs (n=50, same seed=0) ...")
    val_pairs = load_pairs_same_seed_as_run(
        "pfam/precompiled", 256, 20000, 50, seed=0)
    print(f"  loaded {len(val_pairs)} val pairs")
    print()

    print("Cross-evaluation matrix (val_ll/pair):")
    print(f"{'':22s}  {'SVI eval':>12s}  {'Adam eval':>12s}")
    t0 = time.time()
    svi_at_svi = svi_loss_at(p_svi, val_pairs, Q, pi)
    print(f"  At SVI-BW params:       {svi_at_svi:12.4f}", end='', flush=True)
    adam_at_svi = adam_loss_at(p_svi, val_pairs, Q, pi)
    print(f"  {adam_at_svi:12.4f}  ({time.time()-t0:.1f}s)")
    t0 = time.time()
    svi_at_adam = svi_loss_at(p_adam, val_pairs, Q, pi)
    print(f"  At Adam params:         {svi_at_adam:12.4f}", end='', flush=True)
    adam_at_adam = adam_loss_at(p_adam, val_pairs, Q, pi)
    print(f"  {adam_at_adam:12.4f}  ({time.time()-t0:.1f}s)")
    print()

    # Verdict
    delta_at_svi = svi_at_svi - adam_at_svi
    delta_at_adam = svi_at_adam - adam_at_adam
    print(f"Δ (SVI - Adam) at SVI params:  {delta_at_svi:+.4f}")
    print(f"Δ (SVI - Adam) at Adam params: {delta_at_adam:+.4f}")
    if abs(delta_at_svi) < 0.01 and abs(delta_at_adam) < 0.01:
        print("\n  → Losses AGREE. Adam is just stuck in a different basin.")
        print("    Try warm-start Adam at SVI params, OR longer/higher-lr Adam.")
    else:
        print("\n  → Losses DISAGREE. Different objectives at these params.")
        print("    Investigate the L'Hôpital branch boundary or numerical paths.")

    json.dump({
        "p_svi": list(p_svi), "p_adam": list(p_adam),
        "svi_at_svi": svi_at_svi, "adam_at_svi": adam_at_svi,
        "svi_at_adam": svi_at_adam, "adam_at_adam": adam_at_adam,
        "delta_at_svi": delta_at_svi, "delta_at_adam": delta_at_adam,
    }, open("experiments/2dfb/cross_eval.json", "w"), indent=2)


if __name__ == "__main__":
    main()
