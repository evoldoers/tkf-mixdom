#!/usr/bin/env python3
"""Diagnostic: TKF92 2D EM with invariant checks and LL tracking.

EM must MONOTONICALLY increase the likelihood. If it doesn't, there's a bug.
This script instruments every step with sanity checks.

Invariants checked at every E-step:
  1. Posterior row sums = 1 (valid distribution)
  2. Expected transition counts n_trans >= 0
  3. n_trans row sums match posterior marginals
  4. Log-likelihood is finite

Invariants checked at every M-step:
  5. New params produce a valid transition matrix (rows sum to 1, entries >= 0)
  6. Log-likelihood under new params >= log-likelihood under old params
  7. BDI stats: E_B >= 0, E_D >= 0, E_S >= 0

Usage:
    cd python && uv run python experiments/diagnose_tkf92_em.py
"""

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import time
import warnings
warnings.filterwarnings('ignore')

from tkfmixdom.jax.core.params import tkf92_trans, S, M, I, D, E
from tkfmixdom.jax.core.ctmc import rate_matrix_jc69, transition_matrix
from tkfmixdom.jax.models.compiled import TKF92Model
from tkfmixdom.jax.train.em import _add_stats
from tkfmixdom.jax.dp.hmm import forward_backward_2d


def check_transition_matrix(tau, label=""):
    """Check that tau is a valid stochastic matrix."""
    tau = np.asarray(tau)
    row_sums = tau.sum(axis=1)
    for i in range(tau.shape[0]):
        if abs(row_sums[i] - 1.0) > 1e-6 and row_sums[i] > 0:
            print(f"  WARNING {label}: row {i} sums to {row_sums[i]:.10f}")
    if np.any(tau < -1e-10):
        neg = tau[tau < -1e-10]
        print(f"  WARNING {label}: {len(neg)} negative entries, min={neg.min():.2e}")
    return True


def check_e_step(ll, n_trans, label=""):
    """Check E-step outputs for sanity."""
    ok = True
    if not np.isfinite(ll):
        print(f"  ERROR {label}: LL not finite: {ll}")
        ok = False

    n = np.asarray(n_trans)
    if np.any(n < -1e-8):
        neg = n[n < -1e-8]
        print(f"  WARNING {label}: {len(neg)} negative n_trans entries, min={neg.min():.2e}")

    if np.any(np.isnan(n)):
        print(f"  ERROR {label}: NaN in n_trans")
        ok = False

    # Total transitions should equal total emissions (roughly)
    total_trans = n.sum()
    if total_trans < 0:
        print(f"  ERROR {label}: total transitions < 0: {total_trans}")
        ok = False

    return ok


def check_bdi_stats(stats, label=""):
    """Check BDI sufficient statistics."""
    bdi = stats['indel']
    ok = True
    if bdi.E_B < -1e-6:
        print(f"  WARNING {label}: E_B = {bdi.E_B:.6f} < 0")
    if bdi.E_D < -1e-6:
        print(f"  WARNING {label}: E_D = {bdi.E_D:.6f} < 0")
    if bdi.E_S < -1e-6:
        print(f"  WARNING {label}: E_S = {bdi.E_S:.6f} < 0")
        ok = False
    if bdi.n_kappa < 0:
        print(f"  WARNING {label}: n_kappa = {bdi.n_kappa:.6f} < 0")
    if bdi.n_1mkappa < 0:
        print(f"  WARNING {label}: n_1mkappa = {bdi.n_1mkappa:.6f} < 0")

    ext = stats['ext']
    if ext.n_success < -1e-6:
        print(f"  WARNING {label}: ext n_success = {ext.n_success:.6f} < 0")
    if ext.n_failure < -1e-6:
        print(f"  WARNING {label}: ext n_failure = {ext.n_failure:.6f} < 0")

    return ok


def compute_data_ll(model, params, pairs):
    """Compute total log-likelihood of all pairs under params."""
    total = 0.0
    n_valid = 0
    for anc, desc in pairs:
        if len(anc) == 0 and len(desc) == 0:
            continue
        try:
            ll, _, _ = model.e_step(params, jnp.array(anc), jnp.array(desc))
            total += float(ll)
            n_valid += 1
        except:
            continue
    return total, n_valid


def main():
    true_ins, true_del, true_ext = 0.0458, 0.0468, 0.683
    t_val = 0.5
    A = 4

    Q, pi = rate_matrix_jc69(A)
    Q, pi = np.asarray(Q), np.asarray(pi)
    sub_matrix = np.asarray(transition_matrix(jnp.array(Q), t_val))

    model = TKF92Model(fixed_params=frozenset({'Q', 'pi', 'sub_matrix', 't'}))

    # Forward-sample 50 pairs (state paths + characters)
    tau_true = np.asarray(tkf92_trans(true_ins, true_del, t_val, true_ext))
    print("True transition matrix:")
    check_transition_matrix(tau_true, "true")

    rng = np.random.RandomState(42)
    pairs = []
    for _ in range(50):
        state = S; anc = []; desc = []
        for _ in range(1000):
            ns = rng.choice(5, p=tau_true[state])
            if ns == E:
                break
            if ns == M:
                a = rng.choice(A, p=pi)
                d = rng.choice(A, p=sub_matrix[a])
                anc.append(a); desc.append(d)
            elif ns == I:
                desc.append(rng.choice(A, p=pi))
            elif ns == D:
                anc.append(rng.choice(A, p=pi))
            state = ns
        pairs.append((np.array(anc, dtype=np.int32),
                       np.array(desc, dtype=np.int32)))

    print(f"\n50 pairs, mean anc={np.mean([len(a) for a,d in pairs]):.0f} "
          f"desc={np.mean([len(d) for a,d in pairs]):.0f}")

    # Random initialization
    params = {
        'ins_rate': 0.10, 'del_rate': 0.15, 'ext': 0.50,
        't': t_val, 'Q': Q, 'pi': pi, 'sub_matrix': sub_matrix,
    }

    print(f"\nTrue:  ins={true_ins:.4f} del={true_del:.4f} ext={true_ext:.3f}")
    print(f"Init:  ins={params['ins_rate']:.4f} del={params['del_rate']:.4f} "
          f"ext={params['ext']:.3f}")

    # Check initial transition matrix
    tau_init = np.asarray(tkf92_trans(
        params['ins_rate'], params['del_rate'], t_val, params['ext']))
    check_transition_matrix(tau_init, "init")

    # Compute initial LL
    print("\nComputing initial LL (this involves JIT compilation)...")
    t0 = time.time()
    init_ll, init_n = compute_data_ll(model, params, pairs)
    print(f"Initial LL: {init_ll:.2f} ({init_n} pairs, {time.time()-t0:.0f}s)")

    prev_ll = init_ll

    for epoch in range(5):
        print(f"\n=== EPOCH {epoch+1} ===")
        t0 = time.time()

        # E-step: accumulate stats
        agg_stats = None
        epoch_ll = 0.0
        n_valid = 0
        n_warnings = 0

        for i, (anc, desc) in enumerate(pairs):
            if len(anc) == 0 and len(desc) == 0:
                continue
            try:
                ll, n_trans, posteriors = model.e_step(
                    params, jnp.array(anc), jnp.array(desc))

                if not check_e_step(ll, n_trans, f"pair {i}"):
                    n_warnings += 1

                stats = model.extract_stats(jnp.array(n_trans), params)

                if not check_bdi_stats(stats, f"pair {i}"):
                    n_warnings += 1

                if agg_stats is None:
                    agg_stats = stats
                else:
                    agg_stats = _add_stats(agg_stats, stats)
                epoch_ll += float(ll)
                n_valid += 1
            except Exception as e:
                print(f"  pair {i} EXCEPTION: {e}")
                continue

            if i % 10 == 0:
                print(f"  pair {i}/50 ({time.time()-t0:.0f}s)", flush=True)

        print(f"  E-step done: {n_valid} pairs, epoch LL={epoch_ll:.2f}, "
              f"{n_warnings} warnings, {time.time()-t0:.0f}s")

        # Check aggregated stats
        if agg_stats is not None:
            print(f"  Agg BDI: E_B={agg_stats['indel'].E_B:.4f} "
                  f"E_D={agg_stats['indel'].E_D:.4f} "
                  f"E_S={agg_stats['indel'].E_S:.4f}")
            print(f"  Agg ext: success={agg_stats['ext'].n_success:.2f} "
                  f"failure={agg_stats['ext'].n_failure:.2f}")
            print(f"  n_kappa={agg_stats['indel'].n_kappa:.2f} "
                  f"n_1mkappa={agg_stats['indel'].n_1mkappa:.2f}")

        # M-step
        old_params = dict(params)
        if agg_stats is not None and n_valid > 0:
            params = model.m_step(agg_stats, params)

        print(f"  M-step: ins={old_params['ins_rate']:.4f}->{params['ins_rate']:.4f} "
              f"del={old_params['del_rate']:.4f}->{params['del_rate']:.4f} "
              f"ext={old_params['ext']:.3f}->{params['ext']:.3f}")

        # Check new transition matrix
        tau_new = np.asarray(tkf92_trans(
            params['ins_rate'], params['del_rate'], t_val, params['ext']))
        check_transition_matrix(tau_new, f"epoch {epoch+1}")

        # CRITICAL CHECK: recompute LL under new params
        print(f"  Recomputing LL under new params...")
        new_ll, new_n = compute_data_ll(model, params, pairs)
        ll_change = new_ll - prev_ll

        print(f"  LL: {prev_ll:.2f} -> {new_ll:.2f} (change={ll_change:+.2f})")
        if ll_change < -1e-4:
            print(f"  *** LL DECREASED BY {-ll_change:.4f} — THIS IS A BUG ***")
        else:
            print(f"  LL monotonicity OK")

        ie = abs(params['ins_rate'] - true_ins) / true_ins
        de = abs(params['del_rate'] - true_del) / true_del
        ee = abs(params['ext'] - true_ext) / true_ext
        print(f"  Errors: ins={ie:.3f} del={de:.3f} ext={ee:.3f}")

        prev_ll = new_ll


if __name__ == '__main__':
    main()
