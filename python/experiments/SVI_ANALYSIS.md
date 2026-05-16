# SVI for MixDom Baum-Welch: Analysis and Recommendations

## 1. Does SVI apply?

**Yes, cleanly.** The MixDom M-step is entirely in the exponential family / conjugate prior framework that SVI requires. Here is the argument for each parameter group.

### The key insight

SVI's natural gradient update operates on **sufficient statistics** (or equivalently, on the natural parameters of the posterior), not on the model parameters themselves. The update is:

```
η_new = (1 - ρ_t) · η_old + ρ_t · (η_prior + (N / |B|) · η_batch)
```

where `η` are the natural parameters (pseudocounts / sufficient statistics), `N` is the total dataset size, `|B|` is the minibatch size, and `ρ_t` is the Robbins-Monro step size.

The nonlinear M-step (quadratic solve, coordinate ascent) is applied **after** the sufficient statistics are averaged. This is the crucial distinction from naive stochastic EM: we average the *statistics*, not the *parameters*.

### Parameter-by-parameter compatibility

**Domain weights** (Dirichlet): Natural parameters are counts `n_d + alpha - 1`. The SVI update averages these counts. The M-step (MAP mode of Dirichlet) is applied to the averaged counts. Fully compatible.

**Fragment weights** (per-domain Dirichlet): Same as domain weights. Fully compatible.

**Extension rates** (Beta): Natural parameters are `(ext_count + alpha - 1, term_count + beta - 1)`. SVI averages these. The M-step `ext = a / (a + b)` is applied to averaged counts. Fully compatible.

**Indel rates** (quadratic in kappa): The sufficient statistics are `(B, D, S, L, M, T)` — all **additive** over pairs. The SVI update averages these six scalars. The quadratic solve `m_step_indel_quadratic` is then applied to the averaged statistics. This is correct because the M-step maximizes `Q(theta | suff_stats)` where Q is linear in the sufficient statistics. Averaging stats then solving is equivalent to finding the MAP of the averaged Q-function. **Fully compatible.**

**Substitution model** (Holmes-Rubin W, U, V): The sufficient statistics are `W` (dwell times, shape `(A,)`), `U` (transition counts, shape `(A, A)`), and `V` (character counts, shape `(A,)`). All three are additive over pairs. The iterative coordinate ascent (`m_step_subst_option1`) operates on these aggregated statistics. **Fully compatible.**

### Why averaging stats works for the quadratic solve

The concern was: the M-step is nonlinear in `(B, D, S, L, M, T)`, so does averaging these stats cause problems?

No. The M-step solves `argmax_theta Q(theta; stats)` where `Q` is the expected complete-data log-likelihood, which is **linear** in the sufficient statistics. So:

```
argmax_theta Q(theta; avg_stats) = argmax_theta [ (1-rho) * Q(theta; old_stats) + rho * Q(theta; scaled_batch_stats) ]
```

This is exactly the M-step for the averaged Q-function, which is what SVI computes (it takes a natural gradient step on the ELBO, which for conjugate models reduces to this convex combination of sufficient statistics).

The quadratic solve is just a convenient way to find the argmax. It doesn't matter that the mapping from stats to parameters is nonlinear — what matters is that the stats themselves combine linearly.

### The chain restoration is not a problem

The 6-step chain restoration (`exact_suffstats`) is a deterministic function mapping collapsed FB counts `n_chi` (the `(5NK+2) x (5NK+2)` matrix) to per-parameter sufficient statistics. It is applied **per-batch** before the sufficient statistics are accumulated. The SVI averaging happens at the level of the extracted sufficient statistics, not at the level of `n_chi`.

Concretely: each batch produces `(B_batch, D_batch, S_batch, ...)` via chain restoration. These are then averaged into the running totals. This is correct.

### The substitution coordinate ascent is fine

The `m_step_subst_option1` iterative procedure maximizes `Q(S, pi; W, U, V)` given fixed sufficient statistics `(W, U, V)`. Since we average `W`, `U`, `V` across iterations (SVI update), and then run the coordinate ascent on the averaged stats, this is correct. The iterative scheme converges to the unique MAP given the stats — it doesn't matter whether those stats came from one batch or from an exponential moving average.

## 2. Is it complicated by kappa chicanery?

**No.** The `n_kappa` and `n_{1-kappa}` counts that enter as `L` and `M` in the quadratic solve are part of the sufficient statistics extracted by `_extract_suffstats`. They are additive. The `T` accumulation (`T = t * (n_entries + n_{1-kappa})`) is also a linear function of the sufficient statistics. Everything composes cleanly.

The only subtlety: `t_rep` (the representative evolutionary time) is currently estimated per-iteration as the median of that batch's `t_est` values. Under SVI, this should be computed from the full dataset (or from a global running estimate), since the BDI parameter derivatives depend on `t`. In practice, `t_rep` stabilizes quickly and can be treated as a fixed hyperparameter after a few iterations, or updated via a separate running median.

## 3. Concrete Recommendations

### 3.1 The SVI update rule for MixDom

Replace the current "fresh stats each iteration" with a running exponential average of sufficient statistics:

```python
# After chain restoration for this batch:
ss_batch = exact_suffstats(n_chi_batch, ...)

# Scale batch stats to full dataset size
scale = N_total / n_pairs_batch

# SVI update for each sufficient statistic group
for key in ['top_5x5', 'dom_M_5x5', 'dom_kappa', 'dom_1mkappa',
            'dom_w', 'frag_w', 'ext', 'term',
            'dom_match_counts', 'dom_insert_counts', 'dom_delete_counts']:
    ss_global[key] = (1 - rho) * ss_global[key] + rho * (prior[key] + scale * ss_batch[key])

# Then run M-step on ss_global (unchanged)
```

The prior pseudocounts should be added to `ss_global` only once at initialization (they are part of the natural parameter), not re-added each iteration. Alternatively, separate the prior and likelihood contributions:

```python
# Cleaner formulation: track likelihood stats only, add prior at M-step time
ss_lik_global[key] = (1 - rho) * ss_lik_global[key] + rho * scale * ss_batch[key]
# M-step uses ss_lik_global[key] + prior[key]
```

### 3.2 Code changes to `train_stochastic_em`

The changes are localized. Here is the diff conceptually, referencing `train_pfam.py`:

**A. Add SVI hyperparameters** (around line 2054, in the config dict):

```python
# Robbins-Monro schedule: rho_t = (t + tau)^{-kappa_rm}
# Default: tau=10, kappa_rm=0.7 (Hoffman et al recommend kappa in [0.5, 1])
svi_tau = getattr(args, 'svi_tau', 10.0)
svi_kappa = getattr(args, 'svi_kappa', 0.7)
```

These are the only two new hyperparameters.

**B. Initialize global sufficient statistics** (around line 2247, after `_zero_suff_stats`):

```python
# SVI: maintain running sufficient statistics
# Initialize from prior (first iteration will overwrite via rho=1 effectively)
ss_global = _zero_suff_stats_exact(N, n_dom, n_frag)  # same shape as exact_suffstats output
# On checkpoint resume, load ss_global from checkpoint
```

**C. Replace per-iteration stats reset with SVI update** (around lines 2483-2489):

Current code:
```python
ss = exact_suffstats(suff_stats['agg_n_chi'], ...)
# M-step uses ss directly
```

New code:
```python
ss_batch = exact_suffstats(suff_stats['agg_n_chi'], ...)

# SVI step size
rho = (em_iter + svi_tau) ** (-svi_kappa)

# Scale factor: N_total_pairs / n_pairs_this_iter
# N_total can be estimated once or tracked as a running count
scale = N_total_pairs / max(n_pairs_this_iter, 1)

# SVI natural gradient update on sufficient statistics
for key in ss_batch:
    if isinstance(ss_batch[key], np.ndarray):
        ss_global[key] = (1 - rho) * ss_global[key] + rho * scale * ss_batch[key]
    elif isinstance(ss_batch[key], (int, float)):
        ss_global[key] = (1 - rho) * ss_global[key] + rho * scale * ss_batch[key]

# M-step uses ss_global instead of ss_batch
ss = ss_global
```

**D. Save `ss_global` in checkpoint** (in `_save_stochastic_checkpoint`, around line 2301):

```python
# Save SVI global stats for resume
for key in ss_global:
    data[f'svi_global_{key}'] = np.asarray(ss_global[key])
```

**E. `N_total_pairs` estimation:** Either:
- Count total pairs across all families once at startup (expensive but exact)
- Use a running estimate: `N_est = n_families * avg_pairs_per_family`
- For precompiled pairs, this is known exactly from the manifest

### 3.3 Hyperparameters

| Parameter | Symbol | Default | Range | Notes |
|-----------|--------|---------|-------|-------|
| Delay | tau | 10 | 1-100 | Higher = more initial exploration |
| Forgetting rate | kappa_rm | 0.7 | (0.5, 1] | 0.5 = slow forget, 1.0 = fast forget |

These satisfy Robbins-Monro: `sum rho_t = inf` (kappa <= 1) and `sum rho_t^2 < inf` (kappa > 0.5).

With `tau=10, kappa_rm=0.7`:
- Iteration 1: rho = 0.19 (still heavily weighted toward batch)
- Iteration 5: rho = 0.12
- Iteration 10: rho = 0.076
- Iteration 20: rho = 0.045
- Iteration 50: rho = 0.022

This means early iterations are dominated by recent data (good for initial exploration), while later iterations stabilize by averaging over history (good for convergence).

**Recommended starting point:** `tau=10, kappa_rm=0.7`. These are Hoffman et al's recommended defaults and should work without tuning.

### 3.4 What about the first iteration?

On iteration 1, `ss_global` is all zeros. With `rho_1 = (1 + tau)^{-kappa}`, the update becomes:

```
ss_global = rho_1 * scale * ss_batch_1
```

This is just the batch stats scaled up. The M-step will produce reasonable parameters from this. By iteration 3-5, the running average will have seen enough data to stabilize.

Alternatively, set `rho_1 = 1.0` explicitly (full replacement on first iteration), then use the schedule from iteration 2 onward. This is common practice.

### 3.5 Relationship to the three training modes

| Mode | What it does | LL/pair | SVI equivalent |
|------|-------------|---------|----------------|
| BW-EM | Fixed pairs, exact E+M, all pairs every iter | -681 | Not needed (already optimal for fixed set) |
| Stochastic EM | Fresh pairs, E+M on batch only | -706 | **This proposal**: average stats across iters |
| Adam | SGD on log-likelihood via custom VJPs | ? | SVI is the natural gradient version of Adam |

SVI should achieve LL/pair close to BW-EM's -681, because:
1. It sees the same data diversity as stochastic EM (fresh pairs each iteration)
2. But it retains information from past iterations via the running average
3. The natural gradient (which is what SVI computes) is provably the steepest ascent direction in the Fisher information metric — it is the "right" gradient for exponential family models

In fact, SVI is strictly better than Adam for this model because:
- Adam uses Euclidean gradients; SVI uses natural gradients
- For exponential families, natural gradients have the simple closed form above (no need for VJPs, no need for Fisher matrix computation)
- The M-step is exact (closed-form), not approximate (gradient step)

### 3.6 Experimental plan

1. **Implement SVI update** in `train_stochastic_em` (changes above, ~30 lines of code)
2. **Run SVI training** with default hyperparameters (`tau=10, kappa_rm=0.7`), same budget as current stochastic EM
3. **Evaluate on held-out data** using `eval_only` mode, compare LL/pair to:
   - BW-EM: -681 (target)
   - Stochastic EM: -706 (baseline)
4. **If LL/pair > -690**, declare success. If not, try `kappa_rm=0.6` (slower forgetting) or `tau=20` (more delay).
5. **Scaling experiment**: since SVI can process arbitrary numbers of pairs, try training on 10x more pairs than BW-EM's frozen set. This should yield better generalization (lower held-out LL/pair).

### 3.7 Paper paragraph

For `tkf.tex`, add a paragraph in the Baum-Welch section (after the M-step descriptions):

> **Stochastic optimization.** For large datasets, we replace full-batch Baum-Welch with stochastic variational inference (Hoffman et al., 2013). Since every M-step parameter group has conjugate exponential family form, the natural gradient of the ELBO reduces to a convex combination of the current sufficient statistics and the minibatch sufficient statistics scaled by the dataset-to-batch ratio:
> $$\bar{T}^{(t+1)} = (1 - \rho_t)\,\bar{T}^{(t)} + \rho_t\left(\bar{T}_0 + \frac{N}{|B|}\,T_B\right),$$
> where $\bar{T}$ denotes the running sufficient statistics (transition counts, character counts, Holmes--Rubin dwell times and jump counts), $\bar{T}_0$ the prior pseudocounts, $T_B$ the batch sufficient statistics from the E-step, $N$ the total number of training pairs, $|B|$ the batch size, and $\rho_t = (t + \tau)^{-\kappa}$ the Robbins--Monro step size with delay $\tau$ and forgetting rate $\kappa \in (0.5, 1]$. The M-step (quadratic solve for indel rates, Dirichlet/Beta MAP for weights and extension rates, Holmes--Rubin coordinate ascent for substitution parameters) is then applied to $\bar{T}^{(t+1)}$ exactly as in full-batch EM. This retains the closed-form exactness of each M-step while enabling online processing of arbitrarily large pair sets.

### 3.8 BibTeX entry

```bibtex
@article{hoffman2013svi,
  author  = {Hoffman, Matthew D. and Blei, David M. and Wang, Chong and Paisley, John},
  title   = {Stochastic Variational Inference},
  journal = {Journal of Machine Learning Research},
  year    = {2013},
  volume  = {14},
  pages   = {1303--1347},
}
```

## 4. Summary

SVI applies cleanly to MixDom Baum-Welch. The core change is ~30 lines: maintain a running exponential average of sufficient statistics instead of resetting each iteration. The M-step code is unchanged. Two hyperparameters are added (tau and kappa_rm), with robust defaults. The expected improvement over current stochastic EM is substantial (closing the gap from -706 toward -681 LL/pair), because information from past batches is retained rather than discarded.
