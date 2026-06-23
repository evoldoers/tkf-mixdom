# 2D-FB Pfam benchmark

Joint log-likelihood of unaligned Pfam cherry pairs under different
3-parameter TKF92 / GGI-steered TKF92 models, evaluated via the 2D
pair-HMM forward-backward DP.  This benchmark exists to characterise
how well alternative TKF92 reparameterisations and ancestor-prior
choices fit Pfam, holding the descendant-conditional DP machinery
fixed.

## Data

* **Source.**  Pfam protein cherries decoded by
  `PrecompiledPairSource` from `pfam/precompiled/`
  (1,137,844 cherry pairs across 20,864 families, max alignment length
  256).
* **Train.**  20,000 cherries, breadth-first sampled (one per family
  before any second), shuffled by `--seed` so different seeds get
  different train subsamples.
* **Val.**  500 cherries from a held-out *canonical* set of families:
  alphabetically sorted family list → first
  `max(round(5%·|families|), round(1.5·n_val))` families →
  breadth-sampled with a fixed canonical RNG seed
  `CANONICAL_VAL_SEED = 0xCAFE`.
  **The val set is identical across every method, every seed, every
  process** — this is enforced by sorting families before any
  rng.shuffle (so the val/train split is independent of the user
  seed) and using a separate RNG for within-val sampling
  (independent of `--seed`).
  Pinned in `run_tkf92_2dfb_pfam.load_breadth_first_pairs`
  (commit `e9f7483c1`).

## Objective

For each pair (anc, des, t) we compute the joint log-probability
`log P(anc, des | model, t)` via the 2D pair-HMM forward-backward
(`tkfmixdom.jax.dp.hmm.forward_backward_2d`) with the LG amino-acid
substitution model.  No alignment is fixed.  Padded positions are
masked via `real_Lx, real_Ly` so they contribute nothing to the
likelihood — early Adam runs that omitted this paid ~60 nats/pair of
spurious cost from padding emissions (fixed in `369e39c98`).

The minibatch loss is `−Σ log P(anc, des)` summed over `batch_size=16`
pairs.  Held-out val LL/pair is reported every 5 iters and tracked
for early-stop with patience 100.

## Methods compared

All are 3-parameter models (TKF92 has λ, μ, ext; GGI's 4 parameters
λ₀, μ₀, x, y are reduced to 3 by the reversibility constraint
`y(1−y) = x(1−x)/ρ` with `ρ = λ₀/μ₀`).

| Method | Optimizer | What's optimized |
|---|---|---|
| `svi_bw` | Stochastic-VB Baum-Welch (analytic Holmes-Rubin gradient + closed-form M-step on EMA suff stats) | TKF92 joint log P |
| `adam_tkf92` | Adam through `value_and_grad` of 2D F-B | TKF92 joint log P, constant (λ, μ, ext) |
| `adam_ggi` (lower) | Adam | GGI-steered TKF92: per-pair (λ_t, μ_t, r_t) from the GGI flow; GGI-native geometric ancestor prior; x ∈ (0, x_min) |
| `adam_ggi` (upper) | Adam | Same as lower but x ∈ (1−x_min, 1) — required for r_boundary > ~0.4 |
| `adam_ggi --ggi-no-prior-swap` (upper) | Adam | GGI dynamics with the *TKF92* ancestor prior — isolates dynamics from prior |

## Padding fix (`369e39c98`)

`tkf92_log_prob_fb` and `ggi_steered_log_prob_fb` now thread
`real_Lx, real_Ly` into `forward_backward_2d`, masking padded
positions to NEG_INF.  Pre-fix Adam was minimising
`log P(anc_padded, des_padded)` with padded zeros treated as
residue 0; the loss was inflated by ~60 nats/pair from padding
contribution, and the converged optima were biased.

## Knobs (defaults shared by every run unless noted)

```
--max-aln-len 256
--n-train 20000  --n-val 500
--batch-size 16  --lr 0.01
--n-iter 800     --patience 100   --val-every 5
--Q lg
--bin-bucketed   --pre-warm
--no-command-buffers  --max-pad-cap 256
```

`--bin-bucketed` pre-buckets pairs by `(Lx_pad, Ly_pad)` and each
minibatch is drawn from a single bucket — drastically reduces unique
JIT shapes and the resulting CUBIN memory.  `--max-pad-cap 256`
forces 32-multiple padding capped at 256 (≤8 bins/axis).  Without
these, the 11 GiB GPU OOMs on cumulative compiled CUBINs at ~5k pairs.

## Current results

All val LL/pair on the canonical 500-pair and 1500-pair canonical
vals.  Means over seeds where available (σ in parentheses).

### Canonical 1500-pair val (recommended)

| Rank | Method | best val_ll/pair | (λ, μ, ext) or (λ₀, μ₀, x, y) |
|---|---|---|---|
| 1 | Adam-GGI upper, no-prior-swap (2 seeds) | **−614.309** (σ < 0.001) | λ₀=0.014, μ₀=0.014, x≈0.70, y≈0.69 |
| 2 | SVI-BW canonical (3 seeds) | −614.343 (σ = 0.005) | λ=0.035, μ=0.036, ext≈0.58 |
| 3 | Adam-tkf92 (3 seeds) | −614.400 (σ = 0.012) | λ=0.055, μ=0.057, ext=0.75 |
| 4 | Aligned Pfam TKF92(K=1) (CherryML/Maraschino, alignment-given fit) | **−614.385** | λ=0.030, μ=0.030, ext=0.65 |
| 5 | Adam-GGI upper native, long2400 (2 seeds) | −614.642 | λ₀=0.014, μ₀=0.014, x≈0.78, y≈0.78 |
| 6 | Adam-GGI upper native, n800 (3 seeds) | −615.103 (σ = 0.016) | λ₀=0.013, μ₀=0.013, x≈0.81, y≈0.81 |
| 7 | Adam-GGI lower native | −616.405 | λ₀=0.018, μ₀=0.019, x=0.30, y=0.33 |

### Canonical 500-pair val (faster)

| Method | best val_ll/pair |
|---|---|
| Adam-GGI upper, no-prior-swap (seed=0) | **−599.64** |
| SVI-BW canonical (3 seeds avg) | −599.67 |
| Adam-tkf92 (3 seeds avg) | −599.73 |
| Aligned Pfam TKF92(K=1) | −599.74 |
| Adam-GGI upper, prior-swap (3 seeds, n_iter=800) | −600.40 |
| Adam-GGI lower, prior-swap (seed=0) | −601.82 |

(SVI-BW's 50-pair val_ll reported in its training log — `−624.76`
— is on the pre-canonical val set; not directly comparable.)

### Comparison to the alignment-given TKF92 fit

The canonical alignment-given TKF92(K=1) fit
(`fit_tkf92_cherryml.py` → `pfam/tkf92_K1_train.npz`, trained on
2,880 families / 51,390 aligned pairs) lands at
(λ=0.030, μ=0.030, ext=0.65) — DIFFERENT from any of the unaligned
optima.  Evaluated under the SAME 2D-FB joint LL on the canonical
1500-pair val, it scores **−614.385** — between SVI-BW
(−614.343) and Adam-tkf92's cold-start basin (−614.40).

**Three distinct local optima of the joint TKF92 LL on Pfam**:

* (λ=0.035, μ=0.036, ext≈0.58)  → SVI-BW
* (λ=0.030, μ=0.030, ext≈0.65)  → alignment-given fit (CherryML/Maraschino)
* (λ=0.055, μ=0.057, ext≈0.75)  → Adam-on-FB cold-start

All have val_ll/pair within ~0.06 nat of each other on the canonical
1500-pair val.  The flat basin near the SVI-BW optimum is hard to
reach from a cold-start Adam — even when Adam is warmstarted at the
SVI optimum, the gradient through 2D-FB drifts it toward the high-ext
basin.  This is a real (and reproducible) optimizer effect, not just
sampling noise.

### Findings

1. **Adam-tkf92 matches SVI-BW within seed noise** (-599.73 vs
   −599.69) at very different parameter points (ext=0.75 vs 0.58):
   the joint-LL surface has multiple near-equivalent optima.  Both
   are reproducible across seeds to ~0.01 nats.
2. **GGI upper-segment > GGI lower-segment by 1.4 nats** — Pfam's
   r_boundary lives above 0.4 (the lower-segment ceiling for
   ρ≈1), so the lower parameterisation truncates a useful basin.
3. **Adam-GGI with the GGI native geometric ancestor prior loses
   ~0.7 nats** to constant TKF92 — but **swapping in the TKF92
   ancestor prior closes the gap entirely and slightly beats it**.
   The GGI flow's transition dynamics are not the bottleneck; the
   geometric stationary length distribution is a worse fit than
   TKF92's compound-geometric prior.
4. With the canonical val (n=500), the runtime-reported and
   cross-eval val LLs agree exactly: same val pairs, same
   `forward_backward_2d` code path.  Prior runs that reported
   wildly different LLs (-665 vs -599) were a combination of the
   padding bug and the (now-fixed) `set()` hash-randomised val
   sampling.

## Files

```
experiments/run_tkf92_2dfb_pfam.py    # the launcher
experiments/2dfb/                      # this dir
  ├─ benchmark.md                      # this file
  ├─ {svi_bw, adam_tkf92, adam_ggi*}_*.json   # per-run history + best params
  ├─ {svi_bw, adam_tkf92, adam_ggi*}_*.log    # raw stdout
  ├─ eval_all_on_500val.py            # cross-eval helper
  ├─ eval_all_on_500val.json          # cross-eval output table
  ├─ cross_eval.py                    # earlier diagnostic (loss-function comparison)
tkfmixdom/jax/train/tkf92_adam_fb.py  # Adam loss + train loop
tkfmixdom/jax/train/tkf92_svi_bw.py   # SVI-BW loss + train loop
tkfmixdom/jax/dp/hmm.py:1835          # forward_backward_2d (real_Lx/Ly support)
```

## Reproducing

```bash
cd python
# SVI-BW
uv run python -u experiments/run_tkf92_2dfb_pfam.py \
    --mode svi_bw --precompiled-dir pfam/precompiled \
    --n-train 20000 --n-val 500 --batch-size 16 \
    --n-iter 200 --patience 50 \
    --bin-bucketed --pre-warm --no-command-buffers --max-pad-cap 256 \
    --out experiments/2dfb/svi_bw_canonical.json

# Adam-tkf92
uv run python -u experiments/run_tkf92_2dfb_pfam.py \
    --mode adam_tkf92 --precompiled-dir pfam/precompiled \
    --n-train 20000 --n-val 500 --batch-size 16 \
    --n-iter 800 --patience 100 \
    --bin-bucketed --pre-warm --no-command-buffers --max-pad-cap 256 \
    --out experiments/2dfb/adam_tkf92_long.json

# Adam-GGI upper (with GGI native prior)
uv run python -u experiments/run_tkf92_2dfb_pfam.py \
    --mode adam_ggi --precompiled-dir pfam/precompiled \
    --n-train 20000 --n-val 500 --batch-size 16 \
    --init-mu0 0.05 --init-rho 0.9 --init-x 0.7 --ggi-segment upper \
    --n-iter 800 --patience 100 \
    --bin-bucketed --pre-warm --no-command-buffers --max-pad-cap 256 \
    --out experiments/2dfb/adam_ggi_upper_long.json

# Adam-GGI upper (with TKF92 prior, no swap) — current best
uv run python -u experiments/run_tkf92_2dfb_pfam.py \
    --mode adam_ggi --precompiled-dir pfam/precompiled \
    --n-train 20000 --n-val 500 --batch-size 16 \
    --init-mu0 0.05 --init-rho 0.9 --init-x 0.7 --ggi-segment upper \
    --ggi-no-prior-swap \
    --n-iter 800 --patience 100 \
    --bin-bucketed --pre-warm --no-command-buffers --max-pad-cap 256 \
    --out experiments/2dfb/adam_ggi_upper_noswap_seed0_long.json
```

## Known caveats

* SVI-BW currently iterates ~35× slower than Adam-on-FB because its
  E-step and val_eval loop over pairs in Python (sequential
  `expm(t·Q)` + sequential F-B), while Adam vmap's the minibatch.
  A vmap rewrite is in flight.
* SVI-BW does not yet support `--ggi-no-prior-swap`; that comparison
  is currently Adam-only.
* The 0.04-nat gap between Adam-GGI-noswap and Adam-tkf92 is within
  seed variance and not yet replicated across multiple GGI-noswap
  seeds.  More seeds queued.
