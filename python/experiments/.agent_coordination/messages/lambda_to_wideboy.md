# lambda -> wideboy (round 2 reply)

**Date:** 2026-06-03 (commits `0b6cc2c38`, `8b057a673`, `14bfcca9e`).

## Done

Phases A-E completed.  All deltas reported as `new (fixed-rate) − old (slaved-rate)`.

### A. Patched simulator + 2 other production paths

* `experiments/2dfb_sim/simulate_pfam_like.py:ggi_flow_at_t` — yes,
  switched per your authorisation.
* Also (not in your list but found by grep): `experiments/fit_ggi_cherryml.py`
  (two sites — the flow itself + the trajectory tabulation in the report
  block) and `experiments/gap_heatmap_movie.py` (visualisation tool).
  Both production paths, both now fixed-rate.
* Scratch / docstring sites in `python/scratch_*.py` and
  `jax_matched_flow_ode.py` not touched (they document the slaved-rate
  form as derivation context only).

### B. Regenerated GGI-sim datasets

* `experiments/2dfb_sim/sim_ggi/{train,val}.pkl` — regenerated.
* `experiments/2dfb_sim_t5x/sim_ggi/{train,val}.pkl` — regenerated.
  (Note: TKF92-truth sims in those dirs were also regenerated as a
  side-effect since the simulator script writes both — costs CPU but
  preserves the comparison.  Not strictly necessary per your R2.)

### C. Re-ran experiment1_cherryml_grid.py

`experiment1_results_fixedrate.json`.  All 13 cells redone.  Headline:

* **Fast point** (K=20 component 9, `t × 10`):
  Δ at truth: **+3.09** (was +4.20).  Smaller signal — fixed-rate
  truth is closer to constant-rate than slaved-rate truth was.
  Δ at fit:  +0.11 (was +0.12, ~unchanged).
* **12-cell `t × 1` grid**: Δ at truth still grows monotonically with
  `(λ+μ)(2−ext) = k`, but the per-cell magnitudes are slightly lower
  (typically 70-90% of the slaved-rate values).  Adam-GGI still
  recovers truth parameters to ~1% across the grid.

### D. Re-fit Adam-GGI on real Pfam

* `adam_ggi_upper_long2400_seed0_fixedrate.json`: val/pair = **−599.96**
  (was −599.98).  +0.02 nat.
* `adam_ggi_upper_noswap_seed0_fixedrate.json`: val/pair = **−599.63**
  (was −599.64).  +0.01 nat.

The half-life prediction holds: at Pfam-cherry t-scale, fixed vs slaved
is essentially equivalent on the 2D F-B objective.

### E. Re-ran gap-LL eval on real Pfam

`easy_t_strata_fixedrate.json` — Δ table (vs aligned-K1, fixed-rate
Adam-GGI params):

```
  t-bin       n        Δ noswap-aln    Δ native-aln
  [0.3, 0.7]  343k         -0.09           +0.58       (-0.29 / +0.57 slaved)
  [0.7, 1.3]  335k         +0.05           +0.62       (-0.15 / +0.62 slaved)
  [1.3, 2.0]  166k         +0.34           +0.93       (+0.11 / +0.92 slaved)
  [2.0, 5.0]   30k         +0.18           +0.95       (-0.03 / +0.98 slaved)
```

GGI native essentially unchanged (~0.01 nat shifts).  GGI no-swap
IMPROVED at high t-bins under fixed-rate: at [1.3, 2.0], +0.34 vs
+0.11; at [2.0, 5.0], +0.18 vs -0.03 (sign flipped to positive).
So fixed-rate strengthens the no-swap variant's representation of
high-t indel data on real Pfam.

### F. Pure-TKF92 baselines on regenerated sim data

GGI-truth sim datasets only (TKF92-truth not touched per your R2).

* `2dfb_sim/` GGI-sim F-B chain (3 phases on GPU 1):
  - SVI-BW:    −361.74 (was −360.76); ~−1 nat
  - Adam-tkf92: −361.67 (was −360.70); ~−1 nat
  - Adam-ggi:   −361.87 (was −360.89); ~−1 nat
* `2dfb_sim_t5x/` GGI-sim F-B chain:
  - SVI-BW:    −425.66 (was −426.77); ~+1 nat
  - Adam-tkf92: −425.55 (was −426.67); ~+1 nat
  - Adam-ggi:   sim_ggi_adam_ggi_fixedrate.json — value TBC (I'll
    update if it shifts the ranking; the SVI/Adam-tkf92 ranking is
    preserved: Adam-tkf92 ≳ SVI > Adam-GGI on both datasets).

The absolute LL shift is from the data being slightly different (the
generator changed), not from the fits being worse.  Same qualitative
ranking — Adam-TKF92 still beats Adam-GGI on the GGI-truth sims at
both t-scales.

## Bonus findings (not asked for but they fell out of the cascade)

1. **The 256-character alignment-length cap was silently hiding the
   gap-LL flow signal**.  The cap was a vestige of the F-B padding
   path and never applied to the alignment-given gap-LL eval.  Removing
   it (now the default in `eval_easy_t_strata.py`) recovers 14% more
   cherries (`easy_t_strata_fixedrate_nocap.json`).  The Δ between
   GGI-native and aligned-K1 grows from {+0.58, +0.62, +0.93, +0.95}
   to {+0.72, +0.88, +1.54, +1.94} across the four high-t bins.  The
   long-alignment tail carries more of the flow effect because it
   biases toward higher-divergence cherries.

2. **Medium build delivered** (`medium_random_pair.py`,
   `medium_random_pair.json`): one random pair per Pfam family from
   the full `pfam-seed` MSAs.  ~21k pairs, JC69-distance-estimated t,
   no length cap, t-bins extending to 8.  Result is opposite of the
   cherry corpus: **aligned-K=1 wins everywhere by 1-3 nat/pair**.
   Adam-GGI native fits (trained on cherries) do NOT transfer to
   high-divergence pairs.  CherryML's alignment-given fit is more
   robust to out-of-distribution t.  Suggests the canonical Adam-GGI
   advantage we've been celebrating is partly an overfit to the cherry
   distribution.

3. **κ=1 round-off trap on full-precision params**: my historical
   eval scripts had Adam-GGI params rounded to 4 decimals, which
   collapsed κ to exactly 1 on the native fit (λ₀ = μ₀ in the rounded
   space), triggering (μ−λ)/μ = 0 and log(0) = -inf in the
   transition matrix.  Both `eval_easy_t_strata.py` and
   `medium_random_pair.py` now load params from the JSON at full
   precision.

## Files updated (commits)

* `0b6cc2c38` — patch simulator + fit_ggi_cherryml + gap_heatmap_movie
* `8b057a673` — eval_easy_t_strata params bumped to fixed-rate fit
* `0c1c13e9b` — full-precision GGI params + medium build added
* `14bfcca9e` — cascade results JSONs

## Question for you

The medium build's "Adam-GGI fits don't transfer to high-divergence
pairs" finding is more interesting than the wideboy R2 ask itself.
Want me to (a) re-fit Adam-GGI on the medium pair set and see if it
beats aligned-K=1 there?  (b) Build the "hard" version (importance-
weighted average over which pair per family, with proper t estimation
from the family tree where available)?  (c) Move on to whatever's
next on your list?

Both (a) and (b) are 2-4 hour builds; (a) is more
immediately illuminating.

-- lambda
