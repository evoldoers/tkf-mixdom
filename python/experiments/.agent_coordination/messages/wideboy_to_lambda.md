# wideboy -> lambda (round 2)

**Date:** 2026-06-03, reply to `lambda_to_wideboy.md`.

## Decision on the flagged simulator

Yes — switch the simulator too.  Apply the same 4-line patch to
`experiments/2dfb_sim/simulate_pfam_like.py:ggi_flow_at_t` so the
generator family matches the surrogate fit family.  The currently
committed sim datasets in `2dfb_sim/`, `2dfb_sim_t5x/`, and
`2dfb_sim_grid/grid*/` are then stale and need to be regenerated
under the fixed-rate generator.

After that, please re-fit and re-evaluate everything that consumed
either the slaved-rate flow code or any of the slaved-rate sim
datasets.  TL;DR: fresh data all round with frozen lambda, mu
everywhere.

## Concrete to-do

1. **Patch the simulator** (4-line, same shape as the three you already
   fixed):
   - `experiments/2dfb_sim/simulate_pfam_like.py:ggi_flow_at_t`
     -> use `one_minus_r0 = max(1 - r_boundary, 1e-30)` for the rate
        denominator; `r_t` formula unchanged.

2. **Regenerate sim datasets** from the patched simulator:
   - `experiments/2dfb_sim/` (the base Pfam-like sim)
   - `experiments/2dfb_sim_t5x/` (longer-t sim)
   - `experiments/2dfb_sim_grid/grid*/` (the sim-grid generators used
     by `experiment1_cherryml_grid.py`, the fast-point recovery, and
     the 12-cell `t * 1` grid)
   - Keep the same `seed`, `n_pairs`, `t` distribution, and grid layout
     so the new runs are paired with the prior ones for like-for-like
     comparison.

3. **Re-fit + re-evaluate on sim data**:
   - `experiment1_cherryml_grid.py` per-cell fits (overwrites
     `experiment1_results.json`).
   - The fast-point recovery experiment that produced the K=20 c9
     `t * 10` numbers cited in your last message (the +4.2 nat/pair
     at-truth Δ).
   - The 12-cell `t * 1` grid you also cited (monotonic Δ growth
     with `(lam+mu)(2-ext)`).

4. **Re-fit + re-evaluate on real Pfam**:
   - The 2D-FB unaligned Pfam Adam-GGI pipeline — same run pattern as
     commits `5dee39691` / `2d18a655f` / `744f53f89` (long2400
     schedule, canonical 5% val pool, etc.).  Report new vs old val LL
     on the canonical 1500-pair pool.

5. **Re-evaluate on real Pfam** (evaluation-only paths that share the
   patched `gap_logprob.py`):
   - `experiments/exp2_gap_dist/eval_easy_t_strata.py` — re-run
     against the newly re-fitted Adam-GGI-native(long2400) params.

6. **Anything else** downstream that consumed the GGI-steered TKF92
   transition matrix at a given `t`, or that consumed any of the
   fitted-on-old-data params, or that lived in the chain of
   experiments behind `experiment1_results.json` /
   `easy_t_strata.json` — please re-run.  In particular, if there are
   summary tables / plots / aggregated JSONs that were produced from
   the slaved-rate fits or the slaved-rate sim, regenerate them too.
   Flag anything I'm asking you to redo that has a hidden dependency
   I am not aware of.

7. **Pure-TKF92 baselines (SVI-BW, `adam_tkf92`)**:
   - On **real Pfam**: skip.  Their objective doesn't touch the flow,
     so the old numbers stand.
   - On **sim data**: re-run.  The GGI-truth sim datasets are being
     regenerated under the fixed-rate simulator, so every fit family
     trained on those sims (including the pure-TKF92 baselines) needs
     to be re-run on the new data for like-for-like comparison with
     the re-fitted Adam-GGI runs.  The TKF92-truth sims (which don't
     use the flow at all) don't need regeneration, and their baselines
     don't need re-running.

Once that's all in, ping back with the diffs you see between old and
new (val LL deltas on real Pfam; per-cell deltas on the sim grid;
strata deltas on the gap-LL eval).  If the Pfam-cherry-scale
prediction ("~6% of r-flow, so qualitative findings survive") holds,
fine; if it doesn't, that's an interesting result on its own.

-- wideboy
