# Closed-form EM vs gradient-based optimization (paper 1)

Does closed-form Baum–Welch make fitting TKF-family models more efficient
than gradient-based optimization? These experiments compare closed-form EM
with Adam and L-BFGS under controlled conditions. All runs: 1,000 pairs, LG08 held
fixed except where noted, the same initial parameters for every method, and
one objective per (setting, model, corpus).

## Settings

| setting | objective | one "pass" | script |
|---|---|---|---|
| summed over alignments (unaligned) | 2D pair-HMM forward likelihood of the unaligned pairs | EM: forward–backward + closed-form M-step; gradient: forward DP + reverse-mode autodiff (or the custom VJP) | `compare_optimizers.py` |
| alignment given (summarised counts) | supplement B.6 count-tensor likelihood (one small forward–backward per τ bin; no per-pair DP) | EM: `em_fit` E-step + closed-form M-step; gradient: autodiff of the per-bin log-partition | `compare_optimizers_aligned.py` |
| MixDom, alignment given (1D constrained FB) | MAP objective logged by `train_pfam.py` (observed LL + log-prior), per-domain (S, π) estimated | EM: `--svi-bw --svi-kappa 0 --svi-batch 1000` (full-batch, closed-form M-steps); Adam on the Q-function from the same E-step | `train_pfam.py` via `launch_rest*.sh` |
| stochastic (minibatch 16, 20,000 Pfam pairs) | svi-BW vs Adam-on-2D-FB, the protocol of `../2dfb/benchmark.md` | one minibatch E-step / gradient step | `stochastic/run_pfam_stochastic.sh` |

Models: TKF91 (λ, μ), TKF92 (λ, μ, r), MixFrag F=2 (λ, μ, r₁, r₂, w),
MixDom D=3, F=1 with per-domain substitution. Corpora: breadth-first Pfam
cherries (seed 0) and, for each model, a corpus simulated from that model at
Pfam-like branch lengths (`sim_tkf91`, `sim_tkf92`, `sim_mixfrag`; the
`sim_tkf92`/TKF91 and `sim_tkf92`/MixFrag rows are misspecified fits kept for
reference). Methods: closed-form EM (flat priors, so the fixed point is the
MLE), scipy L-BFGS-B on unconstrained parameters, Adam at learning rates 0.1
and 0.01 (stopped when 200–300 passes bring no gain above 1e-7 nat/pair).

Convergence is measured post hoc by `summarize.py`: the first pass whose
log-likelihood per pair is within ε of the best value reached by any method
on that objective (ε = 1e-2, 1e-3, 1e-4), plus wall-clock on a warm JIT
cache. Tables: `results/summary.md` (unaligned), `results_aligned/summary.md`
(alignment given), `results_vjp/` (custom-VJP gradient variants),
`results_mixdom/` (MixDom logs), `stochastic/`.

## Gradients: custom VJP vs autodiff

The gradient of the observed-data log-likelihood equals the gradient of the
Q-function at the current parameters (Fisher's identity), so the E-step
sufficient statistics give the gradient in closed form; this is the custom
VJP route of the paper and of `tkfmixdom/jax/train/vjp.py`. Every unaligned
run prints a three-way check at the initial point and at two random points:
the library's custom VJP (`vjp.tkf91_log_prob` / `tkf92_log_prob`, unpadded
per-pair), reverse-mode autodiff through the padded forward DP, and the
batched score-identity gradient assembled from `estep_batch_*`. The three
agree to a relative 1e-14 (see `gradcheck` in each JSON). They are the same
mathematical object; what differs is cost per pass, which is a property of
the implementation (and of JAX's autodiff) rather than of the method:

| pass type (1,000 pairs) | simulated TKF92 | Pfam |
|---|---|---|
| forward DP + reverse-mode autodiff | 6.0 s | 11 s |
| forward–backward E-step (counts, BDI statistics), as used by EM and by the custom-VJP gradient | 19 s | 28 s |

In this implementation the E-step is about three times the cost of an
autodiff gradient pass because it materialises the full posterior transition
count matrix and resolves it in separate jitted calls; the custom-VJP
gradient costs the same as the E-step (`pass_seconds.vjp`). Adam and L-BFGS
driven by the custom-VJP gradient (`--methods lbfgs_vjp adam_vjp`) follow the
autodiff trajectories pass for pass (`results_vjp/`), so the pass counts below
hold for either gradient implementation; only wall-clock changes.

## Results (passes to within 1e-3 nat/pair of the optimum)

See `results/summary.md` and `results_aligned/summary.md` for the full
tables (ε = 1e-2 / 1e-3 / 1e-4, wall-clock, final parameters). Headline:

| setting | model | corpus | EM | L-BFGS | Adam 0.1 | Adam 0.01 |
|---|---|---|---|---|---|---|
| unaligned | TKF91 | sim (TKF91) | 5 | 6 | 72 | 627 |
| unaligned | TKF91 | Pfam | 6 | 8 | 141 | >1000 |
| unaligned | TKF92 | sim (TKF92) | 52 | 7 | 57 | 888 |
| unaligned | TKF92 | Pfam | 115 | 7 | 132 | >1000 |
| unaligned | MixFrag | sim (MixFrag) | 371 | 15 | 168 | >1000 |
| unaligned | MixFrag | Pfam | >300 | 15 | 307 | >1000 |
| aligned | TKF92 | sim (TKF92) | 41 | 7 | 59 | 1034 |
| aligned | TKF92 | Pfam | 61 | 8 | 99 | 1202 |
| aligned | MixFrag | sim (MixFrag) | 465 | 14 | 184 | 1269 |
| aligned | MixFrag | Pfam | 1084 | 16 | 283 | 1901 |

Every method that ran to convergence reached the same optimum (to 1e-6
nat/pair). On the unaligned MixFrag objectives EM had not converged after its
300-pass budget (1.6e-3 nat/pair short on the simulated corpus, 2.8e-2 on
Pfam). A longer EM-only run on the simulated corpus (`results_emlong/`,
logged every 10 passes) reached 1e-2 at pass 91, 1e-3 at pass 371 and was
within 1e-4 by pass 631; the Pfam run is deferred until the GPUs are free
(`launch_emlong_mixfrag.sh`). Observations:

1. For the BDI rates (TKF91), closed-form EM converges in 5–7 passes, as
   fast as L-BFGS and 10–25× faster than Adam at its best learning rate.
2. Adding the fragment-extension probability (TKF92) slows EM to 40–115
   passes; L-BFGS still needs 7–8. EM remains ahead of Adam at lr 0.01 by an
   order of magnitude and roughly level with Adam at lr 0.1.
3. The fragtype weights of MixFrag converge last (the slow tail predicted
   in the svi-BW appendix): EM needs ~470 passes on the aligned objective,
   more than Adam at lr 0.1 (184) and far more than L-BFGS (14).
4. EM needs no learning rate, no line search and no parameter transform,
   and its objective is monotone; the gradient methods' pass counts depend
   strongly on the learning rate (Adam) or on a working line search
   (L-BFGS), and on the unconstrained parameterisation.

So the defensible statement is: closed-form EM matches quasi-Newton
convergence for the indel rates and beats first-order methods at any safe
learning rate; for mixture weights it converges in more passes than L-BFGS,
and the custom VJP makes the same sufficient statistics available to either
optimizer at the same cost per pass.

## MixDom (alignment given, per-domain substitution estimated)

Both modes start from identical parameters (`results_mixdom/init_d3f1_seed0.npz`)
and log the same MAP objective (observed LL + log-prior) at each E-step
(`results_mixdom/*.log`).

**A bug found on the way.** The first full-batch EM run was not monotone
(peak −571,470 at iteration 41, dip to −571,650, best −571,148 at 176), and a
2,000-step Adam run on the Q-function ended above it (−570,854). A one-step
test (`mixdom_mstep_check.py`) showed the closed-form M-step *lowering* the
Q-function it was handed while a numerical maximiser raised it by 112 nats,
the whole gap sitting in the per-domain exchangeability block; the indel
blocks matched the maximiser to 1e-6. Cause: the svi-BW loop copied only the
count tensors from the E-step into its EMA running statistics, never the
per-pair-t Holmes–Rubin `dom_W`/`dom_U` (nor `class_W`/`class_U`), so the
substitution M-step silently took its "older E-step" fallback and recomputed
W/U at the batch's median time. Fixed in train_pfam.py (commit b349b1825): the
keys are flowed through and the fallbacks raise. Every svi-BW MixDom fit with
substitution estimation since 2026-04-25 — including the paper's
`svibw_d3f1_perdomain_v3` and its LG-fix refit — used the wrong M-step.
Retrained under the fix with the same recipe (`launch_retrain_d3f1.sh` →
`pfam/svibw_d3f1_perdomain_v4_fix.npz`, 540 iterations in 93 min): held-out
path-conditioned −718.50 (was −718.51), order-summed −713.57 (was −712.96);
BAliBASE at gap factor 0: F1 0.375 / SP 0.819 / TC 0.686 (was 0.377 / 0.825
/ 0.692), no per-family difference significant (Wilcoxon p > 0.3). Same
qualitative structure (one indel-rich domain type, two near-conserved ones).
d4f1 and d5f1 are being trained and evaluated the same way
(`launch_train_eval_dNf1.sh`).

**After the fix** (`results_mixdom/em_d3f1_fixed*.log`, same start): the EM
objective is monotone (24 decreases of at most 0.1 nats in 2,000 iterations,
numerical noise of the coordinate-ascent substitution step), reaches −571,188
at iteration 20, −570,909 at 40, −570,803 at 200 and converges by ~500 to
−570,796; one more iteration coincides with the numerical Q-maximiser
(`mstep_check_fixed.log`), and the objective gradient at the converged point
is essentially zero (`gradient_at_points_2000.json`: indel block 3.6 nats per
unit log-rate, substitution 0.04; finite difference agrees). Each iteration
takes 1.9 s instead of 49 s (the fallback evaluated 1,200 Holmes–Rubin
expectations per iteration).

The converged point is a *local* maximum, and a degenerate one: the third
domain type has collapsed to near-zero indel rates (λ₃=0.0010, μ₃=0.0011).
Adam on the same Q-function from the same start, at lr 0.01, climbs to
−575,006 after 200 steps and −570,854 after 2,000 (still below EM's local
maximum); at lr 0.1 it oscillates (−572,037 at step 99, −574,369 at 199) but
visits points up to −570,632, 165 nats above EM's maximum, with all three
domain types active (rates 0.02–0.11). So the MixDom likelihood is
multimodal: in E-step equivalents EM is ahead of Adam at every pass count up
to about 1,000 and reaches its maximum in ~500, but as a local method it
needs restarts (or the noise of minibatch svi-BW) to escape the collapsed
solution, which a noisy first-order method happened to do here.

**Replicates (seeds 1-4, `launch_mixdom_reps.sh`, 1,500 iterations each).**
Full-batch EM from four different random initialisations converges to the
same solution every time (MAP -570,796 to -570,799; rates
(0.001, 0.013, 0.078), weights (0.38, 0.24, 0.38)), so the near-zero-indel
domain type is a reproducible conserved class, not an accident of one start.
Adam from the same four starts has not converged after 1,500 steps and lands
on a different parameter set each time: at lr 0.1 its best visited value is
above EM's for one seed (-570,766, seed 4) and 50-1,200 nats below for the
other three; at lr 0.01 it is 160-450 nats below for every seed. So the
objective is multimodal (points above EM's stationary point exist) but EM is
the method that finds the same mode reliably.

## Held-out likelihood of the retrained d3f1/d4f1/d5f1 (contaminated corpus)

Same v3_val set, corrected M-step, 540 iterations each
(`ordsum_val/pathcond_svibw_d*_perdomain_v4_fix.log`, `score_d*_perdomain_v4_fix.log`):

| model | path-conditioned LL/pair | order-summed LL/pair | BAliBASE SP / TC (gap factor 0) |
|---|---|---|---|
| d3f1 | -718.50 | -713.57 | 0.819 / 0.686 |
| d4f1 | -717.17 | -712.24 | 0.812 / 0.672 |
| d5f1 | -717.46 | -712.26 | 0.815 / 0.678 |

More domain types buy held-out likelihood but not BAliBASE accuracy (no
pairwise difference significant, Wilcoxon p >= 0.24).

## Retraining on a BAliBASE-free corpus (in progress)

A sequence-level check (`check_balibase_train_overlap*.py`) found that 51 of
the 120 BAliBASE references have a same-family relative among the training
pairs (the split's 59-family benchmark list is incomplete). The
`../balibase_exclusion/` pipeline (pyhmmer hmmscan vs Pfam-A 38.2, phmmer and
jackhmmer vs the training sequences, E <= 0.01, clan-expanded) removes 2,284
families / 140,514 pairs (12.4%), and d3f1/d4f1/d5f1 are being retrained on
`pfam/precompiled_v3_train_nobali` for 1,620 iterations (3x), with Adam-on-Q
and L-BFGS tails from the 1,620-iteration point (`launch_retrain_nobali.sh`).
First numbers: d3f1 at 540 iterations, held-out path-conditioned -718.23 /
order-summed -713.19 (vs -718.50 / -713.57 on the full corpus), BAliBASE
0.809 / 0.670 (vs 0.819 / 0.686, p = 0.15 / 0.13); at 1,080 iterations
-718.09 / -713.11.

## Stochastic (minibatch) setting

Same 20,000 breadth-first Pfam pairs, canonical 500-pair validation set,
minibatches of 16, same initial parameters, 800 iterations each
(`stochastic/pfam_{svi_bw,adam_tkf92}_seed0.{json,log}`):

| | svi-BW (closed-form M-step on EMA statistics) | Adam on 2D-FB (lr 0.01) |
|---|---|---|
| validation LL/pair at iteration 10 / 100 / 800 | −599.10 / −599.06 / −599.03 | −607.02 / −600.99 / −599.12 |
| first iteration within 0.1 nat/pair of the best value | 10 | 785 |
| wall-clock for 800 iterations (vmapped E-step) | 361 s | 947 s |

Here svi-BW is ahead by almost two orders of magnitude in iterations: each
minibatch E-step is turned into a full M-step, whereas Adam at a safe
learning rate takes small steps. The advantage is specific to this
minibatch setting; it does not carry over to full-batch optimization against
a quasi-Newton method (above).
