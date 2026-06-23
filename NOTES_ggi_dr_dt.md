# GGI → TKF92 matched-flow dr/dt: working notes

**Status as of 2026-05-31.** The Triad ODE integrator
(`scratch_ggi_cond_kl_quad.py`) is now the *canonical* implementation —
clean, pure-numpy algebraic, no L'Hopital footguns, no kl_fit
workarounds. The dr/dt sign issue described below was **resolved by
switching to the corrected coarse-graining**: null-eliminate the ID
state from the 13-state Triad, then project to the 5-state pair-HMM
via `CMAP_12 = [S, I, M, I, D, I, I, D, D, D, D, E]`.  The old
`scratch_ggi_triad_algebraic.coarse_grain` had `sI` mapped to `S` (it
should be `I`), `mI` mapped to `M` (should be `I`), and `MD` mapped to
`M` (should be `D`); using that coarse-graining flipped the sign of
dr/dt.

On the May 29 commit's asymmetric reference (lam0=0.035, mu0=0.045,
x_del=0.6, y_ins=0.4) the cleaned-up canonical integrator now gives r
decreasing from 0.5317 -> 0.4491, matching both the
fragment-erosion closed-form (r_inf = r0/(2-r0) = 0.362) and the
empirical Gillespie fit table in the original "sign problem" section
below.  On Pfam joint-native parameters r decreases from 0.6392 ->
0.6017.

The original 2026-05-28 NOTES below described the sign issue as
applying to the integrated ODE; in retrospect it was equally
applicable to the hand-derived dr/dt formula in
`scratch_ggi_triad_algebraic.dr_dt`, which uses the same buggy
coarse-graining.  The Triad ODE itself (eq:ode-params) is fine; only
the projection from 13 -> 5 states was wrong.

**Status as of 2026-05-28** (historical).  Algebra is set up; sign of
dr/dt comes out opposite to empirical Gillespie fits; multiple
candidate fixes identified but not yet validated.

## Goal

Derive a closed-form ODE for $r(t)$ for a TKF92 surrogate fitting a GGI
process at branch length $t$, under Born-Oppenheimer (BO) approximation
where $\lambda(t), \mu(t)$ are slow background coordinates and $r(t)$ is
the fast m-projection-tracked coordinate.

## What's in the supplement

Appendix A.8.9 "The Triad HMM" defines the 13-state composite
$\jointpairhmm(\theta, t) \circ \condpairhmm(\theta_\ggi, dt)$:

- States: SS, sI, MM, mI, MD, IM, iI, ID, Ds, Dm, Di, Dd, EE
- Macros use `\mathtt{}` for case-distinguishable rendering
- Equations A.73: path counts $n_{AB} = (I-T)^{-1}_{SA}\,t_{AB}\,(I-T)^{-1}_{BE}$
- Eq A.74: coarse-grained TKF92 counts $m_{XY} = \sum_{C(A)=X, C(B)=Y} n_{AB}$
- Eq A.75: fragment extension/termination decomposition with
  $u_{XX} = r + (1-r)v_{XX}$
- Eq A.76: implicit M-step equation $\sum_X m_{XX}/u_{XX} = N_{tot} - 1$
  (the $-1$ from the start transition's r-independence)
- BO paragraph: $\mu_{TKF} = \mu_0/(1-y)$, $\lambda_{TKF} = \kappa(r)\mu_{TKF}$
  with $\kappa(r)$ via length conservation

## Algebraic framework (in `scratch_ggi_triad_algebraic.py`)

$T = A + B\,dt$ where $A = T|_{dt=0}$, $B = dT/d(dt)|_0$.

$(I-T)^{-1} = R_0 + dt\,R_0 B R_0 + O(dt^2)$ with $R_0 = (I-A)^{-1}$.

Reachable at dt=0: {SS, MM, IM, Ds, Dm, EE} — coarse to {S, M, I, D, D, E},
exactly the TKF92 pair-HMM resolvent. Unreachable: {sI, mI, MD, iI, ID, Di, Dd}.

$B$ is sparse: only 5 nonzero rows (SS, MM, IM, Ds, Dm), with Ds and Dm
identical (because $z_{S,\cdot}$ and $z_{M,\cdot}$ have the same
$dt$-derivatives).

$dr/dt$ formula (linearised M-step around $r_{native}$ + self-consistency):

$$\frac{dr}{dt} = \frac{\sum_X m^{(1)}_{XX}/u_{XX} - N^{(1)}_{tot}}{\sum_X m^{(0)}_{XX}(1-v_{XX})/u_{XX}^2}$$

Closed-form rationals in $(\lambda, \mu, r, t, x, y, \lambda_0, \mu_0)$
once $R_0$ and $R_0 B R_0$ are evaluated.

## What's confirmed working

1. Self-consistency at $dt=0$: $\sum_X m^{(0)}_{XX}/u_{XX} = N^{(0)}_{tot} - 1$
   exactly (selfcheck = -1 numerically).
2. Nonlinear cross-check: $r^*(t, dt) - r$ at small $dt$ matches the linearised
   prediction to high precision across multiple $dt$ values.

## The sign problem

For $(x,y) = (0.4, 0.55)$ evaluated at empirical Gillespie $(\lambda^*, \mu^*, r^*)$:

| $t$ | $r$ empirical | $dr/dt$ algebra | $dr/dt$ empirical |
|---|---|---|---|
| 0.1 | 0.284 | +0.20 | -0.03 |
| 0.5 | 0.256 | +0.31 | -0.09 |
| 1.0 | 0.221 | +0.35 | -0.06 |
| 2.0 | 0.191 | +0.30 | -0.02 |
| 4.0 | 0.167 | +0.12 | -0.02 |

The algebraic ODE consistently predicts $r$ *increasing*; the actual matched
trajectory has $r$ *decreasing*. Signs opposite.

## Diagnosis: Triad ODE ≠ direct-fit trajectory

The Triad construction in eq:ggi-triad is
$\theta(t+dt) = \arg\min D(\tkfwfst'(\theta(t),t) \circ \mathrm{GGI}(dt) \| \tkfwfst'(\theta, t+dt))$
— compose the **current surrogate** with one more GGI step, refit. This is a
*self-composed surrogate flow*.

The direct Gillespie fit at branch $t$ minimises
$D(\mathrm{GGI}(t) \| \tkfwfst'(\theta, t))$ — fit to **the true GGI process**
at each $t$.

These two trajectories agree only at the TKF91 boundary ($r=0$) and drift
apart by accumulated $\Delta_2$ defect. The Triad ODE faithfully captures
"one more GGI event pulls $r$ toward $x, y$" (positive), but misses the
cumulative effect of surrogate-defect mismatch that drives $r$ down in the
true matched trajectory.

## Other issues identified

1. **BO formulae are off:** my $\mu = \mu_0/(1-y) = 2.22$ vs empirical
   $\mu \approx 1.07$. Factor of 2 wrong. The full 3-parameter
   m-projection picks $\mu$ much closer to $\mu_0$ than to $\mu_0/(1-y)$.
   Suggests the "per-residue deletion rate" constraint is heuristic, not
   actually variational.
2. **The matched trajectory shoots up then dribbles down:**
   - At $t=0$: best fit is TKF91 ($r=0$, $\kappa = x/y$, rate undetermined)
     because GGI's stationary length is plain geometric and TKF91 matches
     exactly, while TKF92 has zero-inflation.
   - At $t > 0$: $r$ jumps up to match the fragment-correlation structure
     of GGI's indel events (the "shoot up").
   - At larger $t$: $r$ declines gradually (the "dribble down").
   - This trajectory shape suggests a fast initial transient (or true
     discontinuity) at $t = 0$ followed by a regular flow.

## Attempted fix: zero-corrected TKF92

**Idea (user's, elegant):** TKF92 with probability $R$ of resampling on
empty alignment, with $R = r/w$ chosen to give plain geometric stationary
(parameter $w = r + (1-r)\kappa$). With $w = x/y$ this matches GGI's
stationary exactly. Equivalently: TKF92 with a "ghost fragment continuation"
that removes zero-inflation. This **dissolves the $t=0$ discontinuity**:
at $t=0$ any $r$ along the curve $\kappa(r) = (x/y - r)/(1-r)$ is optimal.

**Implementation:** `tau5_zc(κ, α, r)` in `scratch_ggi_triad_zc.py` rescales
the S row of the pair-HMM matrix: $y_{S,Y} \to y_{S,Y} \cdot w/\kappa$ for
$Y \in \{M,I,D\}$, $y_{S,E} \to y_{S,E} \cdot (1-R) \cdot w/\kappa$.

**Result:** sign did NOT flip. Diff at $dt = 10^{-5}$ was $O(10^{-2})$
(should be $O(dt) = O(10^{-5})$ for a smooth perturbation). This indicates
self-consistency failure at $dt=0$ — the m-projection of the ZC TKF92's own
counts back onto ZC TKF92 isn't giving back the input $r$.

**Suspected reasons:**
1. My matrix-entry rescaling may not be exactly equivalent to the
   path-level resampling (need to re-derive at path-count level).
2. Coarse-graining Ds/Dm onto a single D might break under ZC's S-row changes.
3. The 5-state pair-HMM likelihood I'm using as the cost may not match
   the proper ZC likelihood.

## Three candidate paths forward

**(a) Path-level zero-correction derivation.** Re-derive the ZC pair-HMM by
weighting each path by the resampling factor, then identifying which
matrix entries change. Test self-consistency at $dt=0$. If passes, re-test
sign.

**(b) Direct-fit ODE formulation.** Abandon the Triad eq:ggi-triad. Instead
write
$$d\theta^*/dt = -F(\theta^*)^{-1} \cdot \partial_t \partial_\theta D(\mathrm{GGI}(t) \| \tkfwfst'(\theta^*(t), t))$$
with $\partial_t$ acting through GGI's own generator on the GGI side (not
through the surrogate). This is the proper matched-flow trajectory ODE.
Requires modifying B in the algebraic construction so that the leftmost
matrix is the GGI generator action on GGI(t), not on the surrogate.

**(c) Pause and regroup.** Land everything currently working in the
appendix, commit the scratch files, and revisit later.

## Files and commits

- `~/tkf-mixdom/tkf/composition-renormalization.tex` — main appendix
  (eq:ode-params at lines ~222-237 is the ODE we integrate)
- **`~/tkf-mixdom/python/scratch_ggi_cond_kl_quad.py`** — *canonical*
  pure-numpy algebraic integrator (use this).  Implements eq:ode-params
  end-to-end; non-recursive L'Hopital limit at λ=μ; pure numpy + scipy.
- `~/tkf-mixdom/python/scratch_ggi_cond_kl_ode.py` — thin re-export
  shim for `scratch_ggi_cond_kl_quad`.
- `~/tkf-mixdom/python/scratch_ggi_triad_eliminated.py` — keeps the
  ID-null-elimination helpers (`null_eliminate_ID`, `coarse_grain_12`,
  `triad_counts_eliminated`); re-exports ODE entry points from
  `scratch_ggi_cond_kl_quad`.
- `~/tkf-mixdom/python/scratch_ggi_triad_algebraic.py` — A+B*dt
  expansion (the algebra layer cond_kl_quad sits on top of).
- `~/tkf-mixdom/python/scratch_ggi_triad_zc.py` — zero-correction
  attempt (broken self-consistency at $dt=0$, see "Attempted fix" above)

Pre-cleanup (2026-05-28) commits:
- `tkf-mixdom @ ca011fb71` — algebra + BO mu correction
- `tkf-mixdom @ a3ca599f5` — 13-state Triad with fixed labels

Cleanup (2026-05-31):
- Crude recursive L'Hopital fallback removed from
  `_tkf91_bdi_from_m` (it perturbed by lambda*1e-4 below a 1e-4
  threshold, so for symmetric GGI it never escaped and stack-overflowed).
- `lam < mu` early-return guard removed (the L'Hopital handling now
  covers the singular case directly).
- Old `dtheta_dt_fixed` / `run_flow_fixed` workaround stripped from
  `python/experiments/compare_closedform_vs_odeflow.py`.

### Coarse-graining fix

The bug was localized in `scratch_ggi_triad_algebraic.coarse_grain`,
which maps 13 -> 5 states.  Its `C_map = (S, S, M, M, M, I, I, I, D,
D, D, D, E)` is wrong for index 1 (`sI`), index 3 (`mI`), and index 4
(`MD`).  Correct: `sI -> I` (surrogate at S, GGI inserted), `mI -> I`
(surrogate at M, GGI inserted on top), `MD -> D` (surrogate at M, GGI
deleted what was there -- the X->Z position becomes a deletion).

Cleaned-up `cond_kl_quad.stats_and_increments` now:
  1. builds the 13x13 (A, B) via `build_AB`;
  2. null-eliminates the ID state -> (U, V) on 12 states (sI, mI, MD
     etc. are still PRESENT -- only ID is removed, since a "ghost
     insertion + immediate deletion" produces no X->Z column at all);
  3. computes resolvent C and CVC on 12x12;
  4. coarse-grains n^(0), n^(1) via CMAP_12 = [S, I, M, I, D, I, I, D,
     D, D, D, E];
  5. applies the closed-form TKF92 BDI map.

This matches the pipeline that the old `triad_eliminated.py` used
(but with closed-form BDI instead of kl_fit, so much faster).

### The "Diagnosis" section above is now of historical interest only

The May 28 diagnosis claimed the Triad self-composition flow itself is
fundamentally different from the matched trajectory.  Empirically the
Triad ODE -- with the correct coarse-graining -- does give the same
sign and approximately the same magnitude as the empirical
direct-fit-to-Gillespie trajectory.  No fundamental
self-composition-vs-direct-fit divergence; just a 13->5 projection
bug.

## Key technical notes for resumption

- The reachable-subspace resolvent $R_0$ on {SS, MM, IM, Ds, Dm, EE}
  equals the standard TKF92 pair-HMM resolvent on {S, M, I, D, E}, with
  the Ds+Dm coarse-graining preserving expected counts.
- $B$'s sparsity: 5 nonzero rows (SS, MM, IM, Ds≡Dm), all entries are
  linear in $\lambda_0$ or $\mu_0$ — so $dr/dt$ is a rational function
  with linear-in-$\lambda_0,\mu_0$ numerator.
- TKF92 pair-HMM in `tau5(κ, α, r)`: S-row has no $r$-dependence (this is
  the source of the "-1" in the self-consistency equation).
- Empirical $\lambda$, $\mu$ trajectories from Gillespie are nearly
  constant: $\lambda \approx 0.71-0.76$, $\mu \approx 1.03-1.09$ across
  $t \in [0.1, 5]$. So almost all the "action" is in $r(t)$.
- The "shoot up then dribble down" is qualitatively important — it means
  the matched trajectory is NOT a simple relaxation toward a fixed point.

## Useful equations

GGI reversibility: $\lambda_0 y(1-y) = \mu_0 x(1-x)$.

GGI equilibrium length: $\ell_\ggi = x/(y-x)$.

GGI per-residue deletion rate: $\mu_0/(1-y)$.

TKF92 mean stationary length: $\kappa / ((1-\kappa)(1-r))$.

Length-conservation $\kappa(r)$: $\kappa = x(1-r)/(y - xr)$.

With zero-correction targeting $w = x/y$: $\kappa(r) = (x/y - r)/(1-r)$.
