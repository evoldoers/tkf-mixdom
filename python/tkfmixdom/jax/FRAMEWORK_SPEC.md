# Compositional Parameter & Likelihood Framework

## Motivation

Every TKF-family model (TKF91, TKF92, MixDom, TKFST, TKFBS) follows the same
training pattern:

1. Construct grammar from model parameters via elaboration rules
2. Remove null productions (epsilon nonterminals + null cycles) to get DP-ready form
3. Run Forward-Backward (HMMs) or Inside-Outside (SCFGs) to get expected counts
4. Restore null counts to recover true sufficient statistics
5. M-step: update parameters from sufficient statistics
6. Verify: `grad(log_prob + log_prior)` via autodiff == grad via sufficient stats

Currently each model has bespoke code for steps 1-5. The framework makes this
compositional: grammar elaboration rules (from `grammar-elaboration.tex`) become
Python builder functions that construct aligned pytrees of parameters, priors,
and sufficient statistics. Null removal and restoration are separate,
model-independent post-processing steps.

## Parameter Types

Three fundamental parameter types, each with a conjugate prior:

```python
@dataclass
class RateParam:
    """Positive real parameter (λ, μ, etc.). Prior: Gamma(shape, rate)."""
    value: float
    prior_shape: float = 2.0
    prior_rate: float = 10.0

@dataclass
class SimplexParam:
    """Probability vector (domain weights, fragment weights). Prior: Dirichlet(alpha)."""
    value: jnp.ndarray       # (K,) sums to 1
    prior_alpha: jnp.ndarray  # (K,) pseudocounts

@dataclass
class BernoulliParam:
    """Probability in [0,1] (extension rate). Prior: Beta(a, b)."""
    value: float
    prior_a: float = 1.0
    prior_b: float = 1.0
```

## Sufficient Statistics

Parallel to each parameter type:

```python
@dataclass
class RateSuffStats:
    """For a BDI process: E[B], E[D], E[S] recovered from score identity."""
    E_B: float  # expected births
    E_D: float  # expected deaths
    E_S: float  # expected survivals

@dataclass
class SimplexSuffStats:
    """Posterior category counts (add to Dirichlet pseudocounts for M-step)."""
    counts: jnp.ndarray  # (K,) expected counts per category

@dataclass
class BernoulliSuffStats:
    """Extension loop counts."""
    n_extend: float   # times extended (self-loop)
    n_exit: float     # times exited (1-r transition)
```

## Model Spec: What a Builder Returns

Each grammar builder returns a `GrammarSpec` — the raw grammar before null
removal. A separate `compile` step removes nulls to produce a `CompiledModel`
ready for DP:

```python
@dataclass
class GrammarSpec:
    """Raw grammar produced by elaboration builders."""
    params: dict          # pytree of {RateParam, SimplexParam, BernoulliParam}
    build_rules: Callable # params -> (rules/transitions, state_types, null_info)
    # null_info records which states are null, nullabilities, etc.

@dataclass
class CompiledModel:
    """Grammar after null removal — ready for DP."""
    params: dict              # same pytree as GrammarSpec
    build_trans: Callable     # params -> (chi, state_types) — null-free
    extract_stats: Callable   # (n_counts, chi, params) -> suff_stats pytree
    m_step: Callable          # (suff_stats, params) -> new_params
    dp_type: str              # 'hmm' (Forward-Backward) or 'scfg' (Inside-Outside)
```

The `compile` step:
1. Identifies nullable nonterminals, computes nullabilities
2. Removes ε-producing nonterminals (SCFG nullability correction)
3. Removes null cycles via `(I - T_ZZ)^{-1}` closure (HMM null cycle removal)
4. Records the removal metadata needed to restore counts later

The `extract_stats` in `CompiledModel` automatically restores null counts
before computing sufficient statistics — this is the inverse of null removal.

The key property: `params`, `suff_stats`, and the prior pseudocounts are
**parallel pytrees** with the same structure.

## Grammar Elaboration Builders

Each elaboration rule from `grammar-elaboration.tex` is a Python function that
wraps an inner `ModelSpec`, adding parameters and extending the extract/m_step:

### 1. `bdi_process(ins_rate, del_rate, t)` — Link Grammar

Creates the base BDI (birth-death-immigration) process. This is the TKF91
link sequence generator.

**Params:**
```
{'ins_rate': RateParam, 'del_rate': RateParam}
```

**Sufficient stats:**
```
{'bdi': RateSuffStats}  # E[B], E[D], E[S] for this process
```

**Score identity (the key correctness constraint):**
```
∂(log P)/∂λ = E[S]/λ + E[B]/λ - t     (births + survivals contribute to λ)
∂(log P)/∂μ = -E[S]·t·α/(1-α) + ...   (deaths contribute to μ)
```

These are exactly the formulas in `indel.py:recover_bdi_stats`.

### 2. `ctmc_expansion(Q, pi)` — CTMC Expansion

Decorates each link with a character drawn from equilibrium π, evolving under
rate matrix Q. No new model parameters (Q, pi are fixed or trained separately).
Adds substitution sufficient stats (expected substitution counts).

### 3. `fragment_expansion(inner_spec, ext_rate)` — Fragment Expansion

Wraps an inner model spec, adding a geometric fragment length to each link.

**New params:**
```
{'ext_rate': BernoulliParam}  # extension probability r
```

**New sufficient stats:**
```
{'ext': BernoulliSuffStats}  # n_extend, n_exit
```

**M-step:** `r_new = (n_extend + a - 1) / (n_extend + n_exit + a + b - 2)`

### 4. `mixture_expansion(inner_specs, weights)` — Mixture Expansion

Assigns a latent categorical variable to each link, selecting one of K
component inner specs.

**New params:**
```
{'weights': SimplexParam}  # (K,) mixture weights
```

**New sufficient stats:**
```
{'occupancy': SimplexSuffStats}  # posterior mass per component
```

**M-step:** `w_new = (counts + alpha - 1) / sum(counts + alpha - 1)`

### 5. `non_recursive_nesting(outer_spec, inner_spec)` — Non-recursive Nesting

Splices inner grammar into each outer link. This is the MixDom pattern:
outer TKF91 links each contain a full TKF92 process.

When the inner grammar can produce ε (empty), this creates null states.
The builder records this in `null_info` but does NOT remove them — that's
the job of `compile` (see Null Removal below).

### 6. `evolution(single_spec, t)` — Evolution

Converts single-sequence grammar to pair grammar. Triples each link
nonterminal into M/I/D states. This is where the TKF transition matrix τ
appears.

Not a builder in the parameter sense (doesn't add learnable params beyond
what the BDI process already has), but defines how the pair HMM is
constructed from the single-sequence model.

### 7. `recursive_nesting(outer_spec, inner_spec, split_prob)` — Recursive Nesting

For TKFST: mortal links can either emit a character or spawn a bifurcation.

**New params:**
```
{'split_prob': BernoulliParam}  # probability of spawning inner grammar
```

## Null Removal and Restoration (the `compile` step)

This is model-independent post-processing, not part of any elaboration builder.
It applies uniformly to any grammar produced by the builders.

### Two forms of null

1. **Nullable nonterminals (SCFG):** A nonterminal X is nullable if X ⇒* ε.
   Arises from `non_recursive_nesting` (inner grammar can be empty) and
   `recursive_nesting` (nested sequences can be empty).

2. **Null cycles (HMM and SCFG):** Chains of null transitions X → Y → ... → X
   where all intermediate states produce ε. In HMMs, this is the `T_ZZ`
   sub-matrix among null states. In SCFGs, this is unit production cycles
   among nullable nonterminals.

### `compile(grammar_spec) -> compiled_model`

```python
def compile(grammar_spec):
    """Remove all null productions, return DP-ready model."""

    # 1. Compute nullabilities η(X) for all nonterminals
    #    - Non-recursive: closed form (e.g. z_0, z_t from domain params)
    #    - Recursive (TKFST): fixed-point iteration
    null_info = compute_nullabilities(grammar_spec)

    # 2. Remove ε-producing nonterminals
    #    For each rule X → Y·Z, add:
    #      X' → Y'·Z' (weight WF)
    #      X' → Y'    (weight WF·η(Z))  if Z nullable
    #      X' → Z'    (weight WF·η(Y))  if Y nullable
    rules_no_eps = remove_epsilon_nonterminals(grammar_spec, null_info)

    # 3. Remove null cycles via closure
    #    HMM: T_eff = T_NN + T_Nnull · (I - T_ZZ)^{-1} · T_ZN
    #    SCFG: equivalent closure over unit production cycles
    rules_clean, closure_info = remove_null_cycles(rules_no_eps, null_info)

    # 4. Build extract_stats that restores null counts
    def extract_stats(n_counts, trans, params):
        # a. Decompose DP counts into per-component raw counts
        raw_stats = decompose_counts(n_counts, trans, params)

        # b. Restore null cycle counts (the un-elimination)
        restored = restore_null_counts(raw_stats, closure_info, params)

        # c. Restore ε-nonterminal counts (the un-removal)
        full_stats = restore_epsilon_counts(restored, null_info, params)

        # d. Compute sufficient stats from full counts
        return compute_suff_stats(full_stats, params)

    return CompiledModel(
        params=grammar_spec.params,
        build_trans=lambda p: build_from_clean_rules(rules_clean, p),
        extract_stats=extract_stats,
        m_step=build_m_step(grammar_spec),
        dp_type=infer_dp_type(grammar_spec),
    )
```

### Null cycle restoration (HMM case, current `_uneliminate_null_counts`)

Given `top_counts_eff` (5×5 expected counts on the effective T matrix):

1. **Build upsilon** (7×7 exploded matrix with null states M_null, D_null)
2. **Compute closure** `N* = (I - T_ZZ)^{-1}`
3. **Compute null contribution** `C = T_Ω,Z · N* · T_Z,Ω`
4. For each effective transition `T_eff[u,v] = T_NN[u,v] + C[u,v]`:
   - Direct fraction: `n_direct = n_eff[u,v] · T_NN[u,v] / T_eff[u,v]`
   - Null-mediated fraction: `n_null = n_eff[u,v] · C[u,v] / T_eff[u,v]`
5. **Distribute null-mediated counts** through the closure
6. **Map null states to visible:** M_null → M, D_null → D
7. **Sum** direct + null-mediated counts → adjusted counts with phantom BDI events

### Nullability restoration (SCFG case)

The SCFG analog: when a bifurcation X → Y·Z was split into X' → Y' and
X' → Y'·Z', the Inside-Outside counts for X' → Y' must be redistributed
back to X → Y·Z weighted by η(Z). This restores the expected counts for
"events where Z was present but generated ε."

### Key insight: both restorations are inverses of the removal

The removal step is a linear transformation of the rule weight matrix.
The restoration step applies the inverse (or pseudo-inverse) to the count
matrix. This is why it's cleanly separable from the elaboration builders:
the builders define the grammar structure, and null removal/restoration is
a uniform algebraic post-processing step.

## Compositional Model Definitions

Following the paper's notation:

```python
# TKF91 = tkflinks(subproc(Q, pi))
tkf91 = bdi_process(
    ins_rate=RateParam(0.05),
    del_rate=RateParam(0.10),
    inner=ctmc_expansion(Q, pi),
)

# TKF92 = tkflinks(fragproc(subproc(Q, pi); ext))
tkf92 = bdi_process(
    ins_rate=RateParam(0.05),
    del_rate=RateParam(0.10),
    inner=fragment_expansion(
        inner=ctmc_expansion(Q, pi),
        ext_rate=BernoulliParam(0.5),
    ),
)

# MixDom = tkflinks(mixture_dom(tkflinks(mixture_frag(fragproc(
#            mixture_class(subproc(Q_c, pi_c)); ext_f)); λ_d, μ_d)); λ_0, μ_0)
def build_mixdom(n_dom, n_frag, n_class=1):
    # Innermost: site class mixture over substitution models
    site_classes = [ctmc_expansion(Q_c, pi_c) for c in range(n_class)]

    # Per-fragment: fragment expansion with class mixture
    def make_frag(f):
        inner = mixture_expansion(site_classes, SimplexParam(uniform(n_class)))
        return fragment_expansion(inner, BernoulliParam(0.5))

    # Per-domain: TKF92 (bdi over fragments with fragment mixture)
    def make_domain(d):
        frags = [make_frag(f) for f in range(n_frag)]
        return bdi_process(
            ins_rate=RateParam(0.05),
            del_rate=RateParam(0.10),
            inner=mixture_expansion(frags, SimplexParam(uniform(n_frag))),
        )

    # Outer: TKF91 over domains with domain mixture + nesting
    domains = [make_domain(d) for d in range(n_dom)]
    inner = mixture_expansion(domains, SimplexParam(uniform(n_dom)))
    return non_recursive_nesting(
        outer=bdi_process(
            ins_rate=RateParam(0.05),
            del_rate=RateParam(0.10),
        ),
        inner=inner,
    )
```

## The Uniform Correctness Test

For ANY model built by composition:

```python
def test_grad_consistency(model_spec, x_seq, y_seq, t):
    """Verify: autodiff grad == VJP grad from sufficient statistics."""
    params = model_spec.params

    # Method 1: Pure autodiff through transition matrix construction + DP
    def log_prob_fn(flat_params):
        p = unflatten(flat_params, params)
        chi, stypes = model_spec.build_trans(p)
        lp, _ = forward_2d(jnp.log(chi), stypes, x_seq, y_seq, sub_matrix, pi)
        return lp + log_prior(p)
    grad_auto = jax.grad(log_prob_fn)(flatten(params))

    # Method 2: FB counts -> sufficient stats -> score function identity
    chi, stypes = model_spec.build_trans(params)
    log_prob, _, n_chi = forward_backward_2d(
        jnp.log(chi), stypes, x_seq, y_seq, sub_matrix, pi)
    suff_stats = model_spec.extract_stats(n_chi, chi, params)
    grad_vjp = score_function_grad(suff_stats, params) + log_prior_grad(params)

    # These must match
    assert allclose(grad_auto, grad_vjp, rtol=1e-4)
```

This test is the **single source of truth** for correctness. If it passes,
the E-step is correct. If it fails, the sufficient statistics computation
(including null cycle correction) has a bug.

### What `score_function_grad` computes

For each parameter type:

- **RateParam (λ):** `∂log P/∂λ = E[B]/λ + E[S]·(∂log(survival terms)/∂λ) + ...`
  This is exactly `indel_score()` applied to the BDI stats.

- **SimplexParam (w):** `∂log P/∂w_k = occupancy_k / w_k`
  The mixture weight gradient is just the posterior count divided by the weight.

- **BernoulliParam (r):** `∂log P/∂r = n_extend/r - n_exit/(1-r)`
  Standard Bernoulli score.

The log-prior gradients add the usual conjugate terms:
- Gamma: `(shape-1)/λ - rate`
- Dirichlet: `(alpha_k - 1)/w_k`
- Beta: `(a-1)/r - (b-1)/(1-r)`

## Current Bug: Missing Null Count Restoration

The current EM divergence is caused by skipping the null count restoration
step. The `resolve_counts` function decomposes n_chi into per-component
counts correctly at the chi level, but the top-level BDI counts are computed
from the *effective* T matrix (nulls already removed) without restoring the
phantom BDI events from null cycles.

The fix: after `resolve_counts` produces `top_counts` (5×5 on the effective
matrix), pass them through `_uneliminate_null_counts` (already implemented in
`em_mixdom.py`) before computing BDI stats. This is an instance of the general
null count restoration described above.

## Implementation Plan

### Phase 1: Core types and test (immediate)

1. Define `RateParam`, `SimplexParam`, `BernoulliParam` dataclasses
2. Define `RateSuffStats`, `SimplexSuffStats`, `BernoulliSuffStats`
3. Implement `score_function_grad` for each type
4. Write `test_grad_consistency` for TKF91 (simplest case, no null cycles)
5. Verify it passes

### Phase 2: TKF92 and fragment expansion

1. Add `BernoulliParam` for extension rate
2. Extend `score_function_grad` for extension
3. Write `test_grad_consistency` for TKF92
4. Verify

### Phase 3: MixDom with null count restoration

1. Add `SimplexParam` for domain/fragment weights
2. Implement `compile` step: null removal (already done in `effective_trans`)
   + null count restoration (re-enable `_uneliminate_null_counts`)
3. Write `test_grad_consistency` for MixDom
4. **This is where the current bug will surface and be fixed**

### Phase 4: Compositional builders

1. Implement `bdi_process`, `ctmc_expansion`, `fragment_expansion`,
   `mixture_expansion`, `non_recursive_nesting` as composable `GrammarSpec` builders
2. Implement `compile` as a uniform post-processing step
3. Verify that `compile(build_mixdom())` produces the same `chi` matrix as
   existing `build_nested_trans()`
4. Verify that the composed `extract_stats` (with null restoration) passes
   the uniform test

### Phase 5: EM loop and optimizers

1. Single uniform `em_step(model_spec, data)` that works for any model
2. Single uniform `adam_step(model_spec, data)` using VJP
3. Re-run 5-method comparison with correct BDI stats

## Relationship to Existing Code

| Existing | Framework equivalent |
|----------|---------------------|
| `build_nested_trans()` | `compiled.build_trans(params)` |
| `effective_trans()` | `compile` step: null cycle removal |
| `resolve_counts()` | `compiled.extract_stats` — count decomposition part |
| `_uneliminate_null_counts()` | `compile` step: null count restoration (inverse of removal) |
| `m_step_top_level()` | `compiled.m_step(suff_stats)` — BDI closed form |
| `m_step_indel()` | `compiled.m_step(suff_stats)` — domain-level BDI |
| `mixdom_log_prob()` in vjp.py | `compiled.build_trans + forward_2d` (HMM) or `inside` (SCFG) |
| `score_derivatives()` | `score_function_grad` for RateParam |
| `transition_count_groups()` | Subsumed by `extract_stats` |
| `recover_bdi_stats()` | Part of BDI RateSuffStats extraction |
| `nullability()` | `compile` step: compute_nullabilities |
| `forward_backward_2d()` | DP engine for `dp_type='hmm'` |
| (future) `inside_outside()` | DP engine for `dp_type='scfg'` (TKFST, TKFBS) |

The framework doesn't replace the existing code immediately — it wraps it.
The existing optimized functions (`build_nested_trans`, `forward_backward_2d`)
remain the computational core. The framework provides the correct
grammar→compile→DP→restore→suffstats pipeline on top.

## File & Module Reorganization

The current 45 modules grew organically. The new structure reflects the
architecture: grammar elaboration → compilation → DP → training, with a
clear split between left-regular (HMM) and context-free (SCFG) models.

### Current layout → Proposed layout

```
tkfmixdom/jax/                    # CURRENT (flat, 45 files)
├── dp.py                         # HMM DP (forward, backward, viterbi, banded)
├── beam.py                       # beam-pruned HMM DP
├── grammar.py                    # WCFG framework
├── grammar_dp.py                 # JAX-parallelized WCFG DP
├── factored_dp.py                # factored SCFG×WPTT DP
├── beam_scfg.py                  # beam-pruned phylo Inside
├── hmm.py                        # TKF91/92 Pair HMM construction
├── mixdom.py                     # MixDom nested Pair HMM
├── tkf_grammar.py                # TKF elaboration grammars
├── tkfst_grammar.py              # TKFStack pair SCFG
├── rna_grammar.py                # RNA stem-loop SCFG
├── order1_scfg.py                # order-1 singlet SCFG
├── params.py                     # TKF transition matrices
├── indel.py                      # BDI parameters, score derivatives
├── subst.py                      # reversible CTMC
├── subst_irreversible.py         # irreversible CTMC
├── protein_models.py             # WAG, LG matrices
├── vjp.py                        # custom VJP wrappers
├── em_tkf.py                     # EM for TKF91/92
├── em_mixdom.py                  # EM for MixDom
├── adam_mixdom.py                # Adam for MixDom
├── constrained.py                # alignment-constrained training
├── optimizer.py                  # abstract EM→LBFGS
├── distill.py                    # HMM distillation
├── scfg_distill.py               # SCFG distillation
├── wptt.py                       # WPTT transducer
├── scfg_compose.py               # SCFG composition
├── intersect.py                  # sibling intersection
├── compose_wptt_rec.py           # WPTT×recognizer
├── progressive.py                # progressive reconstruction
├── profile_compress.py           # profile SCFG compression
├── recognizer.py                 # leaf recognizer
├── rna_context.py                # RNA basepair context
├── ancestor.py                   # ancestral reconstruction
├── tree.py                       # Felsenstein pruning
├── tree_transducer.py            # parse tree transducer
├── evolve.py                     # simulation
├── simulate.py                   # TKF simulation
├── fit.py                        # pairwise fitting
├── msa.py                        # MSA likelihood
├── io.py                         # FASTA/Stockholm/Newick I/O
├── data.py                       # data download
├── timing.py                     # benchmarking
└── ...
```

```
tkfmixdom/                        # PROPOSED (organized by concern)
├── core/                         # Foundational types and algorithms
│   ├── types.py                  # RateParam, SimplexParam, BernoulliParam,
│   │                             # RateSuffStats, SimplexSuffStats, BernoulliSuffStats,
│   │                             # GrammarSpec, CompiledModel, GrammarClass
│   ├── ctmc.py                   # CTMC: rate_matrix, transition_matrix, pi,
│   │                             # holmes_rubin, log_prior  (was: subst.py + subst_irreversible.py)
│   ├── bdi.py                    # BDI process: alpha, beta, gamma, kappa,
│   │                             # score_derivatives, recover_bdi_stats
│   │                             # (was: indel.py + parts of params.py)
│   ├── protein.py                # WAG, LG rate matrices (was: protein_models.py)
│   └── rna.py                    # basepair contexts, canonical pairs
│                                 # (was: rna_context.py)
│
├── grammar/                      # Grammar framework and compilation
│   ├── scfg.py                   # Production, SCFG classes, inside/outside/CYK
│   │                             # (was: grammar.py — renamed from WCFG to SCFG)
│   ├── elaborate.py              # bdi_process, ctmc_expansion, fragment_expansion,
│   │                             # mixture_expansion, non_recursive_nesting,
│   │                             # recursive_nesting, evolution
│   │                             # (NEW — the compositional builders)
│   └── compile.py                # compile(): nullability computation,
│                                 # ε-removal, null cycle removal,
│                                 # null count restoration, grammar class detection
│                                 # (NEW — was scattered across mixdom.py, em_mixdom.py)
│
├── models/                       # Model-specific grammar definitions
│   ├── left_regular.py           # Left-regular (HMM) models:
│   │                             # TKF91, TKF92, MixDom pair HMM construction
│   │                             # (was: tkf_grammar.py + hmm.py + mixdom.py + params.py)
│   └── context_free.py           # Context-free (SCFG) models:
│                                 # TKFStack/TKFST, TKFBS pair SCFG construction
│                                 # (was: tkfst_grammar.py + rna_grammar.py)
│
├── dp/                           # Dynamic programming engines
│   ├── hmm.py                    # Forward, Backward, FB, Viterbi, sampling (1D and 2D)
│   │                             # (was: dp.py)
│   ├── hmm_beam.py               # Beam-pruned HMM DP with MSA envelope
│   │                             # (was: beam.py)
│   ├── hmm_banded.py             # Banded HMM DP  (was: dp.py banded functions)
│   ├── scfg.py                   # Inside, Outside, IO, CYK for SCFGs
│   │                             # (was: grammar_dp.py + parts of grammar.py)
│   ├── scfg_beam.py              # Beam-pruned phylo Inside for SCFGs
│   │                             # (was: beam_scfg.py)
│   └── scfg_factored.py          # Factored SCFG×WPTT DP
│                                 # (was: factored_dp.py)
│
├── train/                        # Training algorithms (model-generic)
│   ├── em.py                     # Uniform EM loop: e_step → extract_stats
│   │                             # → restore_null → m_step
│   │                             # (was: em_tkf.py + em_mixdom.py, unified)
│   ├── adam.py                   # Uniform Adam loop using VJP
│   │                             # (was: adam_mixdom.py, generalized)
│   ├── vjp.py                    # Custom VJP: log_posterior with FB/IO fwd,
│   │                             # score function bwd. Returns
│   │                             # log P(data|θ)·P(θ) = log-lik + log-prior
│   │                             # (was: vjp.py, generalized)
│   ├── constrained.py            # Alignment-constrained E-step
│   │                             # (was: constrained.py)
│   └── fit.py                    # Pairwise fitting, branch length
│                                 # (was: fit.py)
│
├── distill/                      # Distillation (structured → order-1 → composable)
│   ├── hmm.py                    # Structured HMM → order-1 HMM
│   │                             # (e.g. MixDom → order-1 Pair HMM)
│   │                             # (was: distill.py)
│   ├── wfst.py                   # Order-1 HMM → WFST for tree composition
│   │                             # (parallel to wptt.py for SCFGs)
│   ├── scfg.py                   # Structured SCFG → order-1 SCFG
│   │                             # (e.g. TKFST → order-1 pair SCFG)
│   │                             # (was: scfg_distill.py)
│   └── wptt.py                   # Order-1 SCFG → WPTT for tree composition
│                                 # (was: wptt.py)
│
├── tree/                         # Phylogenetic tree algorithms
│   ├── compose.py                # SCFG composition, WPTT×recognizer,
│   │                             # WFST composition, sibling intersection
│   │                             # (was: scfg_compose.py + compose_wptt_rec.py
│   │                             #  + intersect.py)
│   ├── progressive.py            # Progressive alignment/reconstruction
│   │                             # (was: progressive.py + profile_compress.py)
│   ├── ancestor.py               # Ancestral reconstruction
│   │                             # (was: ancestor.py)
│   ├── felsenstein.py            # Felsenstein pruning, tree likelihood
│   │                             # (was: tree.py)
│   ├── recognizer.py             # Leaf recognizer construction
│   │                             # (was: recognizer.py)
│   └── transducer.py             # Parse tree transducer
│                                 # (was: tree_transducer.py)
│
├── simulate/                     # Simulation and evolution
│   ├── evolve.py                 # Stochastic sequence evolution (all models)
│   │                             # (was: evolve.py + simulate.py)
│   └── msa.py                    # MSA likelihood, column likelihood
│                                 # (was: msa.py)
│
└── util/                         # I/O, data, timing
    ├── io.py                     # FASTA, Stockholm, Newick parsers
    │                             # (was: io.py)
    ├── data.py                   # Database fetchers (Pfam, Rfam, BaliBase)
    │                             # (was: data.py)
    └── timing.py                 # Benchmark infrastructure
                                  # (was: timing.py)
```

### Naming principles

1. **Module names reflect concern, not model.**
   `dp/hmm.py` not `dp.py`; `dp/scfg.py` not `grammar_dp.py`.

2. **Models are classified by grammar type, not biological application.**
   `models/left_regular.py` (TKF91, TKF92, MixDom — all produce HMMs).
   `models/context_free.py` (TKFST, TKFBS — all produce SCFGs).

3. **Grammar framework uses standard terminology.**
   `grammar/scfg.py` not `grammar/wcfg.py`. (Left-regular grammars are a
   special case of SCFGs; the same SCFG framework handles both.)

4. **Training code is model-generic.**
   `train/em.py` works with any `CompiledModel`. No `em_tkf.py` vs `em_mixdom.py`.

5. **HMM vs SCFG parallelism in DP and distillation.**
   `dp/hmm.py` ∥ `dp/scfg.py`.
   `distill/hmm.py` ∥ `distill/scfg.py` (structured → order-1).
   `distill/wfst.py` ∥ `distill/wptt.py` (order-1 → tree-composable).

6. **`core/` has no model-specific code.** Just parameter types, CTMC, BDI.

### Key function renames

| Current | Proposed | Rationale |
|---------|----------|-----------|
| `forward_backward_2d()` | `dp.hmm.forward_backward()` | model-generic |
| `build_nested_trans()` | `models.left_regular.build_mixdom_trans()` | classified by grammar |
| `effective_trans()` | internal to `grammar.compile` | null removal detail |
| `resolve_counts()` | internal to `train.em` | count decomposition |
| `_uneliminate_null_counts()` | `grammar.compile.restore_null_counts()` | public, testable |
| `make_tkf91_pair_hmm()` | `models.left_regular.tkf91_pair_hmm()` | classified by grammar |
| `tkf91_trans()` | `core.bdi.tkf91_trans()` | foundational |
| `score_derivatives()` | `core.bdi.score_derivatives()` | foundational |
| `inside_jax()` | `dp.scfg.inside()` | parallel to dp.hmm |
| `distill_pair_hmm()` | `distill.hmm.distill_pair()` | structured → order-1 |
| `distill_pair_scfg()` | `distill.scfg.distill_pair()` | structured → order-1 |
| `build_wptt_transducer()` | `distill.wptt.build_transducer()` | order-1 → composable |
| `compose_generator_transducer()` | `tree.compose.generator_transducer()` | namespaced |
| `tkf91_log_prob()` | `train.vjp.log_posterior()` | log P(data|θ)·P(θ) |

### Test file renames

Tests mirror the module structure:

```
tests/
├── core/
│   ├── test_ctmc.py              # (was: test_subst.py + test_subst_irreversible.py)
│   ├── test_bdi.py               # (was: test_indel.py + test_params.py)
│   └── test_protein.py           # (was: test_protein_models.py)
├── grammar/
│   ├── test_elaborate.py         # NEW: test compositional builders
│   ├── test_compile.py           # NEW: test null removal/restoration
│   │                             # (includes golden-reference tests vs closed-form)
│   └── test_scfg.py              # (was: test_grammar.py)
├── models/
│   ├── test_left_regular.py      # (was: test_tkf_grammar.py + test_mixdom.py)
│   └── test_context_free.py      # (was: test_tkfst_grammar.py + test_rna_grammar.py
│                                 #  + test_rna_scfg.py)
├── dp/
│   ├── test_hmm.py               # (was: test_dp.py)
│   ├── test_hmm_beam.py          # (was: test_beam.py)
│   ├── test_scfg.py              # (was: test_grammar_dp.py)
│   ├── test_scfg_beam.py         # (was: test_beam_scfg.py)
│   └── test_scfg_factored.py     # (was: test_factored_dp.py)
├── train/
│   ├── test_em.py                # (was: test_em.py + test_em_mixdom.py)
│   ├── test_vjp.py               # (was: test_vjp.py)
│   ├── test_grad_consistency.py  # (was: test_bw_vs_grad.py — the uniform test)
│   ├── test_constrained.py       # (was: test_cross_validation.py)
│   └── test_fit.py               # (was: test_fit.py)
├── distill/
│   ├── test_hmm.py               # (was: test_distill.py)
│   ├── test_wfst.py              # NEW: WFST composition tests
│   ├── test_scfg.py              # (was: test_scfg_distill.py)
│   └── test_wptt.py              # (was: test_wptt.py)
├── tree/
│   ├── test_compose.py           # (was: test_scfg_compose.py + test_intersect.py +
│   │                             #  test_compose_wptt_rec.py)
│   ├── test_progressive.py       # (was: test_progressive.py)
│   ├── test_ancestor.py          # (was: test_ancestor.py)
│   └── test_felsenstein.py       # (was: test_tree.py)
├── simulate/
│   └── test_evolve.py            # (was: test_evolve.py)
├── statistical/                  # unchanged
│   ├── test_bdi_stats.py
│   ├── test_holmes_rubin.py
│   ├── test_alignment_recovery.py
│   └── test_baum_welch.py
└── conftest.py
```

## Type Annotations

All functions in the refactored codebase must have full type annotations.

### Conventions

- Use `jax.Array` (not `jnp.ndarray`) for JAX arrays
- Use `numpy.typing.NDArray[np.float64]` for NumPy arrays
- Annotate shapes in docstrings: `x: jax.Array  # (L,) integer sequence`
- Use `TypeAlias` for common compound types
- Use `@dataclass` (not NamedTuple) for structured returns
- Use `Protocol` for duck-typed interfaces (e.g. `ModelSpec`)

```python
from typing import Protocol, TypeAlias
import jax
import jax.numpy as jnp

Seq: TypeAlias = jax.Array          # (L,) integer sequence
TransMatrix: TypeAlias = jax.Array  # (N, N) transition matrix
StateTypes: TypeAlias = jax.Array   # (N,) state type indicators

class CompiledModel(Protocol):
    params: dict
    dp_type: str  # 'hmm' | 'scfg'
    def build_trans(self, params: dict) -> tuple[TransMatrix, StateTypes]: ...
    def extract_stats(self, n_counts: jax.Array, trans: TransMatrix,
                      params: dict) -> dict: ...
    def m_step(self, suff_stats: dict, params: dict) -> dict: ...
```

### Scope

- Phase 1: Add types to all new framework code (`core/`, `grammar/`, `models/`)
- Phase 2: Add types to refactored existing code as files move
- Do NOT add types to files that aren't being touched in the refactor

## Integration of Wideboy's TKFST Work

Wideboy's final commit on `main` (c71d531) adds:

1. **Alignment-constrained pair SCFG** — Replaces 4D unconstrained
   `tkfst_pair_inside/outside` with 1D constrained versions. These go in
   `models/context_free.py` (alignment-constrained pair SCFG construction).

2. **TKFST ModelSpec** — `tkfst_e_step`, `tkfst_m_step`, `tkfst_em_optimize`.
   These become golden-reference tests for `train/em.py` + `grammar/compile.py`.
   The hand-rolled EM must match the generic framework's results.

3. **Multi-pair distillation** — `aggregate_distill_stats()` sums IO emission
   counts across pairs. Goes in `distill/scfg.py`.

4. **WPTT epsilon fix** — `tkf_wptt_weights()` p_end floor for kappa≈1.
   Goes in `distill/wptt.py`. Essential bugfix.

5. **End-to-end RNA pipeline** — `exp8_rna_pipeline.py`. Stays in `experiments/`.

## Repository Cleanup

### Files to DELETE

**Inter-agent coordination (entire directory):**
```
python/experiments/.agent_coordination/           # all 6 files
├── InterAgentCommunication.md
├── README.md
├── lambda.status.json
├── wideboy.status.json
└── messages/
    ├── lambda_to_wideboy.md
    └── wideboy_to_lambda.md
```

**Superseded documentation (subsumed by .tex files or code):**
```
math-review-Holmes_Rubin_2002.md     # subsumed by Holmes_Rubin_2002.tex
math-review-tkf.md                   # subsumed by tkf.tex
math-review-lhopital-limits.md       # subsumed by lhopital-limits.tex
ParseTreeTransducers.md              # subsumed by grammar-elaboration.tex + wptt.py
```
Note: the math-verifier agent's MEMORY.md (`.claude/agent-memory/math-verifier/`)
records known errors found in .tex files. These findings are valuable and
should be retained until the errors are fixed in the .tex sources.

**Agent/session ephemera:**
```
python/GPU_AGENT_INSTRUCTIONS.md     # agent setup, obsolete
python/HANDOFF.md                    # session handoff, ephemeral
python/SESSION_NOTES.md              # session notes, ephemeral
python/GuidanceNeeded.md             # all items resolved
python/experiments/bio_test_plan.md  # subsumed by ROADMAP.md Phase 7
```

### Files to KEEP

```
CLAUDE.md                            # project guidance (update during refactor)
README.md                            # project overview (update during refactor)
ROADMAP.md                           # project phases (update to reflect new plan)
${CLAUDE_AGENTS_PATH}      # useful tooling
.claude/agent-memory/math-verifier/  # records of known .tex errors
```

### ROADMAP.md update

ROADMAP.md should be updated to reflect the current state:
- Phases 1-7: COMPLETE (as marked)
- Phase 8: Replace current content with the framework refactor plan
- Remove experiment-specific detail (that lives in experiment scripts now)

### CLAUDE.md update

CLAUDE.md should be updated during the refactor to reflect:
- New module structure (replace the flat `python/tkfmixdom/jax/` references)
- New build/test commands (if paths change)
- Remove references to two-agent workflow

### .gitignore additions

```
__pycache__/
*.pyc
.pytest_cache/
```

## Refactor Execution Plan

### Step 0: Merge and clean — DONE

- Merged `protein-benchmarks` into `main`, deleted 15 cleanup targets
- Commit: 898fdc2

### Step 1: Create directory structure — DONE

- 44 modules moved to `core/`, `grammar/`, `models/`, `dp/`, `train/`,
  `distill/`, `tree/`, `simulate/`, `util/`
- 103 files had imports updated
- 800+ tests pass
- Commit: ebfa0b1

### Step 2: Implement core framework types — DONE

- `core/types.py`: RateParam, SimplexParam, BernoulliParam, BDISuffStats,
  NullInfo, GrammarSpec, CompiledModel Protocol
- `grammar/compile.py`: build_null_info_hmm, effective_trans_from_null_info,
  restore_null_counts
- Verified: generic compile matches existing effective_trans for MixDom
- Note: `grammar/elaborate.py` deferred (compositional builders are a
  future step — existing model-specific builders work correctly)

### Step 3: Uniform gradient test — DONE

- `tests/test_grad_consistency.py`: 18 tests
  - TKF91: 7 tests (fixture + grad match + BDI conservation + 5 seeds)
  - TKF92: 3 tests (3 seeds)
  - MixDom: 5 tests (3 VJP-vs-autodiff + 2 null restoration)
  - Golden reference: 3 tests (generic compile matches hand-derived)
- All 18 pass. MixDom autodiff required `jnp.maximum(chi, 1e-30)` clamp.

### Step 4: Unify training — DONE

- `models/compiled.py`: concrete CompiledModel implementations
  (TKF91Model, TKF92Model, MixDomModel)
- `train/em.py`: generic EM (em_single_pair, em_aggregate) using
  CompiledModel interface, delegating to optimizer.py
- `tests/test_compiled_model.py`: 5 golden-reference tests
  (e_step matches, gradient matches, EM runs)
- Existing specialized em_tkf.py / em_mixdom.py retained
  (they have L-BFGS, safeguards, conditioned mode, etc.)
- train/vjp.py and train/adam_mixdom.py already existed

### Step 5: Add types, clean up old code — DONE

- Type annotations added to optimizer.py, em.py, compiled.py, types.py
- CLAUDE.md updated with new module structure and test commands
- conftest.py updated with file-based progress logger and new test tiers
- Old bespoke code retained (not superseded — specialized features needed)
