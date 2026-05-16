# tkfmixdom JAX API Reference

## Overview

The `tkfmixdom.jax` package implements the TKF family of evolutionary sequence models
in JAX, with differentiable DP algorithms, exact Baum-Welch EM, and phylogenetic inference.

### Package Structure

```
tkfmixdom/jax/
├── core/       # Parameters, rate matrices, BDI statistics
├── models/     # Model constructors (TKF91, TKF92, MixDom, TKFST)
├── dp/         # DP algorithms (Forward, Viterbi, FB, Inside-Outside)
├── grammar/    # WCFG/SCFG framework, elaboration rules
├── train/      # EM training, custom VJPs, optimizers
├── distill/    # Order-1 HMM/WFST/WPTT distillation
├── tree/       # Phylogenetic algorithms (ProgRec, Beam MSA)
├── simulate/   # Sequence simulation
└── util/       # I/O, data loading, timing
```

---

## 1. Core (`core/`)

### BDI Parameters (`core/bdi.py`)

TKF91 Birth-Death-Immigration parameters from indel rates (λ, μ) and time t:

| Function | Returns | L'Hôpital? |
|----------|---------|------------|
| `tkf_alpha(μ, t)` | α = e^{-μt} (survival prob) | No |
| `tkf_beta(λ, μ, t)` | β (offspring param) | Yes, β → s/(1+s) |
| `tkf_gamma(λ, μ, t)` | γ (post-delete insert) | Yes |
| `tkf_kappa(λ, μ)` | κ = λ/μ (stationarity ratio) | No |
| `score_derivatives(λ, μ, t)` | 6 elasticities ∂log(ξ)/∂λ, ∂log(ξ)/∂μ | Yes |
| `indel_score(n_trans, λ, μ, t)` | ∂ℓ/∂λ, ∂ℓ/∂μ from count matrix | Yes |
| `recover_bdi_stats(dλ, dμ, λ, μ, t, i, j)` | E[B], E[D], E[S] | Yes |

All functions gate to L'Hôpital limit formulas when |1-κ| < 10⁻⁴.

### Transition Matrices (`core/params.py`)

| Function | Returns | States |
|----------|---------|--------|
| `tkf91_trans(λ, μ, t)` | 5×5 TKF91 Pair HMM matrix | S, M, I, D, E |
| `tkf92_trans(λ, μ, t, ext)` | 5×5 TKF92 with fragment extension | S, M, I, D, E |
| `tkf91_trans_cond(λ, μ, t)` | 5×5 conditioned (κ factored out) | S, M, I, D, E |

### CTMC (`core/ctmc.py`)

| Function | Purpose |
|----------|---------|
| `rate_matrix_jc69(n)` | JC69 n-state rate matrix |
| `transition_matrix_with_pi(Q, π, t)` | P(t) = exp(Qt) |
| `holmes_rubin_expected_stats(Q, π, t, a, b)` | Expected dwell times & transition counts for a→b |

### Typed Sufficient Statistics (`core/types.py`)

| Type | Conjugate Prior | M-step |
|------|----------------|--------|
| `BDISuffStats(E_B, E_D, E_S, n_κ, n_{1-κ})` | Gamma(α,β) on rates | λ̂ = E[B]/E[S], μ̂ = E[D]/E[S] |
| `SimplexSuffStats(counts)` | Dirichlet(α) | ŵ ∝ counts + α - 1 |
| `BernoulliSuffStats(n_succ, n_fail)` | Beta(a,b) | p̂ = (n_succ+a-1)/(total+a+b-2) |

---

## 2. Models (`models/`)

### Model Hierarchy

| Model | States | Parameters | Pair HMM | Singlet HMM |
|-------|--------|------------|----------|-------------|
| TKF91 | 5 (S,M,I,D,E) | λ, μ, Q, π | `tkf91_trans` | geometric(κ) × π |
| TKF92 | 5 | λ, μ, ext, Q, π | `tkf92_trans` | geometric(κ) × geometric(ext) × π |
| MixDom | 2+5KF | λ₀, μ₀, {λ_k, μ_k, v_k, w_{kf}, ext_{kf}} | `build_nested_trans` | distilled order-1 |
| TKFST | ~49 NTs | stem/loop rates, structural weights | pair SCFG | singlet SCFG |

### MixDom (`models/mixdom.py`)

| Function | Purpose |
|----------|---------|
| `build_nested_trans(λ₀, μ₀, t, λ_k, μ_k, v, w, ext)` | Build collapsed (2+5KF)×(2+5KF) χ matrix |
| `effective_trans(...)` | 5×5 effective top-level T_eff |
| `nullability(λ_k, μ_k, v, t)` | z₀, z_t (domain empty probs) |
| `state_types(K, F)` | State type array (M/I/D per compound state) |

### Exact Sufficient Statistics (`models/exact_suffstats.py`)

```python
exact_suffstats(n_chi, λ₀, μ₀, t, λ_k, μ_k, v, w, ext) → {
    'top_5x5':     (5,5)    # top-level TKF91 count matrix
    'dom_M_5x5':   [(5,5)]  # per-domain M-type TKF91 counts
    'dom_kappa':    (K,)     # I/D-type κ continuation counts
    'dom_1mkappa':  (K,)     # I/D-type 1-κ termination counts
    'ext':          (K,F)    # fragment extension counts
    'term':         (K,F)    # fragment termination counts
    'dom_w':        (K,)     # domain weight counts
    'frag_w':       (K,F)    # fragment weight counts
}
```

Uses the 6-step null elimination chain. Verified against autodiff to 10⁻¹⁴.

### CompiledModel Classes (`models/compiled.py`)

Uniform interface for all models:

```python
class TrainableModel:
    def e_step(params, x, y) → (log_prob, n_trans, posteriors)
    def extract_stats(n_trans, params) → stats_dict
    def m_step(stats, params) → new_params
```

| Class | e_step | extract_stats | m_step |
|-------|--------|---------------|--------|
| `TKF91Model` | 2D FB | count_groups + BDI | Exact closed-form |
| `TKF92Model` | 2D FB | TKF91 decomp + Bernoulli | Exact BDI + Beta |
| `MixDomModel` | 2D FB | `exact_suffstats` (6-step chain) | Exact all groups |
| `TKFSTModel` | IO on pair SCFG | factor count mapping | Exact conjugate |

---

## 3. DP Algorithms (`dp/`)

### HMM DP (`dp/hmm.py`)

| Function | Complexity | Notes |
|----------|-----------|-------|
| `forward_2d(log_trans, st, x, y, sub, π)` | O(LxLyN²) | Anti-diagonal wavefront |
| `backward_2d(...)` | O(LxLyN²) | Same wavefront |
| `forward_backward_2d(...)` | O(LxLyN²) | Returns (log_prob, posteriors, n_trans) |
| `viterbi_2d(...)` | O(LxLyN²) | With traceback |
| `sample_traceback_2d(...)` | O(LxLy) | Stochastic Forward traceback |
| `forward_1d_associative(...)` | O(L log L) | Associative scan, supports `seq_length` padding |
| `forward_backward_1d_associative(...)` | O(L log L) | Full FB with expected counts |
| `forward_backward_1d_padded(...)` | O(L log L) | Padded for JIT cache reuse |
| `safe_log(x)` | O(1) | Maps zeros to NEG_INF (-10³⁰) |

Geometric padding: `_pad_to_bin(L)` rounds L to [4,6,8,12,16,24,...,8192].

### Beam HMM (`dp/hmm_beam.py`)

Beam-pruned Forward-Backward for Pair HMMs with MSA envelope banding.

### SCFG DP (`dp/scfg_beam.py`, `dp/scfg_factored.py`)

Phylogenetic Inside-Outside via WPTT state pushing. Factored Inside-Outside
for SCFG×WPTT composition.

---

## 4. Training (`train/`)

### Custom VJP Wrappers (`train/vjp.py`)

Differentiable log-likelihoods using the score function identity:
d(log P)/d(θ) = Σᵢⱼ E[nᵢⱼ] · d(log τᵢⱼ)/d(θ)

Forward pass: run DP. Backward pass: autodiff through transition matrix construction only.

| Function | Model | Differentiable w.r.t. | Mode |
|----------|-------|----------------------|------|
| `tkf91_log_prob(λ, μ, t, Q, π, x, y)` | TKF91 | λ, μ | Joint |
| `tkf91_log_prob_cond(...)` | TKF91 | λ, μ | Conditional |
| `tkf92_log_prob(λ, μ, t, ext, Q, π, x, y)` | TKF92 | λ, μ, ext | Joint |
| `mixdom_log_prob(λ₀, μ₀, t, λ_k, μ_k, v, w, ext, Q, π, x, y)` | MixDom | All structural | Joint |
| `_chi_weighted_loglik(...)` | MixDom | All structural | Q-function |

### EM Training

| Function | Model | M-step | Exact? |
|----------|-------|--------|--------|
| `em_tkf.em_loop(pairs, Q, π, t, ...)` | TKF91 | Closed-form BDI | Yes |
| `em_tkf.em_loop_tkf92(...)` | TKF92 | Gradient + Newton | Partial |
| `em_mixdom.em_loop(pairs, Q, π, t, ..., exact=True)` | MixDom | `exact_suffstats` | Yes |
| `em_mixdom.em_loop_constrained(aligned_pairs, ..., exact=True)` | MixDom | `exact_suffstats` | Yes |
| `em.em_single_pair(model, params, x, y)` | Any | model.m_step() | Model-dependent |
| `em.em_aggregate(model, params, pairs)` | Any | model.m_step() | Model-dependent |

### Pfam Training (`train_pfam.py`)

Production training script for MixDom on Pfam Stockholm MSAs.

Features:
- Cherry pair selection (greedy nearest-neighbor by p-distance)
- Pair manifest caching (compact metadata, computed once)
- Streaming (one MSA at a time, constant memory)
- Padded 1D FB with geometric bins for JIT cache reuse
- Exact M-steps via `exact_suffstats`
- Reversibility enforcement (both pair directions)
- Auto-checkpoint/resume

### Maraschino Training (`maraschino.py`)

CherryML-like distillation training with four modes:

| Mode | Input | Output | Method |
|------|-------|--------|--------|
| `count` | Stockholm MSAs | Adjacency counts (.npz) | Cherry pairs + column counting |
| `fit` | Counts | MixDom parameters (.npz) | Adam or L-BFGS on LL |
| `distill` | Parameters | Order-1 WFST (JSON) | Algebraic (Woodbury identity) |
| `fetch` | Family IDs | Stockholm files | InterPro/Pfam download |

---

## 5. Distillation (`distill/`)

| Module | Purpose |
|--------|---------|
| `hmm.py` | Order-1 HMM/WFST distillation (adjacency frequencies) |
| `maraschino.py` | Algebraic MixDom distillation (Woodbury identity) |
| `scfg.py` | Order-1 SCFG distillation with 4-position context |
| `wptt.py` | 246-state Weighted Parse Tree Transducer |

---

## 6. Grammar Framework (`grammar/`)

### WCFG/SCFG (`grammar/scfg.py`)

| Algorithm | Function | Complexity |
|-----------|----------|-----------|
| Inside | `inside(grammar, seq)` | O(n³) |
| Outside | `outside(grammar, seq, inside_table)` | O(n³) |
| CYK | `cyk(grammar, seq)` | O(n³) |
| Epsilon elimination | `epsilon_eliminate(grammar)` | Fixed-point |
| Unary closure | `unary_closure(grammar)` | (I-W)⁻¹ |

### Grammar Elaboration (`models/tkf_grammar.py`)

Seven elaboration rules construct TKF grammars:
1. Link (TKF91 transitions)
2. CTMC expansion (substitution model)
3. Fragment (geometric extension)
4. Mixture (domain/fragment/class)
5. Concatenation
6. Non-recursive nesting
7. Recursive nesting

---

## 7. Tree Algorithms (`tree/`)

### Progressive Reconstruction (ProgRec)

| Module | Purpose |
|--------|---------|
| `progrec_felsenstein.py` | Profile-based ProgRec with Felsenstein CL |
| `progrec_dag_dp.py` | DAG-aware Viterbi/Forward |
| `progrec_recognizer.py` | DAG recognizer construction + compression |

### Beam MSA

| Module | Purpose |
|--------|---------|
| `beam_msa.py` | Beam-pruned MSA likelihood |
| `guide_tree.py` | TKF92 distance-based NJ guide tree |

### Phylogenetic SCFG

| Module | Purpose |
|--------|---------|
| `compose_wptt_rec.py` | WPTT × Recognizer composition |
| `recognizer.py` | Leaf recognizer (span-based) |
| `intersect.py` | Sibling recognizer intersection |
| `progressive.py` | Progressive reconstruction pipeline |

---

## 8. Key Design Patterns

### Score Function Identity (Custom VJP)

All HMM-based models use the score function identity for efficient gradients:

```
Forward pass:  log P = Forward algorithm (expensive DP)
Backward pass: d(log P)/dθ = Σ E[n_ij] · d(log τ_ij)/dθ
               (cheap: autodiff through matrix construction only)
```

This avoids propagating gradients through the O(L²N²) DP computation.

### Exact Null-State Count Restoration (MixDom)

The collapsed MixDom Pair HMM has null states eliminated via (I-T_ZZ)⁻¹.
The 6-step elimination chain restores exact expected counts on the fully
exploded model:

| Step | Eliminated | T_ZZ | Closure |
|------|-----------|------|---------|
| 1 | FragType | 0 | I |
| 2 | FragEnd | 0 | I |
| 3 | Frag | 0 | I |
| 4 | DomType | 0 | I |
| 5 | Dom | 0 | I |
| 6 | DomEnd | R·diag(z) | 3×3 analytic |

The vectorized ghost-usage formula at Step 6:
```
n_KZ = T_KZ ⊙ (Scale · H^T)
n_ZK = T_ZK ⊙ (Ct^T · Scale)
n_ZZ = T_ZZ ⊙ (Ct^T · Scale · H^T)
```

### Geometric Padding for JIT Cache Reuse

All DP functions pad sequences to geometric bin sizes [4,6,8,...,8192]
so JAX reuses compiled XLA programs instead of recompiling per length.
Typically saves ~9s per new length on first EM iteration.
