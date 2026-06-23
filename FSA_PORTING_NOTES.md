# Porting the FSA (TKF92) aligner — pointer + analytic τ-derivatives

Handoff note for the TS/Rust port. You won't have autodiff, so the Newton
step on evolutionary time τ must use **analytic** first/second derivatives.
Those are worked out and verified at the bottom — that's the part that isn't
just "read the file".

The reference (JAX) lives in **`tkf-mixdom`**:

> `python/tkfmixdom/jax/tree/fsa_anneal.py`

Everything below cites that repo.

---

## 1. What the algorithm does (and the 2-FB-pass invariant)

Per *selected* sequence pair (pair selection: `select_pairs_full`,
`select_pairs_erdos_renyi`, lines ~1303–1374):

1. **FB pass #1** at `tau_init` (=1.0) → expected transition counts `N` (5×5)
   and expected match-emission counts. `forward_backward_2d` (from
   `..dp.hmm`) returns `(log_prob, posteriors, n_trans)`.
2. Reduce to sufficient statistics: `N` = `n_trans` (5×5), and
   `W` (A×A) = expected match counts by (ancestor residue, descendant
   residue). See `match_W_fixed = einsum('ij,ia,jb->ab', match_post, X_oh, Y_oh)`,
   lines ~234–239.
3. **Newton–Raphson on τ** (fixed iteration count, default 5) maximizing the
   posterior-expected complete-data log-likelihood `Q(τ)` defined below.
   Reference uses autodiff grad/Hess (`_tkf92_tau_grad`, `_tkf92_tau_hess`).
   **You replace this with the closed forms in §4.**
4. **FB pass #2** at the optimized τ → posterior residue-alignment
   probabilities `P(i ~ j)` (lines ~253–263).

So exactly **two FB passes per pair**. Core function:
`_pairwise_posteriors_tkf92_jax` (lines ~212–264).

Then **sequence annealing / AMAP**: `sequence_annealing` (~1499–1586) →
`_amap_align` (~1589–1986): TGF edge weights, Pearce–Kelly cycle detection,
priority-queue column merging, topological sort. Optional remove-and-reinsert
refinement `_refine_one_sequence` (~1989–2156). Top-level entry points:
`fsa_align` (~2247) and `compute_pairwise_posteriors` (~1381). Worked CLI
examples: `python/experiments/fsa_tkf92_balibase.py` and siblings.

Paper-side write-up: `tkf/body-tkf-inference.tex` (FSA section). Method is
Bradley et al. 2009, *Fast Statistical Alignment*, PLoS Comp Biol 5(5):e1000392.

---

## 2. Fixed parameters vs. the optimization variable

Held **constant** during the per-pair τ step (they come from a pre-fit model):

- `λ` = insertion rate, `μ` = deletion rate.
- `r` = fragment-extension probability (`ext`).
- `Q` = substitution **generator** (A×A, rows sum to 0), `π` = equilibrium.

Optimization variable: **τ** (evolutionary time). The "closed-form score
derivatives" in `core/bdi.py` (`score_derivatives`) are **∂/∂λ, ∂/∂μ** — the
*rate* M-step. They are NOT what you want here. This step differentiates the
**same objective w.r.t. τ at fixed rates** — a different derivative. Don't try
to reuse `score_derivatives`; use §4.

---

## 3. The objective Q(τ)

Reference: `_tkf92_expected_ll` (lines ~91–102).

```
Q(τ) = Σ_{i,j} N_{ij} · log χ_{ij}(τ)            (transition term)
     + Σ_{a,b} W_{ab} · ( log π_a + log P_{ab}(τ) )   (emission term)
```

- `χ(τ)` = 5×5 TKF92 pair-HMM transition matrix, state order **S,M,I,D,E =
  0,1,2,3,4**. Built by `tkf92_trans` (`core/params.py:92`).
- `P(τ) = exp(Q·τ)` = substitution probability matrix (`transition_matrix`,
  `core/ctmc.py:31` — plain matrix exponential).
- `W_{ab}` = expected number of matches with ancestor residue `a`, descendant
  residue `b`. `P_{ab}` is read the same way (a = row = ancestor).
- `log π_a` is **constant in τ** → drops out of all derivatives.

Newton needs `dQ/dτ` and `d²Q/dτ²`. The two terms are independent; do them
separately and add.

---

## 4. Analytic derivatives (the part to implement)

Notation: dot = d/dτ. Let `ε = μ − λ`, `κ = λ/μ`.

### 4a. Emission term — clean, exact

Because `Q` commutes with `exp(Qτ)`:

```
P  = exp(Q τ)          (you already need this for the FB pass)
P' = Q · P             (one A×A matmul)
P''= Q · Q · P         (one more)
```

Then

```
dE/dτ   = Σ_{ab} W_{ab} · (QP)_{ab} / P_{ab}
d²E/dτ² = Σ_{ab} W_{ab} · [ (Q²P)_{ab}/P_{ab} − ( (QP)_{ab}/P_{ab} )² ]
```

Efficiency: if you eigendecompose the reversible `Q = V diag(d) V⁻¹` once, then
`P = V diag(e^{dτ}) V⁻¹`, `QP = V diag(d·e^{dτ}) V⁻¹`, `Q²P = V diag(d²·e^{dτ})
V⁻¹` — all τ-evaluations from a single decomposition. (Reference uses Padé
`expm` instead; either is fine.)

### 4b. Transition term — via α, β, γ

```
dT/dτ   = Σ_{ij} N_{ij} · χ̇_{ij} / χ_{ij}
d²T/dτ² = Σ_{ij} N_{ij} · [ χ̈_{ij}/χ_{ij} − (χ̇_{ij}/χ_{ij})² ]
```

So you need χ and its first two τ-derivatives. χ is built from three
time-dependent scalars α, β, γ (κ and r are constant). Build these atoms:

**α (survival), Φ = 1−α:**
```
α  = exp(−μτ)      α̇ = −μ α        α̈ =  μ² α
Φ  = 1 − α         Φ̇ =  μ α        Φ̈ = −μ² α
```

**β (offspring) — single code path, correct for all λ,μ including λ=μ.**
Use the `expm1`-ratio helper `g(x) = (eˣ−1)/x`:
```
g(x)   = (eˣ − 1)/x
g'(x)  = ((x−1)eˣ + 1)/x²
g''(x) = ((x²−2x+2)eˣ − 2)/x³
```
For |x| ≲ 1e-2 use Taylor (avoids cancellation):
```
g   ≈ 1 + x/2 + x²/6 + x³/24 + x⁴/120
g'  ≈ 1/2 + x/3 + x²/8 + x³/30
g'' ≈ 1/3 + x/4 + x²/10 + x³/36
```
Then with `x = ε τ`:
```
ρ  = g(x)          ρ̇  = ε g'(x)        ρ̈  = ε² g''(x)
w  = τ ρ           ẇ  = ρ + τ ρ̇        ẅ  = 2ρ̇ + τ ρ̈
Δ  = μ w + 1
β  = λ w / Δ
β̇  = λ ẇ / Δ²
β̈  = λ (ẅ Δ − 2 μ ẇ²) / Δ³
```
(This equals the textbook `β = λ(η−α)/(μη−λα)`, `η=e^{−λτ}`, away from λ=μ, and
limits to `s/(1+s)`, `s=μτ`, at λ=μ — with no branch and no singularity.)

**γ (orphan):**
```
γ  = 1 − (μ/λ) · β / Φ
m  = β̇ Φ − β Φ̇
γ̇  = −(μ/λ) · m / Φ²
γ̈  = −(μ/λ) · ( (β̈ Φ − β Φ̈) Φ − 2 m Φ̇ ) / Φ³
```
(γ is 0/0 only at τ→0, which the τ-clip below excludes.)

**Column atoms.** For a row whose "offspring/orphan" parameter is `b` (use β
for rows S,M,I; use γ for row D), with ḃ, b̈ its derivatives, the four
destination columns (M, I, D, E) are:

| dest | value | first deriv | second deriv |
|------|-------|-------------|--------------|
| →M | `κ α (1−b)` | `κ(α̇(1−b) − α ḃ)` | `κ(α̈(1−b) − 2α̇ḃ − α b̈)` |
| →I | `b` | `ḃ` | `b̈` |
| →D | `κ Φ (1−b)` | `κ(Φ̇(1−b) − Φ ḃ)` | `κ(Φ̈(1−b) − 2Φ̇ḃ − Φ b̈)` |
| →E | `(1−κ)(1−b)` | `−(1−κ) ḃ` | `−(1−κ) b̈` |

Call the row-S/M/I atoms `A_j` (built with β) and the row-D atoms `B_j` (built
with γ), `j ∈ {M,I,D,E}`.

**Assemble χ** (the fragment self-loop adds `r` on the diagonal of M/I/D rows;
S row has no self-loop and no `(1−r)` factor — `tkf92_trans` lines 100–107):
```
χ[S,j] = A_j
χ[M,j] = (1−r) A_j + r·[j=M]
χ[I,j] = (1−r) A_j + r·[j=I]
χ[D,j] = (1−r) B_j + r·[j=D]
```
The constant `r` and the constant `+const` drop under differentiation, so:
```
χ̇[S,j] = Ȧ_j           χ̈[S,j] = Ä_j
χ̇[M,·] = (1−r) Ȧ       χ̈[M,·] = (1−r) Ä
χ̇[I,·] = (1−r) Ȧ       χ̈[I,·] = (1−r) Ä
χ̇[D,·] = (1−r) Ḃ       χ̈[D,·] = (1−r) B̈
```
Only rows {S,M,I,D} × cols {M,I,D,E} are structurally non-zero; `N` is zero
elsewhere so those terms vanish (don't divide by χ=0 there).

### 4c. Total + Newton step

```
Q̇ = dT/dτ + dE/dτ
Q̈ = d²T/dτ² + d²E/dτ²
```

Reference optimizes **u = log τ** (keeps τ>0; matches the clip logic). Convert:
```
ĝ = τ · Q̇
ĥ = τ² · Q̈ + τ · Q̇
```
Newton update (reference, `_pairwise_posteriors_tkf92_jax` lines ~242–251):
```
safe = (|ĥ| > 1e-10) ? −ĥ : 1
step = clip(ĝ / safe, −1, +1)
u   += step
```
Repeat `n_newton` (=5) times. Finally `τ = clip(exp(u), 1e-4, 10)`.

(If you prefer τ-space: `τ ← τ − Q̇/Q̈` with a positivity guard. Log-space is
recommended — it's what the reference ships and it bounds the step naturally.)

---

## 5. Pitfalls

- **λ = μ (equal indel rates).** Common in fitted models. The textbook
  `β = λ(η−α)/(μη−λα)` is 0/0 here, and *autodiff through the `jnp.where`
  branch in `tkf_beta` returns NaN for the second derivative* (the JAX
  double-`where` grad trap). The §4b `g(x)` form has **no branch and no
  singularity** — it's correct and finite at λ=μ. This is a reason the closed
  form is actually better than the reference's autodiff, not just a substitute.
- **τ range.** Keep τ ∈ [1e-4, 10] as the reference does; γ's `1/Φ` is only
  ill-conditioned as τ→0, which the clip handles.
- **Don't reuse `score_derivatives`** (§2) — wrong variable (rates, not time).
- **W orientation:** `a` = ancestor (= x = row of P), `b` = descendant.
  Match `match_W_fixed`'s `einsum('ij,ia,jb->ab', …)` exactly or you'll
  transpose the substitution counts.
- **Two FB passes only.** Don't add a third "verification" FB; the optimized-τ
  pass *is* the one that produces the posteriors you anneal on.

---

## 6. Verifying your port

`python/scratch_fsa_tau_derivs.py` (in tkf-mixdom) implements exactly the §4
formulas in pure numpy and checks them against the JAX autodiff ground truth:

- λ≠μ: `ĝ`, `ĥ` match autodiff to **~1e-14** (machine precision).
- λ=μ: `ĝ` matches autodiff to ~1e-16; `ĥ` matches the finite-differenced
  *gradient* to ~1e-10 (autodiff `ĥ` is NaN there, as noted above).

Reproduce the same numeric check in TS/Rust: pick a few (λ,μ,r,τ,Q,π,N,W),
evaluate `Q(τ)` at u±h, finite-difference for ĝ and ĥ, and confirm your
closed forms agree to ~1e-7 (central-difference value-based) / ~1e-10
(if you difference your own analytic gradient). The pure-numpy script is the
portable oracle — match its numbers.
