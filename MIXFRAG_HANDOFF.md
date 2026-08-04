# MixFrag — handoff to next session

Status as of 2026-06-23. Author: previous Claude session, working with Ian Holmes.
This file is the authoritative continuity doc for the **MixFrag** effort. Read it
fully before doing any MixFrag work. Everything below is DONE and committed/pushed
unless explicitly marked TODO.

---

## 0. What MixFrag is (one paragraph)

MixFrag = TKF92 with the single fragment-extension parameter `\ext` (r) promoted to
a **per-fragment categorical latent variable**: when each mortal link's fragment is
created it independently draws a *fragtype* `f ∈ {1..𝓕}` with weight `w_f`, then has
geometric length with fragtype-specific parameter `r_f`. The substitution model
`(S, π)` AND the BDI/indel (links) process are **shared** across fragtypes; the only
new parameters are `{r_f}` and `{w_f}`. At `𝓕=1` (`w_1=1`) it reduces **exactly** to
TKF92. It is the single-domain, shared-substitution precursor to MixDom (no null
states → no collapse step). Eventual purpose: use MixFrag in TKF-DP (see §7).

---

## 1. Repos, dev→published split, and how to publish

| role | paper/lib | dir = dev (edit here) | published (generated) |
|---|---|---|---|
| paper | tkf-dp | `~/tkf-dp` = `${REPO_OWNER}/tkfdp` | `~/tkfdp` = `evoldoers/tkfdp` |
| inference lib + paper fragments | tkf-mixdom | `~/tkf-mixdom` = `${REPO_OWNER}/tkf-mixdom` | `evoldoers/tkf-mixdom` (submodule of evoldoers/tkfdp) |
| site | — | — | `~/tkfdp.net` = `evoldoers/tkfdp.net` (committed `supplement.pdf`) |

- **Publish command:** `bash ~/tkf-dp/scripts/publish_drop.sh` — stages the scrubbed
  subset, build-verifies main+supplement (aborts if they don't compile), pushes
  `evoldoers/{tkf-mixdom,tkfdp,bio-datasets}` + refreshes `evoldoers/tkfdp.net`. It
  operates on cache clones under `~/.cache/tkfdp-publish/`, NOT on `~/tkfdp` — so
  after publishing, fast-forward `~/tkfdp` and `~/tkfdp.net` with
  `git pull --ff-only origin main` (+ `git submodule update --init --recursive` in
  `~/tkfdp`).
- **CRITICAL macro gotcha:** the supplement (`~/tkf-dp/math-paper/supplement.tex`)
  **deliberately does NOT `\input preamble-shared.tex`** — it replicates the macros
  inline (search "we deliberately do NOT \input preamble-shared", ~line 58, defs
  near line 154). So **any new macro must be added in BOTH**
  `~/tkf-mixdom/tkf/preamble-shared.tex` AND `~/tkf-dp/math-paper/supplement.tex`.
  This bit us once with `\mixfrag`.
- The "**do not modify `~/tkf-mixdom`**" rule in `~/tkf-dp/CLAUDE.md` is **suspended
  for MixFrag** (Ian authorised editing tkf-mixdom for the model+code+paper).
- Build-verifying a supplement-only section locally: temp-copy the changed/new
  `tkf/*.tex` into the submodule `~/tkf-dp/math-paper/tkf-mixdom/tkf/`, run
  `cd ~/tkf-dp/math-paper && ./build.sh --supp`, check `supplement.toc`, then clean
  up (the submodule sits at an OLD commit, so `mixfrag*.tex` are UNTRACKED there →
  `rm` them; `git -C tkf-mixdom checkout -- tkf/<modified>` for tracked ones).

---

## 2. LaTeX work — DONE and PUBLISHED

All three items are live in `evoldoers/tkfdp` + `tkfdp.net` (latest publish: evoldoers
tkf-mixdom `f927d13`, tkfdp `1e7f85f`, tkfdp.net `7c03d6c`).

### 2a. TKF91 Pair HMM matrix fix (`~/tkf-mixdom/tkf/body-tkf91.tex`)
The joint TKF91 Pair HMM matrix had a **spurious `\kappa` across the entire `\ins`
column** (`\beta\kappa`/`\gamma\kappa` should be `\beta`/`\gamma`). Ian fixed it; I
confirmed against the JAX code. **Load-bearing facts (do not re-break):**
- The JOINT matrix `tkf91_trans` IS row-stochastic (`Σ_Y τ_{XY} = 1`). The
  `\ins` column carries **no** ancestral-length factor (insertions consume no
  ancestor). The **conditional** `tkf91_trans_cond` is the sub-stochastic one (it
  factors out the `κ^i(1-κ)` length prior).
- **TKF91 rows S, M, I are IDENTICAL**; row D differs (uses γ). This `τ_S=τ_M=τ_I`
  coincidence is LOAD-BEARING for B.6 (folds leading gaps into internal gaps, and
  empty alignments into trailing). Do not "simplify" it away.
- JAX `core/params.tkf91_trans` was ALWAYS correct (`tau[·,I]=beta`, `tau[D,I]=gamma`).

### 2b. §A.3 "The TKF92 Model with Fragment Mixtures (MixFrag)" — `~/tkf-mixdom/tkf/mixfrag.tex`
Model opening + §A.3.1 Singlet HMM (`2+𝓕` states `{S, I_f, E}`) and Pair HMM
(`2+3𝓕` states `{S, M_f, I_f, D_f, E}`) transition matrices + §A.3.2 Baum-Welch.
- `\input` in `tkf/tkf.tex`, `tkf/mixdom.tex`, AND `~/tkf-dp/math-paper/supplement.tex`
  (→ appendix **A.3**; pushed "TKF92 WFST by Singlet Division" to **A.4**).
- Labels: `sec:mixfrag`, `sec:mixfrag-machines`, `sec:bw-mixfrag`.
- Construction rule (from TKF91 τ): out of an emitting state ×`(1-r_f)`; into an
  emitting state ×`w_g`; self-loop `+ r_f·δ_{fg}` (same state & fragtype).
- Baum-Welch: per-fragtype `F_{af}=n̂'_{a_f a_f}·r_f/(r_f+(1-r_f)·τ_{aa}·w_f)`,
  `F_f=Σ_a F_{af}`, `E_f=Σ_a Σ_{b_g} n̂'_{a_f,b_g} − F_f`,
  `n̂_{a_f b_g}=n̂'_{a_f b_g}−δ_{ab}δ_{fg}F_{af}`. **The `w_f` in the `F_{af}`
  denominator is essential** (the self-loop's new-link branch re-draws the same
  fragtype). M-step: `r_f←F_f/(F_f+E_f)`, `w_f←E_f/Σ E_{f'}`. `E_f` = expected #
  type-f fragments.
- math-verifier: ALL correct, F=1 reduces to TKF92.

### 2c. §B.6 "MixFrag Training from Alignment-Summary Counts" — `~/tkf-mixdom/tkf/mixfrag-cherrytrain.tex`
The Maraschino-style summarised training. **`\input` in `supplement.tex` ONLY**
(between `body-tkf-inference.tex` = B.5 and `varanc-presence.tex`; it is **B.6**,
varanc → B.7). NOT in tkf.tex/mixdom.tex. Section-local macros are `\providecommand`'d
at the top of the file (`\transwithin`=𝖠, `\transbetween`=𝖡, `\runfac`=φ, `\Nmatch`,
`\NgapM`, `\NgapE`, `\Nstart`, `\Nend`, `\Nempty`, `\Nsub`).
**The math (math-verifier confirmed all 7 items, symbolic+numeric, to machine
precision; do NOT re-derive from scratch):**
- Fragtype-augmented transfers (𝓕×𝓕): within-run `(𝖠_a)_{fg}=r_f δ_{fg}+τ_{aa}(1−r_f)w_g`,
  between-type `(𝖡_ab)_{fg}=τ_{ab}(1−r_f)w_g`. **`𝖡_ab` is rank-one** = `τ_ab·(1−r)·wᵀ`
  (fragtype redrawn fresh at every new link) → it CUTS the transfer product at every
  column-type change.
- Run-factorisation (`eq:mixfrag-runfactor`): indel L = `τ_{S a_1}[∏φ_{a_i}(k_i)][∏τ_{a_i a_{i+1}}]τ_{a_m E}`,
  with run factor `φ_a(k)=wᵀ 𝖠_a^{k−1}(1−r)` (w on entry, (1−r) on exit).
- Run factor forms (`eq:mixfrag-mell`, `eq:mixfrag-gf`): `φ_a(k)=Σ_p τ_aa^{p−1} Σ_{Σℓ=k}∏ m(ℓ)`,
  `m(ℓ)=Σ_f w_f r_f^{ℓ−1}(1−r_f)`; GF `Φ_a(z)=M(z)/(1−τ_aa M(z))`,
  `M(z)=Σ_f w_f(1−r_f)z/(1−r_f z)`.
- Gap factor (`eq:mixfrag-gapprefix`, `eq:mixfrag-gapgf`): prefix recursion
  `P_I=(τ_MI+P_D τ_DI)Φ_I`, `P_D=(τ_MD+P_I τ_ID)Φ_D`; `𝒢_Y=P_I τ_IY+P_D τ_DY` for
  exit `Y∈{M,E}`. `G_Y(i,j)` = coefficient; sums over ALL I/D interleavings; no (0,0)
  term; entry-from-S folds into entry-from-M via the row identity.
- Sufficient statistics (`eq:mixfrag-summary-ll`) — COMPLETE set: `N_match(k)`,
  `N_gap→M(i,j)` (internal+leading), `N_gap→E(i,j)` (trailing + match-free whole),
  `N_start`/`N_end` (begins/ends with match → `τ_MM`/`τ_ME`), `N_empty` (no columns →
  `τ_SE`), and `N_sub(a,b)=π_a exp(Qt)_{ab}` substitution. (0,0) excluded (adjacent
  matches = one M-run).
- EM: per-segment expected stats `B,D,S,L,F_f,E_f` are gradients of log-factors via
  the **score-function identity** (`eq:score-identity` in body-tkf91); `M` (trajectory
  count) and `T` (= bin-time × #cherries) are observed directly; M-step = the
  `sec:bw-mixfrag` closed forms.
- A review-findings memory exists at
  `~/.claude/projects/-Users-yam-tkf-dp/memory/review-mixfrag-cherrytrain.md`.

### LaTeX design decisions (don't second-guess)
- Notation: fragtype index = `\frag`/`\srcfrag`/`\destfrag` (= f/f/g); fragtype COUNT
  = `\nfrag` (= 𝓕). There is NO `\nfrags`. (Ian corrected an early mix-up.)
- `\mixfrag` macro = `\mbox{MixFrag}` (NOT `\tkffrags^\nfrag`/"TKF92^𝓕"). Defined in
  BOTH preamble-shared.tex AND supplement.tex (see §1 gotcha).
- A.3 must NOT reference MixDom / Maraschino / SVI (Ian: appendices may become
  separate papers). B.6 (which IS in appendix B with Maraschino) may reference
  Maraschino (`sec:maraschino-main`), `sec:tkf92-gapprob`, A.3, and `eq:score-identity`.
- Wide displays use `multline`; the Pair HMM matrix uses `\resizebox{\textwidth}`.

---

## 3. Code work — DONE (in `~/tkf-mixdom/python`, pushed to `${REPO_OWNER}/tkf-mixdom`)

All tests pass. Run: `cd ~/tkf-mixdom/python && python3.12 -m pytest tests/test_mixfrag.py tests/test_mixfrag_cherry_counts.py -q` (22 tests). (No `uv` on this machine — use `python3.12`.)

### 3a. Model builders
- `tkfmixdom/jax/core/params.py`: `mixfrag_trans(ins,del,t,exts,weights)` (the
  `(3𝓕+2)`-state Pair HMM), `mixfrag_singlet_trans(...)` (`𝓕+2`-state Singlet),
  `mixfrag_pair_index(F)` → `(S, M0, I0, D0, E)` base indices. **State order
  S, M_1..M_F, I_1..I_F, D_1..D_F, E.**
- `tkfmixdom/jax/models/left_regular.py`: `make_mixfrag_pair_hmm`,
  `make_mixfrag_singlet_hmm`. `state_types` map every `M_f→M, I_f→I, D_f→D`, so the
  generic Pair-HMM emission/DP code (`forward_backward_2d`, `pair_emission_logprob`)
  works UNCHANGED.

### 3b. svi-Baum-Welch (the alignment- AND fragtype-marginalised trainer)
- `tkfmixdom/jax/train/mixfrag_svi_bw.py`: `svi_bw_mixfrag(...)`,
  `estep_batch_mixfrag(...)`, the per-fragtype chi resolution
  (`_mixfrag_chi_resolve_core`), `_aggregate_chi_to_5x5`, and M-steps
  `m_step_ext_per_fragtype` (Beta) + `m_step_weights` (Dirichlet). Reuses
  `tkf92_svi_bw`'s `_bin_bucket_pairs`, `_stack_bucket`, `_bdi_stats_batch`,
  `m_step_lam_mu`. The FB E-step yields per-fragtype `F_f,E_f`; BDI stats run on the
  fragtype-summed 5×5 counts; resolution denominator carries `w_f`.
- This is **path 1** (per-pair FB). It WORKS but is the marginalised path.

### 3c. Data-prep (path-2 input builder) — `~/tkf-mixdom/python/build_mixfrag_cherry_counts.py`
The MixFrag analogue of `build_tkf92_cherry_counts.py` (the template — read it).
Same Pfam cherry-pick / geometric τ-bin / multiprocessing pipeline (JAX-free helpers
IMPORTED from `build_tkf92_cherry_counts`). Per τ-bin accumulates: `match_counts`,
`singlet_counts`, `N_match(k)`, `N_gap_to_match(i,j)`, `N_gap_to_end(i,j)`, and a
`scalars[bin,3]` = `[start, end, empty]`. **i=#deletions, j=#insertions.** The
variable-size run/gap tensors are stored **sparsely** (coordinate arrays
`*_idx`,`*_val`) → no length cap, exact aggregation. Helpers `load_family_counts`,
`aggregate_counts`. CLI mirrors the TKF92 builder. Output: one `.npz` per family.
- Tests: `tests/test_mixfrag_cherry_counts.py` (10). Tests the decomposition,
  invariants (`Σ k·N_match=#M`, etc.), `(n_m,n_i,n_d)` consistency with the TKF92
  builder, sparse round-trip + aggregate, and an end-to-end `process_family` run.

### Code design decisions
- F=1 parity is the canonical correctness oracle (matrix, FB log_p, full E-step
  suff-stats `F_f↔ext_count`, `E_f↔notext_count`) — keep using it.
- Sparse coordinate storage for run/gap count tensors (no caps; exact aggregation).
- **PROJECT RULE — NEVER gradient-ascent / line-search M-steps.** All TKF-based
  models have exact closed-form M-steps. If a closed-form M-step gives wrong results
  it's a BUG to fix, not a reason to switch to gradient ascent. (See tkf-mixdom
  CLAUDE.md "NEVER revert to gradient ascent".)

---

## 4. Publish/commit state (exact)

- LaTeX (TKF91 fix, A.3, B.6) + `\mixfrag` macro: committed on `${REPO_OWNER}/tkf-mixdom`
  main (`b79e8a8`→`51e15c9`) and `${REPO_OWNER}/tkfdp` main (`8a884c6`,`c3580c6`).
  **Published** to evoldoers (tkf-mixdom `f927d13`, tkfdp `1e7f85f`, tkfdp.net
  `7c03d6c`). `~/tkfdp`/`~/tkfdp.net` synced.
- Model builders + svi-BW + their test (`test_mixfrag.py`): `${REPO_OWNER}/tkf-mixdom`
  `d8d054a` — also PUBLISHED to `evoldoers/tkf-mixdom`.
- Data-prep `build_mixfrag_cherry_counts.py` + `test_mixfrag_cherry_counts.py`:
  `${REPO_OWNER}/tkf-mixdom` `14a580e6`,`7e5254d2` — **NOT yet in evoldoers** (held to batch
  with the EM publish). Refresh evoldoers with `publish_drop.sh` when convenient.

---

## 5. Summarised-count EM — DONE (`train/mixfrag_cherry_em.py`)

Implements the §B.6 method; **34/34 MixFrag tests pass** (`pytest tests/test_mixfrag.py
tests/test_mixfrag_cherry_counts.py tests/test_mixfrag_cherry_em.py`). Structure:

- **Factor functions** (per τ-bin; JAX-differentiable): `m_full(s,r,q,Lmax)` =
  `m(ℓ)=Σ_f w_f r_f^{ℓ−1}(1−r_f)`; `run_factor(τ_aa, m)` = `φ_a(k)` via the GF
  power-sum `Φ_a=Σ_p τ_aa^{p−1} M(z)^p` (repeated `jnp.convolve`); `gap_factors(τ,
  φ_I, φ_D, i_max, j_max)` = `G_M(i,j), G_E(i,j)`. **The gap DP is solved as an
  antidiagonal fixed-point sweep** `for _ in range(i_max+j_max)` — i.e. one
  inside-outside per time point up to that bin's max(i+j) wavefront (Ian's framing)
  — using **causal-Toeplitz matmuls** (`_causal_toeplitz`) for the I/D run
  convolutions. **NO `.at[]` scatter anywhere in the module** — a scatter-heavy
  first version SIGILL'd the XLA-CPU LLVM JIT on the 2nd large compile; the
  matmul/`jnp.where` rewrite fixed it (and is faster). φ/G use the observed
  k/(i,j) support only (no dense caps).
- **Log-L**: `corpus_indel_loglik` (eq:mixfrag-summary-ll) + `corpus_substitution_loglik`
  (fixed (Q,π) match-emission term; a constant offset for the indel M-step).
- **E-step** (`estep`): the score/expectation-semiring identity via autodiff of the
  per-bin log-partition, parameterised by the **free TKF91 transition slots τ_ab**
  and the **split fragtype atoms** (start `s_f=w_f`, extension `r_f`, stop
  `q_f=1−r_f`): `n̂_ab = ∂logZ/∂log τ_ab` (the resolved 5×5, extensions already in
  `r` not τ_aa), `F_f = ∂/∂log r_f`, `E_f = ∂/∂log s_f`. `n̂` → `(B,D,S)` via
  `tkf91_stats_from_counts_batch` (batched over bins); `L,M` from
  `transition_count_groups`; `T = Σ_bin t·#cherries`. The per-bin `value_and_grad`
  is **jitted** (cached per grid shape → fast across iterations); BDI is batched.
- **M-step**: REUSE `m_step_indel_quadratic` (κ-quadratic λ,μ) +
  `m_step_ext_per_fragtype` (Beta r_f) + `m_step_weights` (Dirichlet w_f). **Closed
  form, NOT gradient ascent.** (Q,π) fixed, as in svi-BW.
- **`em_fit(agg, n_fragtypes, Q, pi, …)`** is the driver (monotone-LL EM with
  tol-based early stop).

**κ=1 (λ=μ) is a genuinely IMPROPER point** — every `τ[·,E]=(1−β)(1−κ)=0`, so the
indel log-L is −∞ and its autodiff gradient is NaN. The κ-quadratic M-step keeps
κ<1 thereafter, but a **κ≥1 init NaNs the first E-step**; `em_fit` nudges the init
to keep κ≤0.9 (and `_xlogy` guards the boundary scalar terms). Do NOT init λ=μ.

**DECISIVE TEST** (`tests/test_mixfrag_cherry_em.py`): the count-tensor indel log-L
(from the factors) equals an **independent brute-force oracle** to ~1e-9. NB I did
NOT use `forward_backward_2d` as the oracle (the handoff's original wording):
`forward_backward_2d` marginalises over *all* alignments, whereas the count-tensor
likelihood is **alignment-given** and only sums **within-gap I/D orderings**. So the
correct oracle enumerates those orderings and runs a fragtype-summed forward pass
over each explicit column sequence (`_oracle_cherry`) — that ordering sum is exactly
what `G_Y(i,j)` packs. Other tests: F=1→TKF92, factor units, **score-identity finite
difference** (`∂logZ/∂λ = B/λ−S−T + L/λ − Mκ/((1−κ)λ)`, and the μ analogue — note the
κ-geometric terms beyond the WFST part), the `Σ(F_f+E_f)=#emitting-columns`
invariant, monotone-LL, **multi-bin parameter recovery** (simulate at several τ-bin
centres sharing one θ, fit, recover), and the κ=1-init guard.

Recovery sanity (multi-bin, ~12k cherries): θ_true (λ,μ,r,w)=(.08,.11,[.25,.75],
[.65,.35]) → fit (.082,.112,[.246,.75],[.63,.37]). Convergence of `w` is the slow
tail; needs a few hundred iters.

---

## 6. Standalone simulator + cross-path agreement — DONE

- **`simulate/mixfrag_sim.py`** (`simulate_mixfrag_cherries`): forward-generative
  MixFrag sampler — walk the (3𝓕+2)-state Pair HMM from S by the joint τ rows
  (sampling the latent fragtype along the way), emit **real residues** under the
  shared `(S,π)` (match `a~π, b~exp(Qt)[a]`; insert `b~π`; delete `a~π`). Returns
  per cherry the unaligned int seqs `x,y`, the column types, AND the aligned rows
  `row_a,row_b` (so both trainers can consume one dataset). Residues use the
  `build_tkf92_cherry_counts` amino-acid order. (The old MixDom sim was removed —
  built fresh.)
- **Cross-path test** `test_em_and_svibw_agree_on_simulated_data`: simulate one
  corpus (F81 substitution, diverse π so positions are distinguishable), fit with
  BOTH `em_fit` (alignment-given → tight recovery of θ) and `svi_bw_mixfrag`
  (alignment-marginalised → λ/μ within ~20% and the fragtypes separated in the same
  order, but compressed toward the middle by alignment uncertainty + SVI noise).
  Asserts each recovers truth (to its own tolerance) and they agree structurally
  (κ<1 both; the shorter-extension fragtype carries the heavier weight in both;
  λ/μ agree to rtol 0.35). Seeds fixed → deterministic. This is the end-to-end
  closure the F=1 parity tests can't give (recovery of the TRUE generative θ).
- The EM-only recovery test (`test_em_recovers_simulated_parameters`) was
  refactored onto the same standalone simulator (one simulator, not two).

**Training entrypoint — DONE.** `train_mixfrag_cherry_em.py` (the *fit* half of
build→fit): aggregates the chosen families' per-family count `.npz`, loads fixed
LG08 (`core/protein.rate_matrix_lg`, already in the count tensors' AA order), runs
`em_fit`, optionally reports a held-out `--val-split` log-L, saves `.npz`+`.json`.
Importable core `train_mixfrag(counts_paths, …)`; CLI verified end-to-end. Tests:
`tests/test_train_mixfrag_cherry_em.py` (3). So the full Pfam pipeline is now
`build_mixfrag_cherry_counts.py` → `train_mixfrag_cherry_em.py`.

**Still TODO:** update `tkf/implementations.tex` (modules + test counts); run real
training on the GPU box; then §7 (downstream TKF-DP O(L⁴) sampler).

Also TODO: update `tkf/implementations.tex` (modules + test counts; see tkf-mixdom
CLAUDE.md). Real training runs on the AWS GPU box (cloud notes in tkf-mixdom
CLAUDE.md — sub-account <AWS_ACCOUNT>, profile `tkf-gpu`, SkyPilot, surface spend
before launching >$5).

---

## 7. DOWNSTREAM — TKF-DP infinite pair HMM (the eventual point of MixFrag)

MixFrag exists to be used by **TKF-DP** (`~/tkf-dp`, the Potts/Dirichlet-process
coevolution model). Concretely that means building a **MixFrag version of the
O(L⁴) infinite-pair-HMM MCMC sampler** described in
`~/tkf-dp/math-paper/appendix-infinite-phmm.tex` (`sec:infinite-hmm`; it currently
covers the TKF/Potts case at O(L⁴) via an F₂-SCFG inside-outside + the infinite
pair-HMM MCMC). When MixFrag training is validated, that sampler is the integration
target — see `~/tkf-dp/CLAUDE.md` ("MixFrag (planned integration)"). This is a
larger, separate piece; do NOT start it before the EM + simulator are done and Ian
asks.

---

## 8. Where the plans live (keep in sync)
- `~/tkf-mixdom/CLAUDE.md` — "### MixFrag" design note (two training paths; the §B.6
  method recorded as the exact-EM path). Update it as code lands.
- `~/tkf-dp/CLAUDE.md` — "## MixFrag (planned integration)" (the downstream O(L⁴)
  sampler target).
- Memory: `project-publish-tkf91-fix-mixfrag.md` (the TKF91 κ-typo gotcha + shipped
  status), `review-mixfrag-cherrytrain.md` (B.6 verification record), and a pointer
  to THIS file.
