# The Nested TKF Model

A LaTeX paper, JAX reference implementation, and ongoing experimental
program around **MixDom** — a singly-nested TKF92 variant that adds
factorisable context-dependence (per-domain, per-fragment, per-site-class)
to a TKF91 backbone for protein sequence evolution.

See [Large & Holmes (2026)](https://www.biorxiv.org/content/10.64898/2026.02.02.702952v1)
for the original training and evaluation of MixDom on Pfam. The code in
this repo has since been rebuilt around **MixDom2** — a Markovian
fragchar variant whose null-state count restoration gives an exact
closed-form M-step at every level.

## What's here

### Paper (`tkf/`)

- `tkf.tex` — main paper (build with `tkf/build.sh`)
- `exploded-mixdom.tex`, `grammar-elaboration.tex`, `recursive.tex` — the
  6-step elimination chain that turns a fully-exploded MixDom2 into a
  collapsed Pair HMM, and the corresponding restoration that recovers
  exact sufficient statistics
- `substitution-mstep.tex` — closed-form substitution M-steps (rescaling,
  tied-π, tied-π+rescaling, etc.) implemented as `--subst-mode` flags in
  `train_pfam.py`
- `lhopital-limits.tex` — BDI sufficient statistic limits at λ=μ
- `svb-convergence.tex` — SVI-BW convergence theory (minibatch variance,
  ESS, Fisher information per parameter group)
- `partition-recon.tex` — exact 3F+2 partition reconstruction
- `algebraic-distillation.tex`, `mixdom-wfst.tex`, `tkf92-wfst-derivation.tex`
  — order-1 WFST distillation for tree composition
- `implementations.tex` — auto-maintained appendix listing every JAX
  module, CLI tool, and test file

### JAX reference implementation (`python/tkfmixdom/jax/`)

| Model | Pair HMM states | What it adds |
|-------|-----------------|--------------|
| **TKF91** | 5 | single (λ, μ) indel rates |
| **TKF92** | 5 | + geometric fragment extension |
| **MixDom1** | 2 + 5KF | + K domains × F fragments (intra-domain Markov on fragments) |
| **MixDom2** | 2 + 5KF | + Markovian *fragchars* and per-site classes (current focus) |
| **TKFST** | ~49 NTs | + RNA secondary structure (SCFG) |

Capabilities used in the active workflow:
- **Differentiable DP**: forward/backward with custom VJPs via the
  score-function identity; geometric-bin padding so JAX reuses compiled
  graphs across input lengths
- **Exact Baum-Welch**: closed-form M-steps at every level via
  null-state count restoration (6-step elimination chain; verified
  against autodiff to 10⁻¹⁴)
- **Holmes-Rubin substitution**: analytic expected CTMC path counts
  feed BDI sufficient statistics, including the L'Hôpital limits at λ≈μ
- **Restricted M-step regimes**: `--subst-mode {standard, frozen-pi,
  rescaling-rates, rescaling-rates-and-pi, tied-pi, tied-pi-rescaling,
  alt-tied-pi-rescaling}` for ablations of the per-class substitution
  freedom
- **Two training modes** in `train_pfam.py`:
  - `--svi-bw` — Stochastic Variational Baum-Welch with EMA
    pseudocounts (per `tkf/svb-convergence.tex`)
  - `--adam` — Adam on the same E-step δ via the
    `(e_step, expected_ll)` split with parameter-shape-only JIT
- **Maraschino warm-start**: CherryML-style count-then-distill pre-fit
  that produces an `npz` checkpoint consumed via `--init-from-maraschino`

### Top-level Python tools (`python/`)

- `train_pfam.py` — train MixDom on streamed Pfam Stockholm MSAs
- `maraschino.py` — count → fit → distill → fetch (warm-start producer)
- `fit_tkf92_mixture.py`, `fit_banded_mixdom2_mixture.py` — single-model
  fitters used by the K-component baselines
- `build_tkf92_cherry_counts.py`, `build_marcounts_parallel.py` —
  Pfam-cherry suff-stat builders
- `experiments/` — one driver per benchmark or analysis (FSA-MSA on
  BAliBase / OxBench, partition reconstruction on Pfam / TreeFam,
  fels21 / fels40 baselines, BDI consistency figures, etc.)

### Current experimental program

The active comparison is documented in
[`python/experiments/architectural_comparison_protocol.md`](python/experiments/architectural_comparison_protocol.md)
— the **C1–C7 protocol** for separating architecture from
training-strategy confounds. Each cell fixes train split, breadth-sample
stream, optimizer settings, seeds, and reports per-pair val LL on the
non-overlap subset with ΔAIC and ΔBIC against the smallest-k model.

Architectural axes being swept (n_dom, n_frag, n_classes, classdist
structure, per-class S^c, per-class π^c) and the benchmark inputs
(`*_spec.json` files) live alongside the protocol in
`python/experiments/`.

### Test suite

1884 tests under `python/tests/` (organised in level0–level4 by rigour;
gradient-equivalence and parameter-recovery tests dominate the higher
levels). Run with `cd python && uv run pytest tests/`.

## Build

```bash
# Paper
tkf/build.sh                     # pdflatex + biber + open

# Python tests
cd python && uv run pytest tests/

# Train MixDom2 (post-fix defaults — see project_postfix_training_queue
# memory note for the canonical flag set used in the architectural sweep)
cd python && uv run python train_pfam.py \
    --svi-bw --precompiled-pairs pfam/precompiled/ \
    --split train --split-file ~/bio-datasets/data/pfam/seed/splits/v1.json \
    --breadth-sample --svi-batch 200 \
    --n-dom 3 --n-frag 1 --n-classes 1 \
    --n-iter 200 --val-every 5 --patience 20
```

## Data and checkpoints

Datasets, large precomputed counts, and checkpoints live outside the
repo:
- `~/bio-datasets/` — Pfam, BAliBase, OxBench, TreeFam (symlinked into
  `python/` where benchmark scripts expect them)
- S3 bucket `s3://tkf-mixdom-gpu-618647024028/` — canonical store for
  per-family derived data, training checkpoints, and result JSONs.
  Sync utilities in `scripts/sync_data_from_s3.sh` and
  `scripts/upload_results_to_s3.sh`.

The repo intentionally tracks only:
- code, paper, test fixtures
- benchmark *spec* files (`python/experiments/*_spec.json`) — the
  canonical entry-list inputs that the benchmark drivers consume
- a small set of best/test parameter checkpoints under
  `python/params/best/` and `python/params/test/`

Result JSONs, training logs, and per-family derived data are
`.gitignore`'d and pushed to S3 instead.

## Other implementation targets

The JAX reference is the source of truth. WebGPU (browser) and
Rust → WASM (fallback) ports are planned but not yet built.

## Related repos

- [`machineboss`](https://github.com/ihh/machineboss) — finite-state
  transducer toolkit with JAX differentiable DP (composition,
  semirings, anti-diagonal wavefront, silent-transition closure)
- [`subby`](https://github.com/ihh/subby) — phylogenetic substitution
  models with JAX (postorder/preorder via `jax.lax.scan`,
  column-vmapped Felsenstein)
