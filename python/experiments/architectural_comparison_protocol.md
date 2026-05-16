# Controlled Architectural Comparison Protocol

(Saved 2026-05-01.)

**Context.** Current val-LL/pair numbers (d3f1 −666.21, Annabel 3×3×9
−663.16/−627.47, banded K=3 −668.17, BW-from-banded ≈ −667.10) are not
directly comparable: param counts span ~10×, inits range from cold to
imported, and SVI vs Adam converge differently. This protocol
separates architecture from training-strategy confounds.

**User priority (2026-05-01):** efficient direction of architecture +
strategy. Replicating Annabel exactly would be nice but is not the
primary goal. C5 (Annabel arch under our pipeline) was previously
attempted with batch size 2000 and did not get close to Annabel's
result; this is part of what motivated developing the Adam version of
`train_pfam`.

---

## A. Architectural axes

| Axis | Levels | Param impact (per d, f) |
|---|---|---|
| Per-class S^c | full GTR (190) / shared LG (0) / per-class scalar σ_c (1) | dominant |
| Per-class π^c | per-class (19) / tied across blocks (19/B) / frozen (0) | medium |
| #classes per (d,f) (C) | 1, 3, 9, 10 | linear in C |
| classdist[d,f,c] | unconstrained / fragchar-tied / block-diagonal (banded K) / frozen | C·n_dom·n_frag − 1 |
| n_frag, ext structure | f1, f2-diag, f3-banded, f3-unconstrained | ext params |
| n_dom | 3, 4, 5 | dom_weights + per-dom indel BDI |

Flat substitution-only param counts (n_dom=3):
- d3f1 ≈ 3·(190+19) = 627
- K=3 banded (frozen LG, free π, σ_c) ≈ 9·19 + 9 = 180
- d3f1c10 (--estimate-sexch) ≈ 30·(190+19) = 6,270
- Annabel 3×3×9 ≈ 81·(190+19) = 16,929

## B. Training-strategy confounds to control

- **Init**: fresh random / Maraschino-fit / warm-start from another ckpt (and which ckpt). Record `--init` / `--init-from-maraschino` / `--class-pi-init`.
- **Optimizer**: Adam vs SVI-BW; lr, batch, EMA τ/κ; pseudocounts (`--pi-pseudo`, `--S-pseudo`).
- **Seed**: ≥3 seeds per cell; report mean ± SE.
- **Data**: identical train split (`v1.json` train, 21,667 fams) and identical breadth-sample stream; identical val set (8,746 cherries).
- **Convergence**: same `--n-iter`, `--val-every`, `--patience`. Report best-val and last-val.
- **--subst-mode**: standard / rescaling-rates / tied-pi / frozen-pi (per `project_restricted_mstep_modes.md`).

## C. Concrete experiments

| # | Name | Hypothesis | Existing ckpt | New training | Runtime |
|---|---|---|---|---|---|
| C1 | **d3f1c3-fair** (3 dom × 1 frag × 3 site classes; exch-constrained: free π^c + σ_c, S^c = LG; standard SVI-BW from C3-spread init) | matches K=3 banded param count without fragchar/banded structure. Isolates "fragchar+banded" effect | none | yes; ~6h | ~6h |
| C2 | **K=3 banded, SVI-BW from cold** (no Maraschino init) | tests whether Maraschino-fit warm start (vs cold) is hurting/helping K=3 | banded_K3_train3k.npz (warm) | cold variant; ~6h | 6h |
| C3 | **K=3 banded with --estimate-sexch unfrozen** (full S^c, not σ_c only) | tests whether σ_c-only restriction caps capacity below Annabel | warm K=3 ckpt | resume with `--estimate-sexch`; ~4h | 4h |
| C4 | **d3f1 with --subst-mode rescaling-rates** (3 dom, c=1, σ_c only) | strict d3f1 baseline restricted to the same training regime as the running BW-from-banded | adam_d3f1_warm_postfix.npz | resume; ~4h | 4h |
| C5 | **Annabel architecture, our pipeline** (3×3×9, fresh, our SVI-BW) | reproduces Annabel param count under our training | Annabel ckpt for arch reference only | yes; ~12h+ | 12h+ |
| C6 | **Seed sweep on d3f1 and K=3** (3 seeds each) | estimates val-LL noise floor; tells us whether 0.5 nat/pair gaps are signal | partial | 6 short SVI-BW runs (~3h each) | 18h serial / 6h on 3 GPUs |
| C7 | **d3f1-warm → +c3 expand** (warm from d3f1, expand to C=3 with K-means split) | isolates "extra classes given good indels" from "cold MixDom2" | adam_d3f1_warm_postfix.npz | warm-expand SVI-BW; ~6h | 6h |

Trained on identical breadth-sample stream / same 21,667 train fams /
`--svi-batch 200 --svi-tau 1.0 --svi-kappa 0.7 --pi-pseudo 2 --S-pseudo 1`.

**Status note (2026-05-01):** C5 was already attempted with batch
size 2000 and did not close to Annabel's reported numbers under our
SVI-BW pipeline. This motivated the Adam version of `train_pfam`.
A fresh C5 attempt is therefore lower priority than C1/C4/C6.

## D. Confounds we cannot fully control

- **Annabel was trained on her own optimizer with possible val-set overlap** → mitigate by always evaluating on `/tmp/v1_val_minus_annabel_gtr_train.json` non-overlap subset (n=7,579 cherries) for the headline comparison.
- **Local optima depend on init** → mitigate via C6 multi-seed sweep + report range, not just best.
- **Different indel-param init** between ckpts → freeze indel BDI to a common warm-start in cross-arch runs (or add as a sensitivity in C7).
- **SVI noise floor** (`tkf/svb-convergence.tex` §3, eq. minibatch) — at B=200 the per-pair-LL minibatch SE is ~`v_θ/sqrt(200·E[L])`. Compute it once from C6 seed variance and report the noise floor alongside every val-LL.

## E. Headline metric

- **Primary**: per-pair val LL on the **non-overlap subset**, reported as mean ± SE across seeds (C6) and across the last 5 val checkpoints (within-run variability).
- **Comparison correction**: AIC = −2·LL_total + 2·k and BIC = −2·LL_total + k·log(N_pairs), with k = (substitution params + indel params + classdist + ext + dom_weights). Report ΔAIC and ΔBIC per pair vs the smallest-k model in the table.
- **Sanity**: also report per-param LL (LL_total / k) — useful for quick eyeballing but not a substitute for AIC/BIC.

A model "wins" only if (i) ΔBIC ≥ 10 and (ii) the LL gap exceeds the seed-induced noise floor measured in C6.
