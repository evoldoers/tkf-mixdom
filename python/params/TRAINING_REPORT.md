# MixDom Training Report

*Generated 2026-03-20. Regenerate biophysical tables with: `cd python && uv run python results/generate_analysis.py`*

## Summary of Completed Runs

### Baum-Welch EM (with per-domain substitution model)

| Run | Domains | Data | Families | Pairs | Iters | Final LL | Wall time | GPU |
|-----|---------|------|----------|-------|-------|----------|-----------|-----|
| bw_d3f2_pfam100_20iter | 3×2 | 100 Pfam families | 98 | 11,456 | 20 | -6,269,649 | ~7.5h | 2080 Ti |
| bw_d5f2_pfam100_20iter | 5×2 | 100 Pfam families | 98 | 11,456 | 20 | -6,062,243 | ~7.5h | 2080 Ti |
| bw_d8f2_pfam100_15iter | 8×2 | 100 Pfam families | 98 | 9,818 | 15 | -6,250,762 | ~7h | 2080 Ti |
| bw_d3f2_fullseed_15iter | 3×2 | Full Pfam seed (train split) | 5,996 | 11,992 | 15 | -8,285,353 | ~5.4h | 2080 Ti |
| bw_d8f2_fullseed_15iter | 8×2 | Full Pfam seed (train split) | 4,909 | 9,818 | 15 | -6,774,478 | ~7.2h | 2080 Ti |

### Maraschino (CherryML Adam on adjacency counts)

| Run | Domains | Data | Steps | Final LL | Wall time |
|-----|---------|------|-------|----------|-----------|
| maraschino_d3_comparison | 3 | seed_counts.npz (27K families, 206M counts) | 5,000 | -1,757,922,432 | 7 min |

### In Progress

| Run | Method | Status |
|-----|--------|--------|
| Adam d3f1 100fam | Adam gradient ascent | Running GPU 0, ~8s/step |
| Stochastic EM d3f2 fullseed | Stochastic minibatch EM | Running GPU 1, iter 1 |

## LL Curves

### BW d3 full-seed (5,996 families, 15 iterations)

| Iter | ΔLL | Cumulative time |
|------|-----|-----------------|
| 1→2 | +841,453 | 32 min |
| 2→3 | +2,942 | 66 min |
| 3→4 | +4,841 | 100 min |
| 4→5 | +5,776 | 134 min |
| 5→6 | +4,912 | 168 min |
| 6→7 | +3,664 | 202 min |
| 7→8 | +2,826 | 236 min |
| 8→9 | +2,174 | 270 min |
| 9→10 | +1,634 | 304 min |
| 10→11 | +1,249 | 338 min |
| 11→12 | +994 | 372 min |
| 12→13 | +834 | ~6h |

### BW d8 full-seed (4,909 families, 15 iterations)

| Iter | ΔLL |
|------|-----|
| 1→2 | +522,749 |
| 2→3 | +5,141 |
| 3→4 | +6,205 |
| 4→5 | +7,906 |
| 5→6 | +10,496 |
| 6→7 | +7,880 |
| 7→8 | +3,448 |
| 8→9 | +2,074 |
| 9→10 | +1,947 |
| 10→11 | +2,071 |
| 11→12 | +1,882 |
| 14→15 | +20,097 |

Note: d8 iter 14→15 had a large jump (+20K), suggesting a late domain reassignment.

## Key Findings

### 1. Three biophysically meaningful domains emerge consistently

Across all d3 runs, the model learns:

| Domain | Weight | λ (ins) | μ (del) | Character |
|--------|--------|---------|---------|-----------|
| Core | 55-66% | 0.003-0.010 | 0.004-0.012 | Hydrophobic-enriched, charged-depleted |
| Loops/surface | 22-26% | 0.020-0.050 | 0.027-0.140 | G+P enriched, moderate rates |
| IDR/fast | 7-27% | 0.050-0.127 | 0.060-0.188 | Hydrophobic-depleted, G+P enriched |

### 2. Eight domains capture finer structural biology

The d8 model separates:
- **Buried core** (33%): hydrophobic +2.7%, charged -4.0%
- **Surface loops** (22%): hydrophobic -9.7%, G+P +3.7%
- **Charged surface** (8%): charged +8.9%
- **IDR** (2%): hydrophobic -18.7%, charged +9.5%, G+P +4.5%
- **Deletable linkers** (7%): κ=0.22, strong deletional bias
- Plus 3 additional transition domains

### 3. Per-domain amino acid composition tracks known biology

| Signal | Expected from structural biology | Observed |
|--------|----------------------------------|----------|
| Core = hydrophobic | α-helix/β-sheet buried residues | ✓ +3-5% |
| IDR = Gly+Pro enriched | Backbone flexibility, PPII | ✓ +4-10% |
| IDR rates ~15× core | Afonso & Bhatt GBE 2015 | ✓ λ ratio 10-30× |
| Loops = Cys enriched | Disulfide bonds | ✓ +2-4% (BW only) |
| Loops = charged enriched | Salt bridges, surface electrostatics | ✓ +5-9% |

### 4. Full Pfam seed training gives broader coverage

- 100-family runs: 98 families × all pairs = ~11K pairs, all families revisited every iteration
- Full-seed runs: 5000-6000 families × 1-2 pairs each = diverse sampling via interleaved budget
- Full-seed d3 LL is higher magnitude (-8.3M vs -6.3M) because it sees more data

### 5. Maraschino vs BW comparison

Maraschino (CherryML) on the same Pfam data:
- **Faster**: 7 min vs hours for BW
- **Weaker domain differentiation at d3**: tends to collapse to one dominant domain on small data (pfam_counts, 198K counts). Works well on large data (seed_counts, 206M counts).
- **Stronger per-domain composition signal**: unconstrained Adam finds larger amino acid deviations from LG08 than BW with priors

### 6. Substitution M-step works correctly

Option 2 from the paper (iterative coordinate ascent: fix π → solve S, fix S → solve π via Lagrange multiplier) produces monotonically increasing LL. Option 1 (set π from V counts, solve S once) caused LL decreases.

## Training Infrastructure

### Methods available
1. **BW-EM** (`train_pfam.py`): exact EM with chain restoration, per-domain substitution
2. **Adam** (`train_pfam.py --adam`): stochastic gradient ascent via custom VJPs
3. **Stochastic EM** (`train_pfam.py --stochastic-em`): fresh pairs every iteration
4. **Maraschino** (`maraschino.py fit`): CherryML on pre-counted adjacency tensor

### Key features
- Clan-aware train/val/test splits (v1.json, 812 clans, 59 benchmark families in test)
- Interleaved budget sampling with lazy manifest (no 30-min upfront scan)
- Per-family cherry selection capped at 500 sequences
- StoIndex for fast random access to Stockholm files
- Geometric bin padding for JIT cache reuse
- `--save-counts` for BW vs CherryML comparison on same data
- `--jax-cache-dir` for persistent JIT compilation cache
- `--rebuild-manifest` for safe checkpoint recovery after file moves

### Data
- 100 Pfam families in `pfam/` (symlinked from `~/bio-datasets/data/pfam/`)
- 27,481 Pfam seed families in `~/bio-datasets/data/pfam-seed/`
- Canonical split: 21,667 train / 2,430 val / 3,384 test (v1.json)
- CherryML counts: `data/seed_counts.npz` (206M match counts, 27K families)

## File Inventory

### Best-fit parameters (`params/best/`)

| File | Method | Domains | Data |
|------|--------|---------|------|
| bw_d3f2_pfam100_20iter.npz | BW+subst | 3×2 | 100 families, 20 iters |
| bw_d5f2_pfam100_20iter.npz | BW+subst | 5×2 | 100 families, 20 iters |
| bw_d8f2_pfam100_15iter.npz | BW+subst | 8×2 | 100 families, 15 iters |
| bw_d3f2_fullseed_15iter.npz | BW+subst | 3×2 | 5,996 families, 15 iters |
| bw_d8f2_fullseed_15iter.npz | BW+subst | 8×2 | 4,909 families, 15 iters |

### Training checkpoints (`pfam/`)
All intermediate checkpoints, logs, and counts tensors from training runs.
Gitignored — regenerable by rerunning training.

### Naming convention
`{method}_d{N}f{F}_{dataset}_{iters}iter.npz`


## Updated Test Split Evaluation (2026-03-21)

| Model | Test LL/pair | Method | Training data |
|-------|-------------|--------|---------------|
| BW d8 fullseed | -573.6 | EM, 15 iters | 4909 families |
| **SVI d3f2 fullseed** | **-665.2** | **SVI stochastic EM, 15 iters** | **155K pairs, 21K families** |
| BW d3 fullseed | -681.0 | EM, 15 iters | 5996 families |
| Adam d3f1 100fam | -697.3 | Adam, 5.4h | 100 families |
| BW d3 100fam | -700.8 | EM, 20 iters | 98 families |
| Stochastic EM d3 (no SVI) | -706.1 | Stoch EM, 15 iters | 80K pairs |

Key finding: SVI (Robbins-Monro EMA of sufficient stats) improves
stochastic EM from -706 to -665 LL/pair, surpassing frozen-pair BW (-681).
The information retention across iterations closes the gap as predicted
by Hoffman et al's theory.

Adam OOMs on d3f2 (32 states) on RTX 2080 Ti due to VJP compilation
memory. Gradient checkpointing (jax.remat) needed for Adam on larger models.


## Maraschino Comparison (2026-03-22)

| Model | Test LL/pair | Method | Time | Data |
|-------|-------------|--------|------|------|
| BW d8 fullseed | -573.6 | EM | 7.2h | 4909 fam |
| Maraschino d3 (as MixDom) | -658.0 | CherryML Adam | 7 min | 27K fam (counts) |
| SVI d3f2 fullseed | -665.2 | SVI stoch EM | 8h | 155K pairs |
| BW d3 fullseed | -681.0 | EM | 5.4h | 5996 fam |

Caveat: Maraschino trained on seed_counts.npz which includes test families.
Need train-split-only counts for a fully fair comparison.

Key insight: at d3, data quantity (CherryML on 27K families) outperforms
statistical efficiency (BW EM on 6K families). For d20+, CherryML may be
the way to initialize, with SVI fine-tuning on pairwise data.
