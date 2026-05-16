# Data Inventory

Central reference for all data artifacts: source files, intermediate
representations, and trained parameters.

## Source Data

### Pfam Seed Alignments (Stockholm format)

| Location | Families | Size | Notes |
|----------|----------|------|-------|
| `~/bio-datasets/data/pfam-seed/` | 27,481 | 794 MB | Full Pfam 38.1 seed, `.sto` files |
| `~/bio-datasets/data/pfam/` | 100 | 3.8 MB | Selected subset for quick BW training |
| `python/pfam/*.sto` | 100 | symlinks | → `~/bio-datasets/data/pfam/` |

Split file: `~/bio-datasets/data/pfam-seed/splits/v1.json`
- Train: 21,667 families (from 812 clans)
- Val: 2,430 families
- Test: 3,384 families (includes 59 BAliBASE benchmark families)

### Clan Membership
| Location | Size | Notes |
|----------|------|-------|
| `~/bio-datasets/data/pfam-seed/Pfam-A.clans.tsv.gz` | ~500 KB | Maps family → clan for leak-safe splits |

## Intermediate Representations

### CherryML Adjacency Counts Tensors

| File | Source | Families | Match counts | Size |
|------|--------|----------|--------------|------|
| `data/seed_counts.npz` | Full Pfam seed | 27,481 | 206M | 3.4 MB |
| `data/pfam_counts.npz` | 100 selected | ~100 | 198K | 292 KB |

Format: `C_MM[tau_bin, aa1, aa2, aa3, aa4]` etc., 32 tau bins, 20 amino acids.
Used by `maraschino.py fit`.

### BW E-step Counts (from `--save-counts`)

| File pattern | Source | Contents |
|-------------|--------|----------|
| `pfam/counts_d3_fullseed_iter{N}.npz` | BW d3 fullseed | `B[20,20]`, `n_chi[32,32]`, `dom_B/I/D`, per-iteration |

Format: match counts, transition counts, per-domain match/insert/delete counts.
25 KB each. For direct BW vs CherryML comparison on same training data.

### Precompiled Training Pairs (X/A/Y format)

| Location | Source | Format | Status |
|----------|--------|--------|--------|
| `pfam/precompiled/` | Pfam seed train split | zstd-compressed JSONL shards | Building |

Each record: `{x: "ACDE...", a: "M5I1M3D2", y: "ACDF...", t: 0.305, fam: "PF00001", id: "abc123"}`.
Run-length encoded M/I/D alignment paths. Eliminates Stockholm parsing at training time.

Build with: `python precompile_pairs.py --msa-dir ~/bio-datasets/data/pfam-seed/ --split train --out pfam/precompiled/`

## Trained Parameters

### Best-fit (`params/best/`)

| File | Method | Domains | Data | Iters | Final LL |
|------|--------|---------|------|-------|----------|
| `bw_d3f2_pfam100_20iter.npz` | BW+subst | 3×2 | 100 families (98 used) | 20 | -6,269,649 |
| `bw_d5f2_pfam100_20iter.npz` | BW+subst | 5×2 | 100 families | 20 | -6,062,243 |
| `bw_d8f2_pfam100_15iter.npz` | BW+subst | 8×2 | 100 families | 15 | -6,250,762 |
| `bw_d3f2_fullseed_15iter.npz` | BW+subst | 3×2 | 5,996 families | 15 | -8,285,353 |
| `bw_d8f2_fullseed_15iter.npz` | BW+subst | 8×2 | 4,909 families | 15 | -6,774,478 |

Naming: `{method}_d{N}f{F}_{dataset}_{iters}iter.npz`

### Training Checkpoints (`pfam/`)

Gitignored. Regenerable by rerunning training.

| Pattern | Method | Notes |
|---------|--------|-------|
| `train_subst_*.npz` | BW+subst | Various budgets and configs |
| `train_stochastic_*.npz` | Stochastic EM | Fresh pairs each iteration |
| `adam_*.npz` | Adam gradient ascent | Custom VJP-based |
| `maraschino_*.npz` | CherryML Adam | On seed_counts adjacency tensor |
| `train_budget*.npz` | BW indel-only | Early runs, no per-domain subst |

### Maraschino Sweep (`results/maraschino/`)

Gitignored. 12 param files from d3-d20 sweep on seed_counts.

## Benchmarks

### ProteinGym DMS Indels

| Location | Size | Notes |
|----------|------|-------|
| `~/bio-datasets/data/proteingym/DMS_ProteinGym_indels/` | 6.4 MB | 66 assay CSVs |
| `~/bio-datasets/data/proteingym/DMS_indels.csv` | 30 KB | Reference with wildtype sequences |
| `experiments/proteingym_all_results.csv` | — | Pilot results (mean ρ=0.032) |

### Alignment Benchmarks

| Location | Size | Notes |
|----------|------|-------|
| `~/bio-datasets/data/balibase/` | 124 MB | BAliBASE alignment benchmark |
| `~/bio-datasets/data/treefam/` | 16 GB | TreeFam gene families (symlinked) |

## Storage Summary

| Category | Size | Committed to git? |
|----------|------|--------------------|
| Pfam seed MSAs | 794 MB | No (bio-datasets) |
| CherryML counts | 3.7 MB | No (data/) |
| Best params | ~400 KB | Yes (params/best/) |
| Training checkpoints | ~2 MB | No (pfam/) |
| Precompiled pairs | TBD | No (pfam/precompiled/) |
| Benchmarks | ~16 GB | No (bio-datasets) |
