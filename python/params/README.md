# Trained Parameters

## Directory Structure

```
params/
  best/     ← Current best-fit parameters (committed to git)
  test/     ← Small params for unit tests (committed to git)
```

All other params (.npz files from training runs, sweeps, etc.) live in
`pfam/` or `results/` and are gitignored. Only curated best-fits and
test fixtures belong here.

## Naming Convention

```
{method}_d{N}f{F}_{dataset}_{iters}iter.npz
```

- **method**: `bw` (Baum-Welch EM) or `mara` (Maraschino CherryML)
- **N**: number of domains
- **F**: number of fragments per domain
- **dataset**: `pfam100` (100 seed families), `pfamSeed` (full ~27K seed), etc.
- **iters**: number of training iterations

Examples:
- `bw_d3f2_pfam100_20iter.npz` — BW, 3 domains, 2 fragments, 100 Pfam families, 20 iterations
- `bw_d8f2_pfam100_15iter.npz` — BW, 8 domains, 2 fragments
- `mara_d3_pfamSeed_5000step.npz` — Maraschino Adam, 3 domains, 5000 steps

## Current Best-Fit Parameters

| File | Domains | Data | Method | Notes |
|------|---------|------|--------|-------|
| `best/bw_d3f2_pfam100_20iter.npz` | 3×2 | 100 Pfam families (98 used) | BW+subst | Per-domain π, pi_pseudo=3, 11456 pairs |

## Adding New Best-Fits

When a training run produces better results (higher LL on comparable data,
or same LL on more data), copy the checkpoint here with the systematic name:

```bash
cp pfam/train_subst_d8_8h.npz params/best/bw_d8f2_pfam100_15iter.npz
```

Update this README and regenerate the biophysical analysis:
```bash
uv run python results/generate_analysis.py
```

## Test Parameters

Small parameter files for unit tests. These should be minimal (few domains,
small alphabet if possible) and committed to git. They live in `params/test/`
and are referenced by test files via relative imports.
