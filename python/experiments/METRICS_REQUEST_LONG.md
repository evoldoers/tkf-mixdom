# Metrics Request: Long-Family Benchmark (101–500 columns)

This is the companion to `METRICS_REQUEST.md` for longer families.
The short benchmark (`unified_benchmark_spec.json`, ≤100 columns)
tests beam-search methods. This long benchmark tests only methods
that scale to longer alignments without beam search:
Felsenstein, PhyloHMM (partition-recon), ArDCA, CARABS, etc.

## Experimental setup

**EXACTLY** specified in `unified_benchmark_long_spec.json`.

For each of the 200 families:

| Field | Description |
|-------|-------------|
| `family` | Pfam accession (e.g. "PF01524") |
| `held_out` | Exact leaf name to predict |
| `remaining` | Exact list of retained leaf names (tree-traversal order) |
| `true_seq` | True held-out sequence as integer array (0–19 = AA) |
| `true_len` | Length of ungapped true sequence |
| `n_cols` | Number of MSA columns (101–500) |
| `K` | Number of retained leaves |

**Use these EXACTLY. Do not choose your own held-out leaf or filter
the family list.**

## Data locations

- MSAs: `~/bio-datasets/data/pfam/seed/{family}.sto` (Stockholm format)
- Trees: `~/bio-datasets/data/pfam-seed/trees/{family}.nwk` (FastTree ML,
  generated with `FastTree -lg -quiet -nosupport -nopr`)
- Split: `~/bio-datasets/data/pfam/seed/splits/v1.json` (val split)

## Metrics

Identical to `METRICS_REQUEST.md`:

1. **accuracy**: per-position accuracy weighted by FB match posteriors
2. **precision**: matches / pred_len
3. **recall**: matches / true_len
4. **log_prob**: sum of log P(true_char | model) across non-gap columns
5. **pred_seq**: predicted sequence as integer array

## Scoring pair HMM

Use the SAME TKF92 pair HMM for FB-aligning pred vs true:
- ins_rate = 0.046, del_rate = 0.054, ext = 0.68
- LG08 substitution matrix
- t = mean of FastTree pairwise distances from held-out to retained
  leaves, divided by 2

The `score_prediction` function in
`experiments/unified_reconstruction_benchmark.py` implements this.

## Output format

JSON with one entry per family (same format as the short benchmark):

```json
{
  "family": "PF01524",
  "held_out": "CPSF1_HUMAN/800-879",
  "accuracy": 0.65,
  "precision": 0.98,
  "recall": 0.85,
  "log_prob": -245.3,
  "pred_seq": [3, 14, 7, ...],
  "pred_len": 72,
  "true_len": 80,
  "time": 4.1
}
```

## For CARABS

To run CARABS on this benchmark:

1. Load `unified_benchmark_long_spec.json`
2. For each family, read the MSA and tree from the paths above
3. Remove the `held_out` leaf; use `remaining` as input
4. Predict the held-out leaf's sequence
5. Score using `score_prediction` (or replicate its TKF92 FB scoring)
6. Save results to `carabs_long_benchmark_results.json`

The spec file is self-contained: it has the family list, held-out
names, retained names, and true sequences. No ambiguity.

## Families

200 families from the Pfam val split, filtered to:
- 5 ≤ n_seqs ≤ 50
- 101 ≤ n_cols ≤ 500
- Disjoint from the short benchmark (≤100 columns)

Column distribution: min=101, median=194, max=500.
