# Metrics Request for Unified Benchmark Comparison

## CRITICAL: Use the exact same experimental setup

All methods being compared MUST use **identical inputs** for each family:

1. **Same families**: the 200 Pfam val families listed in
   `experiments/unified_benchmark_spec.json`
2. **Same held-out leaf**: for each family, the `held_out` field in the
   spec JSON specifies the exact leaf to predict. Do NOT choose your own.
3. **Same retained leaves**: the `remaining` field lists the exact set of
   leaves to condition on (in tree-traversal order). Do NOT filter or
   reorder them.
4. **Same MSA**: read from `~/bio-datasets/data/pfam/seed/{family}.sto`
   (Stockholm format). Use the alignment columns as-is.
5. **Same tree**: FastTree ML tree from
   `~/bio-datasets/data/pfam-seed/trees/{family}.nwk`, generated with
   `FastTree -lg -quiet -nosupport -nopr`. Pre-computed for all 200
   families.

The spec file `unified_benchmark_spec.json` is the single source of
truth. It contains for each of the 200 families:
- `family`: Pfam accession (e.g. "PF00057")
- `held_out`: exact leaf name to predict (e.g. "LDLR_RABIT/182-218")
- `remaining`: list of retained leaf names
- `true_seq`: the true held-out sequence as integer array (0-19 = AA)
- `true_len`: length of the true (ungapped) sequence
- `n_cols`: number of MSA columns
- `K`: number of retained leaves

## Task

Held-out leaf prediction: for each family, remove the specified held-out
leaf from the MSA, predict its sequence from the remaining rows + tree.
The held-out leaf occupies a subset of MSA columns (non-gap columns for
that leaf).

## Metrics

For each held-out leaf prediction, compute ALL of the following.

### 1. Per-column accuracy (FB-aligned)

Run a pairwise Forward-Backward alignment (e.g. using a TKF92 pair HMM
with LG08, or any pair HMM) between the predicted sequence (ungapped)
and the true sequence (ungapped). From the alignment posteriors, compute:

- **matches**: expected number of positions where pred and true are
  aligned together (sum of match posteriors)
- **inserts**: positions where prediction has a residue but truth has a
  gap (overprediction). = pred_len - matches
- **deletes**: positions where truth has a residue but prediction has a
  gap (underprediction). = true_len - matches
- **precision**: matches / (matches + inserts) = matches / pred_len
- **recall**: matches / (matches + deletes) = matches / true_len
- **accuracy**: of the matched positions, fraction where
  pred_residue == true_residue (weighted by match posterior)

This is the PRIMARY metric. It handles predictions of any length,
including predictions shorter or longer than the truth.

### 2. Log P(true sequence)

For each MSA column where the held-out leaf has a residue, compute the
model's posterior probability of the true residue at that column:

```
log_prob = sum_c log P(true_char_c | remaining MSA, tree, model)
```

Report the raw sum (not per-position average).

### 3. Prediction metadata

- `pred_seq`: the predicted sequence as integer array (0-19)
- `pred_len`: length of predicted sequence
- `time`: wall-clock seconds for prediction

## Output Format

Save results as JSON with one entry per family, keyed by family ID.
Include all metrics above plus the prediction itself:

```json
{
  "family": "PF00057",
  "held_out": "LDLR_RABIT/182-218",
  "accuracy": 0.548,
  "precision": 0.951,
  "recall": 1.000,
  "matches": 35.2,
  "inserts": 1.8,
  "deletes": 0.0,
  "log_prob": -105.1,
  "pred_seq": [3, 14, 7, ...],
  "pred_len": 37,
  "true_len": 37,
  "time": 2.3
}
```

## Scoring pair HMM for FB alignment

To ensure comparable accuracy numbers across methods, use the SAME
scoring pair HMM for the FB alignment between pred and true. The
unified benchmark uses a TKF92 pair HMM with:
- ins_rate = 0.046, del_rate = 0.054, ext = 0.68
- LG08 substitution matrix
- evolutionary time t = mean of FastTree pairwise distances from
  held-out to retained leaves, divided by 2

The `score_prediction` function in
`experiments/unified_reconstruction_benchmark.py` implements this
exactly. External methods should either use this function directly
or replicate its logic.

## Families

The 200 families are from the Pfam val split (2,430 families total),
filtered to 5 ≤ n_seqs ≤ 50 and n_cols ≤ 100. The exact list is in
`experiments/unified_benchmark_spec.json`.
