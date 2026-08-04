# BAliBASE MSA Benchmark Results

## TKF92 ProgRec (Viterbi, no MixDom)

| Tool | SP | TC | Cases | Guide tree |
|------|-----|-----|-------|-----------|
| MAFFT | 0.705 | 0.503 | 20 | — |
| TKF92 ProgRec | 0.521 | 0.300 | 60 | CherryML NJ |
| TKF92 ProgRec | 0.464 | 0.152 | 20 | MAFFT tree |
| FSA TKF92 | 0.454 | 0.184 | 10 | — (pairwise) |
| TKF92 ProgRec | 0.438 | 0.189 | 20 | CherryML NJ |
| Historian | 0.433 | 0.177 | 20 | built-in NJ |

Notes:
- CherryML NJ uses TKF92 FB-based pairwise distances (proper model-based)
- 60-case CherryML NJ result is higher than 20-case because later BAliBASE
  families are more conserved
- FSA uses AMAP-style sequence annealing with pairwise TKF92 posteriors
- Full 386-case run OOM'd at case 61 (CherryML distance on large family)

## MixDom ProgRec
Not yet benchmarked (compose_intersect_virtual is 394× faster than Python
reference but still ~3s per leaf-pair Viterbi on 20aa).

## FSA-MSA: MixFrag(F=2) vs TKF92, cherry-EM Pfam-train params (2026-06-24)

BAliBASE 3 `bali3pdbm`, all **120 families** (full coverage), FSA sequence
annealing on pairwise Pair-HMM posteriors with per-pair branch-length τ
(Newton-Raphson on E[LL(τ)]); LG08 emissions; core-column scoring. Both
models are fit by the **same** summarised-count cherry EM (supp. B.6) on the
Pfam v1 train split (19,660 families); F=1 reduces exactly to TKF92, so this
is a matched single-variable (fragment-mixture) comparison via the same
driver (`experiments/fsa_mixfrag_balibase.py`).

| Model | params (`params/best/`) | SP mean | TC mean | micro-F1 (P / R) |
|-------|-------------------------|---------|---------|-------------------|
| TKF92 (F=1) | `cem_tkf92_pfamTrain.npz` | 0.7745 | 0.6170 | 0.2877 (0.181 / 0.705) |
| MixFrag (F=2) | `cem_mixfrag_F2_pfamTrain.npz` | **0.7902** | **0.6405** | **0.3040** (0.192 / 0.727) |
| Δ (F2 − F1) | | +0.0157 | +0.0235 | +0.0163 |

Per-family (matched, 120/120): SP win/tie/loss = 47/34/39; TC = 40/46/34.
Gains concentrate in the mean (harder families) more than the median,
echoing the +9.9k held-out-LL edge of F=2 at training time. The corpus-micro
F1 is the soft/expected presence-F1 (diffuse posterior mass → low precision);
SP/TC are the headline alignment-accuracy metrics.

Raw per-family JSONs `experiments/balibase_{mixfrag_F2,tkf92_cherryEM}.json`
are gitignored (per-run results → S3 `results/<run-id>/` via
`scripts/upload_results_to_s3.sh`).

Generated: 2026-03-23; FSA MixFrag/TKF92 section added 2026-06-24
