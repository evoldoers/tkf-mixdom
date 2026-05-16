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

Generated: 2026-03-23
