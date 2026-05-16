# Pfam-v1-test Results

Evaluated on 3,310 test families (of 3,384 in v1.json test split).
All models at t=0.305, per-domain substitution where available.

## LL/pair ± SE

| Model | LL/pair ± SE | Training |
|-------|-------------|----------|
| BW d3f2 fullseed | -736.63 ± 10.04 | EM, 15 iter, 5996 fam, 5.4h |
| SVI d3f2 fullseed | -737.14 ± 10.05 | SVI stoch EM, 15 iter, 8h |
| BW d8f2 fullseed | -740.74 ± 10.09 | EM, 15 iter, 4909 fam, 7.2h |
| Mara d3 entreg | -742.71 ± 10.12 | CherryML Adam, 5000 steps, 7 min |
| Mara d8 entreg | (in progress) | CherryML Adam, 5000 steps, 7 min |

## LL/residue (length-normalized)

| Model | LL/residue |
|-------|-----------|
| BW d3f2 | -5.048 |
| SVI d3f2 | -5.052 |
| BW d8f2 | -5.076 |
| Mara d3 entreg | -5.090 |
| Mara d8 entreg | ~-5.108 (partial) |

## Key findings

1. BW d3 and SVI d3 are statistically indistinguishable (0.5 nats/pair)
2. Maraschino d3 is 6 nats/pair behind BW d3 (7 min vs 5.4h training)
3. Per-residue gap: 0.042 nats (BW -5.048 vs Mara -5.090)
4. All d3 models within 0.04 nats/residue — training method matters less than model size
5. d8 models consistently behind d3 (likely undertrained — fewer families, different t)

Generated: 2026-03-23
Split: ~/bio-datasets/data/pfam-seed/splits/v1.json (test)
