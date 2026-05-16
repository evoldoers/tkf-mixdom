# ProteinGym Evaluation Plan

## Target: DMS Indel Benchmark (74 assays)

MixDom's domain-heterogeneous indel rates give it a genuine advantage over
HMMER's heuristic gap penalties on the ProteinGym indel fitness prediction task.

### Leaderboard context
| Model | Spearman ρ (mean, 74 assays) |
|-------|-----|
| PoET 200M (autoregressive LM) | 0.517 |
| Tranception-L (no retrieval) | 0.437 |
| **Target for MixDom** | **0.42–0.45** |
| HMM baseline (HMMER) | 0.389 |
| ProtGPT2 | 0.191 |

### Why MixDom should work
- TKF computes proper P(indel) from birth/death rates (not heuristic gap penalties)
- Domain-heterogeneous rates: IDR insertions scored differently from core insertions
- Order-1 context dependence captures local sequence composition effects

### Pipeline

1. **Download** (bio-datasets/fetch/proteingym/fetch.py)
   - DMS_indels.zip (~200 MB): 74 assay CSVs with variant→fitness
   - DMS_indels.csv: metadata (protein name, MSA filename, etc.)
   - DMS_MSAs.zip (5.2 GB): pre-built MSAs for per-protein training

2. **Per-protein BW fine-tuning** (~15 min/protein, 18 GPU-hours total)
   - Initialize from global Pfam-trained model (params/best/bw_d3f2_*.npz)
   - Run 5–10 BW iterations on each assay protein's MSA
   - Saves per-protein checkpoint

3. **Scoring** (new: `score_single_sequence()`)
   - Build singlet HMM from per-protein MixDom params
   - Forward algorithm: log P(seq) for wildtype and each mutant
   - Score = log P(mutant) - log P(wildtype)
   - Handles variable-length sequences natively (TKF geometric length)

4. **Evaluation**
   - Spearman ρ per assay, compare to baselines
   - Use ProteinGym's official evaluation script

### Missing code
- `score_single_sequence(seq, params)` — singlet HMM forward, ~50 lines JAX
- `train_proteingym.py` — per-protein BW wrapper
- `score_proteingym.py` — variant scoring script

### Pilot (1 day)
Pick 5 diverse indel assays, run full pipeline, check if ρ > 0.389.
If yes, scale to all 74. If no, analyze why and adjust.

### Caveats
- Substitution benchmark: MixDom without per-protein MSA will likely lose to HMMER
- The story is cleanest for indels only
- Per-protein fine-tuning adds complexity but is standard practice (EVE does it too)
