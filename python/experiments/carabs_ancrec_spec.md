# CARABS Ancestral Sequence Reconstruction: Design Spec

## 1. Problem Statement

Given an MSA of N extant protein sequences, a phylogenetic tree with branch
lengths, and a gap pattern for the target ancestor, predict the amino acid
at each ungapped ancestor position. The model is trained and evaluated by
holding out each leaf sequence in turn, treating the remaining sequences as
the "extant" MSA, and comparing the predicted ancestral characters to the
true withheld leaf.

**Baseline performance** (from `ancrec_clean_n100.json`, 850 TreeFam families):
- Felsenstein column-by-column MAP: **39.1%** mean identity (std 19.1%)
- Variational ancestral reconstruction (BURL): **39.4%** mean identity
- Performance degrades with evolutionary distance (tau):
  - tau < 0.05: 48.8%
  - tau 0.05-0.1: 42.9%
  - tau 0.1-0.2: 41.0%
  - tau 0.2-0.5: 35.4%
  - tau > 0.5: 24.0%

The goal is a learned model that substantially outperforms these baselines,
particularly by exploiting sequential context (neighboring columns are not
independent due to structural and functional constraints).


## 2. Architecture

### 2.1 Overview: Felsenstein-Conditioned CARABS with ESM-2 Features

A two-stage architecture:

1. **Feature extraction**: Pre-compute ESM-2 embeddings for each sequence
   (frozen, cached to disk). Compute Felsenstein conditional likelihoods and
   other per-column features from the MSA + tree (no learnable parameters).
2. **Contextual refinement**: A CARABS model processes the feature sequence
   along the ancestor positions, mixing information across columns via
   RotatingMamba monoid scans and across MSA rows via WeightedAvg reduction.

This is a **hybrid** design: ESM-2 embeddings provide rich per-residue
representations capturing protein language model knowledge, the Felsenstein
posterior provides the Bayes-optimal column-independent phylogenetic signal,
and the learned CARABS layers capture sequential dependencies that the
column-independent model misses.

### 2.2 Input Representation

The model operates on a tensor of shape `(B, R+1, L_anc)` where:
- B = batch size (number of families)
- R = number of extant sequences in the MSA
- +1 = one additional "query" row for the ancestor
- L_anc = number of ungapped ancestor positions

Each MSA column that maps to an ungapped ancestor position contributes one
position to the input. The ancestor row is masked (all mask tokens).

### 2.3 Per-Position Input Features (~662-dim)

For each row r and ancestor position j, the input feature vector is the
concatenation of:

1. **ESM-2 embedding** (640-dim): Pre-computed per-residue embedding from
   ESM-2 t30 (150M parameters). Provides rich protein language model
   features encoding local and global sequence context. Pre-computed once
   for all sequences, cached to disk as .npz files, loaded as frozen
   features at training time (no ESM inference during training).

2. **Felsenstein posterior** (20-dim): P(ancestor_aa = a | leaves, tree, LG+F)
   using standard pruning. The strongest single phylogenetic feature.
   Broadcast identically across all rows (it is a column-level feature).

3. **Branch distance to root** (1-dim): The sum of branch lengths from this
   row's leaf/node to the reconstruction target (root ancestor). Gives the
   model a notion of evolutionary distance — nearby leaves are more
   informative than distant ones.

4. **Conservation score** (1-dim): Fraction of leaf characters that agree
   with the plurality character at this column. Broadcast across rows.

Total per-position input dimension: **662**.

A learned linear projection maps this 662-dim input to d_model before
entering the CARABS blocks.

### 2.4 Row Label

Each row carries a scalar **branch length distance to root**: the sum of
branch lengths along the path from that leaf (or internal node) to the
reconstruction target. This is provided as the 1-dim branch distance feature
described above. For the ancestor query row, the distance is zero.

### 2.5 CARABS Block Configuration

```python
MSABlockConfig(
    col_op="fused_rotating_mamba_wavg",  # Fused RotatingMamba + WeightedAvg
    row_reduce="weighted_avg",           # sigma(a*y+b) weighted mean
    row_reduce_kwargs={},                # no extra config needed
    d_model=128,
    bidirectional=True,
    rc="none",                           # proteins, no RC symmetry
    ffn_expansion=4,
    dropout_rate=0.1,
    include_column_summary=True,
    local_attn_window=32,                # local attention for precise neighbor context
)
```

**Why RotatingMamba + WeightedAvg:**
- **RotatingMamba** (registered as `rotating_mamba`, alias for
  `MambaMonoidColumnOp` with `n_rotating=8`) applies rotation to channel
  pairs before/after the associative scan, giving data-dependent gating
  with rotational expressiveness. The fused variant
  `fused_rotating_mamba_wavg` merges the column-op readout with the row
  reduction, eliminating the intermediate `(B, R, L, D)` tensor for better
  memory efficiency.
- **WeightedAvg** (`weighted_avg`): Each row produces a learned weight
  via `sigma(a*y + b)` and a message; the aggregate is the weighted mean.
  Much faster than ENN while sufficient for this task — the branch distance
  feature already encodes sequence informativeness, so the row reduction
  does not need complex pairwise interactions.

**Model hyperparameters:**
- d_model: 128
- n_layers: 6
- vocab_size: 21 (20 amino acids + gap token, though gaps are masked out in
  the ancestor row)
- max_seq_len: 2048 (covers 99%+ of Pfam domains)

### 2.6 Output Head

The model outputs logits `(B, R+1, L_anc, 21)`. We extract the ancestor
row logits `(B, L_anc, 21)` and take the first 20 channels (amino acids)
as the prediction. The output head is the standard tied-embedding LM head
from GeneralizedMSAModel.

**Loss**: Cross-entropy on the ancestor row at ungapped positions, comparing
the predicted distribution to the true withheld amino acid character.

### 2.7 Felsenstein Warm-Start (Optional)

Initialize the ancestor row embedding not with mask tokens but with a
learned projection of the 20-dim Felsenstein posterior:

```python
fels_proj = nn.Dense(d_model)(felsenstein_posterior)  # (B, L_anc, D)
```

This injects the Felsenstein prior directly into the ancestor representation,
giving the model a strong starting point that it can refine with context.


## 3. Training Protocol

### 3.1 Data Splits

| Split      | Source | N families | Usage |
|------------|--------|-----------|-------|
| Train      | Pfam   | 21,667    | Training |
| Validation | Pfam   | 2,430     | Early stopping, hyperparameter selection |
| Test       | Pfam   | 3,384     | Final evaluation |
| Test-OOD   | TreeFam clean | 1,000 | Out-of-distribution generalization |

### 3.2 ESM-2 Embedding Pre-Computation

ESM-2 is NOT currently installed. Setup:

```bash
pip install fair-esm torch
```

Pre-compute per-residue embeddings for all sequences:

1. Run ESM-2 t30 (150M params, 640-dim embeddings) on every sequence in the
   Pfam and TreeFam datasets. ESM-2 t30 fits comfortably on RTX 2080 Ti
   (11GB VRAM).
2. Save per-residue embeddings as `.npz` files (one per family or per
   sequence), keyed by sequence ID.
3. At training time: load cached embeddings from disk, no ESM inference
   needed. The ESM-2 model is never fine-tuned.

Estimated pre-computation cost:
- Pfam (~25K families, ~10 seqs each, mean length ~200): ~2 hours on 1 GPU
- TreeFam (~1K families): ~15 minutes
- Storage: ~20GB total (640 floats x ~50M residues x 4 bytes)

### 3.3 Training Data Generation

For each training family:
1. Load the Pfam seed MSA, NJ tree, and pre-computed ESM-2 embeddings.
2. For each leaf l in the tree:
   a. Remove leaf l from the MSA.
   b. Compute Felsenstein posteriors at the parent of l using remaining leaves.
   c. The target is the true amino acid sequence of l at columns where l is ungapped.
3. Each (family, held-out-leaf) pair is one training example.

**Data augmentation:**
- Subsample leaves: randomly drop 10-50% of non-held-out leaves to create
  diversity in the number of informative sequences.
- Branch length jitter: multiply all branch lengths by exp(N(0, 0.1)) to
  make the model robust to tree estimation error.

### 3.4 Batching

Families vary in size (L_anc: 30-2000, R: 3-200). Use:
- Geometric padding on L_anc (bins: 32, 48, 64, 96, 128, ..., 2048)
- Fixed max R = 64 (subsample if more, pad if fewer)
- Dynamic batching: pack families to fill ~8192 total tokens per batch

### 3.5 Optimization

- Optimizer: AdamW (lr=3e-4, weight_decay=0.01, warmup=1000 steps)
- Training steps: 100K
- Gradient clipping: max_norm=1.0
- Mixed precision: bfloat16 for forward/backward, float32 for loss

### 3.6 Evaluation Metrics

1. **Per-position accuracy**: fraction of ancestor positions where
   argmax(predicted) == true character.
2. **Per-position cross-entropy**: mean NLL in nats.
3. **Per-family identity**: accuracy averaged per family, then across families
   (avoids long families dominating).
4. **Stratified by tau**: report accuracy in tau bins [0, 0.05), [0.05, 0.1),
   [0.1, 0.2), [0.2, 0.5), [0.5+).
5. **Improvement over Felsenstein**: per-family delta (model accuracy -
   Felsenstein accuracy), reported as mean and per-tau-bin.


## 4. CARABS Modules: Reuse vs New

### 4.1 Reused from CARABS (unchanged)

| Module | Purpose |
|--------|---------|
| `carabs/jax/model.py` | GeneralizedMSAModel, GeneralizedMSABlock |
| `carabs/jax/column_ops/rotating_mamba.py` | RotatingMamba column operation |
| `carabs/jax/column_ops/fused_rotating_mamba_wavg.py` | Fused RotatingMamba + WeightedAvg (optimized kernel) |
| `carabs/jax/row_reductions/weighted_avg.py` | WeightedAvg row reduction |
| `carabs/jax/column_summary.py` | Cross-row statistics |
| `carabs/core/config.py` | MSABlockConfig |
| `carabs/core/registry.py` | Column op / row reduction registries |

### 4.2 Reused from tkf-mixdom (unchanged)

| Module | Purpose |
|--------|---------|
| `tkfmixdom/jax/core/ctmc.py` | Substitution matrix computation (LG+F) |
| `tkfmixdom/jax/core/protein.py` | LG rate matrix |
| `tkfmixdom/jax/tree/felsenstein.py` | Felsenstein pruning algorithm |
| `tkfmixdom/jax/tree/progrec_felsenstein.py` | Profile-based Felsenstein |
| `tkfmixdom/jax/util/io.py` | TreeNode, Newick parsing, sequence I/O |

### 4.3 New Code Required

| Module | Purpose | Estimated LOC |
|--------|---------|---------------|
| `evaluation/ancrec_data.py` | Data loader: Pfam/TreeFam families -> (MSA, tree, held-out leaf, features) | 300 |
| `evaluation/ancrec_features.py` | Feature engineering: ESM-2 embedding loading, Felsenstein posteriors, branch distances, conservation | 250 |
| `evaluation/esm_precompute.py` | ESM-2 embedding pre-computation script (batch inference, .npz caching) | 200 |
| `evaluation/train_ancrec.py` | Training loop with evaluation | 400 |
| `evaluation/ancrec_baselines.py` | Felsenstein and consensus baselines for comparison | 150 |
| Total | | ~1300 |


## 5. Why This Architecture

### 5.1 Why Not Column-Independent?

Column-independent models (Felsenstein, consensus) achieve ~39% identity.
Protein sequences have strong sequential correlations: secondary structure
elements span 3-20 residues, conserved motifs create long-range dependencies,
and insertion/deletion events affect neighboring columns jointly. A model
that can look at flanking columns should predict ambiguous positions better.

### 5.2 Why Not a Pure Transformer?

Full self-attention over (R x L_anc) positions is O(R^2 L^2), prohibitively
expensive for large MSAs. CARABS replaces row-attention with O(R) exchangeable
reductions and column-attention with O(L log L) monoid scans, making it
tractable for MSAs with hundreds of sequences and thousands of columns.

### 5.3 Why Hybrid (ESM-2 + Felsenstein + Learned)?

ESM-2 embeddings provide rich per-residue protein language model features
trained on billions of sequences — encoding structural, functional, and
evolutionary information that would be impractical to learn from scratch on
our dataset. The Felsenstein posterior is the Bayes-optimal column-independent
predictor given the substitution model and tree. By combining both as frozen
features, the CARABS model only needs to learn the residual: how to integrate
ESM-2's sequence-level knowledge with Felsenstein's phylogenetic signal, and
capture sequential context effects that column-independent models miss.

### 5.4 Why RotatingMamba + WeightedAvg?

- **RotatingMamba** provides data-dependent gating with rotational channel
  mixing, capturing variable-length sequential dependencies in proteins.
  The fused variant (`fused_rotating_mamba_wavg`) eliminates the
  intermediate `(B, R, L, D)` tensor between the column-op readout and row
  reduction, reducing peak memory and improving throughput.
- **WeightedAvg** is a simple `sigma(a*y+b)` weighted mean across rows.
  It is substantially faster than ENN while sufficient for this task: the
  branch distance feature already encodes per-row informativeness, and
  the column summary provides cross-row statistics. The weighted average
  lets the model learn which rows to attend to without the overhead of
  pairwise interactions.


## 6. Expected Performance

### 6.1 Performance Targets

| Method | Mean Identity | Notes |
|--------|--------------|-------|
| Felsenstein (baseline) | 39.1% | Column-independent, no learning |
| CARABS column-only (no row pool) | 42-44% | Sequential context, no cross-row |
| CARABS full (target) | 45-50% | Sequential + cross-row context |
| Upper bound (conservation ceiling) | ~70% | Columns with >90% conservation |

Rationale for targets:
- MSA transformers (e.g., MSA Transformer, ESM-MSA) achieve 5-15% absolute
  improvement over column-independent methods on contact prediction. Ancestral
  reconstruction is a more direct target, so similar gains are plausible.
- ESM-2 features should provide additional lift by encoding protein language
  model knowledge that is complementary to the phylogenetic signal.
- The improvement should be largest at intermediate tau (0.05-0.2) where
  there is enough divergence for context to help but not so much that the
  signal is lost.

### 6.2 Ablation Predictions

| Ablation | Expected Effect |
|----------|----------------|
| Remove ESM-2 features (keep Felsenstein) | -2 to -4% (lose PLM knowledge) |
| Remove Felsenstein features (keep ESM-2) | -3 to -5% (lose phylogenetic signal) |
| Remove both ESM-2 and Felsenstein | -5 to -8% (lose both priors) |
| Remove conservation score | -0.5 to -1% (lose column-level cue) |
| Remove branch distance | -0.5 to -1% (lose per-row distance weighting) |
| Column-only (no row reduction) | -2 to -4% (lose cross-row mixing) |
| No local attention | -1 to -2% (lose precise local context) |
| Unidirectional scan | -1 to -2% (lose bidirectional context) |
| ENN instead of WeightedAvg | +0 to +0.5% (marginal, much slower) |


## 7. Compute Requirements

### 7.1 Model Size

- d_model=128, n_layers=6, input_dim=662
- Estimated parameters: ~5M (input projection adds ~85K over the 4M base)
- Memory per example: ~50MB for (R=64, L=512) at float32

### 7.2 ESM-2 Pre-Computation

- Model: ESM-2 t30 (150M params, 640-dim per-residue embeddings)
- Hardware: 1x RTX 2080 Ti (11GB VRAM) — sufficient for ESM-2 t30
- Pfam (~250K sequences, mean length ~200): ~2 hours on 1 GPU
- TreeFam (~10K sequences): ~15 minutes
- Storage: ~20GB as .npz files
- One-time cost, cached permanently

### 7.3 Training

- Hardware: 1x RTX 3070 (8GB VRAM) or 1x A100 (40GB)
- On RTX 3070: bfloat16, batch_size=2, gradient accumulation=4
  - ~0.5 sec/step, 100K steps = ~14 hours
- On A100: batch_size=8, no gradient accumulation
  - ~0.2 sec/step, 100K steps = ~5.5 hours

### 7.4 Inference

- ESM-2 embedding lookup: ~1ms per family (cached, disk read)
- Felsenstein feature computation: ~5ms per family (JAX, CPU)
- CARABS forward pass: ~20ms per family (GPU)
- Full Pfam test set (3,384 families): ~1.5 minutes


## 8. Clade-Aware Masking (Preventing Leakage)

When the held-out leaf has close relatives in the MSA, the model could
achieve high accuracy by simply copying the closest relative. To ensure the
model learns genuine ancestral reconstruction rather than nearest-neighbor
copying:

1. **Clade masking**: Use PhyloTree (from carabs/nona/phylo.py) to identify
   clades. When holding out a leaf, also mask the characters of its
   clade-mates in 30% of training examples. This forces the model to use
   more distant sequences.

2. **Distance-weighted loss**: Weight the loss contribution of each family
   inversely to the minimum leaf-to-ancestor distance. Families where the
   held-out leaf is very close to its parent (trivial reconstruction) get
   lower weight.

3. **Evaluation stratification**: Always report results stratified by tau
   (branch length to ancestor). The model's value is demonstrated at
   intermediate and large tau, not at tau < 0.01 where copying suffices.


## 9. Comparison to Existing Methods

| Method | Type | Uses tree? | Uses sequence context? | Uses PLM? | Speed |
|--------|------|-----------|----------------------|-----------|-------|
| Felsenstein MAP | Probabilistic | Yes | No (column-independent) | No | Fast |
| BURL variational | Probabilistic | Yes | Yes (variational) | No | Slow |
| ESM-IF | Neural (inverse folding) | No | Yes (structure-conditioned) | Yes | Medium |
| ProteinMPNN | Neural (graph) | No | Yes (structure-conditioned) | No | Medium |
| **CARABS AncRec** | **Neural (MSA)** | **Yes** | **Yes (RotMamba scan + WeightedAvg)** | **Yes (ESM-2)** | **Medium** |

Key differentiators: CARABS AncRec combines phylogenetic tree information
(via Felsenstein features and branch-distance row labels), protein language
model knowledge (via frozen ESM-2 embeddings), and sequential context (via
RotatingMamba monoid scans). Inverse folding methods like ESM-IF use 3D
structure instead of a tree, which is complementary but requires structure
availability.


## 10. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Overfitting to Pfam training set | Dropout, weight decay, TreeFam OOD test set |
| Nearest-neighbor copying instead of reconstruction | Clade masking, tau-stratified evaluation |
| Felsenstein features dominate (model learns identity) | Ablation: train without Felsenstein features to verify the model can still learn |
| ESM-2 pre-computation bottleneck | One-time cost; ESM-2 t30 is small enough for RTX 2080 Ti |
| ESM-2 storage overhead (~20GB) | Acceptable; compress with float16 if needed (~10GB) |
| Variable-size MSAs cause JIT recompilation | Geometric padding (already standard in tkf-mixdom) |
| Large MSAs (R>100) blow up memory | Subsample to R=64, fused RotMamba+WeightedAvg reduces peak memory |
| Gap pattern errors propagate | Train with both Fitch-parsimony and TKF92-triad gap patterns; evaluate robustly |


## 11. Future Extensions

1. **Structure conditioning**: Add 3D structure features (from ESMFold or
   AlphaFold predictions) as additional input channels. This would combine
   the tree-based, PLM-based, and structure-based approaches.

2. **Internal node reconstruction**: Extend from leaf hold-out to predicting
   true internal ancestral sequences (requires simulated data or resurrected
   ancestral proteins for validation).

3. **Joint gap + character prediction**: Instead of taking the gap pattern
   as input, predict it jointly. The model would output a 21-class distribution
   (20 AA + gap) at each column.

4. **Multi-task pretraining**: Pretrain the CARABS backbone on the standard
   MSA masked language modeling task (Task 2 from the CARABS task catalog),
   then fine-tune on ancestral reconstruction.

5. **Iterative refinement**: Use the model's own predictions to improve the
   MSA (via progressive alignment with predicted ancestral sequences), then
   re-run the model on the improved MSA.

6. **ESM-2 fine-tuning**: After establishing frozen-feature baseline, explore
   LoRA or last-layer fine-tuning of ESM-2 embeddings for marginal gains.

## Corrections (March 26 2026)

### Target row handling
- The target (root/ancestor) row must be **removed** from the MSA input, not masked.
- Columns that become all-gap when the target row is removed should also be removed.
- The model predicts the target sequence at every position in every remaining column,
  with **gaps as a 21st token** in the output vocabulary.
- The ML gapped target sequence and its alignment to the true target can then be read off
  by taking argmax over 21 classes per column.
- This means the model jointly predicts ancestor presence/absence AND character identity.
