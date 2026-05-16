# Comprehensive Model Evaluation Plan

## Goal

Establish which models explain sequence data best, whether order-1
(sequential) correlations are detectable, and whether TreeVarAnc's
ELBO captures them. Do this on controlled simulations first, then
on held-out Pfam data.

## Models under evaluation

| ID | Description | N | D | S_exch | Param file |
|----|-------------|---|---|--------|------------|
| F | Felsenstein LG08 | — | — | LG08 | built-in |
| M3 | MixDom static | 3 | 1 | shared | maraschino_d3_gamma4_marginal.npz |
| M6 | MixDom static | 6 | 1 | shared | maraschino_d6.npz |
| M6C10 | MixDom + C10 dyn | 6 | 10 | shared | maraschino_d6_c10_v2.npz |
| M8D2 | MixDom + D=2 dyn | 8 | 2 | per-dom | maraschino_d8_d2_fit.npz |
| M2D2 | MixDom + D=2 dyn | 2 | 2 | shared | maraschino_n2_d2_fit.npz |

Each MixDom model is used in two ways:
- **Distilled pair HMM**: for pairwise alignment LL
- **Distilled WFST**: for TreeVarAnc ELBO and reconstruction

## Simulation methodology

### Data generation

For each family in the Pfam test set:
1. Parse the real MSA, build NJ tree from pairwise distances (LG08)
2. Optionally scale branch lengths by a factor (1.0, 2.0) to explore longer branches

For each generating model G ∈ {TKF92/LG08, MixDom-M6}:
3. Simulate a ROOT SEQUENCE by:
   - Drawing root length L from geometric(1-r) where r is the model's
     fragment extension rate. Resample until L is within 50% of the
     observed MSA column count (to match Pfam alignment size).
   - Drawing root residues from the model's stationary distribution.
     For MixDom: draw domain assignments, then within each domain draw
     fragment class assignments, then draw residues from class equilibrium.
4. Evolve the root sequence down the tree using the PAIR HMM:
   - For each branch (parent → child), simulate from the compiled
     pair HMM (TKF91/TKF92/MixDom) at the branch length t.
   - This produces ancestor-descendant pairs with insertions and deletions.
   - The pair HMM simulation uses `simulate_pair` from
     `tkfmixdom/jax/simulate/pair_hmm.py` or equivalent.
5. Build the true MSA by tracing all leaf sequences back to the root
   (progressive alignment using the known tree and known homology).
6. Record:
   - True root sequence (with positions mapped to MSA columns)
   - True MSA (leaf sequences aligned to root)
   - True pairwise alignments (for each cherry pair)

### Why simulate from the pair HMM, not the WFST

The pair HMM IS the generative model for MixDom. It generates
ancestor-descendant pairs with the correct indel + substitution
+ domain structure. The WFST is a distillation (approximation)
used for inference, not generation.

For TKF92/LG08: the pair HMM is the standard TKF92 with LG08
substitution model. No domain structure, no order-1 correlations
in the generating process.

For MixDom: the pair HMM has domain structure, so adjacent positions
in the same domain share substitution parameters. This creates
order-1 correlations in the generated sequences.

### What the simulation controls for

- **Tree shape**: from real Pfam (realistic topology + branch lengths)
- **Sequence length**: matched to observed MSA size (±50%)
- **Indel rates**: from the model's trained parameters
- **Substitution model**: LG08 for TKF92, per-domain Q for MixDom

## Evaluation metrics

For each (generating model G, reconstruction model R, family F):

### 1. Order-1 signal (measured once per R)

KL(P(next|prev,cur) || P(next|cur)) averaged over the WFST's
stationary distribution. Units: nats per position. Measured at
both class-exposed (DA alphabet) and class-marginalized (A alphabet)
levels for models with dynamic classes.

### 2. Pairwise alignment LL

For each cherry pair in the tree:
- Align the two sequences using the pair HMM from model R
- Compute log P(observed pair | R, t) via forward algorithm
- Report mean LL per pair, normalized by alignment length

### 3. ELBO after 1 BP sweep

TreeVarAnc ELBO on the true MSA with model R's WFSTs.
After 1 sweep: this is approximately the Felsenstein column LL
plus the singlet HMM prior. The order-1 WFST factors have been
applied once but BP hasn't iterated.

### 4. ELBO after convergence (10 sweeps)

TreeVarAnc ELBO after BP converges. ELBO(10) - ELBO(1) = the
improvement from iterating BP, which measures the value of order-1
structure.

### 5. Root reconstruction accuracy (given correct MSA)

Predict root residues from TreeVarAnc posteriors. Compare to true
root sequence (known from simulation).
Report: fraction correct, per-column accuracy.

### 6. Alignment + reconstruction (combined)

NOT IMPLEMENTED YET. Would require:
- Running FSA or beam search alignment to produce an estimated MSA
- Then running TreeVarAnc on the estimated MSA
- Comparing both the alignment accuracy and the root reconstruction

For now, metrics 1-5 only (using true MSA for the tree-based metrics).

## Stratification

Families stratified by:
- **Divergence**: close (50-70% identity), moderate (30-50%), divergent (18-30%)
- **Length**: short (50-100 cols), long (100-200 cols)
- **Branch scale**: 1.0 (original), 2.0 (doubled for more divergence)

## Execution order

### Phase 1: Verify infrastructure
- [ ] Simulate one family from TKF92/LG08, verify MSA is reasonable
- [ ] Simulate one family from MixDom-M6, verify domain structure present
- [ ] Run all models on one simulated family, verify metrics are reasonable
- [ ] Check ELBO monotonicity (ELBO(10) >= ELBO(1))
- [ ] Check that generating model has best LL on its own data

### Phase 2: Simulation matrix
- [ ] TKF92/LG08 simulation × all reconstruction models × all strata
- [ ] MixDom-M6 simulation × all reconstruction models × all strata
- [ ] Analyze: where does BP help? Which strata are informative?

### Phase 3: Pfam evaluation
- [ ] Run all reconstruction models on real Pfam test set
- [ ] Focus on strata identified as informative in Phase 2
- [ ] Compare ELBO rankings to simulation predictions

## Expected outcomes

1. **BP should help when**: the generating model has order-1 structure
   (MixDom simulation) AND the reconstruction model captures it
   (MixDom WFST). On TKF92 simulation (order-0), BP should not help.

2. **ELBO ranking should match**: the true generating model should
   have the best ELBO on its own simulated data.

3. **Divergent families**: BP improvement should be largest on
   divergent families (weak tree signal, order-1 helps more).

4. **Pfam results**: should interpolate between TKF92 and MixDom
   simulation results, reflecting the true (unknown) level of
   order-1 structure in protein sequences.

## Composite Likelihood Reconstruction (NEW)

### Method

Given N observed sequences F_1,...,F_N at estimated distances t_1,...,t_N
from an unknown ancestor S, find:

    S* = argmax_S  Π_{n=1}^{N}  P(F_n | S, t_n)

using beam search over ancestor sequences. Each P(F_n | S, t_n) is computed
via the MixDom pair HMM forward algorithm.

### Beam state

At each position of the ancestor S:
- Current residue a (or (d, a) if class-exposed)
- Previous residue of S (for order-1 context)
- Pair HMM state (M/I/D) for each of the N descendants
- Cumulative log-likelihood

### Advantages

1. No tree topology needed — just pairwise distances
2. No MSA needed — each pair aligned independently to S
3. Uses pair HMM directly (no WFST distillation approximation)
4. Naturally handles indels
5. With class-exposed beam: tracks dynamic class of ancestor

### Cost

O(beam_width × N × A × |pair_HMM_states|) per beam step.
For N=8, A=20, beam=100, |states|=5: ~80K ops per step.

### Evaluation

Compare to TreeVarAnc reconstruction (given true MSA) and
TreeVarAnc reconstruction (given FSA MSA).
Report: matches, mismatches, insertions, deletions vs true root.
