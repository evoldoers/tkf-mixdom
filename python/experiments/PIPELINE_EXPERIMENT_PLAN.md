# Training Pipeline Experiment Plan

## Goal
Compare training pipelines with controlled data and iterations on Pfam-v1-train,
evaluated on Pfam-v1-test (3384 families).

## Pipelines to compare

### A. Maraschino → SVI-BW
1. Singlet-init maraschino (500 steps, ~2s)
2. Entropy-reg pairwise maraschino (5000 steps, ~7min)
3. Convert maraschino params to MixDom format
4. SVI-BW with maraschino seed, val-based early stopping

### B. Maraschino only (control)
1. Same as A.1-A.2
2. Evaluate directly (no BW refinement)

### C. BW-EM from random init
1. BW-EM, 15 iterations, fixed pairs

### D. SVI-BW from random init
1. SVI stochastic EM, 15 iterations
2. Val-based early stopping

## Controlled variables
- Run each pipeline at BOTH d3f2 AND d8f2 (repeat at both model sizes)
- Same training data: Pfam-v1-train (21667 families)
- Same number of pairs per iteration (e.g., 10K cherry pairs)
- Same number of EM iterations (15) for BW/SVI steps
- Wall time allowed to vary (reported but NOT controlled)
- Each model evaluated at its own t_rep from checkpoint
- Same eval: Pfam-v1-test (3384 families)
- Same hardware: single GPU

## Key hypothesis: d8 beats d3 with matched training data

Current d8 vs d3 comparison is confounded: d8 trained on fewer families
(4909 vs 5996) due to time-based budgets. Per-residue LL is already close
(-5.048 d3 vs -5.080 d8). With matched data + iterations, d8's extra
capacity should win.

Test: SVI with unlimited training data (full 21667 families), 15 iter,
same #pairs per iter. d8 should surpass d3 when not data-starved.

## Metrics
- **Pfam-v1-test LL/residue ± SE** (primary — length-normalized)
- Pfam-v1-test LL/pair ± SE (secondary)
- ProteinGym mean Spearman rho (54 assays)
- BAliBASE SP/TC (CherryML NJ tree, 20 cases)
- Domain weight entropy H/H_max
- Biophysical signal quality (hydrophobic/charged/G+P deviations)
- Wall time per pipeline (reported, not controlled)

## Launch order
1. Maraschino stages (CPU, ~8 min each for d3 and d8)
2. Pipeline A: SVI with maraschino seed (GPU, 15 iter)
3. Pipeline D: SVI from random init (GPU, 15 iter)
4. Pipeline C: BW from random init (GPU, 15 iter)
5. Evaluate all on Pfam-v1-test + ProteinGym + BAliBASE
