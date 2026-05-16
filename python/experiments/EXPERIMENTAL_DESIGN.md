# Experimental Design for MixDom Parameter Estimation

Adapted from the information-maximization philosophy in carabs/doc/experimental_design.md.

## The Question

Which method produces the best MixDom parameters, and how much compute does it need?

### Configuration Space

| Factor | Levels | Notes |
|--------|--------|-------|
| Estimation method | BW-EM, Adam (stochastic), Maraschino (CherryML) | Core comparison |
| N domains | 3, 5, 8 | Model complexity |
| Training data | 100 families, full seed (21K train) | Data scale |
| Budget | 1h, 4h, 8h | Compute investment |
| Substitution model | shared LG08, per-domain pi | Emission complexity |

Full factorial: 3 × 3 × 2 × 3 × 2 = 108 runs. At ~4h each = 432 GPU-hours.
Too much. Use fractional design + sequential elimination.

## Phased Design

### Phase 0: Smoke test (2 GPU-hours)
- Verify all three methods run without crashes on 3 families, 2 domains
- Confirm Adam produces decreasing loss, BW produces increasing LL
- Fix any bugs before investing compute

### Phase 1: Method screening (16 GPU-hours)
Fix: d3, 100 families, per-domain pi, shared seed.

| Run | Method | Budget | GPU-hours |
|-----|--------|--------|-----------|
| 1 | BW-EM | 2h actual | 2 |
| 2 | Adam (stochastic) | 2h actual | 2 |
| 3 | Maraschino (CherryML) | 2h | 2 |
| 4 | BW-EM | 4h actual | 4 |
| 5 | Adam (stochastic) | 4h actual | 4 |
| 6 | Maraschino + BW init | 2h | 2 |

**Evaluation:** Held-out LL on test split (clan-aware, v1.json).
All methods evaluated on the SAME held-out families using the SAME
eval function (singlet + pair FB forward).

**Phase gate:** If Adam and BW produce indistinguishable held-out LL
after 2h, there's no method comparison story. Focus on the best one.

### Phase 2: Scale and complexity (24 GPU-hours)
Take the best method from Phase 1. Vary:
- d3 vs d5 vs d8 on full seed (8h each = 24h)
- Compare BIC/AIC for model selection

### Phase 3: Full production run (16 GPU-hours)
- Best method × best d × full seed × 8h × 2 random seeds = 16h
- Report final params with error bars across seeds
- Save to params/best/ with systematic naming

**Total: ~58 GPU-hours** (fits in ~30h on 2× RTX 2080 Ti).

## Evaluation Protocol

### Held-out LL (primary metric)
- Use test split from v1.json (3384 families, 59 benchmark families)
- Build MixDom transition matrix from trained params
- Run constrained 1D FB on cherry pairs from test families
- Report total LL and LL/pair

### Maraschino-compatible counts comparison
- Save counts tensor during BW E-step (--save-counts)
- Fit maraschino on the same counts
- Compare LL under both parameterizations

### Statistical reporting
- Mean ± SE across seeds (Phase 3)
- Paired comparison: BW vs Adam on same test pairs
- Report wall-clock time per method

## Phase Gates

1. **After Phase 1:** If no method beats the others by >1% relative LL,
   the methods are equivalent and we pick the fastest.
2. **After Phase 2:** If d8 doesn't beat d3, the extra domains aren't
   justified. Report d3 as the recommended model.
3. **Budget overrun:** If any phase exceeds allocation by >20%, stop
   and diagnose.

## Pre-registered Comparisons

1. **Primary:** BW-EM vs Adam on test LL (Phase 1)
2. **Secondary:** BW vs Maraschino on same training data (Phase 1)
3. **Exploratory:** d3 vs d5 vs d8 (Phase 2), everything else
