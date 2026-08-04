# Indel-model likelihood comparison on v3_val (path-conditioned vs order-summed)

Held-out per-pair log-likelihood for three indel–substitution models, all
trained on the **same** data (`precompiled_v3_train`, 1,134,482 cherry pairs)
and scored on the **same** held-out set (`precompiled_v3_val`, 116,640 pairs),
with LG08 substitution and per-pair branch length `t`. Values are mean
log-likelihood per pair (nats).

| Model | Path-conditioned | Order-summed |
|---|---:|---:|
| **MixDom-d3f1 (per-domain)** | **−718.57** | **−713.02** |
| MixFrag (F=2) | −726.44 | −720.09 |
| TKF92 | −726.30 | −721.97 |

**Caption.** *Naive path-conditioning fixes a single indel ordering, which
forces the latent fragment structure in MixFrag/MixDom (each observed indel is
committed to one fragtype/domain); we therefore also report the
ordering-summed likelihood — the match-constrained 2D forward that holds the
matches at the observed alignment but sums over all indel orderings between
match anchors — which marginalizes that latent structure out.* Under the
order-summed (proper marginal) likelihood every model gains 4–6 nats (MixFrag
gains the most, +6.35); MixDom-d3f1 is best under both conventions, leading by
≥7 nats.

**Note on the MixDom-d3f1 checkpoint.** These numbers use
`pfam/svibw_d3f1_perdomain_v3.npz` — a **per-domain** d3f1 (each of the 3
domains has its own substitution matrix, trained 540 svi-BW iters on v3 via
`--classdist-init identity --freeze-classdist`), matching the architecture of
the original released d3f1. An earlier checkpoint frozen at iter 44 with a free
3-class site mixture (`svibw_d3f1_v3_FROZEN_iter45`) was undertrained and
collapsed to a near-shared substitution (class π's differed by only 0.004); it
scored −725.85 / −717.41 here and 0.771/0.612 SP/TC on BAliBASE. The converged
per-domain model above scores 0.812/0.677 SP/TC on BAliBASE (120 families,
`bali3pdbm`), exceeding the old released d3f1 (0.807/0.662) and MAFFT
(0.797/0.653).

## Provenance

- **Convention — path-conditioned:** 1D forward–backward along the fixed
  observed alignment; indel orderings NOT summed.
- **Convention — order-summed:** matches fixed to the observed alignment;
  the 2D pair-HMM forward sums over all indel orderings between match anchors
  (`mask_emissions_match_aligned` + `forward_backward_2d(forward_only=True)`).
- **Numerics:** float64 throughout; LG08 `(S, π)`; τ per pair (exact, not
  binned); no gap-length or match-run caps.
- **DP fixes used** (canonical `~/tkf-mixdom`, `python/tkfmixdom/jax/dp/hmm.py`,
  commit `20768cb23`): forward-only 2D forward routed by state count —
  small-ns via the linear-space anti-diagonal wavefront
  (`_forward_2d_core_diag(return_chart=False)`), large-ns via the
  associative-scan row-scan. Every pair scored finite (`n_bad=0`), including
  the 2048×2048 grids; the linear-space path never materializes the O(L²)
  chart.

### Model parameters (trained, frozen for scoring)

- **MixDom-d3f1c3:** `svibw_d3f1_v3_FROZEN_iter45.npz` (3 domains, 1 fragtype,
  3 site classes; per-class LG substitution; svi-Baum–Welch).
- **MixFrag F=2** (Maraschino cherry-EM, uncapped/log-space run factor):
  λ=0.04608, μ=0.04709, exts=[0.37466, 0.83325], weights=[0.62848, 0.37152].
- **TKF92** (Maraschino cherry-EM): λ=0.04428, μ=0.04521, ext=0.677.

### Reproduce

```bash
# order-summed (per model); LMAX=0 = full val set
cd ~/tkf-mixdom/python
MODEL=d3f1|mixfrag|tkf92 LMAX=0 JAX_ENABLE_X64=1 \
  CUDA_VISIBLE_DEVICES=<gpu> XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python experiments/ordsum_val/score_matchaligned_val.py
```

Raw run output (all three) is preserved in `ordsum_val_RESULTS.txt`.
`mixfrag_from_shards.py` builds the MixFrag/TKF92 count tensors from v3 shards
and fits them with cherry-EM (Maraschino).

### Note on the earlier discrepancy

An earlier comparison showed MixDom-d3f1 ~9 nats *worse* than TKF92/MixFrag.
That was an artifact of the models not being trained on identical data, not a
convention effect: retrained on the same data and scored both ways here,
MixDom-d3f1 is the best model in both columns.
