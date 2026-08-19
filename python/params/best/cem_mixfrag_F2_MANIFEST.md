# MixFrag-F=2 trained parameters (Pfam v1)

Trained **MixFrag** model with **F=2 fragtypes** — TKF92 with the fragment-extension
parameter drawn per-fragment from a categorical over 𝓕 fragtypes (weights `w_f`,
extensions `r_f`), substitution and indel processes shared across fragtypes; reduces to
TKF92 at 𝓕=1. This is the **paper-1 Table-2** (`tab:pfam`) MixFrag row, fit by
closed-form summarised-count EM on the Pfam v1 training split with LG08 held fixed.

## Fitted values

| parameter | value |
|---|---|
| insertion rate `λ` | 0.033924 |
| deletion rate `μ`  | 0.034690 |
| `κ = λ/μ`          | 0.9779 |
| fragment extensions `{r_f}` | [0.41396, 0.86031] |
| fragment weights `{w_f}`    | [0.70086, 0.29914] |
| substitution `(Q, π)`       | LG08 (fixed) |
| held-out val LL (total, nats) | −2.3050749×10⁷ |
| train LL (total, nats)      | −2.4124385×10⁸ |

Fit on the Pfam v1 **train** split (19,660 families / 361,930 cherries); held-out LL on
the v1 **val** split. 500 EM iterations. For comparison, TKF92 (F=1, same pipeline) scores
val LL −2.3060659×10⁷, so the second fragtype buys ≈ +9,900 nats for two extra params.

## Files

- `cem_mixfrag_F2_pfamTrain.npz` — the fitted params + training trace. Keys:
  - `lam`, `mu` — TKF92 indel rates (shared across fragtypes).
  - `exts` (2,) — fragment extension parameters `r_f`.
  - `weights` (2,) — fragment mixture weights `w_f`.
  - `n_fragtypes` — 2.
  - `substitution` — `"lg"` (LG08 `Q`, `π` fixed; not stored — reconstruct with the
    library's LG08).
  - `final_ll`, `sub_ll` — train total LL and its substitution part (nats).
  - `val_ll`, `val_indel_ll`, `val_sub_ll` — held-out totals (nats).
  - `n_families`, `n_cherries`, `val_n_families`, `val_n_cherries` — corpus sizes.
  - `hist_iter`, `hist_total_ll`, `hist_lam`, `hist_mu`, `hist_exts`, `hist_weights`
    (500,) — per-iteration EM trace.
- `cem_mixfrag_F2_pfamTrain.json` — scalar summary (`lam`, `mu`, `exts`, `weights`,
  `final_ll`, `sub_ll`, `val_ll`, `n_families`, `n_cherries`, `n_iter_run`).

## Reproduce

Run from `~/tkf-mixdom/python`. Substitution is LG08 (fixed); only the indel rates and the
fragment mixture (`{w_f}`, `{r_f}`) are fit.

```bash
# 1. Build the per-family MixFrag cherry-count tensors for both splits from the Pfam v1
#    split (CherryML nearest-neighbour cherries; ordering-summed gap factors):
python build_mixfrag_cherry_counts.py \
    --split-file ~/bio-datasets/data/pfam/seed/splits/v1.json --split train
python build_mixfrag_cherry_counts.py \
    --split-file ~/bio-datasets/data/pfam/seed/splits/v1.json --split val

# 2. Closed-form summarised-count EM fit (F=2), scoring the held-out val split:
python train_mixfrag_cherry_em.py \
    --counts-dir pfam/cherries_mixfrag \
    --split-file ~/bio-datasets/data/pfam/seed/splits/v1.json \
    --split train --val-split val \
    --n-fragtypes 2 \
    --out params/best/cem_mixfrag_F2_pfamTrain.npz
#   (--n-fragtypes 1 gives the TKF92 baseline row.)
```

## Load

```python
import numpy as np
d = np.load("cem_mixfrag_F2_pfamTrain.npz", allow_pickle=True)
lam, mu = float(d["lam"]), float(d["mu"])
r_f, w_f = d["exts"], d["weights"]            # per-fragtype extension, weight
# substitution: LG08 (fixed) -- from tkfmixdom.jax.core.protein.rate_matrix_lg()
```
