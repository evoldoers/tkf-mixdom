#!/bin/bash
# svi-BW vs Adam on the same 20k-pair breadth-first Pfam train set, canonical
# 500-pair val, batch 16, same init, 800 iterations each with early stopping
# disabled, so val LL trajectories and wall-clock can be compared post hoc.
# Same knobs as experiments/2dfb/benchmark.md.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1
OUT=experiments/em_vs_gradient/stochastic
COMMON="--precompiled-dir pfam/precompiled --n-train 20000 --n-val 500 --batch-size 16
        --n-iter 800 --patience 100000 --val-every 5 --seed 0
        --init-lam 0.04 --init-mu 0.05 --init-ext 0.5
        --bin-bucketed --pre-warm --no-command-buffers --max-pad-cap 256"
for mode in ${MODES:-svi_bw adam_tkf92}; do
  echo "=== $mode start $(date -Is) ==="
  .venv/bin/python -u experiments/run_tkf92_2dfb_pfam.py --mode $mode $COMMON \
      --out $OUT/pfam_${mode}_seed0.json > $OUT/pfam_${mode}_seed0.log 2>&1
  echo "=== $mode end $(date -Is) ==="
done
