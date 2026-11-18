#!/bin/bash
# Retrain the paper's MixDom-d3f1 (per-domain substitution via the MixDom2
# identity-classdist path) under the per-pair-t substitution M-step fix.
# Recipe reconstructed from pfam/svibw_d3f1_perdomain_v3_lgfix.npz (_config:
# n_dom 3, n_frag 1, n_iter 540, svi_batch 2000, svi_tau 1, svi_kappa 0.7,
# seed 42, default priors) and experiments/ordsum_val/RESULTS.md
# (precompiled_v3_train; --classdist-init identity --freeze-classdist) plus the
# memory note that per-class S needs --estimate-sexch.  --pi-pseudo/--S-pseudo
# left at their defaults (10 / 5), as the original run is believed to have.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
( until grep -q 'end mixdom_adam_long 0.1 ' $LOG 2>/dev/null; do sleep 60; done
  echo "$(date -Is) gpu0R start retrain d3f1 v4_fix"
  .venv/bin/python -u train_pfam.py --svi-bw --precompiled-pairs pfam/precompiled_v3_train \
     --n-dom 3 --n-frag 1 --n-classes 3 --classdist-init identity --freeze-classdist \
     --estimate-subst --estimate-sexch --svi-batch 2000 --svi-tau 1.0 --svi-kappa 0.7 --n-iter 540 --seed 42 \
     --checkpoint pfam/svibw_d3f1_perdomain_v4_fix.npz --fresh > pfam/svibw_d3f1_perdomain_v4_fix.log 2>&1
  echo "$(date -Is) gpu0R end retrain d3f1 v4_fix rc=$?" ) >> $LOG 2>&1 &
echo armed
