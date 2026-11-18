#!/bin/bash
# After the corrected MixDom-d3f1 retrain: (1) BAliBASE posteriors + FSA at
# gap factors 1 and 0 into the pair cache under a new method name; (2) v3_val
# path-conditioned and order-summed LL/pair (Table 2).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
CK=pfam/svibw_d3f1_perdomain_v4_fix.npz
( until grep -q 'end retrain d3f1 v4_fix rc=0' $LOG 2>/dev/null; do sleep 120; done
  echo "$(date -Is) gpu0E start balibase mixdom_d3f1_v4_fix"
  .venv/bin/python -u experiments/expected_pairwise_balibase.py --method mixdom --params $CK \
     --method-name mixdom_d3f1_v4_fix --fsa on --fsa-sps \
     --out experiments/expected_balibase/expected_balibase_mixdom_d3f1_v4_fix_withsps.json \
     > experiments/expected_balibase/expected_balibase_mixdom_d3f1_v4_fix_withsps.log 2>&1
  echo "$(date -Is) gpu0E end balibase rc=$?"
  echo "$(date -Is) gpu0E start ordsum_val d3f1 v4_fix"
  MODEL=d3f1 CKPT=$PWD/$CK .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py \
     > experiments/ordsum_val/score_d3f1_v4_fix.log 2>&1
  echo "$(date -Is) gpu0E end ordsum_val rc=$?" ) >> $LOG 2>&1 &
echo armed
