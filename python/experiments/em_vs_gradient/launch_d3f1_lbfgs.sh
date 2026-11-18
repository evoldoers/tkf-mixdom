#!/bin/bash
# L-BFGS late-stage refinement of d3f1 from the 2000-iteration svi-BW point
# on a fixed 20,000-pair training subset, then scoring; after the Adam switch.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
( until grep -q 'end score d3f1_adam_from2000' $LOG 2>/dev/null; do sleep 120; done
  echo "$(date -Is) gpu0L start lbfgs-refine d3f1 from 2000"
  .venv/bin/python -u experiments/em_vs_gradient/mixdom_lbfgs_refine.py --init pfam/svibw_d3f1_perdomain_v4_fix_2000_init.npz \
     --template pfam/svibw_d3f1_perdomain_v4_fix_2000.npz --n-sub 20000 --maxfun 80 --out pfam/lbfgs_d3f1_from2000.npz > pfam/lbfgs_d3f1_from2000.log 2>&1
  echo "$(date -Is) gpu0L end lbfgs-refine rc=$?"
  echo "$(date -Is) gpu0L start score d3f1_lbfgs_from2000"
  .venv/bin/python -u experiments/em_vs_gradient/score_pathcond_val.py --checkpoint pfam/lbfgs_d3f1_from2000.npz --out experiments/ordsum_val/pathcond_d3f1_lbfgs_from2000.json > experiments/ordsum_val/pathcond_d3f1_lbfgs_from2000.log 2>&1
  MODEL=d3f1 CKPT=$PWD/pfam/lbfgs_d3f1_from2000.npz .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > experiments/ordsum_val/score_d3f1_lbfgs_from2000.log 2>&1
  .venv/bin/python -u experiments/expected_pairwise_balibase.py --method mixdom --params pfam/lbfgs_d3f1_from2000.npz --method-name mixdom_d3f1_lbfgs_from2000 --fsa on --fsa-sps \
     --out experiments/expected_balibase/expected_balibase_mixdom_d3f1_lbfgs_from2000_withsps.json > experiments/expected_balibase/expected_balibase_mixdom_d3f1_lbfgs_from2000_withsps.log 2>&1
  echo "$(date -Is) gpu0L end score d3f1_lbfgs_from2000 rc=$?" ) >> $LOG 2>&1 &
echo armed
