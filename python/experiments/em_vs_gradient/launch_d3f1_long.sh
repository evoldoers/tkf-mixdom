#!/bin/bash
# Longer d3f1 svi-BW run (540 -> 1000 -> 2000 iterations, resumed from the
# corrected-M-step checkpoint) with held-out scoring at each stage, then a
# late-stage switch to Adam-on-Q (minibatch 2000, lr 0.01, 2000 steps) from
# the 2000-iteration point, scored the same way. GPU 0, after the d4f1
# evaluation there has finished.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log; CV=experiments/em_vs_gradient/mixdom_ckpt_convert.py
COMMON="--svi-bw --precompiled-pairs pfam/precompiled_v3_train --n-dom 3 --n-frag 1 --n-classes 3 --classdist-init identity --freeze-classdist --estimate-subst --estimate-sexch --svi-batch 2000 --svi-tau 1.0 --svi-kappa 0.7 --seed 42 --val-every 0"
score() {  # tag ckpt [balibase]
  echo "$(date -Is) gpu0L start score $1"
  .venv/bin/python -u experiments/em_vs_gradient/score_pathcond_val.py --checkpoint $2 --out experiments/ordsum_val/pathcond_$1.json > experiments/ordsum_val/pathcond_$1.log 2>&1
  MODEL=d3f1 CKPT=$PWD/$2 .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > experiments/ordsum_val/score_$1.log 2>&1
  if [ "$3" = balibase ]; then
    .venv/bin/python -u experiments/expected_pairwise_balibase.py --method mixdom --params $2 --method-name mixdom_$1 --fsa on --fsa-sps \
       --out experiments/expected_balibase/expected_balibase_mixdom_$1_withsps.json > experiments/expected_balibase/expected_balibase_mixdom_$1_withsps.log 2>&1
  fi
  echo "$(date -Is) gpu0L end score $1 rc=$?"
}
( until grep -q 'end ordsum_val d4f1_perdomain_v4_fix rc=' $LOG 2>/dev/null; do sleep 120; done
  cp pfam/svibw_d3f1_perdomain_v4_fix.npz pfam/svibw_d3f1_perdomain_v4_fix_1000.npz
  echo "$(date -Is) gpu0L start resume d3f1 540->1000"
  .venv/bin/python -u train_pfam.py $COMMON --n-iter 1000 --checkpoint pfam/svibw_d3f1_perdomain_v4_fix_1000.npz > pfam/svibw_d3f1_perdomain_v4_fix_1000.log 2>&1
  echo "$(date -Is) gpu0L end resume 1000 rc=$?"
  score d3f1_v4_fix_1000 pfam/svibw_d3f1_perdomain_v4_fix_1000.npz
  cp pfam/svibw_d3f1_perdomain_v4_fix_1000.npz pfam/svibw_d3f1_perdomain_v4_fix_2000.npz
  echo "$(date -Is) gpu0L start resume d3f1 1000->2000"
  .venv/bin/python -u train_pfam.py $COMMON --n-iter 2000 --checkpoint pfam/svibw_d3f1_perdomain_v4_fix_2000.npz > pfam/svibw_d3f1_perdomain_v4_fix_2000.log 2>&1
  echo "$(date -Is) gpu0L end resume 2000 rc=$?"
  score d3f1_v4_fix_2000 pfam/svibw_d3f1_perdomain_v4_fix_2000.npz balibase
  JAX_PLATFORMS=cpu .venv/bin/python $CV to-maraschino pfam/svibw_d3f1_perdomain_v4_fix_2000.npz pfam/svibw_d3f1_perdomain_v4_fix_2000_init.npz > /dev/null 2>&1
  echo "$(date -Is) gpu0L start adam-switch d3f1 from 2000"
  .venv/bin/python -u train_pfam.py --adam --precompiled-pairs pfam/precompiled_v3_train --n-dom 3 --n-frag 1 --adam-batch 2000 --adam-steps 2000 --adam-lr 0.01 --seed 42 \
     --init-from-maraschino pfam/svibw_d3f1_perdomain_v4_fix_2000_init.npz --checkpoint pfam/adam_d3f1_from2000.npz --fresh > pfam/adam_d3f1_from2000.log 2>&1
  echo "$(date -Is) gpu0L end adam-switch rc=$?"
  JAX_PLATFORMS=cpu .venv/bin/python $CV to-scoring pfam/adam_d3f1_from2000.npz pfam/svibw_d3f1_perdomain_v4_fix_2000.npz pfam/adam_d3f1_from2000_scoring.npz > /dev/null 2>&1
  score d3f1_adam_from2000 pfam/adam_d3f1_from2000_scoring.npz balibase ) >> $LOG 2>&1 &
echo armed
