#!/bin/bash
# Side investigation (not for the paper unless approved): can svi-BW alone reach the Adam plateau with a slower-decaying
# step and Polyak-Ruppert averaging of the iterates? d2f1, resumed from the 540-iteration svi-BW checkpoint (same
# minibatch RNG state), run to iteration 1,620 saving every iterate:
#   GPU 0: kappa = 0.5;  GPU 1: kappa = 0.7 (the released schedule; control, its final iterate should reproduce
#          pfam/svibw_d2f1_nobali_1620.npz).
# Then, for each run, score on every 10th validation pair (score_pathcond_val.py --stride 10): the final iterate and the
# Polyak-Ruppert averages over iterations 811-1620 and 1081-1620 (average_iterates.py).
# Output: experiments/ordsum_val/kappa_avg_stride10.tsv. Reference points: released svi-BW 1,620 -715.03; the Adam
# plateau about -714.60 (adam_d2f1_traj_stride10.tsv, switch_point_stride10.tsv). Usage: kappa_avg_test.sh
cd "$(dirname "$0")/../.."
Q=experiments/em_vs_gradient/results_rest_queue.log; OUT=experiments/ordsum_val/kappa_avg_stride10.tsv
COMMON="--svi-bw --precompiled-pairs pfam/precompiled_v3_train_nobali --n-dom 2 --n-frag 1 --n-classes 2 --classdist-init identity --freeze-classdist --estimate-subst --estimate-sexch --svi-batch 2000 --svi-tau 1.0 --seed 42 --val-every 0 --jax-cache-dir /home/yam/.cache/tkf-mixdom-jax"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
[ -f $OUT ] || printf "run\tparameters\tll_per_pair\tn_pairs\n" > $OUT
score() {  # tag ckpt
  local tag=$1
  local ck=$2
  local j=experiments/ordsum_val/pathcond_kavg_${tag}_stride10.json
  [ -s $j ] || .venv/bin/python -u experiments/em_vs_gradient/score_pathcond_val.py --checkpoint $ck --stride 10 --out $j > ${j%.json}.log 2>&1
  python3 -c "import json; d=json.load(open('$j')); print('${tag%%_*}\t${tag#*_}\t%.6f\t%d' % (d['ll_per_pair'], d['n_pairs']))" >> $OUT 2>/dev/null || printf "${tag%%_*}\t${tag#*_}\tFAILED\t0\n" >> $OUT
}
one() {  # kappa gpu
  local k=$1
  local gpu=$2
  local tag=k${k/./}
  local B=pfam/svibw_d2f1_${tag}avg_from540
  export CUDA_VISIBLE_DEVICES=$gpu
  echo "$(date -Is) gpu${gpu}KA start svi-BW kappa=$k from iteration 540, saving every iterate"
  [ -f $B.npz ] || cp pfam/svibw_d2f1_nobali_540.npz $B.npz
  .venv/bin/python -u train_pfam.py $COMMON --svi-kappa $k --n-iter 1620 --checkpoint $B.npz --save-every-iter > $B.log 2>&1
  echo "$(date -Is) gpu${gpu}KA end svi-BW kappa=$k rc=$? $(ls ${B}_iter*.npz 2>/dev/null | wc -l) iterates"
  for w in 811 1081; do
    JAX_PLATFORMS=cpu .venv/bin/python experiments/em_vs_gradient/average_iterates.py $B.npz $w 1620 ${B}_avg${w}.npz >> $B.log 2>&1
  done
  score ${tag}_final $B.npz
  score ${tag}_avg811to1620 ${B}_avg811.npz
  score ${tag}_avg1081to1620 ${B}_avg1081.npz
  echo "$(date -Is) gpu${gpu}KA end kappa-avg test kappa=$k"
}
( one 0.5 0 ) >> $Q 2>&1 &
( one 0.7 1 ) >> $Q 2>&1 &
echo "armed kappa/averaging test (kappa 0.5 on GPU 0, 0.7 on GPU 1)"
