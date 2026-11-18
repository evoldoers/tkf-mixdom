#!/bin/bash
# Side investigation (not for the paper unless approved): does it matter WHEN the d2f1 fit switches from svi-BW to Adam?
# From the svi-BW stage checkpoints at 540 and 1,080 iterations (the 1,620 case is the released tail, already scored in
# experiments/ordsum_val/adam_d2f1_traj_stride10.tsv), run 200 Adam steps with the released tail's settings (lr 0.01,
# minibatch 2,000, seed 42), keeping the parameters at steps 100 and 199, and score the svi-BW start and both kept
# checkpoints on every 10th validation pair (score_pathcond_val.py --stride 10, the paper's path-conditioned LL/pair).
# Runs 540 on GPU 0 and 1,080 on GPU 1 in parallel. Output: experiments/ordsum_val/switch_point_stride10.tsv
# Usage: switch_point_test.sh
cd "$(dirname "$0")/../.."
Q=experiments/em_vs_gradient/results_rest_queue.log; CV=experiments/em_vs_gradient/mixdom_ckpt_convert.py
OUT=experiments/ordsum_val/switch_point_stride10.tsv
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
one() {  # <svi-BW iterations> <gpu>
  local it=$1
  local gpu=$2
  local src=pfam/svibw_d2f1_nobali_${it}.npz
  local init=pfam/svibw_d2f1_nobali_${it}_init.npz
  local B=pfam/adam_d2f1_nobali_from${it}_switch
  export CUDA_VISIBLE_DEVICES=$gpu
  echo "$(date -Is) gpu${gpu}SW start switch-point test from svi-BW iteration $it"
  [ -f $init ] || JAX_PLATFORMS=cpu .venv/bin/python $CV to-maraschino $src $init > /dev/null 2>&1
  .venv/bin/python -u train_pfam.py --adam --precompiled-pairs pfam/precompiled_v3_train_nobali --n-dom 2 --n-frag 1 \
     --adam-batch 2000 --adam-steps 200 --adam-lr 0.01 --seed 42 --init-from-maraschino $init \
     --checkpoint $B.npz --fresh --ckpt-keep-every 100 > $B.log 2>&1
  echo "$(date -Is) gpu${gpu}SW end Adam from $it rc=$?"
  for pair in "0 $src" "100 ${B}_step0100.npz" "199 ${B}_step0199.npz"; do
    set -- $pair
    local step=$1
    local ck=$2
    local j=experiments/ordsum_val/pathcond_switch${it}_step${step}_stride10.json
    [ -s $j ] || .venv/bin/python -u experiments/em_vs_gradient/score_pathcond_val.py --checkpoint $ck --stride 10 --out $j > ${j%.json}.log 2>&1
    python3 -c "import json; d=json.load(open('$j')); print('$it\t$step\t%.6f\t%d' % (d['ll_per_pair'], d['n_pairs']))" >> $OUT 2>/dev/null || printf "$it\t$step\tFAILED\t0\n" >> $OUT
  done
  echo "$(date -Is) gpu${gpu}SW end switch-point test from $it"
}
[ -f $OUT ] || printf "svibw_iters\tadam_step\tll_per_pair\tn_pairs\n" > $OUT
( one 540 0 ) >> $Q 2>&1 &
( one 1080 1 ) >> $Q 2>&1 &
echo "armed switch-point test (540 on GPU 0, 1080 on GPU 1)"
