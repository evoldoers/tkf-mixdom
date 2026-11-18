#!/bin/bash
# Test-split (precompiled_v3_test, 213,930 pairs, 3,310 clan-disjoint families) likelihoods for every
# final model: order-summed for TKF92 and MixFrag (fixed Table 2 parameters, MODEL=tkf92|mixfrag) and both
# path-conditioned and order-summed for the MixDom checkpoints. Idempotent; memory-gated. Usage: launch_test_ll.sh <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-1}; MINFREE=${MINFREE:-8000}; T=experiments/test_ll; TEST=$PWD/pfam/precompiled_v3_test
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
# Both conventions through ONE scorer (score_matchaligned_val.py): PATHCOND=1 fixes the whole path,
# PATHCOND=0 fixes the matches and sums the indel orderings.
score2d() {  # kind(pathcond|ordsum) model tag [ckpt]
  local kind=$1 m=$2 t=$3 ck=$4; local pc=0; [ $kind = pathcond ] && pc=1
  for try in 1 2 3 4 5 6; do
    grep -q 'LL/pair  =' $T/${kind}_$t.log 2>/dev/null && return; gate; echo "$(date -Is) gpu${GPU}T2 start test $kind $t (try $try)"
    PATHCOND=$pc MODEL=$m CKPT=${ck:+$PWD/$ck} VAL_DIR=$TEST .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > $T/${kind}_$t.log 2>&1; echo "$(date -Is) gpu${GPU}T2 end test $kind $t rc=$?"
    grep -q 'LL/pair  =' $T/${kind}_$t.log 2>/dev/null && return; sleep 900; done; }
ordsum_fixed() { score2d ordsum $1 $1; }   # test path-conditioned scores are not used by any table
mixdom() {  # tag ckpt
  local t=$1 ck=$2; until [ -f $ck ]; do sleep 600; done
  score2d ordsum d3f1 $t $ck; }
( ordsum_fixed tkf92; ordsum_fixed mixfrag
  mixdom d3f1_table4 pfam/svibw_d3f1_perdomain_v4_fix.npz
  for n in 2 3 4 5; do mixdom d${n}f1_nobali_1620 pfam/svibw_d${n}f1_nobali_1620.npz; done
  mixdom d1f1_nobali_1620 pfam/svibw_d1f1_nobali_1620.npz
  for n in 3 2; do mixdom d${n}f1_nobali_adam pfam/adam_d${n}f1_nobali_from1620_scoring.npz; done   # only the Adam-tail rows the tables use
  mixdom d3f1_full_1620 pfam/svibw_d3f1_full_1620.npz
  echo "$(date -Is) gpu${GPU}T2 end test-ll chain" ) >> $LOG 2>&1 &
echo armed
