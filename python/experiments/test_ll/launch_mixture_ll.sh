#!/bin/bash
# Held-out likelihoods for the two mixture baselines of the models table, through the one scorer
# (score_matchaligned_val.py; PATHCOND=1 path-conditioned, 0 order-summed), plus MixFrag's 2D path-conditioned
# validation score so every path-conditioned column entry comes from the same code path.
#   CherryML C=20: TKF92 (anchor indel params) with mixture emissions -> MODEL=cherryml_C20
#   TKF92 K=20:    20 per-component TKF92 runs with per-pair output -> combine_k20.py (family-level mixture)
# Idempotent; memory-gated. Usage: launch_mixture_ll.sh <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-1}; MINFREE=${MINFREE:-3000}; T=experiments/test_ll; V=experiments/ordsum_val; K20=$T/k20
VAL=$PWD/pfam/precompiled_v3_val; TEST=$PWD/pfam/precompiled_v3_test
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() {  # wait for free memory, then take one of two GPU slots (an flock held on fd 9 until release)
  while true; do until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done
    for i in 1 2; do exec 9>/tmp/tkf_gpu${GPU}_slot$i.lock; flock -n 9 && return; done; exec 9>&-; sleep 60; done; }
release() { flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null; }
run() {  # logfile pathcond(0|1) split(val|test) model [extra env...]
  local out=$1 pc=$2 split=$3 model=$4; shift 4; grep -q 'LL/pair  =' $out 2>/dev/null && return
  local dir=$VAL; [ $split = test ] && dir=$TEST
  for try in 1 2 3 4; do gate; echo "$(date -Is) gpu${GPU}ML start $out (try $try)"
    env "$@" PATHCOND=$pc MODEL=$model VAL_DIR=$dir .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > $out 2>&1; release
    echo "$(date -Is) gpu${GPU}ML end $out rc=$?"; grep -q 'LL/pair  =' $out 2>/dev/null && return; sleep 600; done; }
( run $V/pathcond2d_mixfrag_val.log 1 val mixfrag
  run $V/pathcond_cherryml_C20.log 1 val cherryml_C20;  run $V/score_cherryml_C20.log 0 val cherryml_C20
  run $T/pathcond_cherryml_C20.log 1 test cherryml_C20; run $T/ordsum_cherryml_C20.log 0 test cherryml_C20
  for split in val test; do for pc in 1 0; do [ $split = test ] && [ $pc = 1 ] && continue   # no table uses the test path-conditioned score
    for k in $(seq 0 19); do
    run $K20/${split}_pc${pc}_comp${k}.log $pc $split tkf92 TKF92_PARAMS=$K20/comp_$k.json SUBST_NPZ=$K20/comp_$k.npz PERPAIR_OUT=$K20/${split}_pc${pc}_comp${k}.npz
  done; done; done
  .venv/bin/python experiments/test_ll/combine_k20.py > $K20/combine.log 2>&1
  echo "$(date -Is) gpu${GPU}ML end mixture-ll chain" ) >> $LOG 2>&1 &
