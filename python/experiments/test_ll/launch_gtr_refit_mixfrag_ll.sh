#!/bin/bash
# MixFrag F=2 (Table-2 params) and F=3 (log-space refit) with the single GTR refitted on the training pairs
# (pfam/gtr_refit_Qpi.npz) in place of LG08: held-out likelihoods through the one scorer. Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-1}; MINFREE=${MINFREE:-3000}; T=experiments/test_ll; V=experiments/ordsum_val
VAL=$PWD/pfam/precompiled_v3_val; TEST=$PWD/pfam/precompiled_v3_test
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() {  # wait for free memory, then take one of two GPU slots (an flock held on fd 9 until release)
  while true; do until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done
    for i in 1 2; do exec 9>/tmp/tkf_gpu${GPU}_slot$i.lock; flock -n 9 && return; done; exec 9>&-; sleep 60; done; }
release() { flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null; }
run() { local out=$1 pc=$2 split=$3; shift 3; grep -q 'LL/pair  =' $out 2>/dev/null && return; local dir=$VAL; [ $split = test ] && dir=$TEST
  for try in 1 2 3; do gate; echo "$(date -Is) gpu${GPU}GR start $out (try $try)"
    env "$@" SUBST_NPZ=pfam/gtr_refit_Qpi.npz PATHCOND=$pc MODEL=mixfrag VAL_DIR=$dir .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > $out 2>&1; release
    echo "$(date -Is) gpu${GPU}GR end $out rc=$?"; grep -q 'LL/pair  =' $out 2>/dev/null && return; sleep 600; done; }
( F2=MIXFRAG_PARAMS=experiments/mixfrag_F/mixfrag_F2_logspace.json   # (2026-11-13) F=2 now passes its fitted parameters explicitly too
  run $V/pathcond_mixfrag_gtrrefit.log 1 val $F2; run $V/score_mixfrag_gtrrefit.log 0 val $F2; run $T/ordsum_mixfrag_gtrrefit.log 0 test $F2
  F3=MIXFRAG_PARAMS=experiments/mixfrag_F/mixfrag_F3_logspace.json
  run $V/pathcond_mixfrag3_gtrrefit.log 1 val $F3; run $V/score_mixfrag3_gtrrefit.log 0 val $F3; run $T/ordsum_mixfrag3_gtrrefit.log 0 test $F3
  echo "$(date -Is) gpu${GPU}GR end gtr-refit mixfrag chain" ) >> $LOG 2>&1 &
