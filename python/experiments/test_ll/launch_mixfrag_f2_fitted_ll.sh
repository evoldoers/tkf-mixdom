#!/bin/bash
# MixFrag F=2 path-conditioned likelihoods (validation, test) with its fitted log-space parameters
# (experiments/mixfrag_F/mixfrag_F2_logspace.json, the fit the text describes), replacing the cells scored with the
# scorer's hard-coded default parameter set (archived as *_scorerdefault.log). Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-1}; MINFREE=${MINFREE:-3000}; T=experiments/test_ll; V=experiments/ordsum_val
VAL=$PWD/pfam/precompiled_v3_val; TEST=$PWD/pfam/precompiled_v3_test
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log; P=gpu${GPU}MF
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
run() { local out=$1 split=$2; grep -q 'LL/pair  =' $out 2>/dev/null && return; local dir=$VAL; [ $split = test ] && dir=$TEST
  for try in 1 2 3; do gate; echo "$(date -Is) $P start $out (try $try)"
    MIXFRAG_PARAMS=experiments/mixfrag_F/mixfrag_F2_logspace.json SUBST_NPZ=pfam/gtr_refit_Qpi.npz PATHCOND=1 MODEL=mixfrag VAL_DIR=$dir .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > $out 2>&1
    echo "$(date -Is) $P end $out rc=$? $(grep 'LL/pair  =' $out | tail -1)"; grep -q 'LL/pair  =' $out 2>/dev/null && return; sleep 600; done; }
( run $V/pathcond_mixfrag_gtrrefit.log val; run $T/pathcond_mixfrag_gtrrefit.log test
  echo "$(date -Is) $P end MixFrag F=2 fitted-parameter likelihood chain" ) >> $LOG 2>&1 &
