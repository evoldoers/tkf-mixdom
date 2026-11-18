#!/bin/bash
# TKF92 (its own Pfam indel parameters) with a single GTR substitution matrix refitted on the precompiled training
# pairs at their own clock (fit_paircolumn_sitemix.py --n-classes 1 -> pfam/gtr_refit_C1_paircol.npz): isolates the
# substitution-refit part of every model's likelihood gain over TKF92+LG08. Scored through the CherryML branch of the
# one scorer with one class. Usage: <gpu>
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
run() { local out=$1 pc=$2 split=$3; grep -q 'LL/pair  =' $out 2>/dev/null && return; local dir=$VAL; [ $split = test ] && dir=$TEST
  for try in 1 2 3; do gate; echo "$(date -Is) gpu${GPU}GR start $out (try $try)"
    CHERRYML_NPZ=pfam/gtr_refit_C1_paircol.npz TKF92_PARAMS=$T/tkf92_pfam_params.json PATHCOND=$pc MODEL=cherryml_C20 VAL_DIR=$dir .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > $out 2>&1; release
    echo "$(date -Is) gpu${GPU}GR end $out rc=$?"; grep -q 'LL/pair  =' $out 2>/dev/null && return; sleep 600; done; }
( run $V/pathcond_tkf92_gtrrefit.log 1 val; run $V/score_tkf92_gtrrefit.log 0 val; run $T/ordsum_tkf92_gtrrefit.log 0 test
  echo "$(date -Is) gpu${GPU}GR end gtr-refit chain" ) >> $LOG 2>&1 &
