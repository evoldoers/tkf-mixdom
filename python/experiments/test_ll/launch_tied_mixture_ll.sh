#!/bin/bash
# Held-out likelihoods of the "independent heterogeneity" control at the mixture level: a K=20 mixture of TKF92
# indel-parameter classes (fit_tkf92_mixture.py --tie-subst; one component per family) whose emissions are the
# C=20 site mixture (cherryml_mixture_C20_paircol.npz, fitted on the same training pairs) shared by every component, i.e. protein types by indel
# class and, independently, site types by substitution class.  Scored component by component through the one
# scorer (MODEL=cherryml_C20 with the component's indel parameters) and combined at the family level by
# combine_k20.py (directory k20tied, tag tkf92_K20tied).  Waits for both fits.  Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-1}; MINFREE=${MINFREE:-3000}; T=experiments/test_ll; D=$T/k20tied
IND=pfam/tkf92_mixture_K20_train_tiedsubst.npz; EMIT=pfam/cherryml_mixture_C20_paircol.npz
VAL=$PWD/pfam/precompiled_v3_val; TEST=$PWD/pfam/precompiled_v3_test
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() {  # wait for free memory, then take one of two GPU slots (an flock held on fd 9 until release)
  while true; do until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done
    for i in 1 2; do exec 9>/tmp/tkf_gpu${GPU}_slot$i.lock; flock -n 9 && return; done; exec 9>&-; sleep 60; done; }
release() { flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null; }
run() {  # logfile pathcond split comp
  local out=$1 pc=$2 split=$3 k=$4; grep -q 'LL/pair  =' $out 2>/dev/null && return
  local dir=$VAL; [ $split = test ] && dir=$TEST
  for try in 1 2 3 4; do gate; echo "$(date -Is) gpu${GPU}MT start $out (try $try)"
    CHERRYML_NPZ=$EMIT TKF92_PARAMS=$D/comp_$k.json PERPAIR_OUT=$D/${split}_pc${pc}_comp${k}.npz PATHCOND=$pc MODEL=cherryml_C20 VAL_DIR=$dir .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > $out 2>&1; release
    echo "$(date -Is) gpu${GPU}MT end $out rc=$?"; grep -q 'LL/pair  =' $out 2>/dev/null && return; sleep 600; done; }
( until [ -f $IND ] && [ -f $EMIT ]; do sleep 300; done
  .venv/bin/python experiments/test_ll/k20_components.py $IND k20tied > $D.export.log 2>&1 || { echo "$(date -Is) gpu${GPU}MT ABORT export"; exit 1; }
  echo "$(date -Is) gpu${GPU}MT start tied-mixture chain"
  for split in val test; do for pc in 1 0; do [ $split = test ] && [ $pc = 1 ] && continue   # no table uses the test path-conditioned score
    for k in $(seq 0 19); do run $D/${split}_pc${pc}_comp${k}.log $pc $split $k; done; done; done
  .venv/bin/python experiments/test_ll/combine_k20.py k20tied tkf92_K20tied > $D/combine.log 2>&1
  echo "$(date -Is) gpu${GPU}MT end tied-mixture chain" ) >> $LOG 2>&1 &
