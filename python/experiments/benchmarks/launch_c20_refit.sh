#!/bin/bash
# CherryML C=20 as a pure 20-state substitution mixture fitted on the precompiled Pfam training pairs at their own
# times (fit_paircolumn_sitemix.py; one class per aligned column, the model the pair-HMM scorers and the BAliBASE
# harness use), then held-out likelihoods (val/test, both conventions) and posterior-cached benchmark runs on all
# three benchmarks with the anchored clock, as for the original CML20 row.  Idempotent; memory-gated.
# Usage: launch_c20_refit.sh <gpu>   (TAG / NPZ env override the output names)
cd "$(dirname "$0")/../.."
GPU=${1:-0}; MINFREE=${MINFREE:-3000}; T=experiments/test_ll; V=experiments/ordsum_val; BENCH=$HOME/bio-datasets/data/bench1.0; PM=$PWD/experiments/profile_mixture
TAG=${TAG:-pc}; NPZ=${NPZ:-pfam/cherryml_mixture_C20_paircol.npz}; PY=$PWD/.venv/bin/python; C20=$PM/results/fsa_c20_balibase
VAL=$PWD/pfam/precompiled_v3_val; TEST=$PWD/pfam/precompiled_v3_test
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() {  # wait for free memory, then take one of two GPU slots (an flock held on fd 9 until release)
  while true; do until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done
    for i in 1 2; do exec 9>/tmp/tkf_gpu${GPU}_slot$i.lock; flock -n 9 && return; done; exec 9>&-; sleep 60; done; }
release() { flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null; }
bdir() { case $1 in bali3pdbm) echo $HOME/bio-datasets/data/balibase/bali3pdbm;; *) echo $BENCH/$1;; esac; }
score() {  # logfile pathcond split
  local out=$1 pc=$2 split=$3; grep -q 'LL/pair  =' $out 2>/dev/null && return; local dir=$VAL; [ $split = test ] && dir=$TEST
  for try in 1 2 3 4; do gate; echo "$(date -Is) gpu${GPU}CA start $out (try $try)"
    CHERRYML_NPZ=$NPZ PATHCOND=$pc MODEL=cherryml_C20 VAL_DIR=$dir $PY -u experiments/ordsum_val/score_matchaligned_val.py > $out 2>&1; release
    echo "$(date -Is) gpu${GPU}CA end $out rc=$?"; grep -q 'LL/pair  =' $out 2>/dev/null && return; sleep 600; done; }
c20() {  # bench
  local b=$1 o=cache_runs/tkf92_CML20${TAG}_anchor_$b; [ -f $C20/$o.json ] && return; gate
  echo "$(date -Is) gpu${GPU}CA start c20 $b $o"
  ( cd $PM && CHERRYML_NPZ=$OLDPWD/$NPZ PYTHONPATH=$PM/src:$PM JAX_CACHE_DIR=$HOME/.cache/jax-fsa-c20 $PY experiments/fsa_c20_balibase.py --indel tkf92 --subst CML20 --clock anchor --balibase-dir $(bdir $b) --out $C20/$o.json --cache-name tkf92_CML20${TAG}_anchor__$b > $C20/$o.log 2>&1 ); release
  echo "$(date -Is) gpu${GPU}CA end c20 $b $o rc=$?"; }
FIT_CMD=${FIT_CMD:-"$PY -u experiments/fit_paircolumn_sitemix.py --precompiled pfam/precompiled_v3_train --n-classes 20 --n-outer 60 --n-inner 60 --output $NPZ"}
( if [ ! -f $NPZ ]; then gate; echo "$(date -Is) gpu${GPU}CA start fit C20 $TAG"
    $FIT_CMD > ${NPZ%.npz}.log 2>&1; release
    echo "$(date -Is) gpu${GPU}CA end fit C20 $TAG rc=$?"; fi
  [ -f $NPZ ] || { echo "$(date -Is) gpu${GPU}CA ABORT: no $NPZ"; exit 1; }
  score $V/pathcond_cherryml_C20$TAG.log 1 val; score $V/score_cherryml_C20$TAG.log 0 val
  score $T/pathcond_cherryml_C20$TAG.log 1 test; score $T/ordsum_cherryml_C20$TAG.log 0 test
  for b in bali3pdbm sabrem oxm; do c20 $b; done
  echo "$(date -Is) gpu${GPU}CA end c20-refit chain" ) >> $LOG 2>&1 &
