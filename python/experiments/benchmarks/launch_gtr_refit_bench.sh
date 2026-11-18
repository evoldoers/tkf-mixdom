#!/bin/bash
# Benchmarks (posterior-cached) for TKF92, MixFrag F=2 and F=3 with the refitted GTR (pfam/gtr_refit_Qpi.npz) in
# place of the published LG, through tkf-dp's fsa_c20_balibase.py (--subst GTR --clock one). Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-0}; MINFREE=${MINFREE:-6000}; BENCH=$HOME/bio-datasets/data/bench1.0; TKFDP=/home/yam/tkf-dp
PY=$PWD/.venv/bin/python; C20=$TKFDP/results/fsa_c20_balibase; R=$PWD/experiments/mixfrag_F; GTR=$PWD/pfam/gtr_refit_Qpi.npz
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
bdir() { case $1 in bali3pdbm) echo $HOME/bio-datasets/data/balibase/bali3pdbm;; *) echo $BENCH/$1;; esac; }
c20() {  # bench outname cache extra-args...
  local b=$1 o=$2 cache=$3; shift 3; [ -f $C20/$o.json ] && return; gate
  echo "$(date -Is) gpu${GPU}GB start c20 $b $o"
  ( cd $TKFDP && GTR_NPZ=$GTR PYTHONPATH=$TKFDP/src:$TKFDP JAX_CACHE_DIR=$HOME/.cache/jax-fsa-c20 $PY experiments/fsa_c20_balibase.py --subst GTR --clock one --balibase-dir $(bdir $b) --out $C20/$o.json --cache-name $cache "$@" > $C20/$o.log 2>&1 )
  echo "$(date -Is) gpu${GPU}GB end c20 $b $o rc=$?"; }
( for b in bali3pdbm sabrem oxm; do c20 $b cache_runs/tkf92_GTR_one_$b tkf92_GTR_one__$b --indel tkf92; done
  for b in bali3pdbm sabrem oxm; do c20 $b cache_runs/mixfrag_GTR_one_$b mixfrag_GTR_one__$b --indel mixfrag; done
  for b in bali3pdbm sabrem oxm; do c20 $b cache_runs/mixfrag3_GTR_one_$b mixfrag3_GTR_one__$b --indel mixfrag --indel-params $R/mixfrag_F3_logspace.json; done
  echo "$(date -Is) gpu${GPU}GB end gtr-refit bench chain" ) >> $LOG 2>&1 &
