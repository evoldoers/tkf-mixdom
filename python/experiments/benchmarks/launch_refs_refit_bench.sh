#!/bin/bash
# Benchmarks of TKF92 and MixFrag F=2 with their indel parameters refitted by counts EM to the bali3pdbm reference
# alignments (experiments/balibase_indel_refit/mixfrag_F{1,2}_bali3pdbm_refs.json; the counts-based indel fit does
# not involve the substitution matrix) and the GTR refitted to the Pfam training pairs as substitution model, through
# profile_mixture/experiments/fsa_c20_balibase.py (--subst GTR --clock one), posterior-cached as <model>refs_GTR_one__<bench>. Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-0}; MINFREE=${MINFREE:-4000}; BENCH=$HOME/bio-datasets/data/bench1.0; PM=$PWD/experiments/profile_mixture
PY=$PWD/.venv/bin/python; C20=$PM/results/fsa_c20_balibase; R=$PWD/experiments/balibase_indel_refit; GTR=$PWD/pfam/gtr_refit_Qpi.npz
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
bdir() { case $1 in bali3pdbm) echo $HOME/bio-datasets/data/balibase/bali3pdbm;; *) echo $BENCH/$1;; esac; }
c20() {  # bench outname cache extra-args...
  local b=$1 o=$2 cache=$3; shift 3; [ -f $C20/$o.json ] && return; gate
  echo "$(date -Is) gpu${GPU}RB start c20 $b $o"
  ( cd $PM && GTR_NPZ=$GTR PYTHONPATH=$PM/src:$PM JAX_CACHE_DIR=$HOME/.cache/jax-fsa-c20 $PY experiments/fsa_c20_balibase.py --subst GTR --clock one --balibase-dir $(bdir $b) --out $C20/$o.json --cache-name $cache "$@" > $C20/$o.log 2>&1 )
  echo "$(date -Is) gpu${GPU}RB end c20 $b $o rc=$?"; }
( for b in bali3pdbm sabrem oxm; do c20 $b cache_runs/tkf92refs_GTR_one_$b tkf92refs_GTR_one__$b --indel mixfrag --indel-params $R/mixfrag_F1_bali3pdbm_refs.json; done
  for b in bali3pdbm sabrem oxm; do c20 $b cache_runs/mixfrag2refs_GTR_one_$b mixfrag2refs_GTR_one__$b --indel mixfrag --indel-params $R/mixfrag_F2_bali3pdbm_refs.json; done
  echo "$(date -Is) gpu${GPU}RB end refs-refit bench chain" ) >> $LOG 2>&1 &
