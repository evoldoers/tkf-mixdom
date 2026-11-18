#!/bin/bash
# Fixed (non-MixDom) models for the consolidated models/decoding tables. Idempotent; memory-gated. Usage: <gpu>
#  1. TKF92 and MixFrag F=2 (published LG, one clock; the paper's Table-2 fits) rerun through
#     profile_mixture/experiments/fsa_c20_balibase.py on all three benchmarks WITH posterior caching (<model>_LGp_one__<bench>), so the
#     decoding variants (consistency / progressive / refinement) can be produced from cache.
#  2. TKF92 K=20 indel mixture through the harness on SABmark / OXBench (its BAliBASE row came from the harness).
#  3. TKF92 + anchored CherryML C=20 site mixture through fsa_c20_balibase.py on SABmark / OXBench (slow).
cd "$(dirname "$0")/../.."
GPU=${1:-1}; MINFREE=${MINFREE:-8000}; OUT=experiments/benchmarks; BENCH=$HOME/bio-datasets/data/bench1.0; PM=$PWD/experiments/profile_mixture
PY=$PWD/.venv/bin/python; C20=$PM/results/fsa_c20_balibase
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
bdir() { case $1 in bali3pdbm) echo $HOME/bio-datasets/data/balibase/bali3pdbm;; *) echo $BENCH/$1;; esac; }
c20() {  # indel subst clock bench outname [cache]
  local i=$1 su=$2 cl=$3 b=$4 o=$5 cache=$6; [ -f $C20/$o.json ] && return; gate
  echo "$(date -Is) gpu${GPU}FX start c20 $b $o"
  ( cd $PM && PYTHONPATH=$PM/src:$PM JAX_CACHE_DIR=$HOME/.cache/jax-fsa-c20 $PY experiments/fsa_c20_balibase.py --indel $i --subst $su --clock $cl --balibase-dir $(bdir $b) --out $C20/$o.json ${cache:+--cache-name $cache} > $C20/$o.log 2>&1 )
  echo "$(date -Is) gpu${GPU}FX end c20 $b $o rc=$?"; }
harness() {  # name method bench extra-args...
  local n=$1 m=$2 b=$3; shift 3; [ -f $OUT/${b}_$n.json ] && return; gate
  echo "$(date -Is) gpu${GPU}FX start bench $b $n"
  $PY -u experiments/expected_pairwise_balibase.py --method $m --method-name ${n}__$b --fsa on --fsa-sps --balibase-dir $BENCH/$b --out $OUT/${b}_$n.json "$@" > $OUT/${b}_$n.log 2>&1
  echo "$(date -Is) gpu${GPU}FX end bench $b $n rc=$?"; }
( for b in bali3pdbm sabrem oxm; do c20 tkf92 LGp one $b cache_runs/tkf92_LGp_one_$b tkf92_LGp_one__$b; done
  for b in bali3pdbm sabrem oxm; do c20 mixfrag LGp one $b cache_runs/mixfrag_LGp_one_$b mixfrag_LGp_one__$b; done
  for b in sabrem oxm; do harness tkf92_K20 tkf92_mixture $b --params pfam/tkf92_mixture_K20_train.npz --anchor experiments/tkf92_anchor_mixtures.json; done
  for b in sabrem oxm; do c20 tkf92 CML20 anchor $b tkf92_CML20_anchor_$b; done
  echo "$(date -Is) gpu${GPU}FX end fixed-models chain" ) >> $LOG 2>&1 &
