#!/bin/bash
# Score MixFrag F=2..5 (log-space fits in experiments/mixfrag_F/mixfrag_F{F}_logspace.json):
# validation + test likelihoods under both conventions (one 2D scorer), then BAliBASE,
# SABmark (sabrem) and OXBench (oxm) through the published-LG FSA script. Idempotent,
# memory-gated. Usage: launch_mixfrag_F_scoring.sh <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-0}; MINFREE=${MINFREE:-8000}; R=$PWD/experiments/mixfrag_F; PM=$PWD/experiments/profile_mixture; BENCH=$HOME/bio-datasets/data/bench1.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=$PWD/experiments/em_vs_gradient/results_rest_queue.log; PY=$PWD/.venv/bin/python
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
ll() {  # F split kind
  local F=$1 sp=$2 kind=$3 pc=0; [ $kind = pathcond ] && pc=1; local out=$R/${kind}_F${F}_$sp.log
  if [ -f $out ] && [ $(( $(date +%s) - $(stat -c %Y $out) )) -lt 1200 ]; then until [ $(( $(date +%s) - $(stat -c %Y $out) )) -ge 1200 ] || grep -q 'LL/pair  =' $out; do sleep 120; done; fi
  for try in 1 2 3 4; do grep -q 'LL/pair  =' $out 2>/dev/null && return; gate; echo "$(date -Is) gpu${GPU}MF start mixfrag F=$F $kind $sp"
    PATHCOND=$pc MODEL=mixfrag MIXFRAG_PARAMS=$R/mixfrag_F${F}_logspace.json VAL_DIR=$PWD/pfam/precompiled_v3_$sp $PY -u experiments/ordsum_val/score_matchaligned_val.py > $out 2>&1
    echo "$(date -Is) gpu${GPU}MF end mixfrag F=$F $kind $sp rc=$?"; grep -q 'LL/pair  =' $out 2>/dev/null && return; sleep 900; done; }
bench() {  # F benchdir tag
  local F=$1 bd=$2 tag=$3; local out=$PM/results/fsa_c20_balibase/mixfrag${F}_LGp_one_$tag.json
  [ -f $out ] && return; gate; echo "$(date -Is) gpu${GPU}MF start mixfrag F=$F bench $tag"
  ( cd $PM && PYTHONPATH=$PM/src:$PM JAX_CACHE_DIR=$HOME/.cache/jax-fsa-c20 $PY experiments/fsa_c20_balibase.py --indel mixfrag --subst LGp --clock one --indel-params $R/mixfrag_F${F}_logspace.json --balibase-dir $bd --out $out > $R/bench_F${F}_$tag.log 2>&1 )
  echo "$(date -Is) gpu${GPU}MF end mixfrag F=$F bench $tag rc=$?"; }
( for F in 2 3 4 5; do
    until [ -f $R/mixfrag_F${F}_logspace.json ]; do sleep 300; done
    ll $F val pathcond; ll $F val ordsum; ll $F test pathcond; ll $F test ordsum
    bench $F $HOME/bio-datasets/data/balibase/bali3pdbm bali3pdbm; bench $F $BENCH/sabrem sabrem; bench $F $BENCH/oxm oxm
  done; echo "$(date -Is) gpu${GPU}MF end mixfrag-F scoring chain" ) >> $LOG 2>&1 &
echo armed
