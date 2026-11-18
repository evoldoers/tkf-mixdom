#!/bin/bash
# Fixed (non-MixDom) models on SABmark / OXBench through the same harness as BAliBASE Table 8, with posterior
# caching on (so the decoding grid can be rerun from cache): TKF92, MixFrag F=2, CherryML C=20, TKF92 K=20.
# Idempotent; memory-gated. Usage: launch_fixed_models_bench.sh <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-1}; MINFREE=${MINFREE:-8000}; OUT=experiments/benchmarks; BENCH=$HOME/bio-datasets/data/bench1.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
run() {  # name method bench extra-args...
  local n=$1 m=$2 b=$3; shift 3; [ -f $OUT/${b}_$n.json ] && return; gate
  echo "$(date -Is) gpu${GPU}FX start bench $b $n"
  .venv/bin/python -u experiments/expected_pairwise_balibase.py --method $m --method-name ${n}__$b --fsa on --fsa-sps --balibase-dir $BENCH/$b --out $OUT/${b}_$n.json "$@" > $OUT/${b}_$n.log 2>&1
  echo "$(date -Is) gpu${GPU}FX end bench $b $n rc=$?"; }
( for b in sabrem oxm; do
    run tkf92 tkf92 $b --params experiments/tkf92_fitted_params.json
    run mixfrag_F2 mixfrag $b --params params/best/cem_mixfrag_F2_pfamTrain.npz
  done
  for b in sabrem oxm; do
    run cherryml_C20 cherryml_mixture $b --params pfam/cherryml_mixture_C20_n5000.npz --anchor experiments/tkf92_anchor_mixtures.json
    run tkf92_K20 tkf92_mixture $b --params pfam/tkf92_mixture_K20_train.npz --anchor experiments/tkf92_anchor_mixtures.json
  done
  echo "$(date -Is) gpu${GPU}FX end fixed-models chain" ) >> $LOG 2>&1 &
