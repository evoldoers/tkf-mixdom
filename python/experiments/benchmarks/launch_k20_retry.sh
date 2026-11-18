#!/bin/bash
# Re-run the TKF92 K=20 mixture on the families that ran out of GPU memory in launch_fixed_models_bench.sh
# (shared GPUs; full pair batches), with a small pair chunk, so the posterior caches tkf92_K20__<bench> are complete.
# Usage: launch_k20_retry.sh <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-1}; MINFREE=${MINFREE:-6000}; OUT=experiments/benchmarks; BENCH=$HOME/bio-datasets/data/bench1.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
retry() {  # bench families
  local b=$1 fams=$2; [ -f $OUT/${b}_tkf92_K20_retry.json ] && return; gate
  echo "$(date -Is) gpu${GPU}KR start bench $b tkf92_K20 retry ($fams)"
  .venv/bin/python -u experiments/expected_pairwise_balibase.py --method tkf92_mixture --method-name tkf92_K20__$b --params pfam/tkf92_mixture_K20_train.npz --anchor experiments/tkf92_anchor_mixtures.json --families $fams --pair-chunk 32 --fsa on --fsa-sps --balibase-dir $BENCH/$b --out $OUT/${b}_tkf92_K20_retry.json > $OUT/${b}_tkf92_K20_retry.log 2>&1
  echo "$(date -Is) gpu${GPU}KR end bench $b tkf92_K20 retry rc=$?"; }
( retry sabrem sup_092,sup_126,sup_167,sup_190,sup_215,sup_396,twi_111
  retry oxm 136s31,_22t53,_22t56,___136
  echo "$(date -Is) gpu${GPU}KR end k20 retry chain" ) >> $LOG 2>&1 &
