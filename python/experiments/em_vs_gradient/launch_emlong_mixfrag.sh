#!/bin/bash
# Closed-form EM alone, 1500 passes, on the two unaligned MixFrag objectives
# where 300 passes did not reach the optimum to 1e-3; then the custom-VJP
# Pfam MixFrag run that OOMed. GPU 1, memory-gated (MixFrag 2D needs chunk 8).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
GPU=${1:-1}; MARK=${2:-.}; MINFREE=${MINFREE:-7000}
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log; R=experiments/em_vs_gradient/results_emlong; RV=experiments/em_vs_gradient/results_vjp; mkdir -p $R
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
( until grep -q -E "$MARK" $LOG 2>/dev/null; do sleep 120; done
  for c in ${CORPORA:-sim_mixfrag pfam}; do gate; echo "$(date -Is) gpu${GPU}E start emlong $c mixfrag"
    .venv/bin/python -u experiments/em_vs_gradient/compare_optimizers.py --corpus $c --model mixfrag --n-pairs 1000 --chunk 8 --seed 0 \
      --methods em --em-max-iter 1500 --tol 1e-7 --out $R/${c}_mixfrag.json > $R/${c}_mixfrag.log 2>&1
    rc=$?; echo "$(date -Is) gpu${GPU}E end emlong $c mixfrag rc=$rc"; done
  gate; echo "$(date -Is) gpu${GPU}E start vjp pfam mixfrag (rerun)"
  .venv/bin/python -u experiments/em_vs_gradient/compare_optimizers.py --corpus pfam --model mixfrag --n-pairs 1000 --chunk 8 --seed 0 \
      --methods lbfgs_vjp --adam-lrs 0.1 --lbfgs-max-fun 300 --adam-max-iter 300 --tol 1e-7 --out $RV/pfam_mixfrag.json > $RV/pfam_mixfrag.log 2>&1
  rc=$?; echo "$(date -Is) gpu${GPU}E end vjp pfam mixfrag rc=$rc" ) >> $LOG 2>&1 &
echo armed
