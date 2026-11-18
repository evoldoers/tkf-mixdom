#!/bin/bash
# Remaining runs after the MixFrag OOM: MixFrag unaligned jobs at chunk 8.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
R=experiments/em_vs_gradient/results; RV=experiments/em_vs_gradient/results_vjp
LOG=experiments/em_vs_gradient/results_rest_queue.log
saved() { until grep -q '^saved ' "$1" 2>/dev/null; do sleep 90; done; }
chunk() { [ "$1" = mixfrag ] && echo 8 || echo 16; }
unal() { .venv/bin/python -u experiments/em_vs_gradient/compare_optimizers.py --corpus $1 --model $2 --n-pairs 1000 --chunk $(chunk $2) --seed 0 \
     --methods em lbfgs adam --adam-lrs 0.1 0.01 --em-max-iter 300 --lbfgs-max-fun 300 --adam-max-iter 1000 --tol 1e-7 \
     --out $R/${1}_${2}.json > $R/${1}_${2}.log 2>&1; }
vjp() { local methods="lbfgs_vjp"; [ "$2" = tkf92 ] && methods="lbfgs_vjp adam_vjp"
     .venv/bin/python -u experiments/em_vs_gradient/compare_optimizers.py --corpus $1 --model $2 --n-pairs 1000 --chunk $(chunk $2) --seed 0 \
     --methods $methods --adam-lrs 0.1 --lbfgs-max-fun 300 --adam-max-iter 300 --tol 1e-7 \
     --out $RV/${1}_${2}.json > $RV/${1}_${2}.log 2>&1; }
run() { echo "$(date -Is) $1 start $2 $3 $4"; "$2" $3 $4; echo "$(date -Is) $1 end $2 $3 $4 rc=$?"; }
( saved $RV/sim_tkf92_tkf92.log; export CUDA_VISIBLE_DEVICES=0
  run gpu0A vjp sim_tkf91 tkf91; run gpu0A vjp sim_mixfrag mixfrag ) >> $LOG 2>&1 &
( saved $RV/pfam_tkf91.log; export CUDA_VISIBLE_DEVICES=1
  run gpu1A unal pfam mixfrag; run gpu1A vjp pfam mixfrag ) >> $LOG 2>&1 &
( saved $R/pfam_tkf92.log; export CUDA_VISIBLE_DEVICES=1
  run gpu1B unal sim_mixfrag mixfrag; run gpu1B vjp pfam tkf92 ) >> $LOG 2>&1 &
echo armed
