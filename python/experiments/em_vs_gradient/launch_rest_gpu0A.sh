#!/bin/bash
# Replacement for the gpu0A chain: wait for the running MixDom EM to exit, then
# MixDom Adam (200 steps per learning rate), then the custom-VJP sim runs.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
RV=experiments/em_vs_gradient/results_vjp; MD=experiments/em_vs_gradient/results_mixdom
LOG=experiments/em_vs_gradient/results_rest_queue.log
vjp() { local methods="lbfgs_vjp"; [ "$2" = tkf92 ] && methods="lbfgs_vjp adam_vjp"
     .venv/bin/python -u experiments/em_vs_gradient/compare_optimizers.py --corpus $1 --model $2 --n-pairs 1000 --chunk 16 --seed 0 \
     --methods $methods --adam-lrs 0.1 --lbfgs-max-fun 300 --adam-max-iter 300 --tol 1e-7 \
     --out $RV/${1}_${2}.json > $RV/${1}_${2}.log 2>&1; }
mixdom_adam() { .venv/bin/python -u train_pfam.py --adam --precompiled-pairs pfam/precompiled_sub1000_seed0 --n-dom 3 --n-frag 1 \
     --adam-batch 1000 --adam-steps 200 --adam-lr $1 --seed 0 --init-from-maraschino $MD/init_d3f1_seed0.npz \
     --checkpoint $MD/adam_d3f1_lr$1.npz --fresh > $MD/adam_d3f1_lr$1.log 2>&1; }
run() { echo "$(date -Is) $1 start $2 $3 $4"; "$2" $3 $4; echo "$(date -Is) $1 end $2 $3 $4 rc=$?"; }
stoch() { MODES=adam_tkf92 experiments/em_vs_gradient/stochastic/run_pfam_stochastic.sh >> experiments/em_vs_gradient/stochastic/driver.log 2>&1; }
( while pgrep -f 'train_pfam.py --svi-bw' > /dev/null; do sleep 60; done
  run gpu0A stoch; run gpu0A mixdom_adam 0.01; run gpu0A mixdom_adam 0.1
  run gpu0A vjp sim_tkf92 tkf92; run gpu0A vjp sim_tkf91 tkf91; run gpu0A vjp sim_mixfrag mixfrag ) >> $LOG 2>&1 &
echo "gpu0A replacement armed"
