#!/bin/bash
# Longer MixDom Adam runs (2000 steps per learning rate) after the 200-step pair finishes.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
MD=experiments/em_vs_gradient/results_mixdom; LOG=experiments/em_vs_gradient/results_rest_queue.log
long() { .venv/bin/python -u train_pfam.py --adam --precompiled-pairs pfam/precompiled_sub1000_seed0 --n-dom 3 --n-frag 1 \
     --adam-batch 1000 --adam-steps 2000 --adam-lr $1 --seed 0 --init-from-maraschino $MD/init_d3f1_seed0.npz \
     --checkpoint $MD/adam_d3f1_lr$1_long.npz --fresh > $MD/adam_d3f1_lr$1_long.log 2>&1; }
( until grep -q 'end mixdom_adam 0.1 ' $LOG 2>/dev/null; do sleep 60; done
  for lr in 0.01 0.1; do echo "$(date -Is) gpu0L start mixdom_adam_long $lr"; long $lr; echo "$(date -Is) gpu0L end mixdom_adam_long $lr rc=$?"; done ) >> $LOG 2>&1 &
echo armed
