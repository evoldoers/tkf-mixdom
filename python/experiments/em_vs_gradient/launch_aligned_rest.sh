#!/bin/bash
# Remaining alignment-given jobs: sim jobs on GPU 0 now; pfam:mixfrag on GPU 1 once pfam:tkf92 has saved.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1
OUT=experiments/em_vs_gradient/results_aligned
one() { .venv/bin/python -u experiments/em_vs_gradient/compare_optimizers_aligned.py \
     --corpus $1 --model $2 --n-pairs 1000 --seed 0 --methods em lbfgs adam --adam-lrs 0.1 0.01 \
     --em-max-iter 2000 --lbfgs-max-fun 500 --adam-max-iter 3000 --tol 1e-7 \
     --out $OUT/${1}_${2}.json > $OUT/${1}_${2}.log 2>&1; }
( for j in sim_tkf92:tkf92 sim_tkf92:mixfrag; do echo "$(date -Is) gpu0 start $j"; CUDA_VISIBLE_DEVICES=0 one ${j%:*} ${j#*:}; echo "$(date -Is) gpu0 end $j rc=$?"; done ) >> $OUT/../results_aligned_queue.log 2>&1 &
( until grep -q '^saved' $OUT/pfam_tkf92.log 2>/dev/null; do sleep 60; done; echo "$(date -Is) gpu1 start pfam:mixfrag"; CUDA_VISIBLE_DEVICES=1 one pfam mixfrag; echo "$(date -Is) gpu1 end pfam:mixfrag rc=$?" ) >> $OUT/../results_aligned_queue.log 2>&1 &
