#!/bin/bash
# Alignment-given (summarised-count) EM vs L-BFGS vs Adam: 2 corpora x {TKF92, MixFrag F=2}, sequential on one GPU.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1
OUT=experiments/em_vs_gradient/results_aligned; mkdir -p $OUT
for job in pfam:tkf92 pfam:mixfrag sim_tkf92:tkf92 sim_tkf92:mixfrag; do
  c=${job%:*}; m=${job#*:}
  echo "$(date -Is) start $job"
  .venv/bin/python -u experiments/em_vs_gradient/compare_optimizers_aligned.py \
     --corpus $c --model $m --n-pairs 1000 --seed 0 --methods em lbfgs adam --adam-lrs 0.1 0.01 \
     --em-max-iter 2000 --lbfgs-max-fun 500 --adam-max-iter 3000 --tol 1e-7 \
     --out $OUT/${c}_${m}.json > $OUT/${c}_${m}.log 2>&1
  echo "$(date -Is) end $job rc=$?"
done
