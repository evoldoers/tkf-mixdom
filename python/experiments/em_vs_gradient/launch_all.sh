#!/bin/bash
# Launch the six deterministic full-batch comparisons (2 corpora x 3 models)
# as four sequential queues, two per GPU (four concurrent gradient jobs on one
# 11 GB card OOM at the 256x256 shapes).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1
OUT=experiments/em_vs_gradient/results; mkdir -p $OUT
one() {  # corpus model
  .venv/bin/python -u experiments/em_vs_gradient/compare_optimizers.py \
     --corpus $1 --model $2 --n-pairs 1000 --chunk 16 --seed 0 \
     --methods em lbfgs adam --adam-lrs 0.1 0.01 \
     --em-max-iter 300 --lbfgs-max-fun 300 --adam-max-iter 1000 --tol 1e-7 \
     --out $OUT/${1}_${2}.json > $OUT/${1}_${2}.log 2>&1
}
queue() {  # gpu then list of corpus:model
  local gpu=$1; shift
  ( for job in "$@"; do
      echo "$(date -Is) gpu$gpu start $job"; CUDA_VISIBLE_DEVICES=$gpu one ${job%:*} ${job#*:}; echo "$(date -Is) gpu$gpu end $job rc=$?"
    done ) >> $OUT/queue_gpu$gpu.log 2>&1 &
  echo "queue gpu$gpu pid $! : $*"
}
queue 1 pfam:tkf91 pfam:mixfrag
queue 1 pfam:tkf92 sim_tkf92:mixfrag
queue 0 sim_tkf92:tkf91
queue 0 sim_tkf92:tkf92
