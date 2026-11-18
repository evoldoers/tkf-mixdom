#!/bin/bash
# Alignment-given MixFrag-S rows of Table 3 (EM vs L-BFGS vs Adam on the 1D path-conditioned log-likelihood of the
# 1,000 aligned pairs; compare_optimizers_aligned.py --model mixfrag_s): {pfam, sim_mixfrag_s} x F in {2, 3}, with the
# GTR refit (pfam/gtr_refit_Qpi.npz) as the initial (S_f, pi_f) of every fragtype and as the simulation base, as
# launch_table3_gtr.sh does for the other rows.  Outputs results_aligned_gtr/{pfam,sim_mixfrag_s}_mixfrag_s{F}.json
# (the paths convergence_figure.py reads).  Usage: <gpu>; skips a job whose log already ends in 'saved'.
cd "$(dirname "$0")/../.."
GPU=${1:-0}; MINFREE=${MINFREE:-3000}; SUB=pfam/gtr_refit_Qpi.npz
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
E=experiments/em_vs_gradient; RA=$E/results_aligned_gtr; mkdir -p $RA
LOG=$E/results_aligned_queue.log
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
als() { # corpus F
  for try in 1 2 3; do grep -q '^saved ' $RA/${1}_mixfrag_s${2}.log 2>/dev/null && return; gate
    echo "$(date -Is) gpu${GPU} start aligned $1 mixfrag_s F=$2 (try $try)"
    .venv/bin/python -u $E/compare_optimizers_aligned.py --corpus $1 --model mixfrag_s --n-fragtypes $2 --n-pairs 1000 --seed 0 \
       --methods em lbfgs adam --adam-lrs 0.1 0.01 --em-max-iter 2000 --lbfgs-max-fun 500 --adam-max-iter 3000 --tol 1e-7 \
       --subst-npz $SUB --out $RA/${1}_mixfrag_s${2}.json > $RA/${1}_mixfrag_s${2}.log 2>&1
    echo "$(date -Is) gpu${GPU} end aligned $1 mixfrag_s F=$2 rc=$?"
    grep -q '^saved ' $RA/${1}_mixfrag_s${2}.log 2>/dev/null && return; sleep 600; done; }
( als pfam 2; als sim_mixfrag_s 2; als pfam 3; als sim_mixfrag_s 3
  echo "$(date -Is) gpu${GPU} end aligned MixFrag-S rows" ) >> $LOG 2>&1 &
