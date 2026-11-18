#!/bin/bash
# Re-run the Table 3 Pfam/TKF92 unaligned row (the one shown in the convergence figure) with the gather-based DP
# cores, so that the wall-clock panel and the "one E-step costs N gradient passes" statement reflect the current
# code; the pass counts are unchanged (outputs are bit-identical). Runs after the MixDom d2f1 EM-vs-Adam chain,
# on an otherwise quiet GPU (9 GB free), so that per-pass timings are clean. Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-0}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
E=experiments/em_vs_gradient; R=$E/results_gtr; LOG=$E/results_rest_queue.log
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge 9000 ]; do sleep 120; done; }
( until grep -q 'end mixdom d2f1 EM-vs-Adam chain' $LOG 2>/dev/null; do sleep 300; done
  mv $R/pfam_tkf92.json $R/pfam_tkf92.oldtiming.json 2>/dev/null; mv $R/pfam_tkf92.log $R/pfam_tkf92.oldtiming.log 2>/dev/null
  gate; echo "$(date -Is) gpu${GPU}T3 start unal pfam tkf92 (re-timing with gather-based DP)"
  .venv/bin/python -u $E/compare_optimizers.py --corpus pfam --model tkf92 --n-pairs 1000 --chunk 16 --seed 0 \
     --methods em lbfgs adam --adam-lrs 0.1 0.01 --em-max-iter 300 --lbfgs-max-fun 300 --adam-max-iter 1000 --tol 1e-7 \
     --subst-npz pfam/gtr_refit_Qpi.npz --out $R/pfam_tkf92.json > $R/pfam_tkf92.log 2>&1
  echo "$(date -Is) gpu${GPU}T3 end unal pfam tkf92 (re-timing) rc=$?" ) >> $LOG 2>&1 &
