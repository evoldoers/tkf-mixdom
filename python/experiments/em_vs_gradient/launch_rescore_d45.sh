#!/bin/bash
# Re-run the d4f1 order-summed scoring (OOM on shared GPU 0) and add the
# path-conditioned held-out scoring for d4f1 and d5f1; retry d5f1 ordsum if
# its own chain OOMs. GPU 1, after the d5f1 chain, memory-gated.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 1 | awk -F', ' '{print $2-$1}') -ge 5000 ]; do sleep 120; done; }
( until grep -q 'end ordsum_val d5f1_perdomain_v4_fix rc=' $LOG 2>/dev/null; do sleep 120; done
  echo "$(date -Is) gpu1R start rescore d45"
  for T in d4f1_perdomain_v4_fix d5f1_perdomain_v4_fix; do
    for try in 1 2 3; do grep -q 'LL/pair  =' experiments/ordsum_val/score_$T.log 2>/dev/null && break; gate
      MODEL=d3f1 CKPT=$PWD/pfam/svibw_$T.npz .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > experiments/ordsum_val/score_$T.log 2>&1; sleep 60; done
    for try in 1 2 3; do grep -q PATH-CONDITIONED experiments/ordsum_val/pathcond_svibw_$T.log 2>/dev/null && break; gate
      .venv/bin/python -u experiments/em_vs_gradient/score_pathcond_val.py --checkpoint pfam/svibw_$T.npz --out experiments/ordsum_val/pathcond_svibw_$T.json > experiments/ordsum_val/pathcond_svibw_$T.log 2>&1; sleep 60; done
  done
  echo "$(date -Is) gpu1R end rescore d45" ) >> $LOG 2>&1 &
echo armed
