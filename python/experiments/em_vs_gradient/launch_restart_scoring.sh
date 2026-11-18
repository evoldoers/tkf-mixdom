#!/bin/bash
# Held-out likelihoods of the 1,620-iteration d2f1 restarts (seeds 1-4; same recipe as the released fit): val
# path-conditioned (score_pathcond_val.py; the order-summed column was dropped from the paper), each once its
# checkpoint reaches 1,620 iterations. Outputs experiments/ordsum_val/{pathcond,score}_d2f1_nobali_1620_seed<s>.log.
# Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-1}; MINFREE=${MINFREE:-3000}; LOG=experiments/em_vs_gradient/results_rest_queue.log
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
ckpt_iter() { JAX_PLATFORMS=cpu .venv/bin/python -c "import numpy as np,sys; d=np.load(sys.argv[1],allow_pickle=True); print(int(d['em_iter']) if 'em_iter' in d.files else 0)" $1 2>/dev/null || echo 0; }
( for s in 1 4 3 2; do ck=pfam/svibw_d2f1_nobali_1620_seed$s.npz; tag=d2f1_nobali_1620_seed$s
    until [ -f $ck ] && [ "$(ckpt_iter $ck)" -ge 1620 ]; do sleep 300; done
    if ! grep -q 'LL/pair' experiments/ordsum_val/pathcond_$tag.log 2>/dev/null; then gate; echo "$(date -Is) gpu${GPU}RS start pathcond $tag"
      .venv/bin/python -u experiments/em_vs_gradient/score_pathcond_val.py --checkpoint $ck --out experiments/ordsum_val/pathcond_$tag.json > experiments/ordsum_val/pathcond_$tag.log 2>&1; echo "$(date -Is) gpu${GPU}RS end pathcond $tag rc=$?"; fi
  done; echo "$(date -Is) gpu${GPU}RS end restart scoring chain" ) >> $LOG 2>&1 &
