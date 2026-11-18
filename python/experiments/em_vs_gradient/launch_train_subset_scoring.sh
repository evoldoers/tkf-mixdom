#!/bin/bash
# Training-set path-conditioned log-likelihood of the released d2f1 and its 1,620-iteration restarts on one fixed
# 20,000-pair subset of the BAliBASE-free training corpus (same pairs for every fit). Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-1}; MINFREE=${MINFREE:-3000}; LOG=experiments/em_vs_gradient/results_rest_queue.log; SUB=pfam/precompiled_train_nobali_sub20k_seed0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
ckpt_iter() { JAX_PLATFORMS=cpu .venv/bin/python -c "import numpy as np,sys; d=np.load(sys.argv[1],allow_pickle=True); print(int(d['em_iter']) if 'em_iter' in d.files else 0)" $1 2>/dev/null || echo 0; }
( for tag in d2f1_nobali_1620 d2f1_nobali_1620_seed1 d2f1_nobali_1620_seed2 d2f1_nobali_1620_seed4 d2f1_nobali_1620_seed3; do
    ck=pfam/svibw_$tag.npz; out=experiments/ordsum_val/pathcond_train20k_$tag.log
    until [ -f $ck ] && [ "$(ckpt_iter $ck)" -ge 1620 ]; do sleep 300; done
    grep -q 'LL/pair' $out 2>/dev/null && continue
    gate; echo "$(date -Is) gpu${GPU}TS start train-subset pathcond $tag"
    .venv/bin/python -u experiments/em_vs_gradient/score_pathcond_val.py --checkpoint $ck --precompiled $SUB --out ${out%.log}.json > $out 2>&1
    echo "$(date -Is) gpu${GPU}TS end train-subset pathcond $tag rc=$? $(grep -o 'LL/pair *= *-[0-9.]*' $out | tail -1)"
  done; echo "$(date -Is) gpu${GPU}TS end train-subset scoring chain" ) >> $LOG 2>&1 &
