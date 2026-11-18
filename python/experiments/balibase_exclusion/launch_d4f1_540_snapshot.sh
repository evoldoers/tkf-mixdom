#!/bin/bash
# The first d4f1_nobali 540-iteration run segfaulted at iteration 1 (GPU
# contention) and the chain fell through to a fresh 1080-iteration run, so the
# 540-iteration snapshot is missing. Train it separately (same seed/recipe)
# and score + BAliBASE it, on GPU 1 after the d4f1 1620 stage.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log; CORPUS=pfam/precompiled_v3_train_nobali; B=pfam/svibw_d4f1_nobali
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 1 | awk -F', ' '{print $2-$1}') -ge 5000 ]; do sleep 120; done; }
( until grep -q 'gpu1N end train d4f1_nobali to 1620' $LOG 2>/dev/null; do sleep 120; done
  for try in 1 2 3; do gate; echo "$(date -Is) gpu1S start train d4f1_nobali_540 (snapshot rerun, try $try)"
    .venv/bin/python -u train_pfam.py --svi-bw --precompiled-pairs $CORPUS --n-dom 4 --n-frag 1 --n-classes 4 --classdist-init identity --freeze-classdist \
      --estimate-subst --estimate-sexch --svi-batch 2000 --svi-tau 1.0 --svi-kappa 0.7 --seed 42 --val-every 0 --jax-cache-dir /home/yam/.cache/tkf-mixdom-jax --n-iter 540 --checkpoint ${B}_540.npz --fresh > ${B}_540.log 2>&1
    rc=$?; echo "$(date -Is) gpu1S end train d4f1_nobali_540 rc=$rc"; [ $rc -eq 0 ] && break; sleep 600; done
  gate; .venv/bin/python -u experiments/em_vs_gradient/score_pathcond_val.py --checkpoint ${B}_540.npz --out experiments/ordsum_val/pathcond_d4f1_nobali_540.json > experiments/ordsum_val/pathcond_d4f1_nobali_540.log 2>&1
  gate; MODEL=d3f1 CKPT=$PWD/${B}_540.npz .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > experiments/ordsum_val/score_d4f1_nobali_540.log 2>&1
  gate; .venv/bin/python -u experiments/expected_pairwise_balibase.py --method mixdom --params ${B}_540.npz --method-name mixdom_d4f1_nobali_540 --fsa on --fsa-sps \
     --out experiments/expected_balibase/expected_balibase_mixdom_d4f1_nobali_540_withsps.json > experiments/expected_balibase/expected_balibase_mixdom_d4f1_nobali_540_withsps.log 2>&1
  echo "$(date -Is) gpu1S end score d4f1_nobali_540" ) >> $LOG 2>&1 &
echo armed
