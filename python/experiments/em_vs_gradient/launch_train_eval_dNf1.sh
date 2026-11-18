#!/bin/bash
# Train MixDom dNf1 (per-domain substitution via the identity-classdist
# MixDom2 path, the d3f1 recipe) under the corrected M-step, then evaluate:
# BAliBASE posteriors + FSA at gap factors 1 and 0, and v3_val order-summed LL.
# Usage: launch_train_eval_dNf1.sh <n_dom> <gpu> <wait-marker-regex>
cd "$(dirname "$0")/../.."
N=$1; GPU=$2; MARK=$3
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
TAG=d${N}f1_perdomain_v4_fix; CK=pfam/svibw_${TAG}.npz
( until grep -q -E "$MARK" $LOG 2>/dev/null; do sleep 120; done
  echo "$(date -Is) gpu${GPU}T start train $TAG"
  .venv/bin/python -u train_pfam.py --svi-bw --precompiled-pairs pfam/precompiled_v3_train \
     --n-dom $N --n-frag 1 --n-classes $N --classdist-init identity --freeze-classdist \
     --estimate-subst --estimate-sexch --svi-batch 2000 --svi-tau 1.0 --svi-kappa 0.7 --n-iter 540 --seed 42 \
     --checkpoint $CK --fresh > pfam/svibw_${TAG}.log 2>&1
  echo "$(date -Is) gpu${GPU}T end train $TAG rc=$?"
  echo "$(date -Is) gpu${GPU}T start balibase mixdom_$TAG"
  .venv/bin/python -u experiments/expected_pairwise_balibase.py --method mixdom --params $CK \
     --method-name mixdom_$TAG --fsa on --fsa-sps \
     --out experiments/expected_balibase/expected_balibase_mixdom_${TAG}_withsps.json \
     > experiments/expected_balibase/expected_balibase_mixdom_${TAG}_withsps.log 2>&1
  echo "$(date -Is) gpu${GPU}T end balibase mixdom_$TAG rc=$?"
  echo "$(date -Is) gpu${GPU}T start ordsum_val $TAG"
  MODEL=d3f1 CKPT=$PWD/$CK .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py \
     > experiments/ordsum_val/score_${TAG}.log 2>&1
  echo "$(date -Is) gpu${GPU}T end ordsum_val $TAG rc=$?" ) >> $LOG 2>&1 &
echo "armed dN=$N gpu=$GPU after /$MARK/"
