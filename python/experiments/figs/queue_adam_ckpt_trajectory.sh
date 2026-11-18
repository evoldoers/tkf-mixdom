#!/bin/bash
# Early-stopping test with the paper's own validation metric. After the running cherry-pair diagnostic ends (marker
# 'end gpu1 part' in the queue log), on GPU 1:
#   1. rerun the d2f1 Adam tail (deterministic: it reproduces the released tail's trajectory exactly) keeping the
#      current parameters every 100 steps and at step 1999 -> pfam/adam_d2f1_nobali_from1620_traj_stepNNNN.npz;
#   2. score each kept checkpoint, plus the svi-BW start (step 0) and the released tail checkpoint (= step 1824, the
#      lowest-minibatch-loss step), with score_pathcond_val.py on every 10th pair of the precompiled validation set
#      (11,664 pairs; the paper's path-conditioned LL/pair on a fixed subsample);
#   3. write experiments/ordsum_val/adam_d2f1_traj_stride10.tsv (step, LL/pair).
# Usage: <gpu>   (SCORE_ONLY=1: skip the rerun, score the kept checkpoints)
cd "$(dirname "$0")/../.."
GPU=${1:-1}; Q=experiments/em_vs_gradient/results_rest_queue.log; P=gpu${GPU}CK
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU JAX_PLATFORMS=cuda
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
B=pfam/adam_d2f1_nobali_from1620_traj; OUT=experiments/ordsum_val/adam_d2f1_traj_stride10.tsv
( until grep -q 'FG end gpu1 part' $Q 2>/dev/null; do sleep 300; done
  if [ "${SCORE_ONLY:-0}" != 1 ]; then
  echo "$(date -Is) $P start d2f1 Adam tail rerun with kept checkpoints every 100 steps"
  .venv/bin/python -u train_pfam.py --adam --precompiled-pairs pfam/precompiled_v3_train_nobali --n-dom 2 --n-frag 1 \
     --adam-batch 2000 --adam-steps 2000 --adam-lr 0.01 --seed 42 --init-from-maraschino pfam/svibw_d2f1_nobali_1620_init.npz \
     --checkpoint $B.npz --fresh --ckpt-keep-every 100 > $B.log 2>&1
  echo "$(date -Is) $P end rerun rc=$? kept $(ls ${B}_step*.npz 2>/dev/null | wc -l) checkpoints"
  fi
  printf "step\tcheckpoint\tll_per_pair\tn_pairs\n" > $OUT
  score() { local tag=$1; local ck=$2; local j=experiments/ordsum_val/pathcond_traj_${tag}_stride10.json   # separate locals: bash expands a whole `local` line before assigning
    [ -s $j ] || .venv/bin/python -u experiments/em_vs_gradient/score_pathcond_val.py --checkpoint $ck --stride 10 --out $j > ${j%.json}.log 2>&1
    python3 -c "import json,sys; d=json.load(open('$j')); print('$tag\t$ck\t%.6f\t%d' % (d['ll_per_pair'], d['n_pairs']))" >> $OUT 2>/dev/null || echo "$tag\t$ck\tFAILED\t0" >> $OUT; }
  score 0000 pfam/svibw_d2f1_nobali_1620.npz
  for f in $(ls ${B}_step*.npz | sort); do s=$(echo $f | sed -E 's/.*_step([0-9]+)\.npz/\1/'); score $s $f; done
  score released1824 pfam/adam_d2f1_nobali_from1620.npz
  echo "$(date -Is) $P scored $(($(wc -l < $OUT) - 1)) checkpoints -> $OUT"
  echo "$(date -Is) $P end adam checkpoint trajectory" ) >> $Q 2>&1 &
echo "armed adam checkpoint trajectory on GPU $GPU (after the cherry-pair diagnostic)"
