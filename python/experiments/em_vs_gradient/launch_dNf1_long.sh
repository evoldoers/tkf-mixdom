#!/bin/bash
# Longer svi-BW training + hybrid tails for MixDom dNf1 (identity-classdist
# per-domain recipe, corrected M-step): resume the 540-iteration checkpoint to
# 1000 (score) and 2000 (score + BAliBASE); then from the 2000-iteration point
# (a) Adam-on-Q, minibatch 2000, lr 0.01, 2000 steps and (b) L-BFGS on a fixed
# 20k-pair training subset, each scored + BAliBASE. Waits for a log marker and
# for >= MINFREE MiB free on the GPU before each heavy step.
# Usage: launch_dNf1_long.sh <n_dom> <gpu> <wait-marker-regex>
cd "$(dirname "$0")/../.."
N=$1; GPU=$2; MARK=$3; MINFREE=${MINFREE:-6500}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log; CV=experiments/em_vs_gradient/mixdom_ckpt_convert.py
P=gpu${GPU}L; B=pfam/svibw_d${N}f1_perdomain_v4_fix
COMMON="--svi-bw --precompiled-pairs pfam/precompiled_v3_train --n-dom $N --n-frag 1 --n-classes $N --classdist-init identity --freeze-classdist --estimate-subst --estimate-sexch --svi-batch 2000 --svi-tau 1.0 --svi-kappa 0.7 --seed 42 --val-every 0"
free_mib() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}'; }
gate() { until [ $(free_mib) -ge $MINFREE ]; do sleep 120; done; }
score() {  # tag ckpt [balibase]
  echo "$(date -Is) $P start score $1"
  for try in 1 2 3; do gate
    .venv/bin/python -u experiments/em_vs_gradient/score_pathcond_val.py --checkpoint $2 --out experiments/ordsum_val/pathcond_$1.json > experiments/ordsum_val/pathcond_$1.log 2>&1
    grep -q PATH-CONDITIONED experiments/ordsum_val/pathcond_$1.log && break; sleep 300; done
  for try in 1 2 3; do gate
    MODEL=d3f1 CKPT=$PWD/$2 .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > experiments/ordsum_val/score_$1.log 2>&1
    grep -q 'LL/pair  =' experiments/ordsum_val/score_$1.log && break; sleep 300; done
  if [ "$3" = balibase ]; then gate
    .venv/bin/python -u experiments/expected_pairwise_balibase.py --method mixdom --params $2 --method-name mixdom_$1 --fsa on --fsa-sps \
       --out experiments/expected_balibase/expected_balibase_mixdom_$1_withsps.json > experiments/expected_balibase/expected_balibase_mixdom_$1_withsps.log 2>&1
  fi
  echo "$(date -Is) $P end score $1 rc=$?"
}
( until grep -q -E "$MARK" $LOG 2>/dev/null; do sleep 120; done
  gate; cp $B.npz ${B}_1000.npz
  echo "$(date -Is) $P start resume d${N}f1 540->1000"
  .venv/bin/python -u train_pfam.py $COMMON --n-iter 1000 --checkpoint ${B}_1000.npz > ${B}_1000.log 2>&1
  echo "$(date -Is) $P end resume d${N}f1 1000 rc=$?"
  score d${N}f1_v4_fix_1000 ${B}_1000.npz
  gate; cp ${B}_1000.npz ${B}_2000.npz
  echo "$(date -Is) $P start resume d${N}f1 1000->2000"
  .venv/bin/python -u train_pfam.py $COMMON --n-iter 2000 --checkpoint ${B}_2000.npz > ${B}_2000.log 2>&1
  echo "$(date -Is) $P end resume d${N}f1 2000 rc=$?"
  score d${N}f1_v4_fix_2000 ${B}_2000.npz balibase
  JAX_PLATFORMS=cpu .venv/bin/python $CV to-maraschino ${B}_2000.npz ${B}_2000_init.npz > /dev/null 2>&1
  gate; echo "$(date -Is) $P start adam-switch d${N}f1 from 2000"
  .venv/bin/python -u train_pfam.py --adam --precompiled-pairs pfam/precompiled_v3_train --n-dom $N --n-frag 1 --adam-batch 2000 --adam-steps 2000 --adam-lr 0.01 --seed 42 \
     --init-from-maraschino ${B}_2000_init.npz --checkpoint pfam/adam_d${N}f1_from2000.npz --fresh > pfam/adam_d${N}f1_from2000.log 2>&1
  echo "$(date -Is) $P end adam-switch d${N}f1 rc=$?"
  JAX_PLATFORMS=cpu .venv/bin/python $CV to-scoring pfam/adam_d${N}f1_from2000.npz ${B}_2000.npz pfam/adam_d${N}f1_from2000_scoring.npz > /dev/null 2>&1
  score d${N}f1_adam_from2000 pfam/adam_d${N}f1_from2000_scoring.npz balibase
  gate; echo "$(date -Is) $P start lbfgs-refine d${N}f1 from 2000"
  .venv/bin/python -u experiments/em_vs_gradient/mixdom_lbfgs_refine.py --init ${B}_2000_init.npz --template ${B}_2000.npz --n-sub 20000 --maxfun 80 \
     --out pfam/lbfgs_d${N}f1_from2000.npz > pfam/lbfgs_d${N}f1_from2000.log 2>&1
  echo "$(date -Is) $P end lbfgs-refine d${N}f1 rc=$?"
  score d${N}f1_lbfgs_from2000 pfam/lbfgs_d${N}f1_from2000.npz balibase
  echo "$(date -Is) $P end long-chain d${N}f1" ) >> $LOG 2>&1 &
echo "armed d${N}f1 long chain gpu=$GPU after /$MARK/"
