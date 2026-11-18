#!/bin/bash
# Retrain MixDom dNf1 (identity-classdist per-domain recipe, corrected M-step)
# on the BAliBASE-homolog-excluded corpus, 3x as long as before: staged
# 540 -> 1080 -> 1620 iterations, held-out scoring at each stage, BAliBASE at
# 540 and 1620, then Adam-on-Q and L-BFGS tails from the 1620 point.
# Usage: launch_retrain_nobali.sh <n_dom> <gpu> <wait-marker-regex>
cd "$(dirname "$0")/../.."
N=$1; GPU=$2; MARK=$3; MINFREE=${MINFREE:-6500}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log; CV=experiments/em_vs_gradient/mixdom_ckpt_convert.py
# Env overrides: CORPUS (training corpus dir), TAG (checkpoint/score tag), START540 (an
# existing 540-iteration checkpoint to continue from instead of training afresh),
# SKIP540=1 (skip the 540-iteration scoring, e.g. when it already exists under another tag), SKIP1080=1 likewise.
NFRAG=${NFRAG:-1}; NCLASSES=${NCLASSES:-$N}   # NCLASSES=F with CLASSDIST_OPTS="--classdist-init fragchar --freeze-classdist": one matrix per fragtype
CORPUS=${CORPUS:-pfam/precompiled_v3_train_nobali}; P=gpu${GPU}N; TAG=${TAG:-d${N}f${NFRAG}_nobali}; B=pfam/svibw_$TAG
CLASSDIST_OPTS=${CLASSDIST_OPTS:-"--classdist-init identity --freeze-classdist"}   # e.g. "--tie-classdist --classdist-init auto" for the tied-mixture control
TAILS=${TAILS:-1}   # 0 = EM stages only (no Adam / L-BFGS tails; the tails assume the identity classdist)
EXTRA_OPTS=${EXTRA_OPTS:-}   # extra train_pfam.py flags, e.g. --freeze-ext-offdiag (diagonal fragtype extension) for f>1
COMMON="--svi-bw --precompiled-pairs $CORPUS --n-dom $N --n-frag $NFRAG --n-classes $NCLASSES $CLASSDIST_OPTS --estimate-subst --estimate-sexch --svi-batch 2000 --svi-tau 1.0 --svi-kappa 0.7 --seed 42 --val-every 0 --jax-cache-dir /home/yam/.cache/tkf-mixdom-jax $EXTRA_OPTS"
free_mib() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}'; }
gate() { until [ $(free_mib) -ge $MINFREE ]; do sleep 120; done; }
train_to() {  # n_iter ckpt log  (retries on OOM, resuming from the checkpoint; skipped if already complete)
  if [ -f "$2" ] && [ "$(JAX_PLATFORMS=cpu .venv/bin/python -c "import numpy as np,sys; d=np.load(sys.argv[1],allow_pickle=True); print(int(d['em_iter']) if 'em_iter' in d.files else 0)" $2 2>/dev/null)" -ge "$1" ]; then
    echo "$(date -Is) $P skip train $TAG to $1 (checkpoint already there)"; FRESH=""; return; fi
  # a partially trained stage checkpoint must be resumed, never restarted
  [ -f "$2" ] && FRESH=""
  for try in 1 2 3 4; do gate
    .venv/bin/python -u train_pfam.py $COMMON --n-iter $1 --checkpoint $2 ${FRESH} > $3 2>&1; rc=$?
    FRESH=""; grep -q -i -E 'RESOURCE_EXHAUSTED|Out of memory|OUT_OF_MEMORY|scalar|fallback|dom_W|CUDA error|Failed to launch|INTERNAL:' $3 && { echo "$(date -Is) $P retry train $TAG to $1 (OOM/scalar fallback, try $try)"; sleep 900; continue; }; break; done
  echo "$(date -Is) $P end train $TAG to $1 rc=$rc"
  [ -f $2 ] || { echo "$(date -Is) $P ABORT $TAG: no checkpoint $2"; exit 1; }
}
score() {  # tag ckpt [balibase]   (idempotent: each sub-step is skipped if its result already exists)
  echo "$(date -Is) $P start score $1"
  for try in 1 2 3; do grep -q PATH-CONDITIONED experiments/ordsum_val/pathcond_$1.log 2>/dev/null && break; gate
    .venv/bin/python -u experiments/em_vs_gradient/score_pathcond_val.py --checkpoint $2 --out experiments/ordsum_val/pathcond_$1.json > experiments/ordsum_val/pathcond_$1.log 2>&1
    grep -q PATH-CONDITIONED experiments/ordsum_val/pathcond_$1.log && break; sleep 300; done
  for try in 1 2 3; do grep -q 'LL/pair  =' experiments/ordsum_val/score_$1.log 2>/dev/null && break; gate
    MODEL=d3f1 CKPT=$PWD/$2 .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > experiments/ordsum_val/score_$1.log 2>&1
    grep -q 'LL/pair  =' experiments/ordsum_val/score_$1.log && break; sleep 300; done
  if [ "$3" = balibase ] && [ ! -f experiments/expected_balibase/expected_balibase_mixdom_$1_withsps.json ]; then gate
    .venv/bin/python -u experiments/expected_pairwise_balibase.py --method mixdom --params $2 --method-name mixdom_$1 --fsa on --fsa-sps \
       --out experiments/expected_balibase/expected_balibase_mixdom_$1_withsps.json > experiments/expected_balibase/expected_balibase_mixdom_$1_withsps.log 2>&1
  fi
  rc=$?; echo "$(date -Is) $P end score $1 rc=$rc"
}
( until grep -q -E "$MARK" $LOG 2>/dev/null; do sleep 120; done
  [ -f $CORPUS/manifest.json ] || { echo "$(date -Is) $P ABORT no corpus $CORPUS"; exit 1; }
  echo "$(date -Is) $P start train $TAG (3 stages)"
  if [ -n "$START540" ]; then [ -f ${B}_540.npz ] || cp $START540 ${B}_540.npz; FRESH=""; else FRESH="--fresh"; fi
  train_to 540 ${B}_540.npz ${B}_540.log
  [ -n "$SKIP540" ] || score ${TAG}_540 ${B}_540.npz balibase
  [ -f ${B}_1080.npz ] || cp ${B}_540.npz ${B}_1080.npz; train_to 1080 ${B}_1080.npz ${B}_1080.log
  [ -n "$SKIP1080" ] || score ${TAG}_1080 ${B}_1080.npz
  [ -f ${B}_1620.npz ] || cp ${B}_1080.npz ${B}_1620.npz; train_to 1620 ${B}_1620.npz ${B}_1620.log
  score ${TAG}_1620 ${B}_1620.npz balibase
  if [ "$TAILS" != 1 ]; then echo "$(date -Is) $P end nobali-chain $TAG (EM stages only)"; exit 0; fi
  JAX_PLATFORMS=cpu .venv/bin/python $CV to-maraschino ${B}_1620.npz ${B}_1620_init.npz > /dev/null 2>&1
  if [ ! -f pfam/adam_${TAG}_from1620.npz ]; then
  gate; echo "$(date -Is) $P start adam-switch $TAG from 1620"
  .venv/bin/python -u train_pfam.py --adam --precompiled-pairs $CORPUS --n-dom $N --n-frag $NFRAG --adam-batch 2000 --adam-steps 2000 --adam-lr 0.01 --seed 42 \
     --init-from-maraschino ${B}_1620_init.npz --checkpoint pfam/adam_${TAG}_from1620.npz --fresh > pfam/adam_${TAG}_from1620.log 2>&1
  rc=$?; echo "$(date -Is) $P end adam-switch $TAG rc=$rc"; fi
  JAX_PLATFORMS=cpu .venv/bin/python $CV to-scoring pfam/adam_${TAG}_from1620.npz ${B}_1620.npz pfam/adam_${TAG}_from1620_scoring.npz > /dev/null 2>&1
  score ${TAG}_adam_from1620 pfam/adam_${TAG}_from1620_scoring.npz balibase
  if [ ! -f pfam/lbfgs_${TAG}_from1620.npz ]; then
  gate; echo "$(date -Is) $P start lbfgs-refine $TAG from 1620"
  .venv/bin/python -u experiments/em_vs_gradient/mixdom_lbfgs_refine.py --init ${B}_1620_init.npz --template ${B}_1620.npz --precompiled $CORPUS --n-sub 20000 --maxfun 80 \
     --out pfam/lbfgs_${TAG}_from1620.npz > pfam/lbfgs_${TAG}_from1620.log 2>&1
  rc=$?; echo "$(date -Is) $P end lbfgs-refine $TAG rc=$rc"; fi
  score ${TAG}_lbfgs_from1620 pfam/lbfgs_${TAG}_from1620.npz balibase
  echo "$(date -Is) $P end nobali-chain $TAG" ) >> $LOG 2>&1 &
echo "armed $TAG gpu=$GPU after /$MARK/"
