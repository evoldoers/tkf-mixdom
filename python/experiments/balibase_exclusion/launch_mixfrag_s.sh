#!/bin/bash
# MixFrag-S (MixFrag with one substitution matrix per fragtype) under the released recipe: 1D path-conditioned
# svi-BW on the BAliBASE-free split, staged 540 -> 1080 -> 1620 iterations with held-out scoring at each stage
# (val path-conditioned, val order-summed, test path-conditioned; scorer MODEL=mixfrag_s), then the three benchmarks
# through profile_mixture/experiments/fsa_c20_balibase.py (--indel mixfrag_s, posterior caches {TAG}_1620__<bench>, rows in
# results/fsa_c20_balibase/cache_runs/{TAG}_1620_<bench>.jsonl). Progressive and other decodings are filled from the
# caches by the decoding scripts. Usage: launch_mixfrag_s.sh <F> <gpu> [wait-marker-regex]
cd "$(dirname "$0")/../.."
F=$1; GPU=$2; MARK=${3:-.}; MINFREE=${MINFREE:-4000}; SCORE_STAGES=${SCORE_STAGES:-1620}; TAG=mixfrags_F${F}_nobali; B=pfam/svibw_$TAG; P=gpu${GPU}MS
CORPUS=${CORPUS:-pfam/precompiled_v3_train_nobali}; PM=$PWD/experiments/profile_mixture; C20=$PM/results/fsa_c20_balibase/cache_runs
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
VAL=$PWD/pfam/precompiled_v3_val; TEST=$PWD/pfam/precompiled_v3_test
free_mib() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}'; }
gate() { until [ $(free_mib) -ge $MINFREE ]; do sleep 120; done; }
ckpt_iter() { JAX_PLATFORMS=cpu .venv/bin/python -c "import numpy as np,sys; d=np.load(sys.argv[1],allow_pickle=True); print(int(d['em_iter']) if 'em_iter' in d.files else 0)" $1 2>/dev/null || echo 0; }
train_to() {  # n_iter ckpt  (resumes a partial stage; skipped when complete; retries on OOM)
  if [ -f "$2" ] && [ "$(ckpt_iter $2)" -ge "$1" ]; then echo "$(date -Is) $P skip train $TAG to $1"; return; fi
  local RES=""; [ -f "$2" ] && RES="--resume"
  for try in 1 2 3 4; do gate; echo "$(date -Is) $P start train $TAG to $1 (try $try)"
    .venv/bin/python -u train_mixfrag_s.py --precompiled-pairs $CORPUS --n-fragtypes $F --dp-mode path --n-iter $1 --svi-batch 2000 --svi-tau 1.0 --svi-kappa 0.7 --seed 42 \
      --init-subst-npz pfam/gtr_refit_Qpi.npz --init-mixfrag-json experiments/mixfrag_F/mixfrag_F${F}_logspace.json \
      --checkpoint $2 --ckpt-every 10 --log-every 1 --log-file ${2%.npz}.log --jax-cache-dir $HOME/.cache/jax_mixfrag_s $RES >> ${2%.npz}.stdout 2>&1; rc=$?
    echo "$(date -Is) $P end train $TAG to $1 rc=$rc"
    [ "$(ckpt_iter $2)" -ge "$1" ] && return; RES="--resume"
    tail -c 4000 ${2%.npz}.stdout | grep -q -i -E 'RESOURCE_EXHAUSTED|Out of memory|OUT_OF_MEMORY|CUDA error|INTERNAL:' && { echo "$(date -Is) $P retry train $TAG to $1 (OOM, try $try)"; sleep 600; continue; }
    sleep 300; done; }
score() {  # stem ckpt : val path-cond, val order-summed, test order-summed through the one scorer
  local stem=$1 ck=$2
  for job in "experiments/ordsum_val/pathcond_${stem}.log 1 $VAL" "experiments/test_ll/pathcond_${stem}.log 1 $TEST"; do   # order-summed column dropped
    set -- $job; local out=$1 pc=$2 dir=$3
    grep -q 'LL/pair' $out 2>/dev/null && continue
    for try in 1 2 3; do gate; echo "$(date -Is) $P start score $out (try $try)"
      PATHCOND=$pc MODEL=mixfrag_s MIXFRAG_S_NPZ=$ck VAL_DIR=$dir .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > $out 2>&1
      echo "$(date -Is) $P end score $out rc=$?"; grep -q 'LL/pair' $out 2>/dev/null && break; sleep 600; done; done; }
bench() {  # stem ckpt : the three benchmarks through profile_mixture script (resumable, append-as-you-go)
  local stem=$1 ck=$PWD/$2
  for b in bali3pdbm sabrem oxm; do
    local dir=$HOME/bio-datasets/data/bench1.0/$b; [ $b = bali3pdbm ] && dir=$HOME/bio-datasets/data/balibase/bali3pdbm
    [ -f $C20/${stem}_$b.json ] && continue
    for try in 1 2 3; do gate; echo "$(date -Is) $P start bench $b $stem (try $try)"
      ( cd $PM && PYTHONPATH=$PM/src:$PM JAX_CACHE_DIR=$HOME/.cache/jax-fsa-c20 $OLDPWD/.venv/bin/python experiments/fsa_c20_balibase.py --indel mixfrag_s --mixfrag-s-npz $ck --clock one --balibase-dir $dir --out $C20/${stem}_$b.json --cache-name ${stem}__$b > $C20/${stem}_$b.log 2>&1 )
      echo "$(date -Is) $P end bench $b $stem rc=$?"; [ -f $C20/${stem}_$b.json ] && break; sleep 600; done; done; }
( until grep -q -E "$MARK" $LOG 2>/dev/null; do sleep 120; done
  echo "$(date -Is) $P start MixFrag-S chain $TAG (3 stages)"
  train_to 540 ${B}_540.npz;  echo " $SCORE_STAGES " | grep -q ' 540 ' && score ${TAG}_540 ${B}_540.npz
  [ -f ${B}_1080.npz ] || cp ${B}_540.npz ${B}_1080.npz;  train_to 1080 ${B}_1080.npz;  echo " $SCORE_STAGES " | grep -q ' 1080 ' && score ${TAG}_1080 ${B}_1080.npz
  [ -f ${B}_1620.npz ] || cp ${B}_1080.npz ${B}_1620.npz;  train_to 1620 ${B}_1620.npz;  score ${TAG}_1620 ${B}_1620.npz
  bench ${TAG}_1620 ${B}_1620.npz
  echo "$(date -Is) $P end MixFrag-S chain $TAG" ) >> $LOG 2>&1 &
echo "armed $TAG gpu=$GPU after /$MARK/"
