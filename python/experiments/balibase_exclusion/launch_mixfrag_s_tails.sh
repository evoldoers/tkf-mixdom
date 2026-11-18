#!/bin/bash
# MixFrag-S gradient tails from the 1,620-iteration svi-BW fit, mirroring the MixDom recipe (launch_dNf1_long.sh):
# (a) Adam, 2,000 steps, minibatch 2,000, lr 0.01, seed 42 (mixfrag_s_adam_tail.py); (b) L-BFGS refine on a 20,000-pair
# breadth-first subset, 80 evaluations (mixfrag_s_lbfgs_refine.py). Each tail is then scored (val path-conditioned, val
# order-summed, test path-conditioned; MODEL=mixfrag_s) and benchmarked (the three benchmarks through tkf-dp's fsa_c20_balibase.py,
# caches {stem}__<bench>), stems mixfrags_F<F>_nobali_{adam,lbfgs}_from1620 as paper_tables.py (MFS_SUFFIXES) expects.
# Usage: launch_mixfrag_s_tails.sh <F> <gpu> [wait-marker-regex]   (waits for the marker in the queue log, then for
# pfam/svibw_mixfrags_F<F>_nobali_1620.npz at 1,620 iterations)
cd "$(dirname "$0")/../.."
F=$1; GPU=$2; MARK=${3:-.}; MINFREE=${MINFREE:-4000}; TAG=mixfrags_F${F}_nobali; B=pfam/svibw_$TAG; P=gpu${GPU}MT
TKFDP=/home/yam/tkf-dp; C20=$TKFDP/results/fsa_c20_balibase/cache_runs
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
VAL=$PWD/pfam/precompiled_v3_val; TEST=$PWD/pfam/precompiled_v3_test
free_mib() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}'; }
gate() { until [ $(free_mib) -ge $MINFREE ]; do sleep 120; done; }
ckpt_iter() { JAX_PLATFORMS=cpu .venv/bin/python -c "import numpy as np,sys; d=np.load(sys.argv[1],allow_pickle=True); print(int(d['em_iter']) if 'em_iter' in d.files else 0)" $1 2>/dev/null || echo 0; }
adam_tail() {  # resumable; retried on OOM
  local out=${B}_adam_from1620.npz
  for try in 1 2 3 4; do grep -q 'done\|complete\|saved' ${out%.npz}.log 2>/dev/null && [ -f $out ] && return; gate; echo "$(date -Is) $P start adam tail $TAG (try $try)"
    .venv/bin/python -u experiments/em_vs_gradient/mixfrag_s_adam_tail.py --init ${B}_1620.npz --precompiled pfam/precompiled_v3_train_nobali \
      --adam-steps 2000 --adam-lr 0.01 --adam-batch 2000 --seed 42 --ckpt-every 50 --resume --jax-cache-dir $HOME/.cache/jax_mixfrag_s \
      --out $out --log-file ${out%.npz}.log >> ${out%.npz}.stdout 2>&1; rc=$?; echo "$(date -Is) $P end adam tail $TAG rc=$rc"
    [ $rc -eq 0 ] && [ -f $out ] && return; sleep 600; done; }
lbfgs_refine() {
  local out=${B}_lbfgs_from1620.npz
  for try in 1 2 3; do [ -f $out ] && return; gate; echo "$(date -Is) $P start lbfgs refine $TAG (try $try)"
    .venv/bin/python -u experiments/em_vs_gradient/mixfrag_s_lbfgs_refine.py --init ${B}_1620.npz --precompiled pfam/precompiled_v3_train_nobali \
      --n-sub 20000 --maxfun 80 --seed 42 --out $out --log-file ${out%.npz}.log >> ${out%.npz}.stdout 2>&1; rc=$?; echo "$(date -Is) $P end lbfgs refine $TAG rc=$rc"
    [ $rc -eq 0 ] && [ -f $out ] && return; sleep 600; done; }
score() {  # stem ckpt
  local stem=$1 ck=$2
  for job in "experiments/ordsum_val/pathcond_${stem}.log 1 $VAL" "experiments/test_ll/pathcond_${stem}.log 1 $TEST"; do   # order-summed column dropped
    set -- $job; local out=$1 pc=$2 dir=$3
    grep -q 'LL/pair' $out 2>/dev/null && continue
    for try in 1 2 3; do gate; echo "$(date -Is) $P start score $out (try $try)"
      PATHCOND=$pc MODEL=mixfrag_s MIXFRAG_S_NPZ=$ck VAL_DIR=$dir .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > $out 2>&1
      echo "$(date -Is) $P end score $out rc=$?"; grep -q 'LL/pair' $out 2>/dev/null && break; sleep 600; done; done; }
bench() {  # stem ckpt
  local stem=$1 ck=$PWD/$2
  for b in bali3pdbm sabrem oxm; do
    local dir=$HOME/bio-datasets/data/bench1.0/$b; [ $b = bali3pdbm ] && dir=$HOME/bio-datasets/data/balibase/bali3pdbm
    [ -f $C20/${stem}_$b.json ] && continue
    for try in 1 2 3; do gate; echo "$(date -Is) $P start bench $b $stem (try $try)"
      ( cd $TKFDP && PYTHONPATH=$TKFDP/src:$TKFDP JAX_CACHE_DIR=$HOME/.cache/jax-fsa-c20 $OLDPWD/.venv/bin/python experiments/fsa_c20_balibase.py --indel mixfrag_s --mixfrag-s-npz $ck --clock one --balibase-dir $dir --out $C20/${stem}_$b.json --cache-name ${stem}__$b > $C20/${stem}_$b.log 2>&1 )
      echo "$(date -Is) $P end bench $b $stem rc=$?"; [ -f $C20/${stem}_$b.json ] && break; sleep 600; done; done; }
( until grep -q -E "$MARK" $LOG 2>/dev/null; do sleep 120; done
  until [ -f ${B}_1620.npz ] && [ "$(ckpt_iter ${B}_1620.npz)" -ge 1620 ]; do sleep 300; done
  echo "$(date -Is) $P start MixFrag-S tails $TAG"
  lbfgs_refine; score ${TAG}_lbfgs_from1620 ${B}_lbfgs_from1620.npz
  adam_tail;    score ${TAG}_adam_from1620 ${B}_adam_from1620.npz
  bench ${TAG}_lbfgs_from1620 ${B}_lbfgs_from1620.npz; bench ${TAG}_adam_from1620 ${B}_adam_from1620.npz
  echo "$(date -Is) $P end MixFrag-S tails $TAG" ) >> $LOG 2>&1 &
echo "armed tails $TAG gpu=$GPU"
