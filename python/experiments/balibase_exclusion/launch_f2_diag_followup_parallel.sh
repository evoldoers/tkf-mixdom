#!/bin/bash
# Parallel replacement for the follow-up half of launch_f2_diag_refit.sh: as soon as the 1,620-iteration d<N>f2dg
# checkpoint is complete, run the test-split path-conditioned likelihood and the SABmark / OXBench posteriors side by
# side (each on whichever card has the most free memory, memory-gated, starts staggered), alongside the training
# launcher's own validation + BAliBASE scoring; then progressive decoding of all three benchmarks.
# Usage: launch_f2_diag_followup_parallel.sh <n_dom>   (env SKIP_TEST=1 / SKIP_SABREM=1: those jobs are handled elsewhere)
cd "$(dirname "$0")/../.."
N=$1; TAG=d${N}f2dg_nobali; P=f2dgP$N; CK=pfam/svibw_${TAG}_1620.npz
LOG=experiments/em_vs_gradient/results_rest_queue.log; EB=experiments/expected_balibase; BM=experiments/benchmarks
BENCH=$HOME/bio-datasets/data/bench1.0; BALI=$HOME/bio-datasets/data/balibase/bali3pdbm
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
pick() {  # minfree -> prints the card with the most free memory once one has >= minfree MiB
  while :; do read g f < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $1, $3-$2}' | sort -k2 -n -r | head -1)
    [ $f -ge $1 ] && { echo $g; return; }; sleep 120; done; }
testll() {
  local T=experiments/test_ll/pathcond_${TAG}_1620.log
  for try in 1 2 3; do grep -q 'LL/pair  =' $T 2>/dev/null && break; local g=$(pick 2600); echo "$(date -Is) $P start test pathcond $TAG gpu$g (try $try)"
    CUDA_VISIBLE_DEVICES=$g MODEL=d3f1 CKPT=$PWD/$CK PATHCOND=1 VAL_DIR=$PWD/pfam/precompiled_v3_test .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > $T 2>&1
    echo "$(date -Is) $P end test pathcond $TAG rc=$?"; grep -q 'LL/pair  =' $T || sleep 600; done; }
bench() {  # b
  local b=$1
  local o=$BM/${b}_mixdom_${TAG}_1620.json   # separate local: ${b} is empty within the same local statement
  for try in 1 2 3; do [ -f $o ] && break; local g=$(pick 6000); echo "$(date -Is) $P start bench $b mixdom_${TAG}_1620 gpu$g (try $try)"
    CUDA_VISIBLE_DEVICES=$g .venv/bin/python -u experiments/expected_pairwise_balibase.py --method mixdom --params $CK --method-name mixdom_${TAG}_1620__$b --fsa on --fsa-sps --balibase-dir $BENCH/$b --out $o > ${o%.json}.log 2>&1
    echo "$(date -Is) $P end bench $b mixdom_${TAG}_1620 rc=$?"; [ -f $o ] || sleep 600; done
  decode mixdom_${TAG}_1620__$b $BENCH/$b $BM/${b}_mixdom_${TAG}_1620_progressive_c0_r0; }
decode() {  # cache-method dir out
  [ -s $3.json ] && return
  JAX_PLATFORMS=cpu .venv/bin/python experiments/fsa_gap_factor_from_cache.py --method $1 --balibase-dir $2 --out $3.json --algo progressive --gap-factors 0 --consistency-iters 0 --refine-iters 0 > $3.log 2>&1
  echo "$(date -Is) $P decode $1 rc=$? $(python3 -c "import json;print(json.load(open('$3.json'))['summary'])" 2>/dev/null | cut -c1-100)"; }
( until grep -q 'SVI Baum-Welch complete' pfam/svibw_${TAG}_1620.log 2>/dev/null && grep -q "end train $TAG to 1620" $LOG; do sleep 60; done
  echo "$(date -Is) $P start parallel follow-up $TAG (test LL, SABmark, OXBench)"
  [ -n "$SKIP_TEST" ] || { sleep 240; testll & }; [ -n "$SKIP_SABREM" ] || { sleep 240; bench sabrem & }; sleep 240; bench oxm &
  ( until grep -q "end nobali-chain $TAG" $LOG; do sleep 120; done; decode mixdom_${TAG}_1620 $BALI $EB/msa_mixdom_${TAG}_1620_progressive_c0_r0 ) &
  wait; echo "$(date -Is) $P end f2-diag refit chain $TAG (parallel follow-up)" ) >> $LOG 2>&1 &
echo "armed parallel follow-up $TAG"
