#!/bin/bash
# Refit MixDom d1f2 / d2f2 with a diagonal fragtype-extension matrix (--freeze-ext-offdiag: a fragment keeps its fragtype;
# each diagonal entry has the Beta(2,3) prior of the f1 fits), on the BAliBASE-free split, same recipe and seed as
# launch_retrain_nobali.sh otherwise. Then every cell of the models table: path-conditioned validation and test
# log-likelihood, BAliBASE / SABmark / OXBench posteriors, and progressive MEA decoding (consistency 0, refinement 0).
# Tag d<N>f2dg_nobali (the full-matrix fits keep d<N>f2_nobali). Usage: launch_f2_diag_refit.sh <n_dom> <gpu>
cd "$(dirname "$0")/../.."
N=$1; GPU=$2; TAG=d${N}f2dg_nobali; P=gpu${GPU}N; CK=pfam/svibw_${TAG}_1620.npz
LOG=experiments/em_vs_gradient/results_rest_queue.log; EB=experiments/expected_balibase; BM=experiments/benchmarks
BENCH=$HOME/bio-datasets/data/bench1.0; BALI=$HOME/bio-datasets/data/balibase/bali3pdbm; MINFREE=${MINFREE:-6500}
NFRAG=2 TAG=$TAG EXTRA_OPTS=--freeze-ext-offdiag SKIP540=1 SKIP1080=1 TAILS=0 \
  bash experiments/balibase_exclusion/launch_retrain_nobali.sh $N $GPU '.'
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
bench() {  # b : posteriors on SABmark / OXBench
  local b=$1 o=$BM/${b}_mixdom_${TAG}_1620.json
  for try in 1 2 3; do [ -f $o ] && break; gate; echo "$(date -Is) $P start bench $b mixdom_${TAG}_1620 (try $try)"
    .venv/bin/python -u experiments/expected_pairwise_balibase.py --method mixdom --params $CK --method-name mixdom_${TAG}_1620__$b --fsa on --fsa-sps --balibase-dir $BENCH/$b --out $o > ${o%.json}.log 2>&1
    echo "$(date -Is) $P end bench $b mixdom_${TAG}_1620 rc=$?"; [ -f $o ] || sleep 600; done; }
decode() {  # cache-method dir out
  [ -s $3.json ] && return
  JAX_PLATFORMS=cpu .venv/bin/python experiments/fsa_gap_factor_from_cache.py --method $1 --balibase-dir $2 --out $3.json --algo progressive --gap-factors 0 --consistency-iters 0 --refine-iters 0 > $3.log 2>&1
  echo "$(date -Is) $P decode $1 rc=$? $(python3 -c "import json;print(json.load(open('$3.json'))['summary'])" 2>/dev/null | cut -c1-100)"; }
( until grep -q "$P end nobali-chain $TAG" $LOG 2>/dev/null; do sleep 120; done
  [ -f $CK ] || { echo "$(date -Is) $P ABORT f2-diag follow-up: no $CK"; exit 1; }
  T=experiments/test_ll/pathcond_${TAG}_1620.log
  for try in 1 2 3; do grep -q 'LL/pair  =' $T 2>/dev/null && break; gate; echo "$(date -Is) $P start test pathcond $TAG (try $try)"
    MODEL=d3f1 CKPT=$PWD/$CK PATHCOND=1 VAL_DIR=$PWD/pfam/precompiled_v3_test .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > $T 2>&1
    echo "$(date -Is) $P end test pathcond $TAG rc=$?"; grep -q 'LL/pair  =' $T || sleep 600; done
  bench sabrem & bench oxm & wait
  decode mixdom_${TAG}_1620 $BALI $EB/msa_mixdom_${TAG}_1620_progressive_c0_r0
  for b in sabrem oxm; do decode mixdom_${TAG}_1620__$b $BENCH/$b $BM/${b}_mixdom_${TAG}_1620_progressive_c0_r0; done
  echo "$(date -Is) $P end f2-diag refit chain $TAG" ) >> $LOG 2>&1 &
echo "armed f2-diag follow-up $TAG gpu=$GPU"
