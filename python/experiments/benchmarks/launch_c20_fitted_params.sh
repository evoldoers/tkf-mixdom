#!/bin/bash
# Re-run the "TKF92, C=20 site mixture" row with TKF92's fitted Pfam indel parameters (experiments/test_ll/
# tkf92_pfam_params.json: 0.04428/0.04521/0.677) in place of the LG-era anchor set (experiments/tkf92_anchor_mixtures.json:
# 0.045806/0.046799/0.683455) that its likelihood cells and benchmark posteriors had used. Outputs keep the paper's names
# (ordsum_val/pathcond_cherryml_C20pc.log, test_ll/pathcond_cherryml_C20pc.log, cache tkf92_CML20pc_anchor__<bench>,
# cache_runs/tkf92_CML20pc_anchor_<bench>.*, msa_/<bench>_tkf92_CML20pc_anchor_progressive_c0_r0.json); the anchor-set
# runs are archived as *_anchorparams.log / *_lgparams*. Usage: <gpu> ll|bench
cd "$(dirname "$0")/../.."
GPU=$1; MODE=$2; MINFREE=${MINFREE:-3000}; T=experiments/test_ll; V=experiments/ordsum_val; BENCH=$HOME/bio-datasets/data/bench1.0; PM=$PWD/experiments/profile_mixture
NPZ=pfam/cherryml_mixture_C20_paircol.npz; PY=$PWD/.venv/bin/python; C20=$PM/results/fsa_c20_balibase; CACHE=$HOME/.cache/tkf-mixdom-balibase
EB=experiments/expected_balibase; BM=experiments/benchmarks
VAL=$PWD/pfam/precompiled_v3_val; TEST=$PWD/pfam/precompiled_v3_test
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log; P=gpu${GPU}C20
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
bdir() { case $1 in bali3pdbm) echo $HOME/bio-datasets/data/balibase/bali3pdbm;; *) echo $BENCH/$1;; esac; }
score() {  # logfile split
  local out=$1 split=$2; local dir=$VAL; [ $split = test ] && dir=$TEST
  [ -f $out ] && ! grep -q "tkf92_pfam_params" $out && [ ! -f ${out%.log}_anchorparams.log ] && mv $out ${out%.log}_anchorparams.log
  grep -q 'LL/pair  =' $out 2>/dev/null && return
  for try in 1 2 3; do gate; echo "$(date -Is) $P start $out (try $try)"
    CHERRYML_NPZ=$NPZ TKF92_PARAMS=$T/tkf92_pfam_params.json PATHCOND=1 MODEL=cherryml_C20 VAL_DIR=$dir $PY -u experiments/ordsum_val/score_matchaligned_val.py > $out 2>&1
    echo "$(date -Is) $P end $out rc=$? $(grep 'LL/pair  =' $out | tail -1)"; grep -q 'LL/pair  =' $out 2>/dev/null && return; sleep 600; done; }
archive() {  # bench: move the anchor-set cache, harness outputs and decode aside (idempotent)
  local b=$1 n=tkf92_CML20pc_anchor
  [ -d $CACHE/${n}__$b ] && [ ! -d $CACHE/${n}_lgparams__$b ] && mv $CACHE/${n}__$b $CACHE/${n}_lgparams__$b
  for e in json jsonl log; do [ -f $C20/cache_runs/${n}_$b.$e ] && [ ! -f $C20/cache_runs/${n}_lgparams_$b.$e ] && mv $C20/cache_runs/${n}_$b.$e $C20/cache_runs/${n}_lgparams_$b.$e; done
  local d=$EB/msa_${n}_progressive_c0_r0; [ $b != bali3pdbm ] && d=$BM/${b}_${n}_progressive_c0_r0
  for e in json log; do [ -f $d.$e ] && [ ! -f ${d/${n}/${n}_lgparams}.$e ] && mv $d.$e ${d/${n}/${n}_lgparams}.$e; done; }
bench() {  # bench
  local b=$1 o=cache_runs/tkf92_CML20pc_anchor_$b; archive $b; [ -f $C20/$o.json ] && return; gate
  echo "$(date -Is) $P start c20 $b $o (fitted TKF92 params)"
  ( cd $PM && CHERRYML_NPZ=$OLDPWD/$NPZ PYTHONPATH=$PM/src:$PM JAX_CACHE_DIR=$HOME/.cache/jax-fsa-c20 $PY experiments/fsa_c20_balibase.py --indel tkf92 --indel-params $OLDPWD/experiments/test_ll/tkf92_fit_lmw.json --subst CML20 --clock anchor --balibase-dir $(bdir $b) --out $C20/$o.json --cache-name tkf92_CML20pc_anchor__$b > $C20/$o.log 2>&1 )
  echo "$(date -Is) $P end c20 $b $o rc=$?"; }
decode() {  # bench: progressive MEA, no consistency / refinement (the Table-6 decoding)
  local b=$1 n=tkf92_CML20pc_anchor; local d=$EB/msa_${n}_progressive_c0_r0; [ $b != bali3pdbm ] && d=$BM/${b}_${n}_progressive_c0_r0
  [ -s $d.json ] && return
  JAX_PLATFORMS=cpu $PY experiments/fsa_gap_factor_from_cache.py --method ${n}__$b --balibase-dir $(bdir $b) --out $d.json --algo progressive --gap-factors 0 --consistency-iters 0 --refine-iters 0 > $d.log 2>&1
  echo "$(date -Is) $P decode $b rc=$? $(python3 -c "import json;print(json.load(open('$d.json'))['summary'])" 2>/dev/null | cut -c1-110)"; }
case $MODE in
  ll)    ( score $V/pathcond_cherryml_C20pc.log val; score $T/pathcond_cherryml_C20pc.log test; echo "$(date -Is) $P end C=20 fitted-params likelihood chain" ) >> $LOG 2>&1 & ;;
  bench) ( for b in bali3pdbm sabrem oxm; do bench $b; decode $b; done; echo "$(date -Is) $P end C=20 fitted-params bench chain" ) >> $LOG 2>&1 & ;;
  *) echo "usage: $0 <gpu> ll|bench"; exit 1;; esac
