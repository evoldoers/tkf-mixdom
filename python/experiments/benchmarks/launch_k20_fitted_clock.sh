#!/bin/bash
# Weekend queue: re-time the four family-level K=20 mixture rows of paper 1's Table 6 with TKF92's fitted Pfam
# parameters as the anchor clock (experiments/tkf92_anchor_fitted.json) instead of the LG-era anchor
# (experiments/tkf92_anchor_mixtures.json). Each row's own per-component parameters are unchanged; only the per-pair
# branch length that the harness sets with the anchor TKF92 moves. Outputs keep the paper's names
# (expected_balibase_<name>_withsps.json, <bench>_<name>.json, caches <name>[__<bench>], *_progressive_c0_r0.json);
# the old ones are archived as *_lgclock*. Starts only when no path-conditioned test chain is running (the tied-variant
# test cells come first) and gates on 6 GB free. Usage: <gpu> <row ...>   rows: K20 K20tied K20tiedgtr K20tiedindel
cd "$(dirname "$0")/../.."
GPU=$1; shift; ROWS=${@:-K20 K20tied}; MINFREE=${MINFREE:-6000}; OUT=experiments/benchmarks; EB=experiments/expected_balibase
BENCH=$HOME/bio-datasets/data/bench1.0; CACHE=$HOME/.cache/tkf-mixdom-balibase; ANCHOR=experiments/tkf92_anchor_fitted.json
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
LOG=experiments/em_vs_gradient/results_rest_queue.log; P=gpu${GPU}KC
[ -f $ANCHOR ] || echo '{"ins_rate": 0.04428, "del_rate": 0.04521, "ext_rate": 0.677, "note": "TKF92 fitted Pfam parameters (experiments/test_ll/tkf92_pfam_params.json) as the anchor clock for the K=20 mixture rows (2026-11-13)"}' > $ANCHOR
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
args() { case ${1#tkf92_} in
  K20)         echo "--params pfam/tkf92_mixture_K20_train.npz";;
  K20tied)     echo "--params pfam/tkf92_mixture_K20_train_tiedsubst.npz --sitemix pfam/cherryml_mixture_C20_paircol.npz";;
  K20tiedgtr)  echo "--params pfam/tkf92_mixture_K20_train_tiedsubst_gtr.npz";;
  K20tiedindel) echo "--params pfam/tkf92_mixture_K20_train_tiedindel.npz";; esac; }
bdir() { case $1 in bali3pdbm) echo $HOME/bio-datasets/data/balibase/bali3pdbm;; *) echo $BENCH/$1;; esac; }
paths() {  # bench name -> OUTJSON MN DEC
  local b=$1 n=$2
  if [ $b = bali3pdbm ]; then OUTJ=$EB/expected_balibase_${n}_withsps.json; MN=$n; DEC=$EB/msa_${n}_progressive_c0_r0
  else OUTJ=$OUT/${b}_$n.json; MN=${n}__$b; DEC=$OUT/${b}_${n}_progressive_c0_r0; fi; }
archive() {  # bench name: move the LG-clock outputs aside (idempotent)
  local b=$1 n=$2; paths $b $n
  [ -d $CACHE/$MN ] && [ ! -d $CACHE/${MN}_lgclock ] && mv $CACHE/$MN $CACHE/${MN}_lgclock
  for e in json log; do [ -f ${OUTJ%.json}.$e ] && [ ! -f ${OUTJ%.json}_lgclock.$e ] && mv ${OUTJ%.json}.$e ${OUTJ%.json}_lgclock.$e; done
  for e in json log; do [ -f $DEC.$e ] && [ ! -f ${DEC}_lgclock.$e ] && mv $DEC.$e ${DEC}_lgclock.$e; done; }
run() {  # bench name
  local b=$1 n=$2; paths $b $n; archive $b $n
  for try in 1 2 3; do [ -f $OUTJ ] && break; gate; echo "$(date -Is) $P start bench $b $n (fitted anchor clock, try $try)"
    .venv/bin/python -u experiments/expected_pairwise_balibase.py --method tkf92_mixture --method-name $MN $(args $n) --anchor $ANCHOR --pair-chunk 16 --fsa on --fsa-sps --balibase-dir $(bdir $b) --out $OUTJ > ${OUTJ%.json}.log 2>&1
    echo "$(date -Is) $P end bench $b $n rc=$?"; [ -f $OUTJ ] || sleep 600; done
  [ -f $OUTJ ] || { echo "$(date -Is) $P FAILED bench $b $n"; return; }
  [ -s $DEC.json ] && return
  JAX_PLATFORMS=cpu .venv/bin/python experiments/fsa_gap_factor_from_cache.py --method $MN --balibase-dir $(bdir $b) --out $DEC.json --algo progressive --gap-factors 0 --consistency-iters 0 --refine-iters 0 > $DEC.log 2>&1
  echo "$(date -Is) $P decode $b $n rc=$? $(python3 -c "import json;print(json.load(open('$DEC.json'))['summary'])" 2>/dev/null | cut -c1-100)"; }
( echo "$(date -Is) $P queued K=20 fitted-clock reruns ($ROWS); waiting for the test chains to finish"
  while pgrep -f "launch_pathcond_test.sh" > /dev/null; do sleep 600; done
  for n in $ROWS; do for b in bali3pdbm sabrem oxm; do run $b tkf92_$n; done; done
  echo "$(date -Is) $P end K=20 fitted-clock chain ($ROWS)" ) >> $LOG 2>&1 &
