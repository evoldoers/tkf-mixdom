#!/bin/bash
# Retry the families the tied K=20 control's benchmark runs failed with out-of-memory (FAILED lines in each run's
# log), at a smaller pair chunk and only when the GPU is quiet, then merge them into the main result file
# (merge_retry.py). Waits for each main run to finish. Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-1}; MINFREE=${MINFREE:-6500}; NAME=${NAME:-tkf92_K20tied}; OUT=experiments/benchmarks; BENCH=$HOME/bio-datasets/data/bench1.0
IND=${IND:-pfam/tkf92_mixture_K20_train_tiedsubst.npz}; EMIT=${EMIT:-pfam/cherryml_mixture_C20_paircol.npz}; SITEMIX="--sitemix $EMIT"; [ "$EMIT" = none ] && SITEMIX=
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() {  # wait for free memory, then take one of two GPU slots (an flock held on fd 9 until release)
  while true; do until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done
    for i in 1 2; do exec 9>/tmp/tkf_gpu${GPU}_slot$i.lock; flock -n 9 && return; done; exec 9>&-; sleep 60; done; }
release() { flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null; }
retry() {  # bench
  local b=$1 main=$OUT/${b}_$NAME.json dir=$BENCH/$b mn=${NAME}__$b
  if [ $b = bali3pdbm ]; then main=experiments/expected_balibase/expected_balibase_${NAME}_withsps.json; dir=$HOME/bio-datasets/data/balibase/bali3pdbm; mn=$NAME; fi
  local rj=${main%.json}_retry.json
  until [ -f $main ]; do sleep 300; done
  [ -f $rj ] && return
  local fams=$(grep FAILED ${main%.json}.log | sed 's/^\[[^]]*\]//' | awk '{print $1}' | sort -u | paste -sd,)   # family name = first token after the [i/n] prefix
  [ -z "$fams" ] && { echo "$(date -Is) gpu${GPU}TR $b: no failed families"; return; }
  for try in 1 2 3; do gate; echo "$(date -Is) gpu${GPU}TR start retry $b $NAME ($fams) try $try"
    .venv/bin/python -u experiments/expected_pairwise_balibase.py --method tkf92_mixture --method-name $mn --params $IND --anchor experiments/tkf92_anchor_mixtures.json $SITEMIX --families $fams --pair-chunk 8 --fsa on --fsa-sps --balibase-dir $dir --out $rj > ${rj%.json}.log 2>&1; release
    echo "$(date -Is) gpu${GPU}TR end retry $b rc=$? failed-again=$(grep -c FAILED ${rj%.json}.log)"
    [ -f $rj ] && ! grep -q FAILED ${rj%.json}.log && break; rm -f $rj; sleep 600; done
  [ -f $rj ] && .venv/bin/python experiments/benchmarks/merge_retry.py $main $rj >> $LOG 2>&1; }
( for b in bali3pdbm sabrem oxm; do retry $b; done; echo "$(date -Is) gpu${GPU}TR end tied retry chain" ) >> $LOG 2>&1 &
