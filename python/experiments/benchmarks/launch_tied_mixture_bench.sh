#!/bin/bash
# Benchmarks (posterior-cached) for the mixture-level class-tied control: K=20 TKF92 indel classes (tie-subst fit)
# with the C=20 site mixture (cherryml_mixture_C20_paircol.npz, fitted on the same training pairs) as the shared emission model, on BAliBASE,
# SABmark, OXBench through the harness (--method tkf92_mixture --sitemix). Waits for both fits. Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-1}; MINFREE=${MINFREE:-3000}; OUT=experiments/benchmarks; BENCH=$HOME/bio-datasets/data/bench1.0
IND=pfam/tkf92_mixture_K20_train_tiedsubst.npz; EMIT=pfam/cherryml_mixture_C20_paircol.npz
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() {  # wait for free memory, then take one of two GPU slots (an flock held on fd 9 until release)
  while true; do until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done
    for i in 1 2; do exec 9>/tmp/tkf_gpu${GPU}_slot$i.lock; flock -n 9 && return; done; exec 9>&-; sleep 60; done; }
release() { flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null; }
run() {  # bench
  local b=$1 name=tkf92_K20tied; local out=$OUT/${b}_$name.json; local dir=$BENCH/$b; local mn=${name}__$b
  if [ $b = bali3pdbm ]; then out=experiments/expected_balibase/expected_balibase_${name}_withsps.json; dir=$HOME/bio-datasets/data/balibase/bali3pdbm; mn=$name; fi
  [ -f $out ] && return; gate; echo "$(date -Is) gpu${GPU}MB start bench $b $name"
  .venv/bin/python -u experiments/expected_pairwise_balibase.py --method tkf92_mixture --method-name $mn --params $IND --anchor experiments/tkf92_anchor_mixtures.json --sitemix $EMIT --pair-chunk 32 --fsa on --fsa-sps --balibase-dir $dir --out $out > ${out%.json}.log 2>&1; release
  echo "$(date -Is) gpu${GPU}MB end bench $b $name rc=$?"; }
( until [ -f $IND ] && [ -f $EMIT ]; do sleep 300; done
  for b in bali3pdbm sabrem oxm; do run $b; done
  echo "$(date -Is) gpu${GPU}MB end tied-mixture bench chain" ) >> $LOG 2>&1 &
