#!/bin/bash
# The fourth cell of the family-level 2x2: K=20 per-family SUBSTITUTION matrices with ONE indel process shared by
# every component and held at TKF92's Pfam values (fit_tkf92_mixture.py --tie-indel; the mirror of the
# --tie-subst + one-GTR control). Fit on CPU (minutes), then held-out likelihoods component by component through
# the one scorer (directory k20tiedindel, tag tkf92_K20tiedindel; combine_k20.py), then the three benchmarks
# through the harness (--method tkf92_mixture, per-component Qs as for the full K=20 row), the latter only after
# the TKF92 re-timing has released GPU 0. Usage: <gpu-for-likelihoods> <gpu-for-benchmarks>
cd "$(dirname "$0")/../.."
GPU=${1:-1}; BGPU=${2:-0}; MINFREE=${MINFREE:-3000}; T=experiments/test_ll; D=$T/k20tiedindel
IND=pfam/tkf92_mixture_K20_train_tiedindel.npz; OUT=experiments/benchmarks; BENCH=$HOME/bio-datasets/data/bench1.0
VAL=$PWD/pfam/precompiled_v3_val; TEST=$PWD/pfam/precompiled_v3_test
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() {  # gpu: wait for free memory, then take one of two GPU slots (an flock held on fd 9 until release)
  local g=$1
  while true; do until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $g | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done
    for i in 1 2; do exec 9>/tmp/tkf_gpu${g}_slot$i.lock; flock -n 9 && return; done; exec 9>&-; sleep 60; done; }
release() { flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null; }
run() {  # logfile pathcond split comp
  local out=$1 pc=$2 split=$3 k=$4; grep -q 'LL/pair  =' $out 2>/dev/null && return
  local dir=$VAL; [ $split = test ] && dir=$TEST
  for try in 1 2 3 4; do gate $GPU; echo "$(date -Is) gpu${GPU}MI start $out (try $try)"
    JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=$GPU SUBST_NPZ=$D/comp_$k.npz TKF92_PARAMS=$D/comp_$k.json PERPAIR_OUT=$D/${split}_pc${pc}_comp${k}.npz PATHCOND=$pc MODEL=tkf92 VAL_DIR=$dir .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > $out 2>&1; release
    echo "$(date -Is) gpu${GPU}MI end $out rc=$?"; grep -q 'LL/pair  =' $out 2>/dev/null && return; sleep 600; done; }
bench() {  # bench
  local b=$1 name=tkf92_K20tiedindel; local out=$OUT/${b}_$name.json; local dir=$BENCH/$b; local mn=${name}__$b
  if [ $b = bali3pdbm ]; then out=experiments/expected_balibase/expected_balibase_${name}_withsps.json; dir=$HOME/bio-datasets/data/balibase/bali3pdbm; mn=$name; fi
  [ -f $out ] && return; gate $BGPU; echo "$(date -Is) gpu${BGPU}MI start bench $b $name"
  JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=$BGPU .venv/bin/python -u experiments/expected_pairwise_balibase.py --method tkf92_mixture --method-name $mn --params $IND --anchor experiments/tkf92_anchor_mixtures.json --pair-chunk 32 --fsa on --fsa-sps --balibase-dir $dir --out $out > ${out%.json}.log 2>&1; release
  echo "$(date -Is) gpu${BGPU}MI end bench $b $name rc=$?"; }
# 1. fit (CPU; the mixture EM runs on aggregated tau-binned counts)
( if [ ! -f $IND ]; then echo "$(date -Is) cpuMI start tied-indel fit"
    JAX_PLATFORMS=cpu .venv/bin/python -u fit_tkf92_mixture.py --K 20 --split-file ~/bio-datasets/data/pfam/seed/splits/v1.json --split train \
      --tie-indel --base-lam 0.0443 --base-mu 0.0452 --base-r 0.677 --out $IND > pfam/tkf92_mixture_K20_train_tiedindel.log 2>&1
    echo "$(date -Is) cpuMI end tied-indel fit rc=$?"; fi
  [ -f $IND ] || { echo "$(date -Is) MI ABORT fit produced no checkpoint"; exit 1; }
  .venv/bin/python experiments/test_ll/k20_components.py $IND k20tiedindel > $D.export.log 2>&1 || { echo "$(date -Is) MI ABORT export"; exit 1; }
  # 2. benchmarks in the background on BGPU, once the re-timing has released it
  ( until grep -q 'end unal pfam tkf92 (re-timing)' $LOG; do sleep 300; done
    echo "$(date -Is) gpu${BGPU}MI start tied-indel bench chain"
    for b in bali3pdbm sabrem oxm; do bench $b; done
    echo "$(date -Is) gpu${BGPU}MI end tied-indel bench chain" ) &
  # 3. likelihoods on GPU (gated; queues behind the restarts)
  echo "$(date -Is) gpu${GPU}MI start tied-indel chain"
  for split in val test; do for pc in 1 0; do [ $split = test ] && [ $pc = 1 ] && continue
    for k in $(seq 0 19); do run $D/${split}_pc${pc}_comp${k}.log $pc $split $k; done; done; done
  .venv/bin/python experiments/test_ll/combine_k20.py k20tiedindel tkf92_K20tiedindel > $D/combine.log 2>&1
  echo "$(date -Is) gpu${GPU}MI end tied-indel chain"; wait ) >> $LOG 2>&1 &
