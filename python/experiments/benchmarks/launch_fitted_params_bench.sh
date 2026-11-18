#!/bin/bash
# Re-run the TKF92 and MixFrag F=2 BAliBASE/SABmark/OXBench posteriors with the models' fitted GTR-era indel
# parameters (experiments/test_ll/tkf92_pfam_params.json; experiments/mixfrag_F/mixfrag_F2_logspace.json), replacing
# the runs made with the LG-era values pinned in fsa_c20_balibase.py's INDEL table (archived as *_lgparams*).
# Usage: <gpu> <model: tkf92|mixfrag> [bench ...]   (bali3pdbm is done separately by the paramcheck run)
cd "$(dirname "$0")/../.."
GPU=$1; M=$2; shift 2; BENCHES=${@:-sabrem oxm}; MINFREE=${MINFREE:-5000}; BENCH=$HOME/bio-datasets/data/bench1.0; TKFDP=/home/yam/tkf-dp
PY=$PWD/.venv/bin/python; C20=$TKFDP/results/fsa_c20_balibase; GTR=$PWD/pfam/gtr_refit_Qpi.npz; CACHE=$HOME/.cache/tkf-mixdom-balibase
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
LOG=experiments/em_vs_gradient/results_rest_queue.log; P=gpu${GPU}FP
case $M in
  tkf92)   ARGS="--indel tkf92 --indel-params $PWD/experiments/test_ll/tkf92_fit_lmw.json";;
  mixfrag) ARGS="--indel mixfrag --indel-params $PWD/experiments/mixfrag_F/mixfrag_F2_logspace.json";;
  *) echo "model?"; exit 1;; esac
[ $M = tkf92 ] && [ ! -f experiments/test_ll/tkf92_fit_lmw.json ] && echo '{"lam": 0.04428, "mu": 0.04521, "exts": [0.677], "weights": [1.0]}' > experiments/test_ll/tkf92_fit_lmw.json
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
bdir() { case $1 in bali3pdbm) echo $HOME/bio-datasets/data/balibase/bali3pdbm;; *) echo $BENCH/$1;; esac; }
archive() {  # bench: move the LG-era-parameter cache and harness outputs aside (idempotent)
  local b=$1 n=${M}_GTR_one
  [ -d $CACHE/${n}__$b ] && [ ! -d $CACHE/${M}_GTR_one_lgparams__$b ] && mv $CACHE/${n}__$b $CACHE/${M}_GTR_one_lgparams__$b
  for e in json jsonl log; do [ -f $C20/cache_runs/${n}_$b.$e ] && [ ! -f $C20/cache_runs/${M}_GTR_one_lgparams_$b.$e ] && mv $C20/cache_runs/${n}_$b.$e $C20/cache_runs/${M}_GTR_one_lgparams_$b.$e; done; }
( for b in $BENCHES; do o=cache_runs/${M}_GTR_one_$b; archive $b; [ -f $C20/$o.json ] && continue; gate
    echo "$(date -Is) $P start c20 $b $o (fitted params)"
    ( cd $TKFDP && GTR_NPZ=$GTR PYTHONPATH=$TKFDP/src:$TKFDP JAX_CACHE_DIR=$HOME/.cache/jax-fsa-c20 $PY experiments/fsa_c20_balibase.py --subst GTR --clock one --balibase-dir $(bdir $b) --out $C20/$o.json --cache-name ${M}_GTR_one__$b $ARGS > $C20/$o.log 2>&1 )
    echo "$(date -Is) $P end c20 $b $o rc=$?"; done
  echo "$(date -Is) $P end fitted-params bench chain $M" ) >> $LOG 2>&1 &
