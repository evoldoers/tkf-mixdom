#!/bin/bash
# Worker pool for the SABmark / OXBench model evaluations (replaces the sequential tails of
# launch_benchmarks.sh and launch_benchmarks_rest.sh; the machine is CPU-bound on these, not
# GPU-bound, so several run side by side). Each worker walks the ordered job list, atomically
# claims a job (mkdir), skips jobs whose output exists or whose input checkpoint is not finished
# yet, and sleeps between passes until every output exists. Workers alternate GPUs.
# Usage: launch_benchmarks_pool.sh [n_workers]
cd "$(dirname "$0")/../.."
NW=${1:-4}; MINFREE=${MINFREE:-6000}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=$PWD/experiments/em_vs_gradient/results_rest_queue.log; OUT=$PWD/experiments/benchmarks; BENCH=$HOME/bio-datasets/data/bench1.0
PY=$PWD/.venv/bin/python; TKFDP=/home/yam/tkf-dp; CLAIM=$OUT/.claims; mkdir -p $CLAIM
# job list: "tag|params" (mixdom) or "c20:indel" (published-LG TKF92/MixFrag), sabrem before oxm
MODELS=(
  "mixdom_d1f1_nobali_1620|pfam/svibw_d1f1_nobali_1620.npz"
  "mixdom_d2f1_nobali_1620|pfam/svibw_d2f1_nobali_1620.npz"
  "mixdom_d3f1_nobali_1620|pfam/svibw_d3f1_nobali_1620.npz"
  "mixdom_d3f1_v4_fix|pfam/svibw_d3f1_perdomain_v4_fix.npz"
  "c20:tkf92" "c20:mixfrag"
  "mixdom_d4f1_nobali_1620|pfam/svibw_d4f1_nobali_1620.npz"
  "mixdom_d5f1_nobali_1620|pfam/svibw_d5f1_nobali_1620.npz"
  "mixdom_d3f1_full_1620|pfam/svibw_d3f1_full_1620.npz"
  "mixdom_d3f1_full_adam|pfam/adam_d3f1_full_from1620_scoring.npz"
  "mixdom_d1f1_nobali_adam|pfam/adam_d1f1_nobali_from1620_scoring.npz"
  "mixdom_d2f1_nobali_adam|pfam/adam_d2f1_nobali_from1620_scoring.npz"
  "mixdom_d3f1_nobali_adam|pfam/adam_d3f1_nobali_from1620_scoring.npz"
  "mixdom_d4f1_nobali_adam|pfam/adam_d4f1_nobali_from1620_scoring.npz"
  "mixdom_d5f1_nobali_adam|pfam/adam_d5f1_nobali_from1620_scoring.npz"
  "mixdom_d1f2_nobali_1620|pfam/svibw_d1f2_nobali_1620.npz"
  "mixdom_d2f2_nobali_1620|pfam/svibw_d2f2_nobali_1620.npz"
  "mixdom_d2f1_full_1620|pfam/svibw_d2f1_full_1620.npz"
  "mixdom_d2f1_full_adam|pfam/adam_d2f1_full_from1620_scoring.npz"
  "mixdom_d2f1_tied_nobali_1620|pfam/svibw_d2f1_tied_nobali_1620.npz"
  "mixdom_d2f1_tiedindel_nobali_1620|pfam/svibw_d2f1_tiedindel_nobali_1620.npz"
)
ready() {  # params file finished?
  local p=$1; [ -f "$p" ] || return 1
  case $p in pfam/svibw_*) grep -q 'SVI Baum-Welch complete' ${p%.npz}.log 2>/dev/null ;;
             *) [ $(( $(date +%s) - $(stat -c %Y "$p") )) -gt 120 ] ;; esac
}
outfile() { local b=$1 j=$2; case $j in c20:*) echo $TKFDP/results/fsa_c20_balibase/${j#c20:}_LGp_one_$b.json;; *) echo $OUT/${b}_${j%%|*}.json;; esac; }
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $1 | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
run_job() {  # gpu bench job
  local G=$1 b=$2 j=$3; export CUDA_VISIBLE_DEVICES=$G
  gate $G
  if [[ $j == c20:* ]]; then local m=${j#c20:}
    echo "$(date -Is) gpu${G}BP start bench $b ${m}_LGp"
    ( cd $TKFDP && PYTHONPATH=$TKFDP/src:$TKFDP JAX_CACHE_DIR=$HOME/.cache/jax-fsa-c20 $PY experiments/fsa_c20_balibase.py --indel $m --subst LGp --clock one --balibase-dir $BENCH/$b --out $(outfile $b $j) > $OUT/${b}_${m}_LGp.log 2>&1 )
    echo "$(date -Is) gpu${G}BP end bench $b ${m}_LGp rc=$?"
  else local t=${j%%|*} p=${j#*|}
    echo "$(date -Is) gpu${G}BP start bench $b $t"
    $PY -u experiments/expected_pairwise_balibase.py --method mixdom --params $p --method-name ${t}__$b --fsa on --fsa-sps --balibase-dir $BENCH/$b --out $(outfile $b $j) > $OUT/${b}_$t.log 2>&1
    echo "$(date -Is) gpu${G}BP end bench $b $t rc=$?"
  fi
}
worker() {
  local W=$1 G=$(( $1 % 2 ))
  while :; do
    local did=0 missing=0
    for b in sabrem oxm; do for j in "${MODELS[@]}"; do
      local o=$(outfile $b $j); [ -f $o ] && continue; missing=1
      [[ $j == c20:* ]] || ready "${j#*|}" || continue
      local c=$CLAIM/${b}_${j%%|*}; c=${c//:/_}
      mkdir $c 2>/dev/null || continue
      # machine-wide concurrency cap (file $CLAIM/maxconc, default 3): the training/scoring chains have priority.
      # The count-and-launch is serialised through a lock, and the lock is held until the new process is visible
      # to pgrep, so several workers cannot pass the check together.
      until mkdir $CLAIM/.lock 2>/dev/null; do sleep 7; done
      until [ $(pgrep -fc 'expected_pairwise_balibase.py|fsa_c20_balibase.py') -lt $(cat $CLAIM/maxconc 2>/dev/null || echo 3) ]; do rmdir $CLAIM/.lock; sleep 300; until mkdir $CLAIM/.lock 2>/dev/null; do sleep 7; done; done
      run_job $G $b "$j" & local pid=$!; sleep 45; rmdir $CLAIM/.lock; wait $pid; did=1
      [ -f $o ] || rmdir $c   # failed: release the claim so a later pass retries it
    done; done
    [ $missing = 0 ] && { echo "$(date -Is) BP worker $W: all benchmark outputs present"; return; }
    [ $did = 0 ] && sleep 600
  done
}
for ((w=0; w<NW; w++)); do ( worker $w ) >> $LOG 2>&1 & sleep 2; done
echo "armed $NW workers"
