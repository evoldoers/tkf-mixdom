#!/bin/bash
# Alignment benchmarks beyond BAliBASE (drive5 bench1.0 layout, in/ + ref/), smallest first:
# sabrem (SABmark SABRE subset, 303 families) then oxm (OXBench, 336). For each: MAFFT,
# MUSCLE, the Table 4 MixDom-d3f1, the cleaned-corpus d1f1/d2f1/d3f1 (EM 1620; d1f1 waits
# for its checkpoint), then TKF92 and MixFrag (published LG) via fsa_c20_balibase.py.
# Every step is idempotent (skipped when its output exists). GPU 0, memory-gated.
# Usage: launch_benchmarks.sh [gpu]
cd "$(dirname "$0")/../.."
GPU=${1:-0}; MINFREE=${MINFREE:-5000}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=$PWD/experiments/em_vs_gradient/results_rest_queue.log; OUT=$PWD/experiments/benchmarks; BENCH=$HOME/bio-datasets/data/bench1.0
PY=$PWD/.venv/bin/python; PM=$PWD/experiments/profile_mixture
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
run_model() {  # bench tag method [params]
  local b=$1 t=$2 m=$3 p=$4; local out=$OUT/${b}_$t.json
  [ -f $out ] && return
  gate; echo "$(date -Is) gpu${GPU}B start bench $b $t"
  if [ -n "$p" ]; then $PY -u experiments/expected_pairwise_balibase.py --method $m --params $p --method-name ${t}__$b --fsa on --fsa-sps --balibase-dir $BENCH/$b --out $out > $OUT/${b}_$t.log 2>&1
  else $PY -u experiments/expected_pairwise_balibase.py --method $m --method-name ${t}__$b --balibase-dir $BENCH/$b --out $out > $OUT/${b}_$t.log 2>&1; fi
  echo "$(date -Is) gpu${GPU}B end bench $b $t rc=$?"
}
run_c20() {  # bench indel
  local b=$1 m=$2; local out=$PM/results/fsa_c20_balibase/${m}_LGp_one_$b.json
  [ -f $out ] && return
  gate; echo "$(date -Is) gpu${GPU}B start bench $b ${m}_LGp"
  ( cd $PM && PYTHONPATH=$PM/src:$PM JAX_CACHE_DIR=$HOME/.cache/jax-fsa-c20 $PY experiments/fsa_c20_balibase.py --indel $m --subst LGp --clock one --balibase-dir $BENCH/$b --out $out > $OUT/${b}_${m}_LGp.log 2>&1 )
  echo "$(date -Is) gpu${GPU}B end bench $b ${m}_LGp rc=$?"
}
( for b in sabrem oxm; do
    run_model $b mafft mafft; run_model $b muscle muscle
    run_model $b mixdom_d3f1_v4_fix mixdom pfam/svibw_d3f1_perdomain_v4_fix.npz
    run_model $b mixdom_d2f1_nobali_1620 mixdom pfam/svibw_d2f1_nobali_1620.npz
    run_model $b mixdom_d3f1_nobali_1620 mixdom pfam/svibw_d3f1_nobali_1620.npz
    until [ -f pfam/svibw_d1f1_nobali_1620.npz ] && grep -q 'SVI Baum-Welch complete' pfam/svibw_d1f1_nobali_1620.log 2>/dev/null; do sleep 300; done
    run_model $b mixdom_d1f1_nobali_1620 mixdom pfam/svibw_d1f1_nobali_1620.npz
    run_c20 $b tkf92; run_c20 $b mixfrag
  done; echo "$(date -Is) gpu${GPU}B end benchmarks chain" ) >> $LOG 2>&1 &
echo armed
