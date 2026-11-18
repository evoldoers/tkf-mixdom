#!/bin/bash
# Completes the SABmark / OXBench set (the first chain, launch_benchmarks.sh, does MAFFT, MUSCLE,
# the Table 4 d3f1, cleaned d1f1/d2f1/d3f1 EM 1620, TKF92 and MixFrag F=2; the MixFrag-F chain
# does MixFrag F=2..5 refits). This one adds, per benchmark: cleaned d4f1 / d5f1 (EM 1620),
# the Adam-tail points of d1f1..d5f1, the full-corpus d3f1 (EM 1620 + Adam), and d1f2 / d2f2
# (EM 1620) once their checkpoints are complete. Idempotent; memory-gated. Usage: [gpu]
cd "$(dirname "$0")/../.."
GPU=${1:-1}; MINFREE=${MINFREE:-5000}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=$PWD/experiments/em_vs_gradient/results_rest_queue.log; OUT=$PWD/experiments/benchmarks; BENCH=$HOME/bio-datasets/data/bench1.0
PY=$PWD/.venv/bin/python
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
run_model() {  # bench tag params
  local b=$1 t=$2 p=$3; local out=$OUT/${b}_$t.json
  [ -f $out ] && return
  gate; echo "$(date -Is) gpu${GPU}B2 start bench $b $t"
  $PY -u experiments/expected_pairwise_balibase.py --method mixdom --params $p --method-name ${t}__$b --fsa on --fsa-sps --balibase-dir $BENCH/$b --out $out > $OUT/${b}_$t.log 2>&1
  echo "$(date -Is) gpu${GPU}B2 end bench $b $t rc=$?"
}
wait_ckpt() { until [ -f pfam/svibw_$1.npz ] && grep -q 'SVI Baum-Welch complete' pfam/svibw_$1.log 2>/dev/null; do sleep 300; done; }
( for b in sabrem oxm; do
    run_model $b mixdom_d4f1_nobali_1620 pfam/svibw_d4f1_nobali_1620.npz
    run_model $b mixdom_d5f1_nobali_1620 pfam/svibw_d5f1_nobali_1620.npz
    for d in 1 2 3 4 5; do
      until [ -f pfam/adam_d${d}f1_nobali_from1620_scoring.npz ]; do sleep 300; done
      run_model $b mixdom_d${d}f1_nobali_adam pfam/adam_d${d}f1_nobali_from1620_scoring.npz
    done
    run_model $b mixdom_d3f1_full_1620 pfam/svibw_d3f1_full_1620.npz
    run_model $b mixdom_d3f1_full_adam pfam/adam_d3f1_full_from1620_scoring.npz
  done
  for t in d1f2_nobali d2f2_nobali; do wait_ckpt ${t}_1620; for b in sabrem oxm; do run_model $b mixdom_${t}_1620 pfam/svibw_${t}_1620.npz; done; done
  echo "$(date -Is) gpu${GPU}B2 end benchmarks-rest chain" ) >> $LOG 2>&1 &
echo armed
