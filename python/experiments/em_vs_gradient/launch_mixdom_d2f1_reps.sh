#!/bin/bash
# Full-batch EM on MixDom-d2f1 from four further random initialisations (seeds 1-4; same 1,000 aligned Pfam pairs,
# flat priors, 2,000 iterations), to say whether EM reaches the same stationary point from different starts.
# Outputs results_mixdom/em_d2f1_seed<s>.log. Gated on 5 GB free. Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-0}; MINFREE=${MINFREE:-5000}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
E=experiments/em_vs_gradient; MD=$E/results_mixdom; LOG=$E/results_rest_queue.log
COMMON="--precompiled-pairs pfam/precompiled_sub1000_seed0 --n-dom 2 --n-frag 1 --val-every 0 --fresh"
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
( for s in 1 2 3 4; do
    grep -q 'MAP_LL=' $MD/em_d2f1_seed$s.log 2>/dev/null && [ $(grep -c 'MAP_LL=' $MD/em_d2f1_seed$s.log) -ge 2000 ] && continue
    INIT=$MD/init_d2f1_seed$s.npz; [ -f $INIT ] || JAX_PLATFORMS=cpu .venv/bin/python experiments/em_vs_gradient/make_mixdom_init.py $s $INIT 2 > /dev/null 2>&1
    gate; echo "$(date -Is) gpu${GPU}MR start mixdom_em d2f1 seed$s"
    .venv/bin/python -u train_pfam.py --svi-bw $COMMON --svi-batch 1000 --svi-kappa 0 --svi-tau 1 --n-iter 2000 \
       --init-from-maraschino $INIT --pi-pseudo 0 --S-pseudo 0 --seed $s --checkpoint $MD/em_d2f1_seed$s.npz > $MD/em_d2f1_seed$s.log 2>&1
    echo "$(date -Is) gpu${GPU}MR end mixdom_em d2f1 seed$s rc=$?"
  done; echo "$(date -Is) gpu${GPU}MR end mixdom d2f1 reps chain" ) >> $LOG 2>&1 &
