#!/bin/bash
# MixDom d3f1 full-batch EM vs Adam replicates from different random starts
# (seeds 1-4), on the fixed 1000-pair subset, GPU 1, after the Pfam MixFrag
# unaligned job there has finished.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
MD=experiments/em_vs_gradient/results_mixdom; LOG=experiments/em_vs_gradient/results_rest_queue.log
COMMON="--precompiled-pairs pfam/precompiled_sub1000_seed0 --n-dom 3 --n-frag 1 --val-every 0 --fresh"
( until [ $(grep -c 'end unal pfam mixfrag' $LOG 2>/dev/null) -ge 2 ]; do sleep 120; done
  until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 1 | awk -F', ' '{print $2-$1}') -ge 4000 ]; do sleep 120; done
  for s in 1 2 3 4; do
    INIT=$MD/init_d3f1_seed$s.npz
    [ -f $INIT ] || JAX_PLATFORMS=cpu .venv/bin/python experiments/em_vs_gradient/make_mixdom_init.py $s $INIT > /dev/null 2>&1
    echo "$(date -Is) gpu1P start mixdom_em seed$s"
    .venv/bin/python -u train_pfam.py --svi-bw $COMMON --svi-batch 1000 --svi-kappa 0 --svi-tau 1 --n-iter 1500 \
       --init-from-maraschino $INIT --pi-pseudo 0 --S-pseudo 0 --seed $s --checkpoint $MD/em_d3f1_seed$s.npz > $MD/em_d3f1_seed$s.log 2>&1
    echo "$(date -Is) gpu1P end mixdom_em seed$s rc=$?"
    for lr in 0.1 0.01; do
      echo "$(date -Is) gpu1P start mixdom_adam seed$s lr$lr"
      .venv/bin/python -u train_pfam.py --adam $COMMON --adam-batch 1000 --adam-steps 1500 --adam-lr $lr \
         --init-from-maraschino $INIT --seed $s --checkpoint $MD/adam_d3f1_seed${s}_lr$lr.npz > $MD/adam_d3f1_seed${s}_lr$lr.log 2>&1
      echo "$(date -Is) gpu1P end mixdom_adam seed$s lr$lr rc=$?"
    done
  done ) >> $LOG 2>&1 &
echo armed
