#!/bin/bash
# Table 3's MixDom row on the released model (d2f1) instead of d3f1: full-batch EM (svi-BW with the whole 1,000-pair
# corpus as one batch, flat priors) for 200 then 2,000 iterations, and Adam at two learning rates for 2,000 steps,
# all from the same initial parameters, on the same 1,000 aligned Pfam pairs as the d3f1 experiment. Outputs
# results_mixdom/{em_d2f1_fixed,em_d2f1_fixed_2000,adam_d2f1_lr0.1_long,adam_d2f1_lr0.01_long}.log; then
# EMVG_MIXDOM=d2f1 convergence_figure.py. Waits for the Table 3 simulated-MixFrag one-off (same GPU, both need
# most of it) and for 9 GB free. Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-0}; MINFREE=${MINFREE:-9000}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
E=experiments/em_vs_gradient; MD=$E/results_mixdom; LOG=$E/results_rest_queue.log
COMMON="--precompiled-pairs pfam/precompiled_sub1000_seed0 --n-dom 2 --n-frag 1 --val-every 0"
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
( until grep -q '^saved ' $E/results_gtr/sim_mixfrag_mixfrag.log 2>/dev/null; do sleep 300; done
  INIT=$MD/init_d2f1_seed0.npz
  [ -f $INIT ] || JAX_PLATFORMS=cpu .venv/bin/python experiments/em_vs_gradient/make_mixdom_init.py 0 $INIT 2 > /dev/null 2>&1
  if ! grep -q 'MAP_LL=' $MD/em_d2f1_fixed_2000.log 2>/dev/null; then gate; echo "$(date -Is) gpu${GPU}MX start mixdom_em d2f1 200"
    .venv/bin/python -u train_pfam.py --svi-bw $COMMON --svi-batch 1000 --svi-kappa 0 --svi-tau 1 --n-iter 200 \
       --init-from-maraschino $INIT --pi-pseudo 0 --S-pseudo 0 --seed 0 --checkpoint $MD/em_d2f1_fixed.npz --fresh > $MD/em_d2f1_fixed.log 2>&1
    echo "$(date -Is) gpu${GPU}MX end mixdom_em d2f1 200 rc=$?"
    cp $MD/em_d2f1_fixed.npz $MD/em_d2f1_fixed_2000.npz; echo "$(date -Is) gpu${GPU}MX start mixdom_em d2f1 200->2000"
    .venv/bin/python -u train_pfam.py --svi-bw $COMMON --svi-batch 1000 --svi-kappa 0 --svi-tau 1 --n-iter 2000 \
       --pi-pseudo 0 --S-pseudo 0 --seed 0 --checkpoint $MD/em_d2f1_fixed_2000.npz > $MD/em_d2f1_fixed_2000.log 2>&1
    echo "$(date -Is) gpu${GPU}MX end mixdom_em d2f1 2000 rc=$?"; fi
  for lr in 0.1 0.01; do grep -q 'MAP_LL=' $MD/adam_d2f1_lr${lr}_long.log 2>/dev/null && continue; gate; echo "$(date -Is) gpu${GPU}MX start mixdom_adam_long d2f1 $lr"
    .venv/bin/python -u train_pfam.py --adam $COMMON --adam-batch 1000 --adam-steps 2000 --adam-lr $lr --seed 0 \
       --init-from-maraschino $INIT --checkpoint $MD/adam_d2f1_lr${lr}_long.npz --fresh > $MD/adam_d2f1_lr${lr}_long.log 2>&1
    echo "$(date -Is) gpu${GPU}MX end mixdom_adam_long d2f1 $lr rc=$?"; done
  echo "$(date -Is) gpu${GPU}MX end mixdom d2f1 EM-vs-Adam chain" ) >> $LOG 2>&1 &
