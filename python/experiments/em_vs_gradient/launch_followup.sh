#!/bin/bash
# Follow-up GPU queue, started once the first unaligned queues have finished:
#  (a) custom-VJP gradient variants (L-BFGS and Adam lr 0.1) for the six
#      unaligned (corpus, model) combinations, plus the three-way gradient check;
#  (b) alignment-conditioned MixDom-d3f1 on the fixed 1000-pair subset:
#      full-batch closed-form EM (svi-BW with eta=1, batch=corpus) vs Adam on
#      the Q-function from the same E-step, both with per-domain substitution
#      estimated and the same priors.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
OUT=experiments/em_vs_gradient/results_vjp; mkdir -p $OUT
MD=experiments/em_vs_gradient/results_mixdom; mkdir -p $MD
vjp() {  # corpus model   (Adam with the custom-VJP gradient only for TKF92, 300 passes:
         # enough to show it follows the autodiff-Adam trajectory pass for pass)
  local methods="lbfgs_vjp"; [ "$2" = tkf92 ] && methods="lbfgs_vjp adam_vjp"
  .venv/bin/python -u experiments/em_vs_gradient/compare_optimizers.py \
     --corpus $1 --model $2 --n-pairs 1000 --chunk 16 --seed 0 \
     --methods $methods --adam-lrs 0.1 \
     --lbfgs-max-fun 300 --adam-max-iter 300 --tol 1e-7 \
     --out $OUT/${1}_${2}.json > $OUT/${1}_${2}.log 2>&1
}
mixdom_em() {
  .venv/bin/python -u train_pfam.py --svi-bw --precompiled-pairs pfam/precompiled_sub1000_seed0 \
     --n-dom 3 --n-frag 1 --svi-batch 1000 --svi-kappa 0 --svi-tau 1 --n-iter 200 \
     --init-from-maraschino $MD/init_d3f1_seed0.npz --pi-pseudo 0 --S-pseudo 0 \
     --seed 0 --val-every 0 --checkpoint $MD/em_d3f1.npz --fresh > $MD/em_d3f1.log 2>&1
}
mixdom_adam() {  # lr
  .venv/bin/python -u train_pfam.py --adam --precompiled-pairs pfam/precompiled_sub1000_seed0 \
     --n-dom 3 --n-frag 1 --adam-batch 1000 --adam-steps 1500 --adam-lr $1 --seed 0 \
     --init-from-maraschino $MD/init_d3f1_seed0.npz \
     --checkpoint $MD/adam_d3f1_lr$1.npz --fresh > $MD/adam_d3f1_lr$1.log 2>&1
}
wait_for() { until grep -q "$1" "$2" 2>/dev/null; do sleep 120; done; }
# GPU 0: after the sim_tkf92:tkf91 unaligned job ends -> MixDom EM, Adam(0.01), Adam(0.1), then sim vjp runs
( wait_for 'end sim_tkf92:tkf91' experiments/em_vs_gradient/results/queue_gpu0.log
  for j in em adam0.01 adam0.1; do echo "$(date -Is) gpu0 start mixdom $j"; case $j in em) CUDA_VISIBLE_DEVICES=0 mixdom_em;; adam0.01) CUDA_VISIBLE_DEVICES=0 mixdom_adam 0.01;; adam0.1) CUDA_VISIBLE_DEVICES=0 mixdom_adam 0.1;; esac; echo "$(date -Is) gpu0 end mixdom $j rc=$?"; done
  for j in sim_tkf92:tkf91 sim_tkf92:tkf92 sim_tkf92:mixfrag; do echo "$(date -Is) gpu0 start vjp $j"; CUDA_VISIBLE_DEVICES=0 vjp ${j%:*} ${j#*:}; echo "$(date -Is) gpu0 end vjp $j rc=$?"; done
) >> $OUT/queue_gpu0.log 2>&1 &
# GPU 1: after the pfam:tkf91 unaligned job ends -> pfam vjp runs
( wait_for 'end pfam:tkf91' experiments/em_vs_gradient/results/queue_gpu1.log
  for j in pfam:tkf91 pfam:tkf92 pfam:mixfrag; do echo "$(date -Is) gpu1 start vjp $j"; CUDA_VISIBLE_DEVICES=1 vjp ${j%:*} ${j#*:}; echo "$(date -Is) gpu1 end vjp $j rc=$?"; done
) >> $OUT/queue_gpu1.log 2>&1 &
echo "follow-up queues armed"
