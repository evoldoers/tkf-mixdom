#!/bin/bash
# Single orchestrator for everything still to run, keyed on each job's own
# "saved <json>" completion line (so it does not depend on older queue logs).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
R=experiments/em_vs_gradient/results; RA=experiments/em_vs_gradient/results_aligned
RV=experiments/em_vs_gradient/results_vjp; MD=experiments/em_vs_gradient/results_mixdom
mkdir -p $RV $MD; LOG=experiments/em_vs_gradient/results_rest_queue.log
saved() { until grep -q '^saved ' "$1" 2>/dev/null; do sleep 90; done; }
unal() { .venv/bin/python -u experiments/em_vs_gradient/compare_optimizers.py --corpus $1 --model $2 --n-pairs 1000 --chunk 16 --seed 0 \
     --methods em lbfgs adam --adam-lrs 0.1 0.01 --em-max-iter 300 --lbfgs-max-fun 300 --adam-max-iter 1000 --tol 1e-7 \
     --out $R/${1}_${2}.json > $R/${1}_${2}.log 2>&1; }
vjp() { local methods="lbfgs_vjp"; [ "$2" = tkf92 ] && methods="lbfgs_vjp adam_vjp"
     .venv/bin/python -u experiments/em_vs_gradient/compare_optimizers.py --corpus $1 --model $2 --n-pairs 1000 --chunk 16 --seed 0 \
     --methods $methods --adam-lrs 0.1 --lbfgs-max-fun 300 --adam-max-iter 300 --tol 1e-7 \
     --out $RV/${1}_${2}.json > $RV/${1}_${2}.log 2>&1; }
alig() { .venv/bin/python -u experiments/em_vs_gradient/compare_optimizers_aligned.py --corpus $1 --model $2 --n-pairs 1000 --seed 0 \
     --methods em lbfgs adam --adam-lrs 0.1 0.01 --em-max-iter 2000 --lbfgs-max-fun 500 --adam-max-iter 3000 --tol 1e-7 \
     --out $RA/${1}_${2}.json > $RA/${1}_${2}.log 2>&1; }
mixdom_em() { .venv/bin/python -u train_pfam.py --svi-bw --precompiled-pairs pfam/precompiled_sub1000_seed0 --n-dom 3 --n-frag 1 \
     --svi-batch 1000 --svi-kappa 0 --svi-tau 1 --n-iter 200 --init-from-maraschino $MD/init_d3f1_seed0.npz --pi-pseudo 0 --S-pseudo 0 \
     --seed 0 --val-every 0 --checkpoint $MD/em_d3f1.npz --fresh > $MD/em_d3f1.log 2>&1; }
mixdom_adam() { .venv/bin/python -u train_pfam.py --adam --precompiled-pairs pfam/precompiled_sub1000_seed0 --n-dom 3 --n-frag 1 \
     --adam-batch 1000 --adam-steps 1500 --adam-lr $1 --seed 0 --init-from-maraschino $MD/init_d3f1_seed0.npz \
     --checkpoint $MD/adam_d3f1_lr$1.npz --fresh > $MD/adam_d3f1_lr$1.log 2>&1; }
stoch() { MODES=adam_tkf92 experiments/em_vs_gradient/stochastic/run_pfam_stochastic.sh >> experiments/em_vs_gradient/stochastic/driver.log 2>&1; }
run() { echo "$(date -Is) $1 start $2"; "$2" $3 $4; echo "$(date -Is) $1 end $2 $3 $4 rc=$?"; }
( saved $R/sim_tkf92_tkf91.log; export CUDA_VISIBLE_DEVICES=0
  run gpu0A stoch; run gpu0A mixdom_em; run gpu0A mixdom_adam 0.01; run gpu0A mixdom_adam 0.1
  run gpu0A vjp sim_tkf92 tkf92; run gpu0A vjp sim_tkf91 tkf91; run gpu0A vjp sim_mixfrag mixfrag ) >> $LOG 2>&1 &
( saved $R/sim_tkf92_tkf92.log; export CUDA_VISIBLE_DEVICES=0
  run gpu0B unal sim_tkf91 tkf91 ) >> $LOG 2>&1 &
( saved $R/pfam_tkf91.log; export CUDA_VISIBLE_DEVICES=1
  run gpu1A unal pfam mixfrag; run gpu1A vjp pfam tkf91; run gpu1A vjp pfam mixfrag ) >> $LOG 2>&1 &
( saved $R/pfam_tkf92.log; export CUDA_VISIBLE_DEVICES=1
  run gpu1B unal sim_mixfrag mixfrag; run gpu1B vjp pfam tkf92 ) >> $LOG 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; run gpu1C alig sim_mixfrag mixfrag ) >> $LOG 2>&1 &
echo "rest queues armed"
