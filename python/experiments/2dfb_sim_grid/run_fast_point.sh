#!/bin/bash
# Stand-alone "fast" cell, run in parallel with the slow 12-cell grid.
# Goal: get ANY-signal-from-GGI answer quickly.
#
# Params: K=20 component 9 (highest λ+μ → fastest flow):
#   TKF92 truth: λ=0.0709 μ=0.0720 ext=0.5882, κ=0.985
#   GGI proj   : λ₀=0.0292 μ₀=0.0297 x=0.5882 (upper), y=0.5634
#   k=0.202, t_half=3.44
# At t×10 (median 7.6), fraction (r_b−r_inf) traversed = 78%.
cd /home/yam/tkf-mixdom/python

LOG=experiments/2dfb_sim_grid/fast.log
echo "$(date) FAST: generating sim data (CPU)" > $LOG

DIR=experiments/2dfb_sim_grid/fast
mkdir -p $DIR

# Stage 1 for the fast point (CPU sim, ~25 min).
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 uv run python \
    experiments/2dfb_sim/simulate_pfam_like.py \
    --out-dir $DIR \
    --t-scale 10.0 \
    --tkf92-lam 0.0709 --tkf92-mu 0.0720 --tkf92-ext 0.5882 \
    --ggi-lam0 0.0292 --ggi-mu0 0.0297 --ggi-x 0.5882 --ggi-segment upper \
    > $DIR/simulate.log 2>&1
echo "$(date) FAST: sim done; running at-truth eval" >> $LOG

# Stage 2: at-truth eval on the GGI sim val ONLY (we don't care about TKF92-sim here).
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run python -c "
import os, sys, time, json, pickle
os.environ['JAX_ENABLE_X64'] = '1'
os.environ['TKFMIXDOM_MAX_PAD'] = '256'
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='
sys.path.insert(0, '.')
sys.path.insert(0, 'experiments')
sys.path.insert(0, 'experiments/2dfb')
import jax.numpy as jnp
from tkfmixdom.jax.core.protein import rate_matrix_lg
from eval_all_on_1500val import eval_tkf92, eval_ggi

Q, pi = rate_matrix_lg()
val_pairs = pickle.load(open('$DIR/sim_ggi/val.pkl', 'rb'))
print(f'  loaded {len(val_pairs)} GGI-sim val pairs', flush=True)

# TKF92 truth
lam, mu, ext = 0.0709, 0.0720, 0.5882
t0 = time.time()
ll = eval_tkf92(lam, mu, ext, val_pairs, Q, pi)
v_tkf = ll / len(val_pairs)
print(f'  TKF92(truth) val_ll/pair = {v_tkf:.4f}  ({time.time()-t0:.1f}s)', flush=True)

# GGI truth (swap=True is the matching prior)
lam0, mu0, x, y = 0.0292, 0.0297, 0.5882, 0.5634
t0 = time.time()
ll = eval_ggi(lam0, mu0, x, y, val_pairs, Q, pi, prior_swap=True)
v_ggi_swap = ll / len(val_pairs)
print(f'  GGI(truth, swap=T) val_ll/pair = {v_ggi_swap:.4f}  ({time.time()-t0:.1f}s)', flush=True)
t0 = time.time()
ll = eval_ggi(lam0, mu0, x, y, val_pairs, Q, pi, prior_swap=False)
v_ggi_noswap = ll / len(val_pairs)
print(f'  GGI(truth, swap=F) val_ll/pair = {v_ggi_noswap:.4f}  ({time.time()-t0:.1f}s)', flush=True)
print(f'\\n  Δ (ggi_swap − tkf92) = {v_ggi_swap - v_tkf:+.4f}  (positive ⇒ GGI better)', flush=True)
json.dump({
    'tkf92_at_truth': v_tkf, 'ggi_at_truth_swap': v_ggi_swap, 'ggi_at_truth_noswap': v_ggi_noswap,
    'delta': v_ggi_swap - v_tkf,
}, open('$DIR/eval_at_truth.json', 'w'), indent=2)
" > $DIR/eval_at_truth.log 2>&1
echo "$(date) FAST: at-truth eval done; launching parallel Adam fits" >> $LOG

# Stage 3: Adam-TKF92 || Adam-GGI on the GGI-sim train
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run python -u \
    experiments/run_tkf92_2dfb_pfam.py \
    --mode adam_tkf92 \
    --sim-train-file $DIR/sim_ggi/train.pkl \
    --sim-val-file   $DIR/sim_ggi/val.pkl \
    --batch-size 16 --n-iter 800 --patience 100 \
    --bin-bucketed --pre-warm --no-command-buffers --max-pad-cap 256 \
    --out $DIR/adam_tkf92.json \
    > $DIR/adam_tkf92.log 2>&1 &
PID_T=$!

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run python -u \
    experiments/run_tkf92_2dfb_pfam.py \
    --mode adam_ggi \
    --sim-train-file $DIR/sim_ggi/train.pkl \
    --sim-val-file   $DIR/sim_ggi/val.pkl \
    --init-mu0 0.0297 --init-rho 0.9847 --init-x 0.5882 \
    --ggi-segment upper \
    --batch-size 16 --n-iter 800 --patience 100 \
    --bin-bucketed --pre-warm --no-command-buffers --max-pad-cap 256 \
    --out $DIR/adam_ggi.json \
    > $DIR/adam_ggi.log 2>&1 &
PID_G=$!

wait $PID_T $PID_G
echo "$(date) FAST: all done" >> $LOG
