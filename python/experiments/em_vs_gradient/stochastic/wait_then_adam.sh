#!/bin/bash
# Wait for the first GPU-0 full-batch queue to finish, then run the stochastic Adam job on GPU 0.
cd "$(dirname "$0")/../../.."
until grep -q 'end sim_tkf92:tkf91' experiments/em_vs_gradient/results/queue_gpu0.log 2>/dev/null; do sleep 60; done
echo "$(date -Is) starting stochastic adam on gpu0"
MODES=adam_tkf92 CUDA_VISIBLE_DEVICES=0 experiments/em_vs_gradient/stochastic/run_pfam_stochastic.sh
