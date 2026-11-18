#!/bin/bash
# Post-Table-3 work, reordered by the user 2026-11-15 evening: the Figure 4 error bars and the d2f1 Adam early-stopping
# diagnostic run on GPU 1 as soon as the simulated F=3 job frees it (while the Pfam F=3 job keeps GPU 0); Table 3 and the
# convergence figure are regenerated once both F=3 jobs have saved and the error bars exist.
#   MODE=gpu1  (GPU 1): wait for results_gtr/sim_mixfrag_s_mixfrag_s3.log 'saved' and an empty GPU 1, then
#              (1) perpair_trajectory.py on the TKF92 Pfam run -> results_gtr/pfam_tkf92_perpair.npz;
#              (2) the d2f1 Adam tail rerun with validation every 100 steps, no stopping
#                  -> pfam/adam_d2f1_nobali_from1620_valdiag.{npz,log}.
#   MODE=table (CPU): wait for both F=3 logs 'saved', no compare_optimizers process, and the per-pair file; then
#              regenerate Table 3 + the figure (with the dotted +-SE bands).
# Usage: MODE=gpu1 <script> 1 ; MODE=table <script>
cd "$(dirname "$0")/../.."
MODE=${MODE:-table}; GPU=${1:-1}; E=experiments/em_vs_gradient; R=$E/results_gtr; Q=$E/results_rest_queue.log; P=gpu${GPU}FG
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
gpu_busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU)" ]; }
if [ "$MODE" = gpu1 ]; then
( until grep -q '^saved ' $R/sim_mixfrag_s_mixfrag_s3.log 2>/dev/null && ! gpu_busy; do sleep 300; done
  echo "$(date -Is) $P simulated F=3 saved and GPU $GPU idle; start per-pair trajectory (Figure 4 error bars)"
  JAX_PLATFORMS=cuda .venv/bin/python -u $E/perpair_trajectory.py $R/pfam_tkf92.json > $R/pfam_tkf92_perpair.log 2>&1
  echo "$(date -Is) $P end per-pair trajectory rc=$? $(grep -h 'max |recomputed' $R/pfam_tkf92_perpair.log | tr '\n' ';' | cut -c1-300)"
  echo "$(date -Is) $P start early-stopping diagnostic: d2f1 Adam tail, val every 100 steps, no stopping"
  JAX_PLATFORMS=cuda .venv/bin/python -u train_pfam.py --adam --precompiled-pairs pfam/precompiled_v3_train_nobali --n-dom 2 --n-frag 1 \
     --adam-batch 2000 --adam-steps 2000 --adam-lr 0.01 --seed 42 --init-from-maraschino pfam/svibw_d2f1_nobali_1620_init.npz \
     --checkpoint pfam/adam_d2f1_nobali_from1620_valdiag.npz --fresh --val-every 100 --patience 1000 --split train \
     > pfam/adam_d2f1_nobali_from1620_valdiag.log 2>&1
  echo "$(date -Is) $P end early-stopping diagnostic rc=$? $(grep -c 'Val eval' pfam/adam_d2f1_nobali_from1620_valdiag.log) val evals"
  echo "$(date -Is) $P end gpu1 part" ) >> $Q 2>&1 &
echo "armed error bars + diagnostic on GPU $GPU (after the simulated F=3 job)"
else
P=cpuFG
( until grep -q '^saved ' $R/pfam_mixfrag_s3.log 2>/dev/null && grep -q '^saved ' $R/sim_mixfrag_s_mixfrag_s3.log 2>/dev/null \
        && [ -z "$(pgrep -f 'compare_optimizers.py')" ] && [ -f $R/pfam_tkf92_perpair.npz ]; do sleep 600; done
  echo "$(date -Is) $P Table 3 jobs finished and per-pair file present; regenerating Table 3 + convergence figure"
  EMVG_RESULTS_TAG=_gtr JAX_PLATFORMS=cpu .venv/bin/python $E/convergence_figure.py > experiments/timing/scratch/convfig_after_table3.out 2>&1
  echo "$(date -Is) $P table/figure rc=$? $(grep -h 'error bars from' experiments/timing/scratch/convfig_after_table3.out)"
  echo "$(date -Is) $P end table part" ) >> $Q 2>&1 &
echo "armed Table 3 + figure regeneration (after both F=3 jobs and the error bars)"
fi
