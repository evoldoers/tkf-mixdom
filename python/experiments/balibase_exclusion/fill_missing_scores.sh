#!/bin/bash
# Fill in held-out scores (path-conditioned, order-summed) that a chain's
# scoring step lost to GPU OOM. Loops every 20 min over every nobali checkpoint
# that has a scoring-format file, runs whichever of the two scores is missing
# when the GPU has >= MINFREE MiB free, and exits when nothing is missing and
# no training chain is alive. Usage: fill_missing_scores.sh <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-0}; MINFREE=${MINFREE:-6500}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
recent() { [ -f "$1" ] && [ $(( $(date +%s) - $(stat -c %Y "$1") )) -lt 1200 ]; }   # log written in the last 20 min => someone else is on it
tag_of() { local b=$(basename $1 .npz); b=${b#svibw_}; b=${b/_nobali_/_nobali_}; echo $b; }
( while true; do
    missing=0
    for ck in pfam/svibw_d[1-5]f[12]_nobali_[0-9]*.npz pfam/svibw_d3f1_full_[0-9]*.npz pfam/adam_d[1-5]f[12]_nobali_from1620_scoring.npz pfam/adam_d3f1_full_from1620_scoring.npz pfam/lbfgs_d[1-5]f[12]_nobali_from1620.npz pfam/lbfgs_d3f1_full_from1620.npz; do
      [ -f "$ck" ] || continue
      case $ck in
        pfam/svibw_*) t=$(basename $ck .npz); t=${t#svibw_} ;;                     # dNf1_nobali_540
        pfam/adam_*)  t=$(basename $ck _scoring.npz); t=${t#adam_}; t=${t/_from/_adam_from} ;;   # dNf1_nobali_adam_from1620
        pfam/lbfgs_*) t=$(basename $ck .npz); t=${t#lbfgs_}; t=${t/_from/_lbfgs_from} ;;
      esac
      # only score checkpoints whose chain has finished the training stage (log says complete) or that are tails
      case $ck in pfam/svibw_*) grep -q 'SVI Baum-Welch complete' pfam/$(basename $ck .npz).log 2>/dev/null || continue ;; esac
      # never race a chain: act only once the chain has logged the end of its own scoring step for this tag
      grep -q "end score $t" $LOG 2>/dev/null || continue
      if ! grep -q PATH-CONDITIONED experiments/ordsum_val/pathcond_$t.log 2>/dev/null; then
        # leave very recent chains alone: their own score step is probably running
        recent experiments/ordsum_val/pathcond_$t.log && continue
        missing=1; gate; echo "$(date -Is) gpu${GPU}F fill pathcond $t"
        .venv/bin/python -u experiments/em_vs_gradient/score_pathcond_val.py --checkpoint $ck --out experiments/ordsum_val/pathcond_$t.json > experiments/ordsum_val/pathcond_$t.log 2>&1
      fi
      if ! grep -q 'LL/pair  =' experiments/ordsum_val/score_$t.log 2>/dev/null; then
        recent experiments/ordsum_val/score_$t.log && continue
        missing=1; gate; echo "$(date -Is) gpu${GPU}F fill ordsum $t"
        MODEL=d3f1 CKPT=$PWD/$ck .venv/bin/python -u experiments/ordsum_val/score_matchaligned_val.py > experiments/ordsum_val/score_$t.log 2>&1
      fi
      # BAliBASE at the 540 and 1620 stages and after the tails (not at 1080)
      case $t in *_540|*_1620|*adam_from*|*lbfgs_from*)
        bj=experiments/expected_balibase/expected_balibase_mixdom_${t}_withsps.json; bl=${bj%.json}.log
        if [ ! -f $bj ] && ! recent $bl; then
          missing=1; gate; echo "$(date -Is) gpu${GPU}F fill balibase $t"
          .venv/bin/python -u experiments/expected_pairwise_balibase.py --method mixdom --params $ck --method-name mixdom_$t --fsa on --fsa-sps --out $bj > $bl 2>&1
        fi ;;
      esac
    done
    pgrep -f 'launch_retrain_nobali|launch_d4f1_540' >/dev/null || [ $missing -eq 1 ] || { echo "$(date -Is) gpu${GPU}F fill_missing_scores done"; break; }
    sleep 1200
  done ) >> $LOG 2>&1 &
echo armed
