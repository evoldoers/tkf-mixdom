#!/bin/bash
# Reproducibility of the released model's domain types ON THE SAME DATA AND PRIORS as Table 5: svi-BW MixDom-d2f1
# on the BAliBASE-free training split (precompiled_v3_train_nobali, 993,968 pairs), 1,620 iterations, minibatches
# of 2,000, step size (k+1)^-0.7, default priors, from four random seeds (the Table 5 fit is seed 42), for
# NITER iterations (default 200: the d3f1 log shows the types identifiable by iteration 20 and the rates within a
# few percent of final by 100). Outputs pfam/svibw_d2f1_nobali_<NITER>_seed<s>.{npz,log}. Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-0}; MINFREE=${MINFREE:-5000}; NITER=${NITER:-200}; SEEDS=${SEEDS:-1 2 3 4}; NDOM=${NDOM:-2}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
( for s in $SEEDS; do
    grep -q 'SVI Baum-Welch complete' pfam/svibw_d${NDOM}f1_nobali_${NITER}_seed$s.log 2>/dev/null && continue
    gate; echo "$(date -Is) gpu${GPU}MF start svibw d${NDOM}f1 nobali seed$s"
    .venv/bin/python -u train_pfam.py --svi-bw --precompiled-pairs pfam/precompiled_v3_train_nobali --n-dom $NDOM --n-frag 1 --n-classes $NDOM --classdist-init identity --freeze-classdist --estimate-subst --estimate-sexch \
       --svi-batch 2000 --svi-tau 1 --svi-kappa 0.7 --n-iter $NITER --seed $s --val-every 0 --jax-cache-dir /home/yam/.cache/tkf-mixdom-jax \
       --checkpoint pfam/svibw_d${NDOM}f1_nobali_${NITER}_seed$s.npz --fresh > pfam/svibw_d${NDOM}f1_nobali_${NITER}_seed$s.log 2>&1
    echo "$(date -Is) gpu${GPU}MF end svibw d${NDOM}f1 nobali seed$s rc=$?"
  done; echo "$(date -Is) gpu${GPU}MF end d${NDOM}f1 full-data restarts chain" ) >> $LOG 2>&1 &
