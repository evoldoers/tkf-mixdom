#!/bin/bash
# Unaligned MixFrag-S rows of Table 3 (EM vs L-BFGS vs Adam on the 2D forward log-likelihood of 1,000 unaligned pairs;
# compare_optimizers.py --model mixfrag_s): {pfam, sim_mixfrag_s} x F in {2, 3}, GTR refit as every fragtype's initial
# (S_f, pi_f) and as the simulation base, as launch_table3_gtr.sh does for the other rows; then the custom-VJP L-BFGS
# row (launch_followup.sh's vjp()). Outputs results_gtr/{corpus}_mixfrag_s{F}.json and results_vjp/{corpus}_mixfrag_s{F}.json
# (the paths convergence_figure.py / summarize.py read). Usage: <gpu> [wait-marker-regex]; skips a job whose log ends in 'saved'.
cd "$(dirname "$0")/../.."
GPU=${1:-0}; MARK=${2:-.}; MINFREE=${MINFREE:-4500}; SUB=pfam/gtr_refit_Qpi.npz
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
E=experiments/em_vs_gradient; R=$E/results_gtr; RV=$E/results_vjp; mkdir -p $R $RV
LOG=$E/results_rest_queue.log
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
unal() { # corpus F   (chunk 8 then 4 on retry)
  exec 9>"$R/${1}_mixfrag_s${2}.lock"; flock -n 9 || { exec 9>&-; return 0; }
  for try in 1 2 3; do grep -q '^saved ' $R/${1}_mixfrag_s${2}.log 2>/dev/null && { exec 9>&-; return; }; gate; local ch=8; [ $try -gt 1 ] && ch=4
    echo "$(date -Is) gpu${GPU}T3 start unal $1 mixfrag_s F=$2 (try $try, chunk $ch)"
    .venv/bin/python -u $E/compare_optimizers.py --corpus $1 --model mixfrag_s --n-fragtypes $2 --n-pairs 1000 --chunk $ch --seed 0 \
       --methods em lbfgs adam --adam-lrs 0.1 0.01 --em-max-iter 300 --lbfgs-max-fun 300 --adam-max-iter 1000 --tol 1e-7 \
       --subst-npz $SUB --out $R/${1}_mixfrag_s${2}.json > $R/${1}_mixfrag_s${2}.log 2>&1; echo "$(date -Is) gpu${GPU}T3 end unal $1 mixfrag_s F=$2 rc=$?"
    grep -q '^saved ' $R/${1}_mixfrag_s${2}.log 2>/dev/null && { exec 9>&-; return; }; sleep 600; done; exec 9>&-; }
vjp() { # corpus F   (L-BFGS with the closed-form gradient, 300 evaluations, as for MixFrag)
  exec 9>"$RV/${1}_mixfrag_s${2}.lock"; flock -n 9 || { exec 9>&-; return 0; }
  for try in 1 2 3; do grep -q '^saved ' $RV/${1}_mixfrag_s${2}.log 2>/dev/null && { exec 9>&-; return; }; gate; local ch=8; [ $try -gt 1 ] && ch=4
    echo "$(date -Is) gpu${GPU}T3 start vjp $1 mixfrag_s F=$2 (try $try, chunk $ch)"
    .venv/bin/python -u $E/compare_optimizers.py --corpus $1 --model mixfrag_s --n-fragtypes $2 --n-pairs 1000 --chunk $ch --seed 0 \
       --methods lbfgs_vjp --adam-lrs 0.1 --lbfgs-max-fun 300 --adam-max-iter 300 --tol 1e-7 \
       --subst-npz $SUB --out $RV/${1}_mixfrag_s${2}.json > $RV/${1}_mixfrag_s${2}.log 2>&1; echo "$(date -Is) gpu${GPU}T3 end vjp $1 mixfrag_s F=$2 rc=$?"
    grep -q '^saved ' $RV/${1}_mixfrag_s${2}.log 2>/dev/null && { exec 9>&-; return; }; sleep 600; done; exec 9>&-; }
( until grep -q -E "$MARK" $LOG 2>/dev/null; do sleep 120; done
  JOBS=("unal pfam 2" "unal sim_mixfrag_s 2" "unal pfam 3" "unal sim_mixfrag_s 3" "vjp pfam 2" "vjp sim_mixfrag_s 2" "vjp pfam 3" "vjp sim_mixfrag_s 3")
  [ "${REVERSE:-0}" = 1 ] && { n=${#JOBS[@]}; RJ=(); for ((i=n-1;i>=0;i--)); do RJ+=("${JOBS[i]}"); done; JOBS=("${RJ[@]}"); }   # second chain, other card
  for j in "${JOBS[@]}"; do eval "$j"; done
  echo "$(date -Is) gpu${GPU}T3 end unaligned MixFrag-S rows" ) >> $LOG 2>&1 &
echo "armed unaligned MixFrag-S rows gpu=$GPU after /$MARK/"
