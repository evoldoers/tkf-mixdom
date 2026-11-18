#!/bin/bash
# Path-conditioned TEST-split likelihoods (pfam/precompiled_v3_test, 213,930 pairs) for the rows of the MBE tables that
# still lack one, now that the order-summed columns are dropped and each table keeps a path-conditioned validation
# column and a path-conditioned test column. One job at a time, in priority order, gated on free GPU memory
# (MINFREE MiB, default 2600), skipped when the log already holds a result, retried up to three times after 600 s on failure.
# Every cell comes through the one scorer (experiments/ordsum_val/score_matchaligned_val.py, PATHCOND=1: whole path
# fixed), with exactly the model/env of the chain that produced the matching validation cell:
#   MixDom checkpoints      MODEL=d3f1 CKPT=<npz>                       (as the existing pathcond_d2f1_nobali_1620.log etc.,
#                                                                        launch_test_ll.sh@208237e41; agrees with
#                                                                        score_pathcond_val.py to 1e-6 LL/pair on val)
#   TKF92 + refitted GTR    MODEL=cherryml_C20 CHERRYML_NPZ=pfam/gtr_refit_C1_paircol.npz TKF92_PARAMS=tkf92_pfam_params.json
#                                                                        (launch_gtr_refit_ll.sh)
#   MixFrag F=2/3 + GTR     MODEL=mixfrag SUBST_NPZ=pfam/gtr_refit_Qpi.npz [MIXFRAG_PARAMS=<F3 json>]
#                                                                        (launch_gtr_refit_mixfrag_ll.sh)
#   MixFrag-S               MODEL=mixfrag_s MIXFRAG_S_NPZ=<npz>          (balibase_exclusion/launch_mixfrag_s.sh score())
#   K=20 family mixtures    20 per-component TKF92 runs with PERPAIR_OUT -> <dir>/test_pc1_comp<k>.{log,npz}, then
#                           combine_k20.py <dir> <tag> writes experiments/test_ll/pathcond_<tag>.log and
#                           pathcond_<tag>_perpair.log (launch_mixture_ll.sh / launch_tied_*_ll.sh / launch_tied_indel_control.sh)
# Outputs: experiments/test_ll/pathcond_<stem>.log (the scorer writes no JSON). Queue log lines carry the gpu${GPU}PT prefix.
# Usage: launch_pathcond_test.sh <gpu>        (env: MINFREE=<MiB>)
cd "$(dirname "$0")/../.."
GPU=${1:-1}; MINFREE=${MINFREE:-2600}; T=experiments/test_ll; TEST=$PWD/pfam/precompiled_v3_test; P=gpu${GPU}PT
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
LOG=experiments/em_vs_gradient/results_rest_queue.log
SCORER=experiments/ordsum_val/score_matchaligned_val.py
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
has_result() { grep -q 'LL/pair  =' "$1" 2>/dev/null; }   # the scorer's and combine_k20.py's result line (progress lines never match)
run() {  # logfile [ENV=VALUE ...] : one path-conditioned test-split scorer job; skipped when done or claimed by another chain
  local out=$1; shift; has_result $out && return 0
  exec 9>"$out.lock"; flock -n 9 || { exec 9>&-; return 0; }   # a chain on the other card holds this job
  if ! has_result $out; then for try in 1 2 3 4; do gate; echo "$(date -Is) $P start $out (try $try)"
    env "$@" PATHCOND=1 VAL_DIR=$TEST .venv/bin/python -u $SCORER > $out 2>&1; local rc=$?
    echo "$(date -Is) $P end $out rc=$rc"; has_result $out && break
    [ $try -lt 4 ] && sleep 600; done; fi
  has_result $out || echo "$(date -Is) $P FAILED $out after 4 tries"; exec 9>&-; has_result $out; }
mixdom() { run $T/pathcond_$1.log MODEL=d3f1 CKPT=$PWD/$2; }                       # stem ckpt
mixfrag_s() { run $T/pathcond_$1.log MODEL=mixfrag_s MIXFRAG_S_NPZ=$2; }           # stem ckpt
k20() {  # dir tag kind(percomp|c20|gtr) : the 20 components, then the family-level combine
  local D=$T/$1 tag=$2 kind=$3 k; has_result $T/pathcond_$tag.log && return 0
  echo "$(date -Is) $P start K=20 test path-conditioned components $tag ($1)"
  for k in $(seq 0 19); do
    case $kind in
      percomp) run $D/test_pc1_comp$k.log MODEL=tkf92 SUBST_NPZ=$D/comp_$k.npz TKF92_PARAMS=$D/comp_$k.json PERPAIR_OUT=$D/test_pc1_comp$k.npz ;;
      c20)     run $D/test_pc1_comp$k.log MODEL=cherryml_C20 CHERRYML_NPZ=pfam/cherryml_mixture_C20_paircol.npz TKF92_PARAMS=$D/comp_$k.json PERPAIR_OUT=$D/test_pc1_comp$k.npz ;;
      gtr)     run $D/test_pc1_comp$k.log MODEL=tkf92 SUBST_NPZ=pfam/gtr_refit_Qpi.npz TKF92_PARAMS=$D/comp_$k.json PERPAIR_OUT=$D/test_pc1_comp$k.npz ;;
    esac; done
  # combine_k20.py rewrites every (split, convention) whose 20 per-pair files exist (val/test, pc1/pc0) with the same
  # numbers as before, plus the new test pc1 -> pathcond_<tag>.log and pathcond_<tag>_perpair.log. CPU only.
  JAX_PLATFORMS=cpu .venv/bin/python experiments/test_ll/combine_k20.py $1 $tag > $D/combine_test_pc1.log 2>&1
  if has_result $T/pathcond_$tag.log; then echo "$(date -Is) $P end K=20 combine $tag: $(grep -h 'LL/pair  =' $T/pathcond_$tag.log | tr -s ' ')"
  else echo "$(date -Is) $P K=20 combine $tag produced no test path-conditioned result (missing components?)"; fi; }
JOBS=(
  # 1. the refitted-GTR fixed-parameter rows
  "run $T/pathcond_mixfrag_gtrrefit.log MODEL=mixfrag SUBST_NPZ=pfam/gtr_refit_Qpi.npz"
  "run $T/pathcond_tkf92_gtrrefit.log MODEL=cherryml_C20 CHERRYML_NPZ=pfam/gtr_refit_C1_paircol.npz TKF92_PARAMS=$T/tkf92_pfam_params.json"
  # 3.
  "run $T/pathcond_mixfrag3_gtrrefit.log MODEL=mixfrag SUBST_NPZ=pfam/gtr_refit_Qpi.npz MIXFRAG_PARAMS=experiments/mixfrag_F/mixfrag_F3_logspace.json"
  "mixdom d1f1_nobali_1620 pfam/svibw_d1f1_nobali_1620.npz"
  "mixfrag_s mixfrags_F2_nobali_1620 pfam/svibw_mixfrags_F2_nobali_1620.npz"
  "mixfrag_s mixfrags_F3_nobali_1620 pfam/svibw_mixfrags_F3_nobali_1620.npz"
  # 4.
  "mixdom d1f2_nobali_1620 pfam/svibw_d1f2_nobali_1620.npz"
  "mixdom d2f2_nobali_1620 pfam/svibw_d2f2_nobali_1620.npz"
  "mixdom d2f1_full_1620 pfam/svibw_d2f1_full_1620.npz"
  "mixdom d3f1_full_1620 pfam/svibw_d3f1_full_1620.npz"
  # the per-family K=20 mixture (10 h; the row the text cites on the test split); its combine also yields tkf92_K20_perpair
  "k20 k20 tkf92_K20 percomp"
  # 5. the Adam tails (checkpoints as the ordsum_<stem>.log test runs: launch_test_ll*.sh@208237e41/500f5d85c)
  "mixdom d2f1_nobali_adam pfam/adam_d2f1_nobali_from1620_scoring.npz"
  "mixdom d3f1_nobali_adam pfam/adam_d3f1_nobali_from1620_scoring.npz"
  "mixdom d4f1_nobali_adam pfam/adam_d4f1_nobali_from1620_scoring.npz"
  "mixdom d5f1_nobali_adam pfam/adam_d5f1_nobali_from1620_scoring.npz"
  "mixdom d2f1_full_adam pfam/adam_d2f1_full_from1620_scoring.npz"
  # 6. the remaining family-level mixtures (tkf92_K20_perpair comes out of the K=20 combine)
  "k20 k20tied tkf92_K20tied c20"
  "k20 k20tiedgtr tkf92_K20tiedgtr gtr"
  "k20 k20tiedindel tkf92_K20tiedindel percomp"
)
# REVERSE=1 walks the list from the end: a second chain on the other card takes the long tail-end jobs while the first
# works forward; the per-job locks keep the two from colliding, and each skips what the other has finished.
[ "${REVERSE:-0}" = 1 ] && { n=${#JOBS[@]}; R=(); for ((i=n-1;i>=0;i--)); do R+=("${JOBS[i]}"); done; JOBS=("${R[@]}"); }
# JOBS_FIRST=<regex>: jobs matching the regex run first (in list order), then the rest; combine with REVERSE for the second chain.
if [ -n "${JOBS_FIRST:-}" ]; then A=(); B=(); for j in "${JOBS[@]}"; do if [[ "$j" =~ $JOBS_FIRST ]]; then A+=("$j"); else B+=("$j"); fi; done; JOBS=("${A[@]}" "${B[@]}"); fi
( echo "$(date -Is) $P start path-conditioned test chain (MINFREE=$MINFREE, REVERSE=${REVERSE:-0})"
  for j in "${JOBS[@]}"; do eval "$j"; done
  echo "$(date -Is) $P end path-conditioned test chain" ) >> $LOG 2>&1 &
echo "armed path-conditioned test chain on gpu $GPU (MINFREE=$MINFREE, REVERSE=${REVERSE:-0})"
