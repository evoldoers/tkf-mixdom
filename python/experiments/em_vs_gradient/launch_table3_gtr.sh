#!/bin/bash
# Table 3 (EM vs gradient) rerun with the single GTR refitted to the training pairs (pfam/gtr_refit_Qpi.npz) as the
# fixed substitution model in place of LG08: the six unaligned rows (simulated corpora re-simulated under it), the two
# aligned simulated rows, and the minibatch svi-BW / Adam row. Results go to results_gtr/, results_aligned_gtr/ and
# stochastic_gtr/ (the aligned Pfam rows are count-based and do not involve the matrix: copied over). Then
# EMVG_RESULTS_TAG=_gtr convergence_figure.py regenerates the table and figure. Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-0}; MINFREE=${MINFREE:-4500}; SUB=pfam/gtr_refit_Qpi.npz
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
E=experiments/em_vs_gradient; R=$E/results_gtr; RA=$E/results_aligned_gtr; ST=$E/stochastic_gtr; mkdir -p $R $RA $ST
LOG=$E/results_rest_queue.log
gate() { until [ $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $GPU | awk -F', ' '{print $2-$1}') -ge $MINFREE ]; do sleep 120; done; }
chunk() { [ "$1" = mixfrag ] && echo 8 || echo 16; }
# each job: up to 3 tries (the GPU is shared; an OOM is retried after 10 min), skipped once its output is complete
unal() { for try in 1 2 3; do grep -q '^saved ' $R/${1}_${2}.log 2>/dev/null && return; gate; echo "$(date -Is) gpu${GPU}T3 start unal $1 $2 (try $try)"
  .venv/bin/python -u $E/compare_optimizers.py --corpus $1 --model $2 --n-pairs 1000 --chunk $(chunk $2) --seed 0 \
     --methods em lbfgs adam --adam-lrs 0.1 0.01 --em-max-iter 300 --lbfgs-max-fun 300 --adam-max-iter 1000 --tol 1e-7 \
     --subst-npz $SUB --out $R/${1}_${2}.json > $R/${1}_${2}.log 2>&1; echo "$(date -Is) gpu${GPU}T3 end unal $1 $2 rc=$?"
  grep -q '^saved ' $R/${1}_${2}.log 2>/dev/null && return; sleep 600; done; }
al() { for try in 1 2 3; do grep -q '^saved ' $RA/${1}_${2}.log 2>/dev/null && return; gate; echo "$(date -Is) gpu${GPU}T3 start aligned $1 $2 (try $try)"
  .venv/bin/python -u $E/compare_optimizers_aligned.py --corpus $1 --model $2 --n-pairs 1000 --seed 0 --methods em lbfgs adam --adam-lrs 0.1 0.01 \
     --em-max-iter 2000 --lbfgs-max-fun 500 --adam-max-iter 3000 --tol 1e-7 --subst-npz $SUB \
     --out $RA/${1}_${2}.json > $RA/${1}_${2}.log 2>&1; echo "$(date -Is) gpu${GPU}T3 end aligned $1 $2 rc=$?"
  grep -q '^saved ' $RA/${1}_${2}.log 2>/dev/null && return; sleep 600; done; }
sg() { for try in 1 2 3; do [ -f $ST/pfam_${1}_seed0.json ] && return; gate; echo "$(date -Is) gpu${GPU}T3 start minibatch $1 (try $try)"
  .venv/bin/python -u experiments/run_tkf92_2dfb_pfam.py --mode $1 --precompiled-dir pfam/precompiled --n-train 20000 --n-val 500 --batch-size 16 \
     --n-iter 800 --patience 100000 --val-every 5 --seed 0 --init-lam 0.04 --init-mu 0.05 --init-ext 0.5 \
     --bin-bucketed --pre-warm --no-command-buffers --max-pad-cap 256 --Q npz --subst-npz $SUB \
     --out $ST/pfam_${1}_seed0.json > $ST/pfam_${1}_seed0.log 2>&1; echo "$(date -Is) gpu${GPU}T3 end minibatch $1 rc=$?"
  [ -f $ST/pfam_${1}_seed0.json ] && return; sleep 600; done; }
( for c in pfam_tkf92 pfam_mixfrag; do [ -f $RA/$c.json ] || cp $E/results_aligned/$c.json $E/results_aligned/$c.log $RA/; done
  unal sim_tkf91 tkf91; unal sim_tkf92 tkf92; unal sim_mixfrag mixfrag
  unal pfam tkf91; unal pfam tkf92; unal pfam mixfrag
  al sim_tkf92 tkf92; al sim_mixfrag mixfrag
  sg svi_bw; sg adam_tkf92
  echo "$(date -Is) gpu${GPU}T3 end table-3 GTR rerun" ) >> $LOG 2>&1 &
