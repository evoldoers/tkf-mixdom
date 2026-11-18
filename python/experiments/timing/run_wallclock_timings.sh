#!/bin/bash
# Uncontested wall-clock timings for paper 1 (user decision 2026-11-14: time a fixed block of iterations per model on an
# otherwise idle machine and report cost per iteration x iterations). Waits until both K=20 fitted-clock chains have
# ended and no process is on either GPU, then runs everything on GPU 0, one job at a time:
#   T1  timed passes (E-step, autodiff value+grad, closed-form VJP) of compare_optimizers.py on the 1,000-pair Pfam
#       corpus for TKF92 (3 repeats) and MixFrag F=2 (1 repeat)        -> Table 4 "cost per pass", Fig 4b caption
#   T2  svi-BW: 20 and 80 iterations from scratch for d1f1..d5f1, d1f2, d2f2 and MixFrag-S F=2,3 (BAliBASE-free split,
#       the Table 6 recipe) -> per-iteration cost = (t80 - t20)/60 of the TIMED pass (see below); 1,620-iteration
#       estimate = startup + 1,620 x per-iteration. The trainers' own per-iteration timestamps (median over
#       iterations 21-80) are reported alongside as a cross-check.
# Warm-up: the 20-iteration (25-step) run of each configuration compiles or cache-loads every length-bucket shape
# (every bucket appears within the first few 2,000-pair minibatches; the persistent JIT caches already hold the
# executables from the real trainings). The 80-iteration (125-step) run is then timed from its own per-iteration
# timestamps over iterations 2..80 (median, with the mean alongside), so compilation never enters the number.
#   T3  Adam tail: 25 and 125 steps from the d2f1 and d3f1 1,620-iteration fits -> per-step cost x 2,000
#   T4  count aggregation of the 1.13 M training pairs + 10 EM iterations (mixfrag_from_shards.py, F=2)
# Results: experiments/timing/wallclock_<date>.log (everything) and wallclock_<date>.json (the numbers).
cd "$(dirname "$0")/../.."
MODE=${MODE:-timed}; GPU=${GPU:-0}   # MODE=warm: only the cache-warming runs, now, on $GPU (no idle wait); timed: the measured runs on an idle machine
DATE=${DATE:-$(date +%F)}; T=experiments/timing; mkdir -p $T/scratch; LOGF=$T/wallclock_$DATE.log; OUT=$T/wallclock_$DATE.json
Q=experiments/em_vs_gradient/results_rest_queue.log
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=$GPU
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
log() { echo "$(date -Is) $*" | tee -a $LOGF; }
idle() { [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && awk '{exit !($1 < 3.0)}' /proc/loadavg; }
# ---- wait for an idle machine (timed mode only) ----
if [ "$MODE" = timed ]; then
log "queued; waiting for both K=20 fitted-clock chains and an idle machine"
until grep -q "end K=20 fitted-clock chain (K20 K20tied)" $Q && grep -q "end K=20 fitted-clock chain (K20tiedgtr K20tiedindel)" $Q; do sleep 300; done
until idle; do sleep 120; done; sleep 120; until idle; do sleep 120; done
log "machine idle: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "$(date -Is) gpu${GPU}WT start uncontested wall-clock timing session" >> $Q
else log "warm mode on GPU $GPU: cache-warming runs only"; fi
[ -f $OUT ] || python3 - "$DATE" "$GPU" <<'EOF' > $OUT
import json, sys; print(json.dumps({"date": sys.argv[1], "gpu": int(sys.argv[2]), "runs": {}}, indent=1))
EOF
rec() {  # key seconds [note]
  python3 - "$OUT" "$1" "$2" "${3:-}" <<'EOF'
import json, sys; p, k, s, note = sys.argv[1:5]; d = json.load(open(p)); d["runs"][k] = {"seconds": float(s), "note": note}; json.dump(d, open(p, "w"), indent=1)
EOF
}
PASS=warm
timeit() {  # key cmd... : wall-clock seconds of one command (stdout/stderr to the scratch log); key gets the pass suffix
  local k=${1}_$PASS; shift; local t0=$(date +%s.%N); "$@" > $T/scratch/$k.out 2>&1; local rc=$?; local t1=$(date +%s.%N)
  local s=$(python3 -c "print(f'{$t1-$t0:.1f}')"); rec "$k" "$s" "rc=$rc"; log "$k: ${s}s (rc=$rc)"; }
for PASS in timed; do log "==== single pass (20/25-step runs warm the caches; 80/125-step runs are timed from their own timestamps) ===="
# ---- T1: timed passes (timed mode) ----
if [ "$MODE" = timed ] && [ "${SKIP_T1:-0}" != 1 ]; then for rep in 1 2 3; do
  timeit "T1_tkf92_pfam_rep$rep" .venv/bin/python -u experiments/em_vs_gradient/compare_optimizers.py --corpus pfam --model tkf92 --n-pairs 1000 --max-len 256 --chunk 16 --seed 0 --subst-npz pfam/gtr_refit_Qpi.npz --methods em --em-max-iter 1 --out $T/scratch/timed_pass_tkf92_rep$rep.json
  grep "timed pass" $T/scratch/T1_tkf92_pfam_rep${rep}_$PASS.out | tee -a $LOGF; done
timeit "T1_mixfrag_pfam" .venv/bin/python -u experiments/em_vs_gradient/compare_optimizers.py --corpus pfam --model mixfrag --n-pairs 1000 --max-len 256 --chunk 8 --seed 0 --subst-npz pfam/gtr_refit_Qpi.npz --methods em --em-max-iter 1 --out $T/scratch/timed_pass_mixfrag.json
grep "timed pass" $T/scratch/T1_mixfrag_pfam_$PASS.out | tee -a $LOGF; fi
# ---- T2: svi-BW blocks ----
CORPUS=pfam/precompiled_v3_train_nobali
ITS=80; STEPS=125; [ "$MODE" = warm ] && { ITS=20; STEPS=25; }
svibw() {  # N NFRAG
  local N=$1 F=$2;tag=d${1}f${2}
  local COMMON="--svi-bw --precompiled-pairs $CORPUS --n-dom $N --n-frag $F --n-classes $N --classdist-init identity --freeze-classdist --estimate-subst --estimate-sexch --svi-batch 2000 --svi-tau 1.0 --svi-kappa 0.7 --seed 42 --val-every 0 --jax-cache-dir /home/yam/.cache/tkf-mixdom-jax"
  for it in $ITS; do rm -f $T/scratch/${tag}_$it.npz
    timeit "T2_${tag}_iter$it" .venv/bin/python -u train_pfam.py $COMMON --n-iter $it --checkpoint $T/scratch/${tag}_$it.npz --fresh; done; }
for N in 1 2 3 4 5; do svibw $N 1; done; svibw 1 2; svibw 2 2
mfs() {  # F
  local F=$1
  for it in $ITS; do rm -f $T/scratch/mfs${F}_$it.npz $T/scratch/mfs${F}_$it.log
    timeit "T2_mixfrags_F${F}_iter$it" .venv/bin/python -u train_mixfrag_s.py --precompiled-pairs $CORPUS --n-fragtypes $F --dp-mode path --n-iter $it --svi-batch 2000 --svi-tau 1.0 --svi-kappa 0.7 --seed 42 --init-subst-npz pfam/gtr_refit_Qpi.npz --init-mixfrag-json experiments/mixfrag_F/mixfrag_F${F}_logspace.json --checkpoint $T/scratch/mfs${F}_$it.npz --ckpt-every 1000 --log-every 1 --log-file $T/scratch/mfs${F}_$it.log --jax-cache-dir $HOME/.cache/jax_mixfrag_s; done; }
mfs 2; mfs 3
# ---- T3: Adam tails ----
for N in 2 3; do for st in $STEPS; do rm -f $T/scratch/adam_d${N}f1_$st.npz
  timeit "T3_adam_d${N}f1_steps$st" .venv/bin/python -u train_pfam.py --adam --precompiled-pairs $CORPUS --n-dom $N --n-frag 1 --adam-batch 2000 --adam-steps $st --adam-lr 0.01 --seed 42 --init-from-maraschino pfam/svibw_d${N}f1_nobali_1620_init.npz --checkpoint $T/scratch/adam_d${N}f1_$st.npz --fresh; done; done
# ---- T4: count aggregation + EM iterations (timed mode) ----
if [ "$MODE" = timed ]; then timeit "T4_counts_F2_iter10" .venv/bin/python -u experiments/ordsum_val/mixfrag_from_shards.py --n-frag 2 --n-iter 10 --workers 16 --out $T/scratch/counts_F2_iter10.json
grep -i "agg\|chunk 48/48\|iter *10/" $T/scratch/T4_counts_F2_iter10_$PASS.out | tail -3 | tee -a $LOGF; fi
done
[ "$MODE" = warm ] && { log "warm pass done"; exit 0; }
# ---- summary ----
python3 - "$OUT" "$T" <<'EOF' | tee -a $LOGF
import json, sys, re, statistics as st
out, T = sys.argv[1:3]; d = json.load(open(out)); r = {k: v["seconds"] for k, v in d["runs"].items()}; summ = {}
def stamps(path, pat):
    ts = []
    for line in open(path):
        m = re.match(pat, line)
        if m: ts.append((int(m.group(2)), float(m.group(1))))
    return dict(ts)
PAT_SVI = r'\[\s*([\d.]+)s\] --- SVI-BW iteration (\d+)/'; PAT_MFS = r'\[\s*([\d.]+)s\]\s+iter (\d+)/'; PAT_ADAM = r'\[\s*([\d.]+)s\]\s+Step (\d+)/'
def block(tag, logf, pat, first, last, total, k20):
    try: t = stamps(logf, pat)
    except FileNotFoundError: return None
    iv = [t[i + 1] - t[i] for i in range(first, last) if i in t and i + 1 in t]
    if len(iv) < 5: return None
    med, mean = st.median(iv), st.mean(iv); startup = t.get(first, 0.0) + (r.get(k20, 0) - r.get(k20, 0))  # time to the first timed iteration header
    summ[tag] = dict(per_iter_median=med, per_iter_mean=mean, per_iter_max=max(iv), n_intervals=len(iv), startup=startup, total_h=(startup + total * med) / 3600)
    return med, mean, startup, len(iv), max(iv)
print("== uncontested per-iteration costs (GPU 0 alone; median of the trainer's iteration intervals, iterations 2..80) ==")
for tag in ["d1f1", "d2f1", "d3f1", "d4f1", "d5f1", "d1f2", "d2f2"]:
    x = block(tag, f"{T}/scratch/T2_{tag}_iter80_timed.out", PAT_SVI, 1, 80, 1620, f"T2_{tag}_iter20_timed")
    if x: print(f"{tag:12s} median {x[0]:6.2f}s/it, mean {x[1]:6.2f}, max {x[4]:6.1f} (n={x[3]}); startup {x[2]:5.0f}s; 1,620 it ~ {summ[tag]['total_h']:5.2f} h")
for F in (2, 3):
    tag = f"mixfrags_F{F}"; x = block(tag, f"{T}/scratch/mfs{F}_80.log", PAT_MFS, 1, 80, 1620, f"T2_{tag}_iter20_timed")
    if x: print(f"{tag:12s} median {x[0]:6.2f}s/it, mean {x[1]:6.2f}, max {x[4]:6.1f} (n={x[3]}); startup {x[2]:5.0f}s; 1,620 it ~ {summ[tag]['total_h']:5.2f} h")
for n in (2, 3):
    tag = f"adam_d{n}f1"; x = block(tag, f"{T}/scratch/T3_{tag}_steps125_timed.out", PAT_ADAM, 1, 124, 2000, f"T3_{tag}_steps25_timed")
    if x: print(f"{tag:12s} median {x[0]:6.2f}s/step, mean {x[1]:6.2f}, max {x[4]:6.1f} (n={x[3]}); startup {x[2]:5.0f}s; 2,000 steps ~ {summ[tag]['total_h']:5.2f} h")
d["summary"] = summ; json.dump(d, open(out, "w"), indent=1)
EOF
grep -h "timed pass" $T/scratch/T1_*_timed.out | sed 's/^/timed-pass (timed run): /' | tee -a $LOGF
echo "$(date -Is) gpu0WT end uncontested wall-clock timing session" >> $Q
log "done: $OUT"
