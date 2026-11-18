#!/bin/bash
# The fitted-clock re-timing of tkf92_K20tied on BAliBASE lost 10 large families to GPU OOM (BB50002-BB50015, whole-
# family batches). Re-run them with a small pair chunk, merge into the main result (merge_retry.py), re-decode, and
# regenerate the tables. Waits for the uncontested wall-clock timing session to end first. Usage: <gpu>
cd "$(dirname "$0")/../.."
GPU=${1:-0}; EB=experiments/expected_balibase; Q=experiments/em_vs_gradient/results_rest_queue.log; P=gpu${GPU}KR
N=tkf92_K20tied; MAIN=$EB/expected_balibase_${N}_withsps.json; RETRY=$EB/expected_balibase_${N}_withsps_retry_fitted.json
FAMS=BB50002,BB50004,BB50005,BB50006,BB50007,BB50008,BB50012,BB50013,BB50014,BB50015
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
( until grep -q "end uncontested wall-clock timing session" $Q; do sleep 300; done
  echo "$(date -Is) $P start retry bench bali3pdbm $N ($FAMS, pair-chunk 4)"
  .venv/bin/python -u experiments/expected_pairwise_balibase.py --method tkf92_mixture --method-name $N --params pfam/tkf92_mixture_K20_train_tiedsubst.npz --sitemix pfam/cherryml_mixture_C20_paircol.npz --anchor experiments/tkf92_anchor_fitted.json --families $FAMS --pair-chunk 4 --fsa on --fsa-sps --balibase-dir $HOME/bio-datasets/data/balibase/bali3pdbm --out $RETRY > ${RETRY%.json}.log 2>&1
  echo "$(date -Is) $P end retry bench rc=$?"
  python3 experiments/benchmarks/merge_retry.py $MAIN $RETRY && echo "$(date -Is) $P merged retry into $MAIN"
  mv $EB/msa_${N}_progressive_c0_r0.json $EB/msa_${N}_progressive_c0_r0_partial110.json 2>/dev/null
  JAX_PLATFORMS=cpu .venv/bin/python experiments/fsa_gap_factor_from_cache.py --method $N --balibase-dir $HOME/bio-datasets/data/balibase/bali3pdbm --out $EB/msa_${N}_progressive_c0_r0.json --algo progressive --gap-factors 0 --consistency-iters 0 --refine-iters 0 > $EB/msa_${N}_progressive_c0_r0.log 2>&1
  echo "$(date -Is) $P decode bali3pdbm $N rc=$? $(python3 -c "import json;d=json.load(open('$EB/msa_${N}_progressive_c0_r0.json'));print(d['n_families'],d['summary'])" 2>/dev/null | cut -c1-120)"
  .venv/bin/python experiments/figs/paper_tables.py > experiments/timing/scratch/paper_tables_after_retry.out 2>&1; echo "$(date -Is) $P tables regenerated rc=$?"
  echo "$(date -Is) $P end K20tied retry chain" ) >> $Q 2>&1 &
