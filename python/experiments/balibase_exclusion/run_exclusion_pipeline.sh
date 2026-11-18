#!/bin/bash
# Wait for the Pfam HMM download and the training FASTA export, then run the
# homology search and write the filtered corpus pfam/precompiled_v3_train_nobali.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu
LOG=experiments/em_vs_gradient/results_rest_queue.log
( until [ -f ~/bio-datasets/data/pfam/hmm/download.ok ] && grep -q 'wrote' experiments/balibase_exclusion/export_train_fasta.log 2>/dev/null; do sleep 60; done
  echo "$(date -Is) cpuX start balibase homology search"
  .venv/bin/python -u experiments/balibase_exclusion/find_balibase_homologs.py > experiments/balibase_exclusion/find_balibase_homologs.log 2>&1
  rc=$?; echo "$(date -Is) cpuX end balibase homology search rc=$rc"
  .venv/bin/python -u experiments/balibase_exclusion/filter_precompiled.py pfam/precompiled_v3_train pfam/precompiled_v3_train_nobali experiments/balibase_exclusion/exclude_families.json > experiments/balibase_exclusion/filter_precompiled.log 2>&1
  rc=$?; [ $rc -eq 0 ] && [ -f pfam/precompiled_v3_train_nobali/manifest.json ] && echo "$(date -Is) cpuX end filter corpus OK $(tail -1 experiments/balibase_exclusion/filter_precompiled.log)" || echo "$(date -Is) cpuX filter corpus FAILED rc=$rc" ) >> $LOG 2>&1 &
echo armed
