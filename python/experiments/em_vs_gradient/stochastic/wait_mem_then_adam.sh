#!/bin/bash
# Run the stochastic Adam job on whichever GPU first has >= 6 GB free.
cd "$(dirname "$0")/../../.."
while true; do
  for g in 0 1; do
    free=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $g | awk -F', ' '{print $2-$1}')
    if [ "$free" -ge 6000 ]; then
      echo "$(date -Is) stochastic adam on gpu$g (free ${free} MiB)"
      MODES=adam_tkf92 CUDA_VISIBLE_DEVICES=$g experiments/em_vs_gradient/stochastic/run_pfam_stochastic.sh
      echo "$(date -Is) stochastic adam done rc=$?"; exit 0
    fi
  done
  sleep 120
done
