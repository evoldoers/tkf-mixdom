#!/bin/bash
# Stage 3: for each of the 12 grid points, train Adam-TKF92 + Adam-GGI.
# Adam-TKF92 cold-start from default init.  Adam-GGI cold-start from
# the TKF92→GGI projection of the grid point's truth (= the same
# projection used to make the simulator).  This is the "fair" warmstart:
# the optimizer starts in the GGI region that the data was generated from.
#
# Runs both Adams on different GPUs in parallel per grid point.
# Total: 12 cells × ~20 min wall (Adam-TKF92 ~20 min on GPU 0
# || Adam-GGI ~20 min on GPU 1) ≈ ~4 hours on 2 GPUs.

cd /home/yam/tkf-mixdom/python

LOG=experiments/2dfb_sim_grid/fit_chain.log
META=experiments/2dfb_sim_grid/grid_meta.json
echo "$(date) fit chain start, waiting for grid_meta + all cells" > $LOG
until [ -f $META ]; do sleep 30; done

# Read grid count
N_CELLS=$(python3 -c "import json; print(len(json.load(open('$META'))['grid_points']))")
echo "$(date) META found; n_cells=$N_CELLS" >> $LOG

for i in $(seq 0 $((N_CELLS - 1))); do
  CELL=$(printf 'grid%02d' $i)
  CELL_DIR=experiments/2dfb_sim_grid/$CELL
  echo "$(date) [$CELL] waiting for sim data" >> $LOG
  until [ -f $CELL_DIR/train.pkl ] && [ -f $CELL_DIR/val.pkl ]; do sleep 30; done
  echo "$(date) [$CELL] sim data ready, running Adam-TKF92 || Adam-GGI" >> $LOG

  # Read grid params for warmstart projection
  read LAM MU EXT GGI_LAM0 GGI_MU0 GGI_RHO GGI_X <<<"$(python3 -c "
import json
d = json.load(open('$META'))
pt = next(p for p in d['grid_points'] if p['idx'] == $i)
g = pt['ggi']
print(f\"{pt['lam']:.6f} {pt['mu']:.6f} {pt['ext']:.6f} {g['lam0']:.6f} {g['mu0']:.6f} {g['rho']:.6f} {g['x']:.6f}\")
")"

  # Adam-TKF92 (cold default init)
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run python -u \
      experiments/run_tkf92_2dfb_pfam.py \
      --mode adam_tkf92 \
      --sim-train-file $CELL_DIR/train.pkl \
      --sim-val-file   $CELL_DIR/val.pkl \
      --batch-size 16 --n-iter 800 --patience 100 \
      --bin-bucketed --pre-warm --no-command-buffers --max-pad-cap 256 \
      --out $CELL_DIR/adam_tkf92.json \
      > $CELL_DIR/adam_tkf92.log 2>&1 &
  PID_T=$!

  # Adam-GGI (warmstart at GGI truth projection — same params the sim used)
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run python -u \
      experiments/run_tkf92_2dfb_pfam.py \
      --mode adam_ggi \
      --sim-train-file $CELL_DIR/train.pkl \
      --sim-val-file   $CELL_DIR/val.pkl \
      --init-mu0 $GGI_MU0 --init-rho $GGI_RHO --init-x $GGI_X \
      --ggi-segment upper \
      --batch-size 16 --n-iter 800 --patience 100 \
      --bin-bucketed --pre-warm --no-command-buffers --max-pad-cap 256 \
      --out $CELL_DIR/adam_ggi.json \
      > $CELL_DIR/adam_ggi.log 2>&1 &
  PID_G=$!

  wait $PID_T $PID_G
  echo "$(date) [$CELL] both Adam fits done" >> $LOG
done

echo "$(date) all $N_CELLS cells fit; chain done" >> $LOG
