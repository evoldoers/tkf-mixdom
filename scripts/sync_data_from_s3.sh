#!/bin/bash
# Pull canonical datasets + recent checkpoints from S3 into the local
# checkout. Run after a fresh clone or when you suspect local data is
# stale. Idempotent: skips files that already match S3 (mtime + size).
#
# Required: AWS_PROFILE=tkf-gpu (or any profile with read access to
# s3://tkf-mixdom-gpu-618647024028/).

set -e
PROFILE="${AWS_PROFILE:-tkf-gpu}"
BUCKET="s3://tkf-mixdom-gpu-618647024028"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[sync] using AWS_PROFILE=$PROFILE, bucket=$BUCKET, repo=$REPO_ROOT"

# ── Per-family derived data (gamma labels, etc.) ──
mkdir -p "$REPO_ROOT/python/gamma_labels"
mkdir -p "$REPO_ROOT/python/gamma_labels_G6"
AWS_PROFILE=$PROFILE aws s3 sync "$BUCKET/datasets/gamma_labels/" \
  "$REPO_ROOT/python/gamma_labels/"
AWS_PROFILE=$PROFILE aws s3 sync "$BUCKET/datasets/gamma_labels_G6/" \
  "$REPO_ROOT/python/gamma_labels_G6/"

# ── BAliBASE held-one-out alignments (consumed by
#    experiments/balibase_reconstruction_benchmark.py) ──
mkdir -p "$REPO_ROOT/python/experiments/balibase_recon_alignments"
AWS_PROFILE=$PROFILE aws s3 sync \
  "$BUCKET/datasets/balibase_recon_alignments/" \
  "$REPO_ROOT/python/experiments/balibase_recon_alignments/"

# ── Bio-datasets (Pfam MSAs, BAliBase, OXBench, trees) ──
mkdir -p "$HOME/bio-datasets/data"
AWS_PROFILE=$PROFILE aws s3 sync "$BUCKET/bio-datasets/" "$HOME/bio-datasets/"

# ── Recent checkpoints (small; ~tens of MB) ──
mkdir -p "$REPO_ROOT/python/pfam"
AWS_PROFILE=$PROFILE aws s3 sync "$BUCKET/checkpoints/pfam/" \
  "$REPO_ROOT/python/pfam/"

echo "[sync] done."
