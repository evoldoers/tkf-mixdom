#!/bin/bash
# Push freshly-written experiment results (JSONs + logs) and any new
# checkpoints in python/pfam/ to S3. Run after a benchmark or training
# run completes. Result JSONs are versioned by commit hash so we can
# always reproduce which code wrote which numbers.
#
# Required: AWS_PROFILE=tkf-gpu (or any profile with write access to
# s3://tkf-mixdom-gpu-618647024028/).

set -e
PROFILE="${AWS_PROFILE:-tkf-gpu}"
BUCKET="s3://tkf-mixdom-gpu-618647024028"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="${RUN_ID:-$(cd "$REPO_ROOT" && git rev-parse --short HEAD)-$(date +%Y%m%d-%H%M%S)}"

echo "[upload] using AWS_PROFILE=$PROFILE, RUN_ID=$RUN_ID"

# ── Result JSONs + per-run logs ──
if [ -d "$REPO_ROOT/python/experiments" ]; then
  AWS_PROFILE=$PROFILE aws s3 sync \
    "$REPO_ROOT/python/experiments/" \
    "$BUCKET/results/$RUN_ID/" \
    --exclude "*" --include "*.json" --include "*.log" --include "*.tsv"
fi

# ── Checkpoints (mirror; intent: keep S3 as the canonical store) ──
if [ -d "$REPO_ROOT/python/pfam" ]; then
  AWS_PROFILE=$PROFILE aws s3 sync \
    "$REPO_ROOT/python/pfam/" \
    "$BUCKET/checkpoints/pfam/"
fi

echo "[upload] done. RUN_ID = $RUN_ID"
