#!/usr/bin/env bash
# Download the result files the paper-1 tables are built from (a release asset of this repository) and
# unpack them into this checkout at the paths listed in paper_tables_inputs.txt. Checks the archive's SHA-256.
# Usage: python/experiments/figs/fetch_table_inputs.sh
set -euo pipefail
TAG=results/paper1-tables-2026-11-18
SHA256=866f2773b140b09e55cb9aa66e64299101e55a5eff6ed9119004e4462392d60c
URL="https://github.com/evoldoers/tkf-mixdom/releases/download/$TAG/paper1-table-inputs.tar.gz"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
curl -fsSL "$URL" -o "$TMP/inputs.tar.gz"
echo "$SHA256  $TMP/inputs.tar.gz" | sha256sum -c --quiet -
tar -xzf "$TMP/inputs.tar.gz" -C "$ROOT"
echo "unpacked $(tar -tzf "$TMP/inputs.tar.gz" | wc -l) files into $ROOT"
