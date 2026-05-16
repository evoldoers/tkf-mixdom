#!/bin/sh
# Build the TKF (tkf.tex) and MixDom (mixdom.tex) papers, showing only
# warnings/errors from pdflatex and biber.
# Proceeds through errors (no &&) so all stages run regardless.

# Change to the directory containing this script (tkf/)
cd "$(dirname "$0")"

build_paper() {
  paper="$1"
  echo "===== building ${paper}.tex ====="
  pdflatex -interaction=nonstopmode "${paper}.tex" 2>&1 \
    | grep -E '^!|^l\.|LaTeX Warning|Package .* Warning|undefined|multiply.defined'
  biber "${paper}" 2>&1 | grep -E 'WARN|ERROR|FATAL'
  pdflatex -interaction=nonstopmode "${paper}.tex" 2>&1 \
    | grep -E '^!|^l\.|undefined|multiply.defined'
  pdflatex -interaction=nonstopmode "${paper}.tex" 2>&1 \
    | grep -E '^!|^l\.|undefined|multiply.defined'
}

build_paper tkf
build_paper mixdom

open tkf.pdf
open mixdom.pdf
