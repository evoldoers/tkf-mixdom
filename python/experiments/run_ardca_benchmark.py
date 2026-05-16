#!/usr/bin/env python3
"""Python wrapper for the ArDCA ancestral reconstruction benchmark.

Calls the Julia script `ardca_benchmark.jl` as a subprocess and
processes the output JSON.

Usage:
    uv run python experiments/run_ardca_benchmark.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
JULIA_SCRIPT = SCRIPT_DIR / "ardca_benchmark.jl"
RESULTS_FILE = SCRIPT_DIR / "ardca_benchmark_results.json"

# Comparison baselines
BASELINES = {
    "Felsenstein LG08": 0.543,
    "Felsenstein C10":  0.543,
    "Felsenstein C20":  0.543,
    "CARABS":           0.713,
}


def find_julia():
    """Find the Julia executable."""
    julia_paths = [
        os.path.expanduser("~/.juliaup/bin/julia"),
        "julia",
    ]
    for p in julia_paths:
        try:
            result = subprocess.run(
                [p, "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print(f"Julia: {result.stdout.strip()} at {p}")
                return p
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise FileNotFoundError("Julia not found")


def run_julia_benchmark():
    """Run the Julia ArDCA benchmark script."""
    julia = find_julia()

    env = os.environ.copy()
    julia_dir = os.path.dirname(julia)
    env["PATH"] = julia_dir + ":" + env.get("PATH", "")

    print(f"Running: {julia} {JULIA_SCRIPT}")
    print("=" * 60)

    t0 = time.time()
    proc = subprocess.Popen(
        [julia, str(JULIA_SCRIPT)],
        cwd=str(SCRIPT_DIR.parent),  # python/ directory
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream output
    for line in proc.stdout:
        print(line, end="", flush=True)

    proc.wait()
    elapsed = time.time() - t0

    print("=" * 60)
    print(f"Julia process exited with code {proc.returncode}")
    print(f"Elapsed: {elapsed:.1f}s")

    return proc.returncode


def load_and_report():
    """Load results JSON and print comparison table."""
    if not RESULTS_FILE.exists():
        print(f"Results file not found: {RESULTS_FILE}")
        return None

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    n = results.get("n_families", 0)
    mean_acc = results.get("mean_accuracy", float("nan"))
    median_acc = results.get("median_accuracy", float("nan"))
    std_acc = results.get("std_accuracy", float("nan"))

    print()
    print("=" * 60)
    print("ANCESTRAL RECONSTRUCTION BENCHMARK COMPARISON")
    print("=" * 60)
    print(f"{'Method':<25} {'Accuracy':>10}")
    print("-" * 35)

    for name, acc in BASELINES.items():
        print(f"{name:<25} {acc*100:>9.1f}%")

    if n > 0:
        print(f"{'ArDCA (this run)':<25} {mean_acc*100:>9.1f}%")
    else:
        print("ArDCA: no results")

    print("-" * 35)
    if n > 0:
        print(f"\nArDCA details ({n} families):")
        print(f"  Mean:   {mean_acc*100:.1f}%")
        print(f"  Median: {median_acc*100:.1f}%")
        print(f"  Std:    {std_acc*100:.1f}%")
        print(f"  Min:    {results.get('min_accuracy', float('nan'))*100:.1f}%")
        print(f"  Max:    {results.get('max_accuracy', float('nan'))*100:.1f}%")

        # Per-family details
        families = results.get("families", [])
        if families:
            print(f"\nPer-family results:")
            print(f"  {'Family':<12} {'Seqs':>5} {'Cols':>5} {'Acc':>8} {'BL':>8}")
            for fam in families:
                print(f"  {fam['family']:<12} {fam['n_seqs']:>5} {fam['n_cols']:>5} "
                      f"{fam['accuracy']*100:>7.1f}% {fam.get('holdout_branch_length', 0):>8.4f}")

    print("=" * 60)
    return results


def main():
    rc = run_julia_benchmark()
    results = load_and_report()

    if rc != 0:
        print(f"\nWarning: Julia process exited with code {rc}")
        sys.exit(rc)

    return results


if __name__ == "__main__":
    main()
