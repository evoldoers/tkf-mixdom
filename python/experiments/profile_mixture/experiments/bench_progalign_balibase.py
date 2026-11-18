"""Benchmark the statistical progressive aligner on BAliBASE 3 (bali3pdbm, 120 families).

Methods (see src/progalign): <indel>_<subst>_<decode>[_bound]
  indel  tkf92 | mixfrag (MixFrag F=2), Pfam-trained parameters of paper 1
  subst  lg (LG) | c20 (LG exchangeabilities with the C20 profile mixture)
  decode viterbi | mea (expected-SPS posterior decoding at each merge)
  _bound  use the O(K+A) single-jump bound (re-estimated clades) instead of exact per-class pruning
  _sparse use the sparse-class lower bound (exact partials, 5K + |S|A per cell)
  _pp     posterior-pair score (mixture posterior tops, class-marginal pair ratio; O(A) per cell)
  _le     derived log-expectation score on soft-Fitch summary tops (no pruning; O(A) per cell)
  _r<N>   N rounds of tree-dependent iterative refinement (expected-SPS realignment of each split)
plus muscle (5.3, default), mafft (FFT-NS-2, the default strategy) and mafft_auto (mafft --auto).
Scores: SP and TC on reference core columns (tkfmixdom.util.msa_benchmark.sp_tc_score).
Idempotent: rows already in the CSV are skipped.

Usage:
  JAX_ENABLE_X64=1 python3.12 experiments/bench_progalign_balibase.py --methods tkf92_lg_viterbi mafft
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import progalign  # noqa: E402,F401  (x64, tkfmixdom path)
from progalign.align import align  # noqa: E402
from tkfmixdom.util.msa_benchmark import parse_fasta, sp_tc_score  # noqa: E402

BALI = Path.home() / "bio-datasets" / "data" / "balibase" / "bali3pdbm"
OUT = ROOT / "results" / "progalign_balibase"
FIELDS = ["family", "method", "n_seqs", "sp", "tc", "seconds", "error"]


def progalign_kwargs(method):
    parts = method.split("_")
    indel = {"tkf92": "tkf92", "mixfrag": "mixfrag_F2"}[parts[0]]
    subst = {"lg": "LG", "c20": "C20"}[parts[1]]
    decode = parts[2]
    rest = parts[3:]
    rounds = [int(x[1:]) for x in rest if x.startswith("r") and x[1:].isdigit()]
    rest = [x for x in rest if not (x.startswith("r") and x[1:].isdigit())]
    scoring = rest[0] if rest else "exact"
    return dict(subst=subst, indel=indel, decode=decode, scoring=scoring,
                reestimate=(scoring == "bound"), refine_rounds=rounds[0] if rounds else 0)


def run_external(method, names, seqs):
    with tempfile.TemporaryDirectory() as d:
        inp, out = Path(d) / "in.fa", Path(d) / "out.fa"
        inp.write_text("".join(f">{n}\n{s}\n" for n, s in zip(names, seqs)))
        if method == "muscle":
            cmd = [str(Path.home() / ".local/bin/muscle"), "-align", str(inp), "-output", str(out)]
            subprocess.run(cmd, check=True, capture_output=True)
        elif method == "mafft":
            with open(out, "w") as fh:
                subprocess.run(["mafft", "--retree", "2", "--quiet", str(inp)], check=True, stdout=fh)
        elif method == "mafft_auto":
            with open(out, "w") as fh:
                subprocess.run(["mafft", "--auto", "--quiet", str(inp)], check=True, stdout=fh)
        else:
            raise ValueError(method)
        return parse_fasta(str(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--csv", default=str(OUT / "results.csv"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if os.path.exists(args.csv):
        with open(args.csv) as fh:
            done = {(r["family"], r["method"]) for r in csv.DictReader(fh) if not r["error"]}
    fams = args.families or sorted(os.listdir(BALI / "in"))
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for fam in fams:
            inp = parse_fasta(str(BALI / "in" / fam))
            ref = parse_fasta(str(BALI / "ref" / fam))
            names = list(inp)
            seqs = [inp[n].replace("-", "").replace(".", "") for n in names]
            for m in args.methods:
                if (fam, m) in done:
                    continue
                t0 = time.time()
                row = dict(family=fam, method=m, n_seqs=len(seqs), sp="", tc="", seconds="", error="")
                try:
                    if m in ("muscle", "mafft", "mafft_auto"):
                        aln = run_external(m, names, seqs)
                    else:
                        aln = align(names, seqs, **progalign_kwargs(m))
                    assert all(aln[n].replace("-", "").upper() == s.upper() for n, s in zip(names, seqs))
                    sp, tc = sp_tc_score(aln, ref, core_only=True)
                    row.update(sp=f"{sp:.6f}", tc=f"{tc:.6f}")
                    adir = OUT / "aln" / m
                    adir.mkdir(parents=True, exist_ok=True)
                    (adir / f"{fam}.fa").write_text("".join(f">{n}\n{aln[n]}\n" for n in names))
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {e}"[:300]
                row["seconds"] = f"{time.time() - t0:.1f}"
                w.writerow(row); fh.flush()
                print(fam, m, row["sp"], row["tc"], row["seconds"], row["error"], flush=True)


if __name__ == "__main__":
    main()
