#!/usr/bin/env python3
"""Summarise compare_optimizers.py runs: passes and wall-clock to tolerance.

For each (corpus, model) JSON, LL* = the best per-pair LL reached by ANY
method's trajectory.  For each method and tolerance eps, report the first
pass k whose LL(theta_k) >= LL* - eps and the wall-clock at that pass, plus
the final LL and parameters.  Writes a Markdown table and a JSON summary.

Usage: summarize.py results/*.json --md results/summary.md --json results/summary.json
"""
import argparse
import glob
import json
import os

import numpy as np

EPS = [1e-2, 1e-3, 1e-4]
LABEL = {"em": "closed-form EM", "lbfgs": "L-BFGS",
         "adam_lr0.1": "Adam (lr 0.1)", "adam_lr0.01": "Adam (lr 0.01)"}
MODEL = {"tkf91": "TKF91", "tkf92": "TKF92", "mixfrag": "MixFrag (F=2)",
         "mixfrag_s": "MixFrag-S", "mixfrag_s2": "MixFrag-S (F=2)", "mixfrag_s3": "MixFrag-S (F=3)"}
CORPUS = {"sim_tkf92": "simulated (TKF92)", "sim_tkf91": "simulated (TKF91)", "sim_mixfrag": "simulated (MixFrag)", "sim_mixfrag_s": "simulated (MixFrag-S)", "pfam": "Pfam"}


def summarise(path):
    d = json.load(open(path))
    a = d["args"]
    model = a["model"]
    if model == "mixfrag_s":   # compare_optimizers.py --model mixfrag_s --n-fragtypes F -> row key mixfrag_s{F}
        model = f"mixfrag_s{a.get('n_fragtypes', d.get('n_fragtypes', 2))}"
    methods = d["methods"]
    ll_star = max(max(r["ll"] for r in tr) for tr in methods.values() if tr)
    rows = []
    for m, tr in methods.items():
        if not tr:
            continue
        lls = np.array([r["ll"] for r in tr])
        walls = np.array([r["wall"] for r in tr])
        best_i = int(np.argmax(lls))
        row = dict(method=m, label=LABEL.get(m, m), n_pass=len(tr),
                   wall_total=float(walls[-1]), ll_final=float(lls[-1]),
                   ll_best=float(lls[best_i]), gap_best=float(ll_star - lls[best_i]),
                   params_best=tr[best_i]["params"], to_tol={})
        for eps in EPS:
            hit = np.nonzero(lls >= ll_star - eps)[0]
            if len(hit):
                k = int(hit[0])
                row["to_tol"][str(eps)] = dict(passes=k + 1, wall=float(walls[k]))
            else:
                row["to_tol"][str(eps)] = None
        rows.append(row)
    return dict(path=path, corpus=a["corpus"], model=model, N=d["N"],
                ll_init=d["ll_init_per_pair"], ll_star=ll_star,
                pass_seconds=d["pass_seconds"], rows=rows)


def md_table(s):
    out = [f"### {CORPUS[s['corpus']]} corpus, {MODEL[s['model']]}  "
           f"(N={s['N']} pairs; LL/pair at init {s['ll_init']:.4f}, best {s['ll_star']:.4f}; "
           f"one pass: E-step {s['pass_seconds']['estep']:.1f}s, "
           f"gradient {s['pass_seconds']['grad']:.1f}s)", "",
           "| method | passes to 1e-2 | to 1e-3 | to 1e-4 | wall to 1e-3 (s) | "
           "passes run | best LL/pair | gap to best | parameters at best |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        def tt(eps, key="passes"):
            v = r["to_tol"][str(eps)]
            return "–" if v is None else (f"{v[key]}" if key == "passes" else f"{v[key]:.0f}")
        p = r["params_best"]
        ps = f"λ={p['lam']:.4f} μ={p['mu']:.4f}"
        if len(p["exts"]) == 1 and p["exts"][0] > 0:
            ps += f" r={p['exts'][0]:.3f}"
        elif len(p["exts"]) > 1:
            ps += " r=(" + ",".join(f"{e:.3f}" for e in p["exts"]) + ")"
            ps += " w=(" + ",".join(f"{w:.3f}" for w in p["weights"]) + ")"
        out.append(f"| {r['label']} | {tt(1e-2)} | {tt(1e-3)} | {tt(1e-4)} | "
                   f"{tt(1e-3,'wall')} | {r['n_pass']} | {r['ll_best']:.4f} | "
                   f"{r['gap_best']:.1e} | {ps} |")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    paths = sorted(p for pat in args.paths for p in glob.glob(pat))
    summaries = [summarise(p) for p in paths]
    order = {"sim_tkf91": 0, "sim_tkf92": 1, "sim_mixfrag": 2, "sim_mixfrag_s": 3, "pfam": 4}
    morder = {"tkf91": 0, "tkf92": 1, "mixfrag": 2, "mixfrag_s": 3, "mixfrag_s2": 3, "mixfrag_s3": 4}
    summaries.sort(key=lambda s: (order[s["corpus"]], morder[s["model"]]))
    md = "\n".join(md_table(s) for s in summaries)
    print(md)
    if args.md:
        open(args.md, "w").write(md)
    if args.json:
        json.dump(summaries, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
