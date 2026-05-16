#!/usr/bin/env python3
"""§6 reconstruction summary: F1 across 4 datasets x 3 methods (Fitch / varanc tied / varanc untied).

Reads the 4 _tied_vs_untied.json files and produces:
  (a) Summary table (mean / median F1 per (dataset, method))
  (b) Per-dataset distribution plot (boxplot or violin, F1 by method)
  (c) Pairwise per-family scatter (untied F1 vs tied F1, color by dataset)
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASETS = ['unified_short_test', 'unified_long_test',
            'unified_hard_test', 'unified_xhard_test']
METHODS = ['fitch', 'varanc', 'varanc_untied']
FIG_DIR = 'experiments/figures'

def load(ds):
    p = f'experiments/varanc_presence_{ds}_tied_vs_untied.json'
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

def collect_f1s(d):
    out = {m: [] for m in METHODS}
    for e in d.get('results', d):
        if not e.get('methods'):
            continue
        for m in METHODS:
            v = e['methods'].get(m, {})
            if 'f1' in v:
                out[m].append(v['f1'])
    return out

# Print summary
print(f'{"":<22} {"n":>4}', *[f'{m:>10}'.replace('_', '-') for m in METHODS])
all_data = {}
for ds in DATASETS:
    d = load(ds)
    if d is None:
        print(f'{ds:<22} (not found)')
        continue
    f1s = collect_f1s(d)
    all_data[ds] = f1s
    n = len(f1s['fitch'])
    means = [np.mean(f1s[m]) if f1s[m] else float('nan') for m in METHODS]
    medians = [np.median(f1s[m]) if f1s[m] else float('nan') for m in METHODS]
    print(f'{ds:<22} {n:>4}', *[f'{x:.4f}' for x in means], '(mean)')
    print(f'{"":<22} {"":>4}', *[f'{x:.4f}' for x in medians], '(median)')

# Plot (a) box per dataset
fig, axes = plt.subplots(1, len(DATASETS), figsize=(4*len(DATASETS), 5),
                           sharey=True)
for ax, ds in zip(axes, DATASETS):
    if ds not in all_data: continue
    f1s = all_data[ds]
    data = [f1s[m] for m in METHODS]
    bp = ax.boxplot(data, labels=['fitch', 'varanc\n(tied)', 'varanc\n(untied)'],
                       patch_artist=True, showmeans=True)
    for patch, c in zip(bp['boxes'], ['lightgray', 'lightblue', 'lightcoral']):
        patch.set_facecolor(c)
    ax.set_title(ds.replace('_', ' '))
    ax.set_ylim(0.4, 1.02); ax.grid(alpha=0.3)
axes[0].set_ylabel('F1 (per held-out leaf prediction)')
fig.suptitle('§6 reconstruction F1: Fitch vs varanc (tied) vs varanc (untied)',
              y=1.02)
fig.tight_layout()
os.makedirs(FIG_DIR, exist_ok=True)
fig.savefig(f'{FIG_DIR}/sec6_recon_box.pdf', dpi=150, bbox_inches='tight')
print(f'\nSaved {FIG_DIR}/sec6_recon_box.pdf')

# Plot (b) untied vs tied scatter
fig, ax = plt.subplots(figsize=(6, 6))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for ds, c in zip(DATASETS, colors):
    if ds not in all_data: continue
    tied = np.array(all_data[ds]['varanc'])
    untied = np.array(all_data[ds]['varanc_untied'])
    n = min(len(tied), len(untied))
    ax.scatter(tied[:n], untied[:n], alpha=0.5, s=20, label=ds, color=c)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
ax.set_xlim(0.4, 1.02); ax.set_ylim(0.4, 1.02)
ax.set_xlabel('F1 — varanc tied (E, 2)')
ax.set_ylabel('F1 — varanc untied (E, L, 2)')
ax.set_title('Per-family: tied vs untied F1')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/sec6_tied_vs_untied_scatter.pdf', dpi=150, bbox_inches='tight')
print(f'Saved {FIG_DIR}/sec6_tied_vs_untied_scatter.pdf')
