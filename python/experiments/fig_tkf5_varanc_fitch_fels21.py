#!/usr/bin/env python3
"""TKF paper §5 figure: VarAnc vs Fitch vs gap-augmented LG08 (Fels21).

Section sec:results-varanc-vs-fitch.  Aggregates the existing per-method
benchmark JSONs into a paper-ready table + bar chart with bootstrap CIs.

Results sources:
    experiments/varanc_presence_<spec>_test.json
    contain entries per family with a 'methods' sub-dict keyed by method
    name (varanc, fitch, fels21, best_p) with per-entry F1 / precision /
    recall / time / logp.

Output: experiments/figures/tkf5_varanc_fitch_fels21.pdf
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


SPECS = ['unified_short', 'unified_hard', 'unified_xhard']
METHODS = ['varanc', 'fitch', 'fels21', 'best_p']
METHOD_LABELS = {
    'varanc': 'VarAnc (TKF92)',
    'fitch': 'Fitch parsimony',
    'fels21': 'gap-LG08 (Fels21)',
    'best_p': 'best-of-others',
}
METHOD_COLORS = {
    'varanc': '#1f77b4',
    'fitch': '#ff7f0e',
    'fels21': '#2ca02c',
    'best_p': '#7f7f7f',
}


def load_method_f1(spec):
    """Return dict[method -> list[F1]] for a given spec."""
    path = (f'experiments/varanc_presence_{spec}_test.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for entry in d['results']:
        for m, stats in entry.get('methods', {}).items():
            if 'f1' in stats:
                out.setdefault(m, []).append(stats['f1'])
    return out


def bootstrap_ci(samples, n_boot=2000, ci=0.95, seed=0):
    """Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples)
    if len(samples) == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, len(samples), len(samples))
        boot_means[k] = samples[idx].mean()
    lo = float(np.quantile(boot_means, (1 - ci) / 2))
    hi = float(np.quantile(boot_means, 1 - (1 - ci) / 2))
    return float(samples.mean()), lo, hi


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)

    # Aggregate.
    rows = []
    for spec in SPECS:
        data = load_method_f1(spec)
        for m in METHODS:
            if m not in data:
                continue
            mean, lo, hi = bootstrap_ci(data[m])
            rows.append({
                'spec': spec, 'method': m,
                'mean_f1': mean, 'lo': lo, 'hi': hi,
                'n': len(data[m]),
            })

    print(f'{"Method":<25} {"Spec":<18} {"Mean F1":<10} {"95% CI":<22} {"N":<5}')
    print('-' * 85)
    for r in rows:
        ci_str = f'[{r["lo"]:.3f}, {r["hi"]:.3f}]'
        print(f'{METHOD_LABELS[r["method"]]:<25} {r["spec"]:<18} '
              f'{r["mean_f1"]:.4f}    {ci_str:<22} {r["n"]:<5}')

    # Bar chart with error bars.
    fig, ax = plt.subplots(figsize=(11, 6))
    n_methods = len(METHODS)
    n_specs = len(SPECS)
    bar_width = 0.8 / n_methods
    x_base = np.arange(n_specs)
    for k, m in enumerate(METHODS):
        means = []; lo_err = []; hi_err = []; ns = []
        for spec in SPECS:
            entry = next((r for r in rows
                            if r['method'] == m and r['spec'] == spec), None)
            if entry is None:
                means.append(0)
                lo_err.append(0)
                hi_err.append(0)
                ns.append(0)
            else:
                means.append(entry['mean_f1'])
                lo_err.append(entry['mean_f1'] - entry['lo'])
                hi_err.append(entry['hi'] - entry['mean_f1'])
                ns.append(entry['n'])
        offset = (k - (n_methods - 1) / 2) * bar_width
        bars = ax.bar(x_base + offset, means, bar_width,
                        yerr=[lo_err, hi_err],
                        label=METHOD_LABELS[m], color=METHOD_COLORS[m],
                        alpha=0.85, capsize=3,
                        edgecolor='k', linewidth=0.6)
        for x, mean_val, n in zip(x_base + offset, means, ns):
            if n > 0:
                ax.text(x, mean_val + 0.01, f'n={n}',
                          ha='center', va='bottom', fontsize=7, alpha=0.7)
    ax.set_xticks(x_base)
    ax.set_xticklabels([s.replace('_', '\n') for s in SPECS])
    ax.set_ylabel('Indel-presence F1 score')
    ax.set_title(
        'TKF §5: ancestral indel reconstruction. '
        'VarAnc, Fitch, gap-augmented LG08 across difficulty tiers '
        '(error bars: 95% bootstrap CI).')
    ax.legend(loc='lower left')
    ax.set_ylim(0.7, 1.0)
    ax.grid(alpha=0.3, axis='y')
    out_path = os.path.join(out_dir, 'tkf5_varanc_fitch_fels21.pdf')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
