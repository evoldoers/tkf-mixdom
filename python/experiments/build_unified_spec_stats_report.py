#!/usr/bin/env python3
"""Build a Markdown report summarising the unified test-spec families.

Produces a comparison table over the four specs (short / long / hard /
xhard) and one representative truncated alignment from each. Rebuilds
deterministically from the spec JSONs + Pfam Stockholm + Newick files
on disk; no benchmark results required.

Per-family stats reported (mean / median / min / max over each spec):
  - n_leaves         number of leaves in the spec-pruned tree
                       (held_out + remaining; xhard drops K nearest)
  - true_len         residue count of the held-out target sequence
  - n_cols           alignment width in columns
  - tot_branch       sum of branch lengths in the spec-pruned tree
  - min_nn_dist      tree distance from held-out target to nearest
                       leaf still in `remaining` (post-drop for xhard)
  - col_entropy      mean per-column entropy of the residue distribution
                       (bits, gaps excluded from the count)
  - gap_fraction     fraction of (leaf, column) cells that are gap

Usage:
    cd python && uv run python experiments/build_unified_spec_stats_report.py

Writes `experiments/unified_spec_stats.md`.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT
from experiments.ancrec_benchmark import parse_sto
from experiments.fels21_reconstruction_benchmark import prune_tree_to_msa


SPECS = [
    ('short', 'experiments/unified_benchmark_test_spec.json'),
    ('long',  'experiments/unified_benchmark_long_test_spec.json'),
    ('hard',  'experiments/unified_benchmark_hard_test_spec.json'),
    ('xhard', 'experiments/unified_benchmark_xhard_test_spec.json'),
]
OUT_PATH = 'experiments/unified_spec_stats.md'


def total_branch_length(tree):
    total = 0.0
    def walk(node):
        nonlocal total
        if node.parent is not None:
            total += float(node.branch_length or 0.0)
        for c in node.children:
            walk(c)
    walk(tree)
    return total


def collect_leaves(tree):
    out = []
    def walk(n):
        if not n.children:
            out.append(n)
        for c in n.children:
            walk(c)
    walk(tree)
    return out


def column_entropy_and_gap(seqs, leaf_names):
    """Compute mean per-column residue entropy (bits) + gap fraction +
    all-gap-column fraction.

    Restricts to leaf_names (drops the held-out target's row so the
    entropy reflects what the predictor sees, not what it's predicting
    — but include held_out in gap_fraction since the user asked about
    %gaps in the alignment as a whole).

    Gap chars are EXCLUDED from the per-column distribution before
    entropy is computed (matches how info-theoretic conservation is
    usually reported); the gap_fraction stat reports them separately.

    `all_gap_col_frac` reports how many columns in the kept submatrix
    have NO amino acid in any kept leaf — an artifact of subsampling
    or held-out-set restriction. xhard hits this heavily because it
    drops the K nearest neighbours and caps at leaf_cap=100, so
    columns whose residues lived only in the dropped/subsampled
    rows become empty. Affects benchmark interpretation: all-gap
    columns are trivially predicted as absent and don't really test
    the model.
    """
    if not seqs or not leaf_names:
        return float('nan'), float('nan'), float('nan')
    rows = [seqs[n] for n in leaf_names if n in seqs]
    if not rows:
        return float('nan'), float('nan'), float('nan')
    L = max(len(r) for r in rows)
    rows = [r.ljust(L, '-') for r in rows]
    n_leaves = len(rows)
    n_cells = n_leaves * L
    n_gap = 0
    n_all_gap_col = 0
    entropies = []
    for j in range(L):
        col = [r[j].upper() for r in rows]
        counts = {}
        for ch in col:
            if ch in AA_TO_INT and AA_TO_INT[ch] < 20:
                counts[ch] = counts.get(ch, 0) + 1
            else:
                n_gap += 1
        n_aa = sum(counts.values())
        if n_aa == 0:
            entropies.append(float('nan'))
            n_all_gap_col += 1
        else:
            H = 0.0
            for c in counts.values():
                p = c / n_aa
                if p > 0:
                    H -= p * math.log(p, 2)
            entropies.append(H)
    mean_H = float(np.nanmean(entropies)) if entropies else float('nan')
    gap_frac = float(n_gap) / float(max(n_cells, 1))
    all_gap_frac = float(n_all_gap_col) / float(max(L, 1))
    return mean_H, gap_frac, all_gap_frac


def per_family_stats(entry, pfam_dir, tree_dir):
    fam = entry['family']
    held_out = entry['held_out']
    remaining = entry.get('remaining', [])
    spec_leaves = set(remaining) | {held_out}

    out = {
        'n_leaves': float('nan'),
        'true_len': float(entry.get('true_len', float('nan'))),
        'n_cols':   float(entry.get('n_cols', float('nan'))),
        'tot_branch': float('nan'),
        'min_nn_dist': float('nan'),
        'col_entropy': float('nan'),
        'gap_frac': float('nan'),
        'all_gap_col_frac': float('nan'),
    }

    # Tree side
    nwk_path = os.path.join(tree_dir, f'{fam}.nwk')
    pruned_tree = None
    if os.path.exists(nwk_path):
        try:
            with open(nwk_path) as f:
                tree = parse_newick(f.read().strip())
            tree_leaf_names = {l.name for l in collect_leaves(tree)}
            present = spec_leaves & tree_leaf_names
            if present:
                pruned_tree = prune_tree_to_msa(tree, present)
                out['n_leaves'] = float(len(present))
                out['tot_branch'] = float(total_branch_length(pruned_tree))
        except Exception:
            pass

    # Distance from target to nearest remaining-leaf — already in spec
    # in slightly-different keys depending on which builder produced it.
    if 'min_nn_dist_post_drop' in entry:
        out['min_nn_dist'] = float(entry['min_nn_dist_post_drop'])
    elif 'min_nn_dist' in entry:
        out['min_nn_dist'] = float(entry['min_nn_dist'])
    elif 'mean_dist' in entry:
        # short / long specs only have mean_dist (mean over ALL pairs);
        # not the same metric, but we report what the spec carries.
        out['min_nn_dist'] = float(entry['mean_dist'])

    # MSA side — restrict to spec leaves so the stats reflect the
    # benchmark's actual data exposure (xhard drops the K nearest).
    sto_path = os.path.join(pfam_dir, f'{fam}.sto')
    if os.path.exists(sto_path):
        try:
            seqs = parse_sto(sto_path)
            kept = {n: seqs[n] for n in spec_leaves if n in seqs}
            mean_H, gap_frac, all_gap_col = column_entropy_and_gap(
                kept, [n for n in remaining if n in kept])
            out['col_entropy'] = mean_H
            out['gap_frac'] = gap_frac
            out['all_gap_col_frac'] = all_gap_col
        except Exception:
            pass

    return out


def aggregate(stats_list, key):
    vals = [s[key] for s in stats_list
            if s[key] is not None and not (isinstance(s[key], float)
                                             and math.isnan(s[key]))]
    if not vals:
        return None
    return {
        'mean': float(np.mean(vals)),
        'median': float(np.median(vals)),
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'n': len(vals),
    }


def fmt_agg(d, decimals=2):
    if d is None:
        return '—'
    fmt = f'{{:.{decimals}f}}'
    return (fmt.format(d['mean']) + ' (med ' + fmt.format(d['median'])
            + ', [' + fmt.format(d['min']) + '–' + fmt.format(d['max'])
            + '])')


def truncated_alignment(seqs, leaf_order, max_leaves=8, max_cols=80):
    keep = [n for n in leaf_order if n in seqs][:max_leaves]
    if not keep:
        return '(no sequences could be loaded)'
    name_w = max(len(n) for n in keep)
    out = []
    for n in keep:
        s = seqs[n]
        out.append(f'  {n:<{name_w}}  {s[:max_cols]}')
    return '\n'.join(out)


def pick_representative(spec_entries, per_fam_stats):
    """Pick the family whose true_len is closest to the median true_len."""
    pairs = list(zip(spec_entries, per_fam_stats))
    valid = [(e, s) for e, s in pairs
             if not math.isnan(s['true_len'])]
    if not valid:
        return spec_entries[0]
    median = float(np.median([s['true_len'] for _, s in valid]))
    best = min(valid, key=lambda es: abs(es[1]['true_len'] - median))
    return best[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=OUT_PATH)
    parser.add_argument('--max-fams-per-spec', type=int, default=0,
                        help='Cap families considered per spec (0 = all). '
                        'Useful for fast iteration.')
    args = parser.parse_args()

    # All specs reference the same Pfam tree+seed dirs.
    repo_python = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sections = []
    for label, rel_path in SPECS:
        path = os.path.join(repo_python, rel_path)
        with open(path) as f:
            spec = json.load(f)
        pfam_dir = os.path.expanduser(spec['pfam_dir'])
        tree_dir = os.path.expanduser(spec['tree_dir'])
        fams = spec['families']
        if args.max_fams_per_spec > 0:
            fams = fams[:args.max_fams_per_spec]
        per_fam = [per_family_stats(e, pfam_dir, tree_dir) for e in fams]
        agg = {k: aggregate(per_fam, k) for k in (
            'n_leaves', 'true_len', 'n_cols', 'tot_branch',
            'min_nn_dist', 'col_entropy', 'gap_frac',
            'all_gap_col_frac')}
        # Count families with ≥1 all-gap col (not just the mean fraction).
        n_with_all_gap = sum(1 for s in per_fam
                              if not math.isnan(s['all_gap_col_frac'])
                              and s['all_gap_col_frac'] > 0)
        rep = pick_representative(fams, per_fam)
        sections.append({
            'label': label,
            'spec_path': rel_path,
            'spec_meta': {k: v for k, v in spec.items() if k != 'families'},
            'n_families_target': spec['n_families'],
            'n_families_kept': sum(1 for s in per_fam
                                     if not math.isnan(s['n_leaves'])),
            'n_with_all_gap': n_with_all_gap,
            'agg': agg,
            'rep': rep,
            'pfam_dir': pfam_dir,
            'tree_dir': tree_dir,
        })

    # ------------ build markdown ------------
    L = []
    L.append('# Unified test-spec statistics\n')
    L.append('Per-family aggregates over the four canonical reconstruction '
              'benchmark specs. Each cell formatted as '
              '`mean (median, [min–max])`. Tree-derived stats use the '
              'spec-pruned tree (held_out ∪ remaining; xhard additionally '
              'drops K nearest neighbours). MSA-derived stats use the '
              'same leaf set; `col_entropy` averages per-column '
              'amino-acid entropy in bits (gaps excluded from the '
              'distribution before entropy); `gap_fraction` reports '
              'gaps as a share of all (leaf, column) cells in the '
              'kept rows.\n')
    cols_order = ['short', 'long', 'hard', 'xhard']
    label_to_section = {s['label']: s for s in sections}

    rows = [
        ('Families (target)', lambda s: f'{s["n_families_target"]}'),
        ('Families (kept)',   lambda s: f'{s["n_families_kept"]}'),
        ('n_leaves',           lambda s: fmt_agg(s['agg']['n_leaves'], 1)),
        ('true_len (residues)', lambda s: fmt_agg(s['agg']['true_len'], 1)),
        ('n_cols',             lambda s: fmt_agg(s['agg']['n_cols'], 1)),
        ('total tree length',  lambda s: fmt_agg(s['agg']['tot_branch'], 2)),
        ('target→nearest dist', lambda s: fmt_agg(s['agg']['min_nn_dist'], 2)),
        ('col entropy (bits)', lambda s: fmt_agg(s['agg']['col_entropy'], 2)),
        ('gap fraction',       lambda s: fmt_agg(s['agg']['gap_frac'], 3)),
        ('all-gap col fraction', lambda s: fmt_agg(
            s['agg']['all_gap_col_frac'], 3)),
        ('families with ≥1 all-gap col',
            lambda s: f'{s["n_with_all_gap"]} / {s["n_families_kept"]}'),
    ]

    L.append('| Metric | ' + ' | '.join(cols_order) + ' |')
    L.append('|---|' + '---|' * len(cols_order))
    for label, getter in rows:
        cells = [label] + [getter(label_to_section[c]) for c in cols_order]
        L.append('| ' + ' | '.join(cells) + ' |')
    L.append('')
    L.append('Notes:')
    L.append('- short / long use `mean_dist` (mean tree distance from '
              'the held-out target to all other leaves) for the '
              '`target→nearest dist` row, since neither builder records '
              'a true nearest-neighbour distance. hard uses `min_nn_dist` '
              'over `remaining`; xhard uses `min_nn_dist_post_drop` '
              '(distance to nearest leaf AFTER dropping the K closest '
              'as the spec specifies).')
    L.append('- xhard subsamples large families to leaf_cap=100; '
              '`n_leaves` for xhard reflects (held + remaining) — '
              'i.e. usually 98, occasionally less when the source '
              'tree was already small.')
    L.append('- `all-gap col fraction` is the share of columns in the '
              'kept submatrix that have no amino acid in any kept leaf '
              '(i.e. an "empty column"). A side-effect of subsampling: '
              'when xhard drops the K nearest neighbours and caps at '
              '100 leaves, columns whose residues lived only in the '
              'dropped/subsampled rows become empty. These columns are '
              'trivially predicted as absent and don\'t exercise the '
              'model — interpret xhard headline F1 with this in mind. '
              'short / hard never show this; long has 2 of 200; xhard '
              'has it in over half of families.')
    L.append('')

    # Per-spec representative
    for label in cols_order:
        s = label_to_section[label]
        e = s['rep']
        L.append(f'## Representative `{label}` family\n')
        L.append(f'`{e["family"]}` — held_out=`{e["held_out"]}`, '
                  f'K (remaining leaves) = {len(e.get("remaining", []))}, '
                  f'n_cols = {e.get("n_cols", "?")}, '
                  f'true_len = {e.get("true_len", "?")}.')
        if 'median_div' in e:
            L.append(f'Family median pairwise tree distance = '
                      f'{e["median_div"]:.2f}.')
        if 'min_nn_dist_post_drop' in e:
            L.append(f'min target→nn (post-drop) = '
                      f'{e["min_nn_dist_post_drop"]:.2f}.')
        elif 'min_nn_dist' in e:
            L.append(f'min target→nn = {e["min_nn_dist"]:.2f}.')
        else:
            L.append(f'mean target→other = {e["mean_dist"]:.2f}.')
        if 'gap_frac' in e:
            L.append(f'held-out gap_frac = {e["gap_frac"]:.2f}.')
        if 'dropped_neighbours' in e:
            L.append(f'dropped neighbours = '
                      f'{e["dropped_neighbours"]}.')
        L.append('')
        # Truncated alignment
        sto = os.path.join(s['pfam_dir'], e['family'] + '.sto')
        try:
            seqs = parse_sto(sto)
        except Exception:
            seqs = {}
        leaf_order = [e['held_out']] + e.get('remaining', [])
        block = truncated_alignment(seqs, leaf_order,
                                       max_leaves=8, max_cols=80)
        L.append('```')
        L.append(block)
        L.append('```')
        L.append('')

    out = '\n'.join(L)
    with open(args.out, 'w') as f:
        f.write(out)
    print(f'wrote {args.out} ({len(out)} chars)')


if __name__ == '__main__':
    main()
