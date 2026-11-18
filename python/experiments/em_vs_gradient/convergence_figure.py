#!/usr/bin/env python3
"""Paper-1 Table 3 + convergence figure for closed-form EM vs gradient-based optimizers, from the
trajectories stored by compare_optimizers*.py (results/, results_aligned/), the MixDom full-batch
EM / Adam logs (results_mixdom/) and the minibatch svi-BW / Adam runs (stochastic/).

Writes  <paper1>/tables/em_vs_gradient_table.tex   (whole tabular)
        <paper1>/figures/em_vs_gradient.pdf   (<paper1>: see figs/paper1_dir.py)
"""
import csv, json, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
TAG = os.environ.get('EMVG_RESULTS_TAG', '_gtr')   # the paper's runs (GTR refit); '' = the earlier results/, results_aligned/, stochastic/
MIXDOM = os.environ.get('EMVG_MIXDOM', 'd2f1')   # the MixDom model of the full-batch EM vs Adam row (results_mixdom/{em,adam}_<MIXDOM>_*)   # '' = results/, results_aligned/, stochastic/; '_gtr' = the GTR-refit reruns
import sys; sys.path.insert(0, os.path.join(HERE, '..', 'figs')); from paper1_dir import PAPER1 as PAPER
EPS = (0.1, 1e-3)
METHODS = [('em', 'EM'), ('lbfgs', 'L-BFGS'), ('adam_lr0.1', 'Adam (0.1)'), ('adam_lr0.01', 'Adam (0.01)')]


def passes_to(lls, best, eps):
    hit = np.nonzero(np.asarray(lls) >= best - eps)[0]
    return int(hit[0]) + 1 if len(hit) else None


def load_run(path):
    d = json.load(open(path))
    tr = {m: ([x['ll'] for x in v], [x['wall'] for x in v]) for m, v in d['methods'].items()}
    best = max(max(ll) for ll, _ in tr.values())
    return d, tr, best


# ------------------------------------------------------------------ full-batch rows (count-tensor settings)
ROWS = [('Unaligned', 'TKF91', 'simulated', f'results{TAG}/sim_tkf91_tkf91.json'),
        ('Unaligned', 'TKF91', '\\Pfam', f'results{TAG}/pfam_tkf91.json'),
        ('Unaligned', 'TKF92', 'simulated', f'results{TAG}/sim_tkf92_tkf92.json'),
        ('Unaligned', 'TKF92', '\\Pfam', f'results{TAG}/pfam_tkf92.json'),
        ('Unaligned', '\\MixFrag\\ $\\mathcal F{=}2$', 'simulated', f'results{TAG}/sim_mixfrag_mixfrag.json'),
        ('Unaligned', '\\MixFrag\\ $\\mathcal F{=}2$', '\\Pfam', f'results{TAG}/pfam_mixfrag.json'),
        ('Unaligned', '\\MixFrag-S $\\mathcal F{=}2$', 'simulated', f'results{TAG}/sim_mixfrag_s_mixfrag_s2.json'),
        ('Unaligned', '\\MixFrag-S $\\mathcal F{=}2$', '\\Pfam', f'results{TAG}/pfam_mixfrag_s2.json'),
        ('Unaligned', '\\MixFrag-S $\\mathcal F{=}3$', 'simulated', f'results{TAG}/sim_mixfrag_s_mixfrag_s3.json'),
        ('Unaligned', '\\MixFrag-S $\\mathcal F{=}3$', '\\Pfam', f'results{TAG}/pfam_mixfrag_s3.json'),
        ('Aligned', 'TKF92', 'simulated', f'results_aligned{TAG}/sim_tkf92_tkf92.json'),
        ('Aligned', 'TKF92', '\\Pfam', f'results_aligned{TAG}/pfam_tkf92.json'),
        ('Aligned', '\\MixFrag\\ $\\mathcal F{=}2$', 'simulated', f'results_aligned{TAG}/sim_mixfrag_mixfrag.json'),
        ('Aligned', '\\MixFrag\\ $\\mathcal F{=}2$', '\\Pfam', f'results_aligned{TAG}/pfam_mixfrag.json'),
        ('Aligned', '\\MixFrag-S $\\mathcal F{=}2$', 'simulated', f'results_aligned{TAG}/sim_mixfrag_s_mixfrag_s2.json'),
        ('Aligned', '\\MixFrag-S $\\mathcal F{=}2$', '\\Pfam', f'results_aligned{TAG}/pfam_mixfrag_s2.json'),
        ('Aligned', '\\MixFrag-S $\\mathcal F{=}3$', 'simulated', f'results_aligned{TAG}/sim_mixfrag_s_mixfrag_s3.json'),
        ('Aligned', '\\MixFrag-S $\\mathcal F{=}3$', '\\Pfam', f'results_aligned{TAG}/pfam_mixfrag_s3.json')]
# MixFrag-S (one substitution matrix per fragtype): results{TAG}/{sim_mixfrag_s,pfam}_mixfrag_s{F}.json, same layout


def cell(lls, best, run_len):
    """Two cells: passes to within EPS[0] and within EPS[1] nats."""
    a = passes_to(lls, best, EPS[0]); b = passes_to(lls, best, EPS[1])
    def f(x):
        return str(x) if x is not None else f'$>${run_len}'
    return f'{f(a)} & {f(b)}'


NA = '\\multicolumn{2}{c}{n/a}'; DASH = '\\multicolumn{2}{c}{--}'


lines = []
for setting, model, corpus, path in ROWS:
    if not os.path.exists(path):   # a run not yet made (the MixFrag-S rows until their jobs land): '--' cells
        lines.append(f"{setting} & {model} & {corpus} & " + ' & '.join([DASH] * len(METHODS)) + ' \\\\'); continue
    d, tr, best = load_run(path)
    cells = []
    for m, _ in METHODS:
        if m not in tr: cells.append(DASH); continue
        lls, _ = tr[m]; cells.append(cell(lls, best, len(lls)))
    lines.append(f"{setting} & {model} & {corpus} & " + ' & '.join(cells) + ' \\\\')

# ------------------------------------------------------------------ MixDom (alignment given, 1D FB; MAP objective per pair)
def map_trace(path):
    return [float(x) / 1000 for x in re.findall(r'MAP_LL=(-[\d.]+)', open(path).read())]


em_md = map_trace(f'results_mixdom/em_{MIXDOM}_fixed.log')[:200] + map_trace(f'results_mixdom/em_{MIXDOM}_fixed_2000.log')
adam_md = {lr: map_trace(f'results_mixdom/adam_{MIXDOM}_lr{lr}_long.log') for lr in ('0.1', '0.01')}
em_star = max(em_md)   # EM's stationary value, the reference for this row (Adam 0.1 transiently exceeds it; see text)
mix_cells = [cell(em_md, em_star, len(em_md)), NA]
for lr in ('0.1', '0.01'):
    mix_cells.append(cell(adam_md[lr], em_star, len(adam_md[lr])))
lines.append('\\midrule')
lines.append(f"Aligned & \\MixDom-{MIXDOM}$^{{\\dagger}}$ & \\Pfam\\ & " + ' & '.join(mix_cells) + ' \\\\')

# ------------------------------------------------------------------ minibatch (svi-BW vs Adam, held-out LL)
def val_hist(path):
    h = json.load(open(path))['history']
    return [(x['iter'], x['val_ll_per_pair']) for x in h if x.get('val_ll_per_pair') is not None]


svi = val_hist(f'stochastic{TAG}/pfam_svi_bw_seed0.json'); adam_sg = val_hist(f'stochastic{TAG}/pfam_adam_tkf92_seed0.json')
vbest = max(max(v for _, v in svi), max(v for _, v in adam_sg))


def iters_to(hist, eps):
    for it, v in hist:
        if v >= vbest - eps: return it
    return None


def mcell(hist, n_iter):
    a, b = iters_to(hist, EPS[0]), iters_to(hist, EPS[1])
    f = lambda x: str(x) if x is not None else f'$>${n_iter}'
    return f'{f(a)} & {f(b)}'


lines.append("Unaligned, minibatch$^{\\ddagger}$ & TKF92 & \\Pfam\\ & " + ' & '.join([mcell(svi, 800), NA, DASH, mcell(adam_sg, 800)]) + ' \\\\')

os.makedirs(f'{PAPER}/tables', exist_ok=True)
with open(f'{PAPER}/tables/em_vs_gradient_table.tex', 'w') as fh:
    fh.write('\\begin{tabular}{@{}lllrrrrrrrr@{}}\n\\toprule\n')
    fh.write(' & & & \\multicolumn{2}{c}{EM} & \\multicolumn{2}{c}{L-BFGS} & \\multicolumn{2}{c}{Adam, $\\eta{=}0.1$} & \\multicolumn{2}{c}{Adam, $\\eta{=}0.01$} \\\\\n')
    fh.write('\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\n')
    fh.write('Setting & Model & Corpus & $0.1$ & $10^{-3}$ & $0.1$ & $10^{-3}$ & $0.1$ & $10^{-3}$ & $0.1$ & $10^{-3}$ \\\\\n\\midrule\n')
    fh.write('\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')
print('\n'.join(lines))

# ------------------------------------------------------------------ figure
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'legend.fontsize': 7})
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.4))
cols = {'em': 'C3', 'lbfgs': 'C0', 'adam_lr0.1': 'C2', 'adam_lr0.01': 'C1'}
d, tr, best = load_run(f'results{TAG}/pfam_tkf92.json')
# Wall-clock axis: by default passes x the cost of one pass measured on an idle GPU (EMVG_PASS_SECONDS="estep=..,grad=..",
# from experiments/timing/run_wallclock_timings.sh's timed passes), since the runs themselves were recorded on a shared card
# and their 'wall' stamps include compilation. EMVG_PASS_SECONDS=record uses the recorded stamps instead.
PASS_SEC = os.environ.get('EMVG_PASS_SECONDS', 'estep=3.16,grad=4.60')
if PASS_SEC != 'record':
    PASS_SEC = dict(kv.split('=') for kv in PASS_SEC.split(','))
# Error bars: if perpair_trajectory.py has recomputed the per-pair LLs at every recorded pass
# (results{TAG}/pfam_tkf92_perpair.npz), draw dotted lines at mean +- one standard error of the per-pair gap to the
# optimum (the optimum = the per-pair LLs at the best recorded pass of any method), across the 1,000 pairs.
PP = f'results{TAG}/pfam_tkf92_perpair.npz'; pp = np.load(PP, allow_pickle=True) if os.path.exists(PP) and os.environ.get('EMVG_ERRORBARS') == '1' else None   # off by default (user, 2026-11-16: each curve is one run, so the bands were not what was wanted)
if pp is not None:
    bm, bk = max(((m, int(np.argmax(tr[m][0]))) for m, _ in METHODS), key=lambda mk: tr[mk[0]][0][mk[1]])
    ll_star = pp[f'll_{bm}'][bk]
    print(f'error bars from {PP}: optimum = {bm} pass {bk + 1}; N = {int(pp["N"])}')
for (m, lab) in METHODS:
    lls, walls = tr[m]; gap = np.maximum(best - np.asarray(lls), 1e-6)   # floored at 1e-6 nats (the best run's own gap is 0)
    if PASS_SEC != 'record':
        walls = np.arange(1, len(lls) + 1) * float(PASS_SEC['estep' if m == 'em' else 'grad'])
    xs_pass, xs_wall = np.arange(1, len(lls) + 1), np.asarray(walls) / 3600
    axs[0, 0].loglog(xs_pass, gap, color=cols[m], label=lab, lw=1.2)
    axs[0, 1].loglog(xs_wall, gap, color=cols[m], label=lab, lw=1.2)
    if pp is not None and f'll_{m}' in pp.files:
        g = ll_star[None, :] - pp[f'll_{m}']                       # (n_pass, N) per-pair gaps
        se = g.std(axis=1, ddof=1) / np.sqrt(g.shape[1]); mean = g.mean(axis=1)
        for band in (mean - se, mean + se):
            band = np.maximum(band, 1e-6)
            axs[0, 0].loglog(xs_pass[:len(band)], band, color=cols[m], lw=0.7, ls=':')
            axs[0, 1].loglog(xs_wall[:len(band)], band, color=cols[m], lw=0.7, ls=':')
gmax = max(float(np.max(best - np.asarray(tr[m][0]))) for m, _ in METHODS)
for ax, xl in ((axs[0, 0], 'passes over the corpus'), (axs[0, 1], 'wall-clock (h)')):
    ax.set_xlabel(xl); ax.set_ylabel('nats per pair below the optimum'); ax.set_ylim(5e-7, 3 * gmax)
    for e, ls in ((0.1, ':'), (1e-3, '--')): ax.axhline(e, color='grey', ls=ls, lw=0.8)
axs[0, 0].set_title('(a) TKF92, 1,000 unaligned Pfam pairs'); axs[0, 1].set_title('(b) the same against wall-clock')
axs[0, 0].legend(frameon=False)

ax = axs[1, 0]
ax.plot(np.arange(1, len(em_md) + 1), em_md, color=cols['em'], lw=1.2, label='EM (full batch)')
for lr, c in (('0.1', cols['adam_lr0.1']), ('0.01', cols['adam_lr0.01'])):
    ax.plot(np.arange(1, len(adam_md[lr]) + 1), adam_md[lr], color=c, lw=0.9, label=f'Adam ({lr})')
ax.axhline(em_star, color='grey', ls='--', lw=0.8)
ax.set_xscale('log'); ax.set_ylim(em_star - 12, em_star + 1.5); ax.set_xlabel('E-steps or gradient steps'); ax.set_ylabel('MAP objective per pair')
ax.set_title(f'(c) MixDom-{MIXDOM}, 1,000 aligned Pfam pairs'); ax.legend(frameon=False, loc='lower right')

ax = axs[1, 1]
ax.plot([i for i, _ in svi], [v for _, v in svi], color=cols['em'], lw=1.2, label='svi-BW (minibatch 16)')
ax.plot([i for i, _ in adam_sg], [v for _, v in adam_sg], color=cols['adam_lr0.01'], lw=1.0, label='Adam (minibatch 16)')   # learning rate 0.01 (stochastic*/pfam_adam_tkf92_seed0.json)
ax.axhline(vbest, color='grey', ls='--', lw=0.8)
ax.set_xscale('log'); ax.set_ylim(vbest - 12, vbest + 1.5); ax.set_xlabel('minibatch iterations'); ax.set_ylabel('held-out log-likelihood per pair')
ax.set_title('(d) TKF92, 20,000 Pfam pairs, minibatch 16'); ax.legend(frameon=False, loc='lower right')
fig.tight_layout()
os.makedirs(f'{PAPER}/figures', exist_ok=True)
fig.savefig(f'{PAPER}/figures/em_vs_gradient.pdf'); fig.savefig(os.path.join(HERE, 'em_vs_gradient.png'), dpi=130)
print('svi-BW best', max(v for _, v in svi), 'Adam best', max(v for _, v in adam_sg), '| MixDom EM*', em_star,
      'Adam0.1 best/last', max(adam_md['0.1']), adam_md['0.1'][-1], 'Adam0.01 best/last', max(adam_md['0.01']), adam_md['0.01'][-1])
