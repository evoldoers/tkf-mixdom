#!/usr/bin/env python3
"""Figure: TKF92 (LG08) and MixDom d1f1..d5f1, one panel per domain type.

Rows = models. Within a row each domain type is placed on a shared log-scale
axis of its insertion rate lambda, so similar domain types line up vertically
across models. Each panel shows: the equilibrium amino-acid composition as
side-by-side letters (height = fold ratio of the frequency over LG08; colour =
chemistry class; solid at or above the LG08 frequency, dotted below it); bars for
lambda and mu on one linear scale shared by every panel; and strings of beads (one bead per residue) for the mean fragment
length 1/(1-r) and the mean domain length lambda/(mu-lambda) * 1/(1-r).
MixFrag-S rows (--mixfrag-s): one panel per fragtype, each with its own substitution model (composition and rate),
its weight and extension, and the insertion / deletion rates shared by every fragtype; the panels sit at the
shared insertion rate and are pushed apart by the overlap rule. A MixFrag-S checkpoint that is absent is skipped.
Usage: domain_logos.py --out <pdf> [--models label=ckpt ...] [--mixfrag-s label=ckpt ...]
"""
import argparse, json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch, Circle, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tkfmixdom.jax.core.protein import rate_matrix_lg

AA = 'ACDEFGHIKLMNPQRSTVWY'   # alphabetical: the order of rate_matrix_lg() and of the checkpoints' pis
CLASSES = [('aliphatic', 'AVLIM', '#222222'), ('aromatic', 'FWY', '#ff7f0e'), ('Pro/Gly', 'PG', '#8c564b'),
           ('polar', 'STC', '#2ca02c'), ('amide', 'QN', '#9467bd'), ('basic', 'KRH', '#1f77b4'), ('acidic', 'DE', '#d62728')]
COLOR = {a: c for _, s, c in CLASSES for a in s}
FP = FontProperties(family='DejaVu Sans', weight='bold')
RATE_MAX = None                       # common linear scale for the lambda / mu bars (set from the data below)
BEADS_PER_ROW = 32

DOT_LEVELS = [(-1.5, '..'), (-0.75, '....'), (0.0, '........')]   # dot density for letters below the background
matplotlib.rcParams['hatch.linewidth'] = 0.6

def shade(color, ratio):
    """Letter styling by enrichment over LG08 (ratio = fitted / background frequency). At or above the background
    the letter is solid in its class colour, with a dark outline from 1.7x upward. Below the background it is a
    dotted fill: dots of the class colour on an empty interior, denser the nearer the frequency is to the
    background (three density levels, the sparsest at 1/2.8 and below), with a thin outline in the class colour."""
    rgb = np.array(matplotlib.colors.to_rgb(color)); l2 = np.log2(max(ratio, 1e-9))
    if l2 >= 0.0:
        return rgb, (rgb * 0.4 if l2 >= 0.75 else rgb), None, 0.5
    return 'none', rgb, next(h for lim, h in DOT_LEVELS if l2 <= lim) if l2 <= 0.0 else None, 0.3

# categories for the one-line enrichment summary: (label, members, parent)
CATEGORIES = [('hydrophobic', 'AVLIMFWC', None), ('aliphatic', 'AVLIM', 'hydrophobic'), ('aromatic', 'FWY', 'hydrophobic'),
              ('small', 'GASTC', None), ('Pro/Gly', 'PG', None), ('charged', 'DEKR', None), ('acidic', 'DE', 'charged'),
              ('basic', 'KRH', 'charged'), ('polar', 'STNQ', None), ('amide', 'NQ', 'polar')]

def enrichment_summary(pi, pi_bg, min_mean=0.15, min_consistency=0.7, max_terms=2):
    """Background-weighted mean log2 fold ratio per category; a category qualifies if that mean is >= min_mean
    (twice that for a two-member category) and at least min_consistency of its members are enriched. A child category is reported only when its parent
    does not qualify. Returns up to max_terms labels, strongest first, or '' when nothing is clear."""
    l2 = np.log2(np.asarray(pi) / np.asarray(pi_bg)); idx = {a: i for i, a in enumerate(AA)}
    score = {}
    for name, members, parent in CATEGORIES:
        ii = [idx[a] for a in members]; w = np.asarray(pi_bg)[ii]
        m = float((w * l2[ii]).sum() / w.sum()); cons = float((l2[ii] > 0).mean())
        need = min_mean if len(members) > 2 else 2 * min_mean      # two-member categories need a stronger signal
        if m >= need and cons >= min_consistency: score[name] = m
    out = [(m, name) for name, m in score.items() if dict((n, p) for n, _, p in CATEGORIES)[name] not in score]
    return ', '.join(name for _, name in sorted(out, reverse=True)[:max_terms])

def draw_logo(ax, pi, pi_bg, ymax):
    """Letters scaled by the fold ratio of their fitted frequency to the LG08 background, sorted by that ratio;
    a dashed line marks ratio 1. Letters below 1 are dotted (see shade)."""
    ratio = np.asarray(pi) / np.asarray(pi_bg); order = np.argsort(-ratio)
    for x, i in enumerate(order):
        h = float(ratio[i]); tp = TextPath((0, 0), AA[i], size=1.0, prop=FP); bb = tp.get_extents()
        tr = Affine2D().translate(-bb.x0, -bb.y0).scale(0.82 / bb.width, max(h, 1e-4) / bb.height).translate(x + 0.09, 0)
        fc, ec, hatch, lw = shade(COLOR[AA[i]], h); path = tr.transform_path(tp)
        if hatch: ax.add_patch(PathPatch(path, facecolor='none', edgecolor=COLOR[AA[i]], hatch=hatch, linewidth=0))   # dots take the edge colour
        ax.add_patch(PathPatch(path, facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.axhline(1.0, color='#888888', lw=0.6, ls=(0, (3, 2)), zorder=0)
    cap = enrichment_summary(pi, pi_bg)
    if cap and not GLYPHS: ax.text(0.98, 0.96, cap, transform=ax.transAxes, ha='right', va='top', fontsize=FS * 7, style='italic', color='#333333')
    ax.set_xlim(-0.2, 20.2); ax.set_ylim(0, ymax); ax.set_xticks([]); ax.set_yticks(range(0, int(np.ceil(ymax)) + 1)); ax.tick_params(labelsize=4 * FS, length=2, pad=1)
    if GLYPHS: ax.tick_params(labelleft=False, length=3)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)

def draw_bars(ax, lam, mu, sub=''):
    """Two horizontal bars; length proportional to the rate on one linear scale (0 to RATE_MAX) shared by every panel.
    sub = subscript: the domain type's index (1.. left to right), 0 for the domain level."""
    for y, v, c, lab in ((1.0, lam, '#444444', f'$\\lambda_{{{sub}}}$' if sub != '' else 'λ'), (0.0, mu, '#999999', f'$\\mu_{{{sub}}}$' if sub != '' else 'μ')):
        x = min(max(v / RATE_MAX, 0.0), 1.0)
        if x < 0.004: ax.plot([0, 0], [y + 0.12, y + 0.88], color=c, lw=1.2, solid_capstyle='butt')
        ax.add_patch(Rectangle((0, y + 0.12), x, 0.76, facecolor=c, edgecolor='none'))
        ax.text(-0.03, y + 0.5, lab, ha='right', va='center', fontsize=FS * 7)
        if x < 0.68: ax.text(x + 0.02, y + 0.5, f'{v:.3g}', ha='left', va='center', fontsize=FS * 6, color='#333333')
        else: ax.text(x - 0.02, y + 0.5, f'{v:.3g}', ha='right', va='center', fontsize=FS * 6, color='white')
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 2.05); ax.axis('off')

def draw_beads(ax, n, label, color, note='', open_=False):
    """One bead per residue (a partial bead for the fraction), BEADS_PER_ROW to a row, wrapping onto further rows
    for long domains; the number is printed after the first row."""
    step = 1.0 / (BEADS_PER_ROW + 1); full = int(np.floor(n)); frac = n - full
    rows = max(1, int(np.ceil(max(n, 1e-9) / BEADS_PER_ROW)))
    fc, ec, lw = ('white', color, 0.9) if open_ else (color, 'none', 0)
    if not GLYPHS: ax.text(-0.03, 0.0, label, ha='right', va='center', fontsize=FS * 7)
    for i in range(full):
        r_, c_ = divmod(i, BEADS_PER_ROW); ax.add_patch(Circle(((c_ + 0.5) * step, -r_ * step), 0.46 * step, facecolor=fc, edgecolor=ec, linewidth=lw))
    if frac > 0.05:
        r_, c_ = divmod(full, BEADS_PER_ROW); ax.add_patch(Circle(((c_ + 0.5) * step, -r_ * step), 0.42 * step * np.sqrt(frac), facecolor=fc, edgecolor=ec, linewidth=lw))
    if not GLYPHS: ax.text((min(np.ceil(n), BEADS_PER_ROW) + 0.8) * step, 0.0, (f'{n:.2f}' if n < 100 else f'{n:.1f}') + note, ha='left', va='center', fontsize=FS * 5, color='#333333')
    ax.set_xlim(0, 1); ax.set_ylim(-(rows - 0.5) * step, 0.5 * step); ax.set_aspect('equal', adjustable='box'); ax.set_anchor('NW'); ax.axis('off')

GAP_KMAX, GAP_PMIN = 60, 2.0 ** -13   # common axes for the gap-length sketches (y from log2 P = -13 to 0)
T_SKETCH = 0.76                       # branch length for the pair-HMM sketch: the training corpus's median pair time

def insert_run_pmf(log_chi, st, kmax=GAP_KMAX):
    """Distribution of the number of inserted residues on a path between two consecutive match states of the
    pair HMM (at a fixed branch length), from its transition matrix: with A the transition matrix, M the match
    states and I the insert-emitting states (whatever their domain / fragtype, whether a within-domain insert
    or a whole inserted domain), P(k) is proportional to pi_M A_MI A_II^(k-1) A_IM 1 for k >= 1, where pi_M
    weights the match states by their expected number of visits on a path from the start state."""
    from tkfmixdom.jax.core.params import S as _S, M as _M, I as _I, E as _E
    A = np.exp(np.asarray(log_chi, dtype=float)); st = np.asarray(st); n = len(st)
    keep = np.where(st != _E)[0]; An = A[np.ix_(keep, keep)]
    e = np.zeros(len(keep)); e[list(keep).index(int(np.where(st == _S)[0][0]))] = 1.0
    v = np.linalg.solve((np.eye(len(keep)) - An).T, e)              # expected visits per non-end state
    Mi = [i for i in range(n) if st[i] == _M]; Ii = [i for i in range(n) if st[i] == _I]
    piM = np.array([v[list(keep).index(i)] for i in Mi]); piM = piM / piM.sum()
    A_MI, A_II, A_IM = A[np.ix_(Mi, Ii)], A[np.ix_(Ii, Ii)], A[np.ix_(Ii, Mi)]
    w = piM @ A_MI; out = []
    for _ in range(kmax):
        out.append(float(w @ A_IM.sum(axis=1))); w = w @ A_II
    Z = float(piM @ A_MI @ np.linalg.solve(np.eye(len(Ii)) - A_II, A_IM.sum(axis=1)))
    return np.array(out) / Z

def draw_gap_sketch(ax, pmf, label):
    """log2 P(k) against k on linear axes: the x axis sits at the top where log P = 0, the y axis (left) is log2 P."""
    k = np.arange(1, GAP_KMAX + 1)
    ax.plot(k, np.log2(np.maximum(pmf, GAP_PMIN / 10)), color='#333333', lw=1.1)
    ax.set_xlim(1, GAP_KMAX); ax.set_ylim(np.log2(GAP_PMIN), 0.0)
    ax.set_xticks([1, 20, 40, 60]); ax.set_yticks([-12, -8, -4, 0]); ax.set_ylabel('log$_2$ P', fontsize=FS * 6, labelpad=1)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position('top'); ax.tick_params(labelsize=3.7 * FS, length=2, pad=1)
    if GLYPHS: ax.set_ylabel(''); ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, length=3)
    else: ax.set_title(label, fontsize=FS * 6, pad=6, linespacing=1.0)
    for sp in ('bottom', 'right'): ax.spines[sp].set_visible(False)

def subst_rate(S, pi):
    Q = S * pi[None, :]; np.fill_diagonal(Q, 0.0); return float((pi * Q.sum(1)).sum())

def subst_rate_Q(Q, pi):
    """Expected substitutions per unit time of a rate matrix at its equilibrium: -sum_a pi_a Q_aa."""
    return float(-(np.asarray(pi) * np.diag(np.asarray(Q))).sum())

def domain_stats(pi, S, w, lam, mu, r, frags=None, rate=None):
    """frags: [(weight, extension), ...] for a fragment-type mixture (MixFrag); default one fragtype with extension r.
    Mean fragment length = sum_f w_f / (1 - r_f); mean domain (or sequence) length = lam/(mu-lam) * that.
    rate: the substitution rate, if already known (default: from S and pi)."""
    frags = frags or [(1.0, r)]
    frag = sum(wf / (1.0 - rf) for wf, rf in frags); dom = (lam / max(mu - lam, 1e-9)) * frag if mu > lam else float('inf')
    return dict(pi=pi, w=w, lam=lam, mu=mu, r=r, frags=frags, rate=subst_rate(S, pi) if rate is None else rate, frag_len=frag, dom_len=dom)

def mixfrag_s_of(ckpt):
    """MixFrag-S checkpoint (ins_rate, del_rate, exts (F,), weights (F,), S_exch (F,A,A), pi (F,A), Q (F,A,A)):
    one panel per fragtype, ordered by extension (short to long), each at the shared insertion rate with its own
    composition and substitution rate (-sum_a pi_f[a] Q_f[a,a]), weight w_f and extension r_f; the row's expected
    sequence length is lam/(mu-lam) x the weight-averaged mean fragment length, as for MixFrag; the gap sketch is
    from make_mixfrag_pair_hmm's transition matrix (its emission arguments play no part in the sketch)."""
    d = np.load(ckpt, allow_pickle=True)
    lam, mu = float(d['ins_rate']), float(d['del_rate']); r = np.asarray(d['exts'], float).ravel(); w = np.asarray(d['weights'], float).ravel()
    pis = np.asarray(d['pi'], float); Qs = np.asarray(d['Q'], float); Ss = np.asarray(d['S_exch'], float)
    doms = [domain_stats(pis[f] / pis[f].sum(), Ss[f], float(w[f]), lam, mu, float(r[f]), rate=subst_rate_Q(Qs[f], pis[f] / pis[f].sum())) for f in np.argsort(r)]
    seq_len = (lam / max(mu - lam, 1e-9)) * float(sum(w / (1.0 - r))) if mu > lam else float('inf')
    _lt, _st, _, _ = make_mixfrag_pair_hmm(lam, mu, T_SKETCH, jnp.asarray(r), jnp.asarray(w), jnp.asarray(Qs[0]), jnp.asarray(pis[0]))
    return doms, dict(seq_len=seq_len, hmm=(np.asarray(_lt), np.asarray(_st)))

def domains_of(ckpt):
    d = np.load(ckpt, allow_pickle=True)
    S = np.asarray(d['class_S_exch']) if 'class_S_exch' in d.files else np.asarray(d['dom_S_exch'])
    pis = np.asarray(d['class_pis']) if 'class_pis' in d.files else np.asarray(d['dom_pis'])
    lam = np.asarray(d['dom_ins']).ravel(); mu = np.asarray(d['dom_del']).ravel(); w = np.asarray(d['dom_weights']).ravel(); r = np.asarray(d['ext_rates']).ravel()
    doms = [domain_stats(pis[k] / pis[k].sum(), S[k], float(w[k]), float(lam[k]), float(mu[k]), float(r[k])) for k in np.argsort(lam)]
    top = dict(lam=float(d['main_ins']), mu=float(d['main_del']))     # domain-level (whole-domain) insertion / deletion rates
    from train_pfam import _load_checkpoint, _build_log_chi_stack
    from tkfmixdom.jax.models.mixdom import state_types as mixdom_state_types
    params = _load_checkpoint(ckpt)[0]; cfg = json.loads(str(d['_config']))
    top['hmm'] = (np.asarray(_build_log_chi_stack(params, np.array([T_SKETCH])))[0], np.asarray(mixdom_state_types(int(cfg['n_dom']), int(cfg['n_frag']))))
    # expected number of domains (top-level TKF91 at equilibrium) x mixture-weighted mean domain length in residues
    top['n_dom'] = top['lam'] / max(top['mu'] - top['lam'], 1e-9) if top['mu'] > top['lam'] else float('inf')
    top['mean_dom_len'] = float(sum(x['w'] * x['dom_len'] for x in doms) / sum(x['w'] for x in doms))
    top['seq_len'] = top['n_dom'] * top['mean_dom_len']
    return doms, top

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--background-npz', default='pfam/gtr_refit_Qpi.npz', help="(Q, pi) of the baseline substitution model: letter heights are fold ratios over its equilibrium and 'subst' rates are relative to its mean rate; '' for LG08")
ap.add_argument('--font-scale', type=float, default=None, help='font scale FS (default 1.5 with --paper, else 1); the page shrinks the canvas to about 0.3, so 3 gives ~6-7 pt labels')
ap.add_argument('--skip-fixed', action='store_true', help='omit the TKF92 and MixFrag rows (part (b) of a two-part figure)')
ap.add_argument('--rate-max', type=float, default=None, help='override the shared linear rate scale (use the value printed by a run with every row, so parts share scales)')
ap.add_argument('--ratio-max', type=float, default=None, help='override the shared fold-ratio y scale (same purpose)')
ap.add_argument('--bottom-pad', type=float, default=1.3, help='inches reserved below the last row for the shared x axis and the colour legend (2.0 for a part with few rows)')
ap.add_argument('--top-pad', type=float, default=0.25, help='inches above the first row (0.6 for the two-part paper figure, whose panel headings are two lines)')
ap.add_argument('--glyphs', action='store_true', help='one-page variant: no per-panel text (only the logo letters), w and substitution rate as gauges, open circles for fragment length, and a large legend with scales in the empty top-left region')
ap.add_argument('--paper', action='store_true', help='no suptitle / explanatory footer (they go in the paper caption); colour legend only; 1.5x fonts')
ap.add_argument('--tkf92-json', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'profile_mixture/results/fsa_c20_balibase/cache_runs/tkf92_GTR_one_bali3pdbm.json'), help="harness output JSON whose 'indel_params' are TKF92's fitted Pfam parameters (0.04428/0.04521/0.677)")
ap.add_argument('--mixfrag-json', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'profile_mixture/results/fsa_c20_balibase/cache_runs/mixfrag_GTR_one_bali3pdbm.json'), help="same for MixFrag F=2 (mixfrag_F2_logspace.json values)")
ap.add_argument('--models', nargs='*', default=[f'd{n}f1=pfam/svibw_d{n}f1_nobali_1620.npz' for n in range(1, 6)])
ap.add_argument('--mixfrag-s', nargs='*', default=[f'MixFrag-S\\n(F={F})=pfam/svibw_mixfrags_F{F}_nobali_1620.npz' for F in (2, 3)],
                help="MixFrag-S rows, label=checkpoint (a literal \\n in the label breaks the line), placed after the MixFrag row; absent checkpoints are skipped")
args = ap.parse_args()
GLYPHS = args.glyphs
if GLYPHS: ROW_H_OVERRIDE = 2.2
FS = args.font_scale if args.font_scale else (1.5 if args.paper else 1.0)    # font scale (the page shrinks the 17-inch canvas to about 6 inches)

if args.background_npz:
    _bg = np.load(args.background_npz, allow_pickle=True); Q_lg = np.asarray(_bg['Q'], float); pi_lg = np.asarray(_bg['pi'], float); BG_NAME = 'GTR refit'
else:
    Q_lg, pi_lg = rate_matrix_lg(); Q_lg = np.asarray(Q_lg); pi_lg = np.asarray(pi_lg); BG_NAME = 'LG08'
pi_lg = pi_lg / pi_lg.sum()
S_lg = Q_lg / pi_lg[None, :]; np.fill_diagonal(S_lg, 0.0); rate_lg = subst_rate(S_lg, pi_lg)
tk = json.load(open(args.tkf92_json))['indel_params']
rows = [(f'TKF92' + ('' if BG_NAME == 'GTR refit' else f'\n({BG_NAME})'), [domain_stats(pi_lg, S_lg, 1.0, tk['lam'], tk['mu'], tk['exts'][0])], None)]   # the refitted matrix is the baseline, so it is not named
FIXED_ROWS = 2
from tkfmixdom.jax.models.left_regular import make_tkf92_pair_hmm, make_mixfrag_pair_hmm
import jax.numpy as jnp
_lt, _st, _, _ = make_tkf92_pair_hmm(tk['lam'], tk['mu'], T_SKETCH, tk['exts'][0], jnp.asarray(Q_lg), jnp.asarray(pi_lg))
rows[0] = (rows[0][0], rows[0][1], dict(seq_len=rows[0][1][0]['dom_len'], hmm=(np.asarray(_lt), np.asarray(_st))))   # TKF92: no domain level
mf = json.load(open(args.mixfrag_json))['indel_params']
mfd = domain_stats(pi_lg, S_lg, 1.0, mf['lam'], mf['mu'], None, frags=list(zip(mf['weights'], mf['exts'])))
_lt, _st, _, _ = make_mixfrag_pair_hmm(mf['lam'], mf['mu'], T_SKETCH, jnp.asarray(mf['exts']), jnp.asarray(mf['weights']), jnp.asarray(Q_lg), jnp.asarray(pi_lg))
rows.append((f'MixFrag\n(F=2)' + ('' if BG_NAME == 'GTR refit' else f', {BG_NAME}'), [mfd], dict(seq_len=mfd['dom_len'], hmm=(np.asarray(_lt), np.asarray(_st)))))
if args.skip_fixed: rows = rows[FIXED_ROWS:]
for spec in args.mixfrag_s:   # MixFrag-S: per-fragtype substitution models, one panel per fragtype
    label, ck = spec.rsplit('=', 1); label = label.replace('\\n', '\n')
    if os.path.exists(ck): doms, top = mixfrag_s_of(ck); rows.append((label, doms, top))
    else: print('skipping (no checkpoint yet):', spec)
for spec in args.models:
    label, ck = spec.split('=')
    if os.path.exists(ck): doms, top = domains_of(ck); rows.append((label, doms, top))
    else: print('skipping (no checkpoint yet):', spec)

RATE_MAX = max([max(d['lam'], d['mu']) for _, doms, _ in rows for d in doms] + [max(t['lam'], t['mu']) for _, _, t in rows if t and 'lam' in t]) * 1.02
RATIO_MAX = max(float((d['pi'] / pi_lg).max()) for _, doms, _ in rows for d in doms) * 1.05   # common y scale for the fold-ratio logos
print(f'scales: rate_max {RATE_MAX:.6f} ratio_max {RATIO_MAX:.4f} rows {len(rows)}')
if args.rate_max: RATE_MAX = args.rate_max
if args.ratio_max: RATIO_MAX = args.ratio_max

SUBST_MAX = 2.5   # full length of the substitution-rate gauge (relative rate)
def draw_gauges(ax, w, rel_rate):
    """Two thin horizontal gauges above a panel: the type's weight w (top, grey, full width = 1) and its substitution
    rate relative to the baseline matrix (bottom, orange, full width = SUBST_MAX, tick at 1)."""
    ax.add_patch(Rectangle((0, 0.55), 1, 0.4, facecolor='#dff0dc', edgecolor='none')); ax.add_patch(Rectangle((0, 0.55), min(w, 1), 0.4, facecolor='#3a9d3a', edgecolor='none'))
    ax.add_patch(Rectangle((0, 0.0), 1, 0.4, facecolor='#fbe5cc', edgecolor='none')); ax.add_patch(Rectangle((0, 0.0), min(rel_rate / SUBST_MAX, 1), 0.4, facecolor='#e07b20', edgecolor='none'))
    ax.plot([1 / SUBST_MAX] * 2, [-0.05, 0.45], color='#333333', lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.0); ax.axis('off')

def draw_legend(fig, x0, x1, y0, y1, pi_example, pi_bg, ratio_max, hmm_example):
    """Terse legend for the glyph-only figure: an example glyph at the left, a few words at the right, scales where
    there is one; laid out top-down in figure coordinates inside [x0, x1] x [y0, y1], on a white panel (zorder 1)
    that hides the row separators."""
    from matplotlib.lines import Line2D
    L = FS * 8.0; H = y1 - y0; W = x1 - x0; lh = L * 1.25 / (72.0 * FIG_H)
    gx = x0 + 0.075 * W; gw = 0.20 * W; tx = x0 + 0.31 * W
    y = y1 - 0.8 * lh; made = []
    def block(text, glyph_h, draw, gap=0.9):
        nonlocal y
        n = text.count('\n') + 1; bh = max(n * lh, glyph_h); yc = y - bh / 2
        fig.text(tx, yc, text, fontsize=L, va='center', ha='left', linespacing=1.15, zorder=3)
        ax = fig.add_axes([gx, yc - glyph_h / 2, gw, glyph_h]); ax.set_zorder(2); draw(ax); made.append(ax)
        y -= bh + gap * lh
    def g_weight(ax):
        ax.add_patch(Rectangle((0, 0.3), 1, 0.4, facecolor='#dff0dc', edgecolor='none')); ax.add_patch(Rectangle((0, 0.3), 0.4, 0.4, facecolor='#3a9d3a', edgecolor='none'))
        for xx, lab in ((0, '0'), (0.5, '0.5'), (1, '1')): ax.text(xx, 0.82, lab, ha='center', va='bottom', fontsize=L * 0.7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.5); ax.axis('off')
    def g_subst(ax):
        ax.add_patch(Rectangle((0, 0.6), 1, 0.4, facecolor='#fbe5cc', edgecolor='none')); ax.add_patch(Rectangle((0, 0.6), 1.8 / SUBST_MAX, 0.4, facecolor='#e07b20', edgecolor='none'))
        ax.plot([1 / SUBST_MAX] * 2, [0.55, 1.05], color='#333333', lw=0.8)
        for xx, lab in ((0, '0'), (1 / SUBST_MAX, '1'), (2 / SUBST_MAX, '2')): ax.text(xx, 0.45, lab, ha='center', va='top', fontsize=L * 0.7)
        ax.set_xlim(0, 1); ax.set_ylim(-0.3, 1.1); ax.axis('off')
    block('weight $w$ of component\n(fragment or domain)', 1.8 * lh, g_weight, gap=0.4)
    block('substitution rate\n(ratio to baseline)', 1.8 * lh, g_subst, gap=0.8)
    def g_logo(ax):
        draw_logo(ax, pi_example, pi_bg, ratio_max); ax.tick_params(labelleft=True, labelsize=L * 0.7)
    block('amino acid composition\n(ratio to baseline)', 3.4 * lh, g_logo, gap=0.6)
    def g_bars(ax):
        draw_bars(ax, 0.6 * RATE_MAX, 0.4 * RATE_MAX, sub=''); ax.set_ylim(-1.4, 2.05)
        ax.plot([0, 1], [-0.1, -0.1], color='#333333', lw=0.8, clip_on=False)
        for v in (0.0, 0.1, 0.2):
            xx = v / RATE_MAX; ax.plot([xx, xx], [-0.1, -0.45], color='#333333', lw=0.8, clip_on=False); ax.text(xx, -0.6, f'{v:.1f}', ha='center', va='top', fontsize=L * 0.7)
    block('insertion and deletion rates', 2.6 * lh, g_bars, gap=0.6)
    block('mean fragment length\n(1 circle = 1 residue)', 1.4 * lh, lambda ax: draw_beads(ax, 5.5, '', '#444444', open_=True), gap=0.4)
    block('mean domain length', 1.2 * lh, lambda ax: draw_beads(ax, 8.0, '', '#1f77b4'), gap=0.4)
    block('mean sequence length', 1.2 * lh, lambda ax: draw_beads(ax, 12.0, '', '#17becf'), gap=0.8)
    def g_sketch(ax):
        draw_gap_sketch(ax, insert_run_pmf(*hmm_example), '')
        ax.tick_params(labelbottom=False, labeltop=True, labelleft=True, labelsize=L * 0.65); ax.set_ylabel('log$_2$ P', fontsize=L * 0.7, labelpad=2); ax.set_xlabel('k', fontsize=L * 0.75, labelpad=2); ax.xaxis.set_label_position('top')
    block('gap length distribution', 4.0 * lh, g_sketch, gap=0.5)
    pad = 0.4 * lh
    fig.add_artist(Rectangle((x0 - 0.012, y - pad), W + 0.03, (y1 + pad) - (y - pad), transform=fig.transFigure, facecolor='white', edgecolor='#333333', linewidth=1.2, zorder=1))
    if y - pad < y0: print(f'legend box reaches {(y0 - (y - pad)) / lh:.1f} lines into the row below')
    else: print(f'legend box clears the row below by {((y - pad) - y0) / lh:.1f} lines')

# ---- layout: x = log10(lambda) on a shared axis; panels pushed apart to avoid overlap ----
X0, X1 = -3.9, -0.6; PW = 0.42                       # decades; panel width in decades
FIG_W, ROW_H = 17.0, (ROW_H_OVERRIDE if GLYPHS else 3.1); LEFT, RIGHT = (0.025, 0.80) if GLYPHS else (0.085, 0.875)    # the strip right of RIGHT holds the domain-level rates
BOT = args.bottom_pad; TOP = args.top_pad; FIG_H = ROW_H * len(rows) + BOT + TOP
fig = plt.figure(figsize=(FIG_W, FIG_H))
def fx(decade): return LEFT + (decade - X0) / (X1 - X0) * (RIGHT - LEFT)
row_frac = ROW_H / FIG_H; top_pad = TOP / FIG_H   # room for the suptitle above the first row's panel title
BAR_W = PW / (X1 - X0) * (RIGHT - LEFT) * 0.88               # physical width of a panel's bar axis: reused on the right so the scale is identical
for i, (label, doms, top) in enumerate(rows):
    centres = [np.log10(d['lam']) for d in doms]
    for k in range(1, len(centres)):                 # push apart left to right
        centres[k] = max(centres[k], centres[k - 1] + PW + 0.04)
    overflow = centres[-1] + PW / 2 - X1
    if overflow > 0: centres = [c - overflow for c in centres]
    y_top = 1.0 - top_pad - i * row_frac; y_bot = y_top - row_frac * 0.92
    if GLYPHS: fig.text(0.995, (y_top + y_bot) / 2, label, fontsize=FS * 10, fontweight='bold', va='center', ha='right')
    else: fig.text(0.012, (y_top + y_bot) / 2, label, fontsize=FS * 10, fontweight='bold', va='center', ha='left')
    for k, (dm, c) in enumerate(zip(doms, centres), start=1):
        left, width = fx(c - PW / 2), fx(c + PW / 2) - fx(c - PW / 2)
        h = y_top - y_bot
        ax_logo = fig.add_axes([left + width * 0.04, y_bot + 0.47 * h, width * 0.92, 0.46 * h]); draw_logo(ax_logo, dm['pi'], pi_lg, RATIO_MAX)
        if GLYPHS: ax_g2 = fig.add_axes([left + width * 0.04, y_bot + 0.945 * h, width * 0.92, 0.045 * h]); draw_gauges(ax_g2, dm['w'], dm['rate'] / rate_lg)
        else: ax_logo.set_title(f"w = {dm['w']:.2f}\nsubst {dm['rate']/rate_lg:.2f}", fontsize=FS * 7.5, pad=2, linespacing=1.0)
        ax_bar = fig.add_axes([left + width * 0.08, y_bot + 0.28 * h, width * 0.88, 0.17 * h]); draw_bars(ax_bar, dm['lam'], dm['mu'], sub=k if (top and 'lam' in top) else '')
        if len(dm['frags']) == 1:
            ax_b1 = fig.add_axes([left + width * 0.08, y_bot + 0.14 * h, width * 0.88, 0.11 * h]); draw_beads(ax_b1, dm['frag_len'], 'frag', '#444444', note=f"  (r={dm['r']:.3f})", open_=GLYPHS)
        else:   # fragment-type mixture: one bead row per fragtype
            for fi, (wf, rf) in enumerate(dm['frags']):
                ax_bf = fig.add_axes([left + width * 0.08, y_bot + (0.14 - 0.12 * fi) * h, width * 0.88, 0.11 * h])
                draw_beads(ax_bf, 1.0 / (1.0 - rf), f'frag {fi + 1}', '#444444', note=f"  (w={wf:.3f}, r={rf:.3f})", open_=GLYPHS)
        if top and 'lam' in top:   # single-level models (TKF92, MixFrag) show the sequence length on the right instead
            ax_b2 = fig.add_axes([left + width * 0.08, y_bot - 0.04 * h, width * 0.88, 0.16 * h]); draw_beads(ax_b2, dm['dom_len'], 'dom', '#1f77b4')
    h = y_top - y_bot
    if top and 'lam' in top:   # domain-level insertion / deletion rates, same linear scale and the same bar width as the panels
        ax_top = fig.add_axes([RIGHT + 0.02, y_bot + 0.28 * h, BAR_W, 0.17 * h]); draw_bars(ax_top, top['lam'], top['mu'], sub=0)
        if not GLYPHS: ax_top.set_title('domain level', fontsize=FS * 7.5, pad=2)
    if top:   # gap-length sketch for one event at the top (or only) level, common axes for every row
        ax_g = fig.add_axes([RIGHT + 0.035, y_bot + 0.52 * h, BAR_W - 0.015, 0.28 * h])
        draw_gap_sketch(ax_g, insert_run_pmf(*top['hmm']), 'inserts between\nmatches' if args.paper else f'inserts between matches, t = {T_SKETCH}')
    if top:   # expected sequence length in residues = E[#domains] x mixture-weighted mean domain length
        ax_sq = fig.add_axes([RIGHT + 0.02, y_bot - 0.04 * h, BAR_W * 0.9, 0.16 * h]); draw_beads(ax_sq, top['seq_len'], 'seq', '#17becf')
if GLYPHS:
    from matplotlib.lines import Line2D
    for _i in range(1, len(rows)):
        _y = 1.0 - top_pad - _i * row_frac + 0.03 * row_frac
        fig.add_artist(Line2D([0.01, 0.99], [_y, _y], transform=fig.transFigure, color='#aaaaaa', lw=1.0, ls=(0, (1, 4)), zorder=0))
    _ex = next(r for r in rows if r[0].startswith('d2f1') or r[0].startswith('d3f1'))
    _hmm = next(r for r in rows if r[2] and 'hmm' in r[2])[2]['hmm']
    _yl0 = 1.0 - top_pad - 4 * row_frac + 0.02 * row_frac; _yl1 = 1.0 - top_pad - 0.02 * row_frac
    draw_legend(fig, LEFT + 0.015, LEFT + 0.5 * (RIGHT - LEFT), _yl0, _yl1, _ex[1][0]['pi'], pi_lg, RATIO_MAX, _hmm)
# shared x axis for lambda at the bottom
ax_x = fig.add_axes([LEFT, (BOT + 0.3) / FIG_H, RIGHT - LEFT, 0.001]); ax_x.set_xlim(X0, X1); ax_x.set_yticks([])
ticks = [-3, -2, -1]; ax_x.set_xticks(ticks); ax_x.set_xticklabels([f'$10^{{{t}}}$' for t in ticks], fontsize=FS * 8)
ax_x.set_xlabel('insertion rate λ' if GLYPHS else 'insertion rate λ (panel position; pushed apart where types overlap)', fontsize=FS * 8.5)
for s in ('top', 'left', 'right'): ax_x.spines[s].set_visible(False)
fig.legend(handles=[mpatches.Patch(color=c, label=f'{n} {s}') for n, s, c in CLASSES], loc='lower center', ncol=7, fontsize=FS * 7.5, frameon=False, bbox_to_anchor=(0.5, 0.25 / FIG_H))
if not args.paper: fig.text(0.5, 0.004, f'letter height = fitted / LG08 frequency (solid at or above 1, dark outline from 1.7x; dotted below 1, denser nearer 1); bars: insertion rate λ and deletion rate μ on one linear scale (full width = {RATE_MAX:.3f}), subscript = domain type numbered left to right, 0 = domain level; '
         'beads: mean fragment length 1/(1−r), mean domain length λ/(μ−λ)·1/(1−r), expected sequence length (domain-level λ/(μ−λ) × the weight-averaged domain length), one bead per residue, 32 to a row; gap sketch: log2 probability of the number of inserted residues on a path between two match states of the pair HMM at branch length 0.76 (the corpus median), from its transition matrix, common axes', ha='center', va='bottom', fontsize=FS * 7)
if not args.paper: fig.suptitle('Domain types: composition as fold ratio over LG08 (letter height; dashed line = 1; colour = chemistry class), substitution rate, indel rates, fragment and domain lengths', fontsize=FS * 10.5, y=0.995)
fig.savefig(args.out, dpi=180); print('wrote', args.out)
if args.out.endswith('.pdf'): fig.savefig(args.out[:-4] + '.png', dpi=150)
