#!/usr/bin/env python3
"""Does the single-EM-step branch length used for the benchmark posteriors matter? For a stratified sample of
BAliBASE pairs under a MixDom checkpoint: tau, log-likelihood and the pairwise MEA alignment's recall and
precision against the reference core pairs, with one EM round from tau = 1 (the submitted procedure) and with
n rounds (converged). Usage: tau_rounds_check.py <mixdom.npz> <n_pairs> <rounds> <out.json>"""
import os, sys, json, time
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
from tkfmixdom.jax.tree.fsa_anneal import pairwise_posteriors_mixdom
from tkfmixdom.util.msa_benchmark import parse_fasta
from tkfmixdom.util.expected_pair_f1 import expected_family_f1
from tkfmixdom.jax.distill.maraschino import load_params as load_mixdom_params
from fsa_mixdom_pairhmm_oxbench import encode_seq, make_fsa_params
from expected_pairwise_balibase import _optimal_accuracy_indicator

ckpt, n_pairs, rounds, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
params, n_dom, _ = load_mixdom_params(ckpt); fp, n_frag = make_fsa_params(params, n_dom)
bdir = os.path.expanduser('~/bio-datasets/data/balibase/bali3pdbm')
tc = json.load(open('experiments/balibase_indel_refit/tkf92_bali3pdbm_tcache.json'))
# stratified sample: sort all pairs by the TKF92 anchor time, take every k-th
allp = sorted(tc.items(), key=lambda kv: kv[1]); step = max(1, len(allp) // n_pairs); sample = allp[::step][:n_pairs]
res = []; t0 = time.time()
for key, t_anchor in sample:
    fam, a, b = key.split('|'); raw = parse_fasta(f'{bdir}/in/{fam}'); ref = parse_fasta(f'{bdir}/ref/{fam}')
    if a not in raw or b not in raw: continue
    x = jnp.asarray(encode_seq(raw[a]), dtype=jnp.int32); y = jnp.asarray(encode_seq(raw[b]), dtype=jnp.int32)
    if len(x) == 0 or len(y) == 0 or len(x) > 600 or len(y) > 600: continue
    row = dict(family=fam, a=a, b=b, Lx=int(len(x)), Ly=int(len(y)), t_anchor=t_anchor)
    for label, nr in (('r1', 1), (f'r{rounds}', rounds)):
        try:
            mp, tau, lp = pairwise_posteriors_mixdom(x, y, fp, n_dom, n_frag, n_em_rounds=nr)
        except Exception as e:
            jax.clear_caches(); row[label] = {'error': str(e)[:100]}; continue
        ind = _optimal_accuracy_indicator(np.asarray(mp))
        f = expected_family_f1({(0, 1): ind}, {a: ref[a], b: ref[b]}, [a, b], core_only=True)['micro']
        row[label] = dict(tau=float(tau), lp=float(lp), e_tp=float(f['e_tp']), mass=float(f['total_mass']), gold=float(f['gold']))
    res.append(row)
    if len(res) % 10 == 0:
        json.dump(res, open(out, 'w')); print(f'[{time.time() - t0:6.0f}s] {len(res)} pairs', flush=True)
json.dump(res, open(out, 'w'))
ok = [r for r in res if 'tau' in r.get('r1', {}) and 'tau' in r.get(f'r{rounds}', {})]
def agg(lab):
    tp = sum(r[lab]['e_tp'] for r in ok); mass = sum(r[lab]['mass'] for r in ok); gold = sum(r[lab]['gold'] for r in ok)
    return tp / gold, tp / mass
for lab in ('r1', f'r{rounds}'):
    rec, prec = agg(lab); print(f'{lab}: pairs {len(ok)} recall {rec:.4f} precision {prec:.4f} median tau {np.median([r[lab]["tau"] for r in ok]):.3f} mean lp {np.mean([r[lab]["lp"] for r in ok]):.2f}')
d = np.array([r[f'r{rounds}']['lp'] - r['r1']['lp'] for r in ok]); print('lp gain (nats): median %.2f max %.2f; pairs with gain > 1: %d' % (np.median(d), d.max(), (d > 1).sum()))
