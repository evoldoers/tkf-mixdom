#!/usr/bin/env python3
"""Maximum-likelihood branch length of every sequence pair of a benchmark under a MixDom checkpoint, as the
harness estimates it when it computes the pair's posteriors (pairwise_posteriors_mixdom: Newton-Raphson on the
forward likelihood), written as JSON {"family|name_i|name_j": tau} for fsa_gap_factor_from_cache.py
--tree-dist-json, so that the progressive decoder's guide tree can be built from the model's own distances.
Usage: dump_pair_tau.py <mixdom.npz> <balibase-dir> <out.json> [em-rounds, default 1]"""
import os, sys, json, time
os.environ.setdefault('JAX_ENABLE_X64', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
from tkfmixdom.jax.tree.fsa_anneal import pairwise_posteriors_mixdom
from tkfmixdom.util.msa_benchmark import parse_fasta
from tkfmixdom.jax.distill.maraschino import load_params as load_mixdom_params
from fsa_mixdom_pairhmm_oxbench import encode_seq, make_fsa_params

ckpt, bdir, out = sys.argv[1:4]; ROUNDS = int(sys.argv[4]) if len(sys.argv) > 4 else 1
params, n_dom, _ = load_mixdom_params(ckpt)
fsa_params, n_frag = make_fsa_params(params, n_dom)
in_dir = os.path.join(bdir, 'in')
tau = json.load(open(out)) if os.path.exists(out) else {}
fams = sorted(f for f in os.listdir(in_dir) if not f.startswith('.'))
t0 = time.time()
for fam in fams:
    raw = parse_fasta(os.path.join(in_dir, fam))
    seqs = {n: encode_seq(s) for n, s in raw.items()}; seqs = {n: s for n, s in seqs.items() if len(s) > 0}
    names = list(seqs)
    todo = [(a, b) for a in range(len(names)) for b in range(a + 1, len(names)) if f'{fam}|{names[a]}|{names[b]}' not in tau]
    for a, b in todo:
        try:
            _, t, _ = pairwise_posteriors_mixdom(jnp.asarray(seqs[names[a]], dtype=jnp.int32), jnp.asarray(seqs[names[b]], dtype=jnp.int32), fsa_params, n_dom, n_frag, n_em_rounds=ROUNDS)
            tau[f'{fam}|{names[a]}|{names[b]}'] = float(t)
        except Exception as e:
            jax.clear_caches(); print(f'  {fam} {names[a]} {names[b]}: {type(e).__name__}: {str(e)[:120]}', flush=True)
    if todo:
        json.dump(tau, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
        print(f'[{time.time() - t0:7.1f}s] {fam}: {len(todo)} pairs, {len(tau)} total', flush=True)
print('done', len(tau), 'pairs ->', out)
