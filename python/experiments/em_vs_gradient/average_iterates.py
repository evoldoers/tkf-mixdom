#!/usr/bin/env python3
"""Polyak-Ruppert average of svi-BW iterates saved with train_pfam.py --save-every-iter (<ckpt>_iter<k>.npz).

Averages every float model-parameter array (indel rates, weights, extension rates, per-domain and per-class
exchangeabilities and frequencies) arithmetically over iterations [start, end], rebuilds dom_Qs from the averaged
dom_S_exch and dom_pis, and writes a checkpoint with the structure of the last iterate in the window (so the
scorers load it unchanged).
Usage: average_iterates.py <ckpt.npz> <start> <end> <out.npz>
"""
import sys, os
import numpy as np
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tkfmixdom.jax.core.ctmc import build_Q_from_S_pi

PARAM_KEYS = ['main_ins', 'main_del', 'dom_ins', 'dom_del', 'dom_weights', 'frag_weights', 'ext_rates',
              'dom_pis', 'dom_S_exch', 'class_pis', 'class_S_exch', 'classdist']

ckpt, a, b, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
base = ckpt[:-4] if ckpt.endswith('.npz') else ckpt
paths = [f'{base}_iter{k}.npz' for k in range(a, b + 1)]
missing = [p for p in paths if not os.path.exists(p)]
if missing:
    sys.exit(f'{len(missing)} iterate files missing, e.g. {missing[0]}')
last = dict(np.load(paths[-1], allow_pickle=True))
keys = [k for k in PARAM_KEYS if k in last]
acc = {k: np.zeros_like(np.asarray(last[k], np.float64)) for k in keys}
for p in paths:
    d = np.load(p, allow_pickle=True)
    for k in keys:
        acc[k] += np.asarray(d[k], np.float64)
for k in keys:
    last[k] = acc[k] / len(paths)
if 'dom_S_exch' in last and 'dom_pis' in last:
    last['dom_Qs'] = np.asarray(jax.vmap(build_Q_from_S_pi)(jnp.asarray(last['dom_S_exch']), jnp.asarray(last['dom_pis'])))
np.savez(out, **last)
print(f'averaged {len(paths)} iterates ({a}..{b}) over {keys} -> {out}')
