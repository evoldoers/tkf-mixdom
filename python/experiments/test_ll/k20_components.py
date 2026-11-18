#!/usr/bin/env python3
"""Export a K-component TKF92 mixture (default pfam/tkf92_mixture_K20_train.npz; usage: k20_components.py [npz] [subdir], a MixDom1-format checkpoint with one
domain per component) as per-component TKF92 parameter files for score_matchaligned_val.py:
  experiments/test_ll/k20/comp_<k>.json   {"ins_rate", "del_rate", "ext"}
  experiments/test_ll/k20/comp_<k>.npz    Q (20x20 rate matrix, mean rate 1 not enforced: as fitted), pi
  experiments/test_ll/k20/weights.json    mixture weights
"""
import json, os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PY = os.path.dirname(os.path.dirname(HERE)); os.chdir(PY)
NPZ = sys.argv[1] if len(sys.argv) > 1 else 'pfam/tkf92_mixture_K20_train.npz'
out = os.path.join(HERE, sys.argv[2] if len(sys.argv) > 2 else 'k20'); os.makedirs(out, exist_ok=True)
d = np.load(NPZ, allow_pickle=True)
K = int(d['dom_weights'].shape[0])
for k in range(K):
    json.dump(dict(ins_rate=float(d['dom_ins'][k]), del_rate=float(d['dom_del'][k]), ext=float(np.asarray(d['ext_rates'])[k].ravel()[0])),
              open(f'{out}/comp_{k}.json', 'w'))
    np.savez(f'{out}/comp_{k}.npz', Q=np.asarray(d['dom_Qs'])[k], pi=np.asarray(d['dom_pis'])[k])
json.dump(dict(weights=[float(w) for w in d['dom_weights']]), open(f'{out}/weights.json', 'w'))
print(f'wrote {K} components to {out}; weights sum {float(np.sum(d["dom_weights"])):.6f}')
