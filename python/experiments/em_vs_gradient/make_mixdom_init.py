#!/usr/bin/env python3
"""Write a MixDom dNf1 initial-parameter file (Maraschino format) for a given
seed, using train_pfam's own initialiser, so EM and Adam can start from the
same point.  Usage: make_mixdom_init.py <seed> <out.npz>"""
import sys, types, os
os.environ.setdefault("JAX_ENABLE_X64", "1"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
seed, out = int(sys.argv[1]), sys.argv[2]
N_DOM = int(sys.argv[3]) if len(sys.argv) > 3 else 3   # optional third argument: number of domain types (default 3)
_a = sys.argv; sys.argv = ['x']
import train_pfam
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.models.mixdom_init import init_mixdom2_params_from_args
Q_lg, pi_lg = rate_matrix_lg()
args = types.SimpleNamespace(init_ins=0.1, init_del=0.2, seed=seed, n_classes=0, estimate_subst=True,
                             pi_init_noise_frac=0.2, classdist_init=None, class_pi_init=None,
                             classdist_noise_frac=0.0, class_pis_from_dom_pis=False, n_tied=1,
                             banded_frag_init=False, p_ext=0.6, estimate_sexch=None, subst_mode='standard')
params = init_mixdom2_params_from_args(args, N_DOM, 1, Q_lg, pi_lg, log_fn=lambda *a, **k: None)
train_pfam._save_params_as_maraschino(out, params, N_DOM, 1)
print('wrote', out)
