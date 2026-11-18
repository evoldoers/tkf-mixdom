#!/usr/bin/env python3
"""Gradient of the MixDom MAP objective (observed LL + log-prior, as logged by
train_pfam.py) at given parameter points, via Fisher's identity: one E-step on
the fixed 1000-pair corpus gives the sufficient statistics, and the gradient
of the Q-function + log-prior at the same parameters (adam_loss_from_delta)
equals the gradient of the observed-data MAP objective.  Used to decide
whether the point EM converges to is a stationary point (local maximum) or
whether the closed-form M-step stops short of one (an M-step bug).

Also reports a one-dimensional finite-difference check of the objective
along the main insertion rate, so that the Fisher-identity gradient is
validated against the logged objective itself.

Usage (from ~/tkf-mixdom/python):
  .venv/bin/python experiments/em_vs_gradient/mixdom_gradient_at_point.py \
      --points init=results_mixdom/init_d3f1_seed0.npz:maraschino \
               em=results_mixdom/em_d3f1.npz:checkpoint \
               adam=results_mixdom/adam_d3f1_lr0.01_long.npz:checkpoint
"""
import argparse, json, os, sys, types
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
PY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PY)
sys.argv_backup = sys.argv; sys.argv = ['x']
import jax, jax.numpy as jnp
import train_pfam
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.models.mixdom import state_types as mixdom_state_types
from tkfmixdom.jax.simulate.msa import alignment_to_states
from tkfmixdom.jax.util.pair_loader import PairStream
from tkfmixdom.jax.train.adam_train import to_unconstrained, to_constrained, adam_loss_from_delta
sys.argv = sys.argv_backup

N_DOM, N_FRAG = 3, 1
PRIORS = dict(ins_prior=(2.0, 10.0), del_prior=(2.0, 10.0), dom_dirichlet=1.5,
              frag_dirichlet=1.5, ext_alpha=2.0, ext_beta=3.0)


def load_point(spec):
    path, kind = spec.split(':')
    if kind == 'maraschino':
        return train_pfam._load_maraschino_as_train_params(path, N_DOM, N_FRAG)
    d = np.load(path, allow_pickle=True)
    p = {k: (float(d[k]) if d[k].shape == () else d[k].copy())
         for k in ('main_ins', 'main_del', 'dom_ins', 'dom_del', 'dom_weights',
                   'frag_weights', 'ext_rates', 'dom_Qs', 'dom_pis', 'dom_S_exch')}
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--points', nargs='+', required=True, help='name=path:kind (kind = maraschino|checkpoint)')
    ap.add_argument('--precompiled', default='pfam/precompiled_sub1000_seed0')
    ap.add_argument('--fd-eps', type=float, default=1e-4)
    ap.add_argument('--out', default='experiments/em_vs_gradient/results_mixdom/gradient_at_points.json')
    args = ap.parse_args()

    Q_lg, pi_lg = rate_matrix_lg()
    st = mixdom_state_types(N_DOM, N_FRAG); N = int(np.asarray(st).shape[0])
    pairs = PairStream(args.precompiled, seed=0).sample_batch(10**9)
    pairs = [p for p in pairs if len(p[0]) > 0 and len(p[1]) > 0]
    print(f"{len(pairs)} pairs, {N} states", flush=True)
    args_ns = types.SimpleNamespace(**{k: list(v) if isinstance(v, tuple) else v for k, v in PRIORS.items()},
                                    classdist_dirichlet=1.0, fb_scan_mode='auto')

    def estep(params):
        p = dict(params)
        p['dom_Qs'] = np.asarray(p['dom_Qs'])
        return train_pfam._process_pairs_batched(
            pairs, p, st, Q_lg, pi_lg, N, alignment_to_states,
            dom_Qs=p['dom_Qs'], dom_pis=p['dom_pis'], n_dom=N_DOM, n_frag=N_FRAG,
            fb_scan_mode=train_pfam._resolve_fb_scan_mode(args_ns, N))

    def objective(params):
        delta = estep(params)
        obs = float(delta['total_ll'])
        lp = float(train_pfam._log_prior(params, args_ns, Q_lg=Q_lg, pi_lg=pi_lg))
        return obs, lp, delta

    def loss_fn(uc, delta_jax):
        return adam_loss_from_delta(uc, delta_jax, N_DOM, N_FRAG, **PRIORS)

    results = {}
    for spec in args.points:
        name, rest = spec.split('=', 1)
        params = load_point(rest)
        # make sure dom_Qs is consistent with (S, pi) as to_constrained would build it
        uc = to_unconstrained(params, N_DOM, N_FRAG)
        params_c = to_constrained(uc, N_DOM, N_FRAG)
        params_c = {k: (np.asarray(v) if hasattr(v, 'shape') else v) for k, v in params_c.items()}
        obs, lp, delta = objective(params_c)
        delta_jax = jax.tree.map(jnp.asarray, delta)
        g = jax.grad(loss_fn)(uc, delta_jax)          # gradient of -(Q + prior)/n_pairs
        n_pairs = float(delta.get('n_pairs', len(pairs)))
        groups = {}
        for k, v in g.items():
            gv = -np.asarray(v) * n_pairs               # d(MAP objective)/d uc_k, in nats
            groups[k] = dict(norm=float(np.linalg.norm(gv)), max_abs=float(np.max(np.abs(gv))), size=int(gv.size))
        indel_keys = [k for k in groups if not (k.startswith('log_dom_S') or k.startswith('log_dom_pi'))]
        subst_keys = [k for k in groups if k not in indel_keys]
        tot = lambda keys: float(np.sqrt(sum(groups[k]['norm'] ** 2 for k in keys)))
        rec = dict(obs_ll=obs, log_prior=lp, map_ll=obs + lp, grad_norm_indel=tot(indel_keys),
                   grad_norm_subst=tot(subst_keys), grad_groups=groups,
                   indel_params=dict(main_ins=float(params_c['main_ins']), main_del=float(params_c['main_del']),
                                     dom_ins=np.asarray(params_c['dom_ins']).tolist(), dom_del=np.asarray(params_c['dom_del']).tolist(),
                                     dom_weights=np.asarray(params_c['dom_weights']).tolist(),
                                     ext_rates=np.asarray(params_c['ext_rates']).ravel().tolist()))
        # finite-difference check along log(main_ins): compare the Fisher-identity
        # derivative with a central difference of the logged objective.
        key = 'log_main_ins' if 'log_main_ins' in g else next(k for k in g if 'main_ins' in k)
        e = args.fd_eps
        vals = []
        for sgn in (+1, -1):
            uc2 = dict(uc); uc2[key] = uc[key] + sgn * e
            p2 = to_constrained(uc2, N_DOM, N_FRAG)
            p2 = {k: (np.asarray(v) if hasattr(v, 'shape') else v) for k, v in p2.items()}
            o2, l2, _ = objective(p2); vals.append(o2 + l2)
        fd = (vals[0] - vals[1]) / (2 * e)
        fisher = float(-np.asarray(g[key]) * n_pairs)
        rec['fd_check'] = dict(param=key, finite_difference=fd, fisher_identity=fisher)
        results[name] = rec
        print(f"[{name}] obs_LL={obs:.1f} log_prior={lp:.1f} MAP={obs+lp:.1f} | "
              f"|grad| indel={rec['grad_norm_indel']:.3g} subst={rec['grad_norm_subst']:.3g} | "
              f"d/d{key}: FD={fd:.4g} Fisher={fisher:.4g}", flush=True)
        for k, v in groups.items():
            print(f"    {k:28s} norm={v['norm']:.3g} max|g|={v['max_abs']:.3g} (n={v['size']})", flush=True)
    json.dump(results, open(args.out, 'w'), indent=1, default=float)
    print('saved', args.out)


if __name__ == '__main__':
    main()
