#!/usr/bin/env python3
"""Batch worker: score multiple assays with one model in one process."""
import os, sys, json, csv, time, gc
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_llvm_disable_expensive_passes=true'

import numpy as np
from scipy.special import softmax, expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ORDER)}

def seq_to_int(seq):
    return np.array([AA_TO_IDX.get(c, -1) for c in seq], dtype=int)

def load_params(params_path, fmt):
    d = np.load(params_path, allow_pickle=True)
    if fmt == 'bw':
        params = {
            'main_ins': float(d['main_ins']),
            'main_del': float(d['main_del']),
            'dom_ins': np.array(d['dom_ins'], dtype=np.float64),
            'dom_del': np.array(d['dom_del'], dtype=np.float64),
            'dom_weights': np.array(d['dom_weights'], dtype=np.float64),
            'frag_weights': np.array(d['frag_weights'], dtype=np.float64),
            'ext_rates': np.array(d['ext_rates'], dtype=np.float64),
        }
        if 'dom_pis' in d:
            params['dom_pis'] = np.array(d['dom_pis'], dtype=np.float64)
        if 'dom_S_exch' in d:
            params['dom_S_exch'] = np.array(d['dom_S_exch'], dtype=np.float64)
        n_dom = len(params['dom_ins'])
        n_frag = params['frag_weights'].shape[1] if params['frag_weights'].ndim > 1 else 1
    elif fmt == 'maraschino':
        n_dom = int(d['n_domains'])
        n_frag = 1
        params = {
            'main_ins': float(np.exp(d['log_lam0'])),
            'main_del': float(np.exp(d['log_mu0'])),
            'dom_ins': np.exp(np.array(d['log_lam'], dtype=np.float64)),
            'dom_del': np.exp(np.array(d['log_mu'], dtype=np.float64)),
            'dom_weights': softmax(np.array(d['log_v'], dtype=np.float64)),
            'frag_weights': np.ones((n_dom, 1), dtype=np.float64),
            'ext_rates': expit(np.array(d['logit_r'], dtype=np.float64)).reshape(-1, 1),
        }
        if 'log_pi' in d:
            params['dom_pis'] = softmax(np.array(d['log_pi'], dtype=np.float64), axis=-1)
    return params, n_dom, n_frag

def build_pair_hmm(params, n_dom, n_frag, tau):
    from tkfmixdom.jax.models.mixdom import build_nested_trans, state_types as st_fn
    from tkfmixdom.jax.distill.maraschino import build_rate_matrix
    from tkfmixdom.jax.core.ctmc import transition_matrix
    from tkfmixdom.jax.dp.hmm import safe_log

    chi, _ = build_nested_trans(
        jnp.array(float(params['main_ins'])),
        jnp.array(float(params['main_del'])),
        jnp.array(float(tau)),
        jnp.array(params['dom_ins']),
        jnp.array(params['dom_del']),
        jnp.array(params['dom_weights']),
        jnp.array(params['frag_weights']),
        jnp.array(params['ext_rates']),
    )
    log_chi = safe_log(chi)
    st = st_fn(n_dom, n_frag)

    # Per-domain substitution matrices
    if 'dom_S_exch' in params and 'dom_pis' in params:
        S_exch = jnp.array(params['dom_S_exch'])
        pis = jnp.array(params['dom_pis'])
        sub_matrices = jax.vmap(
            lambda S, pi: transition_matrix(build_rate_matrix(S, pi), tau)
        )(S_exch, pis)
    elif 'dom_pis' in params:
        from tkfmixdom.jax.distill.maraschino import get_lg08
        S_lg, _ = get_lg08()
        pis = jnp.array(params['dom_pis'])
        sub_matrices = jax.vmap(
            lambda pi_n: transition_matrix(build_rate_matrix(S_lg, pi_n), tau)
        )(pis)
    else:
        from tkfmixdom.jax.distill.maraschino import get_lg08
        S_lg, pi_lg = get_lg08()
        Q = build_rate_matrix(S_lg, pi_lg)
        sub_matrix = transition_matrix(Q, tau)
        sub_matrices = jnp.tile(sub_matrix[None], (n_dom, 1, 1))
        pis = jnp.tile(pi_lg[None], (n_dom, 1))

    pi = jnp.sum(jnp.array(params.get('dom_weights', np.ones(n_dom)/n_dom))[:, None]
                 * pis, axis=0)
    pi = pi / pi.sum()
    return log_chi, st, sub_matrices, pis, pi, n_dom, n_frag

def main():
    args = json.loads(sys.argv[1])
    from tkfmixdom.jax.dp.hmm import forward_2d_banded, pair_hmm_emissions_per_domain

    params, n_dom, n_frag = load_params(args['params_path'], args['format'])
    log_chi, st, sub_matrices, pis, pi, n_dom, n_frag = build_pair_hmm(
        params, n_dom, n_frag, args['tau'])

    band_width = 20
    all_results = []

    for assay_info in args['assays']:
        assay_csv = assay_info['csv']
        wt_seq = assay_info['wt_seq']

        with open(assay_csv) as f:
            rows = list(csv.DictReader(f))

        wt_int = seq_to_int(wt_seq)
        wt_int = wt_int[wt_int >= 0]
        Lx = len(wt_int)
        wt_jnp = jnp.array(wt_int)

        predictions = []
        dms_scores = []
        for row in rows:
            mut_int = seq_to_int(row['mutated_sequence'])
            mut_int = mut_int[mut_int >= 0]
            if len(mut_int) == 0:
                continue
            Ly = len(mut_int)
            mut_jnp = jnp.array(mut_int)
            bc = jnp.round(jnp.arange(Lx + 1) * Ly / max(Lx, 1)).astype(jnp.int32)
            bc = jnp.clip(bc, 0, Ly)
            log_emit = pair_hmm_emissions_per_domain(
                st, wt_jnp, mut_jnp, sub_matrices, pis, n_dom, n_frag)
            log_prob, _ = forward_2d_banded(log_chi, st, wt_jnp, mut_jnp,
                                             None, None, bc, band_width,
                                             log_emit_table=log_emit)
            predictions.append(float(log_prob))
            dms_scores.append(float(row['DMS_score']))
            del log_prob

        all_results.append({
            'assay_csv': assay_csv,
            'predictions': predictions,
            'dms_scores': dms_scores,
            'n_scored': len(predictions),
        })
        # Print progress to stderr
        print(f"  scored {assay_csv}: {len(predictions)} variants", file=sys.stderr)
        gc.collect()

    print(json.dumps(all_results))

if __name__ == '__main__':
    main()
