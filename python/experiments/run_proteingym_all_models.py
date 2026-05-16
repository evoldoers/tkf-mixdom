#!/usr/bin/env python3
"""Comprehensive ProteinGym DMS indel evaluation across all MixDom models.

For each model, scores all indel assays (protein length <= 200 aa) using
the pair HMM Forward algorithm. Runs each assay in a subprocess to avoid
JAX/XLA memory accumulation that causes LLVM OOM.

Usage:
    python run_proteingym_all_models.py
    python run_proteingym_all_models.py --max-len 100
    python run_proteingym_all_models.py --models bw_d3f2,mara_d3
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import numpy as np
from scipy import stats
from scipy.special import softmax, expit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

MODEL_DEFS = {
    'bw_d3f2': {
        'path': os.path.join(REPO_ROOT, 'params/best/bw_d3f2_fullseed_15iter.npz'),
        'format': 'bw',
        'label': 'BW d3f2 fullseed',
    },
    'bw_d8f2': {
        'path': os.path.join(REPO_ROOT, 'params/best/bw_d8f2_fullseed_15iter.npz'),
        'format': 'bw',
        'label': 'BW d8f2 fullseed',
    },
    'svi_d3f2': {
        'path': os.path.join(REPO_ROOT, 'params/best/svi_d3f2_fullseed_15iter.npz'),
        'format': 'bw',
        'label': 'SVI d3f2 fullseed',
    },
    'mara_d3': {
        'path': os.path.expanduser('~/tkf-mixdom/python/pfam/maraschino_d3_trainsplit_entreg.npz'),
        'format': 'maraschino',
        'label': 'Maraschino d3 ent-reg',
    },
    'mara_d8': {
        'path': os.path.expanduser('~/tkf-mixdom/python/pfam/maraschino_d8_trainsplit_entreg.npz'),
        'format': 'maraschino',
        'label': 'Maraschino d8 ent-reg',
    },
}


def get_assay_list(proteingym_dir, ref_csv, max_len=200):
    """Get list of (assay_name, wt_seq, protein_length) sorted by length."""
    ref = {}
    if os.path.exists(ref_csv):
        with open(ref_csv) as f:
            for row in csv.DictReader(f):
                ref[row['DMS_id']] = row['target_seq']

    assays = []
    for fname in sorted(os.listdir(proteingym_dir)):
        if not fname.endswith('.csv'):
            continue
        assay_name = fname.replace('.csv', '')
        wt_seq = ref.get(assay_name)
        if wt_seq is None:
            continue
        prot_len = len(wt_seq)
        if prot_len <= max_len:
            assays.append((assay_name, wt_seq, prot_len))

    assays.sort(key=lambda x: x[2])
    return assays


def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    if not all_results:
        print("No results to display.")
        return

    models = sorted(set(r['model'] for r in all_results))
    assays = sorted(set(r['assay'] for r in all_results),
                    key=lambda a: next((r['protein_length'] for r in all_results
                                        if r['assay'] == a), 0))

    lookup = {}
    for r in all_results:
        lookup[(r['model'], r['assay'])] = r['rho']

    model_labels = {r['model']: r['model_label'] for r in all_results}
    col_width = 12
    header = f"{'Assay':<50} {'Len':>4}"
    for m in models:
        header += f" {model_labels.get(m, m):>{col_width}}"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")

    for assay in assays:
        prot_len = next((r['protein_length'] for r in all_results
                         if r['assay'] == assay), '?')
        line = f"{assay:<50} {prot_len:>4}"
        for m in models:
            rho = lookup.get((m, assay))
            if rho is not None:
                line += f" {rho:>{col_width}.4f}"
            else:
                line += f" {'--':>{col_width}}"
        print(line)

    print(f"{'-'*len(header)}")
    for stat_name, stat_fn in [('MEAN', np.mean), ('MEDIAN', np.median), ('STD', np.std)]:
        line = f"{stat_name:<50} {'':>4}"
        for m in models:
            rhos = [lookup[(m, a)] for a in assays if (m, a) in lookup]
            if rhos:
                line += f" {stat_fn(rhos):>{col_width}.4f}"
            else:
                line += f" {'--':>{col_width}}"
        print(line)
    print(f"{'='*len(header)}")


# ---- Subprocess worker script ----
WORKER_SCRIPT = r'''#!/usr/bin/env python3
"""Worker: score one assay with one model. Runs in subprocess to avoid memory leaks."""
import os, sys, json, csv, time
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_llvm_disable_expensive_passes=true'

import numpy as np
from scipy import stats
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
    params_path = args['params_path']
    fmt = args['format']
    tau = args['tau']
    assay_csv = args['assay_csv']
    wt_seq = args['wt_seq']

    from tkfmixdom.jax.dp.hmm import forward_2d_banded, pair_hmm_emissions_per_domain

    params, n_dom, n_frag = load_params(params_path, fmt)
    log_chi, st, sub_matrices, pis, pi, n_dom, n_frag = build_pair_hmm(
        params, n_dom, n_frag, tau)

    # Read variants
    with open(assay_csv) as f:
        rows = list(csv.DictReader(f))

    wt_int = seq_to_int(wt_seq)
    wt_int = wt_int[wt_int >= 0]
    Lx = len(wt_int)
    wt_jnp = jnp.array(wt_int)
    band_width = 20

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

    result = {'predictions': predictions, 'dms_scores': dms_scores, 'n_scored': len(predictions)}
    print(json.dumps(result))

if __name__ == '__main__':
    main()
'''


def score_assay_subprocess(assay_csv, wt_seq, params_path, fmt, tau):
    """Score an assay by running a subprocess (avoids JAX memory leaks)."""
    worker_path = os.path.join(os.path.dirname(__file__), '_worker_score.py')
    if not os.path.exists(worker_path):
        with open(worker_path, 'w') as f:
            f.write(WORKER_SCRIPT)

    args_json = json.dumps({
        'params_path': params_path,
        'format': fmt,
        'tau': tau,
        'assay_csv': assay_csv,
        'wt_seq': wt_seq,
    })

    env = os.environ.copy()
    env['JAX_PLATFORMS'] = 'cpu'
    xla_flags = env.get('XLA_FLAGS', '')
    if '--xla_llvm_disable_expensive_passes' not in xla_flags:
        env['XLA_FLAGS'] = xla_flags + ' --xla_llvm_disable_expensive_passes=true'

    result = subprocess.run(
        [sys.executable, worker_path, args_json],
        capture_output=True, text=True, env=env,
        cwd=REPO_ROOT, timeout=3600
    )

    if result.returncode != 0:
        raise RuntimeError(f"Worker failed: {result.stderr[-500:]}")

    # Parse last line as JSON (ignore JAX warnings on stderr)
    lines = result.stdout.strip().split('\n')
    data = json.loads(lines[-1])
    return data['predictions'], data['dms_scores'], data['n_scored']


def score_assay_subprocess_batch(assay_csvs_and_wt, params_path, fmt, tau):
    """Score multiple assays in a single subprocess (reduces JIT overhead).

    Processes up to BATCH_SIZE assays before returning, to limit memory.
    """
    worker_path = os.path.join(os.path.dirname(__file__), '_worker_score_batch.py')
    # Write batch worker if needed
    if not os.path.exists(worker_path):
        with open(worker_path, 'w') as f:
            f.write(WORKER_BATCH_SCRIPT)

    args_json = json.dumps({
        'params_path': params_path,
        'format': fmt,
        'tau': tau,
        'assays': [{'csv': csv_path, 'wt_seq': wt} for csv_path, wt in assay_csvs_and_wt],
    })

    env = os.environ.copy()
    env['JAX_PLATFORMS'] = 'cpu'
    xla_flags = env.get('XLA_FLAGS', '')
    if '--xla_llvm_disable_expensive_passes' not in xla_flags:
        env['XLA_FLAGS'] = xla_flags + ' --xla_llvm_disable_expensive_passes=true'

    result = subprocess.run(
        [sys.executable, worker_path, args_json],
        capture_output=True, text=True, env=env,
        cwd=REPO_ROOT, timeout=7200
    )

    if result.returncode != 0:
        raise RuntimeError(f"Batch worker failed: {result.stderr[-500:]}")

    lines = result.stdout.strip().split('\n')
    data = json.loads(lines[-1])
    return data  # list of {predictions, dms_scores, n_scored, assay_csv}


WORKER_BATCH_SCRIPT = r'''#!/usr/bin/env python3
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
'''


def count_variants(assay_csv):
    """Count number of variants in an assay CSV."""
    with open(assay_csv) as f:
        return sum(1 for _ in csv.DictReader(f))


def evaluate_model(model_key, model_def, assays, proteingym_dir, tau, results_dir,
                    batch_size=8):
    """Evaluate a model on all assays using subprocess batches."""
    path = model_def['path']
    if not os.path.exists(path):
        alt_path = path.replace('/.claude/worktrees/agent-a24f5243/', '/')
        if os.path.exists(alt_path):
            path = alt_path

    if not os.path.exists(path):
        print(f"  SKIP {model_key}: params not found at {path}")
        return []

    print(f"\n{'='*60}")
    print(f"Model: {model_def['label']} ({model_key})")
    print(f"Params: {path}")
    print(f"Batch size: {batch_size} assays per subprocess")
    print(f"{'='*60}")

    results = []
    total_t0 = time.time()

    # Process in batches
    for batch_start in range(0, len(assays), batch_size):
        batch = assays[batch_start:batch_start + batch_size]
        batch_assay_info = []
        for assay_name, wt_seq, prot_len in batch:
            csv_path = os.path.join(proteingym_dir, assay_name + '.csv')
            if os.path.exists(csv_path):
                batch_assay_info.append((csv_path, wt_seq, assay_name, prot_len))

        if not batch_assay_info:
            continue

        print(f"  Batch {batch_start//batch_size + 1}: assays {batch_start+1}-{batch_start+len(batch)}", flush=True)
        for csv_path, wt_seq, name, plen in batch_assay_info:
            n_vars = count_variants(csv_path)
            print(f"    {name} (len={plen}, {n_vars} variants)", flush=True)

        t0 = time.time()
        try:
            batch_results = score_assay_subprocess_batch(
                [(info[0], info[1]) for info in batch_assay_info],
                path, model_def['format'], tau
            )
        except Exception as e:
            print(f"    BATCH FAILED: {e}")
            # Fall back to individual subprocess per assay
            batch_results = []
            for csv_path, wt_seq, name, plen in batch_assay_info:
                try:
                    preds, dms, n = score_assay_subprocess(csv_path, wt_seq, path,
                                                            model_def['format'], tau)
                    batch_results.append({
                        'assay_csv': csv_path,
                        'predictions': preds,
                        'dms_scores': dms,
                        'n_scored': n
                    })
                except Exception as e2:
                    print(f"    {name}: FAILED ({e2})")
                    continue

        elapsed = time.time() - t0

        for res, (csv_path, wt_seq, assay_name, prot_len) in zip(batch_results, batch_assay_info):
            preds = res['predictions']
            dms = res['dms_scores']
            n = res['n_scored']
            if n >= 10:
                rho, pval = stats.spearmanr(preds, dms)
                print(f"    {assay_name}: rho={rho:.4f} (p={pval:.2e}), {n} variants")
                results.append({
                    'model': model_key,
                    'model_label': model_def['label'],
                    'assay': assay_name,
                    'protein_length': prot_len,
                    'n_variants': n,
                    'rho': rho,
                    'pval': pval,
                    'time': elapsed / len(batch_assay_info),
                    'tau': tau,
                })
            else:
                print(f"    {assay_name}: too few variants ({n}), skipped")

        print(f"    batch time: {elapsed:.1f}s")

    total_elapsed = time.time() - total_t0
    if results:
        rhos = [r['rho'] for r in results]
        print(f"\n  {model_key} summary: mean rho={np.mean(rhos):.4f} "
              f"(+/-{np.std(rhos):.4f}), {len(results)} assays, {total_elapsed:.1f}s total")

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, f'{model_key}_tau{tau:.2f}.csv')
    if results:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"  Saved: {out_csv}")

    return results


def main():
    parser = argparse.ArgumentParser(description='ProteinGym indel evaluation across all models')
    parser.add_argument('--proteingym-dir', type=str,
                        default=os.path.expanduser(
                            '~/bio-datasets/data/proteingym/DMS_ProteinGym_indels/'))
    parser.add_argument('--ref-csv', type=str,
                        default=os.path.expanduser(
                            '~/bio-datasets/data/proteingym/DMS_indels.csv'))
    parser.add_argument('--max-len', type=int, default=200,
                        help='Maximum protein length (default: 200)')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='Evolutionary time (default: 0.1)')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model keys (default: all)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Assays per subprocess batch (default: 8)')
    parser.add_argument('--out-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'proteingym_results'))
    args = parser.parse_args()

    assays = get_assay_list(args.proteingym_dir, args.ref_csv, args.max_len)
    print(f"Found {len(assays)} indel assays with protein length <= {args.max_len}")
    for name, _, plen in assays[:5]:
        print(f"  {name} (len={plen})")
    if len(assays) > 5:
        print(f"  ... and {len(assays) - 5} more")

    if args.models:
        model_keys = [k.strip() for k in args.models.split(',')]
    else:
        model_keys = list(MODEL_DEFS.keys())

    print(f"\nModels to evaluate: {model_keys}")

    all_results = []
    for model_key in model_keys:
        if model_key not in MODEL_DEFS:
            print(f"Unknown model: {model_key}")
            continue

        results = evaluate_model(
            model_key, MODEL_DEFS[model_key],
            assays, args.proteingym_dir, args.tau, args.out_dir,
            batch_size=args.batch_size
        )
        all_results.extend(results)

    # Save combined results
    if all_results:
        combined_csv = os.path.join(os.path.dirname(__file__),
                                     'proteingym_all_models_comparison.csv')
        with open(combined_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nCombined results saved: {combined_csv}")

    print_comparison_table(all_results)


if __name__ == '__main__':
    main()
