#!/usr/bin/env python3
"""Realignment + partition reconstruction benchmark.

For each family in a given spec:
1. Load the unaligned remaining sequences (held-out removed)
2. Realign with FSA (d3f1 or d5f1) or MUSCLE
3. Build FastTree from the realignment
4. Run partition reconstruction (d3f1 or d5f1) to predict the held-out sequence
5. Score prediction via NW identity against the true sequence

Methods:
  fsa_d3f1_part_d3f1  — FSA d3f1 realignment -> partition d3f1 reconstruction
  fsa_d5f1_part_d5f1  — FSA d5f1 realignment -> partition d5f1 reconstruction
  muscle_part_d3f1    — MUSCLE realignment -> partition d3f1 reconstruction
  muscle_part_d5f1    — MUSCLE realignment -> partition d5f1 reconstruction

Usage:
  cd python && JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 uv run python -u \
      experiments/realign_partition_benchmark.py \
      --dataset unified_short \
      --methods fsa_d3f1_part_d3f1,muscle_part_d3f1 \
      --out experiments/realign_partition_unified_short.json
"""

import os
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

import sys
import json
import time
import copy
import shutil
import tempfile
import subprocess
import traceback
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
jax.config.update('jax_enable_x64', True)

from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT
from tkfmixdom.jax.tree.tree_varanc import infer_internal_presence, name_internal_nodes
from experiments.ancrec_benchmark import needleman_wunsch_identity, parse_sto
from experiments.partition_recon_adapter import (
    mixdom_model_from_params, run_partition_reconstruction_method,
    PartitionReconConfig,
)

# --- Constants ---
PYTHON_ROOT = Path(__file__).parent.parent
PFAM_DIR = "/home/yam/bio-datasets/data/pfam-seed"
TREEFAM_DIR = os.path.expanduser("~/bio-datasets/data/treefam/treefam_family_data")
SAVE_EVERY = 5

AA_CHARS = "ACDEFGHIKLMNPQRSTVWY"
AA_MAP = {c: i for i, c in enumerate(AA_CHARS)}

ALL_METHODS = [
    'fsa_d3f1_part_d3f1',
    'fsa_d5f1_part_d5f1',
    'muscle_part_d3f1',
    'muscle_part_d5f1',
]

t0 = time.time()
def log(msg):
    print(f'[{time.time()-t0:.0f}s] {msg}', flush=True)


# --- Encoding helpers ---

def encode_seq(seq_str):
    """Convert amino acid string to integer array. Non-standard -> -1."""
    return np.array([AA_MAP.get(c.upper(), -1) for c in seq_str if c not in '.-~ '],
                    dtype=np.int32)


def encode_aligned_seq(seq_str):
    """Convert aligned sequence string to integer array. Gaps -> -1."""
    result = []
    for c in seq_str:
        if c in '.-~ ':
            result.append(-1)
        else:
            result.append(AA_MAP.get(c.upper(), -1))
    return np.array(result, dtype=np.int32)


def msa_ints_to_strings(msa_dict):
    """Convert integer MSA dict {name: int_array} to {name: string}."""
    result = {}
    for name, row in msa_dict.items():
        chars = []
        for c in row:
            if 0 <= c < 20:
                chars.append(AA_CHARS[c])
            elif c == 20:
                chars.append('X')
            else:
                chars.append('-')
        result[name] = ''.join(chars)
    return result


def write_fasta(seqs, filepath):
    """Write sequences dict {name: string} to FASTA."""
    with open(filepath, 'w') as f:
        for name, seq in seqs.items():
            f.write(f'>{name}\n{seq}\n')


def parse_fasta(path):
    """Parse FASTA, return {name: seq_str}."""
    seqs = {}
    name = None
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('//'):
                continue
            if line.startswith('>'):
                if name is not None:
                    seqs[name] = ''.join(seq_parts)
                name = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
    if name is not None:
        seqs[name] = ''.join(seq_parts)
    return seqs


def remove_gapcols(msa_dict):
    """Remove columns where all sequences are gaps. Returns new dict."""
    names = list(msa_dict.keys())
    if not names:
        return {}
    L = len(msa_dict[names[0]])
    keep = [col for col in range(L)
            if any(msa_dict[n][col] >= 0 for n in names)]
    return {n: np.array([msa_dict[n][c] for c in keep], dtype=np.int32) for n in names}


# --- External tool runners ---

def run_fasttree(aln_path):
    """Run FastTree on aligned FASTA, return Newick string."""
    fasttree_bin = os.path.expanduser('~/bin/FastTree')
    if not os.path.exists(fasttree_bin):
        fasttree_bin = shutil.which('FastTree') or 'FastTree'
    result = subprocess.run(
        [fasttree_bin, '-quiet', '-lg'],
        stdin=open(aln_path),
        capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f'FastTree failed: {result.stderr}')
    return result.stdout.strip()


def run_muscle(input_path, output_path):
    """Run MUSCLE alignment. Tries v5 then v3 syntax."""
    muscle_bin = shutil.which('muscle') or os.path.expanduser('~/bin/muscle')
    if not os.path.exists(muscle_bin):
        raise FileNotFoundError('muscle binary not found in PATH or ~/bin/')
    result = subprocess.run(
        [muscle_bin, '-align', input_path, '-output', output_path],
        capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        # Fall back to MUSCLE v3 syntax
        result = subprocess.run(
            [muscle_bin, '-in', input_path, '-out', output_path],
            capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f'MUSCLE failed: {result.stderr}')


# --- MixDom params loading ---

_params_cache = {}

def load_mixdom_params(model_key):
    """Load MixDom params for a given model key ('d3f1' or 'd5f1').

    Returns (params_dict, n_dom, n_frag, fsa_params_dict).
    """
    if model_key in _params_cache:
        return _params_cache[model_key]

    from tkfmixdom.jax.distill.maraschino import load_params, build_rate_matrix
    import jax.numpy as jnp

    if model_key == 'd3f1':
        npz_path = str(PYTHON_ROOT / 'pfam' / 'svi_bw_d3f1_full_best_val.npz')
    elif model_key == 'd5f1':
        npz_path = str(PYTHON_ROOT / 'pfam' / 'svi_bw_d5f1_full_best_val.npz')
    else:
        raise ValueError(f'Unknown model key: {model_key}')

    log(f'Loading {model_key} params from {npz_path}')
    params, n_dom, n_cls = load_params(npz_path)

    # Build FSA params dict
    v = np.asarray(params['v'])
    pis = np.asarray(params['pi'])
    S_exch = np.asarray(params['S_exch'])
    avg_pi = np.einsum('n,na->a', v, pis)
    avg_pi = avg_pi / avg_pi.sum()
    avg_S = np.einsum('n,nab->ab', v, S_exch)
    fsa_params = {
        'main_ins': float(params['lam0']),
        'main_del': float(params['mu0']),
        'dom_ins': np.asarray(params['lam']),
        'dom_del': np.asarray(params['mu']),
        'dom_weights': np.asarray(params['v']),
        'frag_weights': np.asarray(params['frag_weights']),
        'ext_rates': np.asarray(params['r_frags']),
        'S_exch': np.asarray(params['S_exch']),
        'pi': avg_pi,
        'Q': np.asarray(build_rate_matrix(jnp.array(avg_S), jnp.array(avg_pi))),
    }
    fw = fsa_params['frag_weights']
    n_frag = fw.shape[1] if fw.ndim > 1 else 1

    log(f'  {model_key}: n_dom={n_dom}, n_frag={n_frag}')
    _params_cache[model_key] = (params, n_dom, n_frag, fsa_params)
    return params, n_dom, n_frag, fsa_params


# --- Dataset loading ---

def load_pfam_unaligned(family_id, remaining_names):
    """Load unaligned sequences for a Pfam family.

    Returns dict {name: np.array of ints}.
    """
    sto_path = os.path.join(PFAM_DIR, f'{family_id}.sto')
    seqs_str = parse_sto(sto_path)  # {name: aligned_string}

    int_seqs = {}
    for name in remaining_names:
        if name not in seqs_str:
            raise ValueError(f'{name} not in {sto_path}')
        # Strip gaps to get unaligned sequence, encode to ints
        ungapped = seqs_str[name].replace('-', '').replace('.', '').replace('~', '')
        int_seqs[name] = encode_seq(ungapped)
    return int_seqs


def load_treefam_unaligned(family_id, remaining_names):
    """Load unaligned sequences for a TreeFam family.

    Returns dict {name: np.array of ints}.
    """
    fasta_path = os.path.join(TREEFAM_DIR, f'{family_id}.aa.fasta')
    raw_seqs = parse_fasta(fasta_path)

    int_seqs = {}
    for name in remaining_names:
        if name not in raw_seqs:
            raise ValueError(f'{name} not in {fasta_path}')
        ungapped = raw_seqs[name].replace('-', '').replace('.', '')
        int_seqs[name] = encode_seq(ungapped)
    return int_seqs


# --- Alignment methods ---

def align_fsa(int_seqs, model_key):
    """Run FSA alignment with MixDom model. Returns (msa_int_dict, C)."""
    from tkfmixdom.jax.tree.fsa_anneal import fsa_align

    _, n_dom, n_frag, fsa_params = load_mixdom_params(model_key)

    jax.clear_caches()  # OOM prevention; prefer geometric padding

    try:
        msa_dict, msa_len = fsa_align(
            int_seqs, model='mixdom', pair_selection='erdos_renyi',
            mixdom_params=fsa_params, n_dom=n_dom, n_frag=n_frag,
            n_anneal_iterations=3)
    except Exception as e:
        if 'RESOURCE_EXHAUSTED' in str(e) or 'OUT_OF_MEMORY' in str(e):
            jax.clear_caches()  # OOM prevention; prefer geometric padding
            cpu = jax.devices('cpu')[0]
            with jax.default_device(cpu):
                msa_dict, msa_len = fsa_align(
                    int_seqs, model='mixdom', pair_selection='erdos_renyi',
                    mixdom_params=fsa_params, n_dom=n_dom, n_frag=n_frag,
                    n_anneal_iterations=3)
        else:
            raise

    # Convert to int arrays
    msa_int = {}
    for name, row in msa_dict.items():
        msa_int[name] = np.array(row, dtype=np.int32)

    # Remove gap-only columns
    msa_int = remove_gapcols(msa_int)
    C = len(next(iter(msa_int.values())))
    return msa_int, C


def align_muscle(int_seqs):
    """Run MUSCLE alignment. Returns (msa_int_dict, C)."""
    # Convert int seqs to string for MUSCLE input
    str_seqs = {}
    for name, iarr in int_seqs.items():
        chars = []
        for c in iarr:
            if 0 <= c < 20:
                chars.append(AA_CHARS[c])
            else:
                chars.append('X')
        str_seqs[name] = ''.join(chars)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        tmp_input = f.name
        write_fasta(str_seqs, tmp_input)
    tmp_output = tmp_input + '.aln'

    try:
        run_muscle(tmp_input, tmp_output)
        muscle_seqs = parse_fasta(tmp_output)

        msa_int = {}
        for name in int_seqs:
            if name not in muscle_seqs:
                raise ValueError(f'{name} not in MUSCLE output')
            msa_int[name] = encode_aligned_seq(muscle_seqs[name])

        msa_int = remove_gapcols(msa_int)
        C = len(next(iter(msa_int.values())))
        return msa_int, C
    finally:
        os.unlink(tmp_input)
        if os.path.exists(tmp_output):
            os.unlink(tmp_output)


# --- Core benchmark ---

def run_method(method_name, int_seqs, true_seq, remaining, target_name=None,
               ref_aln_strings=None):
    """Run a single method on a family.

    Args:
        method_name: e.g. 'fsa_d3f1_part_d3f1'
        int_seqs: dict {name: int_array} of REMAINING sequences (unaligned)
        true_seq: int array of true target sequence
        remaining: list of remaining sequence names
        target_name: name of held-out target
        ref_aln_strings: dict {name: aligned_string} of the ORIGINAL BAliBASE
            structural MSA (including target), used to build a reference tree
            that determines the target's sibling

    Returns dict with accuracy, pred_len, true_len, n_matches, n_aligned, time.
    """
    from experiments.reconstruct_util import reroot_at_node

    t_start = time.time()

    # Parse method name
    parts = method_name.split('_part_')
    if len(parts) != 2:
        raise ValueError(f'Invalid method name: {method_name}')
    align_method = parts[0]      # 'fsa_d3f1', 'fsa_d5f1', 'muscle'
    recon_model_key = parts[1]   # 'd3f1', 'd5f1'

    # Step 1: Align REMAINING sequences (target excluded)
    t_align_start = time.time()
    if align_method.startswith('fsa_'):
        fsa_model_key = align_method[4:]  # 'd3f1' or 'd5f1'
        msa_int, C = align_fsa(int_seqs, fsa_model_key)
    elif align_method == 'muscle':
        msa_int, C = align_muscle(int_seqs)
    else:
        raise ValueError(f'Unknown alignment method: {align_method}')
    t_align = time.time() - t_align_start

    if C == 0:
        return {
            'accuracy': 0.0, 'pred_len': 0,
            'true_len': int(len(true_seq)),
            'n_matches': 0, 'n_aligned': 0,
            'time': time.time() - t_start,
            'time_align': t_align, 'time_recon': 0.0,
        }

    # Step 2: Determine target's sibling from the ORIGINAL structural MSA tree.
    # Build FastTree from BAliBASE reference alignment (includes target) to
    # determine where the target attaches in the tree topology.
    sibling_name = None
    if ref_aln_strings is not None and target_name is not None:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            tmp_ref = f.name
            write_fasta(ref_aln_strings, tmp_ref)
        try:
            ref_tree_nwk = run_fasttree(tmp_ref)
        finally:
            os.unlink(tmp_ref)

        ref_tree = parse_newick(ref_tree_nwk)
        # Find target's sibling in the reference tree
        for node in ref_tree.preorder():
            if node.name == target_name and node.is_leaf:
                if node.parent is not None:
                    siblings = [c for c in node.parent.children if c.name != target_name]
                    if siblings:
                        # Take the first sibling — if it's a leaf, use its name;
                        # if internal, find any leaf under it
                        sib = siblings[0]
                        if sib.is_leaf:
                            sibling_name = sib.name
                        else:
                            for leaf in sib.preorder():
                                if leaf.is_leaf:
                                    sibling_name = leaf.name
                                    break
                break

    # Step 3: Build FastTree from the REALIGNED remaining-only MSA
    aln_strings = msa_ints_to_strings(msa_int)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        tmp_aln = f.name
        write_fasta(aln_strings, tmp_aln)
    try:
        tree_nwk = run_fasttree(tmp_aln)
    finally:
        os.unlink(tmp_aln)

    tree = parse_newick(tree_nwk)

    # Step 4: If we found the sibling, re-root at the sibling's parent in the
    # remaining-only tree. The sibling's parent IS the node where the target
    # would have attached (the node to reconstruct).
    if sibling_name is not None:
        sib_node = None
        for node in tree.preorder():
            if node.name == sibling_name and node.is_leaf:
                sib_node = node
                break
        if sib_node is not None and sib_node.parent is not None:
            tree = reroot_at_node(tree, sib_node.parent)

    # Step 3: Partition reconstruction using REMAINING-only MSA
    params, n_dom, n_frag, _ = load_mixdom_params(recon_model_key)
    model = mixdom_model_from_params(params)
    config = PartitionReconConfig(use_jax=True)

    # Replace non-standard residues for reconstruction
    clean_msa = {}
    for name, seq in msa_int.items():
        s = seq.copy()
        s[s >= 20] = -1
        clean_msa[name] = s

    # Run partition reconstruction — use a dummy held_out name
    # The adapter will try to prune it; since it doesn't exist,
    # the tree stays unchanged and all remaining leaves are used.
    # Actually, the adapter requires held_out to be in the tree.
    # Instead, build reconstruction inputs directly.
    t_recon_start = time.time()
    from tkfmixdom.jax.tree.partition_recon import build_inputs
    from tkfmixdom.jax.tree.partition_recon_jax import partition_recon_forward_backward_jax

    name_internal_nodes(tree)
    inputs = build_inputs(tree, clean_msa)
    result = partition_recon_forward_backward_jax(inputs, model)

    L = result.root_residue_map.shape[0]
    pred_seq = np.array(
        [int(result.root_residue_map[c])
         for c in range(L) if result.root_residue_map[c] >= 0],
        dtype=np.int32,
    )
    t_recon = time.time() - t_recon_start

    # Step 4: Score
    identity, n_aligned, n_matches = needleman_wunsch_identity(pred_seq, true_seq)

    return {
        'accuracy': float(identity),
        'pred_len': int(len(pred_seq)),
        'true_len': int(len(true_seq)),
        'n_matches': int(n_matches),
        'n_aligned': int(n_aligned),
        'pred_seq': [int(x) for x in pred_seq],
        'time': time.time() - t_start,
        'time_align': float(t_align),
        'time_recon': float(t_recon),
    }


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description='Realignment + partition reconstruction benchmark')
    parser.add_argument('--dataset', required=True,
                        choices=['unified_short', 'unified_long', 'treefam'],
                        help='Dataset to benchmark on')
    parser.add_argument('--methods', default=','.join(ALL_METHODS),
                        help='Comma-separated list of methods (default: all)')
    parser.add_argument('--out', default=None,
                        help='Output JSON file (default: auto-generated)')
    parser.add_argument('--max-families', type=int, default=0,
                        help='Limit number of families (0=all)')
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(',')]
    for m in methods:
        if m not in ALL_METHODS:
            print(f'ERROR: unknown method {m}. Choose from: {ALL_METHODS}')
            sys.exit(1)

    # Load spec
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    if args.dataset == 'unified_short':
        spec_path = os.path.join(exp_dir, 'unified_benchmark_spec.json')
    elif args.dataset == 'unified_long':
        spec_path = os.path.join(exp_dir, 'unified_benchmark_long_spec.json')
    elif args.dataset == 'treefam':
        spec_path = os.path.join(exp_dir, 'treefam_reconstruction_spec.json')

    log(f'Loading spec from {spec_path}')
    with open(spec_path) as f:
        spec = json.load(f)
    families = spec['families']

    if args.max_families > 0:
        families = families[:args.max_families]

    n_total = len(families)
    log(f'Dataset: {args.dataset}, {n_total} families, methods: {methods}')

    # Output path
    out_path = args.out
    if out_path is None:
        out_path = os.path.join(exp_dir,
                                f'realign_partition_{args.dataset}.json')

    # Resume support: load existing results
    results = []
    done_keys = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        for r in results:
            # A result is "done" for a given family+held_out if all
            # requested methods are present
            key = (r['family'], r['held_out'])
            have_methods = set(r.get('methods', {}).keys())
            if all(m in have_methods for m in methods):
                done_keys.add(key)
        log(f'Resumed: {len(done_keys)} families already complete')

    # Pre-load params for all needed model keys
    needed_model_keys = set()
    for m in methods:
        parts = m.split('_part_')
        align_part = parts[0]
        recon_part = parts[1]
        needed_model_keys.add(recon_part)
        if align_part.startswith('fsa_'):
            needed_model_keys.add(align_part[4:])
    for mk in sorted(needed_model_keys):
        load_mixdom_params(mk)

    # Main loop
    for idx, fam_spec in enumerate(families):
        family = fam_spec['family']
        held_out = fam_spec['held_out']
        remaining = fam_spec['remaining']
        true_seq = np.array(fam_spec['true_seq'], dtype=np.int32)

        key = (family, held_out)
        if key in done_keys:
            continue

        jax.clear_caches()  # OOM prevention; prefer geometric padding

        # Load unaligned sequences and reference alignment (for tree placement)
        ref_aln_strings = None  # Only used for BAliBASE
        try:
            if args.dataset in ('unified_short', 'unified_long'):
                int_seqs = load_pfam_unaligned(family, remaining)
                # For Pfam: use existing Pfam MSA as reference to determine sibling
                pfam_dir = os.path.expanduser(spec.get('pfam_dir', PFAM_DIR))
                sto_path = os.path.join(pfam_dir, f'{family}.sto')
                if os.path.exists(sto_path):
                    ref_aln_strings = parse_sto(sto_path)
            elif args.dataset == 'treefam':
                int_seqs = load_treefam_unaligned(family, remaining)
                # For TreeFam: use existing TreeFam MSA as reference
                treefam_dir = os.path.expanduser(spec.get('treefam_dir',
                    '~/bio-datasets/data/treefam/treefam_family_data'))
                tf_path = os.path.join(treefam_dir, f'{family}.aa.fasta')
                if os.path.exists(tf_path):
                    ref_aln_strings = parse_fasta(tf_path)
        except Exception as e:
            log(f'[{idx+1}/{n_total}] {family}: load error: {e}')
            traceback.print_exc()
            continue

        # Check for existing partial results for this family
        existing_result = None
        for r in results:
            if r['family'] == family and r['held_out'] == held_out:
                existing_result = r
                break

        if existing_result is None:
            existing_result = {
                'family': family,
                'held_out': held_out,
                'true_len': int(len(true_seq)),
                'K': len(remaining),
                'methods': {},
            }
            results.append(existing_result)

        # Run each method
        method_strs = []
        for method in methods:
            if method in existing_result.get('methods', {}):
                # Already done from a previous partial run
                r = existing_result['methods'][method]
                method_strs.append(f'{method} acc={r["accuracy"]*100:.1f}% (cached)')
                continue

            try:
                r = run_method(method, int_seqs, true_seq, remaining,
                               target_name=held_out,
                               ref_aln_strings=ref_aln_strings)
                existing_result.setdefault('methods', {})[method] = r
                method_strs.append(
                    f'{method} acc={r["accuracy"]*100:.1f}% t={r["time"]:.1f}s')
            except Exception as e:
                log(f'[{idx+1}/{n_total}] {family}: {method} error: {e}')
                traceback.print_exc()
                existing_result.setdefault('methods', {})[method] = {
                    'accuracy': -1.0, 'time': 0.0, 'error': str(e),
                }
                method_strs.append(f'{method} ERROR')

        log(f'[{idx+1}/{n_total}] {family}: ' + ' | '.join(method_strs))

        # Save periodically
        if (idx + 1) % SAVE_EVERY == 0:
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
            log(f'  Saved {len(results)} results to {out_path}')

    # Final save
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    log(f'\n=== Summary ({args.dataset}) ===')
    for method in methods:
        accs = [r['methods'][method]['accuracy']
                for r in results
                if method in r.get('methods', {})
                and r['methods'][method]['accuracy'] >= 0]
        if accs:
            log(f'  {method}: mean={np.mean(accs)*100:.1f}% '
                f'median={np.median(accs)*100:.1f}% n={len(accs)}')
        else:
            log(f'  {method}: no results')

    log(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
