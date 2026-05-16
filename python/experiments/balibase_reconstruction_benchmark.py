#!/usr/bin/env python3
"""BAliBASE held-out leaf reconstruction benchmark.

NOTE (2026-05-02): rescued after being deleted earlier. The expected
input alignments live at
  s3://tkf-mixdom-gpu-618647024028/datasets/balibase_recon_alignments/
(structural/ and muscle/ subdirs). They are NOT in the repo by default;
fetch with
  AWS_PROFILE=tkf-gpu aws s3 sync \\
    s3://tkf-mixdom-gpu-618647024028/datasets/balibase_recon_alignments/ \\
    python/experiments/balibase_recon_alignments/
before running. The rescue motivation: BAliBASE's higher
divergence may be useful as a stress test for ancestral reconstruction
methods that saturate on Pfam-val (e.g. varanc-presence vs Fitch).

For each of 120 BAliBASE families, one sequence is held out and predicted
from the remaining sequences using Felsenstein marginal reconstruction.

Setup A ("structural"):
  1. Use BAliBASE reference alignment of remaining sequences
  2. Build FastTree ML tree
  3. Felsenstein reconstruction -> predicted sequence

Setup B ("fsa_d3f1" / "fsa_d5f1"):
  1. Build FSA alignment of remaining sequences using MixDom model
  2. Build FastTree ML tree
  3. Felsenstein reconstruction -> predicted sequence

Score: Needleman-Wunsch identity of predicted vs true held-out sequence.

Environment variables:
  BENCH_METHODS: comma-separated from {structural_fels, fsa_d3f1_fels, fsa_d5f1_fels, muscle_fels}
                 Default: all three.

Usage:
  cd python && JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES="" uv run python -u \
      experiments/balibase_reconstruction_benchmark.py
"""

import os
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

import sys
import json
import time
import tempfile
import subprocess
import traceback
from pathlib import Path

import numpy as np
import jax

os.environ.setdefault('JAX_ENABLE_X64', '1')

# Persistent JAX compilation cache
_JAX_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'pfam', 'jax_cache')
os.makedirs(_JAX_CACHE, exist_ok=True)
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', _JAX_CACHE)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, str(Path("~/bio-datasets/data").expanduser()))

from tkfmixdom.util.msa_benchmark import parse_fasta

from tkfmixdom.jax.util.io import parse_newick, AA_TO_INT
from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.tree.ancestor import marginal_ancestor_all_columns_jax
from tkfmixdom.jax.tree.tree_varanc import infer_internal_presence, name_internal_nodes
from experiments.ancrec_benchmark import needleman_wunsch_identity

# --- Constants ---
PROJ_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = Path(__file__).parent.parent
SPEC_PATH = Path(__file__).parent / "balibase_reconstruction_spec.json"
RESULTS_PATH = Path(__file__).parent / "balibase_reconstruction_benchmark.json"

AA_CHARS = "ACDEFGHIKLMNPQRSTVWY"
AA_MAP = {c: i for i, c in enumerate(AA_CHARS)}

ALL_METHODS = ['structural_fels', 'fsa_d3f1_fels', 'fsa_d5f1_fels', 'muscle_fels']

SAVE_EVERY = 5

t0 = time.time()
def log(msg):
    print(f'[{time.time()-t0:.0f}s] {msg}', flush=True)


# --- Helpers ---

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
            idx = AA_MAP.get(c.upper(), -1)
            result.append(idx)
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


def run_fasttree(aln_path):
    """Run FastTree on aligned FASTA, return Newick string."""
    fasttree_bin = os.path.expanduser('~/bin/FastTree')
    if not os.path.exists(fasttree_bin):
        fasttree_bin = 'FastTree'  # fallback to PATH
    result = subprocess.run(
        [fasttree_bin, '-quiet', '-lg'],
        stdin=open(aln_path),
        capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f'FastTree failed: {result.stderr}')
    return result.stdout.strip()


def remove_gapcols(msa_dict):
    """Remove columns where all sequences are gaps. Returns new dict."""
    names = list(msa_dict.keys())
    if not names:
        return {}
    L = len(msa_dict[names[0]])
    # Find non-gap-only columns
    keep = []
    for col in range(L):
        if any(msa_dict[n][col] >= 0 for n in names):
            keep.append(col)
    return {n: np.array([msa_dict[n][c] for c in keep], dtype=np.int32) for n in names}


def find_target_sibling(ref_aln_strings, target_name):
    """Find the target's sibling in a FastTree built from the reference MSA.

    Returns the name of a leaf that is sibling (or under the sibling subtree)
    of the target in the reference tree. Returns None if not found.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        tmp = f.name
        write_fasta(ref_aln_strings, tmp)
    try:
        tree_nwk = run_fasttree(tmp)
    finally:
        os.unlink(tmp)

    tree = parse_newick(tree_nwk)
    for node in tree.preorder():
        if node.name == target_name and node.is_leaf and node.parent is not None:
            siblings = [c for c in node.parent.children if c.name != target_name]
            if siblings:
                sib = siblings[0]
                # If sibling is a leaf, return its name; if internal, find any leaf
                for leaf in sib.preorder():
                    if leaf.is_leaf:
                        return leaf.name
    return None


def run_felsenstein_on_msa(msa_int, tree, sibling_name=None):
    """Run Felsenstein on MSA + tree, return predicted sequence (no gaps).

    If sibling_name is given, re-root the tree at the sibling's parent
    (the node where the held-out target would have attached) before
    reconstructing. Otherwise reconstructs at the root.
    """
    from experiments.reconstruct_util import reroot_at_node

    Q_lg, pi_lg = rate_matrix_lg()
    Q_lg_np, pi_lg_np = np.asarray(Q_lg), np.asarray(pi_lg)

    # Re-root at sibling's parent if specified
    if sibling_name is not None:
        for node in tree.preorder():
            if node.name == sibling_name and node.is_leaf and node.parent is not None:
                tree = reroot_at_node(tree, node.parent)
                break

    name_internal_nodes(tree)

    # Replace non-standard residues (>=20) with -1 for Felsenstein
    pruned_msa = {}
    for name, seq in msa_int.items():
        s = seq.copy()
        s[s >= 20] = -1
        pruned_msa[name] = s

    C = len(next(iter(pruned_msa.values())))
    ancestor, posteriors = marginal_ancestor_all_columns_jax(
        tree, pruned_msa, Q_lg_np, pi_lg_np)

    # Fitch parsimony for gap pattern
    leaf_pres = {name: np.array(seq >= 0, dtype=bool) for name, seq in pruned_msa.items()}
    all_pres = infer_internal_presence(tree, leaf_pres)
    root_pres = all_pres.get(tree.name, np.ones(C, dtype=bool))

    # Extract predicted residues at present columns
    pred_seq = np.array([int(ancestor[c]) for c in range(len(ancestor))
                         if c < len(root_pres) and root_pres[c] and ancestor[c] >= 0],
                        dtype=np.int32)
    return pred_seq


def score_prediction(pred_seq, true_seq):
    """Score predicted vs true using NW alignment.

    Returns dict with accuracy, pred_len, true_len, n_matches, n_aligned.
    """
    identity, n_aligned, n_matches = needleman_wunsch_identity(pred_seq, true_seq)
    true_len = len(true_seq)
    pred_len = len(pred_seq)
    return {
        'accuracy': float(identity),
        'pred_len': int(pred_len),
        'true_len': int(true_len),
        'n_matches': int(n_matches),
        'n_aligned': int(n_aligned),
        'pred_seq': [int(x) for x in pred_seq],
    }


# --- MixDom params loading ---

_fsa_params_cache = {}

def load_fsa_params(model_key):
    """Load MixDom params for FSA alignment. model_key: 'd3f1' or 'd5f1'."""
    if model_key in _fsa_params_cache:
        return _fsa_params_cache[model_key]

    import jax.numpy as jnp
    from tkfmixdom.jax.distill.maraschino import load_params, build_rate_matrix

    if model_key == 'd3f1':
        npz_path = str(PYTHON_ROOT / 'pfam' / 'svi_bw_d3f1_full_best_val.npz')
    elif model_key == 'd5f1':
        npz_path = str(PYTHON_ROOT / 'pfam' / 'svi_bw_d5f1_full_best_val.npz')
    else:
        raise ValueError(f'Unknown model key: {model_key}')

    log(f'Loading {model_key} params from {npz_path}')
    params, n_dom, n_cls = load_params(npz_path)
    v = np.asarray(params['v'])
    pis = np.asarray(params['pi'])
    S_exch = np.asarray(params['S_exch'])
    avg_pi = np.einsum('n,na->a', v, pis)
    avg_pi = avg_pi / avg_pi.sum()
    avg_S = np.einsum('n,nab->ab', v, S_exch)
    fsa = {
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
    fw = fsa['frag_weights']
    n_frag = fw.shape[1] if fw.ndim > 1 else 1
    log(f'  {model_key}: n_dom={n_dom}, n_frag={n_frag}')
    _fsa_params_cache[model_key] = (fsa, n_dom, n_frag)
    return fsa, n_dom, n_frag


# --- Main methods ---

def method_structural_fels(family_spec, balibase_ref_dir):
    """Setup A: structural alignment + Felsenstein."""
    family = family_spec['family']
    held_out = family_spec['held_out']
    remaining = family_spec['remaining']

    # Parse reference alignment
    ref_path = os.path.join(balibase_ref_dir, family)
    ref_seqs = parse_fasta(ref_path)

    # Build aligned integer MSA of remaining sequences
    remaining_aln = {}
    for name in remaining:
        if name not in ref_seqs:
            raise ValueError(f'{name} not in reference alignment for {family}')
        remaining_aln[name] = encode_aligned_seq(ref_seqs[name])

    # Remove gap-only columns
    remaining_aln = remove_gapcols(remaining_aln)
    C = len(next(iter(remaining_aln.values())))

    if C == 0:
        return {'accuracy': 0.0, 'pred_len': 0, 'true_len': family_spec['true_len'],
                'n_matches': 0, 'n_aligned': 0}

    # Write aligned FASTA for FastTree
    aln_strings = {}
    for name, ints in remaining_aln.items():
        chars = []
        for c in ints:
            if 0 <= c < 20:
                chars.append(AA_CHARS[c])
            else:
                chars.append('-')
        aln_strings[name] = ''.join(chars)

    # Save alignment for external methods
    aln_dir = os.path.join(os.path.dirname(__file__), 'balibase_recon_alignments', 'structural')
    os.makedirs(aln_dir, exist_ok=True)
    write_fasta(aln_strings, os.path.join(aln_dir, f'{family}.fa'))

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        tmp_aln = f.name
        write_fasta(aln_strings, tmp_aln)

    try:
        tree_nwk = run_fasttree(tmp_aln)
        tree = parse_newick(tree_nwk)
        # Find target's sibling from structural MSA tree (includes target)
        sibling = find_target_sibling(ref_seqs, held_out)
        pred_seq = run_felsenstein_on_msa(remaining_aln, tree, sibling_name=sibling)
        true_seq = np.array(family_spec['true_seq'], dtype=np.int32)
        return score_prediction(pred_seq, true_seq)
    finally:
        os.unlink(tmp_aln)


def method_fsa_fels(family_spec, balibase_in_dir, model_key, balibase_ref_dir=None):
    """Setup B: FSA alignment + Felsenstein."""
    from tkfmixdom.jax.tree.fsa_anneal import fsa_align

    family = family_spec['family']
    held_out = family_spec['held_out']
    remaining = family_spec['remaining']

    # Check for cached alignment first
    aln_dir = os.path.join(os.path.dirname(__file__), 'balibase_recon_alignments', model_key)
    cached_path = os.path.join(aln_dir, f'{family}.fa')

    if os.path.exists(cached_path):
        # Load cached alignment
        cached_seqs = parse_fasta(cached_path)
        msa_dict = {}
        for name in remaining:
            if name in cached_seqs:
                msa_dict[name] = np.array(
                    [AA_MAP.get(c.upper(), -1) if c != '-' else -1 for c in cached_seqs[name]],
                    dtype=np.int32)
        aln_strings = cached_seqs
    else:
        # Run FSA alignment from scratch
        from tkfmixdom.jax.tree.fsa_anneal import fsa_align

        in_path = os.path.join(balibase_in_dir, family)
        all_seqs = parse_fasta(in_path)
        int_seqs_remaining = {}
        for name in remaining:
            if name not in all_seqs:
                raise ValueError(f'{name} not in input FASTA for {family}')
            int_seqs_remaining[name] = encode_seq(all_seqs[name])

        fsa_params, n_dom, n_frag = load_fsa_params(model_key)
        jax.clear_caches()  # OOM prevention (forces recompile; prefer geometric padding)

        try:
            msa_dict_raw, msa_len = fsa_align(
                int_seqs_remaining, model='mixdom', pair_selection='erdos_renyi',
                mixdom_params=fsa_params, n_dom=n_dom, n_frag=n_frag,
                n_anneal_iterations=3)
        except Exception as e:
            if 'RESOURCE_EXHAUSTED' in str(e) or 'OUT_OF_MEMORY' in str(e):
                jax.clear_caches()  # OOM prevention; prefer geometric padding
                cpu = jax.devices('cpu')[0]
                with jax.default_device(cpu):
                    msa_dict_raw, msa_len = fsa_align(
                        int_seqs_remaining, model='mixdom', pair_selection='erdos_renyi',
                        mixdom_params=fsa_params, n_dom=n_dom, n_frag=n_frag,
                        n_anneal_iterations=3)
            else:
                raise

        msa_dict = {}
        for name, row in msa_dict_raw.items():
            msa_dict[name] = np.array(row, dtype=np.int32)

        aln_strings = msa_ints_to_strings(msa_dict_raw)
        os.makedirs(aln_dir, exist_ok=True)
        write_fasta(aln_strings, cached_path)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        tmp_aln = f.name
        write_fasta(aln_strings, tmp_aln)

    try:
        tree_nwk = run_fasttree(tmp_aln)
        tree = parse_newick(tree_nwk)

        # Convert msa_dict to integer arrays for Felsenstein
        msa_int = {}
        for name, row in msa_dict.items():
            msa_int[name] = np.array(row, dtype=np.int32)

        # Find target's sibling from structural MSA tree
        sibling = None
        if balibase_ref_dir is not None:
            ref_path = os.path.join(balibase_ref_dir, family)
            if os.path.exists(ref_path):
                sibling = find_target_sibling(parse_fasta(ref_path), held_out)

        pred_seq = run_felsenstein_on_msa(msa_int, tree, sibling_name=sibling)
        true_seq = np.array(family_spec['true_seq'], dtype=np.int32)
        return score_prediction(pred_seq, true_seq)
    finally:
        os.unlink(tmp_aln)


def method_muscle_fels(family_spec, balibase_in_dir, balibase_ref_dir=None):
    """MUSCLE alignment + Felsenstein reconstruction."""
    family = family_spec['family']
    held_out = family_spec['held_out']
    remaining = family_spec['remaining']

    # Check for cached alignment first
    aln_dir = os.path.join(os.path.dirname(__file__), 'balibase_recon_alignments', 'muscle')
    cached_path = os.path.join(aln_dir, f'{family}.fa')

    if os.path.exists(cached_path):
        cached_seqs = parse_fasta(cached_path)
        msa_int = {}
        for name in remaining:
            if name in cached_seqs:
                msa_int[name] = encode_aligned_seq(cached_seqs[name])
        msa_int = remove_gapcols(msa_int)
        C = len(next(iter(msa_int.values())))
        if C == 0:
            return {'accuracy': 0.0, 'pred_len': 0, 'true_len': family_spec['true_len'],
                    'n_matches': 0, 'n_aligned': 0}
        aln_strings = msa_ints_to_strings(msa_int)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            tmp_aln = f.name
            write_fasta(aln_strings, tmp_aln)
        try:
            tree_nwk = run_fasttree(tmp_aln)
            tree = parse_newick(tree_nwk)
            sibling = None
            if balibase_ref_dir is not None:
                ref_path = os.path.join(balibase_ref_dir, family)
                if os.path.exists(ref_path):
                    sibling = find_target_sibling(parse_fasta(ref_path), held_out)
            pred_seq = run_felsenstein_on_msa(msa_int, tree, sibling_name=sibling)
            true_seq = np.array(family_spec['true_seq'], dtype=np.int32)
            return score_prediction(pred_seq, true_seq)
        finally:
            os.unlink(tmp_aln)

    # No cached alignment — run MUSCLE from scratch
    in_path = os.path.join(balibase_in_dir, family)
    all_seqs = parse_fasta(in_path)

    str_seqs_remaining = {}
    for name in remaining:
        if name not in all_seqs:
            raise ValueError(f'{name} not in input FASTA for {family}')
        # Strip gaps from input
        str_seqs_remaining[name] = all_seqs[name].replace('-', '').replace('.', '')

    # Write unaligned FASTA for MUSCLE
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        tmp_input = f.name
        write_fasta(str_seqs_remaining, tmp_input)

    tmp_output = tmp_input + '.aln'

    try:
        # Find muscle binary
        import shutil
        muscle_bin = shutil.which('muscle') or os.path.expanduser('~/bin/muscle')
        if not os.path.exists(muscle_bin):
            raise FileNotFoundError('muscle binary not found in PATH or ~/bin/')
        # Try MUSCLE v5 syntax first
        result = subprocess.run(
            [muscle_bin, '-align', tmp_input, '-output', tmp_output],
            capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            # Fall back to MUSCLE v3 syntax
            result = subprocess.run(
                [muscle_bin, '-in', tmp_input, '-out', tmp_output],
                capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f'MUSCLE failed: {result.stderr}')

        # Parse MUSCLE output alignment
        muscle_seqs = parse_fasta(tmp_output)

        # Build integer MSA
        msa_int = {}
        for name in remaining:
            if name not in muscle_seqs:
                raise ValueError(f'{name} not in MUSCLE output for {family}')
            msa_int[name] = encode_aligned_seq(muscle_seqs[name])

        # Remove gap-only columns
        msa_int = remove_gapcols(msa_int)
        C = len(next(iter(msa_int.values())))

        if C == 0:
            return {'accuracy': 0.0, 'pred_len': 0, 'true_len': family_spec['true_len'],
                    'n_matches': 0, 'n_aligned': 0}

        # Convert to string format for FastTree
        aln_strings = msa_ints_to_strings(msa_int)

        # Save alignment
        aln_dir = os.path.join(os.path.dirname(__file__), 'balibase_recon_alignments', 'muscle')
        os.makedirs(aln_dir, exist_ok=True)
        write_fasta(aln_strings, os.path.join(aln_dir, f'{family}.fa'))

        # Write aligned FASTA for FastTree
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            tmp_aln = f.name
            write_fasta(aln_strings, tmp_aln)

        try:
            tree_nwk = run_fasttree(tmp_aln)
            tree = parse_newick(tree_nwk)
            # Find target's sibling from structural MSA tree
            sibling = None
            if balibase_ref_dir is not None:
                ref_path = os.path.join(balibase_ref_dir, family)
                if os.path.exists(ref_path):
                    sibling = find_target_sibling(parse_fasta(ref_path), held_out)
            pred_seq = run_felsenstein_on_msa(msa_int, tree, sibling_name=sibling)
            true_seq = np.array(family_spec['true_seq'], dtype=np.int32)
            return score_prediction(pred_seq, true_seq)
        finally:
            os.unlink(tmp_aln)
    finally:
        os.unlink(tmp_input)
        if os.path.exists(tmp_output):
            os.unlink(tmp_output)


# --- Main ---

def main():
    # Parse methods
    methods_str = os.environ.get('BENCH_METHODS', ','.join(ALL_METHODS))
    methods = [m.strip() for m in methods_str.split(',') if m.strip()]
    for m in methods:
        if m not in ALL_METHODS:
            print(f'Unknown method: {m}. Available: {ALL_METHODS}')
            sys.exit(1)
    log(f'Methods: {methods}')

    # Load spec
    with open(SPEC_PATH) as f:
        spec = json.load(f)
    families = spec['families']
    balibase_in_dir = str(PROJ_ROOT / spec['balibase_in_dir'])
    balibase_ref_dir = str(PROJ_ROOT / spec['balibase_ref_dir'])
    log(f'Loaded {len(families)} families from spec')

    # Resume: load existing results
    results = []
    done = {}  # family -> set of methods already done
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                existing = json.load(f)
            if isinstance(existing, dict) and 'results' in existing:
                results = list(existing['results'])
                for r in results:
                    fam = r.get('family', '')
                    if fam not in done:
                        done[fam] = set()
                    done[fam].update(k for k in r if k.endswith('_acc'))
                log(f'Resume: {len(results)} prior results')
        except Exception as e:
            log(f'Resume failed: {e}')

    def methods_done(family_name):
        """Check if all requested methods are already done for this family."""
        if family_name not in done:
            return False
        acc_keys = {m + '_acc' for m in methods}
        return acc_keys.issubset(done[family_name])

    n_done = 0
    n_total = len(families)

    for i, fspec in enumerate(families):
        family = fspec['family']
        held_out = fspec['held_out']

        # Clear JIT cache to prevent GPU memory accumulation
        jax.clear_caches()  # OOM prevention (forces recompile; prefer geometric padding)

        if methods_done(family):
            continue

        # Find existing result to update, or create new one
        existing_result = None
        for r in results:
            if r.get('family') == family and r.get('held_out') == held_out:
                existing_result = r
                break
        if existing_result is None:
            existing_result = {
                'family': family,
                'held_out': held_out,
                'true_len': fspec['true_len'],
                'K': fspec['K'],
                'mean_dist': fspec['mean_dist'],
            }
            results.append(existing_result)

        parts = []
        for method in methods:
            acc_key = method + '_acc'
            if acc_key in existing_result:
                parts.append(f'{method} acc={existing_result[acc_key]:.1f}% (cached)')
                continue

            t_start = time.time()
            try:
                if method == 'structural_fels':
                    res = method_structural_fels(fspec, balibase_ref_dir)
                elif method == 'fsa_d3f1_fels':
                    res = method_fsa_fels(fspec, balibase_in_dir, 'd3f1',
                                          balibase_ref_dir=balibase_ref_dir)
                elif method == 'fsa_d5f1_fels':
                    res = method_fsa_fels(fspec, balibase_in_dir, 'd5f1',
                                          balibase_ref_dir=balibase_ref_dir)
                elif method == 'muscle_fels':
                    res = method_muscle_fels(fspec, balibase_in_dir,
                                              balibase_ref_dir=balibase_ref_dir)
                else:
                    continue

                elapsed = time.time() - t_start
                existing_result[acc_key] = round(100.0 * res['accuracy'], 2)
                existing_result[method + '_pred_len'] = res['pred_len']
                existing_result[method + '_n_matches'] = res['n_matches']
                existing_result[method + '_n_aligned'] = res['n_aligned']
                existing_result[method + '_time'] = round(elapsed, 2)
                parts.append(f'{method} acc={existing_result[acc_key]:.1f}% t={elapsed:.1f}s')
            except Exception as e:
                elapsed = time.time() - t_start
                existing_result[acc_key] = None
                existing_result[method + '_error'] = str(e)
                parts.append(f'{method} ERROR: {e}')
                traceback.print_exc()

        status = ' | '.join(parts)
        log(f'[{i+1}/{n_total}] {family} (held_out={held_out}): {status}')
        n_done += 1

        # Save periodically
        if n_done % SAVE_EVERY == 0:
            _save_results(results, methods)

    _save_results(results, methods)
    _print_summary(results, methods)


def _save_results(results, methods):
    """Save results to JSON."""
    # Compute summary stats
    summary = {}
    for method in methods:
        acc_key = method + '_acc'
        accs = [r[acc_key] for r in results if r.get(acc_key) is not None]
        if accs:
            summary[method] = {
                'n': len(accs),
                'mean_acc': round(np.mean(accs), 2),
                'median_acc': round(float(np.median(accs)), 2),
                'std_acc': round(float(np.std(accs)), 2),
            }

    output = {
        'methods': methods,
        'n_families': len(results),
        'summary': summary,
        'results': results,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    log(f'Saved {len(results)} results to {RESULTS_PATH}')


def _print_summary(results, methods):
    """Print summary statistics."""
    log('=== Summary ===')
    for method in methods:
        acc_key = method + '_acc'
        accs = [r[acc_key] for r in results if r.get(acc_key) is not None]
        if accs:
            log(f'  {method}: n={len(accs)} mean={np.mean(accs):.1f}% '
                f'median={np.median(accs):.1f}% std={np.std(accs):.1f}%')
        else:
            log(f'  {method}: no results')


if __name__ == '__main__':
    main()
