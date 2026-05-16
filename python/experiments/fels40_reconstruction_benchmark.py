#!/usr/bin/env python3
"""Fels40 (40-state hidden gap model) held-out leaf reconstruction benchmark.

Unlike fels21 (21-state, gap as observed character 20), fels40 has 40 hidden
states: 20 "present" states (emit their amino acid) and 20 "gapped" states
(always emit gap). Felsenstein pruning uses EMISSION-weighted leaf likelihoods:
  - Observed AA a (0-19): leaf likelihood = one-hot at state a, zeros elsewhere
  - Observed gap (-1 or 20): leaf likelihood = [0]*20 + [1]*20

The 40-state model is loaded from pfam/fels40_em.npz (Q40, pi40).

Runs on four datasets:
  - unified_short: Pfam val families (spec: unified_benchmark_spec.json)
  - unified_long:  Pfam val families, longer (spec: unified_benchmark_long_spec.json)
  - treefam:       TreeFam families (spec: treefam_reconstruction_spec.json)
  - balibase:      BAliBASE families (spec: balibase_reconstruction_spec.json)

Usage:
    cd python && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run python -u \\
        experiments/fels40_reconstruction_benchmark.py --dataset unified_short
"""

import os
import sys
import json
import copy
import time
import argparse
import traceback

os.environ.setdefault('JAX_ENABLE_X64', '1')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.linalg import expm

from tkfmixdom.jax.tree.tree_varanc import name_internal_nodes
from tkfmixdom.jax.util.io import AA_TO_INT, parse_newick
from experiments.ancrec_benchmark import parse_sto, needleman_wunsch_identity, PFAM_DIR
from experiments.unified_reconstruction_benchmark import (
    prune_leaf_keep_parent, _nw_metrics,
)

# --- Constants ---
TREE_DIR = os.path.expanduser("~/bio-datasets/data/pfam-seed/trees")
SAVE_EVERY = 5

t0 = time.time()
def log(msg): print(f'[{time.time()-t0:.0f}s] {msg}', flush=True)


# --- Data loading helpers (same as fels21) ---

def parse_fasta(filepath):
    """Parse FASTA file, return dict {name: sequence_string}."""
    seqs = {}
    name = None
    seq_parts = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    seqs[name] = ''.join(seq_parts)
                name = line[1:].split()[0]
                seq_parts = []
            elif line.startswith('//'):
                continue
            elif name is not None:
                seq_parts.append(line)
    if name is not None:
        seqs[name] = ''.join(seq_parts)
    return seqs


def parse_treefam_tree(emf_path):
    """Parse newick tree from TreeFam EMF file."""
    with open(emf_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('('):
                return parse_newick(line)
    raise ValueError(f"No newick tree found in {emf_path}")


def encode_aligned_seq(seq_str):
    """Convert aligned sequence string to integer array. Gaps -> -1."""
    result = []
    for c in seq_str:
        if c in '.-~ ':
            result.append(-1)
        else:
            idx = AA_TO_INT.get(c.upper(), -1)
            result.append(idx)
    return np.array(result, dtype=np.int32)


def build_msa_int(seqs, n_cols=None):
    """Convert aligned string sequences to integer-encoded MSA."""
    msa = {}
    for name, seq_str in seqs.items():
        C = len(seq_str) if n_cols is None else n_cols
        arr = np.full(C, -1, dtype=np.int32)
        for j, ch in enumerate(seq_str[:C]):
            ch_upper = ch.upper()
            if ch_upper in AA_TO_INT:
                idx = AA_TO_INT[ch_upper]
                if idx >= 20:
                    arr[j] = -1
                else:
                    arr[j] = idx
        msa[name] = arr
    return msa


def prune_tree_to_msa(tree, msa_names):
    """Prune tree to only keep leaves present in msa_names."""
    new_tree = copy.deepcopy(tree)
    changed = True
    while changed:
        changed = False
        for node in list(new_tree.preorder()):
            if node.is_leaf and node.name not in msa_names:
                if node.parent is not None:
                    parent = node.parent
                    parent.children = [c for c in parent.children if c is not node]
                    if len(parent.children) == 1 and parent.parent is not None:
                        child = parent.children[0]
                        child.branch_length += parent.branch_length
                        gp = parent.parent
                        idx = gp.children.index(parent)
                        gp.children[idx] = child
                        child.parent = gp
                    elif len(parent.children) == 1 and parent.parent is None:
                        child = parent.children[0]
                        child.parent = None
                        child.branch_length = 0.0
                        new_tree = child
                    changed = True
                    break
    return new_tree


def run_fasttree(aln_path):
    """Run FastTree on aligned FASTA, return Newick string."""
    import subprocess
    fasttree_bin = os.path.expanduser('~/bin/FastTree')
    if not os.path.exists(fasttree_bin):
        fasttree_bin = 'FastTree'
    result = subprocess.run(
        [fasttree_bin, '-quiet', '-lg'],
        stdin=open(aln_path),
        capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f'FastTree failed: {result.stderr}')
    return result.stdout.strip()


# --- Fels40 reconstruction ---
#
# logp_target math (per-method confidence for ancestral reconstruction):
#   For each held-out leaf prediction, we record the JOINT log posterior
#   of the predicted residue sequence under the 40-state model:
#       logp_target = sum_{c in kept cols} log P(pred_aa[c] | MSA \ target,
#                                                  tree, model)
#   where pred_aa[c] is the argmax-over-the-20-present-states (a in 0..19)
#   at column c at the prediction node, and "kept cols" excludes columns
#   where P(any gapped state) > max_a P(present state a) (predicted gap).
#   The posterior in the log is the *unmarginalised* 40-state posterior at
#   the predicted state — this is the joint log probability of the predicted
#   residue, and so directly compares to fels21's logp_target.
#
#   This is the user-requested "log posterior of the target leaf's predicted
#   sequence" — *not* the old `best_p` whole-MSA log probability.
#
# logp_true math (added 2026-05-04, ground-truth scoring under the same
# per-column posterior):
#       logp_true = sum_{c: held_out_seq[c] is non-gap}
#                       log P(true_aa[c] | MSA \ target, tree, model)
#   where the per-column posterior is the 40-state Felsenstein posterior
#   at the prediction node, and we score at the present-state index for
#   the truth residue (posterior[true_aa]). Skips columns where the
#   held-out leaf is gap (parallel to logp_target's predicted-gap skip).
#   headroom = logp_true - logp_target, interpreted in the same way as
#   for fels21.

def run_fels40(tree, held_out, remaining, msa, Q40, pi40,
               held_out_seq=None):
    """Run 40-state Felsenstein on tree re-rooted at target's parent.

    Leaf likelihoods use emission weighting:
      - Observed AA a (0-19): cond[a]=1, rest 0
      - Observed gap: cond[0:20]=0, cond[20:40]=1

    Returns (pred_seq, elapsed_time, logp_target, logp_true). See math
    notes above the launcher main() for both definitions.
    """
    tf = time.time()

    # Re-root at target's parent
    pruned_tree, _ = prune_leaf_keep_parent(tree, held_out)
    if pruned_tree is None:
        return np.array([], dtype=np.int32), 0.0, 0.0, 0.0
    name_internal_nodes(pruned_tree)

    # Build MSA for remaining leaves
    msa40 = {}
    for name in remaining:
        if name not in msa:
            continue
        seq = msa[name].copy()
        # Normalize: gaps (-1) and X residues (>=20) -> gap code 20
        seq[seq < 0] = 20
        seq[seq >= 20] = 20
        msa40[name] = seq

    if not msa40:
        return np.array([], dtype=np.int32), 0.0, 0.0, 0.0

    pred_seq, logp_target, logp_true = _fels40_reconstruct(
        pruned_tree, msa40, Q40, pi40, held_out_seq=held_out_seq)
    elapsed = time.time() - tf
    return pred_seq, elapsed, logp_target, logp_true


def _fels40_reconstruct(tree_root, msa, Q40, pi40, held_out_seq=None):
    """Column-by-column 40-state Felsenstein pruning + MAP prediction.

    For each column:
      1. Felsenstein inside pass with emission-weighted leaf likelihoods
      2. Root posterior = pi40 * CL_root, normalized
      3. P(AA a) = posterior[a] for a in 0..19 (present states)
         P(gap) = sum(posterior[20..39]) (gapped states)
      4. If P(gap) > max_a P(AA a): predict gap (exclude column)
         Else: predict argmax_a P(AA a)

    Returns:
        pred_seq: int array of predicted AAs (gap columns excluded).
        logp_target: float, joint log posterior of predicted residue
            sequence under the 40-state Felsenstein posterior. For each
            kept column c with prediction pred_aa[c] in 0..19, we add
            log(posterior[c, pred_aa[c]]). Excluded (gap-predicted)
            columns do not contribute.
        logp_true: float, joint log posterior of the GROUND-TRUTH
            held-out leaf residue sequence under the same per-column
            posterior. For each column c where held_out_seq[c] is in
            0..19, we add log(posterior[c, held_out_seq[c]]). Gap
            columns of the held-out leaf are skipped. If
            held_out_seq is None, returns 0.0.
    """
    Q40_np = np.asarray(Q40)
    pi40_np = np.asarray(pi40)

    leaf_names = [n for n in msa.keys()]
    msa_len = len(next(iter(msa.values())))
    N = 40

    # Precompute transition matrices P(t) = expm(Q40 * t)
    sub_matrices = {}
    for node in tree_root.preorder():
        for child in node.children:
            t = max(child.branch_length, 1e-6)
            sub_matrices[id(child)] = expm(Q40_np * t)

    # Build leaf conditional likelihood vectors
    # For AA a: one-hot at state a
    # For gap (20): [0]*20 + [1]*20
    leaf_cond_cache = {}
    for name in leaf_names:
        seq = msa[name]
        conds = np.zeros((msa_len, N))
        for j in range(msa_len):
            c = seq[j]
            if 0 <= c < 20:
                conds[j, c] = 1.0
            else:
                # gap: all gapped states are compatible
                conds[j, 20:] = 1.0
        leaf_cond_cache[name] = conds

    pred_chars = []
    logp_target = 0.0
    logp_true = 0.0

    for col in range(msa_len):
        def _prune(node):
            """Returns (cond_likelihood_40, log_scale)."""
            if node.is_leaf:
                # Leaves not present in msa (e.g. held-out leaf that
                # prune_leaf_keep_parent failed to remove due to
                # duplicate internal-node names) are treated as missing
                # data: uniform conditional likelihood.
                cache = leaf_cond_cache.get(node.name)
                if cache is None:
                    return np.ones(N), 0.0
                cond = cache[col].copy()
                return cond, 0.0

            partial = np.ones(N)
            log_scale = 0.0
            for child in node.children:
                child_cond, child_ls = _prune(child)
                P = sub_matrices[id(child)]
                partial *= (P @ child_cond)
                log_scale += child_ls

            max_val = max(np.max(partial), 1e-300)
            partial /= max_val
            log_scale += np.log(max_val)
            return partial, log_scale

        root_cond, _ = _prune(tree_root)
        joint = pi40_np * root_cond
        posterior = joint / max(np.sum(joint), 1e-300)

        # Marginalize to observed space
        p_aa = posterior[:20]       # P(present state a) for each AA
        p_gap = np.sum(posterior[20:])  # P(any gapped state)

        max_aa = np.max(p_aa)
        if p_gap > max_aa:
            # Predict gap -> exclude column
            pass
        else:
            pred_aa = int(np.argmax(p_aa))
            pred_chars.append(pred_aa)
            # Joint log posterior of the predicted residue, under the
            # full 40-state posterior at the prediction node.
            logp_target += float(np.log(max(posterior[pred_aa], 1e-300)))

        # logp_true: score posterior at the held-out leaf's true residue
        # (skip gap cols).
        if held_out_seq is not None and col < len(held_out_seq):
            aa_true = int(held_out_seq[col])
            if 0 <= aa_true < 20:
                logp_true += float(np.log(max(
                    posterior[aa_true], 1e-300)))

    return np.array(pred_chars, dtype=np.int32), logp_target, logp_true


def _fels40_reconstruct_root(tree_root, msa, Q40, pi40, held_out_seq=None):
    """Same as _fels40_reconstruct but for root reconstruction (no pruning).

    Used for BAliBASE where the tree already has only remaining leaves.
    Returns (pred_seq, logp_target, logp_true). See _fels40_reconstruct
    for both definitions.
    """
    Q40_np = np.asarray(Q40)
    pi40_np = np.asarray(pi40)

    leaf_names = list(msa.keys())
    msa_len = len(next(iter(msa.values())))
    N = 40

    # Precompute transition matrices
    sub_matrices = {}
    for node in tree_root.preorder():
        for child in node.children:
            t = max(child.branch_length, 1e-6)
            sub_matrices[id(child)] = expm(Q40_np * t)

    # Build leaf conditional likelihood vectors
    leaf_cond_cache = {}
    for name in leaf_names:
        seq = msa[name]
        conds = np.zeros((msa_len, N))
        for j in range(msa_len):
            c = seq[j]
            if 0 <= c < 20:
                conds[j, c] = 1.0
            else:
                conds[j, 20:] = 1.0
        leaf_cond_cache[name] = conds

    pred_chars = []
    logp_target = 0.0
    logp_true = 0.0

    for col in range(msa_len):
        def _prune(node):
            if node.is_leaf:
                cond = leaf_cond_cache[node.name][col].copy()
                return cond, 0.0

            partial = np.ones(N)
            log_scale = 0.0
            for child in node.children:
                child_cond, child_ls = _prune(child)
                P = sub_matrices[id(child)]
                partial *= (P @ child_cond)
                log_scale += child_ls

            max_val = max(np.max(partial), 1e-300)
            partial /= max_val
            log_scale += np.log(max_val)
            return partial, log_scale

        root_cond, _ = _prune(tree_root)
        joint = pi40_np * root_cond
        posterior = joint / max(np.sum(joint), 1e-300)

        p_aa = posterior[:20]
        p_gap = np.sum(posterior[20:])

        max_aa = np.max(p_aa)
        if p_gap > max_aa:
            pass
        else:
            pred_aa = int(np.argmax(p_aa))
            pred_chars.append(pred_aa)
            logp_target += float(np.log(max(posterior[pred_aa], 1e-300)))

        if held_out_seq is not None and col < len(held_out_seq):
            aa_true = int(held_out_seq[col])
            if 0 <= aa_true < 20:
                logp_true += float(np.log(max(
                    posterior[aa_true], 1e-300)))

    return np.array(pred_chars, dtype=np.int32), logp_target, logp_true


# --- Dataset loaders ---

def load_pfam_family(fspec, pfam_dir, tree_dir):
    """Load MSA and tree for a Pfam family.

    Prunes both MSA and tree to spec's `held_out ∪ remaining`. Required
    for xhard subsampling: the spec may name far fewer leaves than the
    full Stockholm/Newick contain.
    """
    fam = fspec['family']
    held_out = fspec['held_out']
    remaining = fspec['remaining']
    C = fspec['n_cols']
    spec_leaves = set(remaining) | {held_out}

    tree_path = os.path.join(tree_dir, f'{fam}.nwk')
    if not os.path.exists(tree_path):
        tree_path = os.path.join(tree_dir, f'{fam}.tree')
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f'Tree not found for {fam}')

    sto_path = os.path.join(pfam_dir, f'{fam}.sto')
    if not os.path.exists(sto_path):
        raise FileNotFoundError(f'MSA not found for {fam}')

    seqs = parse_sto(sto_path)
    missing = [n for n in spec_leaves if n not in seqs]
    if missing:
        raise ValueError(f'{fam}: spec names not in Stockholm: {missing[:3]}'
                         f' ({len(missing)} missing)')
    msa_full = {}
    for name in spec_leaves:
        seq = np.full(C, -1, dtype=np.int32)
        for j, ch in enumerate(seqs[name]):
            if ch in AA_TO_INT:
                idx = AA_TO_INT[ch]
                if idx < 20:
                    seq[j] = idx
        msa_full[name] = seq

    # Strip empty columns (see fels21 loader for rationale).
    has_residue = np.zeros(C, dtype=bool)
    for seq in msa_full.values():
        has_residue |= (seq >= 0)
    kept_cols = np.where(has_residue)[0]
    msa = {n: seq[kept_cols] for n, seq in msa_full.items()}
    C = len(kept_cols)

    with open(tree_path) as f:
        tree_text = f.read().strip()
    tree = parse_newick(tree_text)
    tree_leaf_names = {l.name for l in tree.leaves()}
    spec_in_tree = spec_leaves & tree_leaf_names
    if held_out not in spec_in_tree:
        raise ValueError(f'{fam}: held_out {held_out} not in tree')
    if len(spec_in_tree) < 2:
        raise ValueError(f'{fam}: only {len(spec_in_tree)} of {len(spec_leaves)} '
                         f'spec leaves are in the tree')
    tree = prune_tree_to_msa(tree, spec_in_tree)
    name_internal_nodes(tree)

    return msa, tree, C


def load_treefam_family(fspec, treefam_dir):
    """Load MSA and tree for a TreeFam family."""
    fam = fspec['family']
    held_out = fspec['held_out']
    remaining = fspec['remaining']

    fasta_path = os.path.join(treefam_dir, f'{fam}.aa.fasta')
    tree_path = os.path.join(treefam_dir, f'{fam}.nh.emf')

    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f'FASTA not found for {fam}')
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f'Tree not found for {fam}')

    raw_seqs = parse_fasta(fasta_path)
    if held_out not in raw_seqs:
        raise ValueError(f'held_out {held_out} not in FASTA')

    C = len(next(iter(raw_seqs.values())))
    msa = build_msa_int(raw_seqs, n_cols=C)

    tree = parse_treefam_tree(tree_path)
    name_internal_nodes(tree)

    # Prune the tree to ONLY the spec's leaves (held_out + remaining).
    # The TreeFam Newick may contain additional leaves that are also in
    # the FASTA but not in the spec's remaining list (e.g. paralogs that
    # were excluded when sampling the held-out target). Keeping those
    # extra leaves in the tree would cause `_fels40_reconstruct` to
    # KeyError, since `msa40` is built only from `remaining`.
    tree_leaf_names = {l.name for l in tree.leaves()}
    spec_leaves = set(remaining) | {held_out}
    if held_out not in tree_leaf_names:
        raise ValueError(f'held_out {held_out} not in tree')

    tree = prune_tree_to_msa(tree, spec_leaves)
    name_internal_nodes(tree)

    return msa, tree, C


def load_balibase_family(fspec, balibase_ref_dir):
    """Load MSA and tree for a BAliBASE family."""
    import tempfile

    family = fspec['family']
    held_out = fspec['held_out']
    remaining = fspec['remaining']

    ref_path = os.path.join(balibase_ref_dir, family)
    ref_seqs = parse_fasta(ref_path)

    remaining_aln = {}
    for name in remaining:
        if name not in ref_seqs:
            raise ValueError(f'{name} not in reference alignment for {family}')
        remaining_aln[name] = encode_aligned_seq(ref_seqs[name])

    # Remove gap-only columns
    names = list(remaining_aln.keys())
    if not names:
        raise ValueError(f'No remaining sequences for {family}')
    L = len(remaining_aln[names[0]])
    keep = [col for col in range(L)
            if any(remaining_aln[n][col] >= 0 for n in names)]
    msa = {n: np.array([remaining_aln[n][c] for c in keep], dtype=np.int32)
           for n in names}
    C = len(keep)

    if C == 0:
        raise ValueError(f'All-gap alignment for {family}')

    AA_CHARS = "ACDEFGHIKLMNPQRSTVWY"
    aln_strings = {}
    for name, ints in msa.items():
        chars = []
        for c in ints:
            if 0 <= c < 20:
                chars.append(AA_CHARS[c])
            else:
                chars.append('-')
        aln_strings[name] = ''.join(chars)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        tmp_aln = f.name
        for name, seq in aln_strings.items():
            f.write(f'>{name}\n{seq}\n')

    try:
        tree_nwk = run_fasttree(tmp_aln)
        tree = parse_newick(tree_nwk)
        name_internal_nodes(tree)
    finally:
        os.unlink(tmp_aln)

    return msa, tree, C


# --- Main ---

DATASET_CONFIGS = {
    'unified_short_test': {
        'spec_file': 'unified_benchmark_test_spec.json',
        'type': 'pfam',
    },
    'unified_long_test': {
        'spec_file': 'unified_benchmark_long_test_spec.json',
        'type': 'pfam',
    },
    'unified_hard_test': {
        'spec_file': 'unified_benchmark_hard_test_spec.json',
        'type': 'pfam',
    },
    'unified_xhard_test': {
        'spec_file': 'unified_benchmark_xhard_test_spec.json',
        'type': 'pfam',
    },
    'unified_short': {
        'spec_file': 'contaminated_val_split_triage_for_deletion/unified_benchmark_spec.json',
        'type': 'pfam',
    },
    'unified_long': {
        'spec_file': 'contaminated_val_split_triage_for_deletion/unified_benchmark_long_spec.json',
        'type': 'pfam',
    },
    'treefam': {
        'spec_file': 'treefam_reconstruction_spec.json',
        'type': 'treefam',
    },
    'balibase': {
        'spec_file': 'balibase_reconstruction_spec.json',
        'type': 'balibase',
    },
}


def main():
    parser = argparse.ArgumentParser(
        description='Fels40 (40-state hidden gap) reconstruction benchmark')
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to run on')
    parser.add_argument('--recompute-missing-fields', action='store_true',
                        help='Re-run families whose existing fels40 entry '
                        'lacks logp_target / logp_true (or other newly-'
                        'added fields). Use this to backfill these '
                        'confidence fields on previously completed JSONs '
                        'without re-running every entry.')
    args = parser.parse_args()

    dataset = args.dataset
    config = DATASET_CONFIGS[dataset]

    log(f'Dataset: {dataset}')

    # Load fels40 model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'pfam', 'fels40_em.npz')
    data = np.load(model_path)
    Q40 = data['Q40']
    pi40 = data['pi40']
    log(f'Loaded fels40 model: Q40 {Q40.shape}, pi40 {pi40.shape}')

    # Load spec
    spec_path = os.path.join(os.path.dirname(__file__), config['spec_file'])
    with open(spec_path) as f:
        spec = json.load(f)
    families = spec['families']
    log(f'Loaded spec: {len(families)} families')

    # Dataset-specific directories
    if config['type'] == 'pfam':
        pfam_dir = os.path.expanduser(spec.get('pfam_dir', PFAM_DIR))
        tree_dir = os.path.expanduser(spec.get('tree_dir', TREE_DIR))
    elif config['type'] == 'treefam':
        treefam_dir = os.path.expanduser(spec.get('treefam_dir',
            '~/bio-datasets/data/treefam/treefam_family_data'))
    elif config['type'] == 'balibase':
        proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        balibase_ref_dir = os.path.join(proj_root, spec.get('balibase_ref_dir',
            os.path.expanduser('~/bio-datasets/data/balibase/bali3pdbm/ref')))

    # Resume support
    results_path = os.path.join(os.path.dirname(__file__),
                                f'fels40_reconstruction_{dataset}.json')
    results = []
    done_fams = set()
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                rd = json.load(f)
            if isinstance(rd, dict) and 'results' in rd:
                results = list(rd['results'])
                # Newly-added required fields. If --recompute-missing-fields
                # is set, families whose fels40 entry is missing any of
                # these are re-run (rather than skipped) to backfill.
                required_fields = {'logp_target', 'logp_true'}
                done_fams = set()
                for r in results:
                    if 'family' not in r or 'fels40' not in r:
                        continue
                    fres = r['fels40']
                    if args.recompute_missing_fields and not (
                            required_fields.issubset(fres.keys())):
                        continue  # treat as not done; will be recomputed
                    done_fams.add(r['family'])
                log(f'Resume: {len(results)} prior results, '
                    f'{len(done_fams)} families with fels40'
                    + (' (excluding ones missing newly-required fields)'
                       if args.recompute_missing_fields else ''))
        except Exception as e:
            log(f'Resume: failed to load: {e}')

    # Map existing results by family for updating
    results_by_fam = {}
    for ri, r in enumerate(results):
        if 'family' in r:
            results_by_fam[r['family']] = ri

    n_done = 0
    n_errors = 0
    nw_accs = []

    for fi, fspec in enumerate(families):
        fam = fspec['family']
        held_out = fspec['held_out']
        remaining = fspec['remaining']
        true_seq = np.array(fspec['true_seq'], dtype=np.int32)

        if fam in done_fams:
            existing = results[results_by_fam[fam]]
            if 'fels40' in existing and 'nw_accuracy' in existing['fels40']:
                nw_accs.append(existing['fels40']['nw_accuracy'])
            continue

        jax.clear_caches()  # OOM prevention (forces recompile; prefer geometric padding)

        try:
            # Load MSA and tree
            if config['type'] == 'pfam':
                msa, tree, C = load_pfam_family(fspec, pfam_dir, tree_dir)
            elif config['type'] == 'treefam':
                msa, tree, C = load_treefam_family(fspec, treefam_dir)
            elif config['type'] == 'balibase':
                msa, tree, C = load_balibase_family(fspec, balibase_ref_dir)

            if config['type'] == 'balibase':
                # BAliBASE: tree already has only remaining leaves,
                # reconstruct at root directly.
                tf = time.time()
                name_internal_nodes(tree)
                msa40 = {}
                for name in remaining:
                    if name not in msa:
                        continue
                    seq = msa[name].copy()
                    seq[seq < 0] = 20
                    seq[seq >= 20] = 20
                    msa40[name] = seq

                # For logp_true on BAliBASE: rebuild held-out leaf row
                # at the same gap-stripped column set used for posteriors.
                held_out_seq_kept = None
                try:
                    ref_path = os.path.join(balibase_ref_dir, fam)
                    ref_seqs_local = parse_fasta(ref_path)
                    if held_out in ref_seqs_local:
                        held_out_full = encode_aligned_seq(
                            ref_seqs_local[held_out])
                        rem_aln = {n: encode_aligned_seq(ref_seqs_local[n])
                                   for n in remaining if n in ref_seqs_local}
                        if rem_aln:
                            L_full = len(next(iter(rem_aln.values())))
                            keep = [col for col in range(L_full) if any(
                                rem_aln[n][col] >= 0 for n in remaining
                                if n in rem_aln)]
                            held_out_seq_kept = np.array(
                                [held_out_full[c] for c in keep],
                                dtype=np.int32)
                except Exception:
                    held_out_seq_kept = None

                pred_seq, logp_target, logp_true = _fels40_reconstruct_root(
                    tree, msa40, Q40, pi40,
                    held_out_seq=held_out_seq_kept)
                elapsed = time.time() - tf
            else:
                # held_out is in msa for unified / treefam loaders.
                held_out_seq = msa.get(held_out)
                pred_seq, elapsed, logp_target, logp_true = run_fels40(
                    tree, held_out, remaining, msa, Q40, pi40,
                    held_out_seq=held_out_seq)

            # Score
            nw = _nw_metrics(pred_seq, true_seq)
            fels40_result = {
                'pred_seq': pred_seq.tolist(),
                'pred_len': len(pred_seq),
                'true_len': len(true_seq),
                'time': round(elapsed, 3),
                'logp_target': logp_target,
                'logp_true': logp_true,
                **nw,
            }

            nw_accs.append(nw['nw_accuracy'])
            mean_acc = np.mean(nw_accs)

            log(f'[{fi+1}/{len(families)}] {fam}: '
                f'nw_acc={nw["nw_accuracy"]:.3f} '
                f'pred_len={len(pred_seq)} true_len={len(true_seq)} '
                f't={elapsed:.1f}s  (mean={mean_acc:.3f})')

            # Store result
            if fam in results_by_fam:
                results[results_by_fam[fam]]['fels40'] = fels40_result
            else:
                result = {
                    'family': fam,
                    'held_out': held_out,
                    'true_len': len(true_seq),
                    'n_cols': C,
                    'K': len(remaining),
                    'fels40': fels40_result,
                }
                results.append(result)
                results_by_fam[fam] = len(results) - 1

            n_done += 1

        except Exception as e:
            log(f'[{fi+1}/{len(families)}] {fam}: ERROR: {e}')
            traceback.print_exc()
            n_errors += 1
            continue

        # Save periodically
        if n_done % SAVE_EVERY == 0:
            _save(results, nw_accs, results_path, dataset)

    # Final save
    _save(results, nw_accs, results_path, dataset)

    log(f'Done: {n_done} families processed, {n_errors} errors')
    if nw_accs:
        log(f'Mean NW accuracy: {np.mean(nw_accs):.4f} '
            f'(n={len(nw_accs)})')


def _save(results, nw_accs, results_path, dataset):
    """Save results to JSON."""
    summary = {
        'dataset': dataset,
        'method': 'fels40',
        'n_families': len(results),
        'mean_nw_accuracy': float(np.mean(nw_accs)) if nw_accs else 0.0,
        'results': results,
    }
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log(f'Saved {len(results)} results to {results_path}')


if __name__ == '__main__':
    main()
