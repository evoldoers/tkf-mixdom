#!/usr/bin/env python3
"""Fels21 (21-state GTR with gap) held-out leaf reconstruction benchmark.

Unlike standard Felsenstein (20-state LG08 + Fitch parsimony for gaps),
fels21 treats gap as the 21st character. Predicted sequence = argmax over
21 states at each column; if argmax == 20 (gap), that column is excluded
from the output sequence.

Runs on four datasets:
  - unified_short: Pfam val families (spec: unified_benchmark_spec.json)
  - unified_long:  Pfam val families, longer (spec: unified_benchmark_long_spec.json)
  - treefam:       TreeFam families (spec: treefam_reconstruction_spec.json)
  - balibase:      BAliBASE families (spec: balibase_reconstruction_spec.json)

Usage:
    cd python && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run python -u \\
        experiments/fels21_reconstruction_benchmark.py --dataset unified_short
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

from tkfmixdom.jax.tree.ancestor import marginal_ancestor_all_columns_jax
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


# --- Data loading helpers ---

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
    """Convert aligned string sequences to integer-encoded MSA.

    Returns dict {name: (C,) int32 array} with -1 for gaps.
    """
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


# --- Fels21 reconstruction ---
#
# logp_target math (per-method confidence for ancestral reconstruction):
#   For each held-out leaf prediction, we record the JOINT log posterior of
#   the predicted residue sequence under the model:
#       logp_target = sum_{c in kept cols} log P(pred_aa[c] | MSA \ target,
#                                                  tree, model)
#   where pred_aa[c] is the argmax-over-21-states at column c at the
#   prediction node (the held-out leaf's parent in the pruned tree), and
#   "kept cols" excludes columns where argmax == 20 (predicted gap).
#
#   This is the user-requested "log posterior of the target leaf's predicted
#   sequence" — *not* the old `best_p` whole-MSA log probability.
#
# logp_true math (added 2026-05-04, ground-truth scoring under the same
# per-column posterior):
#       logp_true = sum_{c in non-gap cols of true held-out leaf}
#                       log P(true_aa[c] | MSA \ target, tree, model)
#   Skips columns where the held-out leaf's true residue is a gap
#   (parallel to how logp_target skips columns the method predicted as
#   gap). For fels21, true gaps map to state 20, but we skip those rows
#   to keep logp_true / logp_target directly comparable in scope.
#
#   headroom = logp_true - logp_target. headroom <= 0 only when the
#   gap-prediction column sets disagree; on calibrated methods the user
#   reads negative headroom as the model being over-confident relative
#   to truth (mass on its own argmax exceeds mass on the true residue).

def run_fels21(tree, held_out, remaining, msa, Q21, pi21,
               held_out_seq=None):
    """Run 21-state Felsenstein on tree re-rooted at target's parent.

    Gaps are treated as observed character 20 (not missing data).
    The argmax over 21 states determines the prediction:
    - argmax in 0..19: predicted amino acid
    - argmax == 20: predicted gap (excluded from output)

    Returns (pred_seq, elapsed_time, logp_target, logp_true).
    logp_target is the joint log posterior of the predicted residue
    sequence:
        logp_target = sum_{c in kept cols} log P(pred_aa[c] | MSA, tree, Q21)
    where the per-column posterior at the prediction node is computed
    by Felsenstein's pruning algorithm. Predicted-gap (argmax==20)
    columns are excluded — see math notes above the launcher main().

    logp_true is the joint log posterior of the GROUND-TRUTH held-out
    leaf residues under the same per-column posterior:
        logp_true = sum_{c: held_out_seq[c] is non-gap}
                        log P(held_out_seq[c] | MSA, tree, Q21)
    Columns where the held-out leaf is gap are skipped. If
    held_out_seq is None, logp_true is computed as 0.0 (caller should
    pass the held-out leaf's MSA row to get a meaningful number).
    """
    tf = time.time()

    # Re-root at target's parent
    pruned_tree, _ = prune_leaf_keep_parent(tree, held_out)
    if pruned_tree is None:
        return np.array([], dtype=np.int32), 0.0, 0.0, 0.0
    name_internal_nodes(pruned_tree)

    # Encode MSA as 21-state: gaps -> character 20 (observed, not missing)
    msa21 = {}
    for name in remaining:
        if name not in msa:
            continue
        seq = msa[name].copy()
        # Gaps (-1) and X residues (>=20) -> character 20 (gap state)
        seq[seq < 0] = 20
        seq[seq >= 20] = 20
        msa21[name] = seq

    if not msa21:
        return np.array([], dtype=np.int32), 0.0, 0.0, 0.0

    # Run 21-state Felsenstein
    # marginal_ancestor_all_columns_jax works with any alphabet size.
    # With 21-state Q and pi, and leaf observations encoded as 0..20,
    # it will compute 21-state posteriors. Character 20 is treated as
    # observed (one-hot at position 20) rather than missing data.
    ancestor, posteriors = marginal_ancestor_all_columns_jax(
        pruned_tree, msa21, Q21, pi21)

    # Extract prediction: argmax over 21 states
    # ancestor already has argmax per column, but for all-gap columns
    # (where all leaves have char 20 = gap), argmax will be 20 naturally.
    # Columns where argmax == 20 are predicted gaps -> exclude from output.
    pred_chars = []
    logp_target = 0.0
    posteriors_np = np.asarray(posteriors)
    for c in range(len(ancestor)):
        aa = int(ancestor[c])
        if aa == -1:
            # All-gap column (shouldn't happen with 21-state since gap is
            # a valid character, but handle anyway)
            continue
        if 0 <= aa < 20:
            pred_chars.append(aa)
            logp_target += float(np.log(max(posteriors_np[c, aa], 1e-300)))
        # aa == 20: predicted gap, skip — not part of the predicted
        # output sequence so its posterior does not enter logp_target.

    # logp_true: score per-column posteriors against the ground-truth
    # held-out leaf, skipping gap columns.
    logp_true = 0.0
    if held_out_seq is not None:
        n_post = posteriors_np.shape[0]
        for c in range(min(n_post, len(held_out_seq))):
            aa_true = int(held_out_seq[c])
            if 0 <= aa_true < 20:
                logp_true += float(np.log(max(
                    posteriors_np[c, aa_true], 1e-300)))
            # gap (aa_true < 0 or aa_true >= 20): skip per spec.

    pred_seq = np.array(pred_chars, dtype=np.int32)
    elapsed = time.time() - tf
    return pred_seq, elapsed, logp_target, logp_true


# --- Dataset loaders ---

def load_pfam_family(fspec, pfam_dir, tree_dir):
    """Load MSA and tree for a Pfam family.

    Prunes both MSA and tree to spec's `held_out ∪ remaining` so that
    methods see exactly the leaves the spec calls for. (Required for
    xhard subsampling: spec may name far fewer leaves than the full
    family Stockholm/Newick contain.)

    Returns (msa, tree, C) or raises an exception.
    """
    fam = fspec['family']
    held_out = fspec['held_out']
    remaining = fspec['remaining']
    C = fspec['n_cols']
    spec_leaves = set(remaining) | {held_out}

    # Find tree file
    tree_path = os.path.join(tree_dir, f'{fam}.nwk')
    if not os.path.exists(tree_path):
        tree_path = os.path.join(tree_dir, f'{fam}.tree')
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f'Tree not found for {fam}')

    # Find MSA file
    sto_path = os.path.join(pfam_dir, f'{fam}.sto')
    if not os.path.exists(sto_path):
        raise FileNotFoundError(f'MSA not found for {fam}')

    # Parse MSA, restrict to spec leaves only.
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

    # Strip empty columns: cols where NO kept-spec leaf has a residue.
    # (xhard's leaf subsampling can leave columns with residues only in
    # the dropped/subsampled leaves; those are empty in the kept submatrix
    # and contribute trivial all-absent predictions to F1 / NW alignment.
    # Removing them gives a "clean" benchmark that doesn't depend on the
    # spurious-empty-column count. Verified safe via empty_col_ablation:
    # F1 unchanged for parsimony / Felsenstein; <1% drift for d3f1-VBEM.)
    has_residue = np.zeros(C, dtype=bool)
    for seq in msa_full.values():
        has_residue |= (seq >= 0)
    kept_cols = np.where(has_residue)[0]
    msa = {n: seq[kept_cols] for n, seq in msa_full.items()}
    C = len(kept_cols)

    # Parse tree, prune to spec leaves only.
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
    """Load MSA and tree for a TreeFam family.

    Returns (msa, tree, C) or raises an exception.
    """
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
    # extra leaves would be inconsistent with the spec contract; fels40
    # additionally KeyErrors on such leaves (its msa cache is built only
    # from `remaining`).
    tree_leaf_names = {l.name for l in tree.leaves()}
    spec_leaves = set(remaining) | {held_out}
    if held_out not in tree_leaf_names:
        raise ValueError(f'held_out {held_out} not in tree')

    tree = prune_tree_to_msa(tree, spec_leaves)
    name_internal_nodes(tree)

    return msa, tree, C


def load_balibase_family(fspec, balibase_ref_dir):
    """Load MSA and tree for a BAliBASE family.

    Uses the reference (structural) alignment + FastTree.
    Returns (msa, tree, C) or raises an exception.
    """
    import tempfile

    family = fspec['family']
    held_out = fspec['held_out']
    remaining = fspec['remaining']

    ref_path = os.path.join(balibase_ref_dir, family)
    ref_seqs = parse_fasta(ref_path)

    # Build aligned integer MSA of remaining sequences
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

    # Build aligned FASTA for FastTree
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

    # Run FastTree
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
        description='Fels21 (21-state) reconstruction benchmark')
    parser.add_argument('--dataset', required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to run on')
    parser.add_argument('--recompute-missing-fields', action='store_true',
                        help='Re-run families whose existing fels21 entry '
                        'lacks logp_target / logp_true (or other newly-'
                        'added fields). Use this to backfill these '
                        'confidence fields on previously completed JSONs '
                        'without re-running every entry.')
    args = parser.parse_args()

    dataset = args.dataset
    config = DATASET_CONFIGS[dataset]

    log(f'Dataset: {dataset}')

    # Load fels21 model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'pfam', 'fels21_cherryml.npz')
    data = np.load(model_path)
    Q21 = data['Q21']
    pi21 = data['pi21']
    log(f'Loaded fels21 model: Q21 {Q21.shape}, pi21 {pi21.shape}')

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
                                f'fels21_reconstruction_{dataset}.json')
    results = []
    done_fams = set()
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                rd = json.load(f)
            if isinstance(rd, dict) and 'results' in rd:
                results = list(rd['results'])
                # Newly-added required fields. If --recompute-missing-fields
                # is set, families whose fels21 entry is missing any of
                # these are re-run (rather than skipped) to backfill.
                required_fields = {'logp_target', 'logp_true'}
                done_fams = set()
                for r in results:
                    if 'family' not in r or 'fels21' not in r:
                        continue
                    fres = r['fels21']
                    if args.recompute_missing_fields and not (
                            required_fields.issubset(fres.keys())):
                        continue  # treat as not done; will be recomputed
                    done_fams.add(r['family'])
                log(f'Resume: {len(results)} prior results, '
                    f'{len(done_fams)} families with fels21'
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
            # Already done
            existing = results[results_by_fam[fam]]
            if 'fels21' in existing and 'nw_accuracy' in existing['fels21']:
                nw_accs.append(existing['fels21']['nw_accuracy'])
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

            # For BAliBASE, the tree doesn't include held_out
            # (we built the tree from remaining only), so we need
            # to handle reconstruction differently: reconstruct at root.
            if config['type'] == 'balibase':
                # BAliBASE: tree already has only remaining leaves,
                # reconstruct at root directly.
                tf = time.time()
                name_internal_nodes(tree)
                msa21 = {}
                for name in remaining:
                    if name not in msa:
                        continue
                    seq = msa[name].copy()
                    seq[seq < 0] = 20
                    seq[seq >= 20] = 20
                    msa21[name] = seq

                ancestor, posteriors = marginal_ancestor_all_columns_jax(
                    tree, msa21, Q21, pi21)

                # For logp_true on BAliBASE: load the held-out leaf's
                # residues at the same gap-stripped column set used for
                # posteriors. We rebuild from the reference FASTA and
                # apply the same `keep` mask used in load_balibase_family.
                held_out_seq_kept = None
                try:
                    ref_path = os.path.join(balibase_ref_dir, fam)
                    ref_seqs_local = parse_fasta(ref_path)
                    if held_out in ref_seqs_local:
                        held_out_full = encode_aligned_seq(
                            ref_seqs_local[held_out])
                        # Re-derive keep mask from remaining only
                        names_l = remaining
                        rem_aln = {n: encode_aligned_seq(ref_seqs_local[n])
                                   for n in names_l if n in ref_seqs_local}
                        if rem_aln:
                            L_full = len(next(iter(rem_aln.values())))
                            keep = [col for col in range(L_full) if any(
                                rem_aln[n][col] >= 0 for n in names_l
                                if n in rem_aln)]
                            held_out_seq_kept = np.array(
                                [held_out_full[c] for c in keep],
                                dtype=np.int32)
                except Exception:
                    held_out_seq_kept = None

                # Same logp_target computation as run_fels21 (math notes above
                # main()): joint log posterior of predicted residue sequence.
                pred_chars = []
                logp_target = 0.0
                logp_true = 0.0
                posteriors_np = np.asarray(posteriors)
                for c in range(len(ancestor)):
                    aa = int(ancestor[c])
                    if aa == -1:
                        continue
                    if 0 <= aa < 20:
                        pred_chars.append(aa)
                        logp_target += float(
                            np.log(max(posteriors_np[c, aa], 1e-300)))
                if held_out_seq_kept is not None:
                    n_post = posteriors_np.shape[0]
                    for c in range(min(n_post, len(held_out_seq_kept))):
                        aa_true = int(held_out_seq_kept[c])
                        if 0 <= aa_true < 20:
                            logp_true += float(np.log(max(
                                posteriors_np[c, aa_true], 1e-300)))
                pred_seq = np.array(pred_chars, dtype=np.int32)
                elapsed = time.time() - tf
            else:
                # held_out is in msa for unified / treefam loaders.
                held_out_seq = msa.get(held_out)
                pred_seq, elapsed, logp_target, logp_true = run_fels21(
                    tree, held_out, remaining, msa, Q21, pi21,
                    held_out_seq=held_out_seq)

            # Score
            nw = _nw_metrics(pred_seq, true_seq)
            fels21_result = {
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
                results[results_by_fam[fam]]['fels21'] = fels21_result
            else:
                result = {
                    'family': fam,
                    'held_out': held_out,
                    'true_len': len(true_seq),
                    'n_cols': C,
                    'K': len(remaining),
                    'fels21': fels21_result,
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
        'method': 'fels21',
        'n_families': len(results),
        'mean_nw_accuracy': float(np.mean(nw_accs)) if nw_accs else 0.0,
        'results': results,
    }
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log(f'Saved {len(results)} results to {results_path}')


if __name__ == '__main__':
    main()
