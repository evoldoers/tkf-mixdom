#!/usr/bin/env python3
"""Fit per-column gamma rate multiplier labels for Pfam seed alignments.

Uses subby's MixturePosterior to compute posterior P(gamma_class | column)
for each column of each MSA, given a guide tree and base substitution model.
Saves MAP labels and full posteriors as JSON per family.

Usage:
    python fit_gamma_rates.py \
        --msa-dir ~/bio-datasets/data/pfam-seed/ \
        --out gamma_labels/ \
        --n-gamma 4 \
        --alpha 0.5

    # Process specific families:
    python fit_gamma_rates.py \
        --msa-dir ~/bio-datasets/data/pfam-seed/ \
        --out gamma_labels/ \
        --families PF00001,PF00002

Output:
    gamma_labels/PF00001.G4.json
    gamma_labels/PF00002.G4.json
    ...

Each JSON file contains:
    family: str
    n_cols: int
    n_gamma: int (G)
    alpha: float
    rates: [float] (G quantile midpoints, mean-normalized)
    labels: [int] (n_cols, MAP gamma class 0..G-1)
    posteriors: [[float]] (G x n_cols)
"""

import argparse
import json
import gzip
import os
import sys

# Ensure subby is importable (installed at ~/subby)
_subby_path = os.path.expanduser('~/subby')
if _subby_path not in sys.path and os.path.isdir(_subby_path):
    sys.path.insert(0, _subby_path)
import time
import numpy as np

# Subby and JAX imports are deferred to _ensure_imports() to guarantee
# float64 is enabled before JAX initializes. Float32 loses rate category
# discrimination in Felsenstein pruning on deep trees.

HAVE_SUBBY = False
USE_JAX = False
_IMPORTS_DONE = False

def _ensure_imports():
    """Lazily import subby and JAX with float64 enabled."""
    global HAVE_SUBBY, USE_JAX, _IMPORTS_DONE
    global MixturePosterior, LogLike, gamma_rate_categories, scale_model
    global jukes_cantor_model, model_from_rate_matrix
    global SubbyJaxTree, SubbyTree, parse_stockholm, jnp
    global pad_tree_and_alignment

    if _IMPORTS_DONE:
        return

    import jax
    jax.config.update('jax_enable_x64', True)

    try:
        from subby.jax import MixturePosterior as _MP, LogLike as _LL
        from subby.jax import pad_tree_and_alignment as _ptaa
        from subby.jax.models import (
            gamma_rate_categories as _grc, scale_model as _sm,
            jukes_cantor_model as _jcm, model_from_rate_matrix as _mfrm,
        )
        from subby.jax.types import Tree as _JaxTree
        from subby.formats import parse_stockholm as _ps, Tree as _ST
        import jax.numpy as _jnp

        MixturePosterior = _MP; LogLike = _LL
        pad_tree_and_alignment = _ptaa
        gamma_rate_categories = _grc; scale_model = _sm
        jukes_cantor_model = _jcm; model_from_rate_matrix = _mfrm
        SubbyJaxTree = _JaxTree; SubbyTree = _ST
        parse_stockholm = _ps; jnp = _jnp
        HAVE_SUBBY = True; USE_JAX = True
    except ImportError:
        try:
            from subby.oracle import (
                MixturePosterior as _MP, LogLike as _LL,
                gamma_rate_categories as _grc, scale_model as _sm,
                jukes_cantor_model as _jcm,
            )
            from subby.oracle.oracle import model_from_rate_matrix as _mfrm
            from subby.formats import parse_stockholm as _ps, Tree as _ST

            MixturePosterior = _MP; LogLike = _LL
            gamma_rate_categories = _grc; scale_model = _sm
            jukes_cantor_model = _jcm; model_from_rate_matrix = _mfrm
            SubbyTree = _ST; parse_stockholm = _ps
            HAVE_SUBBY = True; USE_JAX = False
        except ImportError:
            pass

    _IMPORTS_DONE = True


def _build_lg08_subby_model():
    """Build LG08 protein model in subby format (JAX or oracle)."""
    _ensure_imports()
    from tkfmixdom.jax.distill.maraschino import get_lg08

    S_lg, pi_lg = get_lg08()
    S_lg, pi_lg = np.array(S_lg), np.array(pi_lg)

    # Build rate matrix Q = S * diag(pi), normalized to mean rate 1
    Q = S_lg * pi_lg[None, :]
    Q = Q - np.diag(np.diag(Q))
    Q = Q - np.diag(Q.sum(axis=1))
    mean_rate = -np.sum(pi_lg * np.diag(Q))
    Q = Q / mean_rate

    if USE_JAX:
        return model_from_rate_matrix(jnp.array(Q), jnp.array(pi_lg))
    else:
        return model_from_rate_matrix(Q, pi_lg)


def parse_sto_file(path):
    """Parse a Pfam Stockholm MSA file (plain or gzipped)."""
    _ensure_imports()
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            text = f.read()
    else:
        with open(path) as f:
            text = f.read()
    return parse_stockholm(text)


def pairwise_distances(alignment, A=20):
    """Compute pairwise p-distance matrix from alignment.

    Args:
        alignment: (R, C) int array (0..A-1 for residues, A for gap/missing)
        A: alphabet size

    Returns:
        (R, R) distance matrix (Poisson-corrected p-distance)
    """
    R, C = alignment.shape
    dist = np.zeros((R, R))
    for i in range(R):
        for j in range(i + 1, R):
            # Count positions where both have residues
            both_present = (alignment[i] < A) & (alignment[j] < A)
            n_sites = both_present.sum()
            if n_sites == 0:
                dist[i, j] = dist[j, i] = 5.0  # max distance
                continue
            n_diff = ((alignment[i] != alignment[j]) & both_present).sum()
            p = n_diff / n_sites
            # Poisson correction for proteins (A=20)
            frac = 1.0 - p * A / (A - 1)
            if frac <= 0.01:
                dist[i, j] = dist[j, i] = 5.0
            else:
                dist[i, j] = dist[j, i] = -(A - 1) / A * np.log(frac)
    return dist


def build_tree_for_subby(dist_matrix, names):
    """Build NJ tree from distance matrix and convert for subby.

    Uses Newick roundtrip: build tree -> serialize to Newick -> parse via subby.
    This ensures correct preorder indexing required by subby's pruning algorithm.

    Returns subby (tree_result, leaf_name_to_idx) from parse_newick + combine.
    """
    _ensure_imports()
    from tkfmixdom.jax.tree.guide_tree import neighbor_joining
    from subby.formats import parse_newick as subby_parse_newick

    tree = neighbor_joining(dist_matrix, names)

    # Serialize to Newick string
    def _to_newick(node):
        if not node.children:
            return f"{node.name}:{node.branch_length:.6f}"
        child_strs = ','.join(_to_newick(c) for c in node.children)
        if node.parent is None:
            return f"({child_strs});"
        return f"({child_strs}):{node.branch_length:.6f}"

    newick = _to_newick(tree)
    return subby_parse_newick(newick)


def fit_family_gamma(msa_path, n_gamma, alpha, base_model=None, max_seqs=200):
    """Fit per-column gamma rates for one family.

    Args:
        msa_path: path to .sto or .sto.gz file
        n_gamma: number of gamma categories (G)
        alpha: gamma shape parameter
        base_model: optional subby model (default: LG08 for proteins, JC otherwise)
        max_seqs: max sequences for tree building (default: 200)

    Returns:
        dict with family, n_cols, n_gamma, alpha, rates, labels, posteriors
        or None if family couldn't be processed
    """
    _ensure_imports()
    # Parse MSA
    try:
        aln_result = parse_sto_file(msa_path)
    except Exception as e:
        print(f"  SKIP: parse error: {e}", file=sys.stderr)
        return None

    family = os.path.basename(msa_path).replace('.sto.gz', '').replace('.sto', '')
    alignment = np.array(aln_result['alignment'], dtype=np.int32)
    leaf_names = aln_result['leaf_names']
    R, C = alignment.shape

    if R < 3:
        print(f"  SKIP: too few sequences ({R})", file=sys.stderr)
        return None
    if C < 1:
        print(f"  SKIP: empty alignment", file=sys.stderr)
        return None

    # Detect alphabet size from the alignment's alphabet metadata
    A = len(aln_result.get('alphabet', [None] * 20))

    # Subsample if too many sequences (NJ is O(n^3))
    if R > max_seqs:
        rng = np.random.RandomState(hash(family) % (2**31))
        keep = rng.choice(R, max_seqs, replace=False)
        alignment = alignment[keep]
        leaf_names = [leaf_names[i] for i in keep]
        R = max_seqs

    # Build NJ tree from pairwise p-distances and combine with alignment.
    # The LG08 model is rate-normalized (mean rate = 1 substitution/site/unit time).
    # Pairwise p-distances measure total substitution path distance. NJ distributes
    # this across branches, so branch lengths approximate evolutionary time in
    # the same units. However, the p-distances from the MSA already include the
    # effect of rate variation across sites. We don't rescale here; the gamma
    # model will partition the rate variation, with the mean rate category near 1.0
    # matching the average rate that the tree was estimated under.
    try:
        dist = pairwise_distances(alignment, A=A)
        tree_result = build_tree_for_subby(dist, leaf_names)
    except Exception as e:
        print(f"  SKIP: tree error: {e}", file=sys.stderr)
        return None

    # Combine tree with alignment via subby.
    # combine_tree_alignment maps leaf sequences to tree positions by name,
    # fills internal nodes with ungapped-unobserved tokens, and returns
    # a properly indexed (n_nodes, C) alignment + Tree namedtuple.
    try:
        from subby.formats import combine_tree_alignment as subby_combine
        combined = subby_combine(tree_result, aln_result)
        full_alignment = combined.alignment
        tree_combined = combined.tree

        tree_dict = tree_combined
    except Exception as e:
        print(f"  SKIP: combine error: {e}", file=sys.stderr)
        return None

    # Base model: use LG08 for proteins, JC for others
    if base_model is None:
        if A == 20:
            base_model = _build_lg08_subby_model()
        else:
            base_model = jukes_cantor_model(A)

    # Gamma rate categories
    rates, weights = gamma_rate_categories(alpha, n_gamma)
    rates = np.array(rates)
    weights = np.array(weights)
    models = [scale_model(base_model, float(r)) for r in rates]

    # Compute per-column posteriors.
    # Pad tree and alignment to geometric bin sizes so that JAX reuses
    # JIT-compiled functions across families with similar sizes, instead
    # of recompiling for every distinct (R, C) pair.
    try:
        if USE_JAX:
            aln_jax = jnp.array(full_alignment)
            tree_jax = SubbyJaxTree(
                parentIndex=jnp.array(tree_dict.parentIndex),
                distanceToParent=jnp.array(tree_dict.distanceToParent))
            # Pad to geometric bins for JIT cache reuse
            tree_pad, aln_pad, R_real, C_real = pad_tree_and_alignment(
                tree_jax, aln_jax)
            lw_jax = jnp.log(jnp.array(weights))
            posteriors_padded = np.array(
                MixturePosterior(aln_pad, tree_pad, models, lw_jax))
            # Extract real columns (discard padded)
            posteriors = posteriors_padded[:, :C_real]
        else:
            posteriors = np.array(MixturePosterior(
                full_alignment, tree_dict, models, np.log(weights)))
    except Exception as e:
        print(f"  SKIP: MixturePosterior error: {e}", file=sys.stderr)
        return None

    # MAP labels
    # For columns with very few observed residues, the posterior is nearly
    # uniform (all rates equally likely). We flag these with label -1.
    n_observed = np.array([(full_alignment[:, c] < A).sum() for c in range(C)])
    min_observed = max(3, R // 4)  # need at least 3 or 25% of leaves
    labels = np.argmax(posteriors, axis=0)
    labels[n_observed < min_observed] = -1  # uninformative column
    labels = labels.tolist()

    return {
        'family': family,
        'n_cols': int(C),
        'n_gamma': int(n_gamma),
        'alpha': float(alpha),
        'rates': rates.tolist(),
        'labels': labels,
        'posteriors': posteriors.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fit per-column gamma rate labels for Pfam families')
    parser.add_argument('--msa-dir', required=True,
                        help='Directory with *.sto / *.sto.gz files')
    parser.add_argument('--out', required=True,
                        help='Output directory for JSON files')
    parser.add_argument('--n-gamma', type=int, default=4,
                        help='Number of gamma categories (default: 4)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Gamma shape parameter (default: 0.5)')
    parser.add_argument('--families', type=str, default=None,
                        help='Comma-separated family IDs (default: all)')
    parser.add_argument('--max-seqs', type=int, default=200,
                        help='Max sequences per family for tree building (default: 200)')
    args = parser.parse_args()

    _ensure_imports()

    if not HAVE_SUBBY:
        print("ERROR: subby not importable. Install or add ~/subby to PYTHONPATH.",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    # Find MSA files
    if args.families:
        families = args.families.split(',')
        msa_files = []
        for fam in families:
            for ext in ['.sto', '.sto.gz']:
                p = os.path.join(args.msa_dir, fam + ext)
                if os.path.exists(p):
                    msa_files.append(p)
                    break
    else:
        msa_files = sorted(
            [os.path.join(args.msa_dir, f)
             for f in os.listdir(args.msa_dir)
             if f.endswith('.sto') or f.endswith('.sto.gz')])

    G = args.n_gamma
    n_done = 0
    n_skip = 0
    t0 = time.time()

    for i, msa_path in enumerate(msa_files):
        family = os.path.basename(msa_path).replace('.sto.gz', '').replace('.sto', '')
        out_path = os.path.join(args.out, f'{family}.G{G}.json')

        if os.path.exists(out_path):
            n_done += 1
            continue

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (n_done + n_skip + 1) / max(elapsed, 0.01)
            remaining = (len(msa_files) - i) / max(rate, 0.01)
            print(f"[{i+1}/{len(msa_files)}] {family}  "
                  f"({n_done} done, {n_skip} skipped, "
                  f"ETA {remaining/60:.0f}m)", file=sys.stderr)

        result = fit_family_gamma(msa_path, G, args.alpha, max_seqs=args.max_seqs)

        if result is None:
            n_skip += 1
            continue

        with open(out_path, 'w') as f:
            json.dump(result, f)

        n_done += 1

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} families, {n_skip} skipped, {elapsed:.1f}s total",
          file=sys.stderr)


if __name__ == '__main__':
    main()
