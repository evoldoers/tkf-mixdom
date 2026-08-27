#!/usr/bin/env python3
"""
train_pfam.py — Train MixDom on Pfam via alignment-constrained Baum-Welch.

Picks cherry pairs from Pfam Stockholm MSAs, runs 1D Forward-Backward
with associative scan (O(log L)) over the MixDom HMM, accumulates
Holmes-Rubin/BDI expected sufficient statistics, and does M-steps
in model parameter space (lambda, mu, fragExt, Q, pi).

Two-phase design:
  Phase 1 (once): Scan MSAs, select cherries, cache a compact "pair manifest"
    storing (file_index, row1_index, row2_index, t_est) — no names, no seqs.
  Phase 2 (per EM iter): Stream through manifest, re-read only needed pairs
    from each file, run 1D FB, accumulate suff stats. No cherry recomputation.

Features:
  - Streaming: one MSA at a time, constant memory regardless of Pfam size
  - Pair manifest cache: cherry selection done once, reused across iterations
  - Auto-checkpointing & auto-resume (params, which files done in current
    E-step, cumulative suff stats — lose almost nothing on interrupt)
  - Configurable pseudocounts (in suff-stat space)
  - Reports suff stats, log-likelihoods, and param snapshots as it goes

Usage:
  # Quick smoke test:
  python train_pfam.py --msa-dir pfam/ --families PF00001,PF00002 --n-dom 2 --n-iter 3

  # Train on all of Pfam:
  python train_pfam.py --msa-dir pfam/ --n-dom 4 --n-frag 2 --n-iter 20

  # Budget-based training: 4 hours total, 10 EM iterations.
  # Samples random Pfam families on iter 1 until E-step time exceeds
  # 4/10 = 0.4h = 24min, then freezes that subset for remaining iters.
  python train_pfam.py --msa-dir pfam/ --n-dom 3 --n-iter 10 --budget-hours 4

  # Resume interrupted run (auto-detects checkpoint):
  python train_pfam.py --msa-dir pfam/ --checkpoint pfam/train_d4_f2.npz

  # Inspect a trained model:
  python train_pfam.py --inspect pfam/train_d4_f2.npz
"""

import argparse
import hashlib
import json
import os
import sys
import time

# Enable float64 in JAX BEFORE any jax import / op, so eigh / FB / M-step
# run at the same precision used by the test suite (conftest.py also
# enables x64). The L'Hôpital and conservation-warning thresholds in
# bdi.py assume float64; running at the JAX default of float32 silently
# inflates cancellation error in the M-step quadratic and the
# transition_matrix spectral decomposition. (Audit ledger #8.)
import jax  # noqa: E402  (must precede the jax_enable_x64 update)
jax.config.update("jax_enable_x64", True)

import numpy as np

# ============================================================
# Logging
# ============================================================
_log_start = time.monotonic()


def _log(msg, end='\n'):
    elapsed = time.monotonic() - _log_start
    line = f"[{elapsed:8.1f}s] {msg}"
    sys.stderr.write(line + end)
    sys.stderr.flush()


# ============================================================
# Stockholm parsing & cherry selection
# ============================================================
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA = len(AMINO_ACIDS)
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}


def parse_stockholm(filepath):
    """Parse Stockholm alignment. Returns list of aligned sequences (ordered)."""
    import gzip
    names = []
    seqs = {}
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                name, seq = parts[0], parts[1]
                if name in seqs:
                    seqs[name] += seq
                else:
                    names.append(name)
                    seqs[name] = seq
    return names, [seqs[n] for n in names]


def _read_two_seqs(filepath, idx1, idx2):
    """Read only two sequences (by row index) from a Stockholm file."""
    import gzip
    names_seen = []
    seqs = {}
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                name, seq = parts[0], parts[1]
                if name not in seqs:
                    names_seen.append(name)
                    seqs[name] = seq
                else:
                    seqs[name] += seq
    if idx1 >= len(names_seen) or idx2 >= len(names_seen):
        return None, None
    return seqs[names_seen[idx1]], seqs[names_seen[idx2]]


def ungap(seq):
    return ''.join(c for c in seq if c in AA_TO_IDX)


def p_distance(seq1, seq2):
    matches = mismatches = 0
    for a, b in zip(seq1, seq2):
        if a in AA_TO_IDX and b in AA_TO_IDX:
            if a == b:
                matches += 1
            else:
                mismatches += 1
    total = matches + mismatches
    return mismatches / total if total > 0 else 1.0


def pdist_to_evo_time(pdist):
    if pdist >= 0.95:
        return 5.0
    corrected = 1.0 - pdist * (AA / (AA - 1.0))
    if corrected <= 0.01:
        return 5.0
    return -np.log(corrected)


def select_cherries(n_seqs, dist_fn, max_pairs=None):
    """Greedy nearest-neighbor cherry pairing. Returns list of (idx1, idx2).

    dist_fn(i, j) returns distance between sequences i and j.
    """
    if n_seqs < 2:
        return []

    # Compute pairwise distances
    dists = np.ones((n_seqs, n_seqs)) * 1e10
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            d = dist_fn(i, j)
            dists[i, j] = d
            dists[j, i] = d

    paired = set()
    pairs = []
    while len(paired) < n_seqs - 1:
        best_d = 1e10
        best_i, best_j = -1, -1
        for i in range(n_seqs):
            if i in paired:
                continue
            for j in range(i + 1, n_seqs):
                if j in paired:
                    continue
                if dists[i, j] < best_d:
                    best_d = dists[i, j]
                    best_i, best_j = i, j
        if best_i < 0:
            break
        pairs.append((best_i, best_j))
        paired.add(best_i)
        paired.add(best_j)
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


def _aa_composition(seq):
    comp = np.zeros(AA)
    for c in seq:
        if c in AA_TO_IDX:
            comp[AA_TO_IDX[c]] += 1
    total = comp.sum()
    if total > 0:
        comp /= total
    return comp


def _aligned_pair_to_int_arrays(seq1, seq2):
    """Convert string pair to gapped integer arrays (-1 = gap)."""
    a = np.array([AA_TO_IDX.get(c, -1) for c in seq1], dtype=np.int32)
    b = np.array([AA_TO_IDX.get(c, -1) for c in seq2], dtype=np.int32)
    mask = (a >= 0) | (b >= 0)
    return a[mask], b[mask]


# ============================================================
# Clan-aware train/val/test splits
# ============================================================

def load_clan_membership(clan_file):
    """Load Pfam-A.clans.tsv.gz → dict mapping family accession → clan_id.

    Families with no clan get a unique pseudo-clan '_singleton_PFxxxxx'.
    """
    import gzip
    clans = {}
    opener = gzip.open if clan_file.endswith('.gz') else open
    with opener(clan_file, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                fam = parts[0]
                clan = parts[1] if parts[1] else f'_singleton_{fam}'
                clans[fam] = clan
    return clans


def clan_aware_split(msa_files, clan_file, split, ratios=(0.8, 0.1, 0.1),
                     seed=0):
    """Split MSA files by clan so no clan spans multiple splits.

    Returns list of file indices for the requested split.
    """
    clans = load_clan_membership(clan_file)

    # Group files by clan
    clan_to_files = {}
    for fi, f in enumerate(msa_files):
        fam = os.path.basename(f).split('.')[0]
        clan = clans.get(fam, f'_singleton_{fam}')
        clan_to_files.setdefault(clan, []).append(fi)

    # Shuffle clans deterministically, assign to splits
    rng = np.random.RandomState(seed)
    clan_ids = sorted(clan_to_files.keys())
    rng.shuffle(clan_ids)

    n_total = len(msa_files)
    train_r, val_r, test_r = ratios
    train_end = int(n_total * train_r)
    val_end = int(n_total * (train_r + val_r))

    split_map = {'train': [], 'val': [], 'test': []}
    cumulative = 0
    for clan_id in clan_ids:
        files = clan_to_files[clan_id]
        if cumulative < train_end:
            split_map['train'].extend(files)
        elif cumulative < val_end:
            split_map['val'].extend(files)
        else:
            split_map['test'].extend(files)
        cumulative += len(files)

    _log(f"  Clan-aware split (seed={seed}): "
         f"train={len(split_map['train'])}, "
         f"val={len(split_map['val'])}, "
         f"test={len(split_map['test'])} families "
         f"({len(clan_to_files)} clans)")

    if split not in split_map:
        _log(f"ERROR: unknown split '{split}'. Use train/val/test.")
        sys.exit(1)
    return sorted(split_map[split])


def _find_clan_file(msa_dir):
    """Auto-detect clan membership file."""
    candidates = [
        os.path.join(msa_dir, 'Pfam-A.clans.tsv.gz'),
        os.path.join(msa_dir, '..', 'Pfam-A.clans.tsv.gz'),
        os.path.expanduser('~/bio-datasets/data/pfam/seed/Pfam-A.clans.tsv.gz'),
        os.path.expanduser('~/bio-datasets/data/pfam-seed/Pfam-A.clans.tsv.gz'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ============================================================
# Pair manifest: compact cache of cherry selections
# ============================================================
# Stored as structured numpy array:
#   file_idx (uint16), row1 (uint16), row2 (uint16), t_est (float32)
# ~8 bytes per pair. 100K pairs = 800KB.

MANIFEST_DTYPE = np.dtype([
    ('file_idx', np.uint16),
    ('row1', np.uint16),
    ('row2', np.uint16),
    ('t_est', np.float32),
    ('n_cols', np.uint16),   # alignment columns (after gap removal)
])


def _build_pairs_for_file(fi, msa_file, sto_index=None):
    """Build cherry pairs for a single MSA file.

    Returns list of (row1, row2, t_est) tuples, or empty list.
    If sto_index is provided, uses it instead of re-parsing.
    """
    if sto_index is not None:
        seqs = [sto_index.get_sequence(i) for i in range(len(sto_index))]
        n = len(seqs)
    else:
        names, seqs = parse_stockholm(msa_file)
        n = len(seqs)
    if n < 2:
        return []

    # Cap sequences considered for cherry selection to avoid O(n²) blowup
    MAX_SEQS_FOR_CHERRIES = 200
    if n > MAX_SEQS_FOR_CHERRIES:
        rng_sub = np.random.RandomState(fi)
        subset = sorted(rng_sub.choice(n, MAX_SEQS_FOR_CHERRIES, replace=False))
        seqs_sub = [seqs[i] for i in subset]
        idx_map = subset  # map back to original indices
    else:
        seqs_sub = seqs
        idx_map = list(range(n))
    n_sub = len(seqs_sub)

    if n_sub <= 50:
        def dist_fn(i, j, _seqs=seqs_sub):
            return p_distance(_seqs[i], _seqs[j])
    else:
        comps = np.array([_aa_composition(ungap(s)) for s in seqs_sub])
        def dist_fn(i, j, _c=comps):
            return float(np.sum((_c[i] - _c[j]) ** 2))

    cherries = select_cherries(n_sub, dist_fn)
    pairs = []
    for idx1, idx2 in cherries:
        # Map subset indices back to original
        orig1, orig2 = idx_map[idx1], idx_map[idx2]
        aln_i, aln_j = _aligned_pair_to_int_arrays(seqs[orig1], seqs[orig2])
        n_cols = len(aln_i)
        if n_cols == 0:
            continue
        pd = p_distance(seqs[orig1], seqs[orig2])
        t_est = pdist_to_evo_time(pd)
        pairs.append((int(orig1), int(orig2), float(t_est)))
    return pairs


def build_pair_manifest(msa_files):
    """First pass: scan all MSAs, select cherries, return manifest array.

    Returns (manifest: structured array, t_median: float).
    """
    records = []
    total_pairs = 0
    for fi, msa_file in enumerate(msa_files):
        fam = os.path.basename(msa_file).split('.')[0]
        names, seqs = parse_stockholm(msa_file)
        n = len(seqs)
        if n < 2:
            continue
        if n <= 50:
            def dist_fn(i, j, _seqs=seqs):
                return p_distance(_seqs[i], _seqs[j])
        else:
            comps = np.array([_aa_composition(ungap(s)) for s in seqs])
            def dist_fn(i, j, _c=comps):
                return float(np.sum((_c[i] - _c[j]) ** 2))
        cherries = select_cherries(n, dist_fn)
        for idx1, idx2 in cherries:
            aln_i, aln_j = _aligned_pair_to_int_arrays(seqs[idx1], seqs[idx2])
            n_cols = len(aln_i)
            if n_cols == 0:
                continue
            pd = p_distance(seqs[idx1], seqs[idx2])
            t_est = pdist_to_evo_time(pd)
            records.append((fi, idx1, idx2, t_est, min(n_cols, 65535)))
            total_pairs += 1

        if (fi + 1) % max(1, len(msa_files) // 20) == 0 or fi == len(msa_files) - 1:
            _log(f"  Manifest: {fi+1}/{len(msa_files)} MSAs, {total_pairs} pairs [{fam}]")

    manifest = np.array(records, dtype=MANIFEST_DTYPE)
    if len(manifest) > 0:
        t_vals = manifest['t_est']
        c_vals = manifest['n_cols']
        t_median = float(np.median(t_vals))
        _log(f"  Manifest summary:")
        _log(f"    Pairs: {len(manifest)}")
        _log(f"    Columns: median={int(np.median(c_vals))}, "
             f"mean={int(np.mean(c_vals))}, "
             f"min={int(c_vals.min())}, max={int(c_vals.max())}, "
             f"total={int(c_vals.sum())}")
        _log(f"    Tau (evo time): median={np.median(t_vals):.3f}, "
             f"mean={np.mean(t_vals):.3f}, "
             f"range=[{t_vals.min():.3f}, {t_vals.max():.3f}]")
        pcts = np.percentile(t_vals, [10, 25, 50, 75, 90])
        _log(f"    Tau percentiles: 10%={pcts[0]:.3f} 25%={pcts[1]:.3f} "
             f"50%={pcts[2]:.3f} 75%={pcts[3]:.3f} 90%={pcts[4]:.3f}")
    else:
        t_median = 1.0
    return manifest, t_median


# ============================================================
# Checkpoint save/load
# ============================================================
def _save_checkpoint(path, params, suff_stats, step_file_idx,
                     em_iter, log_probs, config, manifest=None, t_rep=None,
                     budget_files=None, family_timings=None):
    """Save everything needed to resume.

    step_file_idx: index into file list — how many files fully processed this E-step.
    budget_files: frozen list of file indices selected by budget sampling (or None).
    family_timings: per-family E-step wall times from budget sampling (or None).
    """
    save = {
        'main_ins': np.float64(params['main_ins']),
        'main_del': np.float64(params['main_del']),
        'dom_ins': np.array(params['dom_ins']),
        'dom_del': np.array(params['dom_del']),
        'dom_weights': np.array(params['dom_weights']),
        'frag_weights': np.array(params['frag_weights']),
        'ext_rates': np.array(params['ext_rates']),
    }
    # Per-domain substitution model (optional)
    if 'dom_Qs' in params:
        save['dom_Qs'] = np.array(params['dom_Qs'])
        save['dom_pis'] = np.array(params['dom_pis'])
        save['dom_S_exch'] = np.array(params['dom_S_exch'])
    # MixDom2 per-fragment site class parameters (optional)
    if params.get('n_classes', 0) > 1 and 'classdist' in params:
        save['n_classes_frag'] = np.int32(params['n_classes'])
        save['classdist'] = np.array(params['classdist'])
        save['class_pis'] = np.array(params['class_pis'])
        save['class_S_exch'] = np.array(params['class_S_exch'])
    save.update({
        'em_iter': np.int32(em_iter),
        'step_file_idx': np.int32(step_file_idx),
        'log_probs': np.array(log_probs),
        '_config': np.array(json.dumps(config)),
    })
    if t_rep is not None:
        save['t_rep'] = np.float64(t_rep)
    if manifest is not None:
        save['manifest'] = manifest
    if budget_files is not None:
        if budget_files and isinstance(budget_files[0], tuple):
            # Interleaved: list of (fi, (row1, row2, t_est)) tuples
            bf_arr = np.array(
                [(fi, int(p[0]), int(p[1]), float(p[2]))
                 for fi, p in budget_files],
                dtype=[('file_idx', np.int32), ('row1', np.int32),
                       ('row2', np.int32), ('t_est', np.float32)])
            save['budget_pairs'] = bf_arr
        else:
            # Sequential: list of file indices
            save['budget_files'] = np.array(budget_files, dtype=np.int32)
    if family_timings is not None:
        save['family_timings'] = np.array(family_timings)
    for k, v in suff_stats.items():
        if isinstance(v, np.ndarray):
            save[f'ss_{k}'] = v
        elif isinstance(v, (int, float, np.floating, np.integer)):
            save[f'ss_{k}'] = np.float64(v)
    np.savez_compressed(path, **save)


def _load_checkpoint(path):
    """Load checkpoint."""
    data = np.load(path, allow_pickle=True)

    params = {
        'main_ins': float(data['main_ins']),
        'main_del': float(data['main_del']),
        'dom_ins': data['dom_ins'].copy(),
        'dom_del': data['dom_del'].copy(),
        'dom_weights': data['dom_weights'].copy(),
        'frag_weights': data['frag_weights'].copy(),
        'ext_rates': data['ext_rates'].copy(),
    }
    # Validate and fix frag_weights normalization
    fw = params['frag_weights']
    fw_sums = fw.sum(axis=1)
    if np.any(np.abs(fw_sums - 1.0) > 1e-6):
        print(f"WARNING: frag_weights row sums != 1 (got {fw_sums}), normalizing")
        params['frag_weights'] = fw / fw_sums[:, None]
    # Per-domain substitution model (optional)
    if 'dom_Qs' in data:
        params['dom_Qs'] = data['dom_Qs'].copy()
        params['dom_pis'] = data['dom_pis'].copy()
        params['dom_S_exch'] = data['dom_S_exch'].copy()
    # MixDom2 per-fragment site class parameters (optional)
    if 'n_classes_frag' in data and int(data['n_classes_frag']) > 1:
        params['n_classes'] = int(data['n_classes_frag'])
        params['classdist'] = data['classdist'].copy()
        params['class_pis'] = data['class_pis'].copy()
        S_ex_loaded = data['class_S_exch'].copy()
        n_cls = int(data['n_classes_frag'])
        if S_ex_loaded.ndim == 2:
            # Migrate legacy shared-S checkpoint: replicate per class, then
            # absorb the old class_gamma multipliers into each per-class S.
            S_ex_3d = np.tile(S_ex_loaded[None], (n_cls, 1, 1))
            if 'class_gamma' in data:
                g = np.asarray(data['class_gamma']).reshape(n_cls, 1, 1)
                S_ex_3d = S_ex_3d * g
            params['class_S_exch'] = S_ex_3d
        else:
            params['class_S_exch'] = S_ex_loaded

    suff_stats = {}
    for k in data.files:
        if k.startswith('ss_'):
            name = k[3:]
            v = data[k]
            suff_stats[name] = v.item() if v.ndim == 0 else v.copy()

    step_file_idx = int(data.get('step_file_idx', 0))
    em_iter = int(data.get('em_iter', 0))
    log_probs = list(data.get('log_probs', []))
    config = json.loads(str(data['_config'])) if '_config' in data else {}
    manifest = data['manifest'] if 'manifest' in data else None
    t_rep = float(data['t_rep']) if 't_rep' in data else None
    if 'budget_pairs' in data:
        # Interleaved: structured array of (fi, row1, row2, t_est)
        bp = data['budget_pairs']
        budget_files = [(int(r['file_idx']), (int(r['row1']), int(r['row2']), float(r['t_est'])))
                        for r in bp]
    elif 'budget_files' in data:
        budget_files = list(data['budget_files'])
    else:
        budget_files = None
    family_timings = list(data['family_timings']) if 'family_timings' in data else None

    return (params, suff_stats, step_file_idx, em_iter, log_probs, config,
            manifest, t_rep, budget_files, family_timings)


# ============================================================
# Maraschino parameter conversion
# ============================================================
def _load_maraschino_as_train_params(path, n_dom, n_frag):
    """Load a Maraschino .npz file and convert to train_pfam param format.

    Maraschino format (unconstrained raw params):
        log_lam0, log_mu0, log_lam (N,), log_mu (N,), logit_r (N,) or (N,F),
        log_v (N,), log_pi (N,AA), log_S (AA,AA) or (N,AA,AA),
        log_alpha_gamma, plus optional logit_frag_weights (N,F)

    Train_pfam format:
        main_ins, main_del, dom_ins (N,), dom_del (N,),
        dom_weights (N,), frag_weights (N,F), ext_rates (N,F),
        and optionally dom_pis (N,AA), dom_S_exch (N,AA,AA), dom_Qs (N,AA,AA)
    """
    from tkfmixdom.jax.distill.maraschino import load_params as _mar_load
    params, n_domains_mar, n_classes = _mar_load(path)

    if n_domains_mar != n_dom:
        raise ValueError(
            f"Maraschino file has {n_domains_mar} domains but "
            f"--n-dom={n_dom} was requested")

    # Indel rates
    main_ins = float(params['lam0'])
    main_del = float(params['mu0'])
    dom_ins = np.array(params['lam'])
    dom_del = np.array(params['mu'])

    # Domain weights from v (stationary domain frequencies)
    dom_weights = np.array(params['v'])
    dom_weights = dom_weights / dom_weights.sum()  # ensure normalized

    # Fragment weights and extension rates
    if params.get('frag_weights') is not None and params['frag_weights'].ndim == 2:
        # Multi-fragment: r_frags (N,F), frag_weights (N,F)
        mar_n_frag = params['frag_weights'].shape[1]
        if mar_n_frag != n_frag:
            raise ValueError(
                f"Maraschino file has {mar_n_frag} fragments but "
                f"--n-frag={n_frag} was requested")
        frag_weights = np.array(params['frag_weights'])
        ext_rates = np.array(params['r_frags'])
    else:
        # Single-fragment: r is (N,) effective extension rate
        frag_weights = np.ones((n_dom, n_frag)) / n_frag
        r_eff = np.array(params['r'])  # (N,)
        ext_rates = np.tile(r_eff[:, None], (1, n_frag))

    # Substitution model: build per-domain Q from S_exch and pi
    S_exch = np.array(params['S_exch'])
    pis = np.array(params['pi'])  # (N, AA)
    dom_Qs = np.zeros((n_dom, AA, AA))
    dom_S_exch = np.zeros((n_dom, AA, AA))

    if S_exch.ndim == 3:
        # Per-domain S_exch: (N, AA, AA)
        dom_S_exch = S_exch.copy()
    else:
        # Shared S_exch: (AA, AA) — replicate for each domain
        for d in range(n_dom):
            dom_S_exch[d] = S_exch

    # Build dom_Qs from S × π using the same construction as the per-class
    # E-step (constrained.py L1167-1170) so that MixDom1's stored dom_Qs
    # is bit-identical to a per-class rebuild from class_S_exch × class_pis
    # under classdist=identity. NOT rate-normalized — the paper's M-step
    # (tkf.tex L786, L821-822) doesn't specify a rate convention.
    for d in range(n_dom):
        Q = dom_S_exch[d] * pis[d, None, :]
        np.fill_diagonal(Q, 0.0)
        Q[np.diag_indices(AA)] = -Q.sum(axis=1)
        dom_Qs[d] = Q

    result = {
        'main_ins': main_ins, 'main_del': main_del,
        'dom_ins': dom_ins, 'dom_del': dom_del,
        'dom_weights': dom_weights,
        'frag_weights': frag_weights,
        'ext_rates': ext_rates,
        'dom_Qs': dom_Qs,
        'dom_pis': pis,
        'dom_S_exch': dom_S_exch,
    }

    # MixDom2 per-fragment site class params: when the Maraschino file is a
    # MixDom2 fit checkpoint (class_pi / class_S_exch / class_dist /
    # n_classes_per_frag present in the constrained dict produced by
    # tkfmixdom.jax.distill.maraschino.load_params), forward them to
    # train_pfam under its own naming convention. Without this the
    # downstream training silently runs as MixDom1 (no site classes) — the
    # checkpoint save guard `params.get('n_classes', 0) > 1 and 'classdist'
    # in params` evaluates False and the class data is dropped permanently
    # when the SVI-BW checkpoint is written.
    if ('class_pi' in params and 'class_S_exch' in params
            and 'class_dist' in params and 'n_classes_per_frag' in params):
        n_cls = int(params['n_classes_per_frag'])
        result['n_classes'] = n_cls
        result['class_pis'] = np.asarray(params['class_pi'])      # (C, A)
        result['class_S_exch'] = np.asarray(params['class_S_exch'])  # (C, A, A)
        result['classdist'] = np.asarray(params['class_dist'])    # (D, F, C)

    _log(f"  Loaded Maraschino params from {path}")
    _log(f"    n_dom={n_dom}, n_frag={n_frag}, n_classes={n_classes}")
    if 'n_classes' in result:
        _log(f"    MixDom2 class params loaded: classdist {result['classdist'].shape}, "
             f"class_pis {result['class_pis'].shape}, "
             f"class_S_exch {result['class_S_exch'].shape}")
    _log(f"    lam0={main_ins:.6f}, mu0={main_del:.6f}")
    for d in range(n_dom):
        _log(f"    dom {d}: lam={dom_ins[d]:.6f}, mu={dom_del[d]:.6f}, "
             f"v={dom_weights[d]:.4f}, r={ext_rates[d].tolist()}")
    return result


def _save_params_as_maraschino(path, params, n_dom, n_frag):
    """Save train_pfam params in Maraschino raw (unconstrained) format.

    Inverse of _load_maraschino_as_train_params: converts train_pfam's
    constrained parameters back to Maraschino's unconstrained raw format
    so the result can be loaded by maraschino.py for re-distillation.
    """
    # Inverse softplus: log(exp(x) - 1); for x > 20, ~x
    def inv_softplus(x):
        x = np.float64(x)
        return np.where(x > 20, x, np.log(np.expm1(np.maximum(x, 1e-10))))

    # Inverse sigmoid: log(p / (1-p))
    def logit(p):
        p = np.clip(np.float64(p), 1e-7, 1.0 - 1e-7)
        return np.log(p / (1.0 - p))

    save_dict = {}

    # Indel rates (softplus parameterization, offset by 1e-6)
    save_dict['log_lam0'] = inv_softplus(params['main_ins'] - 1e-6)
    save_dict['log_mu0'] = inv_softplus(params['main_del'] - 1e-6)
    save_dict['log_lam'] = inv_softplus(np.array(params['dom_ins']) - 1e-6)
    save_dict['log_mu'] = inv_softplus(np.array(params['dom_del']) - 1e-6)

    # Domain weights v: log-softmax parameterization
    dw = np.array(params['dom_weights'], dtype=np.float64)
    dw = np.maximum(dw, 1e-10)
    save_dict['log_v'] = np.log(dw)  # softmax input

    # Extension rates: sigmoid parameterization
    ext = np.array(params['ext_rates'], dtype=np.float64)
    if ext.ndim == 3:
        # MixDom2: (N, F, F) fragment transition matrix
        save_dict['logit_r'] = logit(ext)         # (N, F, F)
        fw = np.array(params['frag_weights'], dtype=np.float64)
        fw = np.maximum(fw, 1e-10)
        save_dict['logit_frag_weights'] = np.log(fw)  # softmax input
    elif ext.ndim == 2 and ext.shape[1] > 1:
        # Multi-fragment MixDom1
        save_dict['logit_r'] = logit(ext)         # (N, F)
        fw = np.array(params['frag_weights'], dtype=np.float64)
        fw = np.maximum(fw, 1e-10)
        save_dict['logit_frag_weights'] = np.log(fw)  # softmax input
    else:
        # Single fragment or uniform: save effective r
        if ext.ndim == 2:
            ext = ext[:, 0]
        save_dict['logit_r'] = logit(ext)          # (N,)

    # Equilibrium frequencies pi: log-softmax parameterization
    if 'dom_pis' in params:
        pis = np.array(params['dom_pis'], dtype=np.float64)
    else:
        from tkfmixdom.jax.core.protein import rate_matrix_lg
        _, pi_lg = rate_matrix_lg()
        pis = np.tile(np.array(pi_lg), (n_dom, 1))
    pis = np.maximum(pis, 1e-10)
    save_dict['log_pi'] = np.log(pis)              # (N, AA)

    # Exchangeability matrix S_exch
    if 'dom_S_exch' in params:
        S_exch = np.array(params['dom_S_exch'], dtype=np.float64)
        if S_exch.ndim == 3:
            # Per-domain: (N, AA, AA)
            S_exch = np.maximum(S_exch, 1e-30)
            save_dict['log_S'] = np.log(S_exch)
        else:
            S_exch = np.maximum(S_exch, 1e-30)
            save_dict['log_S'] = np.log(S_exch)
    else:
        from tkfmixdom.jax.core.protein import rate_matrix_lg
        Q_lg, pi_lg = rate_matrix_lg()
        S_lg = np.array(Q_lg / np.maximum(np.array(pi_lg)[None, :], 1e-30)
                        * (1.0 - np.eye(AA)))
        S_lg = (S_lg + S_lg.T) / 2
        S_lg = np.maximum(S_lg, 1e-30)
        save_dict['log_S'] = np.log(S_lg)           # (AA, AA)

    # Gamma shape: default to 1.0 (no rate variation)
    save_dict['log_alpha_gamma'] = inv_softplus(np.float64(1.0) - 0.01)

    # Metadata
    save_dict['n_domains'] = n_dom
    save_dict['n_classes'] = 1  # gamma classes

    np.savez(path, **save_dict)
    _log(f"  Saved Maraschino-format params to {path}")


# ============================================================
# Suff stats
# ============================================================
def _zero_suff_stats(n_states, n_dom=1):
    return {
        'agg_n_chi': np.zeros((n_states, n_states)),
        'total_ll': 0.0,
        'n_pairs': 0,
        'n_files': 0,
        'match_counts': np.zeros((AA, AA)),
        # Per-domain substitution stats (for Holmes-Rubin M-step)
        'dom_match_counts': np.zeros((n_dom, AA, AA)),  # weighted (a,b) pairs
        'dom_insert_counts': np.zeros((n_dom, AA)),     # insert character counts
        'dom_delete_counts': np.zeros((n_dom, AA)),     # delete character counts
    }


def _log_prior(params, args, Q_lg=None, pi_lg=None):
    """MAP log-prior matching the priors implied by the M-step pseudocounts.

    Formal priors (all match corresponding pseudocount additions in the
    closed-form M-step formulas, so that observed_LL + log_prior is the
    objective actually maximized per M-step):
      - Gamma(a_ins, b_ins) on main_ins, dom_ins[d] (CLI --ins-prior)
      - Gamma(a_del, b_del) on main_del, dom_del[d] (CLI --del-prior)
      - Dirichlet(alpha_dom) symmetric on dom_weights (CLI --dom-dirichlet)
      - Dirichlet(alpha_frag) symmetric on frag_weights[d] per domain
        (CLI --frag-dirichlet)
      - Dirichlet on each ext_rates row: alpha_ext for F transition entries,
        beta_ext for the (implicit) termination entry (CLI --ext-alpha,
        --ext-beta)
      - (MixDom2) LG-informed Dirichlet(pi_pseudo*pi_LG + 1) on each
        class_pis[c] (same semantics as per-domain Q via
        m_step_subst_option1)
      - (MixDom2) Gamma(S_pseudo*S_lg_ij+1, S_pseudo) per off-diagonal
        class_S_exch[c, i, j] entry — applied independently per class
      - (MixDom2) Dirichlet(1) on classdist[d,f,:] (flat, contributes 0)

    Returns float. Constants independent of params are dropped.
    """
    lp = 0.0

    # ---- Gamma priors on indel rates ----
    a_ins, b_ins = args.ins_prior
    a_del, b_del = args.del_prior
    mi = float(params['main_ins'])
    md = float(params['main_del'])
    lp += (a_ins - 1) * np.log(max(mi, 1e-30)) - b_ins * mi
    lp += (a_del - 1) * np.log(max(md, 1e-30)) - b_del * md
    for r in np.asarray(params['dom_ins']).ravel():
        r = float(r)
        lp += (a_ins - 1) * np.log(max(r, 1e-30)) - b_ins * r
    for r in np.asarray(params['dom_del']).ravel():
        r = float(r)
        lp += (a_del - 1) * np.log(max(r, 1e-30)) - b_del * r

    # ---- Symmetric Dirichlet on domain weights ----
    alpha_dom = float(args.dom_dirichlet)
    dw = np.maximum(np.asarray(params['dom_weights']), 1e-30)
    lp += (alpha_dom - 1) * float(np.log(dw).sum())

    # ---- Symmetric Dirichlet on fragment weights (per domain) ----
    alpha_frag = float(args.frag_dirichlet)
    fw = np.maximum(np.asarray(params['frag_weights']), 1e-30)
    lp += (alpha_frag - 1) * float(np.log(fw).sum())

    # ---- Dirichlet on ext_rates rows: (alpha_ext,...,alpha_ext; beta_ext) ----
    a_ext = float(args.ext_alpha)
    b_ext = float(args.ext_beta)
    ext = np.asarray(params['ext_rates'])
    # Each row (per domain, per src frag) is a categorical over F transitions
    # plus a termination probability term = 1 - row.sum().
    # Supports (N,F,F) MixDom2, (N,F) legacy, and (N,) scalar ext cases.
    if ext.ndim == 3:      # (N, F, F) — row = ext[d, f_src, :]
        rows = ext.reshape(-1, ext.shape[-1])
    elif ext.ndim == 2:    # (N, F) — single fragment transition per domain
        rows = ext
    else:                  # (N,) scalar
        rows = ext.reshape(-1, 1)
    for row in rows:
        row = np.maximum(row, 1e-30)
        term = max(1.0 - row.sum(), 1e-30)
        lp += (a_ext - 1) * float(np.log(row).sum())
        lp += (b_ext - 1) * np.log(term)

    # ---- MixDom2 site-class priors ----
    if params.get('n_classes', 0) > 1 and 'classdist' in params:
        # LG-informed Dirichlet(pi_pseudo * pi_LG + 1) on each class_pis[c]:
        #   pseudocount per state i = pi_pseudo * pi_LG_i (matches the
        #   V + pi_pseudo*pi_LG_i augmentation in the class_pis M-step).
        if 'class_pis' in params and pi_lg is not None:
            N_pi = float(getattr(args, 'pi_pseudo', 10.0))
            cp = np.maximum(np.asarray(params['class_pis']), 1e-30)
            pi_lg_np = np.asarray(pi_lg)
            lp += N_pi * float((pi_lg_np[None, :] * np.log(cp)).sum())

        # Gamma(S_pseudo * S_lg_ij + 1, S_pseudo) per off-diagonal
        # class_S_exch[c, i, j] — applied independently per class.
        # Data-matching the per-class M-step's
        #   num += S_pseudo*S_lg, den += S_pseudo
        # on each class's S update.
        if 'class_S_exch' in params and Q_lg is not None and pi_lg is not None:
            N_S = float(getattr(args, 'S_pseudo', 5.0))
            S_ex = np.asarray(params['class_S_exch'])
            pi_lg_np = np.asarray(pi_lg)
            Q_lg_np = np.asarray(Q_lg)
            if S_ex.ndim == 2:
                S_ex = S_ex[None]  # treat legacy shared-S as a single-class stack
            C, A, _ = S_ex.shape
            S_lg = Q_lg_np / np.maximum(pi_lg_np[None, :], 1e-30)
            S_lg = S_lg * (1.0 - np.eye(A))
            S_lg = (S_lg + S_lg.T) / 2.0
            triu_i, triu_j = np.triu_indices(A, k=1)
            S_lg_off = S_lg[triu_i, triu_j]
            for cc in range(C):
                S_off = np.maximum(S_ex[cc, triu_i, triu_j], 1e-30)
                lp += float(np.sum(
                    N_S * S_lg_off * np.log(S_off) - N_S * S_off))

        # Symmetric Dirichlet(alpha_classdist) on classdist per (d, f)
        # (exposed via --classdist-dirichlet, default 1.0 = uniform/MLE).
        alpha_cd = float(getattr(args, 'classdist_dirichlet', 1.0))
        if abs(alpha_cd - 1.0) > 1e-12:
            cd = np.maximum(np.asarray(params['classdist']), 1e-30)
            lp += (alpha_cd - 1) * float(np.log(cd).sum())

    return float(lp)


# ============================================================
# PairSource abstraction: unified pair loading for all modes
# ============================================================

class PairSource:
    """Abstract source of training pairs for all modes.

    Yields (file_path_or_id, pairs_list) batches where pairs_list contains
    (row1, row2, t_est) tuples for Stockholm mode or pre-decoded tuples
    for precompiled mode.
    """

    def iter_pairs(self, budget_seconds=None, seed=0):
        """Yield (source_id, pairs_list, pre_decoded_list_or_None) batches.

        For Stockholm mode: yields (msa_file_path, [(row1, row2, t_est), ...], None).
        For precompiled mode: yields (family_id, [], [(x_int, y_int, states,
            anc_chars, desc_chars, t_est), ...]).
        Stops when budget_seconds is exceeded (if set).
        """
        raise NotImplementedError

    def load_all_pairs(self):
        """Load all pairs into memory. Returns list of pre-decoded tuples.

        Each tuple: (x_int, y_int, states, anc_chars, desc_chars, t_est).
        Used by Adam mode which needs everything in memory.
        """
        raise NotImplementedError

    @property
    def n_sources(self):
        """Number of source files/families."""
        raise NotImplementedError

    @property
    def description(self):
        """Human-readable description for logging."""
        return "PairSource"


class StockholmPairSource(PairSource):
    """Loads pairs lazily from Stockholm MSA files (current behavior).

    Wraps the existing cherry-selection + StoIndex machinery.
    """

    def __init__(self, msa_files, pairs_by_file=None, active_files=None,
                 budget_files=None):
        self.msa_files = msa_files
        self.pairs_by_file = pairs_by_file or {}
        self.active_files = active_files or list(range(len(msa_files)))
        self.budget_files = budget_files

    def iter_pairs(self, budget_seconds=None, seed=0):
        """Yield (msa_file, pairs_for_file, None) for each active file."""
        start = time.monotonic()
        for fi in self.active_files:
            if budget_seconds and (time.monotonic() - start) >= budget_seconds:
                break
            msa_file = self.msa_files[fi]
            if self.budget_files and isinstance(self.budget_files, list) \
                    and self.budget_files and isinstance(self.budget_files[0], tuple):
                file_pairs = [p for ffi, p in self.budget_files if ffi == fi]
            else:
                file_pairs = self.pairs_by_file.get(fi, [])
            yield msa_file, file_pairs, None

    def load_all_pairs(self):
        """Load all pairs from Stockholm files. Used by Adam mode."""
        from tkfmixdom.jax.simulate.msa import alignment_to_states
        all_pairs = []
        for fi in self.active_files:
            msa_file = self.msa_files[fi]
            pairs = self.pairs_by_file.get(fi)
            if pairs is None:
                pairs = _build_pairs_for_file(fi, msa_file)
            names, seqs = parse_stockholm(msa_file)
            for row1, row2, t_est in pairs:
                x = _aligned_pair_to_int_arrays(seqs[row1], seqs[row2])
                if len(x[0]) == 0:
                    continue
                states, anc_chars, desc_chars = alignment_to_states(x[0], x[1])
                if states:
                    all_pairs.append((x[0], x[1], states, anc_chars, desc_chars, t_est))
        return all_pairs

    @property
    def n_sources(self):
        return len(self.active_files)

    @property
    def description(self):
        return f"StockholmPairSource({len(self.msa_files)} files, {len(self.active_files)} active)"


class PrecompiledPairSource(PairSource):
    """Streams pairs from precompiled JSONL.zst shards.

    Reads shards produced by precompile_pairs.py, decodes X/A/Y records,
    and yields pre-decoded pairs grouped by source family.
    """

    def __init__(self, precompiled_dir, max_alignment_len=None):
        self.precompiled_dir = precompiled_dir
        self.max_alignment_len = max_alignment_len

        manifest_path = os.path.join(precompiled_dir, 'manifest.json')
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.shard_files = self.manifest['shard_files']
        self.n_pairs = self.manifest['n_pairs']
        self.n_families = self.manifest['n_families']
        self._all_decoded = None  # lazy cache

    def _read_shard(self, shard_name):
        """Read and decompress a single shard, returning list of record dicts."""
        import zstandard as zstd
        shard_path = os.path.join(self.precompiled_dir, shard_name)
        with open(shard_path, 'rb') as f:
            compressed = f.read()
        dctx = zstd.ZstdDecompressor()
        data = dctx.decompress(compressed)
        lines = data.decode('utf-8').split('\n')
        return [json.loads(line.strip()) for line in lines if line.strip()]

    def _decode_all(self):
        """Decode all records from all shards. Cached after first call."""
        if self._all_decoded is not None:
            return self._all_decoded

        from tkfmixdom.jax.util.pair_format import decode_pair

        all_decoded = []
        for shard_name in self.shard_files:
            records = self._read_shard(shard_name)
            for rec in records:
                x_int, y_int, states, anc_chars, desc_chars, t_est, *_ = decode_pair(rec)
                if self.max_alignment_len and len(states) > self.max_alignment_len:
                    continue
                all_decoded.append((
                    x_int, y_int, states, anc_chars, desc_chars, t_est,
                    rec.get('fam', 'unknown')))
        self._all_decoded = all_decoded
        return all_decoded

    def iter_pairs(self, budget_seconds=None, seed=0):
        """Yield (family_id, [], pre_decoded_list) grouped by family."""
        all_decoded = self._decode_all()

        # Group by family
        from collections import defaultdict
        by_family = defaultdict(list)
        for item in all_decoded:
            x_int, y_int, states, anc_chars, desc_chars, t_est, fam = item
            by_family[fam].append((x_int, y_int, states, anc_chars, desc_chars, t_est))

        # Shuffle family order for interleaving
        rng = np.random.RandomState(seed)
        families = list(by_family.keys())
        rng.shuffle(families)

        start = time.monotonic()
        for fam in families:
            if budget_seconds and (time.monotonic() - start) >= budget_seconds:
                break
            yield fam, [], by_family[fam]

    def load_all_pairs(self):
        """Load all pairs into memory (fast -- no parsing overhead)."""
        all_decoded = self._decode_all()
        return [(x, y, st, ac, dc, t) for x, y, st, ac, dc, t, _fam in all_decoded]

    @property
    def n_sources(self):
        return self.n_families

    @property
    def description(self):
        return (f"PrecompiledPairSource({self.precompiled_dir}, "
                f"{self.n_pairs} pairs, {self.n_families} families, "
                f"{len(self.shard_files)} shards)")


# ============================================================
# Process pairs from one file using manifest
# ============================================================
from tkfmixdom.jax.util.padding import pad_to_bin as _pad_to_bin, GEOM_BINS as _GEOM_BINS


def _get_classdist_params(params):
    """Extract per-fragment site class params from train params, or None."""
    if params.get('n_classes', 0) > 1 and 'classdist' in params:
        return {
            'classdist': params['classdist'],
            'class_pis': params['class_pis'],
            'class_S_exch': params['class_S_exch'],
        }
    return None


def _process_file_pairs(msa_file, pairs_for_file, chi_params, st, Q_lg, pi_lg,
                        N, alignment_to_states,
                        dom_Qs=None, dom_pis=None, n_dom=1, n_frag=1,
                        sto_index=None, pre_decoded=None,
                        classdist_params=None):
    """Process all cherry pairs from one file.

    Uses padded FB (forward_backward_1d_padded) with geometric bin sizes
    to eliminate JIT recompilation for different sequence lengths.
    One-time JIT cost per bin size (~16s); subsequent calls ~155ms.

    Per-pair t coherence: each pair's chi (TKF nested transitions) is built
    at that pair's own t_est, matching the per-pair t_est used to construct
    substitution emissions. (Previously took a single ``log_chi`` matrix
    pre-built at the run's representative t — that path was incoherent
    with per-pair emissions and produced biased BDI suff stats. See
    `_process_pairs_batched` docstring.)

    pairs_for_file: list of (row1, row2, t_est) from manifest.
    chi_params: params dict with keys main_ins, main_del, dom_ins, dom_del,
        dom_weights, frag_weights, ext_rates — used to build per-pair
        log_chi inside the per-pair loop via `_build_log_chi_stack`.
    dom_Qs: optional (n_dom, A, A) per-domain rate matrices
    dom_pis: optional (n_dom, A) per-domain equilibrium distributions
    sto_index: optional StoIndex for fast random access (avoids re-parsing)
    pre_decoded: optional list of (x_int, y_int, states, anc_chars, desc_chars, t_est)
        When provided, skip file parsing entirely and use these decoded pairs.
    Returns suff_stats delta dict, or None.
    """
    import jax.numpy as jnp
    from tkfmixdom.jax.dp.hmm import forward_backward_1d_padded, NEG_INF
    from tkfmixdom.jax.core.ctmc import transition_matrix
    from tkfmixdom.jax.core.params import M, I, D

    SS, EE = 0, 1  # MixDom start/end
    per_domain_subst = dom_Qs is not None and dom_pis is not None

    # When pre_decoded is provided, skip all file parsing
    if pre_decoded is not None:
        seqs = None  # not needed
        names = None
    elif sto_index is not None:
        seqs = [sto_index.get_sequence(i) for i in range(len(sto_index))]
        names = sto_index.names
    else:
        names, seqs = parse_stockholm(msa_file)
    if seqs is not None and not seqs:
        return None

    delta = {
        'agg_n_chi': np.zeros((N, N)),
        'total_ll': 0.0,
        'n_pairs': 0,
        'n_files': 1,
        'match_counts': np.zeros((AA, AA)),
        'dom_match_counts': np.zeros((n_dom, AA, AA)),
        'dom_insert_counts': np.zeros((n_dom, AA)),
        'dom_delete_counts': np.zeros((n_dom, AA)),
    }
    # Eagerly initialise per-class accumulators when classdist is provided,
    # so that empty pairs (which short-circuit before the per-pair body)
    # don't leave the dict missing keys that later non-empty pairs will
    # try to update.
    if (classdist_params is not None
            and 'classdist' in classdist_params):
        n_cls_init = int(np.asarray(classdist_params['classdist']).shape[2])
        delta['class_match_counts'] = np.zeros((n_cls_init, AA, AA))
        delta['class_insert_counts'] = np.zeros((n_cls_init, AA))
        delta['class_delete_counts'] = np.zeros((n_cls_init, AA))
        delta['classdist_counts'] = np.zeros((n_dom, n_frag, n_cls_init))
    st_np = np.array(st)
    m_mask = st_np == M
    i_mask = st_np == I
    d_mask = st_np == D

    # Build per-domain state masks for posterior splitting
    dom_m_masks = []  # dom_m_masks[d] = boolean mask over states for domain d's M-type
    dom_i_masks = []
    dom_d_masks = []
    for dd in range(n_dom):
        dm = np.zeros(N, dtype=bool)
        di_m = np.zeros(N, dtype=bool)
        dd_m = np.zeros(N, dtype=bool)
        for s in range(2, N):
            sd = (s - 2) // (5 * n_frag)
            if sd == dd:
                if st_np[s] == M:
                    dm[s] = True
                elif st_np[s] == I:
                    di_m[s] = True
                elif st_np[s] == D:
                    dd_m[s] = True
        dom_m_masks.append(dm)
        dom_i_masks.append(di_m)
        dom_d_masks.append(dd_m)

    # Build the list of (states, anc_chars, desc_chars, t_est) tuples to process
    if pre_decoded is not None:
        # Pre-decoded pairs: each item is (x_int, y_int, states, anc_chars, desc_chars, t_est)
        pair_tuples = [(item[2], item[3], item[4], item[5]) for item in pre_decoded]
    else:
        # Stockholm pairs: parse from file
        pair_tuples = []
        for row1, row2, t_est in pairs_for_file:
            if row1 >= len(seqs) or row2 >= len(seqs):
                continue
            aln_i, aln_j = _aligned_pair_to_int_arrays(seqs[row1], seqs[row2])
            if len(aln_i) == 0:
                continue
            # Process both directions for reversibility enforcement
            for anc_aln, desc_aln in [(aln_i, aln_j), (aln_j, aln_i)]:
                states, anc_chars, desc_chars = alignment_to_states(anc_aln, desc_aln)
                if states:
                    pair_tuples.append((states, anc_chars, desc_chars, t_est))

    for states, anc_chars, desc_chars, t_est in pair_tuples:
        # NOTE: empty pairs (len(states) == 0) are a real event in the
        # MixDom2 model distribution — they correspond to the SS→EE
        # direct transition (probability 1-κ at stationary). Discarding
        # them conditions the M-step on "non-empty alignment", which
        # biases λ̂ upward by removing zero-event samples from the
        # Poisson-rate denominator. Process them through FB; the L=0
        # case yields pair_ll = log chi[SS, EE] and a single n_chi
        # entry at (SS, EE), which correctly contributes to T_total
        # without any spurious transition counts.
        L = len(states)

        # Per-pair chi at this pair's t_est (coherent with per-pair emissions
        # below). Single-element t_array → squeeze to (N, N).
        log_chi = _build_log_chi_stack(
            chi_params, jnp.asarray([float(t_est)]))[0]

        # Empty-pair fast path: no emissions, FB yields pair_ll =
        # log chi[SS, EE] and n_chi[SS, EE] = 1.
        if L == 0:
            padded_L = _pad_to_bin(1)
            padded_emit = jnp.full((padded_L, N), NEG_INF)
            pair_ll, posteriors, n_chi = forward_backward_1d_padded(
                log_chi, padded_emit, seq_length=0,
                init_state=SS, final_state=EE)
            delta['agg_n_chi'] += np.asarray(n_chi)
            delta['total_ll'] += float(pair_ll)
            delta['n_pairs'] += 1
            continue

        # Build per-domain substitution matrices if available
        use_classdist = (classdist_params is not None
                         and 'classdist' in classdist_params)

        # Hoisted: build full-length anc/desc arrays once for either
        # branch — the JAX emit builder consumes the filled-length form.
        sa_h = np.array(states, dtype=np.int32)
        af_h = np.zeros(len(states), dtype=np.int32)
        df_h = np.zeros(len(states), dtype=np.int32)
        _ai, _di = 0, 0
        for _idx, _s in enumerate(states):
            if _s == M:
                af_h[_idx] = anc_chars[_ai]; df_h[_idx] = desc_chars[_di]
                _ai += 1; _di += 1
            elif _s == I:
                df_h[_idx] = desc_chars[_di]; _di += 1
            elif _s == D:
                af_h[_idx] = anc_chars[_ai]; _ai += 1

        if use_classdist:
            # MixDom2 per-fragment-per-class emissions
            cdp = classdist_params
            n_cls = cdp['classdist'].shape[2]
            # Build per-class P(t) — keep linear and (optionally) log forms.
            class_subs_lin = np.zeros((n_cls, AA, AA))
            class_pis_lin = np.zeros((n_cls, AA))
            S_per_class = np.array(cdp['class_S_exch'])  # (C, A, A)
            for cc in range(n_cls):
                pi_c = np.array(cdp['class_pis'][cc])
                Q_c = S_per_class[cc] * pi_c[None, :]
                np.fill_diagonal(Q_c, 0.0)
                Q_c[np.diag_indices(AA)] = -Q_c.sum(axis=1)
                P_c = np.array(transition_matrix(jnp.array(Q_c), t_est))
                class_subs_lin[cc] = P_c
                class_pis_lin[cc] = pi_c

            sa, af, df = sa_h, af_h, df_h

            # Per-domain sub/pi (linear) as fallback for SS/EE states.
            if per_domain_subst:
                subs_dom_lin = np.stack([
                    np.array(transition_matrix(jnp.array(dom_Qs[dd]), t_est))
                    for dd in range(n_dom)])
                pis_dom_lin = np.array(dom_pis)
            else:
                P_lg = np.array(transition_matrix(Q_lg, t_est))
                subs_dom_lin = np.broadcast_to(
                    P_lg[None, :, :], (n_dom, AA, AA)).copy()
                pis_dom_lin = np.broadcast_to(
                    np.array(pi_lg)[None, :], (n_dom, AA)).copy()

            # JAX-native emit (replaces NumPy-only
            # `mixdom_constrained_emissions_vectorized`). One code path
            # for emission building across the SVI-BW E-step paths.
            from tkfmixdom.jax.dp.hmm import pair_hmm_emissions_constrained as _emit_jax
            log_emit = _emit_jax(
                jnp.asarray(st), jnp.asarray(sa),
                jnp.asarray(af), jnp.asarray(df),
                jnp.asarray(subs_dom_lin), jnp.asarray(pis_dom_lin),
                n_dom, n_frag,
                classdist=jnp.asarray(np.array(cdp['classdist'])),
                class_sub_matrices=jnp.asarray(class_subs_lin),
                class_pis=jnp.asarray(class_pis_lin))

        elif per_domain_subst:
            sub_matrices_t = np.stack([
                np.array(transition_matrix(jnp.array(dom_Qs[dd]), t_est))
                for dd in range(n_dom)])
            from tkfmixdom.jax.dp.hmm import pair_hmm_emissions_constrained as _emit_jax
            log_emit = _emit_jax(
                jnp.asarray(st), jnp.asarray(sa_h),
                jnp.asarray(af_h), jnp.asarray(df_h),
                jnp.asarray(sub_matrices_t), jnp.asarray(np.array(dom_pis)),
                n_dom, n_frag)
        else:
            sub_matrix_t = transition_matrix(Q_lg, t_est)
            # Broadcast the shared single-Q to the n_dom-axis the JAX
            # builder expects.
            subs_dom = np.broadcast_to(
                np.asarray(sub_matrix_t)[None, :, :], (n_dom, AA, AA)).copy()
            pis_dom = np.broadcast_to(
                np.asarray(pi_lg)[None, :], (n_dom, AA)).copy()
            from tkfmixdom.jax.dp.hmm import pair_hmm_emissions_constrained as _emit_jax
            log_emit = _emit_jax(
                jnp.asarray(st), jnp.asarray(sa_h),
                jnp.asarray(af_h), jnp.asarray(df_h),
                jnp.asarray(subs_dom), jnp.asarray(pis_dom),
                n_dom, n_frag)

        # Pad emissions to geometric bin for JIT cache reuse
        L = len(states)
        padded_L = _pad_to_bin(L)
        if padded_L > L:
            padded_emit = jnp.full((padded_L, N), NEG_INF)
            padded_emit = padded_emit.at[:L].set(log_emit)
        else:
            padded_emit = log_emit

        # Padded 1D FB
        pair_ll, posteriors, n_chi = forward_backward_1d_padded(
            log_chi, padded_emit, seq_length=L, init_state=SS, final_state=EE)

        delta['agg_n_chi'] += np.asarray(n_chi)
        delta['total_ll'] += float(pair_ll)
        delta['n_pairs'] += 1

        # Accumulate substitution match counts (aggregate and per-domain)
        posteriors_np = np.asarray(posteriors[:L])

        # Build per-fragment state masks for classdist accumulation
        has_classdist = use_classdist and classdist_params is not None
        if has_classdist:
            cdp = classdist_params
            n_cls = cdp['classdist'].shape[2]
            # Pre-compute per-class likelihoods at this t for class posterior
            S_per_class = np.array(cdp['class_S_exch'])  # (C, A, A)
            class_P_t = np.zeros((n_cls, AA, AA))
            class_pi = np.array(cdp['class_pis'])
            for cc in range(n_cls):
                Q_c = S_per_class[cc] * class_pi[cc, None, :]
                np.fill_diagonal(Q_c, 0.0)
                Q_c[np.diag_indices(AA)] = -Q_c.sum(axis=1)
                class_P_t[cc] = np.array(transition_matrix(jnp.array(Q_c), t_est))
            cd = np.array(cdp['classdist'])  # (N, F, C)
            # Initialize per-class count arrays in delta
            if 'class_match_counts' not in delta:
                delta['class_match_counts'] = np.zeros((n_cls, AA, AA))
                delta['class_insert_counts'] = np.zeros((n_cls, AA))
                delta['class_delete_counts'] = np.zeros((n_cls, AA))
                delta['classdist_counts'] = np.zeros((n_dom, n_frag, n_cls))

            # Build frag state masks
            frag_m_masks = {}
            for dd in range(n_dom):
                for ff in range(n_frag):
                    mask = np.zeros(N, dtype=bool)
                    for s in range(2, N):
                        d_s = (s - 2) // (5 * n_frag)
                        f_s = ((s - 2) % (5 * n_frag)) % n_frag
                        if d_s == dd and f_s == ff and st[s] == M:
                            mask[s] = True
                    frag_m_masks[(dd, ff)] = mask

        ai, di = 0, 0
        for s_idx, state in enumerate(states):
            if state == M:
                a, d_char = int(anc_chars[ai]), int(desc_chars[di])
                m_post = float(posteriors_np[s_idx, m_mask].sum())
                delta['match_counts'][a, d_char] += m_post
                # Per-domain match counts (weighted by domain posterior)
                for dd in range(n_dom):
                    w = float(posteriors_np[s_idx, dom_m_masks[dd]].sum())
                    delta['dom_match_counts'][dd, a, d_char] += w

                # Per-class accumulation via class posterior
                if has_classdist:
                    for dd in range(n_dom):
                        for ff in range(n_frag):
                            frag_post = float(posteriors_np[s_idx, frag_m_masks[(dd, ff)]].sum())
                            if frag_post < 1e-30:
                                continue
                            # P(c | d,f,a,b) = classdist[d,f,c] * P_c(t)[a,b] / E
                            class_liks = cd[dd, ff, :] * class_P_t[:, a, d_char] * class_pi[:, a]
                            E = class_liks.sum()
                            if E < 1e-30:
                                continue
                            class_post = class_liks / E  # (C,)
                            for cc in range(n_cls):
                                w_c = frag_post * class_post[cc]
                                delta['class_match_counts'][cc, a, d_char] += w_c
                                delta['classdist_counts'][dd, ff, cc] += w_c

                ai += 1
                di += 1
            elif state == I:
                d_char = int(desc_chars[di])
                for dd in range(n_dom):
                    w = float(posteriors_np[s_idx, dom_i_masks[dd]].sum())
                    delta['dom_insert_counts'][dd, d_char] += w

                if has_classdist:
                    for dd in range(n_dom):
                        for ff in range(n_frag):
                            # Insert: similar class posterior
                            frag_mask_i = np.zeros(N, dtype=bool)
                            for s in range(2, N):
                                d_s = (s - 2) // (5 * n_frag)
                                f_s = ((s - 2) % (5 * n_frag)) % n_frag
                                if d_s == dd and f_s == ff and st[s] == I:
                                    frag_mask_i[s] = True
                            frag_post = float(posteriors_np[s_idx, frag_mask_i].sum())
                            if frag_post < 1e-30:
                                continue
                            class_liks = cd[dd, ff, :] * class_pi[:, d_char]
                            E = class_liks.sum()
                            if E < 1e-30:
                                continue
                            class_post = class_liks / E
                            for cc in range(n_cls):
                                w_c = frag_post * class_post[cc]
                                delta['class_insert_counts'][cc, d_char] += w_c
                                delta['classdist_counts'][dd, ff, cc] += w_c

                di += 1
            elif state == D:
                a = int(anc_chars[ai])
                for dd in range(n_dom):
                    w = float(posteriors_np[s_idx, dom_d_masks[dd]].sum())
                    delta['dom_delete_counts'][dd, a] += w

                if has_classdist:
                    for dd in range(n_dom):
                        for ff in range(n_frag):
                            # Delete: only ancestor observed, emission = pi_c(a).
                            # Class likelihood = classdist[d,f,c] * pi_c(a).
                            frag_mask_d = np.zeros(N, dtype=bool)
                            for s in range(2, N):
                                d_s = (s - 2) // (5 * n_frag)
                                f_s = ((s - 2) % (5 * n_frag)) % n_frag
                                if d_s == dd and f_s == ff and st[s] == D:
                                    frag_mask_d[s] = True
                            frag_post = float(posteriors_np[s_idx, frag_mask_d].sum())
                            if frag_post < 1e-30:
                                continue
                            class_liks = cd[dd, ff, :] * class_pi[:, a]
                            E = class_liks.sum()
                            if E < 1e-30:
                                continue
                            class_post = class_liks / E
                            for cc in range(n_cls):
                                w_c = frag_post * class_post[cc]
                                delta['class_delete_counts'][cc, a] += w_c
                                delta['classdist_counts'][dd, ff, cc] += w_c
                ai += 1

    if delta['n_pairs'] == 0:
        return None
    return delta


# ============================================================
# Batched E-step: vmap over pairs for GPU throughput
# ============================================================

# Module-level cache for the JIT'd vmapped FB function. Keyed by
# scan_mode so 'associative' and 'sequential' variants coexist without
# clobbering each other.
_batched_fb_jit_cache = {}
_batched_fwd_jit_cache = {}
_batched_fb_per_chi_jit_cache = {}
_batched_fwd_per_chi_jit_cache = {}
_log_chi_stack_jit_cache = None


# Phase 6: scan-mode resolution. The user-visible CLI flag
# `--fb-scan-mode {associative, sequential, auto}` plumbs through to
# every 1D FB JIT cache. `_resolve_fb_scan_mode` does the auto
# heuristic: pick `sequential` when n_states · padded_L_max is large
# enough that the associative prefix-product tensor (B · L · n²) would
# dominate the GPU memory budget.
def _resolve_fb_scan_mode(args, n_states, padded_L_max=None):
    """Resolve user `--fb-scan-mode` choice to a concrete mode.

    Args:
        args: argparse Namespace (must have `fb_scan_mode` attribute).
        n_states: total state count of the model (e.g. 5DF+2 for MixDom).
        padded_L_max: optional max padded sequence length seen in the
            current minibatch / family. If None, uses a conservative
            default (1024).

    Returns:
        'associative' or 'sequential'.

    Heuristic for `auto` (cf. forward_backward_1d_padded docstring):
      - associative scan footprint = O(B · L · n²) bytes (×8 for fp64).
      - sequential scan footprint  = O(L · n + n²) bytes per pair.
      - threshold n_states · padded_L_max ≥ 16384 picks `sequential`.
        Calibrated so d3f3 (ns=47, L=1024) → 47*1024 ≈ 48k → sequential,
        and TKF91 (ns=5, L=1024) → 5*1024 ≈ 5k → associative.
    """
    mode = getattr(args, 'fb_scan_mode', 'auto')
    if mode in ('associative', 'sequential'):
        return mode
    if mode != 'auto':
        raise ValueError(f"Unknown fb_scan_mode={mode!r}")
    L = padded_L_max if padded_L_max is not None else 1024
    if n_states * L >= 16384:
        return 'sequential'
    return 'associative'


def _get_batched_fb_jit(scan_mode='associative'):
    """Lazily create and cache a JIT'd vmapped forward_backward_1d_padded
    with a SHARED log_chi (DEPRECATED — use _get_batched_fb_per_chi_jit;
    the shared-chi path is incoherent with per-pair t_est emissions)."""
    if scan_mode not in _batched_fb_jit_cache:
        import jax
        from tkfmixdom.jax.dp.hmm import forward_backward_1d_padded

        def _fb_core(log_chi, emit_stack, lengths):
            return jax.vmap(
                lambda emit, sl: forward_backward_1d_padded(
                    log_chi, emit, seq_length=sl, init_state=0, final_state=1,
                    scan_mode=scan_mode),
                in_axes=(0, 0)
            )(emit_stack, lengths)
        _batched_fb_jit_cache[scan_mode] = jax.jit(_fb_core)
    return _batched_fb_jit_cache[scan_mode]


def _get_batched_fwd_jit(scan_mode='associative'):
    """Lazily create and cache a JIT'd vmapped forward-only function with a
    SHARED log_chi (DEPRECATED — see _get_batched_fwd_per_chi_jit)."""
    if scan_mode not in _batched_fwd_jit_cache:
        import jax
        from tkfmixdom.jax.dp.hmm import forward_backward_1d_padded

        def _fwd_core(log_chi, emit_stack, lengths):
            return jax.vmap(
                lambda emit, sl: forward_backward_1d_padded(
                    log_chi, emit, seq_length=sl, init_state=0, final_state=1,
                    forward_only=True, scan_mode=scan_mode),
                in_axes=(0, 0)
            )(emit_stack, lengths)
        _batched_fwd_jit_cache = jax.jit(_fwd_core)
    return _batched_fwd_jit_cache


def _get_batched_fb_per_chi_jit(scan_mode='associative'):
    """JIT'd vmapped FB with per-pair log_chi (B, N, N).

    Required for coherence with per-pair t_est: chi (TKF transitions) must
    be at the same t as emissions. See `_get_log_chi_stack_jit` for chi
    construction.

    See `_get_batched_fwd_per_chi_jit` for `scan_mode` semantics; the
    sequential mode trades parallel depth for the O(L·n²) intermediate
    that dominates the GPU budget at large ns.
    """
    if scan_mode not in _batched_fb_per_chi_jit_cache:
        import jax
        from tkfmixdom.jax.dp.hmm import forward_backward_1d_padded

        def _fb_core(log_chis, emit_stack, lengths):
            return jax.vmap(
                lambda chi, emit, sl: forward_backward_1d_padded(
                    chi, emit, seq_length=sl, init_state=0, final_state=1,
                    scan_mode=scan_mode),
                in_axes=(0, 0, 0)
            )(log_chis, emit_stack, lengths)
        _batched_fb_per_chi_jit_cache[scan_mode] = jax.jit(_fb_core)
    return _batched_fb_per_chi_jit_cache[scan_mode]


def _get_batched_fwd_per_chi_jit(scan_mode='associative'):
    """JIT'd vmapped forward-only with per-pair log_chi (B, N, N).

    `scan_mode='sequential'` selects the lax.scan FB path with O(L·n + n²)
    per-pair memory (instead of O(L·n²) for associative). At MixDom's
    d3f3 ns=47 with B-sized buckets reaching 1000+ pairs at L=1024, the
    associative variant materialises a multi-GB prefix-product tensor
    that OOMs at the val eval stage; the sequential variant fits.
    """
    if scan_mode not in _batched_fwd_per_chi_jit_cache:
        import jax
        from tkfmixdom.jax.dp.hmm import forward_backward_1d_padded

        def _fwd_core(log_chis, emit_stack, lengths):
            return jax.vmap(
                lambda chi, emit, sl: forward_backward_1d_padded(
                    chi, emit, seq_length=sl, init_state=0, final_state=1,
                    forward_only=True, scan_mode=scan_mode),
                in_axes=(0, 0, 0)
            )(log_chis, emit_stack, lengths)
        _batched_fwd_per_chi_jit_cache[scan_mode] = jax.jit(_fwd_core)
    return _batched_fwd_per_chi_jit_cache[scan_mode]


def _get_log_chi_stack_jit():
    """JIT'd vmapped builder for per-pair log_chi.

    Returns a function (t_arr, main_ins, main_del, dom_ins, dom_del,
    dom_weights, frag_weights, ext_rates) -> (B, N, N) log_chi stack,
    where each slice is `log(build_nested_trans(... t=t_arr[b] ...).chi)`.

    Couples with per-pair t_est emissions to give a coherent joint
    pair-HMM evaluation at each pair's actual evolutionary time.
    """
    global _log_chi_stack_jit_cache
    if _log_chi_stack_jit_cache is None:
        import jax
        import jax.numpy as jnp
        from tkfmixdom.jax.models.mixdom import build_nested_trans

        def _chi_at_t(t, main_ins, main_del, dom_ins, dom_del, dw, fw, ext):
            chi, _ = build_nested_trans(main_ins, main_del, t,
                                         dom_ins, dom_del, dw, fw, ext)
            # Use a moderate floor (1e-30 -> log = -69) rather than
            # safe_log's NEG_INF (-1e30). At extreme per-pair t_est, more
            # chi entries float-underflow to zero; mapping those to NEG_INF
            # triggers downstream logsumexp overflow (max-shift of -1e30
            # plus a valid term computes exp(huge) = +inf, which corrupts
            # the FB final log_prob to ~-1.6e27). A bounded log keeps the
            # FB stable at edge t while still rendering forbidden
            # transitions effectively-zero (e^-69 ≈ 1e-30).
            return jnp.log(jnp.maximum(chi, 1e-30))

        def _stack(t_arr, main_ins, main_del, dom_ins, dom_del, dw, fw, ext):
            return jax.vmap(_chi_at_t,
                            in_axes=(0, None, None, None, None, None, None, None))(
                t_arr, main_ins, main_del, dom_ins, dom_del, dw, fw, ext)

        _log_chi_stack_jit_cache = jax.jit(_stack)
    return _log_chi_stack_jit_cache


def _build_log_chi_stack(params, t_array):
    """Convenience wrapper: build per-pair log_chi stack from a params dict."""
    import jax.numpy as jnp
    fn = _get_log_chi_stack_jit()
    return fn(
        jnp.asarray(t_array),
        jnp.asarray(params['main_ins']),
        jnp.asarray(params['main_del']),
        jnp.asarray(params['dom_ins']),
        jnp.asarray(params['dom_del']),
        jnp.asarray(params['dom_weights']),
        jnp.asarray(params['frag_weights']),
        jnp.asarray(params['ext_rates']),
    )


# Module-level caches for per-chi 2D and match-aligned JITs (FB and forward-only)
_batched_fb_2d_per_chi_jit_cache = None
_batched_fb_2d_match_aligned_per_chi_jit_cache = None
_fwd_2d_per_chi_jit_cache = None
_fwd_ma_per_chi_jit_cache = None


def _get_batched_fb_2d_per_chi_jit():
    """JIT'd vmapped 2D FB + suff-stat reduction with per-pair log_chi
    and per-pair sub_matrices.

    Each pair's chi (TKF nested transitions) and substitution matrices are
    built at the pair's own t_est, so the joint pair HMM is coherent at one
    well-defined t per pair (per-pair-t_est analog of the 1D per-chi fix).

    Args of the returned JIT:
      log_chis            : (B, N, N) per-pair log chi
      st                  : (N,) state types
      xs                  : (B, Lx_pad)
      ys                  : (B, Ly_pad)
      real_Lxs, real_Lys  : (B,)
      sub_matrices_stack  : (B, n_dom, A, A) per-pair substitution matrices
      pis                 : (n_dom, A) shared (does not depend on t)
      n_dom, n_frag       : static
      dom_m_mask, dom_i_mask, dom_d_mask, m_mask_f : (N, n_dom) / (N,)
    """
    global _batched_fb_2d_per_chi_jit_cache
    if _batched_fb_2d_per_chi_jit_cache is None:
        import jax
        import jax.numpy as _jnp
        from tkfmixdom.jax.dp.hmm import (
            forward_backward_2d, pair_hmm_emissions_per_domain)

        def _fb_2d_core(log_chis, st, xs, ys, real_Lxs, real_Lys,
                        sub_matrices_stack, pis, n_dom, n_frag,
                        dom_m_mask, dom_i_mask, dom_d_mask, m_mask_f):
            jnp = _jnp

            def _single(log_chi, x, y, real_Lx, real_Ly, sub_matrices):
                A = sub_matrices.shape[-1]
                log_emit = pair_hmm_emissions_per_domain(
                    st, x, y, sub_matrices, pis, n_dom, n_frag)
                log_prob, posteriors, expected_trans = forward_backward_2d(
                    log_chi, st, x, y, None, None,
                    log_emit_table=log_emit, real_Lx=real_Lx, real_Ly=real_Ly)

                Lx_pad = x.shape[0]
                Ly_pad = y.shape[0]

                X_oh = jax.nn.one_hot(x, A)
                Y_oh = jax.nn.one_hot(y, A)

                post_MM = posteriors[1:Lx_pad+1, 1:Ly_pad+1, :]
                W_M = jnp.einsum('ijs,sd->dij', post_MM, dom_m_mask)
                tmp_M = jnp.einsum('dij,ia->daj', W_M, X_oh)
                dom_match = jnp.einsum('daj,jb->dab', tmp_M, Y_oh)

                post_I = posteriors[0:Lx_pad+1, 1:Ly_pad+1, :]
                I_per_j = jnp.einsum('ijs,sd->dj', post_I, dom_i_mask)
                dom_insert = jnp.einsum('dj,ja->da', I_per_j, Y_oh)

                post_D = posteriors[1:Lx_pad+1, 0:Ly_pad+1, :]
                D_per_i = jnp.einsum('ijs,sd->di', post_D, dom_d_mask)
                dom_delete = jnp.einsum('di,ia->da', D_per_i, X_oh)

                agg_match = dom_match.sum(axis=0)

                return log_prob, expected_trans, dom_match, dom_insert, dom_delete, agg_match

            return jax.vmap(_single, in_axes=(0, 0, 0, 0, 0, 0))(
                log_chis, xs, ys, real_Lxs, real_Lys, sub_matrices_stack)

        _batched_fb_2d_per_chi_jit_cache = jax.jit(
            _fb_2d_core, static_argnames=('n_dom', 'n_frag'))
    return _batched_fb_2d_per_chi_jit_cache


def _get_batched_fb_2d_match_aligned_per_chi_jit():
    """JIT'd vmapped match-aligned 2D FB + suff-stat reduction with per-pair
    log_chi and per-pair sub_matrices."""
    global _batched_fb_2d_match_aligned_per_chi_jit_cache
    if _batched_fb_2d_match_aligned_per_chi_jit_cache is None:
        import jax
        import jax.numpy as _jnp
        from tkfmixdom.jax.dp.hmm import (
            forward_backward_2d, pair_hmm_emissions_per_domain,
            mask_emissions_match_aligned,
        )

        def _fb_2d_match_core(log_chis, st, xs, ys, real_Lxs, real_Lys,
                              match_is, match_js, match_lens,
                              sub_matrices_stack, pis, n_dom, n_frag,
                              dom_m_mask, dom_i_mask, dom_d_mask, m_mask_f):
            jnp = _jnp

            def _single(log_chi, x, y, real_Lx, real_Ly,
                        mi, mj, mlen, sub_matrices):
                A = sub_matrices.shape[-1]
                log_emit = pair_hmm_emissions_per_domain(
                    st, x, y, sub_matrices, pis, n_dom, n_frag)
                mi_real = jnp.where(jnp.arange(mi.shape[0]) < mlen, mi, 0)
                mj_real = jnp.where(jnp.arange(mj.shape[0]) < mlen, mj, 0)
                log_emit = mask_emissions_match_aligned(
                    log_emit, st, mi_real, mj_real)

                log_prob, posteriors, expected_trans = forward_backward_2d(
                    log_chi, st, x, y, None, None,
                    log_emit_table=log_emit, real_Lx=real_Lx, real_Ly=real_Ly)

                Lx_pad = x.shape[0]
                Ly_pad = y.shape[0]

                X_oh = jax.nn.one_hot(x, A)
                Y_oh = jax.nn.one_hot(y, A)

                post_MM = posteriors[1:Lx_pad+1, 1:Ly_pad+1, :]
                W_M = jnp.einsum('ijs,sd->dij', post_MM, dom_m_mask)
                tmp_M = jnp.einsum('dij,ia->daj', W_M, X_oh)
                dom_match = jnp.einsum('daj,jb->dab', tmp_M, Y_oh)

                post_I = posteriors[0:Lx_pad+1, 1:Ly_pad+1, :]
                I_per_j = jnp.einsum('ijs,sd->dj', post_I, dom_i_mask)
                dom_insert = jnp.einsum('dj,ja->da', I_per_j, Y_oh)

                post_D = posteriors[1:Lx_pad+1, 0:Ly_pad+1, :]
                D_per_i = jnp.einsum('ijs,sd->di', post_D, dom_d_mask)
                dom_delete = jnp.einsum('di,ia->da', D_per_i, X_oh)

                agg_match = dom_match.sum(axis=0)

                return log_prob, expected_trans, dom_match, dom_insert, dom_delete, agg_match

            return jax.vmap(
                _single, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))(
                log_chis, xs, ys, real_Lxs, real_Lys,
                match_is, match_js, match_lens, sub_matrices_stack)

        _batched_fb_2d_match_aligned_per_chi_jit_cache = jax.jit(
            _fb_2d_match_core, static_argnames=('n_dom', 'n_frag'))
    return _batched_fb_2d_match_aligned_per_chi_jit_cache


def _get_fwd_2d_per_chi_jit():
    """JIT'd vmapped 2D forward-only with per-pair log_chi and per-pair
    sub_matrices (used by val eval, no posteriors)."""
    global _fwd_2d_per_chi_jit_cache
    if _fwd_2d_per_chi_jit_cache is None:
        import jax
        from tkfmixdom.jax.dp.hmm import (
            forward_backward_2d, pair_hmm_emissions_per_domain)

        def _fwd_2d_fn(log_chis, st_j, xs, ys, real_Lxs, real_Lys,
                       sub_matrices_stack, pis, n_dom, n_frag):
            def _single(log_chi, x, y, lx, ly, sub_matrices):
                log_emit = pair_hmm_emissions_per_domain(
                    st_j, x, y, sub_matrices, pis, n_dom, n_frag)
                return forward_backward_2d(
                    log_chi, st_j, x, y, None, None,
                    log_emit_table=log_emit, real_Lx=lx, real_Ly=ly,
                    forward_only=True)
            return jax.vmap(_single, in_axes=(0, 0, 0, 0, 0, 0))(
                log_chis, xs, ys, real_Lxs, real_Lys, sub_matrices_stack)

        _fwd_2d_per_chi_jit_cache = jax.jit(
            _fwd_2d_fn, static_argnames=('n_dom', 'n_frag'))
    return _fwd_2d_per_chi_jit_cache


def _get_fwd_ma_per_chi_jit():
    """JIT'd vmapped match-aligned 2D forward-only with per-pair log_chi and
    per-pair sub_matrices (used by val eval, no posteriors)."""
    global _fwd_ma_per_chi_jit_cache
    if _fwd_ma_per_chi_jit_cache is None:
        import jax
        import jax.numpy as _jnp
        from tkfmixdom.jax.dp.hmm import (
            forward_backward_2d, pair_hmm_emissions_per_domain,
            mask_emissions_match_aligned,
        )

        def _fwd_ma_fn(log_chis, st_j, xs, ys, real_Lxs, real_Lys,
                       match_is, match_js, match_lens,
                       sub_matrices_stack, pis, n_dom, n_frag):
            jnp = _jnp

            def _single(log_chi, x, y, lx, ly, mi, mj, mlen, sub_matrices):
                log_emit = pair_hmm_emissions_per_domain(
                    st_j, x, y, sub_matrices, pis, n_dom, n_frag)
                mi_real = jnp.where(jnp.arange(mi.shape[0]) < mlen, mi, 0)
                mj_real = jnp.where(jnp.arange(mj.shape[0]) < mlen, mj, 0)
                log_emit = mask_emissions_match_aligned(
                    log_emit, st_j, mi_real, mj_real)
                return forward_backward_2d(
                    log_chi, st_j, x, y, None, None,
                    log_emit_table=log_emit, real_Lx=lx, real_Ly=ly,
                    forward_only=True)
            return jax.vmap(
                _single, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))(
                log_chis, xs, ys, real_Lxs, real_Lys,
                match_is, match_js, match_lens, sub_matrices_stack)

        _fwd_ma_per_chi_jit_cache = jax.jit(
            _fwd_ma_fn, static_argnames=('n_dom', 'n_frag'))
    return _fwd_ma_per_chi_jit_cache


# Module-level JIT caches for vmapped per-pair CTMC ops.
# Hoisted out of per-pair Python loops to amortise CUDA-launch overhead;
# previously each batch dispatched B*n_dom (and B*n_classes) tiny eigh calls.
_per_pair_sub_matrices_jit_cache = None
_per_pair_hr_stats_jit_cache = None


def _get_per_pair_sub_matrices_jit():
    """Vmap `transition_matrix` over (B, n_dom).

    Inputs (passed via the wrapper):
      Qs:      (n_dom, A, A) per-domain rate matrices (frozen for batch).
      t_array: (B,) per-pair evolutionary times.

    Returns:
      sub_matrices_stack: (B, n_dom, A, A) — element [p, d] = exp(Q_d t_p).

    Per-pair-t coherence: pair p's slice uses exactly t_array[p]; no
    cross-pair averaging or representative-t shortcut.
    """
    global _per_pair_sub_matrices_jit_cache
    if _per_pair_sub_matrices_jit_cache is None:
        import jax
        from tkfmixdom.jax.core.ctmc import (
            transition_matrix as _tmp)
        # Inner vmap: over n_dom (Q axis 0, t shared).
        per_dom = jax.vmap(_tmp, in_axes=(0, None))
        # Outer vmap: over B (Q shared, t axis 0).
        per_pair_per_dom = jax.vmap(per_dom, in_axes=(None, 0))
        _per_pair_sub_matrices_jit_cache = jax.jit(per_pair_per_dom)
    return _per_pair_sub_matrices_jit_cache


def _get_per_pair_hr_stats_jit():
    """Vmap `holmes_rubin_weighted_stats` over (B, n_dom).

    Inputs (passed via the wrapper):
      Qs:       (n_dom, A, A) per-domain rate matrices (frozen for batch).
      pis:      (n_dom, A) per-domain stationary distributions (frozen).
      t_array:  (B,) per-pair evolutionary times.
      mc_stack: (B, n_dom, A, A) per-pair per-domain match counts.

    Returns:
      W_stack: (B, n_dom, A) — element [p, d, i] = Σ_{a,b} mc_p[d,a,b]
               · I_d^{ab}_{ii}(t_p) / M_d^{ab}(t_p).
      U_stack: (B, n_dom, A, A) — element [p, d, i, j] = Q_d[i,j]
               · Σ_{a,b} mc_p[d,a,b] · I_d^{ab}_{ij}(t_p) / M_d^{ab}(t_p).

    Per-pair-t coherence: each pair-domain slice uses that pair's own t_p
    and that domain's own (Q_d, pi_d) — never a shared / averaged t.
    `holmes_rubin_weighted_stats` is linear in mc, so pairs with all-zero
    mc contribute zero W/U and the prior `if mc_pd.sum() < 1e-12: continue`
    skip is mathematically redundant under vmap.
    """
    global _per_pair_hr_stats_jit_cache
    if _per_pair_hr_stats_jit_cache is None:
        import jax
        from tkfmixdom.jax.core.ctmc import (
            holmes_rubin_weighted_stats as _hrws)
        # Inner: over n_dom (Q axis 0, pi axis 0, t shared, mc axis 0).
        per_dom = jax.vmap(_hrws, in_axes=(0, 0, None, 0))
        # Outer: over B (Q shared, pi shared, t axis 0, mc axis 0).
        per_pair_per_dom = jax.vmap(per_dom, in_axes=(None, None, 0, 0))
        _per_pair_hr_stats_jit_cache = jax.jit(per_pair_per_dom)
    return _per_pair_hr_stats_jit_cache


def _build_per_pair_sub_matrices(t_array, params, Q_lg, pi_lg, n_dom):
    """Build per-pair (B, n_dom, A, A) substitution matrices and (n_dom, A) pis.

    Each pair's sub_matrices are evaluated at that pair's t_est via a single
    vmapped+JIT'd JAX call (over B and n_dom simultaneously). Replaces the
    earlier Python double-loop that dispatched B*n_dom tiny eigh calls — the
    bulk of host-side overhead in the E-step.

    Per-pair-t coherence: result[p, d] = exp(Q_d t_array[p]). Pair p's chi
    (built via `_build_log_chi_stack` at the same t_array[p]) and emissions
    (built from result[p]) thus live at one well-defined t per pair.

    When the caller does not provide per-domain (Q, pi), the LG fallback Q_lg
    and pi_lg are tiled to (n_dom, A, A) / (n_dom, A) before vmapping — so
    every domain shares the same M(t_p), matching the prior behavior of the
    `np.tile(sub_b[None], (n_dom, 1, 1))` LG branch.

    Returns (sub_matrices_stack, pis_arr) where:
      sub_matrices_stack: np.ndarray (B, n_dom, A, A)
      pis_arr           : np.ndarray (n_dom, A) — t-independent
    """
    import jax.numpy as jnp

    per_domain_subst = (params is not None
                        and params.get('dom_Qs') is not None
                        and params.get('dom_pis') is not None)
    if per_domain_subst:
        Qs = np.asarray(params['dom_Qs'])
        pis = np.asarray(params['dom_pis'])
        assert Qs.shape[0] == n_dom, (
            f"dom_Qs leading axis ({Qs.shape[0]}) must match n_dom ({n_dom})")
        assert pis.shape[0] == n_dom, (
            f"dom_pis leading axis ({pis.shape[0]}) must match n_dom ({n_dom})")
    else:
        # LG fallback: tile to (n_dom, A, A) so all domains share the same M(t).
        Qs = np.tile(np.asarray(Q_lg)[None], (n_dom, 1, 1))
        pis = np.tile(np.asarray(pi_lg)[None], (n_dom, 1))

    fn = _get_per_pair_sub_matrices_jit()
    # Preserve JAX's native dtype (float64 with JAX_ENABLE_X64). The prior
    # inline 1D path produced float64 log_subs; the M-step regression tests
    # check at 1e-12 tolerance, so a silent float32 downcast would break
    # them. The 2D/match-aligned paths previously stored float32 here but
    # internally re-uplift to float64 inside JIT'd FB; keeping float64
    # throughout costs one-time recompile but never regresses precision.
    sub_matrices_stack = np.asarray(
        fn(jnp.asarray(Qs), jnp.asarray(t_array)))
    pis_arr = np.asarray(pis)
    return sub_matrices_stack, pis_arr


def _build_per_pair_hr_stats(Qs, pis, t_array, mc_stack, chunk_size=128):
    """Vmapped per-pair-per-domain Holmes-Rubin weighted stats.

    Single vmapped+JIT'd call replaces the per-pair-per-domain Python loop
    over `holmes_rubin_weighted_stats(Q_d, pi_d, t_p, mc_p_d)` that dominated
    host-side overhead during accumulation.

    Per-pair-t coherence: for each pair p and domain d, the call uses that
    pair's own t_p (from t_array[p]) and that domain's own (Q_d, pi_d). No
    averaging, no consensus t, no shared mc — pure element-wise vmap.

    Memory: `holmes_rubin_integrals` materialises a (B, n_dom, A, A, A, A)
    tensor whose float64 footprint is B*n_dom*A^4*8 bytes (e.g.
    2000*3*160000*8 ≈ 7.7 GiB). To stay within GPU memory and avoid cuBLAS
    autotune failures on huge GEMMs, this wrapper splits B into chunks and
    concatenates results. Chunking is purely a memory-management technique
    — the per-pair-t computation is identical to a single all-at-once call
    (no precision change, no aggregation across pairs).

    Args:
        Qs:         np.ndarray (n_dom, A, A) per-domain rate matrices, frozen
                    across the batch (same Qs that drive FB emissions).
        pis:        np.ndarray (n_dom, A) per-domain stationary distributions.
        t_array:    np.ndarray (B,) per-pair t_p (each pair's own t_est).
        mc_stack:   np.ndarray (B, n_dom, A, A) per-pair per-domain match
                    counts (zero rows for empty match positions are fine —
                    HR is linear in mc, so HR(0) = 0).
        chunk_size: int, number of pairs per JAX call. 128 keeps the
                    intermediate tensor under ~250 MiB at A=20, n_dom=5.

    Returns:
        W_stack: np.ndarray (B, n_dom, A) per-pair per-domain dwell times.
        U_stack: np.ndarray (B, n_dom, A, A) per-pair per-domain transition
                 counts (off-diagonal).
    """
    import jax.numpy as jnp
    fn = _get_per_pair_hr_stats_jit()
    Qs_j = jnp.asarray(Qs)
    pis_j = jnp.asarray(pis)
    B = int(t_array.shape[0])
    if B == 0:
        n_dom = Qs.shape[0]
        A = Qs.shape[-1]
        return (np.zeros((0, n_dom, A)),
                np.zeros((0, n_dom, A, A)))
    W_chunks = []
    U_chunks = []
    for s in range(0, B, chunk_size):
        e = min(s + chunk_size, B)
        W_c, U_c = fn(
            Qs_j, pis_j,
            jnp.asarray(t_array[s:e]),
            jnp.asarray(mc_stack[s:e]))
        W_chunks.append(np.asarray(W_c))
        U_chunks.append(np.asarray(U_c))
    return np.concatenate(W_chunks, axis=0), np.concatenate(U_chunks, axis=0)


def _process_pairs_batched(batch_pairs, chi_params, st, Q_lg, pi_lg, N,
                           alignment_to_states,
                           dom_Qs=None, dom_pis=None, n_dom=1, n_frag=1,
                           classdist_params=None,
                           fb_scan_mode='associative'):
    """Process a batch of pre-decoded pairs with vmapped forward-backward.

    Per-pair t coherence: each pair's chi (TKF nested transitions) is built
    at the pair's own t_est, matching the per-pair t_est used to construct
    substitution emissions. Joint pair-HMM is thus evaluated at one
    well-defined t per pair.

    batch_pairs: list of (x_int, y_int, states, anc_chars, desc_chars, t_est)
        Pre-decoded pair tuples (precompiled format).
    chi_params: params dict with keys main_ins, main_del, dom_ins, dom_del,
        dom_weights, frag_weights, ext_rates — used to build per-pair log_chi
        via vmap inside this function. (Replaces the legacy `log_chi` arg
        which was a single (N, N) matrix at a global representative t — the
        global-chi path was incoherent with per-pair emissions.)

    Returns suff_stats delta dict (same format as _process_file_pairs), or None.
    """
    import jax
    import jax.numpy as jnp
    from tkfmixdom.jax.dp.hmm import NEG_INF
    from tkfmixdom.jax.core.ctmc import transition_matrix
    from tkfmixdom.jax.core.params import M, I, D

    per_domain_subst = dom_Qs is not None and dom_pis is not None

    st_np = np.array(st)
    m_mask = st_np == M
    i_mask = st_np == I
    d_mask = st_np == D

    # Build domain state mask matrices for vectorized posterior accumulation
    # dom_m_matrix[s, d] = 1.0 iff state s is a match state for domain d
    dom_m_matrix = np.zeros((N, n_dom))
    dom_i_matrix = np.zeros((N, n_dom))
    dom_d_matrix = np.zeros((N, n_dom))
    for dd in range(n_dom):
        for s in range(2, N):
            sd = (s - 2) // (5 * n_frag)
            if sd == dd:
                if st_np[s] == M:
                    dom_m_matrix[s, dd] = 1.0
                elif st_np[s] == I:
                    dom_i_matrix[s, dd] = 1.0
                elif st_np[s] == D:
                    dom_d_matrix[s, dd] = 1.0

    # Step 1: Build emissions for each pair and bucket by padded length.
    # Uses the JAX-native pair_hmm_emissions_constrained (no NumPy emit
    # path — single, JIT-friendly code path).
    from collections import defaultdict
    buckets = defaultdict(list)  # padded_L -> [(padded_emit_np, L_real, states_arr, anc_chars, desc_chars)]

    # Pre-compute per-pair per-domain substitution matrices in one batched
    # JAX call (vmap over B and n_dom). Each pair's slice
    # `sub_matrices_stack_all[p]` is M_d(t_p) at that pair's own t_p — no
    # representative-t shortcut, no averaging across pairs. Replaces the
    # B*n_dom Python-side `transition_matrix` dispatches that
    # dominated host overhead in the prior per-pair-loop construction.
    t_array_all = np.array(
        [item[5] for item in batch_pairs], dtype=np.float64)
    sub_params_for_pp = ({'dom_Qs': dom_Qs, 'dom_pis': dom_pis}
                         if per_domain_subst else {})
    # n_dom_for_subs: when per-domain substs are supplied use n_dom;
    # otherwise use 1 to preserve the prior LG-fallback shape (1, A, A) on
    # the emissions builder's input.
    n_dom_for_subs = n_dom if per_domain_subst else 1
    sub_matrices_stack_all, pis_subs_arr_linear = _build_per_pair_sub_matrices(
        t_array_all, sub_params_for_pp, Q_lg, pi_lg, n_dom_for_subs)
    # Linear shapes: (B, n_dom_for_subs, A, A) and (n_dom_for_subs, A).
    # Log forms are derived for downstream HR / NumPy-side accumulators
    # that still expect log-scale inputs; the JAX emit builder consumes
    # the linear forms directly.
    log_subs_stack_all = np.log(np.maximum(sub_matrices_stack_all, 1e-30))
    log_pis_subs_arr = np.log(np.maximum(pis_subs_arr_linear, 1e-30))

    # Pre-compute per-pair per-class substitution matrices ONCE per batch
    # (analog of the per-domain hoisting just above). Replaces the
    # B*n_classes Python-side `transition_matrix` dispatches that
    # dominated host overhead for class-aware runs (d3f1c3, d3f1c10,
    # d3f3c27). Each (pair, class) slice uses that pair's own t_p with the
    # class-specific (Q_c, pi_c) — no representative-t, no per-pair-class
    # averaging. The (Q_c, pi_c) themselves are t-independent, so they're
    # built once.
    use_classdist_outer = (classdist_params is not None
                           and 'classdist' in classdist_params)
    if use_classdist_outer:
        cdp_outer = classdist_params
        n_cls_outer = int(np.asarray(cdp_outer['classdist']).shape[2])
        S_per_class_outer = np.asarray(cdp_outer['class_S_exch'])  # (C, A, A)
        class_pis_outer = np.asarray(cdp_outer['class_pis'])       # (C, A)
        # Build per-class Q from (S_per_class, class_pis): t-independent.
        class_Qs_outer = np.zeros((n_cls_outer, AA, AA))
        for cc in range(n_cls_outer):
            Q_c = S_per_class_outer[cc] * class_pis_outer[cc, None, :]
            np.fill_diagonal(Q_c, 0.0)
            Q_c[np.diag_indices(AA)] = -Q_c.sum(axis=1)
            class_Qs_outer[cc] = Q_c
        cd_classdist_outer = np.asarray(cdp_outer['classdist'])  # (n_dom, n_frag, n_cls)
        # Vmapped per-pair per-class M_c(t_p): (B, n_cls, A, A).
        class_sub_params = {'dom_Qs': class_Qs_outer,
                            'dom_pis': class_pis_outer}
        class_sub_stack_all, _ = _build_per_pair_sub_matrices(
            t_array_all, class_sub_params, Q_lg, pi_lg, n_cls_outer)
        cd_log_subs_stack_all = np.log(np.maximum(class_sub_stack_all, 1e-300))
        cd_log_pis_shared = np.log(np.maximum(class_pis_outer, 1e-300))
    else:
        cdp_outer = None
        n_cls_outer = 0
        class_Qs_outer = None
        class_pis_outer = None
        cd_classdist_outer = None
        cd_log_subs_stack_all = None
        cd_log_pis_shared = None

    for ipi, item in enumerate(batch_pairs):
        x_int, y_int, states, anc_chars, desc_chars, t_est = item
        # Empty pairs (states=[]) are valid samples from the model
        # distribution (SS→EE direct transition, prob 1-κ at stationary).
        # FB on L=0 correctly yields pair_ll = log chi[SS, EE] and
        # n_chi[SS, EE] += 1, contributing (0, 0, 0, T=t) to BDI suff
        # stats. Discarding them conditioned the M-step on "non-empty
        # alignment" and biased λ̂ upward.

        L = len(states)
        states_arr = np.array(states, dtype=np.int32)

        # Build full-length character arrays for vectorized emission construction
        m_or_d = (states_arr == M) | (states_arr == D)
        m_or_i = (states_arr == M) | (states_arr == I)
        anc_full = np.zeros(L, dtype=np.int32)
        desc_full = np.zeros(L, dtype=np.int32)
        anc_full[m_or_d] = anc_chars
        desc_full[m_or_i] = desc_chars

        # Slice the pre-computed per-pair per-class sub matrices (built
        # outside this loop via the same vmapped helper as per-domain).
        # Each pair's M_c(t_p) is the pair's own t_p evaluated against the
        # frozen-for-batch class (Q_c, pi_c) — no per-pair JAX dispatch.
        use_classdist = use_classdist_outer
        if use_classdist:
            cd_log_subs = cd_log_subs_stack_all[ipi]   # (n_cls, A, A)
            cd_log_pis = cd_log_pis_shared             # (n_cls, A) — shared
            cd_classdist = cd_classdist_outer
            # Linear forms for the JAX emit builder.
            cd_subs_lin = class_sub_stack_all[ipi]     # (n_cls, A, A)
            cd_pis_lin = class_pis_outer               # (n_cls, A)
            cd_classdist_lin = cd_classdist_outer
        else:
            cd_log_subs = None
            cd_log_pis = None
            cd_classdist = None
            cd_subs_lin = None
            cd_pis_lin = None
            cd_classdist_lin = None

        # Slice the pre-computed per-pair sub matrices for this pair.
        # Shape: (n_dom_for_subs, A, A); pis shared across pairs.
        log_subs = log_subs_stack_all[ipi]
        log_pis_arr = log_pis_subs_arr
        # Linear forms for the JAX emit builder. When per_domain_subst is
        # off, n_dom_for_subs=1 and we broadcast the shared (1, A, A) sub
        # to (n_dom, A, A) so the JAX builder's dom_idx lookup is safe.
        if per_domain_subst:
            subs_lin_pair = sub_matrices_stack_all[ipi]    # (n_dom, A, A)
            pis_lin_pair = pis_subs_arr_linear             # (n_dom, A)
        else:
            subs_lin_pair = np.broadcast_to(
                sub_matrices_stack_all[ipi][0], (n_dom, AA, AA))
            pis_lin_pair = np.broadcast_to(
                pis_subs_arr_linear[0], (n_dom, AA))

        # JAX-native emission construction (replaces the NumPy-only
        # `mixdom_constrained_emissions_vectorized`). Output is a JAX
        # array; converted back to NumPy for the existing NumPy-padded
        # bucket layout below. Eliminates the NumPy code path inside
        # the SVI-BW E-step and is JIT-friendly for downstream
        # restructuring (vmap over bucket pairs).
        from tkfmixdom.jax.dp.hmm import pair_hmm_emissions_constrained as _emit_jax
        log_emit_j = _emit_jax(
            jnp.asarray(st), jnp.asarray(states_arr),
            jnp.asarray(anc_full), jnp.asarray(desc_full),
            jnp.asarray(subs_lin_pair), jnp.asarray(pis_lin_pair),
            n_dom, n_frag,
            classdist=jnp.asarray(cd_classdist_lin) if cd_classdist_lin is not None else None,
            class_sub_matrices=jnp.asarray(cd_subs_lin) if cd_subs_lin is not None else None,
            class_pis=jnp.asarray(cd_pis_lin) if cd_pis_lin is not None else None)
        log_emit = np.asarray(log_emit_j)

        padded_L = _pad_to_bin(L)
        padded_emit_np = np.full((padded_L, N), np.float32(NEG_INF), dtype=np.float32)
        padded_emit_np[:L] = log_emit
        # Store per-pair classdist info for posterior computation
        pair_cd_info = (cd_log_subs, cd_log_pis) if use_classdist else None
        # Store t_est too so we can build per-pair log_chi inside the
        # bucket loop (chi must be at the same t as emissions).
        buckets[padded_L].append((padded_emit_np, L, states_arr, anc_chars, desc_chars, pair_cd_info, t_est))

    if not buckets:
        return None

    # Initialize delta
    delta = {
        'agg_n_chi': np.zeros((N, N)),
        'total_ll': 0.0,
        'n_pairs': 0,
        'n_files': 1,
        'match_counts': np.zeros((AA, AA)),
        'dom_match_counts': np.zeros((n_dom, AA, AA)),
        'dom_insert_counts': np.zeros((n_dom, AA)),
        'dom_delete_counts': np.zeros((n_dom, AA)),
        # Per-pair-t Holmes-Rubin sufficient stats: dom_W[d, A] and
        # dom_U[d, A, A] are accumulated PER PAIR using each pair's own
        # t_p with the per-domain Q_d, pi_d (from current params, or LG
        # fallback). These are time-free aggregated stats consumed
        # directly by the M-step (no per-(a, b) HR loop at t_rep).
        'dom_W': np.zeros((n_dom, AA)),
        'dom_U': np.zeros((n_dom, AA, AA)),
    }
    # Eagerly initialise per-class accumulators when classdist is provided.
    if (classdist_params is not None
            and 'classdist' in classdist_params):
        n_cls_init = int(np.asarray(classdist_params['classdist']).shape[2])
        delta['class_match_counts'] = np.zeros((n_cls_init, AA, AA))
        delta['class_insert_counts'] = np.zeros((n_cls_init, AA))
        delta['class_delete_counts'] = np.zeros((n_cls_init, AA))
        delta['classdist_counts'] = np.zeros((n_dom, n_frag, n_cls_init))
        delta['class_W'] = np.zeros((n_cls_init, AA))
        delta['class_U'] = np.zeros((n_cls_init, AA, AA))
    # Per-pair (n_chi, t) records: we run `exact_suffstats` PER PAIR at each
    # pair's own t_p (instead of at a single t_rep on the aggregated n_chi),
    # then sum the resulting BDI tensors. This preserves additivity of the
    # count-conversion across pairs with different t.
    _pair_n_chi_t_records = []

    # Per-pair (mc_p_d, t_p) records for batched per-pair-t Holmes-Rubin.
    # mc_p_d is the per-domain match-count tensor at this pair's own t_p;
    # zero-mc pairs (e.g. pairs with n_match=0) contribute mc_p_d=0 here
    # and HR(0, ...) = 0, so the prior `if mc.sum() < 1e-12: continue` skip
    # is mathematically redundant under the vmapped HR call.
    _pair_mc_d_records = []   # list of (n_dom, A, A) match-count tensors
    _pair_t_records = []      # list of float t_p values, parallel to _pair_mc_d_records

    # Same pattern for per-class match counts cm_p (class-aware runs).
    # When `use_classdist` is False, this list stays empty and the post-
    # loop class-HR call is a no-op.
    _pair_cm_records = []     # list of (n_cls, A, A) class match-count tensors

    # Per-domain Q, pi at E-step time (frozen across this batch). Used for
    # per-pair-t Holmes-Rubin W/U accumulation. Falls back to LG when the
    # caller hasn't supplied dom_Qs / dom_pis (fresh init or estimate-subst
    # disabled).
    if dom_Qs is not None and dom_pis is not None:
        _hr_Qs = np.asarray(dom_Qs)        # (n_dom, A, A)
        _hr_pis = np.asarray(dom_pis)      # (n_dom, A)
    else:
        _hr_Qs = np.tile(np.asarray(Q_lg)[None], (n_dom, 1, 1))
        _hr_pis = np.tile(np.asarray(pi_lg)[None], (n_dom, 1))

    batched_fb_per_chi = _get_batched_fb_per_chi_jit(scan_mode=fb_scan_mode)
    eye_AA = np.eye(AA)

    # Step 2: For each bucket, build per-pair log_chi at each pair's t_est
    # (so chi and emissions live at the same t — coherent joint pair HMM)
    # then vmap the FB and accumulate stats.
    for padded_L, bucket in buckets.items():
        B = len(bucket)
        emit_stack = jnp.array(np.stack([item[0] for item in bucket]))  # (B, padded_L, N)
        lengths = jnp.array([item[1] for item in bucket], dtype=jnp.int64)  # (B,)
        t_ests = jnp.array([item[6] for item in bucket], dtype=jnp.float32)  # (B,)

        # Per-pair log_chi: (B, N, N), each slice at this pair's t_est.
        log_chi_stack = _build_log_chi_stack(chi_params, t_ests)

        # Vmapped forward-backward: log_probs (B,), posteriors (B, padded_L, N),
        # n_chis (B, N, N).
        log_probs, posteriors_batch, n_chis_batch = batched_fb_per_chi(
            log_chi_stack, emit_stack, lengths)

        # Transfer to NumPy for float64 accumulation (avoids float32 sum drift)
        posteriors_np = np.asarray(posteriors_batch)
        n_chis_np = np.asarray(n_chis_batch)
        log_probs_np = np.asarray(log_probs)

        # Step 3: Per-pair accumulation in float64
        t_ests_np = np.asarray(t_ests)
        for i in range(B):
            delta['agg_n_chi'] += n_chis_np[i]
            # Track per-pair (n_chi, t) for downstream per-pair exact_suffstats.
            _pair_n_chi_t_records.append(
                (n_chis_np[i].astype(np.float64).copy(), float(t_ests_np[i])))
            delta['total_ll'] += float(log_probs_np[i])
            delta['n_pairs'] += 1

            _, L_real, states_arr, anc_chars, desc_chars, pair_cd_info_i, _ = bucket[i]
            post_i = posteriors_np[i, :L_real]  # (L_real, N)

            m_pos = states_arr == M
            i_pos = states_arr == I
            d_pos = states_arr == D
            m_or_d = m_pos | d_pos
            m_or_i = m_pos | i_pos

            # Build full-length character arrays indexed by position
            anc_full = np.zeros(L_real, dtype=int)
            desc_full = np.zeros(L_real, dtype=int)
            anc_full[m_or_d] = anc_chars
            desc_full[m_or_i] = desc_chars

            # Match: aggregate and per-domain counts via einsum
            n_match = m_pos.sum()
            t_p_pair = float(t_ests_np[i])
            # Per-pair per-domain match counts (zero-init; filled below if
            # n_match > 0). Stashed for the post-loop batched HR call so we
            # use this pair's own t_p with the per-domain (Q_d, pi_d) frozen
            # across the iteration (no representative-t shortcut, no shared
            # mc across pairs).
            mc_p_d = np.zeros((n_dom, AA, AA), dtype=np.float64)
            if n_match > 0:
                post_m = post_i[m_pos]  # (n_match, N)
                anc_m = anc_full[m_pos]
                desc_m = desc_full[m_pos]
                anc_oh = eye_AA[anc_m]   # (n_match, AA)
                desc_oh = eye_AA[desc_m]  # (n_match, AA)

                agg_m_post = post_m[:, m_mask].sum(axis=1)  # (n_match,)
                delta['match_counts'] += np.einsum('m,ma,mb->ab', agg_m_post, anc_oh, desc_oh)

                dom_m_post = post_m @ dom_m_matrix  # (n_match, n_dom)
                # Per-pair per-domain match counts mc_p[d, a, b].
                mc_p_d = np.einsum('md,ma,mb->dab', dom_m_post, anc_oh, desc_oh)
                delta['dom_match_counts'] += mc_p_d

            # Stash per-pair (mc_p_d, t_p) for the batched HR call. Pairs
            # with n_match=0 still get pushed (zero mc); HR(0)=0 so they
            # contribute nothing to dom_W / dom_U.
            _pair_mc_d_records.append(mc_p_d)
            _pair_t_records.append(t_p_pair)

            # Insert: per-domain counts via einsum
            n_ins = i_pos.sum()
            if n_ins > 0:
                post_ins = post_i[i_pos]  # (n_ins, N)
                desc_ins_oh = eye_AA[desc_full[i_pos]]  # (n_ins, AA)
                dom_i_post = post_ins @ dom_i_matrix  # (n_ins, n_dom)
                delta['dom_insert_counts'] += np.einsum('md,ma->da', dom_i_post, desc_ins_oh)

            # Delete: per-domain counts via einsum
            n_del = d_pos.sum()
            if n_del > 0:
                post_del = post_i[d_pos]  # (n_del, N)
                anc_del_oh = eye_AA[anc_full[d_pos]]  # (n_del, AA)
                dom_d_post = post_del @ dom_d_matrix  # (n_del, n_dom)
                delta['dom_delete_counts'] += np.einsum('md,ma->da', dom_d_post, anc_del_oh)

            # Per-fragment-per-class counts (if classdist active).
            # Per-pair-t coherence: cd_log_subs_i is THIS pair's M_c(t_p)
            # log-table (sliced from the precomputed B*n_cls stack); same
            # t_p as the rest of this pair's stats. Vectorised over the
            # match position axis to replace the prior triple-nested
            # Python loop (positions × n_dom × n_frag), with a final
            # logsumexp over n_cls for the class posterior. cm_p is the
            # per-pair per-class match-count tensor stashed for the
            # post-loop batched HR call (analog of the per-domain mc_p_d
            # path); zero-mc pairs contribute zero W/U so the prior
            # `if cm.sum() < 1e-12: continue` skip is redundant under
            # vmap.
            pair_cd_log_subs_i = pair_cd_info_i[0] if pair_cd_info_i else None
            pair_cd_log_pis_i = pair_cd_info_i[1] if pair_cd_info_i else None
            cm_p_for_hr = np.zeros((n_cls_outer, AA, AA), dtype=np.float64)
            if use_classdist and n_match > 0 and pair_cd_log_subs_i is not None:
                cd = cd_classdist_outer  # (n_dom, n_frag, n_cls)
                log_cd = np.log(np.maximum(cd, 1e-300))  # (n_dom, n_frag, n_cls)

                # Build frag-level posterior matrix once per pair:
                # frag_m_matrix[s, d*F+f] = 1 iff state s is M for (d,f).
                frag_m_matrix = np.zeros((N, n_dom * n_frag))
                for s in range(2, N):
                    d_s = (s - 2) // (5 * n_frag)
                    f_s = ((s - 2) % (5 * n_frag)) % n_frag
                    if st_np[s] == M:
                        frag_m_matrix[s, d_s * n_frag + f_s] = 1.0
                frag_m_post = post_m @ frag_m_matrix  # (n_match, n_dom*n_frag)
                fp_resh = frag_m_post.reshape(n_match, n_dom, n_frag)

                # Per-position class-likelihood log-terms:
                #   log_obs[t, c] = log_subs_p[c, anc[t], desc[t]]
                #                 + log_pis_p[c, anc[t]]
                # (t-coherent: log_subs_p / log_pis_p use this pair's t_p.)
                log_obs = (pair_cd_log_subs_i[:, anc_m, desc_m].T  # (n_match, n_cls)
                           + pair_cd_log_pis_i[:, anc_m].T)
                # log_class_liks[t, d, f, c] = log_cd[d, f, c] + log_obs[t, c]
                log_class_liks = (log_cd[None, :, :, :]
                                  + log_obs[:, None, None, :])
                # Stable logsumexp over the class axis for the posterior denom.
                lcl_max = log_class_liks.max(axis=-1, keepdims=True)
                log_E = lcl_max[..., 0] + np.log(
                    np.sum(np.exp(log_class_liks - lcl_max), axis=-1))
                cp = np.exp(log_class_liks - log_E[..., None])
                # Mask out (t, d, f) tuples where E ≈ 0 (matches prior
                # `if E < 1e-30: continue` skip — sets per-class posterior
                # contribution to zero, NOT to a fallback uniform).
                e_skip = (log_E < np.log(1e-30))[..., None]
                cp = np.where(e_skip, 0.0, cp)
                # Mask out (t, d, f) tuples where the per-frag posterior
                # weight is essentially zero (matches prior `if fp < 1e-30:
                # continue`). This lets the einsum ignore numerical noise
                # at unselected fragments without averaging them in.
                fp_skip = (fp_resh < 1e-30)[..., None]
                w = np.where(fp_skip, 0.0, fp_resh[..., None] * cp)
                # w has shape (n_match, n_dom, n_frag, n_cls).

                # Accumulators (sum w into the right output tensor).
                # delta['class_match_counts'][c, a, b]
                #     += Σ_{t, d, f} w[t, d, f, c] · 1[anc[t]=a, desc[t]=b]
                anc_oh_m = eye_AA[anc_m]    # (n_match, AA)
                desc_oh_m = eye_AA[desc_m]
                w_dom_frag = w.sum(axis=(1, 2))  # (n_match, n_cls)
                class_match_contrib = np.einsum(
                    'tc,ta,tb->cab', w_dom_frag, anc_oh_m, desc_oh_m)
                delta['class_match_counts'] += class_match_contrib
                # delta['classdist_counts'][d, f, c] += Σ_t w[t, d, f, c]
                delta['classdist_counts'] += w.sum(axis=0)
                # cm_p (this pair) = same tensor as class_match_contrib —
                # stashed for the post-loop batched HR call.
                cm_p_for_hr = class_match_contrib

            _pair_cm_records.append(cm_p_for_hr)

            # Per-fragment-per-class insert counts (vectorised same way).
            # At I positions only the descendant is observed: class likelihood
            # = classdist[d, f, c] * pi_c(b). No HR contribution.
            if use_classdist and n_ins > 0 and pair_cd_log_pis_i is not None:
                cd = cd_classdist_outer
                log_cd = np.log(np.maximum(cd, 1e-300))

                frag_i_matrix = np.zeros((N, n_dom * n_frag))
                for s in range(2, N):
                    d_s = (s - 2) // (5 * n_frag)
                    f_s = ((s - 2) % (5 * n_frag)) % n_frag
                    if st_np[s] == I:
                        frag_i_matrix[s, d_s * n_frag + f_s] = 1.0
                post_ins_i = post_i[i_pos]
                frag_i_post = post_ins_i @ frag_i_matrix
                fp_ins = frag_i_post.reshape(n_ins, n_dom, n_frag)
                desc_ins = desc_full[i_pos]

                log_obs_ins = pair_cd_log_pis_i[:, desc_ins].T  # (n_ins, n_cls)
                log_class_liks_ins = (log_cd[None, :, :, :]
                                      + log_obs_ins[:, None, None, :])
                lcl_max_ins = log_class_liks_ins.max(axis=-1, keepdims=True)
                log_E_ins = lcl_max_ins[..., 0] + np.log(
                    np.sum(np.exp(log_class_liks_ins - lcl_max_ins), axis=-1))
                cp_ins = np.exp(log_class_liks_ins - log_E_ins[..., None])
                cp_ins = np.where(
                    (log_E_ins < np.log(1e-30))[..., None], 0.0, cp_ins)
                w_ins = np.where(
                    (fp_ins < 1e-30)[..., None],
                    0.0, fp_ins[..., None] * cp_ins)
                w_ins_df = w_ins.sum(axis=(1, 2))   # (n_ins, n_cls)
                desc_oh_ins = eye_AA[desc_ins]
                delta['class_insert_counts'] += np.einsum(
                    'tc,tb->cb', w_ins_df, desc_oh_ins)
                delta['classdist_counts'] += w_ins.sum(axis=0)

            # Per-fragment-per-class delete counts (mirror of insert: only
            # the ancestor is observed, so class lik = classdist[d,f,c] · pi_c(a)).
            if use_classdist and n_del > 0 and pair_cd_log_pis_i is not None:
                cd = cd_classdist_outer
                log_cd = np.log(np.maximum(cd, 1e-300))

                frag_d_matrix = np.zeros((N, n_dom * n_frag))
                for s in range(2, N):
                    d_s = (s - 2) // (5 * n_frag)
                    f_s = ((s - 2) % (5 * n_frag)) % n_frag
                    if st_np[s] == D:
                        frag_d_matrix[s, d_s * n_frag + f_s] = 1.0
                post_del_d = post_i[d_pos]
                frag_d_post = post_del_d @ frag_d_matrix
                fp_del = frag_d_post.reshape(n_del, n_dom, n_frag)
                anc_del = anc_full[d_pos]

                log_obs_del = pair_cd_log_pis_i[:, anc_del].T
                log_class_liks_del = (log_cd[None, :, :, :]
                                      + log_obs_del[:, None, None, :])
                lcl_max_del = log_class_liks_del.max(axis=-1, keepdims=True)
                log_E_del = lcl_max_del[..., 0] + np.log(
                    np.sum(np.exp(log_class_liks_del - lcl_max_del), axis=-1))
                cp_del = np.exp(log_class_liks_del - log_E_del[..., None])
                cp_del = np.where(
                    (log_E_del < np.log(1e-30))[..., None], 0.0, cp_del)
                w_del = np.where(
                    (fp_del < 1e-30)[..., None],
                    0.0, fp_del[..., None] * cp_del)
                w_del_df = w_del.sum(axis=(1, 2))   # (n_del, n_cls)
                anc_oh_del = eye_AA[anc_del]
                delta['class_delete_counts'] += np.einsum(
                    'tc,ta->ca', w_del_df, anc_oh_del)
                delta['classdist_counts'] += w_del.sum(axis=0)

    if delta['n_pairs'] == 0:
        return None

    # Batched per-pair-t Holmes-Rubin: single vmapped JAX call replaces the
    # B*n_dom Python loop over `holmes_rubin_weighted_stats`. Each pair-domain
    # slice uses that pair's own t_p and the (frozen-for-batch) per-domain
    # (Q_d, pi_d), so per-pair-t coherence with the FB pass is preserved.
    if _pair_mc_d_records:
        mc_stack = np.stack(_pair_mc_d_records)        # (B_proc, n_dom, A, A)
        t_stack = np.array(_pair_t_records, dtype=np.float64)  # (B_proc,)
        W_stack, U_stack = _build_per_pair_hr_stats(
            _hr_Qs, _hr_pis, t_stack, mc_stack)
        delta['dom_W'] += W_stack.sum(axis=0)  # sum over pairs → (n_dom, A)
        delta['dom_U'] += U_stack.sum(axis=0)  # sum over pairs → (n_dom, A, A)

    # Same batched per-pair-t HR pattern, this time over (B, n_cls) for
    # the per-class match counts. (Q_c, pi_c) are the SAME class params
    # that drove the per-class FB emissions for this batch — frozen across
    # the iteration, so per-pair-t coherence is preserved bit-for-bit.
    # `_pair_cm_records` is empty when use_classdist is False, in which
    # case this block is a no-op and `delta['class_W'] / class_U` stay at
    # their (eagerly-initialised) zeros.
    if use_classdist_outer and _pair_cm_records:
        cm_stack = np.stack(_pair_cm_records)          # (B_proc, n_cls, A, A)
        t_stack_for_cls = np.array(_pair_t_records, dtype=np.float64)
        # Length sanity check — must be one cm_p record per processed pair,
        # in the same order as t_records (they're populated in the same
        # inner loop iteration).
        assert cm_stack.shape[0] == t_stack_for_cls.shape[0], (
            f"cm_records ({cm_stack.shape[0]}) and t_records "
            f"({t_stack_for_cls.shape[0]}) length mismatch")
        W_cls_stack, U_cls_stack = _build_per_pair_hr_stats(
            class_Qs_outer, class_pis_outer, t_stack_for_cls, cm_stack)
        delta['class_W'] += W_cls_stack.sum(axis=0)
        delta['class_U'] += U_cls_stack.sum(axis=0)

    # Per-pair exact_suffstats: convert each pair's n_chi into BDI tensors at
    # its own t, then sum. The driver consumes these pre-converted tensors
    # directly, skipping the t_rep-based aggregate exact_suffstats call.
    if _pair_n_chi_t_records:
        from tkfmixdom.jax.models.exact_suffstats import (
            exact_suffstats_per_pair_batch)
        delta['exact_ss'] = exact_suffstats_per_pair_batch(
            _pair_n_chi_t_records,
            chi_params['main_ins'], chi_params['main_del'],
            np.asarray(chi_params['dom_ins']),
            np.asarray(chi_params['dom_del']),
            np.asarray(chi_params['dom_weights']),
            np.asarray(chi_params['frag_weights']),
            np.asarray(chi_params['ext_rates']))
    return delta


# ============================================================
# Batched 2D E-step: unaligned (full pair HMM) via vmap
# ============================================================

# Module-level cache for JIT'd vmapped 2D FB
_batched_fb_2d_jit_cache = None


def _get_batched_fb_2d_jit():
    """Lazily create and cache a JIT'd vmapped 2D E-step kernel."""
    global _batched_fb_2d_jit_cache
    if _batched_fb_2d_jit_cache is None:
        import jax
        import jax.numpy as _jnp
        from tkfmixdom.jax.dp.hmm import forward_backward_2d, pair_hmm_emissions_per_domain

        def _fb_2d_core(log_chi, st, xs, ys, real_Lxs, real_Lys,
                        sub_matrices, pis, n_dom, n_frag,
                        dom_m_mask, dom_i_mask, dom_d_mask, m_mask_f):
            """Single-pair 2D FB + suff stat reduction (vmapped over batch)."""
            jnp = _jnp  # capture in closure for vmap'd function

            def _single(x, y, real_Lx, real_Ly):
                A = sub_matrices.shape[-1]
                log_emit = pair_hmm_emissions_per_domain(
                    st, x, y, sub_matrices, pis, n_dom, n_frag)
                log_prob, posteriors, expected_trans = forward_backward_2d(
                    log_chi, st, x, y, None, None,
                    log_emit_table=log_emit, real_Lx=real_Lx, real_Ly=real_Ly)

                Lx_pad = x.shape[0]
                Ly_pad = y.shape[0]

                # One-hot encode sequences
                X_oh = jax.nn.one_hot(x, A)  # (Lx_pad, A)
                Y_oh = jax.nn.one_hot(y, A)  # (Ly_pad, A)

                # Match counts: (n_dom, A, A)
                # Three-step einsum to avoid cuBLAS autotuner issues under vmap
                post_MM = posteriors[1:Lx_pad+1, 1:Ly_pad+1, :]  # (Lx_pad, Ly_pad, ns)
                W_M = jnp.einsum('ijs,sd->dij', post_MM, dom_m_mask)  # (n_dom, Lx_pad, Ly_pad)
                tmp_M = jnp.einsum('dij,ia->daj', W_M, X_oh)  # (n_dom, A, Ly_pad)
                dom_match = jnp.einsum('daj,jb->dab', tmp_M, Y_oh)  # (n_dom, A, A)

                # Insert counts: (n_dom, A)
                post_I = posteriors[0:Lx_pad+1, 1:Ly_pad+1, :]  # (Lx_pad+1, Ly_pad, ns)
                I_per_j = jnp.einsum('ijs,sd->dj', post_I, dom_i_mask)  # (n_dom, Ly_pad)
                dom_insert = jnp.einsum('dj,ja->da', I_per_j, Y_oh)  # (n_dom, A)

                # Delete counts: (n_dom, A)
                post_D = posteriors[1:Lx_pad+1, 0:Ly_pad+1, :]  # (Lx_pad, Ly_pad+1, ns)
                D_per_i = jnp.einsum('ijs,sd->di', post_D, dom_d_mask)  # (n_dom, Lx_pad)
                dom_delete = jnp.einsum('di,ia->da', D_per_i, X_oh)  # (n_dom, A)

                # Aggregate match counts: sum over domains
                agg_match = dom_match.sum(axis=0)  # (A, A)

                return log_prob, expected_trans, dom_match, dom_insert, dom_delete, agg_match

            return jax.vmap(_single, in_axes=(0, 0, 0, 0))(xs, ys, real_Lxs, real_Lys)

        _batched_fb_2d_jit_cache = jax.jit(_fb_2d_core,
                                            static_argnames=('n_dom', 'n_frag'))
    return _batched_fb_2d_jit_cache


def _process_pairs_batched_2d(batch_pairs, chi_params, st, Q_lg, pi_lg, N,
                               dom_Qs=None, dom_pis=None, n_dom=1, n_frag=1,
                               params=None):
    """Process a batch of pairs with vmapped 2D forward-backward (unaligned).

    Instead of using the MSA alignment to constrain the DP, runs the full
    2D pair HMM Forward-Backward on raw sequences and extracts sufficient
    statistics from the 2D posteriors.

    Per-pair t coherence: chi (TKF nested transitions) and substitution
    matrices are both built at each pair's own t_est, so the joint pair HMM
    is coherent at one well-defined t per pair (analog of the 1D per-chi
    fix in `_process_pairs_batched`).

    batch_pairs: list of (x_int, y_int, states, anc_chars, desc_chars, t_est)
        Pre-decoded pair tuples. Only x_int, y_int, and t_est are used;
        states/anc_chars/desc_chars are ignored (no alignment constraint).
    chi_params: params dict with keys main_ins, main_del, dom_ins, dom_del,
        dom_weights, frag_weights, ext_rates — used to build per-pair log_chi
        via vmap. (Replaces the legacy `log_chi` arg which was a single
        (N, N) matrix at a global representative t — incoherent with the
        per-pair-t emissions.)

    Returns suff_stats delta dict (same format as _process_file_pairs), or None.
    """
    import jax
    import jax.numpy as jnp
    from tkfmixdom.jax.dp.hmm import NEG_INF
    from tkfmixdom.jax.core.params import M, I, D
    from tkfmixdom.jax.tree.fsa_anneal import build_per_domain_sub_matrices
    from tkfmixdom.jax.models.mixdom import build_nested_trans

    if not batch_pairs:
        return None

    st_np = np.array(st)
    ns = len(st_np)
    is_M = st_np == M
    is_I = st_np == I
    is_D = st_np == D

    # Domain masks for einsum reduction
    body_idx = np.maximum((np.arange(ns) - 2) // (5 * n_frag), 0)
    dom_m_mask = np.zeros((ns, n_dom))  # (ns, n_dom)
    dom_i_mask = np.zeros((ns, n_dom))
    dom_d_mask = np.zeros((ns, n_dom))
    m_mask_f = np.zeros(ns)
    for s in range(2, ns):
        dd = body_idx[s]
        if st_np[s] == M:
            dom_m_mask[s, dd] = 1.0
            m_mask_f[s] = 1.0
        elif st_np[s] == I:
            dom_i_mask[s, dd] = 1.0
        elif st_np[s] == D:
            dom_d_mask[s, dd] = 1.0

    dom_m_mask_j = jnp.array(dom_m_mask)
    dom_i_mask_j = jnp.array(dom_i_mask)
    dom_d_mask_j = jnp.array(dom_d_mask)
    m_mask_f_j = jnp.array(m_mask_f)

    # Group pairs by t_est and build per-t substitution matrices
    # Then bucket by (Lx_pad, Ly_pad) for vmap
    from collections import defaultdict

    # For simplicity, build sub_matrices per unique t_est.
    # But pairs have varying t_est. We'll use per-pair t_est for emissions
    # but a common log_chi. Actually, the transition matrix chi depends on t
    # too... for v1, just use t_est per pair for sub_matrices and the
    # median t for chi (matching the current SVI-BW approach).

    # Build params dict for build_per_domain_sub_matrices
    if params is not None and 'S_exch' in params:
        sub_params = {
            'S_exch': params.get('dom_S_exch', params.get('S_exch')),
            'pi': params.get('dom_pis', pi_lg),
        }
    elif dom_Qs is not None and dom_pis is not None:
        sub_params = None  # will compute directly
    else:
        sub_params = None

    # Preprocess pairs: extract x_int, y_int, t_est, compute Lx_pad, Ly_pad
    processed = []
    for item in batch_pairs:
        x_int, y_int, states, anc_chars, desc_chars, t_est = item
        x = np.asarray(x_int, dtype=np.int32)
        y = np.asarray(y_int, dtype=np.int32)
        Lx = len(x)
        Ly = len(y)
        Lx_pad = _pad_to_bin(Lx)
        Ly_pad = _pad_to_bin(Ly)
        processed.append((x, y, Lx, Ly, Lx_pad, Ly_pad, t_est))

    # Bucket by (Lx_pad, Ly_pad)
    buckets = defaultdict(list)
    for item in processed:
        x, y, Lx, Ly, Lx_pad, Ly_pad, t_est = item
        buckets[(Lx_pad, Ly_pad)].append(item)

    # Initialize delta
    delta = {
        'agg_n_chi': np.zeros((N, N)),
        'total_ll': 0.0,
        'n_pairs': 0,
        'n_files': 1,
        'match_counts': np.zeros((AA, AA)),
        'dom_match_counts': np.zeros((n_dom, AA, AA)),
        'dom_insert_counts': np.zeros((n_dom, AA)),
        'dom_delete_counts': np.zeros((n_dom, AA)),
        # Per-pair-t Holmes-Rubin sufficient stats (see 1D path comment).
        'dom_W': np.zeros((n_dom, AA)),
        'dom_U': np.zeros((n_dom, AA, AA)),
    }
    # Per-pair (n_chi, t) records for per-pair exact_suffstats accumulation.
    _pair_n_chi_t_records = []
    # Per-pair (mc_p_d, t_p) records for the post-loop batched HR call
    # (analog of the 1D path's records — same per-pair-t coherence guarantee).
    _pair_mc_d_records_2d = []
    _pair_t_records_2d = []

    # Per-domain Q, pi for per-pair Holmes-Rubin (LG fallback).
    if dom_Qs is not None and dom_pis is not None:
        _hr_Qs_2d = np.asarray(dom_Qs)
        _hr_pis_2d = np.asarray(dom_pis)
    else:
        _hr_Qs_2d = np.tile(np.asarray(Q_lg)[None], (n_dom, 1, 1))
        _hr_pis_2d = np.tile(np.asarray(pi_lg)[None], (n_dom, 1))

    batched_fb_2d_per_chi = _get_batched_fb_2d_per_chi_jit()
    from tkfmixdom.jax.dp.hmm import safe_log

    # Build a params dict that _build_per_pair_sub_matrices will see as
    # "per-domain subst" iff dom_Qs/dom_pis were supplied here. (The 1D
    # path threads dom_Qs/dom_pis explicitly; do the same shim here so we
    # don't accidentally fall through to LG when params['dom_Qs'] is None.)
    sub_params = {'dom_Qs': dom_Qs, 'dom_pis': dom_pis} if (
        dom_Qs is not None and dom_pis is not None) else {}

    for (Lx_pad, Ly_pad), bucket in buckets.items():
        B = len(bucket)

        # Stack sequences (pad with zeros)
        xs = np.zeros((B, Lx_pad), dtype=np.int32)
        ys = np.zeros((B, Ly_pad), dtype=np.int32)
        real_Lxs = np.zeros(B, dtype=np.int32)
        real_Lys = np.zeros(B, dtype=np.int32)

        for i, (x, y, Lx, Ly, _, _, t_est) in enumerate(bucket):
            xs[i, :Lx] = x
            ys[i, :Ly] = y
            real_Lxs[i] = Lx
            real_Lys[i] = Ly

        # Per-pair t_est array for this bucket (item[6] is t_est).
        t_ests = np.array([item[6] for item in bucket], dtype=np.float32)

        # Per-pair sub_matrices stack (B, n_dom, A, A): each pair's sub
        # matrices at the pair's own t_est.
        sub_matrices_stack, pis_arr = _build_per_pair_sub_matrices(
            t_ests, sub_params, Q_lg, pi_lg, n_dom)

        # Per-pair log_chi stack (B, N, N): each slice at this pair's t_est.
        t_ests_j = jnp.asarray(t_ests)
        log_chi_stack = _build_log_chi_stack(chi_params, t_ests_j)

        # Vmapped 2D FB with per-pair chi + per-pair sub matrices
        results = batched_fb_2d_per_chi(
            log_chi_stack, jnp.array(st),
            jnp.array(xs), jnp.array(ys),
            jnp.array(real_Lxs), jnp.array(real_Lys),
            jnp.array(sub_matrices_stack), jnp.array(pis_arr),
            n_dom, n_frag,
            dom_m_mask_j, dom_i_mask_j, dom_d_mask_j, m_mask_f_j)

        log_probs, n_chis, dom_matches, dom_inserts, dom_deletes, agg_matches = results

        # Accumulate per-pair in float64 (skip NaN pairs from numerical issues)
        log_probs_np = np.asarray(log_probs)
        n_chis_np = np.asarray(n_chis)
        dom_matches_np = np.asarray(dom_matches)
        dom_inserts_np = np.asarray(dom_inserts)
        dom_deletes_np = np.asarray(dom_deletes)
        agg_matches_np = np.asarray(agg_matches)

        for i in range(B):
            if np.isnan(log_probs_np[i]) or np.isinf(log_probs_np[i]):
                continue  # skip numerically failed pairs (long sequences)
            delta['agg_n_chi'] += n_chis_np[i]
            _pair_n_chi_t_records.append(
                (n_chis_np[i].astype(np.float64).copy(), float(t_ests[i])))
            delta['total_ll'] += float(log_probs_np[i])
            delta['n_pairs'] += 1
            delta['match_counts'] += agg_matches_np[i]
            delta['dom_match_counts'] += dom_matches_np[i]
            delta['dom_insert_counts'] += dom_inserts_np[i]
            delta['dom_delete_counts'] += dom_deletes_np[i]
            # Stash per-pair (mc_p_d, t_p) for the batched HR call. Each
            # pair's match counts (dom_matches_np[i]) are paired with that
            # pair's own t_est — preserved per-pair-t coherence with the
            # FB pass above.
            _pair_mc_d_records_2d.append(
                dom_matches_np[i].astype(np.float64).copy())
            _pair_t_records_2d.append(float(t_ests[i]))

    if delta['n_pairs'] == 0:
        return None

    # Batched per-pair-t Holmes-Rubin: replaces the per-pair-per-domain
    # Python loop. Same (Q_d, pi_d) frozen across the batch as the FB pass.
    if _pair_mc_d_records_2d:
        mc_stack_2d = np.stack(_pair_mc_d_records_2d)
        t_stack_2d = np.array(_pair_t_records_2d, dtype=np.float64)
        W_stack_2d, U_stack_2d = _build_per_pair_hr_stats(
            _hr_Qs_2d, _hr_pis_2d, t_stack_2d, mc_stack_2d)
        delta['dom_W'] += W_stack_2d.sum(axis=0)
        delta['dom_U'] += U_stack_2d.sum(axis=0)

    # Per-pair exact_suffstats accumulation (see _process_pairs_batched).
    if _pair_n_chi_t_records:
        from tkfmixdom.jax.models.exact_suffstats import (
            exact_suffstats_per_pair_batch)
        delta['exact_ss'] = exact_suffstats_per_pair_batch(
            _pair_n_chi_t_records,
            chi_params['main_ins'], chi_params['main_del'],
            np.asarray(chi_params['dom_ins']),
            np.asarray(chi_params['dom_del']),
            np.asarray(chi_params['dom_weights']),
            np.asarray(chi_params['frag_weights']),
            np.asarray(chi_params['ext_rates']))
    return delta


# ============================================================
# Match-aligned 2D E-step: match states constrained to alignment
# ============================================================

def _extract_match_positions(states):
    """Extract match positions from an alignment state sequence.

    Args:
        states: list/array of state codes (M=1, I=2, D=3)

    Returns:
        (match_i, match_j): numpy arrays of 0-indexed positions in the
        ancestor (x) and descendant (y) sequences where matches occur.
    """
    i = j = 0
    match_i, match_j = [], []
    for s in states:
        if s == 1:  # M
            match_i.append(i)
            match_j.append(j)
            i += 1
            j += 1
        elif s == 2:  # I
            j += 1
        elif s == 3:  # D
            i += 1
    return np.array(match_i, dtype=np.int32), np.array(match_j, dtype=np.int32)


# Module-level cache for JIT'd vmapped match-aligned 2D FB
_batched_fb_2d_match_aligned_cache = None


def _get_batched_fb_2d_match_aligned_jit():
    """Lazily create and cache a JIT'd vmapped match-aligned 2D E-step kernel."""
    global _batched_fb_2d_match_aligned_cache
    if _batched_fb_2d_match_aligned_cache is not None:
        return _batched_fb_2d_match_aligned_cache

    import jax
    import jax.numpy as _jnp
    from tkfmixdom.jax.dp.hmm import (
        forward_backward_2d, pair_hmm_emissions_per_domain,
        mask_emissions_match_aligned,
    )

    def _fb_2d_match_core(log_chi, st, xs, ys, real_Lxs, real_Lys,
                          match_is, match_js, match_lens,
                          sub_matrices, pis, n_dom, n_frag,
                          dom_m_mask, dom_i_mask, dom_d_mask, m_mask_f):
        """Single-pair match-aligned 2D FB + suff stat reduction (vmapped)."""
        jnp = _jnp

        def _single(x, y, real_Lx, real_Ly, mi, mj, mlen):
            A = sub_matrices.shape[-1]
            log_emit = pair_hmm_emissions_per_domain(
                st, x, y, sub_matrices, pis, n_dom, n_frag)

            # Apply match-aligned mask: constrain match states to alignment positions
            # mi/mj are padded arrays; only use first mlen entries
            mi_real = jnp.where(jnp.arange(mi.shape[0]) < mlen, mi, 0)
            mj_real = jnp.where(jnp.arange(mj.shape[0]) < mlen, mj, 0)
            log_emit = mask_emissions_match_aligned(log_emit, st, mi_real, mj_real)

            log_prob, posteriors, expected_trans = forward_backward_2d(
                log_chi, st, x, y, None, None,
                log_emit_table=log_emit, real_Lx=real_Lx, real_Ly=real_Ly)

            Lx_pad = x.shape[0]
            Ly_pad = y.shape[0]

            # One-hot encode sequences
            X_oh = jax.nn.one_hot(x, A)  # (Lx_pad, A)
            Y_oh = jax.nn.one_hot(y, A)  # (Ly_pad, A)

            # Match counts: (n_dom, A, A)
            post_MM = posteriors[1:Lx_pad+1, 1:Ly_pad+1, :]
            W_M = jnp.einsum('ijs,sd->dij', post_MM, dom_m_mask)
            tmp_M = jnp.einsum('dij,ia->daj', W_M, X_oh)
            dom_match = jnp.einsum('daj,jb->dab', tmp_M, Y_oh)

            # Insert counts: (n_dom, A)
            post_I = posteriors[0:Lx_pad+1, 1:Ly_pad+1, :]
            I_per_j = jnp.einsum('ijs,sd->dj', post_I, dom_i_mask)
            dom_insert = jnp.einsum('dj,ja->da', I_per_j, Y_oh)

            # Delete counts: (n_dom, A)
            post_D = posteriors[1:Lx_pad+1, 0:Ly_pad+1, :]
            D_per_i = jnp.einsum('ijs,sd->di', post_D, dom_d_mask)
            dom_delete = jnp.einsum('di,ia->da', D_per_i, X_oh)

            # Aggregate match counts
            agg_match = dom_match.sum(axis=0)

            return log_prob, expected_trans, dom_match, dom_insert, dom_delete, agg_match

        return jax.vmap(_single, in_axes=(0, 0, 0, 0, 0, 0, 0))(
            xs, ys, real_Lxs, real_Lys, match_is, match_js, match_lens)

    _batched_fb_2d_match_aligned_cache = jax.jit(
        _fb_2d_match_core, static_argnames=('n_dom', 'n_frag'))
    return _batched_fb_2d_match_aligned_cache


def _process_pairs_batched_match_aligned(batch_pairs, chi_params, st, Q_lg, pi_lg, N,
                                          dom_Qs=None, dom_pis=None, n_dom=1, n_frag=1,
                                          params=None):
    """Process a batch of pairs with match-aligned 2D forward-backward.

    Like _process_pairs_batched_2d but constrains match states to fire only
    at positions aligned in the training data. Insert/delete states are
    unconstrained, so the DP sums over all indel orderings.

    Per-pair t coherence: chi (TKF nested transitions) and substitution
    matrices are both built at each pair's own t_est, so the joint pair HMM
    is coherent at one well-defined t per pair (analog of the 1D per-chi
    fix in `_process_pairs_batched`).

    batch_pairs: list of (x_int, y_int, states, anc_chars, desc_chars, t_est)
        Pre-decoded pair tuples. x_int, y_int, states, and t_est are used.
        states provides the alignment path to extract match positions.
    chi_params: params dict — see `_process_pairs_batched_2d`. (Replaces the
        legacy `log_chi` arg, which was a single (N, N) matrix at a global t
        and incoherent with per-pair-t emissions.)

    Returns suff_stats delta dict (same format as _process_file_pairs), or None.
    """
    import jax
    import jax.numpy as jnp
    from tkfmixdom.jax.dp.hmm import NEG_INF
    from tkfmixdom.jax.core.params import M as _M, I as _I, D as _D
    from tkfmixdom.jax.models.mixdom import build_nested_trans

    if not batch_pairs:
        return None

    st_np = np.array(st)
    ns = len(st_np)
    is_M = st_np == _M
    is_I = st_np == _I
    is_D = st_np == _D

    # Domain masks for einsum reduction (same as _process_pairs_batched_2d)
    body_idx = np.maximum((np.arange(ns) - 2) // (5 * n_frag), 0)
    dom_m_mask = np.zeros((ns, n_dom))
    dom_i_mask = np.zeros((ns, n_dom))
    dom_d_mask = np.zeros((ns, n_dom))
    m_mask_f = np.zeros(ns)
    for s in range(2, ns):
        dd = body_idx[s]
        if st_np[s] == _M:
            dom_m_mask[s, dd] = 1.0
            m_mask_f[s] = 1.0
        elif st_np[s] == _I:
            dom_i_mask[s, dd] = 1.0
        elif st_np[s] == _D:
            dom_d_mask[s, dd] = 1.0

    dom_m_mask_j = jnp.array(dom_m_mask)
    dom_i_mask_j = jnp.array(dom_i_mask)
    dom_d_mask_j = jnp.array(dom_d_mask)
    m_mask_f_j = jnp.array(m_mask_f)

    # Preprocess pairs: extract x_int, y_int, t_est, match positions
    from collections import defaultdict

    processed = []
    for item in batch_pairs:
        x_int, y_int, states, anc_chars, desc_chars, t_est = item
        x = np.asarray(x_int, dtype=np.int32)
        y = np.asarray(y_int, dtype=np.int32)
        Lx = len(x)
        Ly = len(y)
        Lx_pad = _pad_to_bin(Lx)
        Ly_pad = _pad_to_bin(Ly)
        mi, mj = _extract_match_positions(states)
        processed.append((x, y, Lx, Ly, Lx_pad, Ly_pad, t_est, mi, mj))

    # Bucket by (Lx_pad, Ly_pad)
    buckets = defaultdict(list)
    for item in processed:
        x, y, Lx, Ly, Lx_pad, Ly_pad, t_est, mi, mj = item
        buckets[(Lx_pad, Ly_pad)].append(item)

    # Initialize delta
    delta = {
        'agg_n_chi': np.zeros((N, N)),
        'total_ll': 0.0,
        'n_pairs': 0,
        'n_files': 1,
        'match_counts': np.zeros((AA, AA)),
        'dom_match_counts': np.zeros((n_dom, AA, AA)),
        'dom_insert_counts': np.zeros((n_dom, AA)),
        'dom_delete_counts': np.zeros((n_dom, AA)),
        'dom_W': np.zeros((n_dom, AA)),
        'dom_U': np.zeros((n_dom, AA, AA)),
    }
    # Per-pair (n_chi, t) records for per-pair exact_suffstats accumulation.
    _pair_n_chi_t_records = []
    # Per-pair (mc_p_d, t_p) records for the post-loop batched HR call
    # (analog of the 1D path's records — same per-pair-t coherence guarantee).
    _pair_mc_d_records_ma = []
    _pair_t_records_ma = []

    # Per-domain Q, pi for per-pair Holmes-Rubin (LG fallback).
    if dom_Qs is not None and dom_pis is not None:
        _hr_Qs_ma = np.asarray(dom_Qs)
        _hr_pis_ma = np.asarray(dom_pis)
    else:
        _hr_Qs_ma = np.tile(np.asarray(Q_lg)[None], (n_dom, 1, 1))
        _hr_pis_ma = np.tile(np.asarray(pi_lg)[None], (n_dom, 1))

    batched_fb_per_chi = _get_batched_fb_2d_match_aligned_per_chi_jit()
    from tkfmixdom.jax.dp.hmm import safe_log

    sub_params = {'dom_Qs': dom_Qs, 'dom_pis': dom_pis} if (
        dom_Qs is not None and dom_pis is not None) else {}

    for (Lx_pad, Ly_pad), bucket in buckets.items():
        B = len(bucket)

        # Stack sequences (pad with zeros)
        xs = np.zeros((B, Lx_pad), dtype=np.int32)
        ys = np.zeros((B, Ly_pad), dtype=np.int32)
        real_Lxs = np.zeros(B, dtype=np.int32)
        real_Lys = np.zeros(B, dtype=np.int32)

        # Find max match count in this bucket for padding match arrays
        max_n_match = max(len(item[7]) for item in bucket)
        if max_n_match == 0:
            max_n_match = 1  # need at least 1 for array shape
        match_is = np.zeros((B, max_n_match), dtype=np.int32)
        match_js = np.zeros((B, max_n_match), dtype=np.int32)
        match_lens = np.zeros(B, dtype=np.int32)

        for i, (x, y, Lx, Ly, _, _, t_est, mi, mj) in enumerate(bucket):
            xs[i, :Lx] = x
            ys[i, :Ly] = y
            real_Lxs[i] = Lx
            real_Lys[i] = Ly
            n_m = len(mi)
            match_is[i, :n_m] = mi
            match_js[i, :n_m] = mj
            match_lens[i] = n_m

        # Per-pair t_est for this bucket (item[6] is t_est).
        t_ests = np.array([item[6] for item in bucket], dtype=np.float32)

        # Per-pair sub_matrices stack (B, n_dom, A, A) at each pair's t_est
        sub_matrices_stack, pis_arr = _build_per_pair_sub_matrices(
            t_ests, sub_params, Q_lg, pi_lg, n_dom)

        # Per-pair log_chi stack (B, N, N) at each pair's t_est
        t_ests_j = jnp.asarray(t_ests)
        log_chi_stack = _build_log_chi_stack(chi_params, t_ests_j)

        # Vmapped match-aligned 2D FB with per-pair chi + per-pair sub matrices
        results = batched_fb_per_chi(
            log_chi_stack, jnp.array(st),
            jnp.array(xs), jnp.array(ys),
            jnp.array(real_Lxs), jnp.array(real_Lys),
            jnp.array(match_is), jnp.array(match_js), jnp.array(match_lens),
            jnp.array(sub_matrices_stack), jnp.array(pis_arr),
            n_dom, n_frag,
            dom_m_mask_j, dom_i_mask_j, dom_d_mask_j, m_mask_f_j)

        log_probs, n_chis, dom_matches, dom_inserts, dom_deletes, agg_matches = results

        # Accumulate per-pair in float64 (skip NaN pairs)
        log_probs_np = np.asarray(log_probs)
        n_chis_np = np.asarray(n_chis)
        dom_matches_np = np.asarray(dom_matches)
        dom_inserts_np = np.asarray(dom_inserts)
        dom_deletes_np = np.asarray(dom_deletes)
        agg_matches_np = np.asarray(agg_matches)

        for i in range(B):
            if np.isnan(log_probs_np[i]) or np.isinf(log_probs_np[i]):
                continue
            delta['agg_n_chi'] += n_chis_np[i]
            _pair_n_chi_t_records.append(
                (n_chis_np[i].astype(np.float64).copy(), float(t_ests[i])))
            delta['total_ll'] += float(log_probs_np[i])
            delta['n_pairs'] += 1
            delta['match_counts'] += agg_matches_np[i]
            delta['dom_match_counts'] += dom_matches_np[i]
            delta['dom_insert_counts'] += dom_inserts_np[i]
            delta['dom_delete_counts'] += dom_deletes_np[i]
            # Stash per-pair (mc_p_d, t_p) for the batched HR call.
            _pair_mc_d_records_ma.append(
                dom_matches_np[i].astype(np.float64).copy())
            _pair_t_records_ma.append(float(t_ests[i]))

    if delta['n_pairs'] == 0:
        return None

    # Batched per-pair-t Holmes-Rubin (replaces per-pair-per-domain loop).
    if _pair_mc_d_records_ma:
        mc_stack_ma = np.stack(_pair_mc_d_records_ma)
        t_stack_ma = np.array(_pair_t_records_ma, dtype=np.float64)
        W_stack_ma, U_stack_ma = _build_per_pair_hr_stats(
            _hr_Qs_ma, _hr_pis_ma, t_stack_ma, mc_stack_ma)
        delta['dom_W'] += W_stack_ma.sum(axis=0)
        delta['dom_U'] += U_stack_ma.sum(axis=0)

    # Per-pair exact_suffstats accumulation (see _process_pairs_batched).
    if _pair_n_chi_t_records:
        from tkfmixdom.jax.models.exact_suffstats import (
            exact_suffstats_per_pair_batch)
        delta['exact_ss'] = exact_suffstats_per_pair_batch(
            _pair_n_chi_t_records,
            chi_params['main_ins'], chi_params['main_del'],
            np.asarray(chi_params['dom_ins']),
            np.asarray(chi_params['dom_del']),
            np.asarray(chi_params['dom_weights']),
            np.asarray(chi_params['frag_weights']),
            np.asarray(chi_params['ext_rates']))
    return delta


# ============================================================
# Main training loop
# ============================================================
def train(args):
    """Train MixDom on Pfam MSAs via constrained Baum-Welch."""
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    from tkfmixdom.jax.core.ctmc import transition_matrix
    from tkfmixdom.jax.core.params import S, M, I, D, E
    from tkfmixdom.jax.models.mixdom import (
        build_nested_trans, n_states as mixdom_n_states,
        state_types as mixdom_state_types
    )
    from tkfmixdom.jax.train.constrained import mixdom_constrained_e_step
    from tkfmixdom.jax.models.compiled import init_mixdom_params as _init_params
    from tkfmixdom.jax.models.exact_suffstats import exact_suffstats
    from tkfmixdom.jax.core.bdi import bdi_stats_from_counts, m_step_indel_quadratic
    from tkfmixdom.jax.simulate.msa import alignment_to_states
    from tkfmixdom.jax.dp.hmm import safe_log

    n_dom = args.n_dom
    n_frag = args.n_frag
    n_iter = args.n_iter
    checkpoint_path = args.checkpoint or os.path.join(
        args.msa_dir, f'train_d{n_dom}_f{n_frag}.npz')
    checkpoint_every = args.checkpoint_every

    budget_hours = args.budget_hours
    config = {
        'n_dom': n_dom, 'n_frag': n_frag, 'n_iter': n_iter,
        'init_ins': args.init_ins, 'init_del': args.init_del,
        'seed': args.seed, 'msa_dir': args.msa_dir,
        'families': args.families,
        'budget_hours': budget_hours,
        'pseudocounts': {
            'ins_prior': list(args.ins_prior),
            'del_prior': list(args.del_prior),
            'dom_dirichlet': args.dom_dirichlet,
            'frag_dirichlet': args.frag_dirichlet,
            'ext_alpha': args.ext_alpha,
            'ext_beta': args.ext_beta,
        },
    }

    N = mixdom_n_states(n_dom, n_frag)
    st = mixdom_state_types(n_dom, n_frag)
    Q_lg, pi_lg = rate_matrix_lg()

    # ---- Check for precompiled pairs ----
    precompiled_source = None
    if getattr(args, 'precompiled_pairs', None):
        precompiled_source = PrecompiledPairSource(args.precompiled_pairs)
        _log(f"Using precompiled pairs: {precompiled_source.description}")

    # ---- Resolve MSA directory via bio-datasets if available ----
    # Only auto-resolve via bio-datasets when using the default msa-dir.
    # If the user explicitly passed --msa-dir, respect it.
    # Skip MSA directory resolution when using precompiled pairs.
    if precompiled_source is not None:
        # Precompiled mode: skip MSA file discovery entirely
        msa_files = []
        _log(f"  Skipping MSA file discovery (using precompiled pairs)")
    else:
        from tkfmixdom.jax.util.bio_datasets import (
            apply_bio_datasets_arg, resolve_data_dir, ensure_symlinks)
        apply_bio_datasets_arg(args)
        _msa_dir_is_default = (args.msa_dir == 'pfam/')
        if _msa_dir_is_default:
            msa_dir_resolved = str(resolve_data_dir("pfam/seed", local_fallback=args.msa_dir))
            if msa_dir_resolved != args.msa_dir:
                _log(f"  bio-datasets: {args.msa_dir} → {msa_dir_resolved}")
                ensure_symlinks(
                    __import__('pathlib').Path(msa_dir_resolved),
                    __import__('pathlib').Path(args.msa_dir), pattern="*.sto")
                ensure_symlinks(
                    __import__('pathlib').Path(msa_dir_resolved),
                    __import__('pathlib').Path(args.msa_dir), pattern="*.sto.gz")
            args.msa_dir = msa_dir_resolved

    # ---- Find MSA files ----
    if precompiled_source is None:
        import glob as glob_mod
        if args.families:
            family_list = [f.strip() for f in args.families.split(',')]
            msa_files = []
            for fam in family_list:
                for ext in ['.sto', '.sto.gz', '.stockholm', '.stockholm.gz']:
                    p = os.path.join(args.msa_dir, fam + ext)
                    if os.path.exists(p):
                        msa_files.append(p)
                        break
                else:
                    _log(f"Warning: {fam} not found in {args.msa_dir}")
        else:
            msa_files = sorted(
                glob_mod.glob(os.path.join(args.msa_dir, '*.sto')) +
                glob_mod.glob(os.path.join(args.msa_dir, '*.sto.gz')))

        if not msa_files:
            _log(f"No MSA files found in {args.msa_dir}")
            sys.exit(1)
        _log(f"Found {len(msa_files)} MSA files")

    # ---- Data split ----
    if args.split and precompiled_source is None:
        # Try to load pre-made split file
        split_file = args.split_file
        if not split_file:
            # Auto-detect canonical split
            for candidate in [
                os.path.expanduser('~/bio-datasets/data/pfam/seed/splits/v1.json'),
                os.path.expanduser('~/bio-datasets/data/pfam-seed/splits/v1.json'),
                os.path.join(args.msa_dir, 'splits', 'v1.json'),
            ]:
                if os.path.exists(candidate):
                    split_file = candidate
                    break

        if split_file and os.path.exists(split_file):
            import json as json_mod
            with open(split_file) as f:
                split_data = json_mod.load(f)
            split_fams = set(split_data.get(args.split, []))
            if not split_fams:
                _log(f"ERROR: split '{args.split}' not found in {split_file}")
                sys.exit(1)
            # Filter msa_files to those in the split
            msa_files = [f for f in msa_files
                         if os.path.basename(f).split('.')[0] in split_fams]
            _log(f"  Split file: {split_file}")
            _log(f"  Using {args.split} split: {len(msa_files)} families "
                 f"(of {len(split_fams)} in split)")
        else:
            # Fall back to ad-hoc clan-aware split
            clan_file = args.clan_file or _find_clan_file(args.msa_dir)
            if not clan_file:
                _log("ERROR: --split requires --split-file or a clan file.")
                sys.exit(1)
            ratios = tuple(float(r) for r in args.split_ratios.split(','))
            split_indices = clan_aware_split(
                msa_files, clan_file, args.split, ratios, args.split_seed)
            msa_files = [msa_files[i] for i in split_indices]
            _log(f"  Ad-hoc {args.split} split: {len(msa_files)} families")

    # ---- Load checkpoint or initialize ----
    manifest = None
    t_rep = None  # representative t
    budget_files = None  # frozen family subset from budget sampling
    family_timings = None  # per-family E-step wall times
    if os.path.exists(checkpoint_path) and not args.fresh:
        _log(f"Resuming from checkpoint: {checkpoint_path}")
        params, suff_stats, step_file_idx, em_iter, log_probs, \
            saved_config, manifest, t_rep, budget_files, family_timings = \
            _load_checkpoint(checkpoint_path)
        if saved_config.get('n_dom') != n_dom or saved_config.get('n_frag') != n_frag:
            _log(f"ERROR: checkpoint n_dom/n_frag mismatch. Use --fresh.")
            sys.exit(1)
        if not suff_stats:
            suff_stats = _zero_suff_stats(N, n_dom)
        if getattr(args, 'rebuild_manifest', False):
            _log("  --rebuild-manifest: discarding cached manifest and budget files")
            _log("  Parameters and iteration count preserved")
            manifest = None
            budget_files = None
            family_timings = None
            suff_stats = _zero_suff_stats(N, n_dom)
            step_file_idx = 0
        _log(f"  EM iter {em_iter}, file_idx {step_file_idx}")
    else:
        mar_path = getattr(args, 'init_from_maraschino', None)
        if mar_path:
            _log(f"Initializing from Maraschino: {mar_path}")
            params = _load_maraschino_as_train_params(mar_path, n_dom, n_frag)
            args.estimate_subst = True
        else:
            _log("Initializing fresh parameters...")
            main_ins, main_del, dom_ins, dom_del, dom_weights, frag_weights, ext_rates = \
                _init_params(args.init_ins, args.init_del, n_dom, n_frag, seed=args.seed)
            if getattr(args, 'banded_frag_init', False):
                from tkfmixdom.jax.train.restricted_mstep import banded_3fc_init
                frag_weights, ext_rates = banded_3fc_init(
                    n_dom, float(getattr(args, 'p_ext', 0.6)))
                _log(f"  banded-frag-init: p_ext="
                     f"{float(getattr(args, 'p_ext', 0.6)):.4f}")
            params = {
                'main_ins': float(main_ins), 'main_del': float(main_del),
                'dom_ins': np.array(dom_ins), 'dom_del': np.array(dom_del),
                'dom_weights': np.array(dom_weights),
                'frag_weights': np.array(frag_weights),
                'ext_rates': np.array(ext_rates),
            }
            # Per-domain (S_exch, π) seed: tile LG with Dirichlet (π) and
            # Gamma (S_exch) symmetry-breaking perturbations per domain.
            # Always populated since Phase 5.6 Sub-phase B; the legacy
            # `--estimate-subst` toggle is gone. Concentration tuned to
            # produce ~5-10% per-entry perturbation: large enough to
            # break domain symmetry on Adam's first steps, small enough
            # that the LG seed structure dominates initially.
            rng = np.random.RandomState(args.seed + 1000)
            dom_pis_init = np.tile(np.array(pi_lg), (n_dom, 1))
            S_lg_seed = (np.array(Q_lg) / np.maximum(np.array(pi_lg)[None, :], 1e-30)
                          * (1.0 - np.eye(AA)))
            S_lg_seed = (S_lg_seed + S_lg_seed.T) / 2
            dom_S_init = np.tile(S_lg_seed[None, :, :], (n_dom, 1, 1))
            for dd in range(n_dom):
                pi_noise = rng.dirichlet(np.ones(AA) * 10.0)
                dom_pis_init[dd] = 0.85 * dom_pis_init[dd] + 0.15 * pi_noise
                dom_pis_init[dd] /= dom_pis_init[dd].sum()
                S_noise = rng.gamma(2.0, 0.05, (AA, AA))
                S_noise = (S_noise + S_noise.T) / 2
                np.fill_diagonal(S_noise, 0.0)
                dom_S_init[dd] = dom_S_init[dd] + S_noise
                dom_S_init[dd] = (dom_S_init[dd] + dom_S_init[dd].T) / 2
            params['dom_Qs'] = np.tile(np.array(Q_lg)[None, :, :], (n_dom, 1, 1))
            params['dom_pis'] = dom_pis_init
            params['dom_S_exch'] = dom_S_init
            _log(f"  Per-domain substitution: {n_dom} domains × {AA} aa "
                 f"(LG-tiled, Dirichlet+Gamma symmetry-breaking; trainable "
                 f"under Adam since Phase 5.6 Sub-phase B)")
        suff_stats = _zero_suff_stats(N, n_dom)
        step_file_idx = 0
        em_iter = 0
        log_probs = []

    # ---- Build pair manifest ----
    # Skip manifest entirely when using precompiled pairs.
    # For interleaved budget mode with no frozen pairs yet, skip the full
    # manifest scan — pairs are built lazily during the E-step.
    if precompiled_source is not None:
        # Precompiled mode: no manifest needed
        manifest = np.array([], dtype=MANIFEST_DTYPE)
        if t_rep is None:
            # Compute t_rep from precompiled stats
            stats = precompiled_source.manifest.get('stats', {})
            t_rep = stats.get('t_est', {}).get('median', 1.0)
        _log(f"  Precompiled mode: t_rep={t_rep:.3f}")
        _lazy_manifest = False
    else:
        _lazy_manifest = False  # default, may be overridden below

    if precompiled_source is None and (args.interleaved and budget_hours is not None
                      and budget_files is None and manifest is None):
        _lazy_manifest = True

    if _lazy_manifest:
        _log("  Lazy manifest: pairs will be built on-the-fly during E-step")
        manifest = np.array([], dtype=MANIFEST_DTYPE)
        if t_rep is None:
            t_rep = 1.0  # will be updated after first E-step
    elif manifest is None:
        _log("Building pair manifest (cherry selection, one-time)...")
        manifest, t_rep = build_pair_manifest(msa_files)
        _log(f"  {len(manifest)} pairs, median t={t_rep:.3f}")
        _save_checkpoint(checkpoint_path, params, suff_stats, step_file_idx,
                         em_iter, log_probs, config, manifest, t_rep,
                         budget_files, family_timings)
        _log(f"  Manifest cached in checkpoint")
    else:
        _log(f"  Loaded cached manifest: {len(manifest)} pairs, t={t_rep:.3f}")

        # Validate manifest (skip if empty — lazy mode with frozen pairs)
        if len(manifest) == 0 and budget_files is not None:
            _log("  (empty manifest from lazy mode, using frozen pairs)")
        elif len(manifest) > 0:
            pass  # validate below
        # Check that all referenced file indices exist
        max_fi = int(max(rec['file_idx'] for rec in manifest)) if len(manifest) > 0 else -1
        bad_files = []
        for fi in sorted(set(int(rec['file_idx']) for rec in manifest)):
            if fi >= len(msa_files):
                bad_files.append(f"  index {fi} out of range (only {len(msa_files)} files found)")
            elif not os.path.exists(msa_files[fi]):
                bad_files.append(f"  index {fi}: {msa_files[fi]} (missing)")
        if bad_files:
            _log("ERROR: Manifest references missing or invalid MSA files:")
            for b in bad_files[:10]:
                _log(b)
            if len(bad_files) > 10:
                _log(f"  ... and {len(bad_files) - 10} more")
            _log("")
            _log("The manifest was built against a different set of MSA files.")
            _log("To fix, add --rebuild-manifest to rebuild the file index.")
            _log("This keeps trained parameters and iteration count intact;")
            _log("only the manifest and budget file list are rebuilt.")
            _log(f"  Example: python train_pfam.py ... --checkpoint {checkpoint_path} --rebuild-manifest")
            sys.exit(1)

    if len(manifest) == 0 and not _lazy_manifest and budget_files is None \
            and precompiled_source is None:
        _log("No valid pairs found!")
        sys.exit(1)

    t = t_rep

    # ---- Group manifest by file_idx for streaming ----
    # pairs_by_file[fi] = [(row1, row2, t_est), ...]
    # In lazy mode, this starts empty and is populated during E-step.
    pairs_by_file = {}
    for rec in manifest:
        fi = int(rec['file_idx'])
        pairs_by_file.setdefault(fi, []).append(
            (int(rec['row1']), int(rec['row2']), float(rec['t_est'])))

    # Sorted unique file indices that have pairs
    # In lazy mode, all files are potentially active
    if _lazy_manifest:
        all_active_files = list(range(len(msa_files)))
    else:
        all_active_files = sorted(pairs_by_file.keys())

    # ---- Family selection ----
    # Priority: checkpoint budget_files > --max-families > --budget-hours > all
    if budget_files is not None:
        if isinstance(budget_files, list) and budget_files and isinstance(budget_files[0], tuple):
            # Interleaved frozen pairs: extract unique file indices
            active_files = sorted(set(fi for fi, _ in budget_files))
            _log(f"  Using frozen interleaved pairs: {len(budget_files)} pairs "
                 f"from {len(active_files)} families")
        else:
            active_files = budget_files
            _log(f"  Using frozen budget subset: {len(active_files)} families")
    elif args.max_families is not None and args.max_families < len(all_active_files):
        rng = np.random.RandomState(args.seed)
        shuffled = list(all_active_files)
        rng.shuffle(shuffled)
        active_files = shuffled[:args.max_families]
        _log(f"  --max-families: selected {len(active_files)} of "
             f"{len(all_active_files)} families (seed={args.seed})")
    else:
        active_files = list(all_active_files)
    n_active = len(active_files)

    # ---- Report manifest stats ----
    _log(f"\n{'='*60}")
    _log(f"MixDom training: {n_dom} domains, {n_frag} fragments, "
         f"{N} states, t={t:.3f}")
    if precompiled_source is not None:
        _log(f"  {precompiled_source.description}")
    elif len(manifest) > 0:
        total_cols = int(manifest['n_cols'].sum())
        _log(f"  {len(manifest)} pairs from {len(all_active_files)} MSAs, "
             f"{total_cols} total columns")
    else:
        _log(f"  {len(all_active_files)} MSAs (lazy manifest, pairs built on-the-fly)")
    if budget_hours is not None:
        budget_per_iter_s = budget_hours * 3600.0 / n_iter
        _log(f"  Budget: {budget_hours}h total, {budget_per_iter_s:.0f}s/iter "
             f"({n_iter} iters)")
    if len(manifest) > 0:
        _log(f"  Columns/pair: median={int(np.median(manifest['n_cols']))}, "
             f"mean={int(np.mean(manifest['n_cols']))}")
    _log(f"  Pseudocounts: ins={args.ins_prior}, del={args.del_prior}")
    _log(f"  Backend: {jax.default_backend()}, devices: {jax.devices()}")
    _log(f"  Checkpoint: {checkpoint_path} (every {checkpoint_every} files)")
    _log(f"{'='*60}\n")

    while em_iter < n_iter:
        iter_start = time.monotonic()
        _log(f"--- EM iteration {em_iter + 1}/{n_iter} ---")
        _report_params(params, n_dom)

        # Per-pair chi is built inside `_process_file_pairs` at each
        # pair's own t_est (coherent with per-pair-t_est emissions);
        # no global chi at t_rep is needed here.

        # ---- E-step ----
        # On the first iteration with --budget-hours and no frozen subset yet,
        # sample families randomly, time each, and stop when the cumulative
        # E-step time exceeds budget_hours/n_iter. Freeze the subset for
        # subsequent iterations. On all other iterations, iterate over the
        # (possibly frozen) active_files in order.

        is_budget_sampling = (budget_hours is not None
                              and budget_files is None
                              and em_iter == 0)

        if precompiled_source is not None:
            # Precompiled mode: skip iter_files setup
            iter_files = []
        elif is_budget_sampling and args.interleaved:
            # Interleaved budget sampling with LAZY manifest building.
            # Shuffles family order, visits families round-robin, builds
            # cherry pairs on-the-fly for each family the first time it's
            # visited. Takes one pair per family per round. Stops when
            # budget is hit. Freezes the exact (fi, pair) list for
            # subsequent iterations.
            rng = np.random.RandomState(args.seed)
            budget_limit_s = budget_hours * 3600.0 / n_iter

            family_order = list(all_active_files)
            rng.shuffle(family_order)

            # Lazy cache: build StoIndex + pairs for a family only when first visited
            from tkfmixdom.jax.util.sto_index import StoIndex
            sto_cache = {}   # fi -> StoIndex (fast random access)
            pairs_cache = {} # fi -> list of (row1, row2, t_est)
            def _get_index(fi):
                if fi not in sto_cache:
                    sto_cache[fi] = StoIndex(msa_files[fi])
                return sto_cache[fi]
            def _get_pairs(fi):
                if fi not in pairs_cache:
                    idx = _get_index(fi)
                    pairs_cache[fi] = _build_pairs_for_file(fi, msa_files[fi], sto_index=idx)
                    pairs_by_file[fi] = pairs_cache[fi]
                return pairs_cache[fi]

            _log(f"  Interleaved budget sampling: {len(family_order)} families, "
                 f"limit {budget_limit_s:.0f}s (lazy manifest)")

            iter_files = None  # signal to use interleaved pair-by-pair loop
        elif is_budget_sampling:
            # Sequential budget sampling (old behavior)
            rng = np.random.RandomState(args.seed)
            sampling_order = list(all_active_files)
            rng.shuffle(sampling_order)
            budget_limit_s = budget_hours * 3600.0 / n_iter
            _log(f"  Budget sampling: shuffled {len(sampling_order)} families, "
                 f"limit {budget_limit_s:.0f}s")
            iter_files = sampling_order
        else:
            iter_files = active_files

        e_step_start = time.monotonic()
        per_family_times = []

        if precompiled_source is not None:
            # ---- Precompiled E-step ----
            # Iterate through all pre-decoded pairs grouped by family.
            fam_count = 0
            for source_id, _pairs_for_file, pre_decoded in precompiled_source.iter_pairs(seed=args.seed):
                if not pre_decoded:
                    continue
                fam_count += 1
                fam_start = time.monotonic()
                result = _process_file_pairs(
                    source_id, [], params, st,
                    Q_lg, pi_lg, N, alignment_to_states,
                    dom_Qs=params.get('dom_Qs'), dom_pis=params.get('dom_pis'),
                    n_dom=n_dom, n_frag=n_frag,
                    pre_decoded=pre_decoded)
                fam_elapsed = time.monotonic() - fam_start
                if result is not None:
                    for k in suff_stats:
                        if isinstance(suff_stats[k], np.ndarray):
                            suff_stats[k] = suff_stats[k] + result[k]
                        else:
                            suff_stats[k] = suff_stats[k] + result[k]
                step_file_idx = fam_count

                if fam_count <= 3 or fam_count % max(1, precompiled_source.n_sources // 20) == 0:
                    elapsed = time.monotonic() - e_step_start
                    _log(f"  E-step: {fam_count}/{precompiled_source.n_sources} families, "
                         f"{int(suff_stats['n_pairs'])} pairs, "
                         f"LL={suff_stats['total_ll']:.1f} [{source_id}] "
                         f"{elapsed:.1f}s")

                if checkpoint_every > 0 and fam_count % checkpoint_every == 0:
                    _save_checkpoint(checkpoint_path, params, suff_stats,
                                     step_file_idx, em_iter, log_probs, config,
                                     manifest, t_rep, budget_files, family_timings)

        elif is_budget_sampling and args.interleaved:
            # ---- Interleaved budget E-step (lazy) ----
            # Visit families round-robin, build cherry pairs on-the-fly,
            # take one pair per family per round. Freeze exact pair list.
            frozen_pairs = []
            pair_cursors = {}  # fi -> next pair index to use

            def _process_one_pair(fi, pair):
                """Process a single pair from one file using cached StoIndex."""
                fam_start = time.monotonic()
                idx = _get_index(fi)
                result = _process_file_pairs(
                    msa_files[fi], [pair], params, st,
                    Q_lg, pi_lg, N, alignment_to_states,
                    dom_Qs=params.get('dom_Qs'), dom_pis=params.get('dom_pis'),
                    n_dom=n_dom, n_frag=n_frag,
                    sto_index=idx)
                elapsed = time.monotonic() - fam_start
                per_family_times.append(elapsed)
                if result is not None:
                    for k in suff_stats:
                        if isinstance(suff_stats[k], np.ndarray):
                            suff_stats[k] = suff_stats[k] + result[k]
                        else:
                            suff_stats[k] = suff_stats[k] + result[k]

            # Iterate through shuffled family list, lazily building pairs.
            # Take one pair from each family, then cycle back for seconds.
            # Each family's cherry selection happens only when first visited.
            budget_hit = False
            family_idx = 0  # position in family_order
            _log(f"  Starting lazy sampling over {len(family_order)} families...")

            while not budget_hit and family_idx < len(family_order):
                fi = family_order[family_idx]
                family_idx += 1

                # Lazily build pairs for this family
                pairs = _get_pairs(fi)
                if not pairs:
                    continue

                cursor = pair_cursors.get(fi, 0)
                if cursor >= len(pairs):
                    continue  # exhausted this family
                pair = pairs[cursor]
                pair_cursors[fi] = cursor + 1

                _process_one_pair(fi, pair)
                frozen_pairs.append((fi, pair))

                # Progress
                if len(frozen_pairs) <= 3 or len(frozen_pairs) % 20 == 0:
                    elapsed = time.monotonic() - e_step_start
                    n_fams = len(pair_cursors)
                    _log(f"  E-step: {len(frozen_pairs)} pairs, "
                         f"{n_fams} families, "
                         f"LL={suff_stats['total_ll']:.1f} "
                         f"| budget: {elapsed:.1f}/{budget_limit_s:.0f}s")

                # Budget cutoff
                if time.monotonic() - e_step_start >= budget_limit_s:
                    budget_hit = True

            n_fams = len(set(fp[0] for fp in frozen_pairs))
            if budget_hit:
                _log(f"\n  *** Budget limit reached: "
                     f"{time.monotonic()-e_step_start:.1f}s >= {budget_limit_s:.0f}s")
            else:
                _log(f"\n  *** All pairs exhausted in {round_num} rounds")
            _log(f"  *** Frozen: {len(frozen_pairs)} pairs from {n_fams} families")
            budget_files = frozen_pairs
            active_files = sorted(set(fp[0] for fp in frozen_pairs))
            family_timings = per_family_times
            # Update t_rep from frozen pairs
            t_vals = [p[2] for _, p in frozen_pairs]
            if t_vals:
                t_rep = float(np.median(t_vals))
                t = t_rep
                _log(f"  Median t from frozen pairs: {t_rep:.3f}")

        elif is_budget_sampling:
            # ---- Sequential budget E-step (old behavior) ----
            for af_pos in range(step_file_idx, len(iter_files)):
                fi = iter_files[af_pos]
                msa_file = msa_files[fi]
                fam_start = time.monotonic()
                result = _process_file_pairs(
                    msa_file, pairs_by_file[fi], params, st,
                    Q_lg, pi_lg, N, alignment_to_states,
                    dom_Qs=params.get('dom_Qs'), dom_pis=params.get('dom_pis'),
                    n_dom=n_dom, n_frag=n_frag)
                fam_elapsed = time.monotonic() - fam_start
                if result is not None:
                    for k in suff_stats:
                        if isinstance(suff_stats[k], np.ndarray):
                            suff_stats[k] = suff_stats[k] + result[k]
                        else:
                            suff_stats[k] = suff_stats[k] + result[k]
                step_file_idx = af_pos + 1
                per_family_times.append(fam_elapsed)
                if step_file_idx % max(1, len(iter_files) // 20) == 0:
                    cum = sum(per_family_times)
                    fam = os.path.basename(msa_file).split('.')[0]
                    _log(f"  E-step: {step_file_idx}/{len(iter_files)} files, "
                         f"{int(suff_stats['n_pairs'])} pairs, "
                         f"LL={suff_stats['total_ll']:.1f} [{fam}] "
                         f"| budget: {cum:.1f}/{budget_limit_s:.0f}s")
                cum_time = sum(per_family_times)
                if cum_time >= budget_limit_s:
                    budget_files = sampling_order[:step_file_idx]
                    family_timings = per_family_times
                    active_files = budget_files
                    _log(f"\n  *** Budget: {step_file_idx} families frozen")
                    break
            else:
                budget_files = sampling_order
                family_timings = per_family_times

        else:
            # ---- Standard E-step (frozen families, all pairs) ----
            n_iter_files = len(iter_files)
            start_pos = step_file_idx
            for af_pos in range(step_file_idx, n_iter_files):
                fi = iter_files[af_pos]
                msa_file = msa_files[fi]

                # For frozen interleaved pairs, only process the frozen subset
                if isinstance(budget_files, list) and budget_files and isinstance(budget_files[0], tuple):
                    file_pairs = [p for ffi, p in budget_files if ffi == fi]
                else:
                    file_pairs = pairs_by_file[fi]

                fam_start = time.monotonic()
                result = _process_file_pairs(
                    msa_file, file_pairs, params, st,
                    Q_lg, pi_lg, N, alignment_to_states,
                    dom_Qs=params.get('dom_Qs'), dom_pis=params.get('dom_pis'),
                    n_dom=n_dom, n_frag=n_frag)
                fam_elapsed = time.monotonic() - fam_start
                if result is not None:
                    for k in suff_stats:
                        if isinstance(suff_stats[k], np.ndarray):
                            suff_stats[k] = suff_stats[k] + result[k]
                        else:
                            suff_stats[k] = suff_stats[k] + result[k]
                step_file_idx = af_pos + 1

                if step_file_idx % max(1, n_iter_files // 20) == 0 or step_file_idx == n_iter_files:
                    elapsed = time.monotonic() - e_step_start
                    done = step_file_idx - start_pos
                    rate = elapsed / max(done, 1)
                    remaining = (n_iter_files - step_file_idx) * rate
                    fam = os.path.basename(msa_file).split('.')[0]
                    _log(f"  E-step: {step_file_idx}/{n_iter_files} files, "
                         f"{int(suff_stats['n_pairs'])} pairs, "
                         f"LL={suff_stats['total_ll']:.1f} [{fam}] "
                         f"{rate:.2f}s/file, "
                         f"{'ETA ' + str(int(remaining)) + 's' if remaining > 0 else 'done'}")

                if checkpoint_every > 0 and step_file_idx % checkpoint_every == 0:
                    _save_checkpoint(checkpoint_path, params, suff_stats,
                                     step_file_idx, em_iter, log_probs, config,
                                     manifest, t_rep, budget_files, family_timings)

        total_ll = suff_stats['total_ll']
        # MAP objective = observed LL + log-prior (at params used in E-step)
        lp_prior = _log_prior(params, args, Q_lg=Q_lg, pi_lg=pi_lg)
        map_ll = total_ll + lp_prior
        log_probs.append(map_ll)
        _log(f"  E-step done: obs_LL={total_ll:.2f}, log_prior={lp_prior:+.2f}, "
             f"MAP_LL={map_ll:.2f}, "
             f"{int(suff_stats['n_pairs'])} pairs from "
             f"{int(suff_stats['n_files'])} files")

        # ---- Save maraschino-compatible counts tensor (optional) ----
        if getattr(args, 'save_counts', None):
            counts_path = args.save_counts
            if em_iter > 0:
                base, ext = os.path.splitext(counts_path)
                counts_path = f"{base}_iter{em_iter}{ext}"
            _log(f"  Saving counts tensor: {counts_path}")
            np.savez(counts_path,
                     B=suff_stats['match_counts'],
                     dom_B=suff_stats['dom_match_counts'],
                     dom_I=suff_stats['dom_insert_counts'],
                     dom_D=suff_stats['dom_delete_counts'],
                     n_chi=suff_stats['agg_n_chi'],
                     n_pairs=int(suff_stats['n_pairs']),
                     n_files=int(suff_stats['n_files']),
                     total_ll=total_ll,
                     t_rep=t,
                     em_iter=em_iter,
                     _format='train_pfam_counts_v1')

        # ---- Exact count restoration via chain ----
        # Restores FB counts to the fully exploded model, then extracts
        # per-parameter BDI sufficient statistics. Verified to match
        # autodiff to 1e-14 precision. See exact_suffstats.py.
        ss = exact_suffstats(
            suff_stats['agg_n_chi'],
            params['main_ins'], params['main_del'], t,
            params['dom_ins'], params['dom_del'],
            params['dom_weights'], params['frag_weights'],
            params['ext_rates'])

        _report_suff_stats_exact(ss, n_dom, n_frag)

        # ---- Log counts and check for anomalies before M-step ----
        top_n = jnp.array(ss['top_5x5'])
        kappa_main = params['main_ins'] / max(params['main_del'], 1e-30)
        _log(f"  Pre-M-step diagnostics:")
        _log(f"    top_5x5:\n{np.array(top_n)}")
        _log(f"    agg_n_chi sum: {suff_stats['agg_n_chi'].sum():.1f}")
        _log(f"    Params: λ={params['main_ins']:.6f}, μ={params['main_del']:.6f}, "
             f"κ={kappa_main:.4f}, λ-μ={params['main_ins']-params['main_del']:.6f}")
        for d in range(n_dom):
            kd = params['dom_ins'][d] / max(params['dom_del'][d], 1e-30)
            _log(f"    dom {d}: λ={params['dom_ins'][d]:.6f}, μ={params['dom_del'][d]:.6f}, "
                 f"κ={kd:.4f}")

        # Check for negative counts in exact_suffstats output
        for key in ['top_5x5']:
            arr = ss[key]
            if np.any(arr < -1e-10):
                _log(f"  *** ANOMALY: negative {key}: min={np.min(arr):.6e}")
        for d in range(n_dom):
            arr = ss['dom_M_5x5'][d]
            if np.any(arr < -1e-10):
                _log(f"  *** ANOMALY: negative dom_M_5x5[{d}]: min={np.min(arr):.6e}")
            if ss['dom_kappa'][d] < -1e-10:
                _log(f"  *** ANOMALY: negative dom_kappa[{d}]={ss['dom_kappa'][d]:.6e}")
        if np.any(suff_stats['agg_n_chi'] < -1e-10):
            _log(f"  *** ANOMALY: negative agg_n_chi: min={suff_stats['agg_n_chi'].min():.6e}")

        # ---- Exact M-steps (all closed-form, matching compiled.py) ----
        # Top-level indel rates via quadratic M-step (eq:kappa-quadratic)
        top_L = float(jnp.sum(top_n[:, M]) + jnp.sum(top_n[:, D]))
        top_M = float(jnp.sum(top_n[:, E]))
        top_T = t * top_M  # T = t × M (one BDI process per pair, M_top=1 per pair)
        E_B, E_D, E_S = bdi_stats_from_counts(
            top_n, params['main_ins'], params['main_del'], t, T=top_T)
        _log(f"    BDI: E_B={E_B:.4f}, E_D={E_D:.4f}, E_S={E_S:.4f}")
        _log(f"    L={top_L:.4f}, M={top_M:.4f}, T={top_T:.4f}")
        if E_S < 0:
            _log(f"  *** ANOMALY: E_S={E_S:.4f} < 0 at top level")
            _log(f"    Conservation: E_B-E_D={E_B-E_D:.4f}, "
                 f"I_col-D_col={float(jnp.sum(top_n[:,I])-jnp.sum(top_n[:,D])):.4f}")
            # Log WFST score breakdown
            from tkfmixdom.jax.core.bdi import _wfst_scores, transition_count_groups, score_derivatives
            groups = transition_count_groups(top_n)
            lam_s, mu_s = _wfst_scores(groups, params['main_ins'], params['main_del'], t)
            eps = params['main_ins'] - params['main_del']
            cons = float(jnp.sum(top_n[:, I]) - jnp.sum(top_n[:, D]))
            numer = cons + float(mu_s) - float(lam_s) - params['main_ins'] * t
            _log(f"    WFST breakdown: lam_score={float(lam_s):.6f}, mu_score={float(mu_s):.6f}")
            _log(f"    cons={cons:.4f}, eps={eps:.6f}, numer={numer:.6f}")
            _log(f"    E_S = numer/eps = {numer:.6f}/{eps:.6f} = {numer/eps:.4f}")
            derivs = score_derivatives(params['main_ins'], params['main_del'], t)
            for name, count in groups.items():
                if abs(count) < 1e-10: continue
                dl, dm = derivs[name]
                _log(f"      {name}: n={count:.4f}, d_λ={float(dl):.8f}, d_μ={float(dm):.8f}")
        if E_B < 0:
            _log(f"  *** ANOMALY: E_B={E_B:.4f} < 0 at top level")
        if E_D < 0:
            _log(f"  *** ANOMALY: E_D={E_D:.4f} < 0 at top level")

        main_ins_new, main_del_new = m_step_indel_quadratic(
            float(E_B), float(E_D), float(E_S),
            L=top_L, M=top_M, T=top_T,
            prior_alpha_lam=args.ins_prior[0], prior_alpha_mu=args.del_prior[0],
            prior_beta=args.ins_prior[1])
        main_ins_new = float(main_ins_new) if np.isfinite(main_ins_new) \
            else float(params['main_ins'])
        main_del_new = float(main_del_new) if np.isfinite(main_del_new) \
            else float(params['main_del'])
        _log(f"    M-step result: λ_new={main_ins_new:.6f}, μ_new={main_del_new:.6f}")

        # Per-domain indel rates via quadratic M-step
        dom_ins_new = np.array(params['dom_ins'], dtype=float)
        dom_del_new = np.array(params['dom_del'], dtype=float)
        for d in range(n_dom):
            dom_n = jnp.array(ss['dom_M_5x5'][d])
            nk_ID = ss['dom_kappa'][d]
            n1k_ID = ss['dom_1mkappa'][d]
            if float(dom_n.sum()) < 0.01 and nk_ID < 0.01:
                continue
            # T^(k) = t × (M-type entries + I/D-type entries)
            n_entries_M = float(jnp.sum(dom_n[S, :]))
            T_d = t * (n_entries_M + n1k_ID)
            eb, ed, es = bdi_stats_from_counts(
                dom_n, params['dom_ins'][d], params['dom_del'][d], t, T=T_d)
            # L = ALL kappa events: M-type (M+D col sums) + I/D-type
            nk_M = float(jnp.sum(dom_n[:, M]) + jnp.sum(dom_n[:, D]))
            n1k_M = float(jnp.sum(dom_n[:, E]))
            _log(f"    dom {d} BDI: E_B={eb:.4f}, E_D={ed:.4f}, E_S={es:.4f}, "
                 f"L={nk_M+nk_ID:.4f}, M={n1k_M+n1k_ID:.4f}, T={T_d:.4f}")
            if es < 0:
                _log(f"  *** ANOMALY: E_S={es:.4f} < 0 for domain {d}")
            if eb < 0:
                _log(f"  *** ANOMALY: E_B={eb:.4f} < 0 for domain {d}")
            if ed < 0:
                _log(f"  *** ANOMALY: E_D={ed:.4f} < 0 for domain {d}")
            ni, nd = m_step_indel_quadratic(
                float(eb), float(ed), float(es),
                L=nk_M + nk_ID, M=n1k_M + n1k_ID, T=T_d,
                prior_alpha_lam=args.ins_prior[0], prior_alpha_mu=args.del_prior[0],
                prior_beta=args.ins_prior[1])
            ni = ni if np.isfinite(ni) else params['dom_ins'][d]
            nd = nd if np.isfinite(nd) else params['dom_del'][d]
            dom_ins_new[d] = float(ni)
            dom_del_new[d] = float(nd)
            _log(f"    dom {d} M-step: λ_new={dom_ins_new[d]:.6f}, μ_new={dom_del_new[d]:.6f}")

        # Domain weights: Dirichlet MAP (counts + alpha - 1, normalized)
        dom_alpha = args.dom_dirichlet
        dom_w_counts = np.array(ss['dom_w'])
        dom_w_post = np.maximum(dom_w_counts + dom_alpha - 1, 0)
        dom_w_total = dom_w_post.sum()
        if dom_w_total > 1e-10:
            dom_weights_new = dom_w_post / dom_w_total
        else:
            dom_weights_new = np.ones(n_dom) / n_dom
        _log(f"    dom_weights: counts={dom_w_counts} → new={dom_weights_new}")

        # Fragment weights: Dirichlet MAP per domain
        frag_alpha = args.frag_dirichlet
        frag_weights_new = np.zeros((n_dom, n_frag))
        if getattr(args, 'banded_frag_init', False):
            # Banded mode: pin frag_weights at init ([1, 0, 0]) by skipping
            # the M-step. Pseudocounts on all entries are zero; FB-restored
            # counts on f>0 are vanishing (chi forbids start at f>0).
            frag_weights_new = np.array(params['frag_weights']).copy()
            _log(f"    frag_weights: PINNED at banded init "
                 f"(frag_weights[d,0]=1)")
        else:
            for d in range(n_dom):
                fw_post = np.maximum(np.array(ss['frag_w'][d]) + frag_alpha - 1, 0)
                fw_total = fw_post.sum()
                if fw_total > 1e-10:
                    frag_weights_new[d] = fw_post / fw_total
                else:
                    frag_weights_new[d] = np.ones(n_frag) / n_frag
            _log(f"    frag_weights: {[ss['frag_w'][d].tolist() for d in range(n_dom)]}")

        # Fragment extension: MixDom2 row-normalized Dirichlet MAP
        # ss['ext'] is (D, F, F) transition counts, ss['term'] is (D, F) termination counts
        # Note: to keep ext diagonal (MixDom1 equivalence at n_frag>1),
        # initialise ext diagonal AND set --ext-alpha=1 --ext-beta=1 so
        # off-diagonal pseudocount is zero; zero counts then stay zero.
        freeze_offdiag = bool(getattr(args, 'freeze_ext_offdiag', False))
        banded = bool(getattr(args, 'banded_frag_init', False))
        if banded:
            from tkfmixdom.jax.train.restricted_mstep import (
                banded_3fc_pseudocounts, banded_3fc_ext_mask)
            ext_pseudo_2d, term_pseudo_1d = banded_3fc_pseudocounts(
                n_frag, args.ext_alpha, args.ext_beta)
            ext_mask_2d, term_mask_1d = banded_3fc_ext_mask()
        ext_rates_new = np.zeros((n_dom, n_frag, n_frag))
        for d in range(n_dom):
            for f in range(n_frag):
                # Row f: counts for transitions to each fragment g, plus termination
                row_ext = ss['ext'][d, f, :]  # (F,) transition counts
                term = ss['term'][d, f]       # scalar termination count
                row_counts = np.append(row_ext, term)
                # Dirichlet pseudocounts
                pseudocounts = np.append(
                    np.full(n_frag, args.ext_alpha - 1.0),
                    args.ext_beta - 1.0)
                if banded:
                    # Banded mask: structurally-zero entries get 0
                    # pseudocount, the rest get ext_alpha-1 / ext_beta-1.
                    pseudocounts[:n_frag] = ext_pseudo_2d[f, :]
                    pseudocounts[n_frag] = term_pseudo_1d[f]
                elif freeze_offdiag:
                    # Zero off-diagonal pseudocounts so the prior cannot push
                    # off-diag mass above zero. Combined with diagonal init,
                    # any FB-restored off-diag counts are vanishing
                    # (ext_rates[d, f, g≠f]=0 ⇒ chi forbids the transition
                    # up to the 1e-30 log-floor), and the M-step normalises
                    # them to ~0. Off-diag stays strictly at 0 through EM.
                    pseudocounts[:n_frag] = 0.0
                    pseudocounts[f] = args.ext_alpha - 1.0
                posterior = np.maximum(row_counts + pseudocounts, 0.0)
                if banded:
                    # Force structural zeros to remain exactly zero, even
                    # if FB restored ε-floor counts on a forbidden entry.
                    posterior[:n_frag] = np.where(
                        ext_mask_2d[f, :], posterior[:n_frag], 0.0)
                    if not term_mask_1d[f]:
                        posterior[n_frag] = 0.0
                total = posterior.sum()
                if total > 1e-10:
                    normalized = posterior / total
                    # Termination prob (the (n_frag)-th entry, sliced off
                    # below) must remain strictly positive — if it is 0
                    # the geometric fragment-length distribution is
                    # degenerate (infinite-length fragment with prob 1).
                    # Surface that as a hard failure rather than clipping
                    # silently, so the user fixes the prior (--ext-beta > 1)
                    # or the upstream data instead of getting biased output.
                    # Banded mode permits structurally-zero termination
                    # (FragMid: term_mask=False) so we exempt those rows.
                    if not (banded and not term_mask_1d[f]):
                        assert normalized[n_frag] > 1e-12, (
                            f"ext_rates M-step at (d={d}, f={f}) produced "
                            f"termination prob = {normalized[n_frag]:.3e} "
                            f"(ext_counts={row_ext}, term_count={term:g}, "
                            f"ext_beta={args.ext_beta}). Increase --ext-beta "
                            f"above 1 or check for degenerate data.")
                    ext_rates_new[d, f, :] = normalized[:n_frag]
                else:
                    if banded:
                        # Preserve banded-init row when no data hit it.
                        ext_rates_new[d, f, :] = np.array(
                            params['ext_rates'])[d, f, :]
                    else:
                        ext_rates_new[d, f, f] = 0.3  # default: diagonal
        _log(f"    ext: counts={ss['ext'].tolist()}, term={ss['term'].tolist()}")
        _log(f"    ext_new={ext_rates_new.tolist()}")

        old_params = params  # preserve dynamic class params
        params = {
            'main_ins': main_ins_new, 'main_del': main_del_new,
            'dom_ins': np.array(dom_ins_new), 'dom_del': np.array(dom_del_new),
            'dom_weights': np.array(dom_weights_new),
            'frag_weights': np.array(frag_weights_new),
            'ext_rates': np.array(ext_rates_new),
        }
        # Carry forward per-domain subst + MixDom2 site class params
        for _k in ('dom_S_exch', 'dom_Qs', 'dom_pis',
                    'n_classes', 'classdist', 'class_pis', 'class_S_exch'):
            if _k in old_params:
                params[_k] = old_params[_k]

        # Per-domain substitution M-step (paper option 1, lines 655-658)
        if getattr(args, 'estimate_subst', False):
            from tkfmixdom.jax.core.ctmc import (
                m_step_subst_option1, holmes_rubin_expected_stats)
            S_lg = np.array(Q_lg / np.maximum(np.array(pi_lg)[None, :], 1e-30)
                            * (1.0 - np.eye(AA)))
            S_lg = (S_lg + S_lg.T) / 2  # symmetrize

            dom_Qs_new = np.zeros((n_dom, AA, AA))
            dom_pis_new = np.zeros((n_dom, AA))
            dom_S_new = np.zeros((n_dom, AA, AA))
            for dd in range(n_dom):
                mc = suff_stats['dom_match_counts'][dd]
                ic = suff_stats['dom_insert_counts'][dd]
                dc = suff_stats['dom_delete_counts'][dd]
                total_mc = mc.sum()

                if total_mc < 1.0:
                    # Not enough data — keep current
                    dom_Qs_new[dd] = params.get('dom_Qs', np.tile(Q_lg, (n_dom,1,1)))[dd]
                    dom_pis_new[dd] = params.get('dom_pis', np.tile(pi_lg, (n_dom,1)))[dd]
                    dom_S_new[dd] = params.get('dom_S_exch', np.tile(S_lg[None], (n_dom,1,1)))[dd]
                    continue

                # V_i = match-anc + insert + delete character counts under
                # the joint pair HMM (tkf.tex L755): every observable
                # equilibrium-pi draw of character i. Match-position
                # descendants enter via U column-sum inside
                # m_step_subst_option1, NOT raw V.
                V_d = mc.sum(axis=1) + ic + dc

                # W, U from Holmes-Rubin expected stats weighted by match counts
                Q_old = params.get('dom_Qs', np.tile(Q_lg, (n_dom,1,1)))[dd]
                pi_old = params.get('dom_pis', np.tile(pi_lg, (n_dom,1)))[dd]
                W_d = np.zeros(AA)
                U_d = np.zeros((AA, AA))
                for a in range(AA):
                    for b in range(AA):
                        if mc[a, b] < 1e-10:
                            continue
                        w_ab, u_ab = holmes_rubin_expected_stats(
                            jnp.array(Q_old), jnp.array(pi_old), t, a, b)
                        W_d += mc[a, b] * np.array(w_ab)
                        U_d += mc[a, b] * np.array(u_ab)

                _subst_mode = getattr(args, 'subst_mode', 'standard')
                S_old_d = (np.asarray(params['dom_S_exch'])[dd]
                           if 'dom_S_exch' in params else S_lg)
                if _subst_mode == 'rescaling-rates' or (
                    _subst_mode == 'alt-tied-pi-rescaling'
                    and (em_iter % 2 == 1)
                ):
                    from tkfmixdom.jax.train.restricted_mstep import (
                        m_step_subst_rescaling)
                    S_d, pi_d, _sigma = m_step_subst_rescaling(
                        W_d, U_d, S_old_d, pi_old)
                    Q_d = S_d * pi_d[None, :]
                    np.fill_diagonal(Q_d, 0.0)
                    Q_d[np.diag_indices(AA)] = -Q_d.sum(axis=1)
                elif _subst_mode == 'rescaling-rates-and-pi':
                    from tkfmixdom.jax.train.restricted_mstep import (
                        m_step_subst_rescaling_pi)
                    sigma_warm = (float(np.linalg.norm(S_old_d.ravel())
                                       / max(np.linalg.norm(S_lg.ravel()), 1e-30)))
                    S_d, pi_d, _sigma = m_step_subst_rescaling_pi(
                        W_d, U_d, V_d, S_old_d, sigma_warm, pi_old)
                    Q_d = S_d * pi_d[None, :]
                    np.fill_diagonal(Q_d, 0.0)
                    Q_d[np.diag_indices(AA)] = -Q_d.sum(axis=1)
                elif _subst_mode == 'frozen-pi':
                    S_d, _pi_drop, _ = m_step_subst_option1(
                        W_d, U_d, V_d,
                        S_prior=S_lg, pi_prior=np.array(pi_lg),
                        pi_pseudo=args.pi_pseudo, S_pseudo=args.S_pseudo)
                    pi_d = pi_old
                    Q_d = S_d * pi_d[None, :]
                    np.fill_diagonal(Q_d, 0.0)
                    Q_d[np.diag_indices(AA)] = -Q_d.sum(axis=1)
                else:
                    S_d, pi_d, Q_d = m_step_subst_option1(
                        W_d, U_d, V_d,
                        S_prior=S_lg, pi_prior=np.array(pi_lg),
                        pi_pseudo=args.pi_pseudo, S_pseudo=args.S_pseudo)

                dom_Qs_new[dd] = Q_d
                dom_pis_new[dd] = pi_d
                dom_S_new[dd] = S_d
                _log(f"    dom {dd} subst: {total_mc:.0f} match counts, "
                     f"pi entropy={float(-np.sum(pi_d * np.log(pi_d + 1e-30))):.3f}")

            params['dom_Qs'] = dom_Qs_new
            params['dom_pis'] = dom_pis_new
            params['dom_S_exch'] = dom_S_new

        # ---- Next iteration ----
        em_iter += 1
        step_file_idx = 0
        suff_stats = _zero_suff_stats(N, n_dom)

        iter_time = time.monotonic() - iter_start
        _log(f"  Iteration done in {iter_time:.1f}s")

        if len(log_probs) >= 2:
            delta = log_probs[-1] - log_probs[-2]
            _log(f"  LL change: {delta:+.4f}")
            if abs(delta) < args.convergence_tol:
                _log(f"  Converged (|delta| < {args.convergence_tol})")
                em_iter = n_iter

        _save_checkpoint(checkpoint_path, params, suff_stats, step_file_idx,
                         em_iter, log_probs, config, manifest, t_rep,
                         budget_files, family_timings)
        _log(f"  [checkpoint saved]")

    # ---- Final report ----
    _log(f"\n{'='*60}")
    _log(f"Training complete: {len(log_probs)} iterations")
    if log_probs:
        _log(f"Final MAP LL: {log_probs[-1]:.2f}")
        _log(f"MAP LL history: {['%.2f' % x for x in log_probs]}")
    _log(f"Model saved to: {checkpoint_path}")
    _report_params(params, n_dom)
    _log(f"{'='*60}")


def _report_budget_summary(per_family_times, n_iter, n_total_families):
    """Report N(H) estimates for standard budget values based on observed timings."""
    times = np.array(per_family_times)
    n_sampled = len(times)
    cum_time = times.sum()
    mean_time = times.mean()
    _log(f"\n  Budget sampling summary:")
    _log(f"    Families sampled: {n_sampled} / {n_total_families}")
    _log(f"    Cumulative E-step time: {cum_time:.1f}s")
    _log(f"    Mean time per family: {mean_time:.4f}s")
    _log(f"    Median time per family: {np.median(times):.4f}s")
    _log(f"    Std time per family: {np.std(times):.4f}s")
    _log(f"    Min/Max: {times.min():.4f}s / {times.max():.4f}s")

    # Estimate N(H) for standard budget values
    _log(f"\n    Estimated N(H) for K={n_iter} iterations:")
    _log(f"    {'H (hours)':>10}  {'Budget/iter (s)':>15}  "
         f"{'N(H) families':>14}  {'N(H) pairs':>10}")
    _log(f"    {'─'*10}  {'─'*15}  {'─'*14}  {'─'*10}")
    for H in [1, 2, 4, 8, 16, 24]:
        budget_s = H * 3600.0 / n_iter
        # Estimate: how many families fit in budget_s at observed mean rate?
        n_est = min(int(budget_s / mean_time), n_total_families)
        # Also compute from cumulative: how many of the sampled families
        # would have fit if we stopped at budget_s?
        cum = np.cumsum(times)
        n_actual = int(np.searchsorted(cum, budget_s, side='right'))
        n_actual = min(n_actual, n_sampled)
        marker = " <-- active" if abs(budget_s - cum_time) < cum_time * 0.1 else ""
        _log(f"    {H:>10}  {budget_s:>15.0f}  "
             f"{n_est:>14}  (sampled: {n_actual}){marker}")
    _log("")


def _evaluate_on_split(params, split_name, split_fams, msa_dir, n_dom, n_frag,
                       t, Q_lg, pi_lg, max_pairs_per_fam=2, unaligned=False,
                       match_aligned=False):
    """Evaluate current params on a split (val/test). Returns (total_ll, n_pairs, ll_per_pair).

    Walks ALL families in the split in deterministic (sorted) order, builds
    cherry pairs, runs constrained 1D FB with current params to get
    log P(x,y|params). No M-step. The val/test pair set is thus identical
    across training iterations and across training runs, which is required
    for reliable early stopping and cross-run comparison.

    Args:
        params: current MixDom param dict
        split_name: 'val' or 'test'
        split_fams: set of family IDs in this split
        msa_dir: directory with .sto/.sto.gz files
        n_dom, n_frag: model dimensions
        t: evolutionary time
        Q_lg, pi_lg: LG08 rate matrix and equilibrium
        max_pairs_per_fam: max cherry pairs per family (default 2)

    Returns:
        (total_ll, n_pairs, ll_per_pair)
    """
    import jax.numpy as jnp
    import glob as glob_mod
    from tkfmixdom.jax.models.mixdom import (
        n_states as mixdom_n_states,
        state_types as mixdom_state_types
    )
    from tkfmixdom.jax.simulate.msa import alignment_to_states
    from tkfmixdom.jax.util.sto_index import StoIndex

    N = mixdom_n_states(n_dom, n_frag)
    st = mixdom_state_types(n_dom, n_frag)

    # No global log_chi here: all three eval paths (1D, 2D unaligned,
    # match-aligned) build chi PER PAIR at each pair's own t_est for
    # coherence with per-pair-t_est emissions. See `_build_log_chi_stack`
    # and the per-chi vmapped FB / forward-only kernels.
    # `t` (kwarg, kept for back-compat) is unused in this scope.

    # Find val MSA files
    all_sto = sorted(
        glob_mod.glob(os.path.join(msa_dir, '*.sto')) +
        glob_mod.glob(os.path.join(msa_dir, '*.sto.gz')))
    val_files = [f for f in all_sto
                 if os.path.basename(f).split('.')[0] in split_fams]

    if not val_files:
        _log(f"  Val eval: no {split_name} files found in {msa_dir}")
        return 0.0, 0, 0.0

    eval_start = time.monotonic()

    # Deterministic order: sorted by filename. No shuffle, no time budget.
    # This guarantees identical val pair set across iterations and runs.
    val_files = sorted(val_files)

    # Collect all val pairs, then batch the FB for throughput
    all_val_pairs = []
    for vf in val_files:
        try:
            idx = StoIndex(vf)
        except Exception:
            continue
        seqs = [idx.get_sequence(i) for i in range(len(idx))]
        if not seqs:
            continue
        pairs_raw = _build_pairs_for_file(0, vf, sto_index=idx)
        if not pairs_raw:
            continue
        for row1, row2, t_est in pairs_raw[:max_pairs_per_fam]:
            if row1 >= len(seqs) or row2 >= len(seqs):
                continue
            aln_i, aln_j = _aligned_pair_to_int_arrays(seqs[row1], seqs[row2])
            if len(aln_i) == 0:
                continue
            for anc_aln, desc_aln in [(aln_i, aln_j), (aln_j, aln_i)]:
                states, anc_chars, desc_chars = alignment_to_states(anc_aln, desc_aln)
                if states:
                    x_int = np.array([int(c) for c in anc_aln if c >= 0])
                    y_int = np.array([int(c) for c in desc_aln if c >= 0])
                    all_val_pairs.append(
                        (x_int, y_int, states, anc_chars, desc_chars, t_est))

    if not all_val_pairs:
        _log(f"  Val eval ({split_name}): no pairs found")
        return 0.0, 0, 0.0

    # Batched forward-only over all val pairs (no backward — only need log P)
    from tkfmixdom.jax.core.ctmc import transition_matrix
    from tkfmixdom.jax.dp.hmm import NEG_INF
    from tkfmixdom.jax.core.params import M as _M, I as _I, D as _D

    per_domain_subst = params.get('dom_Qs') is not None and params.get('dom_pis') is not None

    if unaligned:
        # 2D forward-only: run pair HMM on raw sequences. Per-pair t coherence:
        # chi and sub_matrices are both built at each pair's own t_est (analog
        # of the 1D per-chi val fix).
        from collections import defaultdict

        buckets_2d = defaultdict(list)  # (Lx_pad, Ly_pad) -> [(x, y, Lx, Ly, t_est)]
        for item in all_val_pairs:
            x_int, y_int, states, anc_chars, desc_chars, t_est = item
            x = np.asarray(x_int, dtype=np.int32)
            y = np.asarray(y_int, dtype=np.int32)
            Lx_pad = _pad_to_bin(len(x))
            Ly_pad = _pad_to_bin(len(y))
            buckets_2d[(Lx_pad, Ly_pad)].append((x, y, len(x), len(y), t_est))

        fwd_2d_per_chi = _get_fwd_2d_per_chi_jit()

        sub_params = {'dom_Qs': params.get('dom_Qs'),
                      'dom_pis': params.get('dom_pis')} if per_domain_subst else {}

        total_ll = 0.0
        n_pairs = 0
        st_j = jnp.array(st)
        for (Lx_pad, Ly_pad), bucket in buckets_2d.items():
            B = len(bucket)
            xs = np.zeros((B, Lx_pad), dtype=np.int32)
            ys = np.zeros((B, Ly_pad), dtype=np.int32)
            real_Lxs = np.zeros(B, dtype=np.int32)
            real_Lys = np.zeros(B, dtype=np.int32)
            for i, (x, y, lx, ly, te) in enumerate(bucket):
                xs[i, :lx] = x
                ys[i, :ly] = y
                real_Lxs[i] = lx
                real_Lys[i] = ly

            # Per-pair t_est (item[4] is t_est)
            t_ests = np.array([item[4] for item in bucket], dtype=np.float32)

            # Per-pair sub matrices stack and per-pair log_chi stack
            sub_matrices_stack, pis_arr = _build_per_pair_sub_matrices(
                t_ests, sub_params, Q_lg, pi_lg, n_dom)
            t_ests_j = jnp.asarray(t_ests)
            log_chi_stack = _build_log_chi_stack(params, t_ests_j)

            # Chunk to avoid OOM on large buckets
            CHUNK_2D = 20
            for c_start in range(0, B, CHUNK_2D):
                c_end = min(c_start + CHUNK_2D, B)
                log_probs = fwd_2d_per_chi(
                    log_chi_stack[c_start:c_end], st_j,
                    jnp.array(xs[c_start:c_end]),
                    jnp.array(ys[c_start:c_end]),
                    jnp.array(real_Lxs[c_start:c_end]),
                    jnp.array(real_Lys[c_start:c_end]),
                    jnp.array(sub_matrices_stack[c_start:c_end]),
                    jnp.array(pis_arr),
                    n_dom, n_frag)
                log_probs_np = np.asarray(log_probs)
                for i in range(c_end - c_start):
                    if not np.isnan(log_probs_np[i]):
                        total_ll += float(log_probs_np[i])
                        n_pairs += 1
    elif match_aligned:
        # Match-aligned 2D forward-only: constrain match states to alignment
        # positions. Per-pair t coherence: chi and sub_matrices are both
        # built at each pair's own t_est (analog of the 1D per-chi val fix).
        from collections import defaultdict

        buckets_ma = defaultdict(list)
        for item in all_val_pairs:
            x_int, y_int, states, anc_chars, desc_chars, t_est = item
            x = np.asarray(x_int, dtype=np.int32)
            y = np.asarray(y_int, dtype=np.int32)
            mi, mj = _extract_match_positions(states)
            Lx_pad = _pad_to_bin(len(x))
            Ly_pad = _pad_to_bin(len(y))
            buckets_ma[(Lx_pad, Ly_pad)].append(
                (x, y, len(x), len(y), t_est, mi, mj))

        fwd_ma_per_chi = _get_fwd_ma_per_chi_jit()

        sub_params = {'dom_Qs': params.get('dom_Qs'),
                      'dom_pis': params.get('dom_pis')} if per_domain_subst else {}

        total_ll = 0.0
        n_pairs = 0
        st_j = jnp.array(st)
        for (Lx_pad, Ly_pad), bucket in buckets_ma.items():
            B = len(bucket)
            xs = np.zeros((B, Lx_pad), dtype=np.int32)
            ys = np.zeros((B, Ly_pad), dtype=np.int32)
            real_Lxs = np.zeros(B, dtype=np.int32)
            real_Lys = np.zeros(B, dtype=np.int32)
            max_n_match = max(len(item[5]) for item in bucket)
            if max_n_match == 0:
                max_n_match = 1
            match_is_arr = np.zeros((B, max_n_match), dtype=np.int32)
            match_js_arr = np.zeros((B, max_n_match), dtype=np.int32)
            match_lens_arr = np.zeros(B, dtype=np.int32)
            for i, (x, y, lx, ly, te, mi, mj) in enumerate(bucket):
                xs[i, :lx] = x
                ys[i, :ly] = y
                real_Lxs[i] = lx
                real_Lys[i] = ly
                n_m = len(mi)
                match_is_arr[i, :n_m] = mi
                match_js_arr[i, :n_m] = mj
                match_lens_arr[i] = n_m

            # Per-pair t_est (item[4] is t_est)
            t_ests = np.array([item[4] for item in bucket], dtype=np.float32)

            # Per-pair sub matrices stack and per-pair log_chi stack
            sub_matrices_stack, pis_arr = _build_per_pair_sub_matrices(
                t_ests, sub_params, Q_lg, pi_lg, n_dom)
            t_ests_j = jnp.asarray(t_ests)
            log_chi_stack = _build_log_chi_stack(params, t_ests_j)

            CHUNK_MA = 20
            for c_start in range(0, B, CHUNK_MA):
                c_end = min(c_start + CHUNK_MA, B)
                log_probs = fwd_ma_per_chi(
                    log_chi_stack[c_start:c_end], st_j,
                    jnp.array(xs[c_start:c_end]),
                    jnp.array(ys[c_start:c_end]),
                    jnp.array(real_Lxs[c_start:c_end]),
                    jnp.array(real_Lys[c_start:c_end]),
                    jnp.array(match_is_arr[c_start:c_end]),
                    jnp.array(match_js_arr[c_start:c_end]),
                    jnp.array(match_lens_arr[c_start:c_end]),
                    jnp.array(sub_matrices_stack[c_start:c_end]),
                    jnp.array(pis_arr),
                    n_dom, n_frag)
                log_probs_np = np.asarray(log_probs)
                for i in range(c_end - c_start):
                    if not np.isnan(log_probs_np[i]):
                        total_ll += float(log_probs_np[i])
                        n_pairs += 1
    else:
        # 1D forward-only with vectorized emission construction
        from collections import defaultdict
        buckets = defaultdict(list)
        st_np = np.asarray(st)
        # Precompute per-class (Q_c, pi_c) if model has MixDom2 site classes
        has_classdist = (params.get('n_classes', 0) > 1
                         and 'classdist' in params)
        if has_classdist:
            n_cls = int(params['n_classes'])
            S_per_class = np.asarray(params['class_S_exch'])  # (C, A, A)
            class_pis_np = np.asarray(params['class_pis'])
            classdist_np = np.asarray(params['classdist'])
            class_Qs = np.zeros((n_cls, AA, AA))
            for cc in range(n_cls):
                Q_c = S_per_class[cc] * class_pis_np[cc, None, :]
                np.fill_diagonal(Q_c, 0.0)
                Q_c[np.diag_indices(AA)] = -Q_c.sum(axis=1)
                class_Qs[cc] = Q_c
            class_log_pis_np = np.log(np.maximum(class_pis_np, 1e-30)).astype(np.float32)

        # Hoist per-pair sub_matrices computation out of the per-pair loop
        # via the vmapped+JIT'd `_build_per_pair_sub_matrices` helper. Each
        # batch element b uses its own t_est_b — no representative-t shortcut.
        # Filter empty-state pairs up-front (val skips them just like before).
        val_pairs_filt = [it for it in all_val_pairs if it[2]]
        t_array_val = np.array(
            [it[5] for it in val_pairs_filt], dtype=np.float64)

        if per_domain_subst:
            sub_params_for_pp = {'dom_Qs': params['dom_Qs'],
                                 'dom_pis': params['dom_pis']}
        else:
            sub_params_for_pp = {}
        # Chunk val-pair sub-matrix construction to keep the (B, K, A, A)
        # vmap intermediate within GPU memory. Without chunking, B = n_val_pairs
        # (up to ~5000) × K (n_cls=27 for d3f3c27) × Padé-expm internal
        # workspace (~20× data size) was OOMing at ~10 GiB on the 11 GiB GPU.
        # Per-pair sub-matrices are independent across pairs, so chunking is
        # exact (no boundary overlap). Default chunk = 256 picks a working set
        # of ~50 MB per chunk for d3f3c27 — well within budget.
        _val_chunk = 256

        def _chunked_build(t_arr, sp, n_lead):
            sm_chunks = []
            pis_arr_seen = None
            for i in range(0, len(t_arr), _val_chunk):
                chunk_t = t_arr[i:i + _val_chunk]
                sm_chunk, pis_chunk = _build_per_pair_sub_matrices(
                    chunk_t, sp, Q_lg, pi_lg, n_lead)
                sm_chunks.append(np.asarray(sm_chunk))
                pis_arr_seen = pis_chunk
            return (np.concatenate(sm_chunks, axis=0)
                    if sm_chunks else np.zeros((0, n_lead, AA, AA))), pis_arr_seen

        sub_matrices_stack_all, pis_for_subs = _chunked_build(
            t_array_val, sub_params_for_pp, n_dom)
        log_subs_stack_all = np.log(
            np.maximum(sub_matrices_stack_all, 1e-30)).astype(np.float32)
        log_pis_arr_shared = np.log(
            np.maximum(pis_for_subs, 1e-30)).astype(np.float32)

        if has_classdist:
            # Reuse the same vmapped helper, treating n_cls as the leading
            # "domain" axis. class_Qs has shape (n_cls, A, A), class_pis_np
            # has shape (n_cls, A) — exactly the (D, A, A) / (D, A) layout
            # the helper expects. Chunked over val pairs to bound memory.
            class_sub_params = {'dom_Qs': class_Qs,
                                'dom_pis': class_pis_np}
            class_sub_stack_all, _ = _chunked_build(
                t_array_val, class_sub_params, n_cls)
            class_log_subs_stack_all = np.log(
                np.maximum(class_sub_stack_all, 1e-30)).astype(np.float32)

        for ipi, item in enumerate(val_pairs_filt):
            x_int, y_int, states, anc_chars, desc_chars, t_est = item
            L = len(states)
            states_arr = np.array(states, dtype=np.int32)
            m_or_d = (states_arr == _M) | (states_arr == _D)
            m_or_i = (states_arr == _M) | (states_arr == _I)
            anc_full = np.zeros(L, dtype=np.int32)
            desc_full = np.zeros(L, dtype=np.int32)
            anc_full[m_or_d] = anc_chars
            desc_full[m_or_i] = desc_chars

            # Slice this pair's per-domain sub matrices. For the LG fallback,
            # `_build_per_pair_sub_matrices` already tiled (n_dom, A, A) so
            # `mixdom_constrained_emissions_vectorized`'s dom_idx indexing
            # works for n_dom > 1 (preserves the prior tile-via-broadcast
            # behavior bit-for-bit).
            log_subs = log_subs_stack_all[ipi]
            log_pis_arr = log_pis_arr_shared

            # JAX-native emit (replaces the NumPy-only
            # `mixdom_constrained_emissions_vectorized`). Same per-(state,
            # column) output, fp32-precise. Cast back to NumPy so the
            # surrounding NumPy-padded bucket layout below is unchanged.
            from tkfmixdom.jax.dp.hmm import pair_hmm_emissions_constrained as _emit_jax
            subs_lin_pair = sub_matrices_stack_all[ipi]   # (n_dom, A, A)
            pis_lin_pair = pis_for_subs                   # (n_dom, A)
            if has_classdist:
                cls_subs_lin = class_sub_stack_all[ipi]   # (n_cls, A, A)
                cls_pis_lin = class_pis_np                # (n_cls, A)
                log_emit = np.asarray(_emit_jax(
                    jnp.asarray(st), jnp.asarray(states_arr),
                    jnp.asarray(anc_full), jnp.asarray(desc_full),
                    jnp.asarray(subs_lin_pair), jnp.asarray(pis_lin_pair),
                    n_dom, n_frag,
                    classdist=jnp.asarray(classdist_np),
                    class_sub_matrices=jnp.asarray(cls_subs_lin),
                    class_pis=jnp.asarray(cls_pis_lin)))
            else:
                log_emit = np.asarray(_emit_jax(
                    jnp.asarray(st), jnp.asarray(states_arr),
                    jnp.asarray(anc_full), jnp.asarray(desc_full),
                    jnp.asarray(subs_lin_pair), jnp.asarray(pis_lin_pair),
                    n_dom, n_frag))

            padded_L = _pad_to_bin(L)
            padded_emit = np.full((padded_L, N), np.float32(NEG_INF), dtype=np.float32)
            padded_emit[:L] = log_emit
            # Store t_est so the bucket loop can build per-pair log_chi at
            # the matching t — chi at t_rep with emissions at t_est would
            # be incoherent.
            buckets[padded_L].append((padded_emit, L, t_est))

        # Val eval bundles all val pairs into per-bucket vmaps; at large
        # ns the associative-scan FB materialises a (B, L, n, n) tensor
        # that hits 13+ GiB on d3f3c27. Sequential scan trades parallel
        # depth for memory (O(L·n + n²) per pair) and runs comfortably.
        # Val happens infrequently so the speed cost is negligible vs.
        # training.
        batched_fwd_per_chi = _get_batched_fwd_per_chi_jit(scan_mode='sequential')
        total_ll = 0.0
        n_pairs = 0
        for padded_L, bucket in buckets.items():
            B = len(bucket)
            emit_stack = jnp.array(np.stack([item[0] for item in bucket]))
            lengths = jnp.array([item[1] for item in bucket], dtype=jnp.int32)
            t_ests = jnp.array([item[2] for item in bucket], dtype=jnp.float32)
            log_chi_stack = _build_log_chi_stack(params, t_ests)  # (B, N, N)
            log_probs = batched_fwd_per_chi(log_chi_stack, emit_stack, lengths)
            log_probs_np = np.asarray(log_probs)
            for i in range(B):
                total_ll += float(log_probs_np[i])
                n_pairs += 1

    ll_per_pair = total_ll / max(n_pairs, 1)
    elapsed = time.monotonic() - eval_start
    _log(f"  Val eval ({split_name}): {n_pairs} pairs from "
         f"{len(val_files)} families, LL/pair={ll_per_pair:.2f} "
         f"(total={total_ll:.1f}, {elapsed:.1f}s)")
    return total_ll, n_pairs, ll_per_pair


def _report_params(params, n_dom):
    _log(f"  Params: main_ins={params['main_ins']:.4f}, main_del={params['main_del']:.4f}")
    for d in range(n_dom):
        ext = params['ext_rates'][d]
        if np.ndim(ext) == 0:
            ext_str = f"{ext:.4f}"
        elif np.ndim(ext) == 1:
            ext_str = str([f"{x:.4f}" for x in ext])
        else:
            # (F, F) matrix — show row sums
            ext_str = f"F×F ext_row_sums={[f'{s:.3f}' for s in np.asarray(ext).sum(axis=-1)]}"
        _log(f"    dom {d}: ins={params['dom_ins'][d]:.4f}, del={params['dom_del'][d]:.4f}, "
             f"w={params['dom_weights'][d]:.3f}, ext={ext_str}")


def _report_suff_stats(resolved, n_dom, n_frag):
    top_eff = resolved['top_counts'].sum()
    top_rest = resolved['top_counts_restored'].sum()
    phantom_top = top_rest - top_eff
    _log(f"  Suff stats:")
    _log(f"    top_counts: {top_eff:.1f} effective, "
         f"+{phantom_top:.1f} phantom = {top_rest:.1f} restored")
    for d in range(n_dom):
        dc = resolved['dom_counts'][d]
        dc_m = resolved['dom_counts_M'][d]
        occ = resolved['dom_occupancy'][d]
        phantom_d = dc_m.sum() - dc.sum()
        _log(f"    dom {d}: occ={occ:.1f}, counts={dc.sum():.1f}"
             f"+{phantom_d:.1f}phantom, "
             f"ext_self={resolved['ext_self'][d].sum():.1f}, "
             f"ext_exit={resolved['ext_exit'][d].sum():.1f}")


def _report_suff_stats_exact(ss, n_dom, n_frag):
    _log(f"  Exact suff stats (via chain restoration):")
    _log(f"    top_5x5 sum: {ss['top_5x5'].sum():.1f}")
    for d in range(n_dom):
        dm = ss['dom_M_5x5'][d].sum()
        _log(f"    dom {d}: M_5x5={dm:.1f}, κ={ss['dom_kappa'][d]:.1f}, "
             f"1-κ={ss['dom_1mkappa'][d]:.1f}, "
             f"ext={ss['ext'][d].sum():.1f}, term={ss['term'][d].sum():.1f}")


# ============================================================
# Adam training mode (separate from EM)
# ============================================================
def train_adam(args):
    """Train MixDom via Adam gradient ascent on stochastic minibatches.

    Completely separate code path from Baum-Welch EM. Uses custom VJPs
    on the pair HMM log-likelihood for efficient gradients.
    """
    import jax
    import jax.numpy as jnp
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    from tkfmixdom.jax.core.ctmc import transition_matrix
    from tkfmixdom.jax.models.compiled import init_mixdom_params as _init_params
    from tkfmixdom.jax.simulate.msa import alignment_to_states

    n_dom = args.n_dom
    n_frag = args.n_frag
    Q_lg, pi_lg = rate_matrix_lg()

    _log(f"Adam mode: lr={args.adam_lr}, batch={args.adam_batch}, "
         f"steps={args.adam_steps}")

    # ---- Check for precompiled pairs ----
    precompiled_source = None
    pair_stream = None
    if getattr(args, 'precompiled_pairs', None):
        precompiled_source = PrecompiledPairSource(args.precompiled_pairs)
        _log(f"Using precompiled pairs: {precompiled_source.description}")

    if precompiled_source is not None:
        # Use streaming mode: one shard at a time, constant memory
        from tkfmixdom.jax.util.pair_loader import PairStream
        pair_stream = PairStream(
            args.precompiled_pairs, seed=args.seed,
            max_alignment_len=getattr(args, 'max_alignment_len', None))
        all_pairs = None  # not needed in streaming mode
        _log(f"  Streaming: {len(pair_stream)} pairs across "
             f"{pair_stream.n_shards} shards (constant memory)")
    else:
        # ---- Resolve MSA directory ----
        from tkfmixdom.jax.util.bio_datasets import (
            apply_bio_datasets_arg, resolve_data_dir)
        apply_bio_datasets_arg(args)
        _msa_dir_is_default = (args.msa_dir == 'pfam/')
        if _msa_dir_is_default:
            msa_dir_resolved = str(resolve_data_dir("pfam/seed", local_fallback=args.msa_dir))
            if msa_dir_resolved != args.msa_dir:
                args.msa_dir = msa_dir_resolved

        # ---- Find MSA files ----
        import glob as glob_mod
        if args.families:
            family_list = [f.strip() for f in args.families.split(',')]
            msa_files = []
            for fam in family_list:
                for ext in ['.sto', '.sto.gz']:
                    p = os.path.join(args.msa_dir, fam + ext)
                    if os.path.exists(p):
                        msa_files.append(p)
                        break
        else:
            msa_files = sorted(
                glob_mod.glob(os.path.join(args.msa_dir, '*.sto')) +
                glob_mod.glob(os.path.join(args.msa_dir, '*.sto.gz')))

        # ---- Apply split ----
        if args.split:
            split_file = args.split_file
            if not split_file:
                for candidate in [
                    os.path.expanduser('~/bio-datasets/data/pfam/seed/splits/v1.json'),
                os.path.expanduser('~/bio-datasets/data/pfam-seed/splits/v1.json'),
                    os.path.join(args.msa_dir, 'splits', 'v1.json'),
                ]:
                    if os.path.exists(candidate):
                        split_file = candidate
                        break
            if split_file:
                import json as json_mod
                with open(split_file) as f:
                    split_data = json_mod.load(f)
                split_fams = set(split_data.get(args.split, []))
                msa_files = [f for f in msa_files
                             if os.path.basename(f).split('.')[0] in split_fams]

        _log(f"Found {len(msa_files)} MSA files")

        # ---- Build pairs ----
        _log("Building cherry pairs...")
        all_pairs = []
        for fi, msa_file in enumerate(msa_files):
            pairs = _build_pairs_for_file(fi, msa_file)
            names, seqs = parse_stockholm(msa_file)
            for row1, row2, t_est in pairs:
                x = _aligned_pair_to_int_arrays(seqs[row1], seqs[row2])
                if len(x[0]) == 0:
                    continue
                # Convert to gapped integer pair for alignment_to_states
                states, anc_chars, desc_chars = alignment_to_states(x[0], x[1])
                if states:
                    all_pairs.append((
                        jnp.array(x[0], dtype=jnp.int32),
                        jnp.array(x[1], dtype=jnp.int32)))
            if (fi + 1) % max(1, len(msa_files) // 10) == 0:
                _log(f"  {fi+1}/{len(msa_files)} files, {len(all_pairs)} pairs")

        _log(f"  Total: {len(all_pairs)} pairs from {len(msa_files)} families")

    # ---- Initialize params ----
    mar_path = getattr(args, 'init_from_maraschino', None)
    if mar_path:
        _log(f"Initializing from Maraschino: {mar_path}")
        init_params = _load_maraschino_as_train_params(mar_path, n_dom, n_frag)
        args.estimate_subst = True
    else:
        main_ins, main_del, dom_ins, dom_del, dom_weights, frag_weights, ext_rates = \
            _init_params(args.init_ins, args.init_del, n_dom, n_frag, seed=args.seed)
        if getattr(args, 'banded_frag_init', False):
            from tkfmixdom.jax.train.restricted_mstep import banded_3fc_init
            frag_weights, ext_rates = banded_3fc_init(
                n_dom, float(getattr(args, 'p_ext', 0.6)))
            _log(f"  banded-frag-init: p_ext="
                 f"{float(getattr(args, 'p_ext', 0.6)):.4f}")
        init_params = {
            'main_ins': float(main_ins), 'main_del': float(main_del),
            'dom_ins': np.array(dom_ins), 'dom_del': np.array(dom_del),
            'dom_weights': np.array(dom_weights),
            'frag_weights': np.array(frag_weights),
            'ext_rates': np.array(ext_rates),
        }
        # Per-domain (S_exch, π) seed: tile LG with Dirichlet (π) and
        # Gamma (S_exch) symmetry-breaking perturbations. Required by
        # `to_unconstrained` since Phase 5.6 Sub-phase B; without this
        # block, cold-start Adam fails at startup with a KeyError. The
        # Dirichlet concentration (10) and Gamma noise (shape=2, scale=0.05)
        # produce ~5-10% per-entry perturbation per domain — small enough
        # that the LG seed structure is preserved, large enough that the
        # gradient asymmetry between domains is non-vanishing on Adam's
        # first steps.
        AA = int(np.asarray(pi_lg).shape[0])
        rng_subst = np.random.RandomState(args.seed + 1000)
        S_lg_seed = (np.asarray(Q_lg)
                     / np.maximum(np.asarray(pi_lg)[None, :], 1e-30)
                     * (1.0 - np.eye(AA)))
        S_lg_seed = (S_lg_seed + S_lg_seed.T) / 2
        dom_pis_init = np.tile(np.asarray(pi_lg), (n_dom, 1))
        dom_S_init = np.tile(S_lg_seed[None, :, :], (n_dom, 1, 1))
        for dd in range(n_dom):
            pi_noise = rng_subst.dirichlet(np.ones(AA) * 10.0)
            dom_pis_init[dd] = 0.85 * dom_pis_init[dd] + 0.15 * pi_noise
            dom_pis_init[dd] /= dom_pis_init[dd].sum()
            S_noise = rng_subst.gamma(2.0, 0.05, (AA, AA))
            S_noise = (S_noise + S_noise.T) / 2
            np.fill_diagonal(S_noise, 0.0)
            dom_S_init[dd] = dom_S_init[dd] + S_noise
            dom_S_init[dd] = (dom_S_init[dd] + dom_S_init[dd].T) / 2
        init_params['dom_S_exch'] = dom_S_init
        init_params['dom_pis'] = dom_pis_init
        init_params['dom_Qs'] = np.tile(np.asarray(Q_lg)[None, :, :],
                                         (n_dom, 1, 1))
        _log(f"  Adam cold-start per-domain seed: {n_dom} domains × {AA} aa "
             f"(LG-tiled, Dirichlet+Gamma symmetry-breaking)")

    # ---- Budget ----
    budget_s = None
    if args.budget_hours:
        budget_s = args.budget_hours * 3600.0

    # ---- Validation callback for early stopping ----
    # Single-element list used as a mutable closure variable so the
    # val callback can record the most-recent val LL/pair for the
    # diagnostic TSV. Initialised to NaN; updated inside the callback.
    _adam_val_ll_box = [float('nan')]
    val_cb = None
    val_every = getattr(args, 'val_every', 0)
    patience = getattr(args, 'patience', 5)
    if val_every > 0 and args.split == 'train':
        from tkfmixdom.jax.train.early_stopping import EarlyStopper
        split_file = args.split_file
        if not split_file:
            for candidate in [
                os.path.expanduser('~/bio-datasets/data/pfam/seed/splits/v1.json'),
                os.path.expanduser('~/bio-datasets/data/pfam-seed/splits/v1.json'),
            ]:
                if os.path.exists(candidate):
                    split_file = candidate
                    break
        if split_file and os.path.exists(split_file):
            import json as json_mod
            with open(split_file) as f:
                split_data = json_mod.load(f)
            val_fams = set(split_data.get('val', []))
            adam_stopper = EarlyStopper(patience=patience)

            def _adam_val_callback(cur_params, step):
                # Cadence is gated by the train_adam loop
                # (step % val_every == 0), matching SVI-BW semantics 1:1.
                # This callback always runs the val eval when called.
                _val_ll, _val_n, val_ll_pp = _evaluate_on_split(
                    cur_params, 'val', val_fams, args.msa_dir,
                    n_dom, n_frag, 1.0, Q_lg, pi_lg,
                    max_pairs_per_fam=2)
                _adam_val_ll_box[0] = float(val_ll_pp)
                should_stop = adam_stopper.step(val_ll_pp, cur_params, iter_num=step)
                _log(f"  Early stopping: {adam_stopper.report()}")
                return should_stop

            val_cb = _adam_val_callback
            _log(f"  Early stopping: val every {val_every} steps, patience={patience}, "
                 f"{len(val_fams)} val families")

    # ---- Run Adam (Phase 5: BDI-form chi Q via shared SVI-BW E-step) ----
    # The chi-axis Q is the BDI form (tkf.tex eq:l_1, lines 783-786):
    # t-parameter-free; t enters only through aggregated S, T computed
    # per-pair-t-correctly inside `exact_suffstats_per_pair_batch`.
    # Substitution + π + classdist consume per-pair-t-correct HR
    # aggregates (dom_W / dom_U / class_W / class_U / ...). NO `t_rep`.
    import optax
    from tkfmixdom.jax.train.adam_train import (
        to_unconstrained as _to_unc,
        to_constrained as _to_con,
        make_adam_step as _make_adam_step,
    )
    from tkfmixdom.jax.models.mixdom import state_types as _state_types_fn

    if pair_stream is None:
        raise NotImplementedError(
            "Phase 5 train_adam requires --precompiled-pairs. The legacy "
            "in-memory all_pairs path was deprecated by the BDI-form chi "
            "Q rewrite. Re-run with `--precompiled-pairs <dir>`.")

    dp_mode = getattr(args, 'dp_mode', 'constrained')
    uc = _to_unc(init_params, n_dom, n_frag)
    is_mixdom2 = 'log_class_pis_logits' in uc

    # MixDom2 + 2D unaligned dispatch is not supported: the 2D batched
    # E-step `_process_pairs_batched_2d` does not propagate
    # classdist_params, so it would not produce class_W / class_U /
    # class_*_counts / classdist_counts. Adam loss for MixDom2 needs
    # those keys. Fail loud rather than KeyError mid-step.
    if is_mixdom2 and dp_mode != 'constrained':
        raise NotImplementedError(
            "MixDom2 + --dp-mode full is not supported in the Phase 5 "
            "rewrite (the 2D batched E-step doesn't carry classdist "
            "machinery). Use --dp-mode constrained (the production "
            "default) for any MixDom2 run.")
    optimizer = optax.adam(args.adam_lr)
    opt_state = optimizer.init(uc)

    # Phase 5.6 Sub-phase B: no Q/pi external kwargs. The substitution
    # model is per-domain and lives entirely in `uc` (`log_dom_S_upper`,
    # `log_dom_pi_logits`); MixDom2 layers per-class on top.
    step_fn = _make_adam_step(
        optimizer, n_dom, n_frag,
        ins_prior=tuple(args.ins_prior),
        del_prior=tuple(args.del_prior),
        dom_dirichlet=args.dom_dirichlet,
        frag_dirichlet=args.frag_dirichlet,
        ext_alpha=args.ext_alpha,
        ext_beta=args.ext_beta,
        classdist_dirichlet=getattr(args, 'classdist_dirichlet', 1.0),
        class_pi_dirichlet=getattr(args, 'class_pi_dirichlet', 1.0),
        class_S_gamma=tuple(getattr(args, 'class_S_gamma', (1.0, 0.0))),
    )

    st = _state_types_fn(n_dom, n_frag)
    N = int(np.asarray(st).shape[0])
    # Match SVI-BW's per-iter logging cadence (line 5323 / 5993): log
    # every Adam step. Phase 7 (CLI parity audit) will replace this
    # with a `--log-every` flag shared across SVI-BW and Adam.
    log_every = 1
    val_every_safe = val_every if val_every > 0 else None

    history = []
    best_loss = float('inf')      # best minibatch (training) loss
    best_uc = uc
    start_time = time.monotonic()
    n_pairs_total = len(pair_stream)
    # Capture median t_est from the manifest (used as checkpoint
    # metadata; NOT consumed by the Adam loss — the BDI Q is t-free).
    _t_median_for_meta = (pair_stream.manifest.get('stats', {})
                            .get('t_est', {}).get('median', 1.0))
    _log(f"Adam training (Phase 5, dp-mode={dp_mode}): {args.adam_steps} steps, "
         f"lr={args.adam_lr}, batch={args.adam_batch}, "
         f"MixDom2={is_mixdom2}, {n_pairs_total} pairs (streaming)")

    # Phase 7 CLI/UX parity: Adam diagnostic TSV mirrors SVI-BW's
    # `pfam/svi_bw_*_diag.tsv`. Columns: step, time, loss, ll_pair,
    # val_ll_pair, n_pairs. Header is written for fresh files; on
    # resume we append.
    _adam_diag_ckpt = getattr(args, 'checkpoint', None)
    _adam_diag_path = (_adam_diag_ckpt.replace('.npz', '_adam_diag.tsv')
                       if _adam_diag_ckpt else None)
    _adam_diag_tsv = None
    if _adam_diag_path is not None:
        _adam_diag_new = not os.path.exists(_adam_diag_path)
        _adam_diag_tsv = open(_adam_diag_path, 'a', buffering=1)
        if _adam_diag_new:
            _adam_diag_tsv.write(
                'step\ttime\tloss\tll_pair\tval_ll_pair\tn_pairs\n')

    def _np_constrained_params(uc_now):
        """Build a NumPy params dict from current uc. After Phase 5.6
        Sub-phase B, `to_constrained(uc)` always populates
        `dom_S_exch`, `dom_pis`, `dom_Qs` from `uc`'s
        `log_dom_S_upper` / `log_dom_pi_logits`; the SVI-BW E-step
        consumes those directly from `params` (the LG-tile fallback
        in `_build_per_pair_sub_matrices` is now unreachable)."""
        cp = _to_con(uc_now, n_dom, n_frag)
        out = {}
        for k, v in cp.items():
            if hasattr(v, 'shape'):
                out[k] = np.asarray(v)
            else:
                out[k] = v
        return out

    for step in range(args.adam_steps):
        raw_batch = pair_stream.sample_batch(args.adam_batch,
                                              seed=args.seed + step)
        raw_batch = [p for p in raw_batch if len(p[0]) > 0 and len(p[1]) > 0]
        if not raw_batch:
            _log(f"  Step {step}: empty batch from stream, skipping")
            continue

        cur_params_np = _np_constrained_params(uc)
        cd_params = (_get_classdist_params(cur_params_np)
                     if is_mixdom2 else None)

        if dp_mode == 'constrained':
            delta = _process_pairs_batched(
                raw_batch, cur_params_np, st, Q_lg, pi_lg, N,
                alignment_to_states,
                dom_Qs=cur_params_np.get('dom_Qs'),
                dom_pis=cur_params_np.get('dom_pis'),
                classdist_params=cd_params,
                n_dom=n_dom, n_frag=n_frag,
                fb_scan_mode=_resolve_fb_scan_mode(args, N))
        else:
            delta = _process_pairs_batched_2d(
                raw_batch, cur_params_np, st, Q_lg, pi_lg, N,
                dom_Qs=cur_params_np.get('dom_Qs'),
                dom_pis=cur_params_np.get('dom_pis'),
                n_dom=n_dom, n_frag=n_frag,
                params=cur_params_np)

        if delta is None or delta.get('n_pairs', 0) == 0:
            _log(f"  Step {step}: E-step returned None / 0 pairs, skipping")
            continue

        # NumPy → JAX. jax.tree.map handles nested dicts (including
        # delta['exact_ss']). delta enters the JIT'd step as a leaf
        # input — autograd treats it as a fixed tensor; no FB in bwd.
        delta_jax = jax.tree.map(jnp.asarray, delta)

        uc, opt_state, loss = step_fn(uc, opt_state, delta_jax)
        loss_f = float(loss)
        elapsed = time.monotonic() - start_time

        # best_uc tracks the minimum training-loss uc seen so far,
        # used as the saved checkpoint params if no val_callback runs.
        if loss_f < best_loss:
            best_loss = loss_f
            best_uc = uc

        # history entry shape: {'step', 'loss', 'll', 'time'}.
        # `loss` is the optimiser's negative-LL-per-pair (the loss is
        # normalised by n_pairs inside `adam_loss_from_delta`).
        # `ll` is the corresponding LL/pair (positive, larger is better).
        history.append({'step': step, 'loss': loss_f,
                        'll': -loss_f, 'time': elapsed})

        if step % log_every == 0 or step == args.adam_steps - 1:
            _log(f"  Step {step}/{args.adam_steps}: loss={loss_f:.4f} "
                 f"LL/pair={-loss_f:.4f} ({elapsed:.1f}s)")

        if _adam_diag_tsv is not None:
            _adam_diag_tsv.write(
                f"{step}\t{elapsed:.2f}\t{loss_f:.6f}\t{-loss_f:.6f}\t"
                f"{_adam_val_ll_box[0]:.6f}\t{int(delta.get('n_pairs', 0))}\n")

        # Val callback (semantics matched to SVI-BW's val_every).
        if val_every_safe and step % val_every_safe == 0 and val_cb is not None:
            cur_params_full = _np_constrained_params(uc)
            should_stop = val_cb(cur_params_full, step)
            if should_stop:
                _log(f"  Early stopping triggered at step {step}")
                break

        if budget_s and elapsed >= budget_s:
            _log(f"  Budget reached: {elapsed:.1f}s >= {budget_s:.0f}s")
            break

    # Use best (lowest training loss) uc for the final checkpoint.
    # If the val_callback's EarlyStopper has tracked val-best params,
    # those would be a better target — but the EarlyStopper's
    # best_params live inside the closure and are not retrieved here;
    # downstream tools rely on the val log + best_val.npz produced
    # separately by the EarlyStopper itself.
    final_params = _np_constrained_params(best_uc)
    if not history:
        # Couldn't run any successful step (e.g. empty stream). Surface
        # rather than silently returning init_params.
        raise RuntimeError(
            "Adam training produced no successful steps — empty pair "
            "stream or all batches filtered. Check --precompiled-pairs "
            "and --adam-batch.")

    # ---- Save ----
    checkpoint_path = args.checkpoint or os.path.join(
        args.msa_dir, f'adam_d{n_dom}_f{n_frag}.npz')
    _save_checkpoint(checkpoint_path, final_params,
                     _zero_suff_stats(1, n_dom), 0,
                     len(history), [h['ll'] for h in history],
                     {'method': 'adam', 'n_dom': n_dom, 'n_frag': n_frag,
                      'lr': args.adam_lr, 'batch_size': args.adam_batch,
                      'seed': args.seed},
                     None, _t_median_for_meta, None, None)
    _log(f"\nSaved: {checkpoint_path}")
    _report_params(final_params, n_dom)



# ============================================================
# SVI Baum-Welch training mode
# ============================================================
def train_svi_bw(args):
    """Train MixDom via Stochastic Variational Inference Baum-Welch.

    Proper SVI (Hoffman et al. 2013): maintains a running exponential
    moving average (EMA) of sufficient statistics across minibatch
    iterations.  At each iteration k:

        1. Sample a minibatch of B pairs
        2. Run E-step to get batch sufficient statistics
        3. Scale batch stats by (N_est / B)
        4. EMA-blend: running = (1 - eta_k) * running + eta_k * scaled_batch
        5. M-step from running stats

    Step size: eta_k = (k + tau)^{-kappa},  kappa in (0.5, 1], tau >= 0.
    """
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    from tkfmixdom.jax.core.ctmc import transition_matrix
    from tkfmixdom.jax.core.params import S, M, I, D, E
    from tkfmixdom.jax.models.mixdom import (
        build_nested_trans, n_states as mixdom_n_states,
        state_types as mixdom_state_types
    )
    from tkfmixdom.jax.models.compiled import init_mixdom_params as _init_params
    from tkfmixdom.jax.models.exact_suffstats import exact_suffstats
    from tkfmixdom.jax.core.bdi import bdi_stats_from_counts, m_step_indel_quadratic
    from tkfmixdom.jax.simulate.msa import alignment_to_states
    from tkfmixdom.jax.dp.hmm import safe_log
    from tkfmixdom.jax.util.sto_index import StoIndex

    n_dom = args.n_dom
    n_frag = args.n_frag
    n_iter = args.n_iter
    svi_batch = getattr(args, 'svi_batch', 50)
    svi_tau = getattr(args, 'svi_tau', 1.0)
    svi_kappa = getattr(args, 'svi_kappa', 0.7)

    checkpoint_path = args.checkpoint or os.path.join(
        args.msa_dir, f'svi_bw_d{n_dom}_f{n_frag}.npz')

    config = {
        'n_dom': n_dom, 'n_frag': n_frag, 'n_iter': n_iter,
        'init_ins': args.init_ins, 'init_del': args.init_del,
        'seed': args.seed, 'msa_dir': args.msa_dir,
        'families': args.families,
        'method': 'svi_bw',
        'svi_tau': svi_tau, 'svi_kappa': svi_kappa, 'svi_batch': svi_batch,
        'pseudocounts': {
            'ins_prior': list(args.ins_prior),
            'del_prior': list(args.del_prior),
            'dom_dirichlet': args.dom_dirichlet,
            'frag_dirichlet': args.frag_dirichlet,
            'ext_alpha': args.ext_alpha,
            'ext_beta': args.ext_beta,
        },
    }

    N = mixdom_n_states(n_dom, n_frag)
    st = mixdom_state_types(n_dom, n_frag)
    Q_lg, pi_lg = rate_matrix_lg()

    # ---- Check for precompiled pairs ----
    precompiled_source = None
    if getattr(args, 'precompiled_pairs', None):
        precompiled_source = PrecompiledPairSource(args.precompiled_pairs)
        _log(f"Using precompiled pairs: {precompiled_source.description}")

    if precompiled_source is not None:
        from collections import defaultdict
        _precompiled_by_family = defaultdict(list)
        for item in precompiled_source._decode_all():
            x_int, y_int, states, anc_chars, desc_chars, t_est, fam = item
            _precompiled_by_family[fam].append(
                (x_int, y_int, states, anc_chars, desc_chars, t_est))
        _precompiled_families = sorted(_precompiled_by_family.keys())
        n_families = len(_precompiled_families)
        _log(f"  Precompiled: {n_families} families, "
             f"{sum(len(v) for v in _precompiled_by_family.values())} pairs")
        stats = precompiled_source.manifest.get('stats', {})
        t_rep = stats.get('t_est', {}).get('median', 1.0)
        # Build flat list of all pairs with family index
        all_pairs = []
        for fi, fam in enumerate(_precompiled_families):
            for pair in _precompiled_by_family[fam]:
                all_pairs.append((fi, pair))
        # For breadth-sampling: group pair indices by family
        _pair_idxs_by_family = [[] for _ in _precompiled_families]
        for _idx, (_fi, _pair) in enumerate(all_pairs):
            _pair_idxs_by_family[_fi].append(_idx)
        msa_files = []
    else:
        _precompiled_by_family = None
        _precompiled_families = None

        # ---- Resolve MSA directory ----
        from tkfmixdom.jax.util.bio_datasets import (
            apply_bio_datasets_arg, resolve_data_dir, ensure_symlinks)
        apply_bio_datasets_arg(args)
        _msa_dir_is_default = (args.msa_dir == 'pfam/')
        if _msa_dir_is_default:
            msa_dir_resolved = str(resolve_data_dir("pfam/seed", local_fallback=args.msa_dir))
            if msa_dir_resolved != args.msa_dir:
                _log(f"  bio-datasets: {args.msa_dir} -> {msa_dir_resolved}")
                ensure_symlinks(
                    __import__('pathlib').Path(msa_dir_resolved),
                    __import__('pathlib').Path(args.msa_dir), pattern="*.sto")
                ensure_symlinks(
                    __import__('pathlib').Path(msa_dir_resolved),
                    __import__('pathlib').Path(args.msa_dir), pattern="*.sto.gz")
            args.msa_dir = msa_dir_resolved

        # ---- Find MSA files ----
        import glob as glob_mod
        if args.families:
            family_list = [f.strip() for f in args.families.split(',')]
            msa_files = []
            for fam in family_list:
                for ext in ['.sto', '.sto.gz', '.stockholm', '.stockholm.gz']:
                    p = os.path.join(args.msa_dir, fam + ext)
                    if os.path.exists(p):
                        msa_files.append(p)
                        break
                else:
                    _log(f"Warning: {fam} not found in {args.msa_dir}")
        else:
            msa_files = sorted(
                glob_mod.glob(os.path.join(args.msa_dir, '*.sto')) +
                glob_mod.glob(os.path.join(args.msa_dir, '*.sto.gz')))

        if not msa_files:
            _log(f"No MSA files found in {args.msa_dir}")
            sys.exit(1)
        _log(f"Found {len(msa_files)} MSA files")

        # ---- Data split ----
        if args.split:
            split_file = args.split_file
            if not split_file:
                for candidate in [
                    os.path.expanduser('~/bio-datasets/data/pfam/seed/splits/v1.json'),
                os.path.expanduser('~/bio-datasets/data/pfam-seed/splits/v1.json'),
                    os.path.join(args.msa_dir, 'splits', 'v1.json'),
                ]:
                    if os.path.exists(candidate):
                        split_file = candidate
                        break
            if split_file and os.path.exists(split_file):
                import json as json_mod
                with open(split_file) as f:
                    split_data = json_mod.load(f)
                split_fams = set(split_data.get(args.split, []))
                if not split_fams:
                    _log(f"ERROR: split '{args.split}' not found in {split_file}")
                    sys.exit(1)
                msa_files = [f for f in msa_files
                             if os.path.basename(f).split('.')[0] in split_fams]
                _log(f"  Split file: {split_file}")
                _log(f"  Using {args.split} split: {len(msa_files)} families")
            elif args.split:
                clan_file = args.clan_file or _find_clan_file(args.msa_dir)
                if not clan_file:
                    _log("ERROR: --split requires --split-file or a clan file.")
                    sys.exit(1)
                ratios = tuple(float(r) for r in args.split_ratios.split(','))
                split_indices = clan_aware_split(
                    msa_files, clan_file, args.split, ratios, args.split_seed)
                msa_files = [msa_files[i] for i in split_indices]
                _log(f"  Ad-hoc {args.split} split: {len(msa_files)} families")

        n_families = len(msa_files)
        t_rep = 1.0

        # Build flat list of all (family_index, pair) for sampling
        _log("Building pair index for SVI-BW...")
        all_pairs = []
        for fi in range(n_families):
            try:
                pairs = _build_pairs_for_file(fi, msa_files[fi])
                for pair in pairs:
                    all_pairs.append((fi, pair))
            except Exception as e:
                _log(f"  Warning: skipping {os.path.basename(msa_files[fi])}: {e}")
        _log(f"  Total pairs available: {len(all_pairs)} from {n_families} families")

    if len(all_pairs) == 0:
        _log("ERROR: no pairs found")
        sys.exit(1)

    N_est = len(all_pairs)  # total dataset size

    # ---- Load checkpoint or initialize ----
    if os.path.exists(checkpoint_path) and not args.fresh:
        _log(f"Resuming from checkpoint: {checkpoint_path}")
        (params, suff_stats, step_file_idx, em_iter, log_probs, saved_config,
         manifest, t_rep_loaded, budget_files, family_timings) = \
            _load_checkpoint(checkpoint_path)
        if saved_config.get('n_dom') != n_dom or saved_config.get('n_frag') != n_frag:
            _log(f"ERROR: checkpoint n_dom/n_frag mismatch. Use --fresh.")
            sys.exit(1)
        if t_rep_loaded is not None:
            t_rep = t_rep_loaded

        # Load SVI running stats from checkpoint — we lift these to the
        # α̃ pytree below, once svb_prior is built from args + params.
        # Keep the on-disk format unchanged (stored as running counts,
        # i.e. α̃ − α_prior) so existing checkpoints resume cleanly.
        data = np.load(checkpoint_path, allow_pickle=True)
        _loaded_running = None
        if 'svi_running_top_5x5' in data:
            _loaded_running = {
                'top_5x5': data['svi_running_top_5x5'],
                'dom_M_5x5': [data[f'svi_running_dom_M_5x5_{d}']
                              for d in range(n_dom)],
                'dom_kappa': data['svi_running_dom_kappa'],
                'dom_1mkappa': data['svi_running_dom_1mkappa'],
                'ext': data['svi_running_ext'],
                'term': data['svi_running_term'],
                'dom_w': data['svi_running_dom_w'],
                'frag_w': [data[f'svi_running_frag_w_{d}']
                           for d in range(n_dom)],
            }
        if 'svi_rng_state_keys' in data:
            svi_rng_state = (
                'MT19937',
                data['svi_rng_state_keys'],
                int(data['svi_rng_state_pos']),
                int(data['svi_rng_state_has_gauss']),
                float(data['svi_rng_state_cached_gauss']))
        else:
            svi_rng_state = None
        data.close()
        _log(f"  Resumed at EM iter {em_iter}")
    else:
        mar_path = getattr(args, 'init_from_maraschino', None)
        if mar_path:
            _log(f"Initializing from Maraschino: {mar_path}")
            params = _load_maraschino_as_train_params(mar_path, n_dom, n_frag)
            # Maraschino init always provides substitution model
            args.estimate_subst = True
        else:
            from tkfmixdom.jax.models.mixdom_init import (
                init_mixdom2_params_from_args)
            params = init_mixdom2_params_from_args(
                args, n_dom, n_frag, Q_lg, pi_lg, log_fn=_log)
        suff_stats = _zero_suff_stats(N, n_dom)
        em_iter = 0
        log_probs = []
        _loaded_running = None
        svi_rng_state = None

    # ---- Build posterior pseudocount prior and lift any loaded stats ----
    # svb_prior is the α_prior pytree (keys mirror svi_running_stats, with
    # values set from the CLI pseudocount hyperparameters). The primary
    # SVI-BW state below is α̃ = α_prior + EMA(N/B · s_batch), updated each
    # iter via ema_update_pseudocount. See tkf/svb-convergence.tex
    # §Pseudocount representation.
    from tkfmixdom.jax.train.pseudocounts import (
        build_prior_pseudocounts, lift_to_pseudocount, lower_to_running,
        ema_update_pseudocount, ema_weight_history, ess_from_weights,
    )
    svb_prior = build_prior_pseudocounts(
        args, n_dom, n_frag,
        n_classes=int(params.get('n_classes', 1)), AA=AA)
    if _loaded_running is not None:
        # Only lift keys that are actually present in the loaded checkpoint.
        # Keys in svb_prior but NOT in _loaded_running (e.g. classdist_counts
        # on a pre-MixDom2 checkpoint) are deliberately LEFT OUT of α̃ —
        # they will be initialised on their first batch exactly as the
        # legacy code did (as `scale * batch` → α̃ = prior + scale * batch),
        # via ema_update_pseudocount's missing-key branch.
        _resume_prior = {k: svb_prior[k] for k in _loaded_running
                         if k in svb_prior}
        svi_alpha_tilde = lift_to_pseudocount(_loaded_running, _resume_prior)
    else:
        svi_alpha_tilde = None

    # ---- Set up RNG ----
    rng = np.random.RandomState(args.seed + 7777)
    if svi_rng_state is not None:
        rng.set_state(svi_rng_state)

    save_every_iter = getattr(args, 'save_every_iter', False)

    def _save_svi_checkpoint():
        """Save checkpoint with SVI running stats."""
        _save_checkpoint(checkpoint_path, params, suff_stats, 0,
                         em_iter, log_probs, config, None, t_rep,
                         None, None)
        data = dict(np.load(checkpoint_path, allow_pickle=True))
        # Save running stats (α̃ − α_prior) for on-disk format compat.
        # The primary in-memory state is svi_alpha_tilde (posterior
        # pseudocounts); convert back to data counts at save time so old
        # checkpoint consumers still work.
        if svi_alpha_tilde is not None:
            _ckpt_running = lower_to_running(svi_alpha_tilde, svb_prior)
            data['svi_running_top_5x5'] = np.array(_ckpt_running['top_5x5'])
            for d in range(n_dom):
                data[f'svi_running_dom_M_5x5_{d}'] = np.array(
                    _ckpt_running['dom_M_5x5'][d])
            data['svi_running_dom_kappa'] = np.array(_ckpt_running['dom_kappa'])
            data['svi_running_dom_1mkappa'] = np.array(_ckpt_running['dom_1mkappa'])
            data['svi_running_ext'] = np.array(_ckpt_running['ext'])
            data['svi_running_term'] = np.array(_ckpt_running['term'])
            data['svi_running_dom_w'] = np.array(_ckpt_running['dom_w'])
            for d in range(n_dom):
                data[f'svi_running_frag_w_{d}'] = np.array(
                    _ckpt_running['frag_w'][d])
            # Per-class & classdist counts (class_match_counts,
            # class_insert_counts, classdist_counts) are intentionally NOT
            # saved here — they are regenerated from the first batch on
            # resume, matching existing behavior exactly.
        # Save RNG state
        rng_state = rng.get_state()
        data['svi_rng_state_keys'] = rng_state[1]
        data['svi_rng_state_pos'] = np.int64(rng_state[2])
        data['svi_rng_state_has_gauss'] = np.int64(rng_state[3])
        data['svi_rng_state_cached_gauss'] = np.float64(rng_state[4])
        # Save val tracking state for resume
        data['best_val_ll'] = np.float64(best_val_ll)
        data['val_no_improve'] = np.int64(val_no_improve)
        data['best_val_n_pairs'] = np.int64(best_val_n_pairs)
        np.savez_compressed(checkpoint_path, **data)
        # Optionally save in Maraschino format for re-distillation
        mar_save = getattr(args, 'svi_bw_save_maraschino', None)
        if mar_save:
            _save_params_as_maraschino(mar_save, params, n_dom, n_frag)

    # ---- Report ----
    _log(f"\n{'='*60}")
    _log(f"SVI Baum-Welch: {n_dom} domains, {n_frag} fragments, {N} states")
    _log(f"  {N_est} total pairs, batch_size={svi_batch}, {n_iter} iterations")
    _log(f"  Step size: eta_k = (k + {svi_tau})^(-{svi_kappa})")
    _log(f"  Backend: {jax.default_backend()}, devices: {jax.devices()}")
    _log(f"  Checkpoint: {checkpoint_path}")
    _log(f"{'='*60}\n")

    # ---- Main SVI loop ----
    # Run at least n_iter iterations. If val_every > 0, continue past n_iter
    # as long as val LL is improving (patience-based early stopping).
    patience = getattr(args, 'patience', 5)
    val_every = getattr(args, 'val_every', 0)
    # Restore best_val_ll from checkpoint if available
    if os.path.exists(checkpoint_path):
        _ckpt_data = np.load(checkpoint_path, allow_pickle=True)
        best_val_ll = float(_ckpt_data['best_val_ll']) if 'best_val_ll' in _ckpt_data else -float('inf')
        val_no_improve = int(_ckpt_data['val_no_improve']) if 'val_no_improve' in _ckpt_data else 0
        best_val_n_pairs = int(_ckpt_data['best_val_n_pairs']) if 'best_val_n_pairs' in _ckpt_data else 0
        del _ckpt_data
    else:
        best_val_ll = -float('inf')
        val_no_improve = 0
        best_val_n_pairs = 0
    # `--n-iter` is the hard cap on iteration count. Earlier behavior
    # treated it as a soft cap that extended up to 10× when val_every > 0
    # — surprising and inconsistent with the CLI flag's name. Now `n_iter`
    # is exactly the iteration ceiling regardless of val_every; early
    # stopping (patience-based) can still terminate sooner.
    max_iter = n_iter

    # ---- Per-iter α̃ / ESS / ‖Δα̃‖ diagnostics ----
    # Writes a TSV alongside the checkpoint with pseudocount-space
    # diagnostics defined in tkf/svb-convergence.tex
    # §Pseudocount representation:
    #   ESS_K = (Σ w_{j,K})^2 / Σ w_{j,K}^2  (effective sample size of EMA)
    #   ‖Δα̃‖_1 / ‖α̃‖_1                    (rel. change per param group)
    #   ‖α̃‖_∞                              (magnitude per param group)
    #   H(classdist)                         (entropy of classdist per d,f
    #                                         — low = class collapse /drift)
    # Header only written for fresh files; append on resume. Per-run eta
    # history starts at the current process start (ESS is cumulative only
    # from this process's first iter; acceptable for drift-level signals).
    _diag_tsv_path = checkpoint_path.replace('.npz', '_diag.tsv')
    _diag_new_file = not os.path.exists(_diag_tsv_path)
    _diag_tsv = open(_diag_tsv_path, 'a', buffering=1)
    if _diag_new_file:
        _diag_tsv.write(
            'iter\teta\tess\t'
            'norm_dom_w\tnorm_frag_w\tnorm_ext\tnorm_classdist\t'
            'drel_dom_w\tdrel_frag_w\tdrel_ext\tdrel_classdist\t'
            'entropy_cd_min\tentropy_cd_mean\n')
    _diag_prev_atilde = None
    _diag_eta_history = []

    while em_iter < max_iter:
        iter_start = time.monotonic()
        label = f"{em_iter + 1}/{n_iter}"
        _log(f"--- SVI-BW iteration {label} ---")
        _report_params(params, n_dom)

        # Step size (Robbins-Monro schedule)
        eta = (em_iter + svi_tau) ** (-svi_kappa)
        _log(f"  eta_{em_iter} = {eta:.6f}")

        # All four E-step paths (1D-constrained batched, 2D unaligned
        # batched, match-aligned batched, and the scalar
        # `_process_file_pairs` fallback used on OOM) build chi PER PAIR
        # at each pair's own t_est, coherently with per-pair-t_est
        # emissions. Pre-computing a single log_chi at t_rep here would
        # be unused (and was the source of an earlier per-pair-t bug).
        t = t_rep  # representative t kept for diagnostics / iteration logging

        # ---- E-step: sample a minibatch of pairs ----
        # Default: uniform over pairs (biases toward large families).
        # --breadth-sample: prefer under-sampled families; one pair per
        # distinct family until B pairs collected (wraps around for B>N_fam).
        # Rationale: large Pfam families contribute hundreds of pairs,
        # small ones only a handful; uniform-over-pairs overweights the
        # large ones. Breadth-first fixes this, giving each family
        # equal contribution per epoch.
        use_breadth = getattr(args, 'breadth_sample', False)
        if use_breadth and precompiled_source is not None:
            # Lazy-init family sample counter (persists across iters)
            if '_breadth_fam_counts' not in locals():
                _breadth_fam_counts = np.zeros(
                    len(_pair_idxs_by_family), dtype=np.int64)
            # Score families: (sample count) + small jitter for tie-break
            scores = _breadth_fam_counts + rng.rand(
                len(_breadth_fam_counts)) * 1e-6
            order = np.argsort(scores)  # least-used families first
            batch_indices = []
            fi = 0
            while len(batch_indices) < min(svi_batch, len(all_pairs)):
                fam_idx = order[fi % len(order)]
                pair_idxs = _pair_idxs_by_family[fam_idx]
                if pair_idxs:
                    chosen = pair_idxs[rng.randint(len(pair_idxs))]
                    batch_indices.append(chosen)
                    _breadth_fam_counts[fam_idx] += 1
                fi += 1
            batch_indices = np.array(batch_indices)
        else:
            batch_indices = rng.choice(
                len(all_pairs), size=min(svi_batch, len(all_pairs)),
                replace=False)

        use_unaligned = getattr(args, 'unaligned', False)
        use_match_aligned = getattr(args, 'match_aligned', False)

        if precompiled_source is not None:
            # Batched path: collect pre-decoded pairs, vmap the FB
            batch_pair_list = []
            t_vals = []
            for idx in batch_indices:
                fi, pair = all_pairs[idx]
                batch_pair_list.append(pair)
                t_vals.append(pair[5])

            try:
                if use_match_aligned:
                    batch_suff = _process_pairs_batched_match_aligned(
                        batch_pair_list, params, st, Q_lg, pi_lg, N,
                        dom_Qs=params.get('dom_Qs'), dom_pis=params.get('dom_pis'),
                        n_dom=n_dom, n_frag=n_frag, params=params)
                elif use_unaligned:
                    batch_suff = _process_pairs_batched_2d(
                        batch_pair_list, params, st, Q_lg, pi_lg, N,
                        dom_Qs=params.get('dom_Qs'), dom_pis=params.get('dom_pis'),
                        n_dom=n_dom, n_frag=n_frag, params=params)
                else:
                    batch_suff = _process_pairs_batched(
                        batch_pair_list, params, st, Q_lg, pi_lg, N,
                        alignment_to_states,
                        dom_Qs=params.get('dom_Qs'), dom_pis=params.get('dom_pis'),
                        classdist_params=_get_classdist_params(params),
                        n_dom=n_dom, n_frag=n_frag,
                        fb_scan_mode=_resolve_fb_scan_mode(args, N))
            except (jax.errors.JaxRuntimeError, RuntimeError) as e:
                if 'RESOURCE_EXHAUSTED' in str(e) or 'out of memory' in str(e).lower():
                    _log(f"  OOM in batched E-step, falling back to scalar")
                    batch_suff = None
                else:
                    raise

            if batch_suff is None:
                # Fallback: scalar per-pair processing
                batch_suff = _zero_suff_stats(N, n_dom)
                n_pairs_ok = 0
                for idx in batch_indices:
                    fi, pair = all_pairs[idx]
                    fam_name = _precompiled_families[fi] if fi < len(_precompiled_families) else f"fam_{fi}"
                    try:
                        result = _process_file_pairs(
                            fam_name, [], params, st,
                            Q_lg, pi_lg, N, alignment_to_states,
                            dom_Qs=params.get('dom_Qs'), dom_pis=params.get('dom_pis'),
                                classdist_params=_get_classdist_params(params),
                            n_dom=n_dom, n_frag=n_frag,
                            pre_decoded=[pair])
                    except (jax.errors.JaxRuntimeError, RuntimeError):
                        result = None
                    if result is not None:
                        for k in batch_suff:
                            if isinstance(batch_suff[k], np.ndarray):
                                batch_suff[k] = batch_suff[k] + result[k]
                            else:
                                batch_suff[k] = batch_suff[k] + result[k]
                        n_pairs_ok += 1
                if n_pairs_ok == 0:
                    batch_suff = None

            n_pairs_ok = batch_suff['n_pairs'] if batch_suff is not None else 0

        else:
            # Non-precompiled: scalar per-pair processing (unchanged)
            batch_suff = _zero_suff_stats(N, n_dom)
            n_pairs_ok = 0
            t_vals = []
            for idx in batch_indices:
                fi, pair = all_pairs[idx]
                try:
                    result = _process_file_pairs(
                        msa_files[fi], [pair], params, st,
                        Q_lg, pi_lg, N, alignment_to_states,
                        dom_Qs=params.get('dom_Qs'), dom_pis=params.get('dom_pis'),
                        classdist_params=_get_classdist_params(params),
                        n_dom=n_dom, n_frag=n_frag)
                    if result is not None:
                        t_vals.append(pair[2])
                except (jax.errors.JaxRuntimeError, RuntimeError) as e:
                    if 'RESOURCE_EXHAUSTED' in str(e) or 'out of memory' in str(e).lower():
                        _log(f"  OOM on pair {idx}, skipping")
                        result = None
                    else:
                        raise
                if result is not None:
                    for k in batch_suff:
                        if isinstance(batch_suff[k], np.ndarray):
                            batch_suff[k] = batch_suff[k] + result[k]
                        else:
                            batch_suff[k] = batch_suff[k] + result[k]
                    n_pairs_ok += 1

        if n_pairs_ok == 0:
            _log("  WARNING: no pairs processed this iteration!")
            em_iter += 1
            continue

        # Update t_rep from this batch
        if t_vals:
            t_rep = float(np.median(t_vals))
            t = t_rep

        total_ll = batch_suff['total_ll']
        # MAP objective = observed batch LL + log-prior (at params used in E-step).
        # With closed-form conjugate M-steps the MAP_LL is what gets maximized;
        # logging it here gives a monotone diagnostic (modulo SVI stochasticity).
        lp_prior = _log_prior(params, args, Q_lg=Q_lg, pi_lg=pi_lg)
        map_ll = total_ll + lp_prior
        log_probs.append(map_ll)
        _log(f"  E-step: {n_pairs_ok}/{len(batch_indices)} pairs, "
             f"obs_LL={total_ll:.1f}, log_prior={lp_prior:+.1f}, "
             f"MAP_LL={map_ll:.1f}")

        # ---- Exact count restoration via chain ----
        # Prefer per-pair exact_suffstats (computed inside the batched E-step
        # at each pair's own t_p, then summed) over the legacy
        # aggregate-n_chi-at-t_rep call. The two are mathematically
        # equivalent only when all pairs share the same t; with mixed t
        # the per-pair form is correct (additivity of the chain
        # restoration across pairs at fixed model params).
        if 'exact_ss' in batch_suff and batch_suff['exact_ss'] is not None:
            ss_batch = batch_suff['exact_ss']
        else:
            ss_batch = exact_suffstats(
                batch_suff['agg_n_chi'],
                params['main_ins'], params['main_del'], t,
                params['dom_ins'], params['dom_del'],
                params['dom_weights'], params['frag_weights'],
                params['ext_rates'])

        # Add substitution sufficient stats from batch_suff to ss_batch.
        # Per-domain match/insert/delete and per-class match/insert/delete
        # are all flowed through the EMA pipeline so that BOTH MixDom1
        # (per-domain M-step) and MixDom2 (per-class M-step) consume
        # EMA-smoothed running stats — never raw batch — to keep the two
        # paths numerically identical under classdist=identity.
        if batch_suff is not None:
            for key in ['dom_match_counts', 'dom_insert_counts', 'dom_delete_counts',
                        'class_match_counts', 'class_insert_counts',
                        'class_delete_counts', 'classdist_counts']:
                if key in batch_suff:
                    ss_batch[key] = batch_suff[key]

        # ---- SVI EMA update in posterior-pseudocount space ----
        # Paper form (eq:svi-update / eq:ema-pseudocount):
        #     α̃_{k+1} = (1 − η) α̃_k + η (α_prior + (N/|B|) s_batch)
        # M-steps downstream consume data-only running counts, which we
        # recover as lower_to_running(α̃, α_prior) = α̃ − α_prior.
        # The α̃-form is numerically identical at every iter to the legacy
        # "data-only EMA + prior-at-M-step" representation (commit
        # 5f957fb6c tests). Switching to α̃ as the primary state exposes
        # the posterior natural parameters to the diagnostics block below.
        B = n_pairs_ok
        scale = N_est / max(B, 1)
        _was_first = svi_alpha_tilde is None
        if svi_alpha_tilde is None:
            svi_alpha_tilde = {}
        svi_alpha_tilde = ema_update_pseudocount(
            svi_alpha_tilde, svb_prior, ss_batch, scale, eta)
        if _was_first:
            _log(f"    SVI: initialized α̃ "
                 f"(scale={scale:.1f}, N_est={N_est})")
        else:
            _log(f"    SVI: α̃ EMA update (eta={eta:.4f}, scale={scale:.1f}, "
                 f"N_est={N_est}, B={B})")

        # Recover data-only running counts for M-step consumption
        # (M-steps internally re-add α_prior; this matches existing behavior
        # exactly because α̃ − α_prior is the running count the old loop
        # stored in svi_running_stats).
        ss = lower_to_running(svi_alpha_tilde, svb_prior)
        _report_suff_stats_exact(ss, n_dom, n_frag)

        # ---- Pseudocount-space diagnostics (one TSV row per iter) ----
        _diag_eta_history.append(float(eta))
        _diag_atilde = svi_alpha_tilde
        _diag_ess = ess_from_weights(ema_weight_history(_diag_eta_history))

        def _diag_flat(x):
            if isinstance(x, list):
                return np.concatenate(
                    [np.asarray(y, dtype=float).ravel() for y in x])
            return np.asarray(x, dtype=float).ravel()

        def _diag_linf(key):
            if key in _diag_atilde:
                return float(np.abs(_diag_flat(_diag_atilde[key])).max())
            return 0.0

        def _diag_drel(key):
            if (_diag_prev_atilde is None or key not in _diag_prev_atilde
                    or key not in _diag_atilde):
                return float('nan')
            a = _diag_flat(_diag_atilde[key])
            p = _diag_flat(_diag_prev_atilde[key])
            num = float(np.abs(a - p).sum())
            den = float(np.abs(a).sum())
            return num / den if den > 1e-30 else 0.0

        _ent_cd_min = _ent_cd_mean = float('nan')
        if 'classdist' in params and np.asarray(params['classdist']).ndim == 3:
            _cd = np.asarray(params['classdist'])
            _cd_safe = np.maximum(_cd, 1e-30)
            _ent = -(_cd_safe * np.log(_cd_safe)).sum(axis=-1)
            _ent_cd_min = float(_ent.min())
            _ent_cd_mean = float(_ent.mean())

        _diag_tsv.write(
            f"{em_iter}\t{eta:.6f}\t{_diag_ess:.3f}\t"
            f"{_diag_linf('dom_w'):.6f}\t{_diag_linf('frag_w'):.6f}\t"
            f"{_diag_linf('ext'):.6f}\t{_diag_linf('classdist_counts'):.6f}\t"
            f"{_diag_drel('dom_w'):.6e}\t{_diag_drel('frag_w'):.6e}\t"
            f"{_diag_drel('ext'):.6e}\t{_diag_drel('classdist_counts'):.6e}\t"
            f"{_ent_cd_min:.4f}\t{_ent_cd_mean:.4f}\n")
        _log(f"    α̃ diag: ESS={_diag_ess:.2f} "
             f"‖Δα̃‖/‖α̃‖ "
             f"dom={_diag_drel('dom_w'):.2e} "
             f"frag={_diag_drel('frag_w'):.2e} "
             f"ext={_diag_drel('ext'):.2e} "
             f"cd={_diag_drel('classdist_counts'):.2e} "
             f"H(cd): min={_ent_cd_min:.3f} mean={_ent_cd_mean:.3f}")
        _diag_prev_atilde = _diag_atilde

        # ---- M-steps (closed-form, from running stats) ----
        # Prefer per-pair-t-aware POST-DIVIDE BDI sufficient stats from the
        # batched E-step (key 'top_E_B' etc.). These are the natural BDI
        # statistics (E[B]_p, E[D]_p, E[S]_p) computed PER PAIR at each
        # pair's own (t_p, λ_K, μ_K) and summed; pushing the (λ-μ) divide
        # inside the per-pair loop means the EMA operates on the natural
        # sufficient stats themselves, with no cross-iteration θ-staleness.
        # Fall back to the legacy bdi_stats_from_counts(top_n, ..., t_rep)
        # for paths that don't supply per-pair stats (scalar OOM only).
        top_n = jnp.array(ss['top_5x5'])
        top_L = float(jnp.sum(top_n[:, M]) + jnp.sum(top_n[:, D]))
        top_M = float(jnp.sum(top_n[:, E]))
        if 'top_E_B' in ss:
            top_T = float(ss['top_T_obs'])
            E_B = float(ss['top_E_B'])
            E_D = float(ss['top_E_D'])
            E_S = float(ss['top_E_S'])
        else:
            top_T = t * top_M
            E_B, E_D, E_S = bdi_stats_from_counts(
                top_n, params['main_ins'], params['main_del'], t, T=top_T)
        _log(f"    BDI: E_B={E_B:.4f}, E_D={E_D:.4f}, E_S={E_S:.4f}")

        main_ins_new, main_del_new = m_step_indel_quadratic(
            float(E_B), float(E_D), float(E_S),
            L=top_L, M=top_M, T=top_T,
            prior_alpha_lam=args.ins_prior[0], prior_alpha_mu=args.del_prior[0],
            prior_beta=args.ins_prior[1])
        main_ins_new = float(main_ins_new) if np.isfinite(main_ins_new) \
            else float(params['main_ins'])
        main_del_new = float(main_del_new) if np.isfinite(main_del_new) \
            else float(params['main_del'])
        _log(f"    M-step: lambda={main_ins_new:.6f}, mu={main_del_new:.6f}")

        # Per-domain indel rates
        dom_ins_new = np.array(params['dom_ins'], dtype=float)
        dom_del_new = np.array(params['dom_del'], dtype=float)
        for d in range(n_dom):
            dom_n = jnp.array(ss['dom_M_5x5'][d])
            nk_ID = ss['dom_kappa'][d]
            n1k_ID = ss['dom_1mkappa'][d]
            if float(dom_n.sum()) < 0.01 and nk_ID < 0.01:
                continue
            n_entries_M = float(jnp.sum(dom_n[S, :]))
            if 'dom_E_B' in ss:
                T_d = float(np.asarray(ss['dom_T_obs'])[d])
                eb = float(np.asarray(ss['dom_E_B'])[d])
                ed = float(np.asarray(ss['dom_E_D'])[d])
                es = float(np.asarray(ss['dom_E_S'])[d])
            else:
                T_d = t * (n_entries_M + n1k_ID)
                eb, ed, es = bdi_stats_from_counts(
                    dom_n, params['dom_ins'][d], params['dom_del'][d], t, T=T_d)
            nk_M = float(jnp.sum(dom_n[:, M]) + jnp.sum(dom_n[:, D]))
            n1k_M = float(jnp.sum(dom_n[:, E]))
            ni, nd = m_step_indel_quadratic(
                float(eb), float(ed), float(es),
                L=nk_M + nk_ID, M=n1k_M + n1k_ID, T=T_d,
                prior_alpha_lam=args.ins_prior[0], prior_alpha_mu=args.del_prior[0],
                prior_beta=args.ins_prior[1])
            ni = ni if np.isfinite(ni) else params['dom_ins'][d]
            nd = nd if np.isfinite(nd) else params['dom_del'][d]
            dom_ins_new[d] = float(ni)
            dom_del_new[d] = float(nd)

        # Domain weights: Dirichlet MAP
        dom_alpha = args.dom_dirichlet
        dom_w_counts = np.array(ss['dom_w'])
        dom_w_post = np.maximum(dom_w_counts + dom_alpha - 1, 0)
        dom_w_total = dom_w_post.sum()
        dom_weights_new = dom_w_post / dom_w_total if dom_w_total > 1e-10 \
            else np.ones(n_dom) / n_dom

        # Fragment weights: Dirichlet MAP per domain
        frag_alpha = args.frag_dirichlet
        frag_weights_new = np.zeros((n_dom, n_frag))
        if getattr(args, 'banded_frag_init', False):
            frag_weights_new = np.array(params['frag_weights']).copy()
            _log(f"    frag_weights: PINNED at banded init "
                 f"(frag_weights[d,0]=1)")
        else:
            for d in range(n_dom):
                fw_post = np.maximum(np.array(ss['frag_w'][d]) + frag_alpha - 1, 0)
                fw_total = fw_post.sum()
                frag_weights_new[d] = fw_post / fw_total if fw_total > 1e-10 \
                    else np.ones(n_frag) / n_frag

        # Fragment extension: MixDom2 row-normalized Dirichlet MAP
        freeze_offdiag = bool(getattr(args, 'freeze_ext_offdiag', False))
        banded = bool(getattr(args, 'banded_frag_init', False))
        if banded:
            from tkfmixdom.jax.train.restricted_mstep import (
                banded_3fc_pseudocounts, banded_3fc_ext_mask)
            ext_pseudo_2d, term_pseudo_1d = banded_3fc_pseudocounts(
                n_frag, args.ext_alpha, args.ext_beta)
            ext_mask_2d, term_mask_1d = banded_3fc_ext_mask()
        ext_rates_new = np.zeros((n_dom, n_frag, n_frag))
        for d in range(n_dom):
            for f in range(n_frag):
                row_ext = ss['ext'][d, f, :]
                term = ss['term'][d, f]
                row_counts = np.append(row_ext, term)
                pseudocounts = np.append(
                    np.full(n_frag, args.ext_alpha - 1.0),
                    args.ext_beta - 1.0)
                if banded:
                    pseudocounts[:n_frag] = ext_pseudo_2d[f, :]
                    pseudocounts[n_frag] = term_pseudo_1d[f]
                elif freeze_offdiag:
                    # Zero off-diagonal pseudocounts: with diagonal init
                    # the prior cannot create off-diag mass, and FB-restored
                    # off-diag counts are vanishing (forbidden chi entries
                    # sit at the 1e-30 log-floor). Off-diag stays at 0.
                    pseudocounts[:n_frag] = 0.0
                    pseudocounts[f] = args.ext_alpha - 1.0
                posterior = np.maximum(row_counts + pseudocounts, 0.0)
                if banded:
                    posterior[:n_frag] = np.where(
                        ext_mask_2d[f, :], posterior[:n_frag], 0.0)
                    if not term_mask_1d[f]:
                        posterior[n_frag] = 0.0
                total = posterior.sum()
                if total > 1e-10:
                    normalized = posterior / total
                    # See non-SVI driver above for rationale: termination
                    # prob must be > 0 to avoid degenerate geometric
                    # fragment-length distribution; surface as a hard
                    # failure rather than silently clipping.
                    # Banded mode permits structurally-zero termination
                    # (FragMid: term_mask=False).
                    if not (banded and not term_mask_1d[f]):
                        assert normalized[n_frag] > 1e-12, (
                            f"ext_rates M-step at (d={d}, f={f}) produced "
                            f"termination prob = {normalized[n_frag]:.3e} "
                            f"(ext_counts={row_ext}, term_count={term:g}, "
                            f"ext_beta={args.ext_beta}). Increase --ext-beta "
                            f"above 1 or check for degenerate data.")
                    ext_rates_new[d, f, :] = normalized[:n_frag]
                else:
                    if banded:
                        ext_rates_new[d, f, :] = np.array(
                            params['ext_rates'])[d, f, :]
                    else:
                        ext_rates_new[d, f, f] = 0.3

        old_params = params  # preserve dynamic class params
        params = {
            'main_ins': main_ins_new, 'main_del': main_del_new,
            'dom_ins': np.array(dom_ins_new), 'dom_del': np.array(dom_del_new),
            'dom_weights': np.array(dom_weights_new),
            'frag_weights': np.array(frag_weights_new),
            'ext_rates': np.array(ext_rates_new),
        }
        # Carry forward per-domain subst + MixDom2 site class params
        for _k in ('dom_S_exch', 'dom_Qs', 'dom_pis',
                    'n_classes', 'classdist', 'class_pis', 'class_S_exch'):
            if _k in old_params:
                params[_k] = old_params[_k]

        # Per-domain substitution M-step
        if getattr(args, 'estimate_subst', False):
            from tkfmixdom.jax.core.ctmc import (
                m_step_subst_option1, holmes_rubin_expected_stats)
            S_lg = np.array(Q_lg / np.maximum(np.array(pi_lg)[None, :], 1e-30)
                            * (1.0 - np.eye(AA)))
            S_lg = (S_lg + S_lg.T) / 2
            dom_Qs_new = np.zeros((n_dom, AA, AA))
            dom_pis_new = np.zeros((n_dom, AA))
            dom_S_new = np.zeros((n_dom, AA, AA))
            # Use EMA-smoothed running stats (consistent with the rest of
            # the SVI pipeline). The MixDom2 per-class M-step also pulls
            # from `ss[...]` (EMA), so under classdist=identity the two
            # paths see numerically identical sufficient statistics.
            for dd in range(n_dom):
                mc = ss['dom_match_counts'][dd]
                ic = ss['dom_insert_counts'][dd]
                dc = ss['dom_delete_counts'][dd]
                total_mc = mc.sum()
                if total_mc < 1.0:
                    dom_Qs_new[dd] = params.get('dom_Qs', np.tile(Q_lg, (n_dom,1,1)))[dd]
                    dom_pis_new[dd] = params.get('dom_pis', np.tile(pi_lg, (n_dom,1)))[dd]
                    dom_S_new[dd] = params.get('dom_S_exch', np.tile(S_lg[None], (n_dom,1,1)))[dd]
                    continue
                # V_d is the equilibrium-character composition count under
                # the joint pair HMM P(x, y | t): every position where a
                # residue is an i.i.d. pi-draw contributes. The ancestral
                # residues at match positions (mc.sum(axis=1)[a]) AND at
                # delete positions (dc[a]) are pi-draws at time 0; the
                # descendant residues at insert positions (ic[b]) are
                # pi-draws at the time of insertion. See tkf.tex L755.
                # (Match-position descendants are CTMC bridge endpoints,
                # contributing via U_col_sum inside m_step_subst_option1's
                # V_prime construction, NOT raw V.)
                V_d = mc.sum(axis=1) + ic + dc
                Q_old = params.get('dom_Qs', np.tile(Q_lg, (n_dom,1,1)))[dd]
                pi_old = params.get('dom_pis', np.tile(pi_lg, (n_dom,1)))[dd]
                # Prefer per-pair-t-aggregated W/U from the batched E-step
                # (key 'dom_W'). These were accumulated PER PAIR at each
                # pair's own t_p, so the M-step is purely closed-form on
                # time-free aggregated stats. Fall back to the legacy
                # HR-after-EMA loop at t_rep when 'dom_W' is missing
                # (scalar OOM path / older E-step code).
                if 'dom_W' in ss:
                    W_d = np.asarray(ss['dom_W'])[dd].copy()
                    U_d = np.asarray(ss['dom_U'])[dd].copy()
                else:
                    W_d = np.zeros(AA)
                    U_d = np.zeros((AA, AA))
                    for a_idx in range(AA):
                        for b_idx in range(AA):
                            if mc[a_idx, b_idx] < 1e-10:
                                continue
                            w_ab, u_ab = holmes_rubin_expected_stats(
                                jnp.array(Q_old), jnp.array(pi_old), t, a_idx, b_idx)
                            W_d += mc[a_idx, b_idx] * np.array(w_ab)
                            U_d += mc[a_idx, b_idx] * np.array(u_ab)
                # Dispatch on --subst-mode for restricted regimes.
                _subst_mode = getattr(args, 'subst_mode', 'standard')
                S_old_d = (np.asarray(params['dom_S_exch'])[dd]
                           if 'dom_S_exch' in params else S_lg)
                if _subst_mode == 'rescaling-rates' or (
                    _subst_mode == 'alt-tied-pi-rescaling'
                    and (em_iter % 2 == 1)
                ):
                    from tkfmixdom.jax.train.restricted_mstep import (
                        m_step_subst_rescaling)
                    S_d, pi_d, _sigma = m_step_subst_rescaling(
                        W_d, U_d, S_old_d, pi_old)
                    Q_d = S_d * pi_d[None, :]
                    np.fill_diagonal(Q_d, 0.0)
                    Q_d[np.diag_indices(AA)] = -Q_d.sum(axis=1)
                elif _subst_mode == 'rescaling-rates-and-pi':
                    from tkfmixdom.jax.train.restricted_mstep import (
                        m_step_subst_rescaling_pi)
                    sigma_warm = (float(np.linalg.norm(S_old_d.ravel())
                                       / max(np.linalg.norm(S_lg.ravel()), 1e-30)))
                    S_d, pi_d, _sigma = m_step_subst_rescaling_pi(
                        W_d, U_d, V_d, S_old_d, sigma_warm, pi_old)
                    Q_d = S_d * pi_d[None, :]
                    np.fill_diagonal(Q_d, 0.0)
                    Q_d[np.diag_indices(AA)] = -Q_d.sum(axis=1)
                elif _subst_mode == 'frozen-pi':
                    S_d, _pi_drop, _ = m_step_subst_option1(
                        W_d, U_d, V_d,
                        S_prior=S_lg, pi_prior=np.array(pi_lg),
                        pi_pseudo=args.pi_pseudo, S_pseudo=args.S_pseudo)
                    pi_d = pi_old
                    Q_d = S_d * pi_d[None, :]
                    np.fill_diagonal(Q_d, 0.0)
                    Q_d[np.diag_indices(AA)] = -Q_d.sum(axis=1)
                else:
                    # tied-pi at the per-domain level reduces to standard
                    # (each "domain" is its own class; tying happens at the
                    # MixDom2 site-class level instead). Standard / tied-pi /
                    # alt-tied-pi-rescaling-on-even-iter all fall through.
                    S_d, pi_d, Q_d = m_step_subst_option1(
                        W_d, U_d, V_d,
                        S_prior=S_lg, pi_prior=np.array(pi_lg),
                        pi_pseudo=args.pi_pseudo, S_pseudo=args.S_pseudo)
                dom_Qs_new[dd] = Q_d
                dom_pis_new[dd] = pi_d
                dom_S_new[dd] = S_d
            params['dom_Qs'] = dom_Qs_new
            params['dom_pis'] = dom_pis_new
            params['dom_S_exch'] = dom_S_new

        # ---- MixDom2 per-fragment site class M-step ----
        # Implements substitution-mstep.tex sec:mstep-mixture (alternating closed-form):
        #   step (a) update S_exch (gamma fixed)   — eq:mix-S-update
        #   step (b) update gamma_c (S_exch fixed) — eq:mix-gamma-update
        #   pi_c update is shared across both steps — eq:mix-pi-update
        # classdist update is the standard Dirichlet MAP from posterior assignments.
        if params.get('n_classes', 0) > 1 and 'classdist' in params:
            n_cls = params['n_classes']
            # alpha_classdist is the symmetric Dirichlet concentration per
            # (d, f); pseudocount = alpha - 1 added to classdist counts.
            # Default 1.0 (uniform → zero pseudocount → plain MLE).
            cd_alpha = float(getattr(args, 'classdist_dirichlet', 1.0))
            # pi_pseudo is the effective sample size N_pi of an LG-informed
            # Dirichlet(N_pi * pi_LG_i + 1) on each class_pis[c] — matches
            # the per-domain subst M-step's use of pi_pseudo in
            # m_step_subst_option1, so --pi-pseudo has one meaning everywhere.
            pi_pseudo = getattr(args, 'pi_pseudo', 10.0)
            pi_lg_np = np.asarray(pi_lg)

            if 'classdist_counts' in ss and ss['classdist_counts'] is not None:
                # classdist M-step: symmetric Dirichlet(cd_alpha) MAP,
                # unless --freeze-classdist is set (keep at initial uniform
                # values; tests whether free classdist is the source of
                # optimization-landscape "flabbiness").
                if getattr(args, 'freeze_classdist', False):
                    cd_new = np.asarray(params['classdist'])
                    _log(f"    classdist FROZEN (--freeze-classdist): "
                         f"keeping initial uniform values")
                else:
                    cd_counts = np.array(ss['classdist_counts'])
                    cd_new = cd_counts + (cd_alpha - 1.0)
                    cd_new = np.maximum(cd_new, 1e-6)
                    cd_new = cd_new / cd_new.sum(axis=2, keepdims=True)
                    params['classdist'] = cd_new

                if 'class_match_counts' in ss:
                    from tkfmixdom.jax.core.ctmc import holmes_rubin_expected_stats
                    cm = np.array(ss['class_match_counts'])    # (C, A, A)
                    ci = np.array(ss['class_insert_counts'])   # (C, A)
                    # class_delete_counts mirrors dom_delete_counts in the
                    # MixDom1 path: V = insert + delete counts is the
                    # equilibrium-character count for the per-class pi
                    # M-step. Older checkpoints may not have it; fall back
                    # to zeros (degrades gracefully to the previous, biased
                    # behavior, but new training accumulates it correctly).
                    cd_del = np.array(ss.get(
                        'class_delete_counts', np.zeros_like(ci)))  # (C, A)
                    S_old = np.array(params['class_S_exch'])   # (C, A, A)
                    pi_old = np.array(params['class_pis'])     # (C, A)

                    # ---- Per-class Holmes-Rubin sufficient stats ----
                    # Prefer per-pair-t-aggregated W/U from the batched
                    # E-step (key 'class_W'). If absent (older E-step or
                    # scalar OOM fallback), fall back to the legacy
                    # HR-after-EMA loop at t_rep.
                    if 'class_W' in ss:
                        W_c = np.asarray(ss['class_W']).copy()
                        U_pair_c = np.asarray(ss['class_U']).copy()
                    else:
                        W_c = np.zeros((n_cls, AA))
                        U_pair_c = np.zeros((n_cls, AA, AA))
                        for cc in range(n_cls):
                            Q_c = S_old[cc] * pi_old[cc, None, :]
                            np.fill_diagonal(Q_c, 0.0)
                            Q_c[np.diag_indices(AA)] = -Q_c.sum(axis=1)
                            Wc_acc = np.zeros(AA)
                            Uc_acc = np.zeros((AA, AA))
                            for a_idx in range(AA):
                                for b_idx in range(AA):
                                    w_ab = cm[cc, a_idx, b_idx]
                                    if w_ab < 1e-10:
                                        continue
                                    w_hat, u_hat = holmes_rubin_expected_stats(
                                        jnp.array(Q_c), jnp.array(pi_old[cc]),
                                        t, a_idx, b_idx)
                                    Wc_acc += w_ab * np.array(w_hat)
                                    Uc_acc += w_ab * np.array(u_hat)
                            W_c[cc] = Wc_acc
                            U_pair_c[cc] = Uc_acc

                    # ---- Joint per-class (S_c, pi_c) M-step ----
                    # Calls m_step_subst_option1 (paper's Option 2 from
                    # tkf.tex L823-824): iterative coordinate ascent on
                    # the strictly-concave joint objective. Converges to
                    # the EXACT joint MLE up to floating-point precision
                    # (geometric convergence; n_iter=50, tol=1e-10).
                    #
                    # Replaces the earlier paper Option 1 (one-shot
                    # empirical pi normalization + closed-form S, exact
                    # only at the F81 limit) so the per-class M-step is
                    # numerically identical to MixDom1's per-domain M-step
                    # under identity classdist.
                    #
                    # V_c = match-pos ancestor + insert + delete (joint
                    # pair HMM, tkf.tex L755). m_step_subst_option1
                    # internally builds V'_c = V_c + U_col_sum and applies
                    # the LG pseudocount as before.
                    from tkfmixdom.jax.core.ctmc import m_step_subst_option1
                    from tkfmixdom.jax.train.restricted_mstep import (
                        class_mixture_mstep)
                    cm_anc = cm.sum(axis=2)  # (C, A): match-pos ancestor count per class
                    V_per_class = np.stack(
                        [cm_anc[cc] + ci[cc] + cd_del[cc]
                         for cc in range(n_cls)], axis=0)
                    S_lg = (np.array(Q_lg / np.maximum(
                                np.array(pi_lg)[None, :], 1e-30))
                            * (1.0 - np.eye(AA)))
                    S_lg = (S_lg + S_lg.T) / 2.0
                    subst_mode = getattr(args, 'subst_mode', 'standard')
                    n_tied = int(getattr(args, 'n_tied', 1))

                    if getattr(args, 'estimate_sexch', False):
                        S_pseudo = getattr(args, 'S_pseudo', 5.0)
                        S_new, pi_new = class_mixture_mstep(
                            subst_mode, em_iter,
                            W_c, U_pair_c, V_per_class,
                            S_old, pi_old,
                            S_prior=S_lg, pi_prior=pi_lg_np,
                            pi_pseudo=pi_pseudo, S_pseudo=S_pseudo,
                            n_tied=n_tied, log_fn=_log)
                        params['class_S_exch'] = S_new
                    else:
                        # estimate-sexch off: S held fixed at S_old; only
                        # pi moves via the standard / tied-pi solver.
                        # Rescaling-rates is a no-op here (S frozen by
                        # both the mode and the flag → keep current σ).
                        if subst_mode in ('rescaling-rates',
                                          'rescaling-rates-and-pi'):
                            # S held fixed by --no-estimate-sexch; for these
                            # modes pi cannot be updated independently of σ
                            # (rescaling-rates) or is jointly determined with
                            # σ from S (rescaling-rates-and-pi). Without the
                            # σ update path that --estimate-sexch enables,
                            # the only consistent action is to leave pi at
                            # warm.
                            pi_new = pi_old.copy()
                        else:
                            # Use frozen-pi → pi update via standard solver
                            # but treat the existing flag (no-estimate-sexch)
                            # as authoritative for S only.
                            pi_new = np.zeros_like(pi_old)
                            for cc in range(n_cls):
                                V_c = V_per_class[cc]
                                _, pi_c, _ = m_step_subst_option1(
                                    W_c[cc], U_pair_c[cc], V_c,
                                    S_prior=None, pi_prior=pi_lg_np,
                                    pi_pseudo=pi_pseudo, S_pseudo=0.0)
                                pi_new[cc] = pi_c
                            if subst_mode == 'tied-pi':
                                # Pool pi across blocks to honour the tying
                                # constraint when classes share a block.
                                from tkfmixdom.jax.train.restricted_mstep import (
                                    tied_pi_blocks)
                                blocks = tied_pi_blocks(n_cls, n_tied)
                                for blk in blocks:
                                    # Weight by V'_c contributions per class
                                    # for the pooled pi (closed-form reduction
                                    # to F81 limit; exact when S held fixed).
                                    pooled = pi_new[blk].mean(axis=0)
                                    pooled = pooled / max(pooled.sum(), 1e-30)
                                    for c in blk:
                                        pi_new[c] = pooled
                    params['class_pis'] = pi_new

                    # ---- LG-anchor proximal pull (post-M-step) ----
                    # After the mixture M-step finishes, pull class_pis and
                    # class_S_exch a fraction s ∈ [0, 1] toward the LG
                    # equilibrium / LG exchangeability. Equivalent to a
                    # Gaussian penalty with weight s/(1−s) on drift from LG.
                    # Default s=0 preserves the old behaviour exactly.
                    lg_anchor = float(
                        getattr(args, 'lg_anchor_strength', 0.0))
                    if lg_anchor > 0.0:
                        s = min(max(lg_anchor, 0.0), 1.0)
                        pi_arr = np.asarray(params['class_pis'])
                        params['class_pis'] = (
                            (1.0 - s) * pi_arr
                            + s * pi_lg_np[None, :]).astype(pi_arr.dtype)
                        if 'class_S_exch' in params:
                            S_arr = np.asarray(params['class_S_exch'])
                            S_lg_anchor = (np.array(
                                Q_lg / np.maximum(
                                    np.array(pi_lg)[None, :], 1e-30))
                                * (1.0 - np.eye(AA)))
                            S_lg_anchor = (S_lg_anchor + S_lg_anchor.T) / 2.0
                            params['class_S_exch'] = (
                                (1.0 - s) * S_arr
                                + s * S_lg_anchor[None, :, :]).astype(S_arr.dtype)
                        _log(f"    LG-anchor pull: s={s:.3f} applied to "
                             f"class_pis / class_S_exch")

                    _log(f"    Site classes M-step: C={n_cls}")
                    _log(f"      classdist range: [{cd_new.min():.3f}, {cd_new.max():.3f}]")
                    S_diag_rms = float(np.sqrt(
                        ((params['class_S_exch'] - S_old) ** 2).mean()))
                    _log(f"      class_S_exch RMS change: {S_diag_rms:.4f}")
                    _log(f"      class_pis pi[0] range: "
                         f"[{params['class_pis'][:,0].min():.4f}, "
                         f"{params['class_pis'][:,0].max():.4f}]")
                    # Class-differentiation diagnostics:
                    # - classdist entropy per (d, f): 0 = identity, log(C) = uniform
                    # - class_pis L1 pairwise spread: 0 = all classes identical
                    # - class_S_exch L2 pairwise spread
                    cd_safe = np.maximum(np.asarray(cd_new), 1e-30)
                    cd_ent = -(cd_safe * np.log(cd_safe)).sum(axis=-1)
                    _log(f"      classdist H(c|d,f): "
                         f"min={float(cd_ent.min()):.4f} "
                         f"mean={float(cd_ent.mean()):.4f} "
                         f"max={float(cd_ent.max()):.4f} "
                         f"(uniform={float(np.log(n_cls)):.4f})")
                    pi_arr = np.asarray(params['class_pis'])  # (C, A)
                    pi_l1 = np.zeros((n_cls, n_cls))
                    for ci in range(n_cls):
                        for cj in range(ci + 1, n_cls):
                            pi_l1[ci, cj] = pi_l1[cj, ci] = float(
                                np.abs(pi_arr[ci] - pi_arr[cj]).sum())
                    pi_offdiag = pi_l1[np.triu_indices(n_cls, k=1)]
                    _log(f"      class_pis L1 pairwise: "
                         f"min={float(pi_offdiag.min()):.4f} "
                         f"mean={float(pi_offdiag.mean()):.4f} "
                         f"max={float(pi_offdiag.max()):.4f}")
                    S_arr = np.asarray(params['class_S_exch'])  # (C, A, A)
                    S_l2 = np.zeros((n_cls, n_cls))
                    for ci in range(n_cls):
                        for cj in range(ci + 1, n_cls):
                            S_l2[ci, cj] = S_l2[cj, ci] = float(
                                np.linalg.norm(S_arr[ci] - S_arr[cj]))
                    S_offdiag = S_l2[np.triu_indices(n_cls, k=1)]
                    _log(f"      class_S_exch L2 pairwise: "
                         f"min={float(S_offdiag.min()):.4f} "
                         f"mean={float(S_offdiag.mean()):.4f} "
                         f"max={float(S_offdiag.max()):.4f}")

        # ---- Next iteration ----
        em_iter += 1
        iter_time = time.monotonic() - iter_start
        _log(f"  Iteration done in {iter_time:.1f}s")

        if len(log_probs) >= 2:
            delta = log_probs[-1] - log_probs[-2]
            _log(f"  LL change: {delta:+.4f}")

        _save_svi_checkpoint()
        _log(f"  [checkpoint saved]")

        if save_every_iter:
            iter_path = checkpoint_path.replace('.npz', f'_iter{em_iter}.npz')
            _save_checkpoint(iter_path, params, _zero_suff_stats(N, n_dom), 0,
                             em_iter, log_probs, config, None, t_rep, None, None)
            _log(f"  [iter {em_iter} params saved to {iter_path}]")

        # ---- Validation ----
        if val_every > 0 and em_iter % val_every == 0:
            val_split_fams = None
            _sf = getattr(args, 'split_file', None) or locals().get('split_file', None)
            if not _sf:
                for _cand in [
                    os.path.expanduser('~/bio-datasets/data/pfam/seed/splits/v1.json'),
                os.path.expanduser('~/bio-datasets/data/pfam-seed/splits/v1.json'),
                    os.path.join(args.msa_dir, 'splits', 'v1.json'),
                ]:
                    if os.path.exists(_cand):
                        _sf = _cand
                        break
            if _sf and os.path.exists(_sf):
                import json as json_mod2
                with open(_sf) as f2:
                    sd = json_mod2.load(f2)
                val_split_fams = set(sd.get('val', []))
            if val_split_fams:
                val_ll, val_n, val_mean = _evaluate_on_split(
                    params, 'val', val_split_fams, args.msa_dir,
                    n_dom, n_frag, t_rep, Q_lg, pi_lg,
                    max_pairs_per_fam=2,
                    unaligned=getattr(args, 'unaligned', False),
                    match_aligned=getattr(args, 'match_aligned', False))
                _log(f"  [VAL iter {em_iter}] LL={val_ll:.1f}, "
                     f"n_pairs={val_n}, mean={val_mean:.1f}/pair")
                # Reset stopper state if the val set has changed since the
                # checkpointed best_val_ll was computed (e.g. after a code
                # change to _evaluate_on_split). Stale baselines from a
                # different val set make LL comparisons meaningless.
                if best_val_n_pairs and val_n != best_val_n_pairs:
                    _log(f"  [val set changed: {best_val_n_pairs} -> {val_n} "
                         f"pairs; resetting stopper baseline]")
                    best_val_ll = -float('inf')
                    val_no_improve = 0
                if val_ll > best_val_ll:
                    best_val_ll = val_ll
                    best_val_n_pairs = val_n
                    val_no_improve = 0
                    # Save best-val checkpoint
                    best_path = checkpoint_path.replace('.npz', '_best_val.npz')
                    _save_checkpoint(best_path, params,
                                     _zero_suff_stats(N, n_dom), 0,
                                     em_iter, log_probs, config,
                                     None, t_rep, None, None)
                    _log(f"  [NEW BEST val LL — saved to {best_path}]")
                else:
                    val_no_improve += 1
                    _log(f"  [val no improve {val_no_improve}/{patience}]")
                # Early stop: patience exhausted after a small warm-up.
                # --n-iter is the MAX; --min-iter is a floor (default 10)
                # to let the EMA settle before we trust "no improvement"
                # as a stationarity signal. The outer `while em_iter <
                # n_iter` handles the max-iters stop.
                min_iter = getattr(args, 'min_iter', 10)
                if em_iter >= min_iter and val_no_improve >= patience:
                    _log(f"  [EARLY STOP: val LL not improving for "
                         f"{patience} evals (min_iter={min_iter} reached); "
                         f"best val_LL={best_val_ll:.1f}]")
                    break

    # ---- Final report ----
    _log(f"\n{'='*60}")
    _log(f"SVI Baum-Welch complete: {len(log_probs)} iterations")
    if log_probs:
        _log(f"Final MAP LL: {log_probs[-1]:.2f}")
        _log(f"MAP LL history: {['%.2f' % x for x in log_probs]}")
    _log(f"Model saved to: {checkpoint_path}")
    _report_params(params, n_dom)
    _log(f"{'='*60}")


# ============================================================
# Eval-only mode
# ============================================================
def eval_only(args):
    """Evaluate a trained checkpoint on held-out data (E-step only, no M-step).

    Loads params from checkpoint, runs constrained 1D FB on cherry pairs from
    the specified families/split, and reports total LL, n_pairs, LL/pair.
    """
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tkfmixdom.jax.core.protein import rate_matrix_lg
    from tkfmixdom.jax.models.mixdom import (
        build_nested_trans, n_states as mixdom_n_states,
        state_types as mixdom_state_types
    )
    from tkfmixdom.jax.simulate.msa import alignment_to_states
    from tkfmixdom.jax.dp.hmm import safe_log
    from tkfmixdom.jax.util.sto_index import StoIndex

    # ---- Load checkpoint ----
    checkpoint_path = args.checkpoint
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        _log(f"ERROR: --eval-only requires a valid --checkpoint path")
        sys.exit(1)

    _log(f"Loading checkpoint: {checkpoint_path}")
    (params, _suff_stats, _step_file_idx, em_iter, log_probs, config,
     _manifest, t_rep, _budget_files, _family_timings) = \
        _load_checkpoint(checkpoint_path)

    n_dom = config.get('n_dom', len(params['dom_ins']))
    n_frag = config.get('n_frag', 1)
    if 'frag_weights' in params and params['frag_weights'].ndim == 2:
        n_frag = params['frag_weights'].shape[1]

    N = mixdom_n_states(n_dom, n_frag)
    st = mixdom_state_types(n_dom, n_frag)
    Q_lg, pi_lg = rate_matrix_lg()

    # Use t_rep from checkpoint, fall back to 1.0
    t = t_rep if t_rep is not None else 1.0

    _log(f"  Model: {n_dom} domains, {n_frag} fragments, {N} states, t={t:.3f}")
    _log(f"  Checkpoint EM iter: {em_iter}")
    _report_params(params, n_dom)

    # ---- Resolve MSA directory ----
    from tkfmixdom.jax.util.bio_datasets import (
        apply_bio_datasets_arg, resolve_data_dir, ensure_symlinks)
    apply_bio_datasets_arg(args)
    _msa_dir_is_default = (args.msa_dir == 'pfam/')
    if _msa_dir_is_default:
        msa_dir_resolved = str(resolve_data_dir("pfam/seed", local_fallback=args.msa_dir))
        if msa_dir_resolved != args.msa_dir:
            _log(f"  bio-datasets: {args.msa_dir} -> {msa_dir_resolved}")
            ensure_symlinks(
                __import__('pathlib').Path(msa_dir_resolved),
                __import__('pathlib').Path(args.msa_dir), pattern="*.sto")
            ensure_symlinks(
                __import__('pathlib').Path(msa_dir_resolved),
                __import__('pathlib').Path(args.msa_dir), pattern="*.sto.gz")
        args.msa_dir = msa_dir_resolved

    # ---- Find MSA files ----
    import glob as glob_mod
    if args.families:
        family_list = [f.strip() for f in args.families.split(',')]
        msa_files = []
        for fam in family_list:
            for ext in ['.sto', '.sto.gz', '.stockholm', '.stockholm.gz']:
                p = os.path.join(args.msa_dir, fam + ext)
                if os.path.exists(p):
                    msa_files.append(p)
                    break
            else:
                _log(f"Warning: {fam} not found in {args.msa_dir}")
    else:
        msa_files = sorted(
            glob_mod.glob(os.path.join(args.msa_dir, '*.sto')) +
            glob_mod.glob(os.path.join(args.msa_dir, '*.sto.gz')))

    if not msa_files:
        _log(f"No MSA files found in {args.msa_dir}")
        sys.exit(1)
    _log(f"Found {len(msa_files)} MSA files")

    # ---- Data split ----
    if args.split:
        split_file = args.split_file
        if not split_file:
            for candidate in [
                os.path.expanduser('~/bio-datasets/data/pfam/seed/splits/v1.json'),
                os.path.expanduser('~/bio-datasets/data/pfam-seed/splits/v1.json'),
                os.path.join(args.msa_dir, 'splits', 'v1.json'),
            ]:
                if os.path.exists(candidate):
                    split_file = candidate
                    break
        if split_file and os.path.exists(split_file):
            import json as json_mod
            with open(split_file) as f:
                split_data = json_mod.load(f)
            split_fams = set(split_data.get(args.split, []))
            if not split_fams:
                _log(f"ERROR: split '{args.split}' not found in {split_file}")
                sys.exit(1)
            msa_files = [f for f in msa_files
                         if os.path.basename(f).split('.')[0] in split_fams]
            _log(f"  Split file: {split_file}")
            _log(f"  Using {args.split} split: {len(msa_files)} families "
                 f"(of {len(split_fams)} in split)")
        else:
            clan_file = args.clan_file or _find_clan_file(args.msa_dir)
            if not clan_file:
                _log("ERROR: --split requires --split-file or a clan file.")
                sys.exit(1)
            ratios = tuple(float(r) for r in args.split_ratios.split(','))
            split_indices = clan_aware_split(
                msa_files, clan_file, args.split, ratios, args.split_seed)
            msa_files = [msa_files[i] for i in split_indices]
            _log(f"  Ad-hoc {args.split} split: {len(msa_files)} families")

    # ---- Eval via _evaluate_on_split (batched forward-only, vectorized emissions) ----
    _log(f"\n{'='*60}")
    _log(f"Eval-only mode: {len(msa_files)} families, t={t:.3f}")
    _log(f"  Backend: {jax.default_backend()}, devices: {jax.devices()}")
    _log(f"{'='*60}\n")

    split_fams = set(os.path.basename(f).split('.')[0] for f in msa_files)
    split_name = args.split if args.split else 'all'
    total_ll, n_pairs, ll_per_pair = _evaluate_on_split(
        params, split_name, split_fams, args.msa_dir, n_dom, n_frag,
        t, Q_lg, pi_lg,
        unaligned=getattr(args, 'unaligned', False),
        match_aligned=getattr(args, 'match_aligned', False))

    _log(f"\n{'='*60}")
    _log(f"EVAL SUMMARY")
    _log(f"  Checkpoint:  {checkpoint_path}")
    _log(f"  EM iter:     {em_iter}")
    _log(f"  Pairs:       {n_pairs}")
    _log(f"  Total LL:    {total_ll:.4f}")
    _log(f"  LL/pair:     {ll_per_pair:.6f}")
    elapsed = 0.0  # already reported by _evaluate_on_split
    if args.split:
        _log(f"  Split:       {args.split}")
    _log(f"{'='*60}")

    # Print a machine-readable summary line to stdout
    split_str = args.split if args.split else "all"
    print(f"eval\t{os.path.basename(checkpoint_path)}\t{split_str}\t"
          f"{len(split_fams)}\t{n_pairs}\t{total_ll:.4f}\t{ll_per_pair:.6f}")


# ============================================================
# ============================================================
# Inspect mode
# ============================================================
def inspect(args):
    path = args.inspect
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    (params, suff_stats, step_file_idx, em_iter, log_probs, config,
     manifest, t_rep, budget_files, family_timings) = _load_checkpoint(path)
    print(f"Checkpoint: {path}")
    print(f"Config: {json.dumps(config, indent=2)}")
    print(f"EM iteration: {em_iter}")
    print(f"Files done in current step: {step_file_idx}")
    print(f"Log-likelihoods: {log_probs}")
    if manifest is not None:
        print(f"Pair manifest: {len(manifest)} pairs")
    if t_rep is not None:
        print(f"Representative t: {t_rep:.3f}")
    if budget_files is not None:
        print(f"Budget-selected families: {len(budget_files)}")
    if family_timings is not None:
        timings = np.array(family_timings)
        print(f"Family timings: {len(timings)} families, "
              f"mean={timings.mean():.3f}s, total={timings.sum():.1f}s")
    print()
    n_dom = config.get('n_dom', len(params['dom_ins']))
    _report_params(params, n_dom)


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train MixDom on Pfam via alignment-constrained Baum-Welch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test:
  python train_pfam.py --msa-dir pfam/ --families PF00001,PF00002 --n-dom 2 --n-iter 3

  # Train on all of Pfam:
  python train_pfam.py --msa-dir pfam/ --n-dom 4 --n-frag 2 --n-iter 20

  # Budget 4h total over 10 EM iterations (samples families on iter 1):
  python train_pfam.py --msa-dir pfam/ --n-dom 3 --n-iter 10 --budget-hours 4

  # Resume interrupted run:
  python train_pfam.py --msa-dir pfam/ --checkpoint pfam/train_d4_f2.npz

  # Inspect trained model:
  python train_pfam.py --inspect pfam/train_d4_f2.npz
""")

    parser.add_argument('--inspect', type=str, default=None,
                        help='Inspect a checkpoint file and exit')
    parser.add_argument('--msa-dir', type=str, default='pfam/',
                        help='Directory with Stockholm MSA files (default: pfam/)')
    from tkfmixdom.jax.util.bio_datasets import add_bio_datasets_arg
    add_bio_datasets_arg(parser)
    parser.add_argument('--families', type=str, default=None,
                        help='Comma-separated family IDs (default: all .sto in msa-dir)')
    parser.add_argument('--n-dom', type=int, default=3,
                        help='Number of domains (default: 3)')
    parser.add_argument('--n-frag', type=int, default=2,
                        help='Fragments per domain (default: 2)')
    parser.add_argument('--n-classes', type=int, default=0,
                        help='Site classes per fragment (default: max(n_dom, n_frag)). '
                             'Set 0 for auto = max(n_dom, n_frag).')
    parser.add_argument('--init-ins', type=float, default=0.1,
                        help='Initial insertion rate (default: 0.1)')
    parser.add_argument('--init-del', type=float, default=0.2,
                        help='Initial deletion rate (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--n-iter', type=int, default=15,
                        help='Max EM iterations (default: 15)')
    parser.add_argument('--convergence-tol', type=float, default=0.1,
                        help='|delta LL| convergence threshold (default: 0.1)')
    parser.add_argument('--ins-prior', type=float, nargs=2, default=[2.0, 10.0],
                        metavar=('A', 'B'), help='Gamma prior for ins rate (default: 2 10)')
    parser.add_argument('--del-prior', type=float, nargs=2, default=[2.0, 10.0],
                        metavar=('A', 'B'), help='Gamma prior for del rate (default: 2 10)')
    parser.add_argument('--dom-dirichlet', type=float, default=1.5,
                        help='Dirichlet alpha for domain weights (default: 1.5)')
    parser.add_argument('--frag-dirichlet', type=float, default=1.5,
                        help='Dirichlet alpha for fragment weights (default: 1.5)')
    parser.add_argument('--ext-alpha', type=float, default=2.0,
                        help='Beta prior alpha for fragment extension (default: 2.0)')
    parser.add_argument('--ext-beta', type=float, default=3.0,
                        help='Beta prior beta for fragment extension (default: 3.0)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path (default: msa-dir/train_dN_fM.npz)')
    parser.add_argument('--checkpoint-every', type=int, default=50,
                        help='Checkpoint every N files (default: 50, 0=off)')
    parser.add_argument('--init-from-maraschino', type=str, default=None,
                        metavar='PATH',
                        help='Initialize parameters from a Maraschino .npz file '
                        '(warm-start SVI-BW from Maraschino-fitted params). '
                        'Overrides --init-ins/--init-del. Enables --estimate-subst.')
    parser.add_argument('--svi-bw-save-maraschino', type=str, default=None,
                        metavar='PATH',
                        help='After each SVI-BW checkpoint, also save parameters '
                        'in Maraschino raw format for re-distillation.')
    parser.add_argument('--fresh', action='store_true',
                        help='Ignore existing checkpoint, start fresh')
    parser.add_argument('--rebuild-manifest', action='store_true',
                        help='Rebuild file manifest from current MSA directory, '
                        'keeping trained parameters. Use when MSA files have '
                        'moved or symlinks changed.')
    parser.add_argument('--budget-hours', type=float, default=None,
                        help='Total CPU budget H (hours). On the first EM iteration, '
                        'families are sampled randomly (without replacement) and timed; '
                        'the subset N(H) is frozen when cumulative E-step time exceeds '
                        'H / n_iter. Subsequent iterations reuse that frozen subset.')
    parser.add_argument('--max-families', type=int, default=None,
                        help='Limit training to N randomly selected families '
                        '(uses --seed for reproducibility)')
    parser.add_argument('--estimate-subst', action='store_true',
                        help='[DEPRECATED — no-op since Phase 5.6 Sub-phase B] '
                        'Per-domain substitution model (dom_S_exch, dom_pis) '
                        'is now ALWAYS estimated under Adam and SVI-BW; the '
                        'flag is retained for backward-compat CLI but has no '
                        'effect. To freeze the substitution model, train '
                        'without --adam (chi-axis-only EM) or use a custom '
                        'init that pins (dom_S_exch, dom_pis).')
    # --estimate-gamma / --gamma-prior-ess are deprecated: class_gamma
    # was removed when class_S_exch became per-class (C, A, A). The rate
    # scale is now absorbed into S_c. Flags accepted but ignored for
    # backward-compatible CLI.
    parser.add_argument('--estimate-gamma', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--gamma-prior-ess', type=float, default=10.0,
                        help=argparse.SUPPRESS)
    parser.add_argument('--classdist-dirichlet', type=float, default=1.0,
                        help='Symmetric Dirichlet concentration alpha on each '
                        'classdist[d,f,:] row; pseudocount = alpha - 1 '
                        '(default: 1.0 = uniform MLE, no pseudocount)')
    parser.add_argument('--classdist-init',
                        choices=['auto', 'identity', 'fragchar'],
                        default='auto',
                        help='Initialisation for classdist[d,f,c]. '
                        'auto (default): uniform, or the C10/C20 mixture '
                        'weights if --class-pi-init selects a profile bank. '
                        'identity: set classdist[d,f,c] = 1 iff c==d '
                        '(requires --n-classes == n_dom). Combined with '
                        '--freeze-classdist this reduces MixDom2 to '
                        'MixDom1 (class c becomes domain c\'s dedicated '
                        'substitution profile). For n_frag>1, also set '
                        '--ext-alpha 1 --ext-beta 1 (zero pseudocounts) '
                        'so the diagonal ext init stays diagonal under '
                        'the M-step (each fragment is a single fragchar). '
                        'fragchar: set classdist[d,f,c] = 1 iff c==f '
                        '(requires --n-classes == n_frag). With '
                        '--freeze-classdist + n_dom=1, this approximates '
                        'a TKF92 variant where each fragment carries a '
                        'fixed fragchar tied to its own substitution class.')
    parser.add_argument('--class-pi-init',
                        choices=['lg_noisy', 'c10', 'c10_topN', 'c20',
                                 'c20_plus_uniform_noisy'],
                        default='lg_noisy',
                        help='Initialisation for per-class equilibrium pi_c '
                        'when --n-classes > 1. lg_noisy (default) = LG pi + '
                        'Dirichlet noise per class. c10 = Le-Gascuel C10 '
                        'profile mixture (requires --n-classes 10). '
                        'c20 = Le-Gascuel C20 (requires --n-classes 20). '
                        'classdist also seeded with the C10/C20 mixture '
                        'weights when using a C10/C20 profile init.')
    parser.add_argument('--estimate-sexch', action='store_true', default=None,
                        help='Estimate per-class S_exch (shape (C, A, A)) from '
                        'per-class HR counts. Each class gets its own GTR '
                        'exchangeability. Requires --n-classes > 1. '
                        'DEFAULT: auto-on when --n-classes > 1, auto-off otherwise. '
                        'Pass --no-estimate-sexch to disable for MixDom2.')
    parser.add_argument('--no-estimate-sexch', dest='estimate_sexch',
                        action='store_false',
                        help='Disable per-class S_exch estimation (overrides '
                        'the default-on-for-MixDom2 behaviour).')
    # ----- Restricted substitution-mstep modes -----
    # See tkf/substitution-mstep.tex (sec:mstep-rescaling, sec:mstep-tied-pi).
    parser.add_argument('--subst-mode',
                        choices=['standard', 'frozen-pi', 'rescaling-rates',
                                 'rescaling-rates-and-pi',
                                 'tied-pi', 'tied-pi-rescaling',
                                 'alt-tied-pi-rescaling'],
                        default='standard',
                        help='Restricted substitution M-step mode (mutually '
                        'exclusive). standard: full GTR / per-class M-step. '
                        'frozen-pi: keep equilibrium pi fixed; only update '
                        'exchangeabilities S. rescaling-rates: keep S shape '
                        'AND pi fixed up to a per-class scalar sigma_c '
                        '(sigma_c = U_dot / D maximises ell_2). '
                        'rescaling-rates-and-pi: keep S shape fixed but '
                        'jointly update (sigma_c, pi_c) via the closed-form '
                        'reduction in substitution-mstep.tex sec:mstep-'
                        'rescaling-pi (1-D Newton in sigma_c, then pi_c by '
                        'closed form). tied-pi: pi tied across blocks of '
                        '--n-tied classes (requires C %% n_tied == 0); '
                        'per-class S_c free. tied-pi-rescaling: pi tied '
                        'within blocks of --n-tied classes, per-class '
                        'sigma_c free, S_c shape frozen at LG. '
                        'alt-tied-pi-rescaling: alternate tied-pi (even iters) '
                        'and rescaling-rates (odd iters).')
    parser.add_argument('--n-tied', type=int, default=1,
                        help='Block size for --subst-mode=tied-pi or '
                        'alt-tied-pi-rescaling. Must divide --n-classes. '
                        'n_tied=1 reduces to the standard per-class M-step. '
                        'Default: 1.')
    parser.add_argument('--banded-frag-init', action='store_true',
                        help='Custom 3-fragchar banded init for fragdist + '
                        'ext_rates (requires --n-frag 3). FragStart=0, '
                        'FragMid=1, FragEnd=2: fragments start at FragStart, '
                        'optionally pass through FragMid, and always end at '
                        'FragEnd. Pseudocounts on the structural-zero entries '
                        'are zeroed so they remain exactly 0 under the M-step; '
                        'fragdist M-step is skipped (initial fragdist[d,0]=1 '
                        'is preserved). Extension probability set by --p-ext.')
    parser.add_argument('--p-ext', type=float, default=0.6,
                        help='Extension probability for --banded-frag-init '
                        '(default: 0.6). Used for the FragStart -> FragMid '
                        'and FragMid -> FragMid transitions.')
    parser.add_argument('--val-every', type=int, default=0,
                        help='Evaluate on validation split every N iterations (0=off). '
                        'Requires --split train (uses val split automatically).')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping: stop after N val evals without improvement '
                        '(default: 5, requires --val-every)')
    parser.add_argument('--min-iter', type=int, default=10,
                        help='Minimum iterations before early stopping on '
                        'patience can fire (lets the SVI EMA warm up). '
                        '--n-iter is the MAX; this is a small floor '
                        '(default: 10, much less than typical n-iter)')
    parser.add_argument('--save-every-iter', action='store_true',
                        help='Save params checkpoint after every iteration '
                        '(default: only save final and best-val)')
    parser.add_argument('--pi-pseudo', type=float, default=10.0,
                        help='Dirichlet pseudocount weight for pi prior (LG08 base)')
    parser.add_argument('--S-pseudo', type=float, default=5.0,
                        help='Pseudocount weight for exchangeability prior (LG08 base)')
    parser.add_argument('--pi-init-noise-frac', type=float, default=0.2,
                        help='For --class-pi-init=lg_noisy, the fraction of '
                        'Dirichlet(50) noise mixed into each class_pis init '
                        '(remainder is π_LG). Default 0.2 → class_pis start '
                        'at 80%% π_LG + 20%% noise. Use 0.01 for a tight LG '
                        'init that keeps symmetry-breaking but starts near '
                        'the pure-LG point.')
    parser.add_argument('--class-pis-from-dom-pis', action='store_true',
                        help='After class_pis/dom_pis init, overwrite '
                        'class_pis[c] = dom_pis[c] for c=0..n_dom-1. '
                        'Requires --classdist-init=identity and '
                        'n_classes==n_dom. Makes V1\' iter-1 emissions '
                        'bit-identical to MixDom1 (d3f1) emissions for a '
                        'controlled superset diagnostic — only delta is '
                        'classdist off-diagonal pseudocounts.')
    parser.add_argument('--classdist-noise-frac', type=float, default=0.0,
                        help='Symmetry-breaking noise on classdist init. '
                        'When > 0, blends per-(d, f) Dirichlet(1) draws into '
                        'the uniform 1/n_classes init: '
                        'classdist[d, f] = (1-frac) * uniform + frac * dirichlet. '
                        'Default 0 (uniform) is a saddle point under the M-step '
                        'gradient; values like 0.1-0.5 break the saddle and let '
                        'classdist specialise per (d, f).')
    parser.add_argument('--freeze-classdist', action='store_true',
                        help='Freeze classdist at its initial (usually uniform) '
                        'values. Tests whether free classdist is the source of '
                        'optimization-landscape flabbiness. Per-class π_c, '
                        'S_exch, indel rates, etc. continue to train normally.')
    parser.add_argument('--freeze-ext-offdiag', action='store_true',
                        help='Zero out off-diagonal Dirichlet pseudocounts on '
                        'ext_rates rows so the prior cannot create off-diag '
                        'mass. Combined with the default diagonal '
                        'init_mixdom_params init (off-diag exactly 0), '
                        'off-diag entries stay at 0 through every M-step '
                        '(zero-init params have zero gradient under EM). '
                        'Useful as a phase-1 warm-start for TKF-FragChar-'
                        'style runs: converge (diag, term) two-way params '
                        'first, then drop this flag for phase 2 ext relax.')
    parser.add_argument('--lg-anchor-strength', type=float, default=0.0,
                        help='Proximal LG-anchor strength s ∈ [0, 1]. After '
                        'each mixture M-step, pulls class_pis and '
                        'class_S_exch toward LG π and LG S by fraction s '
                        '(applied to every class of S). Equivalent to '
                        'Gaussian penalty weight s/(1−s) on drift from LG. '
                        'Default 0 (no anchor). Useful range 0.01–0.1 to '
                        'dampen drift while still allowing data signal.')
    parser.add_argument('--interleaved', action='store_true', default=True,
                        help='Sample pairs round-robin across families (default: True)')
    parser.add_argument('--breadth-sample', action='store_true', default=True,
                        help='SVI-BW: prefer sampling from under-used '
                        'families at each iter (one pair per distinct '
                        'family, least-used first). Fixes oversampling '
                        'of large-family pairs under the uniform-over-pairs '
                        'scheme. DEFAULT: ON (was opt-in before; flip to '
                        'default-on per `feedback_breadth_sampling.md`). '
                        'Pass --no-breadth-sample to revert to uniform-over-pairs.')
    parser.add_argument('--no-breadth-sample', dest='breadth_sample',
                        action='store_false',
                        help='Revert to uniform-over-pairs sampling (overrides '
                        'the default-on breadth-sampling behaviour).')
    parser.add_argument('--no-interleaved', dest='interleaved', action='store_false',
                        help='Process families sequentially (old behavior)')
    parser.add_argument('--split', type=str, default=None, metavar='SPLIT',
                        help='Data split: train/val/test. Uses --split-file if provided, '
                        'otherwise generates an ad-hoc clan-aware split.')
    parser.add_argument('--split-file', type=str, default=None,
                        help='Pre-made split JSON (from bio-datasets make_split.py). '
                        'DEFAULT: auto-detect from candidate paths '
                        '(~/bio-datasets/data/pfam/seed/splits/v1.json, '
                        '~/bio-datasets/data/pfam-seed/splits/v1.json) — '
                        'first existing path wins. Pass an explicit path to override.')
    parser.add_argument('--clan-file', type=str, default=None,
                        help='Pfam-A.clans.tsv.gz for ad-hoc splitting (if no --split-file)')
    parser.add_argument('--split-seed', type=int, default=0,
                        help='Random seed for ad-hoc split (default: 0)')
    parser.add_argument('--split-ratios', type=str, default='0.8,0.1,0.1',
                        help='Train/val/test ratios for ad-hoc split')
    parser.add_argument('--adam', action='store_true',
                        help='Use Adam gradient ascent instead of Baum-Welch EM. '
                        'Completely separate code path using custom VJPs.')
    parser.add_argument('--adam-lr', type=float, default=1e-3,
                        help='Adam learning rate (default: 1e-3)')
    parser.add_argument('--adam-batch', type=int, default=32,
                        help='Adam minibatch size (default: 32)')
    parser.add_argument('--adam-steps', type=int, default=5000,
                        help='Maximum Adam steps (default: 5000)')
    parser.add_argument('--dp-mode', type=str, default='constrained',
                        choices=['constrained', 'full'],
                        help='Adam DP mode. "constrained" (default) uses '
                        'the alignment-aware 1D forward-backward via '
                        'mixdom_constrained_log_prob — JIT-cache size O(log L_max), '
                        'no 2D blowup. "full" runs the 2D pair-HMM '
                        'forward_backward over (Lx, Ly) — kept as a '
                        'reference / fallback path; expect compile-storms.')
    parser.add_argument('--fb-scan-mode', type=str, default='auto',
                        choices=['associative', 'sequential', 'auto'],
                        help='1D forward-backward scan mode (Phase 6). '
                        '"associative" (jax.lax.associative_scan) is O(log L) '
                        'depth but O(L·n²) memory per pair — fast on small '
                        'state spaces. "sequential" (jax.lax.scan) is O(L) '
                        'depth but O(L·n + n²) memory — preferred when n_states '
                        'is large (e.g. d3f3 with ns=47, B=200, L=1024 spent '
                        '~1.8 GiB on the associative prefix-product tensor). '
                        '"auto" (default) picks sequential when '
                        'n_states · padded_L_max ≥ 16384 else associative.')
    parser.add_argument('--svi-bw', action='store_true',
                        help='SVI Baum-Welch: proper Stochastic Variational Inference '
                        'with EMA of sufficient statistics (Hoffman et al. 2013). '
                        'Uses fixed batch size instead of budget-based pair selection.')
    parser.add_argument('--svi-tau', type=float, default=1.0,
                        help='SVI delay parameter tau (Robbins-Monro schedule, default: 1)')
    parser.add_argument('--svi-kappa', type=float, default=0.7,
                        help='SVI forgetting rate kappa (0.5=slow, 1.0=fast, default: 0.7)')
    parser.add_argument('--svi-batch', type=int, default=50,
                        help='SVI minibatch size: pairs per iteration (default: 50)')
    parser.add_argument('--unaligned', action='store_true',
                        help='Use 2D pair HMM FB (unconstrained by MSA alignment). '
                        'Marginalizes over all alignments instead of using the MSA. '
                        'Much slower but avoids alignment bias.')
    parser.add_argument('--match-aligned', action='store_true',
                        help='Match-aligned training: uses 2D DP but constrains match '
                        'states to fire only at positions aligned in the training data, '
                        'while summing over all indel orderings. Mutually exclusive '
                        'with --unaligned.')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force JAX to use CPU only (avoids GPU contention)')
    parser.add_argument('--jax-cache-dir', type=str, default=None,
                        help='Directory for persistent JAX compilation cache '
                        '(speeds up restarts by reusing compiled kernels)')
    parser.add_argument('--precompiled-pairs', type=str, default=None, metavar='DIR',
                        help='Use precompiled JSONL.zst shards instead of Stockholm files. '
                        'DIR should contain manifest.json and shard_*.jsonl.zst files '
                        'produced by precompile_pairs.py. Works with all modes '
                        '(EM, stochastic EM, Adam). '
                        'DEFAULT: pfam/precompiled/ if it exists, else None '
                        '(scan Stockholm dir).')
    parser.add_argument('--save-counts', type=str, default=None, metavar='PATH',
                        help='Save maraschino-compatible counts tensor after '
                        'each E-step (for direct BW vs CherryML comparison)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Evaluate a trained checkpoint on held-out data '
                        '(E-step only, no M-step, no checkpoint saving). '
                        'Requires --checkpoint.')

    args = parser.parse_args()

    # ---- CLI default auto-resolution ----
    # Per the CLI tidy-up (this conversation): make commonly-used flags
    # default-on, with explicit opt-out flags + idempotent retention of the
    # legacy explicit flag for backward-compat.
    #
    # 1. --estimate-sexch: default = True if --n-classes > 1 else False.
    if args.estimate_sexch is None:
        args.estimate_sexch = (getattr(args, 'n_classes', 0) > 1)
        if args.estimate_sexch:
            print(f"  --estimate-sexch auto-on (n_classes={args.n_classes} > 1). "
                  f"Pass --no-estimate-sexch to disable.")

    # 2. --precompiled-pairs: default to pfam/precompiled/ if it exists.
    if args.precompiled_pairs is None:
        _candidate = 'pfam/precompiled/'
        if os.path.isdir(_candidate):
            args.precompiled_pairs = _candidate
            print(f"  --precompiled-pairs auto-detected: {_candidate}")

    # 3. --split-file: auto-detect from candidate paths.
    if args.split_file is None:
        for _candidate in [
            os.path.expanduser('~/bio-datasets/data/pfam/seed/splits/v1.json'),
            os.path.expanduser('~/bio-datasets/data/pfam-seed/splits/v1.json'),
        ]:
            if os.path.exists(_candidate):
                args.split_file = _candidate
                print(f"  --split-file auto-detected: {_candidate}")
                break

    # 4. --breadth-sample: default-on already; log resolved value at startup
    #    so silent default-changes don't surprise.
    print(f"  --breadth-sample={args.breadth_sample} "
          f"(default-on; pass --no-breadth-sample to revert)")

    user_passed_estimate_subst = getattr(args, 'estimate_subst', False)
    args.estimate_subst = user_passed_estimate_subst

    if user_passed_estimate_subst:
        print("WARNING: --estimate-subst is deprecated (no-op since Phase "
              "5.6 Sub-phase B). Per-domain substitution model is now "
              "always estimated under Adam and SVI-BW. The flag is "
              "accepted for backward-compat but has no effect.")

    if getattr(args, 'estimate_gamma', False):
        print("WARNING: --estimate-gamma is deprecated and ignored. "
              "class_gamma was removed when class_S_exch became per-class "
              "(C, A, A); rate scale is now absorbed in S_c.")

    if getattr(args, 'unaligned', False) and getattr(args, 'match_aligned', False):
        parser.error('--unaligned and --match-aligned are mutually exclusive')

    # --subst-mode validation (mutually exclusive restricted regimes; see
    # tkf/substitution-mstep.tex sec:mstep-rescaling, sec:mstep-tied-pi).
    from tkfmixdom.jax.train.restricted_mstep import (
        validate_subst_mode as _validate_subst_mode)
    try:
        _validate_subst_mode(
            args.subst_mode, getattr(args, 'n_tied', 1),
            int(getattr(args, 'n_classes', 0)))
    except ValueError as e:
        parser.error(str(e))

    if getattr(args, 'banded_frag_init', False):
        if getattr(args, 'n_frag', None) != 3:
            parser.error('--banded-frag-init requires --n-frag 3')
        if not (0.0 < getattr(args, 'p_ext', 0.6) < 1.0):
            parser.error('--p-ext must be in (0, 1)')

    if args.cpu_only:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    if args.jax_cache_dir:
        import jax
        jax.config.update("jax_compilation_cache_dir", args.jax_cache_dir)
        os.makedirs(args.jax_cache_dir, exist_ok=True)

    if args.inspect:
        inspect(args)
        return

    if not args.msa_dir and not getattr(args, 'precompiled_pairs', None):
        parser.error('--msa-dir is required (unless using --precompiled-pairs)')

    if getattr(args, 'eval_only', False):
        eval_only(args)
    elif args.adam:
        train_adam(args)
    elif getattr(args, 'svi_bw', False):
        train_svi_bw(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
