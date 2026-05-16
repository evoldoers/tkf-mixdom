"""I/O utilities for simulation-evaluation pipeline intermediate files.

Each pipeline stage produces self-contained output files (NPZ for arrays,
JSON for metadata and scores). Each stage reads ONLY from the previous
stage's files, ensuring no data leakage by construction.

Pipeline stages:
  01_tree.json      — Tree specification + metadata
  02_sim.npz        — Simulated MSA with full annotations (ground truth)
  03_observed.npz   — Partially observed data (leaf sequences, unaligned)
  04_recon_{method}.npz — Reconstruction output per method
  05_scores_{method}.json — Evaluation scores per method
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def save_tree(path: str, newick: str, metadata: Optional[Dict] = None):
    """Save tree specification as JSON."""
    data = {'newick': newick}
    if metadata:
        data['metadata'] = metadata
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_tree(path: str) -> Dict:
    """Load tree specification from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def save_simulation(path: str,
                    msa: Dict[str, np.ndarray],
                    annotations: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                    metadata: Optional[Dict] = None):
    """Save simulated MSA with annotations as NPZ.

    Args:
        msa: {node_name: (L,) int residue array, -1=gap}
        annotations: optional {node_name: {
            'domain_ids': (L,) int,
            'domain_types': (L,) int,
            'frag_ids': (L,) int,
            'frag_types': (L,) int,
            'class_ids': (L,) int,
        }}
        metadata: optional dict (stored as JSON string in NPZ).
    """
    save_dict = {}
    for name, seq in msa.items():
        save_dict[f'msa_{name}'] = np.asarray(seq, dtype=np.int32)
    save_dict['node_names'] = np.array(list(msa.keys()), dtype=object)

    if annotations:
        for name, ann in annotations.items():
            for key, arr in ann.items():
                save_dict[f'ann_{name}_{key}'] = np.asarray(arr, dtype=np.int32)

    if metadata:
        save_dict['metadata_json'] = np.array(json.dumps(metadata))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **save_dict)


def load_simulation(path: str) -> Dict:
    """Load simulated MSA with annotations from NPZ.

    Returns dict with 'msa', 'annotations', 'metadata'.
    """
    data = np.load(path, allow_pickle=True)
    names = list(data['node_names'])

    msa = {}
    for name in names:
        msa[name] = data[f'msa_{name}']

    annotations = {}
    ann_keys = ['domain_ids', 'domain_types', 'frag_ids',
                'frag_types', 'class_ids']
    for name in names:
        ann = {}
        for key in ann_keys:
            k = f'ann_{name}_{key}'
            if k in data:
                ann[key] = data[k]
        if ann:
            annotations[name] = ann

    metadata = None
    if 'metadata_json' in data:
        metadata = json.loads(str(data['metadata_json']))

    return {'msa': msa, 'annotations': annotations, 'metadata': metadata}


def save_observation(path: str,
                     leaf_seqs_aligned: Dict[str, np.ndarray],
                     leaf_seqs_unaligned: Optional[Dict[str, np.ndarray]] = None):
    """Save partially observed data (leaf sequences only)."""
    save_dict = {}
    names = sorted(leaf_seqs_aligned.keys())
    save_dict['leaf_names'] = np.array(names, dtype=object)
    for name in names:
        save_dict[f'aligned_{name}'] = np.asarray(
            leaf_seqs_aligned[name], dtype=np.int32)
    if leaf_seqs_unaligned:
        for name in names:
            save_dict[f'unaligned_{name}'] = np.asarray(
                leaf_seqs_unaligned[name], dtype=np.int32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **save_dict)


def load_observation(path: str) -> Dict:
    """Load observed leaf sequences from NPZ."""
    data = np.load(path, allow_pickle=True)
    names = list(data['leaf_names'])
    aligned = {n: data[f'aligned_{n}'] for n in names}
    unaligned = {}
    for n in names:
        k = f'unaligned_{n}'
        if k in data:
            unaligned[n] = data[k]
    return {'aligned': aligned, 'unaligned': unaligned or None}


def save_reconstruction(path: str, method: str,
                        results: Dict[str, Any]):
    """Save reconstruction output for one method."""
    save_dict = {'method': np.array(method)}
    for key, val in results.items():
        if isinstance(val, np.ndarray):
            save_dict[key] = val
        elif isinstance(val, (int, float)):
            save_dict[key] = np.array(val)
        elif isinstance(val, str):
            save_dict[key] = np.array(val)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **save_dict)


def load_reconstruction(path: str) -> Dict:
    """Load reconstruction output from NPZ."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def save_scores(path: str, method: str, scores: Dict[str, Any]):
    """Save evaluation scores as JSON."""
    out = {'method': method, 'scores': scores}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)


def load_scores(path: str) -> Dict:
    """Load evaluation scores from JSON."""
    with open(path, 'r') as f:
        return json.load(f)
