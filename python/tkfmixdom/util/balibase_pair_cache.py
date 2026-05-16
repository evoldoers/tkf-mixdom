"""Pair-posterior cache for the BAliBase TFPN harness.

A central, machine-local, gitignored directory holds the raw
2D forward-backward pair posteriors per ``(method_name, family)``.
This lets the caller re-run downstream FSA variants (different
``gap_factor``, different anneal seeds, different alignment-pruning
threshold) without recomputing the expensive 2D DP.

Cache layout:

    ${CACHE_DIR}/${method_name}/${family}.npz   raw pair posteriors
    ${CACHE_DIR}/${method_name}/${family}.json  metadata + params_key

Each NPZ stores
    names     (S,)        sequence names in family order
    pairs     (P, 2)      list of (i, j) pair indices into ``names``
    kind      str         'soft' (probabilities) or 'hard' (0/1)
    post_<k>  (Li, Lj)    posterior for pairs[k]
    failed_pairs_json   list of dicts (failed-pair records)

Each JSON stores
    method_name, family, params_key, timestamp, runner_version,
    n_pairs, n_failed_pairs.

``params_key`` should be a stable string identifying the parameter
set (sha256 of params file contents is recommended). On load the
cache is rejected if ``params_key`` does not match the caller's.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


DEFAULT_CACHE_DIR = Path(os.environ.get(
    'TKFMIXDOM_BALIBASE_PAIR_CACHE',
    str(Path.home() / '.cache' / 'tkf-mixdom-balibase')))


def file_params_key(*paths: os.PathLike,
                     extra: str | None = None) -> str:
    """sha256 over file contents (sorted by path) + optional extra
    payload. Returns a short hex prefix."""
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        try:
            with open(p, 'rb') as f:
                while True:
                    chunk = f.read(1 << 16)
                    if not chunk:
                        break
                    h.update(chunk)
            h.update(b'\n')
        except FileNotFoundError:
            h.update(f'MISSING:{p}\n'.encode())
    if extra is not None:
        h.update(extra.encode())
    return h.hexdigest()[:16]


def cache_paths(method_name: str, family: str,
                  base: Path | None = None) -> tuple[Path, Path]:
    base = Path(base) if base is not None else DEFAULT_CACHE_DIR
    base = base / method_name
    base.mkdir(parents=True, exist_ok=True)
    return base / f'{family}.npz', base / f'{family}.json'


def save(method_name: str, family: str, names: list[str],
          pair_posteriors: dict[tuple[int, int], np.ndarray],
          kind: str, params_key: str,
          failed_pairs: list[dict] | None = None,
          base: Path | None = None,
          dtype: np.dtype = np.float32) -> None:
    """Write the per-family posteriors + metadata. Atomic via
    write-to-tmp-and-rename."""
    npz_path, json_path = cache_paths(method_name, family, base=base)
    pairs = sorted(pair_posteriors.keys())
    arrays: dict[str, np.ndarray] = {}
    arrays['names'] = np.asarray(names, dtype=object)
    arrays['pairs'] = np.asarray(pairs, dtype=np.int32)
    for k, (i, j) in enumerate(pairs):
        arrays[f'post_{k}'] = np.asarray(pair_posteriors[(i, j)], dtype=dtype)
    # numpy.savez_compressed appends ``.npz`` if the path doesn't end
    # in it, which clobbers a naive ``.tmp`` atomic write. Use a sibling
    # name without the .npz extension and let numpy add it, then rename.
    tmp_stem = npz_path.parent / (npz_path.stem + '.tmp')
    np.savez_compressed(str(tmp_stem), **arrays)
    written = tmp_stem.parent / (tmp_stem.name + '.npz')
    os.replace(written, npz_path)
    meta = {
        'method_name': method_name,
        'family': family,
        'params_key': params_key,
        'kind': kind,
        'timestamp': int(time.time()),
        'n_pairs': int(len(pairs)),
        'failed_pairs': failed_pairs or [],
    }
    tmp_json = json_path.with_suffix('.json.tmp')
    tmp_json.write_text(json.dumps(meta, indent=2))
    os.replace(tmp_json, json_path)


def load(method_name: str, family: str, params_key: str,
          base: Path | None = None
          ) -> tuple[dict[tuple[int, int], np.ndarray], str, list[dict]] | None:
    """Return ``(pair_posteriors, kind, failed_pairs)`` if the cache
    hits with a matching ``params_key``; otherwise ``None``."""
    npz_path, json_path = cache_paths(method_name, family, base=base)
    if not (npz_path.exists() and json_path.exists()):
        return None
    try:
        meta = json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if meta.get('params_key') != params_key:
        return None
    d = np.load(npz_path, allow_pickle=True)
    pairs = np.asarray(d['pairs'])
    pair_posteriors: dict[tuple[int, int], np.ndarray] = {}
    for k, (i, j) in enumerate(pairs.tolist()):
        pair_posteriors[(int(i), int(j))] = np.asarray(d[f'post_{k}'])
    return pair_posteriors, meta.get('kind', 'soft'), meta.get('failed_pairs', [])


def invalidate(method_name: str, family: str | None = None,
                base: Path | None = None) -> int:
    """Delete all cache entries for one method (or one family).
    Returns the number of files removed."""
    base = Path(base) if base is not None else DEFAULT_CACHE_DIR
    target = base / method_name
    if not target.exists():
        return 0
    n = 0
    for p in target.iterdir():
        if family and not p.name.startswith(f'{family}.'):
            continue
        p.unlink()
        n += 1
    return n
