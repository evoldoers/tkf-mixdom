"""Streaming data loader for pre-compiled MixDom training pairs.

Reads zstd-compressed JSONL shards produced by precompile_pairs.py,
decodes X/A/Y records, and yields batches grouped by geometric bin size
for JIT cache reuse.

Usage:
    loader = PairDataLoader('pfam/precompiled/', batch_size=32, seed=42)
    for batch in loader:
        # batch is a dict with keys:
        #   'states_list': list of state lists
        #   'anc_chars_list': list of anc_char lists
        #   'desc_chars_list': list of desc_char lists
        #   't_est': list of floats
        #   'families': list of family strings
        #   'bin_size': int (geometric bin size for this batch)
        ...

Streaming shard iteration:
    stream = PairStream('pfam/precompiled/', seed=42)
    batch = stream.next_batch(32)   # decode up to 32 pairs
    stream.reset()                  # restart from first shard
    stream.shuffle(seed=99)         # re-shuffle shard order
"""

import json
import os
from collections import defaultdict

import numpy as np

from .padding import pad_to_bin
from .pair_format import decode_pair


class PairStream:
    """Streaming iterator over precompiled shards.

    Decompresses one shard at a time (~3000 pairs, ~800 KB).
    Only one shard in memory at once. Suitable for SGD training
    where approximate uniform sampling is sufficient.
    """

    def __init__(self, precompiled_dir, seed=42, max_alignment_len=None):
        """Load manifest and prepare shard order.

        Args:
            precompiled_dir: path to directory with manifest.json and shards
            seed: random seed for shard order shuffling
            max_alignment_len: if set, skip pairs with alignment length > this
        """
        self.precompiled_dir = precompiled_dir
        self.max_alignment_len = max_alignment_len

        manifest_path = os.path.join(precompiled_dir, 'manifest.json')
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.shard_files = list(self.manifest['shard_files'])
        self.n_pairs = self.manifest['n_pairs']
        self.n_families = self.manifest['n_families']

        # Shuffle shard order
        self._rng = np.random.RandomState(seed)
        self._shard_order = np.arange(len(self.shard_files))
        self._rng.shuffle(self._shard_order)

        # Iteration state
        self._shard_idx = 0       # index into _shard_order
        self._buffer = []         # decoded pairs from current shard
        self._buffer_pos = 0      # position within buffer
        self._exhausted = False   # all shards consumed

    def _read_shard(self, shard_name):
        """Read and decompress a single shard, returning list of record dicts."""
        import zstandard as zstd

        shard_path = os.path.join(self.precompiled_dir, shard_name)
        with open(shard_path, 'rb') as f:
            compressed = f.read()

        dctx = zstd.ZstdDecompressor()
        data = dctx.decompress(compressed)
        lines = data.decode('utf-8').split('\n')

        records = []
        for line in lines:
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def _decode_shard(self, shard_name):
        """Decode all records in a shard to tuples.

        Returns list of (x_int, y_int, states, anc_chars, desc_chars, t_est).
        """
        records = self._read_shard(shard_name)
        decoded = []
        for rec in records:
            x_int, y_int, states, anc_chars, desc_chars, t_est, *_ = decode_pair(rec)
            if self.max_alignment_len and len(states) > self.max_alignment_len:
                continue
            decoded.append((x_int, y_int, states, anc_chars, desc_chars, t_est))
        return decoded

    def _refill_buffer(self):
        """Load the next shard into the buffer. Returns True if successful."""
        while self._shard_idx < len(self._shard_order):
            si = self._shard_order[self._shard_idx]
            self._shard_idx += 1
            shard_name = self.shard_files[si]
            decoded = self._decode_shard(shard_name)
            if decoded:
                # Shuffle within shard for diversity
                self._rng.shuffle(decoded)
                self._buffer = decoded
                self._buffer_pos = 0
                return True
        self._exhausted = True
        return False

    def next_batch(self, n):
        """Return up to n decoded pairs. Refills from next shard when empty.

        Returns list of (x_int, y_int, states, anc_chars, desc_chars, t_est).
        Returns empty list when all shards are exhausted.
        """
        result = []
        while len(result) < n:
            # Try to pull from current buffer
            if self._buffer_pos < len(self._buffer):
                remaining_in_buffer = len(self._buffer) - self._buffer_pos
                take = min(n - len(result), remaining_in_buffer)
                result.extend(
                    self._buffer[self._buffer_pos:self._buffer_pos + take])
                self._buffer_pos += take
            else:
                # Buffer exhausted, load next shard
                if self._exhausted or not self._refill_buffer():
                    break
        return result

    def sample_batch(self, n, seed=None):
        """Sample n pairs from a random shard with constant memory.

        Picks one random shard, decodes it, and samples n pairs from it.
        Does NOT advance the streaming position. Useful for Adam where
        approximate uniform sampling is sufficient.

        Args:
            n: number of pairs to sample
            seed: optional seed for reproducibility; if None, uses internal RNG

        Returns:
            list of (x_int, y_int, states, anc_chars, desc_chars, t_est)
        """
        rng = np.random.RandomState(seed) if seed is not None else self._rng
        si = rng.randint(len(self.shard_files))
        decoded = self._decode_shard(self.shard_files[si])
        if not decoded:
            return []
        if n >= len(decoded):
            return decoded
        indices = rng.choice(len(decoded), size=n, replace=False)
        return [decoded[i] for i in indices]

    def reset(self):
        """Start over from first shard (same order). For EM repeated iters."""
        self._shard_idx = 0
        self._buffer = []
        self._buffer_pos = 0
        self._exhausted = False

    def shuffle(self, seed):
        """Re-shuffle shard order. For diversity across iterations."""
        rng = np.random.RandomState(seed)
        rng.shuffle(self._shard_order)
        self.reset()

    def __len__(self):
        """Total pair count from manifest."""
        return self.n_pairs

    @property
    def n_shards(self):
        """Number of shards."""
        return len(self.shard_files)


class PairDataLoader:
    """Streaming loader for pre-compiled pairs.

    Yields batches of decoded pairs grouped by geometric bin size
    for JIT cache reuse.

    Features:
    - Streaming decompression (zstd)
    - Deterministic shuffling from seed
    - Group by sequence length bin
    - Epoch-based iteration
    """

    def __init__(self, precompiled_dir, batch_size=32, seed=42,
                 max_alignment_len=None):
        """Initialize loader.

        Args:
            precompiled_dir: path to directory with manifest.json and shards
            batch_size: number of pairs per batch
            seed: random seed for shuffling
            max_alignment_len: if set, skip pairs longer than this
        """
        self.precompiled_dir = precompiled_dir
        self.batch_size = batch_size
        self.seed = seed
        self.max_alignment_len = max_alignment_len

        manifest_path = os.path.join(precompiled_dir, 'manifest.json')
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.shard_files = self.manifest['shard_files']
        self.n_pairs = self.manifest['n_pairs']
        self.n_families = self.manifest['n_families']

    def _read_shard(self, shard_name):
        """Read and decompress a single shard, returning list of record dicts."""
        import zstandard as zstd

        shard_path = os.path.join(self.precompiled_dir, shard_name)
        with open(shard_path, 'rb') as f:
            compressed = f.read()

        dctx = zstd.ZstdDecompressor()
        data = dctx.decompress(compressed)
        lines = data.decode('utf-8').split('\n')

        records = []
        for line in lines:
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def _load_all_records(self):
        """Load all records from all shards."""
        all_records = []
        for shard_name in self.shard_files:
            records = self._read_shard(shard_name)
            all_records.extend(records)
        return all_records

    def __iter__(self):
        """Yield batches of pairs, grouped by length bin.

        Each batch is a dict with:
            'states_list': list of state lists (each is list of int)
            'anc_chars_list': list of anc_char lists
            'desc_chars_list': list of desc_char lists
            't_est': list of floats
            'families': list of family strings
            'pair_ids': list of pair ID strings
            'bin_size': int (geometric bin for this batch)
            'lengths': list of actual alignment lengths
        """
        all_records = self._load_all_records()

        # Shuffle records
        rng = np.random.RandomState(self.seed)
        rng.shuffle(all_records)

        # Decode and group by bin size
        bins = defaultdict(list)
        for rec in all_records:
            x_int, y_int, states, anc_chars, desc_chars, t_est, *_ = decode_pair(rec)
            aln_len = len(states)

            if self.max_alignment_len and aln_len > self.max_alignment_len:
                continue

            bin_size = pad_to_bin(aln_len)
            bins[bin_size].append({
                'states': states,
                'anc_chars': anc_chars,
                'desc_chars': desc_chars,
                't_est': t_est,
                'family': rec['fam'],
                'pair_id': rec['id'],
                'length': aln_len,
            })

        # Shuffle bin order, then yield batches within each bin
        bin_keys = list(bins.keys())
        rng.shuffle(bin_keys)

        for bin_size in bin_keys:
            items = bins[bin_size]
            rng.shuffle(items)

            for start in range(0, len(items), self.batch_size):
                end = min(start + self.batch_size, len(items))
                batch_items = items[start:end]

                yield {
                    'states_list': [it['states'] for it in batch_items],
                    'anc_chars_list': [it['anc_chars'] for it in batch_items],
                    'desc_chars_list': [it['desc_chars'] for it in batch_items],
                    't_est': [it['t_est'] for it in batch_items],
                    'families': [it['family'] for it in batch_items],
                    'pair_ids': [it['pair_id'] for it in batch_items],
                    'bin_size': bin_size,
                    'lengths': [it['length'] for it in batch_items],
                }

    def __len__(self):
        """Approximate number of batches per epoch."""
        return max(1, self.n_pairs // self.batch_size)

    def stats(self):
        """Return summary statistics about the dataset."""
        return {
            'n_pairs': self.n_pairs,
            'n_families': self.n_families,
            'n_shards': len(self.shard_files),
            'batch_size': self.batch_size,
            'approx_batches': len(self),
            'manifest_stats': self.manifest.get('stats', {}),
        }
