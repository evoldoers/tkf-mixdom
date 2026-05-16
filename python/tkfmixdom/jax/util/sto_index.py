"""Fast random-access to Stockholm alignment sequences via byte-offset index.

Scans a Stockholm file once to record byte offsets of sequence lines.
Subsequent access fetches individual sequences by seeking, avoiding
full file re-parse. Supports both plain and gzip files.

Usage:
    idx = StoIndex(filepath)           # scans file, caches offsets
    seq = idx.get_sequence(row_idx)    # random access by row number
    pair = idx.get_pair(i, j)          # get two sequences
"""

import gzip
import os
from typing import List, Tuple, Optional


class StoIndex:
    """Byte-offset index for fast random access to Stockholm sequences.

    After construction, get_sequence(i) seeks to the i-th sequence line
    and reads it in ~microseconds (no full file parse).

    For gzip files, we can't seek efficiently, so we cache all sequences
    in memory on first parse (still only one parse).
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.is_gz = filepath.endswith('.gz')
        self.names: List[str] = []

        if self.is_gz:
            # Gzip: can't seek, so cache everything
            self._seqs: List[str] = []
            self._offsets = None
            with gzip.open(filepath, 'rt') as f:
                for line in f:
                    line = line.rstrip('\n')
                    if not line or line.startswith('#') or line.startswith('//'):
                        continue
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        self.names.append(parts[0])
                        self._seqs.append(parts[1])
        else:
            # Plain text: record byte offsets for seek-based access
            self._seqs = None
            self._offsets: List[Tuple[int, int, int]] = []  # (offset, name_end, line_end)
            with open(filepath, 'rb') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line[0:1] in (b'#', b'/', b'\n', b'\r'):
                        continue
                    # Sequence line: "NAME/range  SEQUENCE"
                    stripped = line.rstrip(b'\n\r')
                    parts = stripped.split(None, 1)
                    if len(parts) == 2:
                        name = parts[0].decode('ascii', errors='replace')
                        self.names.append(name)
                        # Record: byte offset, position of sequence start, line length
                        seq_start = offset + stripped.index(parts[1])
                        seq_len = len(parts[1])
                        self._offsets.append((seq_start, seq_len))

    def __len__(self):
        return len(self.names)

    def get_sequence(self, row_idx: int) -> str:
        """Get aligned sequence string by row index (0-based)."""
        if self._seqs is not None:
            return self._seqs[row_idx]
        offset, length = self._offsets[row_idx]
        with open(self.filepath, 'rb') as f:
            f.seek(offset)
            return f.read(length).decode('ascii', errors='replace')

    def get_pair(self, i: int, j: int) -> Tuple[str, str]:
        """Get two aligned sequences by row indices."""
        return self.get_sequence(i), self.get_sequence(j)
