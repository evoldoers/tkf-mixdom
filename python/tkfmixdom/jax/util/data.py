"""Data download and preparation utilities for experiments.

Functions for fetching protein sequences and alignments from public databases.
"""

import os
import urllib.request
import json

import numpy as np

from .io import read_fasta, read_stockholm, alignment_to_pairs, seq_to_int


PFAM_BASE = "https://www.ebi.ac.uk/interpro/api/entry/pfam"
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
RFAM_BASE = "https://rfam.org"


def fetch_url(url, cache_dir=None):
    """Fetch a URL, optionally caching the result.

    Args:
        url: URL to fetch
        cache_dir: directory to cache downloads (None = no caching)

    Returns:
        response text as string
    """
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        # Use URL hash as cache filename
        import hashlib
        cache_file = os.path.join(cache_dir, hashlib.md5(url.encode()).hexdigest())
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                return f.read()

    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
        # Handle gzip-compressed responses
        if raw[:2] == b'\x1f\x8b':
            import gzip
            raw = gzip.decompress(raw)
        text = raw.decode("utf-8")

    if cache_dir is not None:
        with open(cache_file, "w") as f:
            f.write(text)

    return text


def fetch_pfam_stockholm(pfam_id, cache_dir=None, fixtures_dir=None):
    """Fetch a Pfam seed alignment in Stockholm format.

    Args:
        pfam_id: Pfam accession (e.g. "PF00001")
        cache_dir: directory to cache downloads
        fixtures_dir: directory with bundled .sto.gz files (checked first)

    Returns:
        dict from read_stockholm()
    """
    if fixtures_dir is not None:
        fixture = os.path.join(fixtures_dir, f"{pfam_id}.sto.gz")
        if os.path.exists(fixture):
            import gzip
            with gzip.open(fixture, "rt") as f:
                return read_stockholm(f.read(), is_file=False)
    url = f"https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/{pfam_id}/?annotation=alignment:seed&download"
    text = fetch_url(url, cache_dir)
    return read_stockholm(text, is_file=False)


def fetch_uniprot_fasta(accession, cache_dir=None):
    """Fetch a UniProt sequence in FASTA format.

    Args:
        accession: UniProt accession (e.g. "P12345")
        cache_dir: directory to cache downloads

    Returns:
        (name, sequence) tuple
    """
    url = f"{UNIPROT_BASE}/{accession}.fasta"
    text = fetch_url(url, cache_dir)
    records = read_fasta(text, is_file=False)
    if not records:
        raise ValueError(f"No sequence found for {accession}")
    return records[0]


def fetch_rfam_stockholm(rfam_id, cache_dir=None, fixtures_dir=None):
    """Fetch an Rfam seed alignment in Stockholm format.

    Args:
        rfam_id: Rfam accession (e.g. "RF00001")
        cache_dir: directory to cache downloads
        fixtures_dir: directory with bundled .sto.gz files (checked first)

    Returns:
        dict from read_stockholm()
    """
    if fixtures_dir is not None:
        fixture = os.path.join(fixtures_dir, f"{rfam_id}.sto.gz")
        if os.path.exists(fixture):
            import gzip
            with gzip.open(fixture, "rt") as f:
                return read_stockholm(f.read(), is_file=False)
    url = f"{RFAM_BASE}/family/{rfam_id}/alignment?acc={rfam_id}&format=stockholm&download=0"
    text = fetch_url(url, cache_dir)
    return read_stockholm(text, is_file=False)


def fetch_rfam_tree(rfam_id, cache_dir=None):
    """Fetch the Rfam seed tree in Newick format.

    Args:
        rfam_id: Rfam accession (e.g. "RF00001")
        cache_dir: directory to cache downloads

    Returns:
        Newick string
    """
    url = f"{RFAM_BASE}/family/{rfam_id}/tree/label/acc?download=0"
    return fetch_url(url, cache_dir)


def load_fasta_sequences(fasta_path, alphabet="protein"):
    """Load sequences from a FASTA file as integer arrays.

    Args:
        fasta_path: path to FASTA file
        alphabet: "protein" or "dna"

    Returns:
        dict of {name: integer_array}
    """
    records = read_fasta(fasta_path)
    return {name: seq_to_int(seq, alphabet) for name, seq in records}


def load_stockholm_sequences(stockholm_path, alphabet="protein"):
    """Load ungapped sequences from a Stockholm alignment file.

    Args:
        stockholm_path: path to Stockholm file
        alphabet: "protein" or "dna"

    Returns:
        dict of {name: integer_array} (ungapped)
    """
    result = read_stockholm(stockholm_path)
    return alignment_to_pairs(result["sequences"], alphabet)


# ---------------------------------------------------------------------------
# BAliBase / BRaliBase benchmark proxies
# ---------------------------------------------------------------------------

# Small Pfam families used as BAliBase proxy (protein, < 20 seqs, < 200 residues)
BALIBASE_PROXY_FAMILIES = [
    "PF00014",  # Kunitz/BPTI (trypsin inhibitor), short
    "PF00018",  # SH3 domain
    "PF00032",  # Cytochrome b(c1) complex
    "PF00046",  # Homeodomain
    "PF00076",  # RRM (RNA recognition motif)
]

# Small Rfam families used as BRaliBase proxy (RNA, < 10 seqs, < 150 nt)
BRALIBASE_PROXY_FAMILIES = [
    "RF00001",  # 5S ribosomal RNA
    "RF00005",  # tRNA
    "RF00010",  # RNaseP
    "RF00020",  # U5 spliceosomal RNA
    "RF00023",  # tmRNA
]


def fetch_balibase_reference(ref_name, cache_dir=None, max_seqs=20, max_len=200,
                             fixtures_dir=None):
    """Fetch a BAliBase proxy reference alignment from Pfam.

    Uses Pfam seed alignments as a proxy for BAliBase reference alignments.
    Filters to at most max_seqs sequences of at most max_len residues.

    Args:
        ref_name: Pfam accession (e.g. "PF00014") or one of
                  BALIBASE_PROXY_FAMILIES
        cache_dir: directory to cache downloads
        max_seqs: maximum number of sequences to keep
        max_len: maximum ungapped sequence length
        fixtures_dir: directory with bundled .sto.gz files (checked first)

    Returns:
        dict with keys:
            'name': family name
            'aligned': dict of {seq_name: gapped_int_array} (-1 = gap)
            'ungapped': dict of {seq_name: int_array}
            'n_seqs': number of sequences
            'avg_len': average ungapped length
    """
    result = fetch_pfam_stockholm(ref_name, cache_dir, fixtures_dir=fixtures_dir)
    aligned = {}
    ungapped = {}

    for name, aln_seq in result["sequences"].items():
        raw = aln_seq.replace(".", "-").upper()
        ug = raw.replace("-", "")
        if len(ug) == 0 or len(ug) > max_len:
            continue
        int_aln = _gapped_seq_to_int(raw, "protein")
        int_ug = seq_to_int(ug, "protein")
        # Skip sequences with too many unknown residues
        if np.sum(int_ug == -1) > len(int_ug) * 0.3:
            continue
        aligned[name] = int_aln
        ungapped[name] = int_ug
        if len(aligned) >= max_seqs:
            break

    avg_len = float(np.mean([len(s) for s in ungapped.values()])) if ungapped else 0.0
    return {
        "name": ref_name,
        "aligned": aligned,
        "ungapped": ungapped,
        "n_seqs": len(aligned),
        "avg_len": avg_len,
    }


def fetch_bralibase_reference(ref_name, cache_dir=None, max_seqs=10, max_len=150,
                              fixtures_dir=None):
    """Fetch a BRaliBase proxy reference alignment from Rfam.

    Uses Rfam seed alignments as a proxy for BRaliBase reference alignments.
    Filters to at most max_seqs sequences of at most max_len nucleotides.

    Args:
        ref_name: Rfam accession (e.g. "RF00005") or one of
                  BRALIBASE_PROXY_FAMILIES
        cache_dir: directory to cache downloads
        max_seqs: maximum number of sequences to keep
        max_len: maximum ungapped sequence length
        fixtures_dir: directory with bundled .sto.gz files (checked first)

    Returns:
        dict with keys:
            'name': family name
            'aligned': dict of {seq_name: gapped_int_array} (-1 = gap)
            'ungapped': dict of {seq_name: int_array}
            'n_seqs': number of sequences
            'avg_len': average ungapped length
    """
    result = fetch_rfam_stockholm(ref_name, cache_dir, fixtures_dir=fixtures_dir)
    aligned = {}
    ungapped = {}

    for name, aln_seq in result["sequences"].items():
        raw = aln_seq.replace(".", "-").upper()
        ug = raw.replace("-", "")
        if len(ug) == 0 or len(ug) > max_len:
            continue
        int_aln = _gapped_seq_to_int(raw, "dna")
        int_ug = seq_to_int(ug, "dna")
        if np.sum(int_ug == -1) > len(int_ug) * 0.3:
            continue
        aligned[name] = int_aln
        ungapped[name] = int_ug
        if len(aligned) >= max_seqs:
            break

    avg_len = float(np.mean([len(s) for s in ungapped.values()])) if ungapped else 0.0
    return {
        "name": ref_name,
        "aligned": aligned,
        "ungapped": ungapped,
        "n_seqs": len(aligned),
        "avg_len": avg_len,
    }


def _gapped_seq_to_int(gapped_seq, alphabet):
    """Convert a gapped sequence string to integer array with -1 for gaps.

    Args:
        gapped_seq: sequence string (may contain '-' for gaps)
        alphabet: "protein" or "dna"

    Returns:
        numpy int array with -1 for gap positions
    """
    from .io import AA_TO_INT, NT_TO_INT
    mapping = AA_TO_INT if alphabet == "protein" else NT_TO_INT
    result = []
    for c in gapped_seq:
        if c == '-':
            result.append(-1)
        else:
            result.append(mapping.get(c, -1))
    return np.array(result, dtype=np.int32)


# ---------------------------------------------------------------------------
# Alignment scoring (SP and TC scores)
# ---------------------------------------------------------------------------

def _extract_aligned_pairs(aligned_seqs):
    """Extract the set of aligned residue pairs from a gapped alignment.

    For each pair of sequences (i, j) with i < j, finds all pairs of
    residue positions (p_i, p_j) that are aligned (both non-gap in the
    same column).

    Args:
        aligned_seqs: dict of {name: gapped_int_array} where -1 = gap

    Returns:
        dict of {(name_i, name_j): set of (pos_i, pos_j)} where pos_i, pos_j
        are positions in the ungapped sequences (0-based)
    """
    names = sorted(aligned_seqs.keys())
    pairs = {}

    for idx_i in range(len(names)):
        for idx_j in range(idx_i + 1, len(names)):
            ni, nj = names[idx_i], names[idx_j]
            si = aligned_seqs[ni]
            sj = aligned_seqs[nj]
            aligned_set = set()
            pos_i = 0
            pos_j = 0
            for col in range(len(si)):
                ci = si[col]
                cj = sj[col]
                is_gap_i = (ci == -1)
                is_gap_j = (cj == -1)
                if not is_gap_i and not is_gap_j:
                    aligned_set.add((pos_i, pos_j))
                if not is_gap_i:
                    pos_i += 1
                if not is_gap_j:
                    pos_j += 1
            pairs[(ni, nj)] = aligned_set

    return pairs


def compute_sp_score(test_alignment, ref_alignment):
    """Compute Sum-of-Pairs (SP) score.

    For each pair of sequences, counts the fraction of aligned residue
    pairs in the reference that are also aligned in the test alignment.

    Args:
        test_alignment: dict of {name: gapped_int_array} (-1 = gap)
        ref_alignment: dict of {name: gapped_int_array} (-1 = gap)

    Returns:
        float: SP score in [0, 1]
    """
    # Only score sequences present in both alignments
    common_names = sorted(set(test_alignment.keys()) & set(ref_alignment.keys()))
    if len(common_names) < 2:
        return 0.0

    test_sub = {n: test_alignment[n] for n in common_names}
    ref_sub = {n: ref_alignment[n] for n in common_names}

    test_pairs = _extract_aligned_pairs(test_sub)
    ref_pairs = _extract_aligned_pairs(ref_sub)

    total_ref = 0
    total_correct = 0

    for key, ref_set in ref_pairs.items():
        test_set = test_pairs.get(key, set())
        total_ref += len(ref_set)
        total_correct += len(ref_set & test_set)

    if total_ref == 0:
        return 1.0
    return total_correct / total_ref


def compute_tc_score(test_alignment, ref_alignment):
    """Compute Total Column (TC) score.

    Fraction of columns in the reference alignment that are exactly
    reproduced in the test alignment. A column is a set of
    (sequence_name, residue_position) pairs for non-gap entries.

    Args:
        test_alignment: dict of {name: gapped_int_array} (-1 = gap)
        ref_alignment: dict of {name: gapped_int_array} (-1 = gap)

    Returns:
        float: TC score in [0, 1]
    """
    common_names = sorted(set(test_alignment.keys()) & set(ref_alignment.keys()))
    if len(common_names) < 2:
        return 0.0

    def _columns_to_sets(aligned, names):
        """Convert alignment columns to sets of (name, ungapped_pos)."""
        n_cols = len(next(iter(aligned.values())))
        # Track ungapped position per sequence
        positions = {n: 0 for n in names}
        columns = []
        for col in range(n_cols):
            col_set = set()
            for n in names:
                c = aligned[n][col]
                if c != -1:
                    col_set.add((n, positions[n]))
                    positions[n] += 1
                # else gap: don't increment
            if len(col_set) >= 2:
                columns.append(frozenset(col_set))
        return columns

    ref_sub = {n: ref_alignment[n] for n in common_names}
    test_sub = {n: test_alignment[n] for n in common_names}

    ref_columns = _columns_to_sets(ref_sub, common_names)
    test_column_set = set(_columns_to_sets(test_sub, common_names))

    if len(ref_columns) == 0:
        return 1.0

    correct = sum(1 for c in ref_columns if c in test_column_set)
    return correct / len(ref_columns)


def prepare_sequence_pairs(sequences, max_pairs=None):
    """Prepare all pairs from a sequence dict.

    Args:
        sequences: dict of {name: integer_array}
        max_pairs: maximum number of pairs to return (None = all)

    Returns:
        list of (name_i, name_j, seq_i, seq_j) tuples
    """
    names = sorted(sequences.keys())
    pairs = []
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            x = sequences[n1]
            y = sequences[n2]
            if len(x) > 0 and len(y) > 0:
                pairs.append((n1, n2, x, y))
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs
    return pairs
