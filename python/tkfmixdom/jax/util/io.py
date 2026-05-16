"""Data I/O: FASTA, Stockholm, Newick parsers.

Pure Python implementations with no external dependencies beyond numpy.
Sequences are converted to integer arrays using alphabet mappings.
"""

import re
import numpy as np


# --- Alphabets ---

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
NUCLEOTIDES = "ACGT"

WILDCARD_IDX = 20  # index for unknown/ambiguous amino acids

AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
# Map ambiguous/non-standard amino acids to wildcard index
for _c in "XBZJUO":
    AA_TO_INT[_c] = WILDCARD_IDX
INT_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}
INT_TO_AA[WILDCARD_IDX] = 'X'

NT_TO_INT = {nt: i for i, nt in enumerate(NUCLEOTIDES)}
NT_TO_INT['U'] = NT_TO_INT['T']  # RNA U -> same index as T
INT_TO_NT = {i: nt for i, nt in enumerate(NUCLEOTIDES)}


def seq_to_int(seq, alphabet="protein"):
    """Convert a sequence string to integer array.

    Args:
        seq: sequence string (uppercase)
        alphabet: "protein" or "dna"

    Returns:
        numpy int array, unknown chars mapped to -1
    """
    mapping = AA_TO_INT if alphabet == "protein" else NT_TO_INT
    default = WILDCARD_IDX if alphabet == "protein" else -1
    return np.array([mapping.get(c, default) for c in seq.upper()], dtype=np.int32)


def int_to_seq(arr, alphabet="protein"):
    """Convert integer array back to sequence string."""
    mapping = INT_TO_AA if alphabet == "protein" else INT_TO_NT
    return "".join(mapping.get(int(i), "X") for i in arr)


# --- FASTA ---

def read_fasta(path_or_text, is_file=True):
    """Parse FASTA format.

    Args:
        path_or_text: file path or FASTA string
        is_file: if True, read from file; if False, parse string

    Returns:
        list of (name, sequence) tuples
    """
    if is_file:
        with open(path_or_text) as f:
            text = f.read()
    else:
        text = path_or_text

    records = []
    name = None
    seq_parts = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(seq_parts)))
            name = line[1:].split()[0]
            seq_parts = []
        else:
            seq_parts.append(line)

    if name is not None:
        records.append((name, "".join(seq_parts)))

    return records


def write_fasta(records, path=None, wrap=80):
    """Write FASTA format.

    Args:
        records: list of (name, sequence) tuples
        path: output file path (if None, return string)
        wrap: line width for sequence wrapping

    Returns:
        FASTA string if path is None
    """
    lines = []
    for name, seq in records:
        lines.append(f">{name}")
        for i in range(0, len(seq), wrap):
            lines.append(seq[i:i + wrap])
    text = "\n".join(lines) + "\n"

    if path is not None:
        with open(path, "w") as f:
            f.write(text)
    return text


# --- Stockholm ---

def read_stockholm(path_or_text, is_file=True):
    """Parse Stockholm alignment format (Pfam).

    Args:
        path_or_text: file path or Stockholm string
        is_file: if True, read from file

    Returns:
        dict with keys:
            'sequences': dict of {name: aligned_sequence}
            'gc': dict of {tag: annotation} (e.g. SS_cons for consensus structure)
            'gs': dict of {(name, tag): annotation}
            'gr': dict of {(name, tag): annotation}
    """
    if is_file:
        with open(path_or_text) as f:
            text = f.read()
    else:
        text = path_or_text

    sequences = {}
    gc = {}
    gs = {}
    gr = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("# STOCKHOLM") or line == "//":
            continue
        if line.startswith("#=GC"):
            parts = line.split(None, 2)
            if len(parts) >= 3:
                gc[parts[1]] = parts[2]
        elif line.startswith("#=GS"):
            parts = line.split(None, 3)
            if len(parts) >= 4:
                gs[(parts[1], parts[2])] = parts[3]
        elif line.startswith("#=GR"):
            parts = line.split(None, 3)
            if len(parts) >= 4:
                gr[(parts[1], parts[2])] = parts[3]
        elif line.startswith("#"):
            continue
        else:
            parts = line.split(None, 1)
            if len(parts) == 2:
                name, seq = parts
                if name in sequences:
                    sequences[name] += seq
                else:
                    sequences[name] = seq

    return {
        "sequences": sequences,
        "gc": gc,
        "gs": gs,
        "gr": gr,
    }


def alignment_to_pairs(sequences, alphabet="protein"):
    """Extract ungapped sequences from a Stockholm/FASTA alignment.

    Args:
        sequences: dict of {name: aligned_sequence} (with gap chars .-~)
        alphabet: "protein" or "dna"

    Returns:
        dict of {name: numpy int array} (ungapped)
    """
    result = {}
    for name, aln_seq in sequences.items():
        ungapped = aln_seq.replace("-", "").replace(".", "").replace("~", "")
        result[name] = seq_to_int(ungapped, alphabet)
    return result


# --- Newick ---

class TreeNode:
    """Simple tree node for Newick trees."""
    __slots__ = ["name", "branch_length", "children", "parent"]

    def __init__(self, name=None, branch_length=0.0):
        self.name = name
        self.branch_length = branch_length
        self.children = []
        self.parent = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        return child

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_root(self):
        return self.parent is None

    def leaves(self):
        if self.is_leaf:
            return [self]
        result = []
        for c in self.children:
            result.extend(c.leaves())
        return result

    def postorder(self):
        for c in self.children:
            yield from c.postorder()
        yield self

    def preorder(self):
        yield self
        for c in self.children:
            yield from c.preorder()

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf({self.name}:{self.branch_length:.4f})"
        return f"Node({self.name}:{self.branch_length:.4f}, {len(self.children)} children)"


def parse_newick(text):
    """Parse a Newick tree string.

    Args:
        text: Newick string, e.g. "((A:0.1,B:0.2):0.3,C:0.4);"

    Returns:
        TreeNode (root)
    """
    text = text.strip()
    if text.endswith(";"):
        text = text[:-1]

    pos = [0]

    def _parse():
        node = TreeNode()
        if text[pos[0]] == "(":
            pos[0] += 1  # skip (
            child = _parse()
            node.add_child(child)
            while text[pos[0]] == ",":
                pos[0] += 1  # skip ,
                child = _parse()
                node.add_child(child)
            if pos[0] < len(text) and text[pos[0]] == ")":
                pos[0] += 1  # skip )
            # Parse optional name and branch length after )
            name, bl = _parse_label()
            node.name = name
            node.branch_length = bl
        else:
            name, bl = _parse_label()
            node.name = name
            node.branch_length = bl
        return node

    def _parse_label():
        name_chars = []
        while pos[0] < len(text) and text[pos[0]] not in ":,();":
            name_chars.append(text[pos[0]])
            pos[0] += 1
        name = "".join(name_chars).strip() or None

        bl = 0.0
        if pos[0] < len(text) and text[pos[0]] == ":":
            pos[0] += 1
            bl_chars = []
            while pos[0] < len(text) and text[pos[0]] not in ",();":
                bl_chars.append(text[pos[0]])
                pos[0] += 1
            try:
                bl = float("".join(bl_chars))
            except ValueError:
                bl = 0.0
        return name, bl

    return _parse()


def write_newick(node):
    """Convert a TreeNode back to Newick string."""
    if node.is_leaf:
        name = node.name or ""
        return f"{name}:{node.branch_length}"
    children_str = ",".join(write_newick(c) for c in node.children)
    name = node.name or ""
    return f"({children_str}){name}:{node.branch_length}"


def tree_to_adjacency(root):
    """Convert tree to adjacency list with branch lengths.

    Returns:
        nodes: list of TreeNode in preorder
        edges: list of (parent_idx, child_idx, branch_length)
        leaf_names: dict of {name: node_index}
    """
    nodes = list(root.preorder())
    node_to_idx = {id(n): i for i, n in enumerate(nodes)}
    edges = []
    leaf_names = {}
    for i, node in enumerate(nodes):
        if node.is_leaf and node.name:
            leaf_names[node.name] = i
        for child in node.children:
            j = node_to_idx[id(child)]
            edges.append((i, j, child.branch_length))
    return nodes, edges, leaf_names
