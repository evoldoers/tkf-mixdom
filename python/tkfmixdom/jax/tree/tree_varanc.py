"""Product-of-trees variational ancestral reconstruction.

Implements the algorithm from misc/tree-varanc.tex: variational inference
over latent ancestral characters at internal MSA nodes, using column-wise
tree-structured blocks and order-1 WFST factors.

Plain Python/NumPy implementation (no JAX). Follows Phases 1-3 of the spec.
"""

import numpy as np
from collections import defaultdict

# State constants
from ..core.params import S, M, I, D, E

NEG_INF = -1e30
WILDCARD_IDX = 20  # unknown/ambiguous amino acid → uniform evidence
BOS = -1  # sentinel for beginning-of-sequence context
EOS = -2  # sentinel for end-of-sequence context

# Extended state types for BOS context tracking.
# BOS_I: insert state with ancestor context permanently BOS (before any M/D).
# BOS_D: delete state with descendant context permanently BOS (before any M/I).
BOS_I = 5
BOS_D = 6


# ======================================================================
# Phase 1: Structural preprocessing
# ======================================================================

class BranchInfo:
    """Precomputed structural data for one tree branch (u, v) where u = parent(v)."""
    __slots__ = [
        'parent_name', 'child_name',
        'active_cols',       # list of active MSA columns (sorted)
        'col_to_reduced',    # dict: MSA col -> reduced coordinate (1-based)
        'reduced_to_col',    # dict: reduced coordinate -> MSA col
        'branch_types',      # dict: reduced coord m -> M/I/D (1/2/3)
        'effective_types',   # dict: reduced coord m -> M/I/D/BOS_I/BOS_D (with BOS tracking)
        'n_reduced',         # U_{uv}
        'prev_anc',          # dict: m -> P^A(m) or None
        'prev_desc',         # dict: m -> P^D(m) or None
    ]

    def __init__(self):
        pass


class FactorInfo:
    """Metadata for one Potts factor."""
    __slots__ = [
        'factor_type',    # 'branch_transition', 'branch_end', 'root_transition', 'root_end'
        'branch_key',     # (parent_name, child_name) or 'root'
        'reduced_coord',  # m (1-based) or None for end factors
        'scope',          # list of (row_name, col) tuples — the latent variables
        'cols',           # set of columns touched
        'col_intersections',  # dict: col -> list of (row_name, col) in scope at that col
        'prev_state_type',    # S/M/I/D — state type at m-1
        'curr_state_type',    # M/I/D/E — state type at m (or E for end)
        # Context resolution metadata
        'anc_ctx_var',    # (row, col) or BOS — where a^- comes from
        'desc_ctx_var',   # (row, col) or BOS — where b^- comes from
        'anc_emit_var',   # (row, col) or BOS/carried — where a^+ comes from
        'desc_emit_var',  # (row, col) or BOS/carried — where b^+ comes from
    ]

    def __init__(self):
        self.scope = []
        self.cols = set()
        self.col_intersections = {}


class ColumnInfo:
    """Precomputed structural data for one MSA column."""
    __slots__ = [
        'col_idx',
        'present_nodes',   # list of node names present at this column
        'present_edges',   # list of (parent_name, child_name) edges present
        'latent_nodes',    # list of latent (internal) node names
        'latent_edges',    # list of (parent, child) edges where both are present
        'root_of_subtree', # name of shallowest present node
        'is_leaf',         # dict: node_name -> bool
    ]

    def __init__(self):
        pass


class PrecomputedStructure:
    """All immutable structural data for the variational solver."""

    def __init__(self, tree, msa_presence, leaf_seqs_aligned, wfst_per_edge, singlet_wfst):
        self.tree = tree
        self.msa_presence = msa_presence
        self.leaf_seqs_aligned = leaf_seqs_aligned
        self.wfst_per_edge = wfst_per_edge
        self.singlet_wfst = singlet_wfst

        # Determine alphabet size from singlet WFST
        # Use log_p_si shape: (A, A) for root singlet
        self.A = singlet_wfst['log_p_si'].shape[-1]
        self.L_msa = next(iter(msa_presence.values())).shape[0]

        # Build all node lists
        self.all_nodes = list(tree.preorder())
        self.node_names = [n.name for n in self.all_nodes]
        self.node_by_name = {n.name: n for n in self.all_nodes}
        self.leaf_names = set(n.name for n in self.all_nodes if n.is_leaf)
        self.internal_names = set(n.name for n in self.all_nodes if not n.is_leaf)

        # Build is_latent lookup: internal nodes at present positions
        # Leaves are observed, not latent
        self.is_latent = {}  # (node_name, col) -> bool
        for name in self.node_names:
            pres = msa_presence[name]
            is_leaf = name in self.leaf_names
            for c in range(self.L_msa):
                if pres[c] and not is_leaf:
                    self.is_latent[(name, c)] = True

        # Ungapped positions per node
        self.ungapped_positions = {}  # node_name -> list of MSA cols where present
        for name in self.node_names:
            self.ungapped_positions[name] = np.where(msa_presence[name])[0].tolist()

        # Build branch info
        self.branches = {}  # (parent_name, child_name) -> BranchInfo
        self._build_branches()

        # Build column info
        self.columns = {}  # col_idx -> ColumnInfo
        self._build_columns()

        # Build factors
        self.factors = []  # list of FactorInfo
        self._build_all_factors()

        # Index factors by column
        self.factors_by_col = defaultdict(list)  # col -> list of (factor_idx, intersection_type)
        self._index_factors_by_column()

    def _build_branches(self):
        """Build BranchInfo for every tree edge."""
        for node in self.all_nodes:
            if node.is_root:
                continue
            parent = node.parent
            bi = BranchInfo()
            bi.parent_name = parent.name
            bi.child_name = node.name

            pres_p = self.msa_presence[parent.name]
            pres_c = self.msa_presence[node.name]

            # Active columns: where either parent or child is present
            active = []
            for c in range(self.L_msa):
                if pres_p[c] or pres_c[c]:
                    active.append(c)
            bi.active_cols = active
            bi.n_reduced = len(active)

            # Maps between MSA cols and reduced coords (1-based)
            bi.col_to_reduced = {}
            bi.reduced_to_col = {}
            for m_idx, c in enumerate(active):
                m = m_idx + 1  # 1-based
                bi.col_to_reduced[c] = m
                bi.reduced_to_col[m] = c

            # Branch types
            bi.branch_types = {}
            for m in range(1, bi.n_reduced + 1):
                c = bi.reduced_to_col[m]
                p_pres = pres_p[c]
                c_pres = pres_c[c]
                if p_pres and c_pres:
                    bi.branch_types[m] = M
                elif not p_pres and c_pres:
                    bi.branch_types[m] = I
                elif p_pres and not c_pres:
                    bi.branch_types[m] = D
                else:
                    raise ValueError(f"Branch ({parent.name},{node.name}) col {c}: both absent but in active set")

            # Predecessor maps
            bi.prev_anc = {}
            bi.prev_desc = {}
            last_anc = None
            last_desc = None
            for m in range(1, bi.n_reduced + 1):
                bi.prev_anc[m] = last_anc
                bi.prev_desc[m] = last_desc
                bt = bi.branch_types[m]
                if bt in (M, D):  # ancestor emits
                    last_anc = m
                if bt in (M, I):  # descendant emits
                    last_desc = m

            # Store final predecessor state for end factor
            bi.prev_anc[bi.n_reduced + 1] = last_anc
            bi.prev_desc[bi.n_reduced + 1] = last_desc

            # Effective state types: distinguish BOS_I / BOS_D from regular I / D.
            # BOS_I: insert where ancestor context is BOS (no prior M/D on branch).
            # BOS_D: delete where descendant context is BOS (no prior M/I on branch).
            bi.effective_types = {}
            for m in range(1, bi.n_reduced + 1):
                bt = bi.branch_types[m]
                if bt == I and bi.prev_anc[m] is None:
                    bi.effective_types[m] = BOS_I
                elif bt == D and bi.prev_desc[m] is None:
                    bi.effective_types[m] = BOS_D
                else:
                    bi.effective_types[m] = bt

            self.branches[(parent.name, node.name)] = bi

    def _build_columns(self):
        """Build ColumnInfo for every MSA column."""
        for c in range(self.L_msa):
            ci = ColumnInfo()
            ci.col_idx = c

            # Present nodes
            ci.present_nodes = []
            ci.is_leaf = {}
            for name in self.node_names:
                if self.msa_presence[name][c]:
                    ci.present_nodes.append(name)
                    ci.is_leaf[name] = name in self.leaf_names

            # Present edges (parent-child pairs where both present)
            ci.present_edges = []
            ci.latent_nodes = []
            ci.latent_edges = []
            for name in ci.present_nodes:
                if name not in self.leaf_names:
                    ci.latent_nodes.append(name)
                node = self.node_by_name[name]
                if not node.is_root and node.parent.name in [n for n in ci.present_nodes]:
                    ci.present_edges.append((node.parent.name, name))
                    # latent edge if both endpoints present (for BP purposes)
                    ci.latent_edges.append((node.parent.name, name))

            # Root of the present subtree
            # Find shallowest present node by tree depth
            if ci.present_nodes:
                # Use preorder - first encountered is shallowest
                for node in self.tree.preorder():
                    if node.name in ci.present_nodes:
                        ci.root_of_subtree = node.name
                        break
            else:
                ci.root_of_subtree = None

            self.columns[c] = ci

    def _resolve_context_var(self, branch_key, m, which):
        """Resolve where a context variable comes from.

        Args:
            branch_key: (parent_name, child_name)
            m: reduced coordinate (1-based)
            which: 'anc_ctx', 'desc_ctx', 'anc_emit', 'desc_emit'

        Returns:
            (row_name, col) tuple if latent variable, or
            BOS sentinel, or
            ('observed', row_name, col, char_idx) if leaf observed
        """
        bi = self.branches[branch_key]
        parent_name, child_name = branch_key

        if which == 'anc_ctx':
            # a^- : previous ancestral context
            pa = bi.prev_anc.get(m)
            if pa is None:
                return BOS
            c_prev = bi.reduced_to_col[pa]
            # The ancestor emitted at pa, so the variable is (parent, c_prev)
            return (parent_name, c_prev)

        elif which == 'desc_ctx':
            # b^- : previous descendant context
            pd = bi.prev_desc.get(m)
            if pd is None:
                return BOS
            c_prev = bi.reduced_to_col[pd]
            return (child_name, c_prev)

        elif which == 'anc_emit':
            # a^+ : ancestor emit at current position
            bt = bi.branch_types[m]
            if bt in (M, D):
                c = bi.reduced_to_col[m]
                return (parent_name, c)
            else:  # I: ancestor context carried
                return self._resolve_context_var(branch_key, m, 'anc_ctx')

        elif which == 'desc_emit':
            # b^+ : descendant emit at current position
            bt = bi.branch_types[m]
            if bt in (M, I):
                c = bi.reduced_to_col[m]
                return (child_name, c)
            else:  # D: descendant context carried
                return self._resolve_context_var(branch_key, m, 'desc_ctx')

    def _var_is_latent(self, var):
        """Check if a variable reference is a latent variable."""
        if var == BOS or var == EOS:
            return False
        if isinstance(var, tuple) and len(var) == 2:
            return var in self.is_latent
        return False

    def _extract_scope(self, branch_key, m, is_end=False):
        """Extract factor scope for branch transition at reduced coord m.

        Returns list of (row_name, col) for latent variables in scope.
        """
        bi = self.branches[branch_key]
        parent_name, child_name = branch_key
        scope_set = set()

        if is_end:
            # End factor: context from final position
            # prev_anc and prev_desc at U+1
            pa = bi.prev_anc.get(bi.n_reduced + 1)
            pd = bi.prev_desc.get(bi.n_reduced + 1)
            if pa is not None:
                c_pa = bi.reduced_to_col[pa]
                var = (parent_name, c_pa)
                if self._var_is_latent(var):
                    scope_set.add(var)
            if pd is not None:
                c_pd = bi.reduced_to_col[pd]
                var = (child_name, c_pd)
                if self._var_is_latent(var):
                    scope_set.add(var)
        else:
            bt = bi.branch_types[m]
            c = bi.reduced_to_col[m]

            # Current emitted variables
            if bt in (M, D):
                var = (parent_name, c)
                if self._var_is_latent(var):
                    scope_set.add(var)
            if bt in (M, I):
                var = (child_name, c)
                if self._var_is_latent(var):
                    scope_set.add(var)

            # Previous context variables
            pa = bi.prev_anc[m]
            if pa is not None:
                c_pa = bi.reduced_to_col[pa]
                var = (parent_name, c_pa)
                if self._var_is_latent(var):
                    scope_set.add(var)

            pd = bi.prev_desc[m]
            if pd is not None:
                c_pd = bi.reduced_to_col[pd]
                var = (child_name, c_pd)
                if self._var_is_latent(var):
                    scope_set.add(var)

        # Sort scope canonically: by column, then by tree depth
        scope = sorted(scope_set, key=lambda rc: (rc[1], self._node_depth(rc[0])))
        return scope

    def _node_depth(self, name):
        """Get depth of a node in the tree."""
        node = self.node_by_name[name]
        depth = 0
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth

    def _build_all_factors(self):
        """Build all factor metadata."""
        self.factors = []

        # Branch factors
        for node in self.all_nodes:
            if node.is_root:
                continue
            branch_key = (node.parent.name, node.name)
            bi = self.branches[branch_key]

            # Transition factors for each reduced coordinate
            for m in range(1, bi.n_reduced + 1):
                fi = FactorInfo()
                fi.factor_type = 'branch_transition'
                fi.branch_key = branch_key
                fi.reduced_coord = m
                fi.curr_state_type = bi.effective_types[m]
                fi.prev_state_type = S if m == 1 else bi.effective_types[m - 1]
                fi.scope = self._extract_scope(branch_key, m)

                # Resolve context variables for this factor
                fi.anc_ctx_var = self._resolve_context_var(branch_key, m, 'anc_ctx')
                fi.desc_ctx_var = self._resolve_context_var(branch_key, m, 'desc_ctx')
                fi.anc_emit_var = self._resolve_context_var(branch_key, m, 'anc_emit')
                fi.desc_emit_var = self._resolve_context_var(branch_key, m, 'desc_emit')

                fi.cols = set(rc[1] for rc in fi.scope)
                fi.col_intersections = {}
                for c in fi.cols:
                    fi.col_intersections[c] = [rc for rc in fi.scope if rc[1] == c]
                self.factors.append(fi)

            # End factor
            fi = FactorInfo()
            fi.factor_type = 'branch_end'
            fi.branch_key = branch_key
            fi.reduced_coord = None
            fi.curr_state_type = E
            fi.prev_state_type = bi.effective_types[bi.n_reduced] if bi.n_reduced > 0 else S
            fi.scope = self._extract_scope(branch_key, None, is_end=True)

            # Context for end factor
            pa = bi.prev_anc.get(bi.n_reduced + 1)
            pd = bi.prev_desc.get(bi.n_reduced + 1)
            fi.anc_ctx_var = BOS if pa is None else (branch_key[0], bi.reduced_to_col[pa])
            fi.desc_ctx_var = BOS if pd is None else (branch_key[1], bi.reduced_to_col[pd])
            fi.anc_emit_var = EOS
            fi.desc_emit_var = EOS

            fi.cols = set(rc[1] for rc in fi.scope)
            fi.col_intersections = {}
            for c in fi.cols:
                fi.col_intersections[c] = [rc for rc in fi.scope if rc[1] == c]
            self.factors.append(fi)

        # Root factors
        root_name = self.tree.name
        root_cols = self.ungapped_positions[root_name]

        for idx, c in enumerate(root_cols):
            m = idx + 1  # 1-based
            fi = FactorInfo()
            fi.factor_type = 'root_transition'
            fi.branch_key = 'root'
            fi.reduced_coord = m
            fi.curr_state_type = BOS_I  # root positions are all inserts with BOS ancestor ctx
            fi.prev_state_type = S if m == 1 else BOS_I

            # Root scope: current root position + previous root position
            scope = []
            if self._var_is_latent((root_name, c)):
                scope.append((root_name, c))
            if m >= 2:
                c_prev = root_cols[idx - 1]
                if self._var_is_latent((root_name, c_prev)):
                    scope.append((root_name, c_prev))
            # Sort: by column then depth
            fi.scope = sorted(scope, key=lambda rc: (rc[1], self._node_depth(rc[0])))

            # Context for root: ancestor is always BOS
            fi.anc_ctx_var = BOS
            fi.desc_ctx_var = BOS if m == 1 else (root_name, root_cols[idx - 1])
            fi.anc_emit_var = BOS
            fi.desc_emit_var = (root_name, c)

            fi.cols = set(rc[1] for rc in fi.scope)
            fi.col_intersections = {}
            for col in fi.cols:
                fi.col_intersections[col] = [rc for rc in fi.scope if rc[1] == col]
            self.factors.append(fi)

        # Root end factor
        fi = FactorInfo()
        fi.factor_type = 'root_end'
        fi.branch_key = 'root'
        fi.reduced_coord = None
        fi.curr_state_type = E
        fi.prev_state_type = BOS_I if len(root_cols) > 0 else S

        scope = []
        if len(root_cols) > 0:
            c_last = root_cols[-1]
            if self._var_is_latent((root_name, c_last)):
                scope.append((root_name, c_last))
        fi.scope = scope
        fi.anc_ctx_var = BOS
        fi.desc_ctx_var = BOS if len(root_cols) == 0 else (root_name, root_cols[-1])
        fi.anc_emit_var = EOS
        fi.desc_emit_var = EOS

        fi.cols = set(rc[1] for rc in fi.scope)
        fi.col_intersections = {}
        for c in fi.cols:
            fi.col_intersections[c] = [rc for rc in fi.scope if rc[1] == c]
        self.factors.append(fi)

    def _index_factors_by_column(self):
        """Index factors by their participating columns with intersection type."""
        self.factors_unary_by_col = defaultdict(list)   # col -> [(factor_idx, (row,col))]
        self.factors_pair_by_col = defaultdict(list)     # col -> [(factor_idx, (parent_row,col), (child_row,col))]

        for fi_idx, fi in enumerate(self.factors):
            for c, vars_in_col in fi.col_intersections.items():
                if len(vars_in_col) == 1:
                    self.factors_unary_by_col[c].append((fi_idx, vars_in_col[0]))
                elif len(vars_in_col) == 2:
                    # Determine parent-child order
                    v0, v1 = vars_in_col
                    n0 = self.node_by_name[v0[0]]
                    n1 = self.node_by_name[v1[0]]
                    if n1.parent is not None and n1.parent.name == v0[0]:
                        self.factors_pair_by_col[c].append((fi_idx, v0, v1))
                    elif n0.parent is not None and n0.parent.name == v1[0]:
                        self.factors_pair_by_col[c].append((fi_idx, v1, v0))
                    else:
                        raise ValueError(f"Factor {fi_idx} col {c}: pair {vars_in_col} not parent-child")


# ======================================================================
# Phase 2: Factor table construction
# ======================================================================

def _get_observed_char(precomp, row_name, col):
    """Get observed character at a leaf position, or None if latent."""
    if row_name in precomp.leaf_names:
        return int(precomp.leaf_seqs_aligned[row_name][col])
    return None


def _resolve_var_value(var, precomp, assignment=None):
    """Resolve a context/emit variable to a concrete character index or sentinel.

    Args:
        var: BOS, EOS, or (row_name, col)
        precomp: PrecomputedStructure
        assignment: dict (row_name, col) -> character index for latent vars

    Returns:
        Character index (int >= 0), or BOS/EOS sentinel
    """
    if var == BOS:
        return BOS
    if var == EOS:
        return EOS
    if isinstance(var, tuple) and len(var) == 2:
        row_name, col = var
        # Check if observed
        obs = _get_observed_char(precomp, row_name, col)
        if obs is not None:
            return obs
        # Must be latent — look up in assignment
        if assignment is not None and var in assignment:
            return assignment[var]
        return None  # Variable, not resolved
    return None


def build_factor_table(fi, precomp):
    """Build a dense log-factor table for one factor (vectorized).

    Directly slices and permutes WFST tensors instead of looping over
    all A^arity assignments. For arity-4 at A=20, this is ~160,000x faster.

    Returns:
        log_table: ndarray of shape (A,) * len(fi.scope)
        The axes correspond to fi.scope in order.
    """
    A = precomp.A
    scope = fi.scope
    arity = len(scope)

    if arity == 0:
        val = _evaluate_factor_single(fi, precomp, {})
        result = np.array(val)
        assert np.isfinite(result), (
            f"Scalar factor is -inf: type={fi.factor_type}, "
            f"branch={fi.branch_key}, m={fi.reduced_coord}")
        return result

    table = _build_factor_table_vectorized(fi, precomp)

    assert np.max(table) > NEG_INF, (
        f"All-(-inf) factor table: type={fi.factor_type}, "
        f"branch={fi.branch_key}, m={fi.reduced_coord}, scope={fi.scope}")

    return table


def _build_factor_table_vectorized(fi, precomp):
    """Vectorized factor table construction via tensor slicing.

    For a factor with state transition X→Y, the WFST tensor has axes
    corresponding to (a-, b-, a+, b+) with some axes absent depending
    on state types. Each present axis maps to either:
      - a scope variable (kept as free axis)
      - an observed leaf character (sliced at fixed index)
      - BOS (handled by state type: S/BOS_I/BOS_D tensors have fewer axes)

    The result is sliced, then transposed to match fi.scope order.
    """
    A = precomp.A
    scope = fi.scope
    X = fi.prev_state_type
    Y_raw = fi.curr_state_type

    # Effective dest type for key lookup
    Y = I if Y_raw == BOS_I else (D if Y_raw == BOS_D else Y_raw)

    # Get WFST
    if fi.factor_type in ('branch_transition', 'branch_end'):
        wfst = precomp.wfst_per_edge[fi.branch_key]
    else:
        wfst = precomp.singlet_wfst

    # Determine tensor key and axis semantics based on (X, Y)
    # Returns: tensor, list of (var_or_fixed, ...) per axis
    tensor, axis_vars = _get_tensor_and_axes(fi, wfst, X, Y, precomp)
    tensor = np.asarray(tensor, dtype=np.float64)

    if tensor.ndim == 0:
        # Scalar — should have been caught by arity==0 check
        return tensor

    # Build the output by slicing fixed axes and keeping free axes
    # We need to map each tensor axis to either:
    #   - a scope variable index (keep as output axis)
    #   - a fixed integer (slice)

    # First pass: determine which axes are free vs fixed
    var_to_scope_idx = {v: i for i, v in enumerate(scope)}
    slices = []
    free_axes = []  # (tensor_axis, scope_idx) for free axes
    for tax, var in enumerate(axis_vars):
        if var in var_to_scope_idx:
            slices.append(slice(None))
            free_axes.append((tax, var_to_scope_idx[var]))
        else:
            # Fixed: either observed char or BOS→0
            val = _resolve_fixed_var(var, fi, precomp)
            slices.append(val)

    result = tensor[tuple(slices)]

    # Transpose to match scope order if needed
    if len(free_axes) > 1:
        # free_axes is [(tensor_axis_after_slicing, scope_idx), ...]
        # After slicing, the remaining axes are in order of free_axes
        current_order = [si for _, si in free_axes]
        if current_order != list(range(len(scope))):
            # Need to transpose
            perm = [0] * len(scope)
            for out_pos, (_, scope_idx) in enumerate(free_axes):
                perm[scope_idx] = out_pos
            result = np.transpose(result, perm)

    return result


def _get_tensor_and_axes(fi, wfst, X, Y, precomp):
    """Get the WFST tensor and its axis-to-variable mapping.

    Returns:
        tensor: numpy array
        axis_vars: list of variable references per axis. Each is either:
            - (row_name, col): a latent or observed variable
            - BOS: BOS sentinel
            - an int: a fixed observed character index
    """
    state_suffix = {M: 'm', I: 'i', D: 'd', E: 'e'}
    y_suf = state_suffix[Y]

    # Resolve the four context/emit variables to their sources
    a_minus_var = fi.anc_ctx_var
    b_minus_var = fi.desc_ctx_var
    a_plus_var = fi.anc_emit_var
    b_plus_var = fi.desc_emit_var

    if X == S:
        key = f'log_p_s{y_suf}'
        tensor = wfst[key]
        if Y == E:
            return tensor, []
        elif Y == M:
            # (A, A) = [a+, b+]
            return tensor, [a_plus_var, b_plus_var]
        elif Y == I:
            # (A, A) = [dummy_anc(BOS), b+]
            return tensor, [BOS, b_plus_var]
        elif Y == D:
            # (A, A) = [a+, dummy_desc(BOS)]
            return tensor, [a_plus_var, BOS]

    elif X == BOS_I:
        key_bos = f'log_p_bos_i_{y_suf}'
        if key_bos in wfst:
            tensor = wfst[key_bos]
            if Y == E:
                return tensor, [b_minus_var]
            elif Y == M:
                return tensor, [b_minus_var, a_plus_var, b_plus_var]
            elif Y == I:
                return tensor, [b_minus_var, b_plus_var]
            elif Y == D:
                return tensor, [b_minus_var, a_plus_var]
        else:
            # Fallback: regular I tensor, anc axis at 0
            key = f'log_p_i{y_suf}'
            tensor = wfst[key]
            if Y == E:
                return tensor, [BOS, b_minus_var]
            elif Y == M:
                return tensor, [BOS, b_minus_var, a_plus_var, b_plus_var]
            elif Y == I:
                return tensor, [BOS, b_minus_var, b_plus_var]
            elif Y == D:
                return tensor, [BOS, b_minus_var, a_plus_var]

    elif X == BOS_D:
        key_bos = f'log_p_bos_d_{y_suf}'
        if key_bos in wfst:
            tensor = wfst[key_bos]
            if Y == E:
                return tensor, [a_minus_var]
            elif Y == M:
                return tensor, [a_minus_var, a_plus_var, b_plus_var]
            elif Y == I:
                return tensor, [a_minus_var, b_plus_var]
            elif Y == D:
                return tensor, [a_minus_var, a_plus_var]
        else:
            key = f'log_p_d{y_suf}'
            tensor = wfst[key]
            if Y == E:
                return tensor, [a_minus_var, BOS]
            elif Y == M:
                return tensor, [a_minus_var, BOS, a_plus_var, b_plus_var]
            elif Y == I:
                return tensor, [a_minus_var, BOS, b_plus_var]
            elif Y == D:
                return tensor, [a_minus_var, BOS, a_plus_var]

    else:
        # Regular M, I, D
        state_prefix = {M: 'm', I: 'i', D: 'd'}
        key = f'log_p_{state_prefix[X]}{y_suf}'
        tensor = wfst[key]
        if Y == E:
            return tensor, [a_minus_var, b_minus_var]
        elif Y == M:
            return tensor, [a_minus_var, b_minus_var, a_plus_var, b_plus_var]
        elif Y == I:
            return tensor, [a_minus_var, b_minus_var, b_plus_var]
        elif Y == D:
            return tensor, [a_minus_var, b_minus_var, a_plus_var]

    # Shouldn't reach here
    return np.array(NEG_INF), []


def _resolve_fixed_var(var, fi, precomp):
    """Resolve a non-scope variable to a fixed integer index for slicing."""
    if var == BOS or var == EOS:
        return 0
    if isinstance(var, tuple) and len(var) == 2:
        row_name, col = var
        # Check if observed leaf
        obs = _get_observed_char(precomp, row_name, col)
        if obs is not None:
            return int(obs)
        # It's a latent variable not in scope — this shouldn't happen
        # for well-formed factors, but handle gracefully
        raise ValueError(f"Latent variable {var} not in scope {fi.scope}")
    return 0


def build_factor_table_naive(fi, precomp):
    """Build factor table by enumeration (reference implementation for testing).

    Retained as a cross-check for the vectorized version.
    """
    A = precomp.A
    scope = fi.scope
    arity = len(scope)

    if arity == 0:
        val = _evaluate_factor_single(fi, precomp, {})
        return np.array(val)

    shape = tuple([A] * arity)
    table = np.full(shape, NEG_INF)

    for flat_idx in range(A ** arity):
        assignment = {}
        idx = flat_idx
        multi_idx = []
        for i in range(arity - 1, -1, -1):
            assignment[scope[i]] = idx % A
            multi_idx.insert(0, idx % A)
            idx //= A
        val = _evaluate_factor_single(fi, precomp, assignment)
        table[tuple(multi_idx)] = val

    return table


def _evaluate_factor_single(fi, precomp, assignment):
    """Evaluate a single factor at a specific assignment of its scope variables.

    Returns log W(X, a-, b-, Y, a+, b+).
    """
    # Determine which WFST to use
    if fi.factor_type in ('branch_transition', 'branch_end'):
        wfst = precomp.wfst_per_edge[fi.branch_key]
    else:
        wfst = precomp.singlet_wfst

    # Resolve all four context/emit variables
    a_minus = _resolve_var_value(fi.anc_ctx_var, precomp, assignment)
    b_minus = _resolve_var_value(fi.desc_ctx_var, precomp, assignment)
    a_plus = _resolve_var_value(fi.anc_emit_var, precomp, assignment)
    b_plus = _resolve_var_value(fi.desc_emit_var, precomp, assignment)

    X = fi.prev_state_type
    Y = fi.curr_state_type

    return _lookup_wfst_weight(wfst, X, a_minus, b_minus, Y, a_plus, b_plus, precomp.A)


def _lookup_wfst_weight(wfst, X, a_minus, b_minus, Y, a_plus, b_plus, A):
    """Look up log W(X, a-, b-, Y, a+, b+) from WFST tensors.

    Handles S, M, I, D, BOS_I, BOS_D as source state types.
    BOS_I/BOS_D use separate tensor keys so BOS context is never mapped
    to an alphabet index — the BOS axis is eliminated from the tensor.

    The WFST tensor convention:
        4D: [prev_anc_ctx, prev_desc_ctx, curr_anc_emit, curr_desc_emit]
        3D: missing one context or emit axis
        2D: two axes
        1D: one axis
        scalar: no axes (S->E)
    """
    # Determine effective dest type for WFST key (BOS_I self-loop)
    Y_key = Y
    if Y == BOS_I:
        Y_key = I  # dest key is 'i' for inserts (BOS_I is about source state)
    elif Y == BOS_D:
        Y_key = D

    state_suffix = {M: 'm', I: 'i', D: 'd', E: 'e'}
    y_suffix = state_suffix.get(Y_key, state_suffix.get(Y, 'e'))

    def _to_int(val):
        """Convert to integer index. BOS/EOS should not reach here for real axes."""
        if val == BOS or val == EOS:
            raise ValueError(f"BOS/EOS reached _to_int — should have been handled by state type")
        return int(val)

    if X == S:
        key = f'log_p_s{y_suffix}'
        tensor = np.asarray(wfst[key])

        if Y_key == E:
            return float(tensor)
        elif Y_key == M:
            # (A, A) — [curr_anc, curr_desc]
            return float(tensor[_to_int(a_plus), _to_int(b_plus)])
        elif Y_key == I:
            # S→I: (A, A) — [dummy_anc_ctx(=0), curr_desc]
            # For S, ancestor context is always BOS. Use index 0.
            return float(tensor[0, _to_int(b_plus)])
        elif Y_key == D:
            # S→D: (A, A) — [curr_anc, dummy_desc_ctx(=0)]
            return float(tensor[_to_int(a_plus), 0])

    elif X == BOS_I:
        # BOS_I: ancestor context is BOS. Tensors from WFST keyed by 'bos_i_*'
        # with shapes that drop the ancestor context axis.
        # Fallback: use regular I tensors, index ancestor axis at 0.
        key_bos = f'log_p_bos_i_{y_suffix}'
        if key_bos in wfst:
            tensor = np.asarray(wfst[key_bos])
            if Y_key == E:
                # (A,) — [prev_desc_ctx]
                return float(tensor[_to_int(b_minus)])
            elif Y_key == M:
                # (A, A, A) — [prev_desc_ctx, curr_anc, curr_desc]
                return float(tensor[_to_int(b_minus), _to_int(a_plus), _to_int(b_plus)])
            elif Y_key == I:
                # (A, A) — [prev_desc_ctx, curr_desc]
                return float(tensor[_to_int(b_minus), _to_int(b_plus)])
            elif Y_key == D:
                # (A, A) — [prev_desc_ctx, curr_anc]
                return float(tensor[_to_int(b_minus), _to_int(a_plus)])
        else:
            # Fallback: use regular I tensor, index ancestor ctx at 0.
            key = f'log_p_i{y_suffix}'
            tensor = np.asarray(wfst[key])
            if Y_key == E:
                return float(tensor[0, _to_int(b_minus)])
            elif Y_key == M:
                return float(tensor[0, _to_int(b_minus), _to_int(a_plus), _to_int(b_plus)])
            elif Y_key == I:
                return float(tensor[0, _to_int(b_minus), _to_int(b_plus)])
            elif Y_key == D:
                return float(tensor[0, _to_int(b_minus), _to_int(a_plus)])

    elif X == BOS_D:
        # BOS_D: descendant context is BOS. Tensors keyed by 'bos_d_*'.
        key_bos = f'log_p_bos_d_{y_suffix}'
        if key_bos in wfst:
            tensor = np.asarray(wfst[key_bos])
            if Y_key == E:
                # (A,) — [prev_anc_ctx]
                return float(tensor[_to_int(a_minus)])
            elif Y_key == M:
                # (A, A, A) — [prev_anc_ctx, curr_anc, curr_desc]
                return float(tensor[_to_int(a_minus), _to_int(a_plus), _to_int(b_plus)])
            elif Y_key == I:
                # (A, A) — [prev_anc_ctx, curr_desc]
                return float(tensor[_to_int(a_minus), _to_int(b_plus)])
            elif Y_key == D:
                # (A, A) — [prev_anc_ctx, curr_anc]
                return float(tensor[_to_int(a_minus), _to_int(a_plus)])
        else:
            # Fallback: use regular D tensor, index descendant ctx at 0.
            key = f'log_p_d{y_suffix}'
            tensor = np.asarray(wfst[key])
            if Y_key == E:
                return float(tensor[_to_int(a_minus), 0])
            elif Y_key == M:
                return float(tensor[_to_int(a_minus), 0, _to_int(a_plus), _to_int(b_plus)])
            elif Y_key == I:
                return float(tensor[_to_int(a_minus), 0, _to_int(b_plus)])
            elif Y_key == D:
                return float(tensor[_to_int(a_minus), 0, _to_int(a_plus)])

    else:
        # Regular M, I, D source states — all contexts must be real characters
        state_prefix = {M: 'm', I: 'i', D: 'd'}
        key = f'log_p_{state_prefix[X]}{y_suffix}'
        tensor = np.asarray(wfst[key])

        if Y_key == E:
            return float(tensor[_to_int(a_minus), _to_int(b_minus)])
        elif Y_key == M:
            return float(tensor[_to_int(a_minus), _to_int(b_minus), _to_int(a_plus), _to_int(b_plus)])
        elif Y_key == I:
            return float(tensor[_to_int(a_minus), _to_int(b_minus), _to_int(b_plus)])
        elif Y_key == D:
            return float(tensor[_to_int(a_minus), _to_int(b_minus), _to_int(a_plus)])

    return NEG_INF


# ======================================================================
# TKF91 WFST factories (conditional form — no kappa in branch)
# ======================================================================

def build_tkf91_branch_wfst(ins_rate, del_rate, Q, pi, t):
    """Build TKF91 branch WFST in conditional (WFST) form.

    Uses tkf91_trans_cond: rows don't sum to 1, kappa is NOT included.
    The kappa factor lives in the root singlet.
    """
    if ins_rate >= del_rate:
        raise ValueError(f"TKF91 requires ins_rate < del_rate, got {ins_rate} >= {del_rate}")
    if t < 0:
        raise ValueError(f"Branch length must be non-negative, got {t}")

    from ..core.params import tkf91_trans_cond
    from ..core.ctmc import transition_matrix

    A = pi.shape[0]
    tau = np.asarray(tkf91_trans_cond(ins_rate, del_rate, t))
    P = np.asarray(transition_matrix(Q, t))

    sl = lambda x: np.log(np.maximum(x, 1e-300))

    # The WFST is context-independent for TKF91, but we construct
    # full-rank tensors that broadcast correctly for the order-1 code path.

    # Match transitions: log(tau[X,M]) + log(P[a_plus, b_plus])
    # 4D tensors are constant in context axes
    log_P = sl(P)

    log_p_mm = np.broadcast_to(sl(tau[M, M]) + log_P[None, None, :, :], (A, A, A, A)).copy()
    log_p_mi = np.broadcast_to(sl(tau[M, I]) + sl(pi)[None, None, :], (A, A, A)).copy()
    log_p_md = np.full((A, A, A), sl(tau[M, D]))
    log_p_me = np.full((A, A), sl(tau[M, E]))

    log_p_im = np.broadcast_to(sl(tau[I, M]) + log_P[None, None, :, :], (A, A, A, A)).copy()
    log_p_ii = np.broadcast_to(sl(tau[I, I]) + sl(pi)[None, None, :], (A, A, A)).copy()
    log_p_id = np.full((A, A, A), sl(tau[I, D]))
    log_p_ie = np.full((A, A), sl(tau[I, E]))

    log_p_dm = np.broadcast_to(sl(tau[D, M]) + log_P[None, None, :, :], (A, A, A, A)).copy()
    log_p_dd = np.full((A, A, A), sl(tau[D, D]))
    log_p_di = np.broadcast_to(sl(tau[D, I]) + sl(pi)[None, None, :], (A, A, A)).copy()
    log_p_de = np.full((A, A), sl(tau[D, E]))

    # Start transitions: S row same as M/I rows in TKF91
    log_p_sm = (sl(tau[S, M]) + log_P).copy()       # (A, A)
    log_p_si = np.broadcast_to(sl(tau[S, I]) + sl(pi)[None, :], (A, A)).copy()
    log_p_sd = np.full((A, A), sl(tau[S, D]))   # (A, A) - [curr_anc, desc_ctx]
    log_p_se = float(sl(tau[S, E]))

    # BOS_I tensors: insert with ancestor context = BOS.
    # For TKF91 (order-0), these are slices at index 0 of the I tensors.
    # For order-1 WFSTs, these would have BOS-specific values.
    log_p_bos_i_m = log_p_im[0]          # (A, A, A) = [b-, a+, b+]
    log_p_bos_i_i = log_p_ii[0]          # (A, A)    = [b-, b+]
    log_p_bos_i_d = log_p_id[0]          # (A, A)    = [b-, a+]
    log_p_bos_i_e = log_p_ie[0]          # (A,)      = [b-]

    # BOS_D tensors: delete with descendant context = BOS.
    log_p_bos_d_m = log_p_dm[:, 0]       # (A, A, A) = [a-, a+, b+]
    log_p_bos_d_i = log_p_di[:, 0]       # (A, A)    = [a-, b+]
    log_p_bos_d_d = log_p_dd[:, 0]       # (A, A)    = [a-, a+]
    log_p_bos_d_e = log_p_de[:, 0]       # (A,)      = [a-]

    return {
        'log_p_mm': log_p_mm, 'log_p_mi': log_p_mi, 'log_p_md': log_p_md, 'log_p_me': log_p_me,
        'log_p_im': log_p_im, 'log_p_ii': log_p_ii, 'log_p_id': log_p_id, 'log_p_ie': log_p_ie,
        'log_p_dm': log_p_dm, 'log_p_dd': log_p_dd, 'log_p_di': log_p_di, 'log_p_de': log_p_de,
        'log_p_sm': log_p_sm, 'log_p_si': log_p_si, 'log_p_sd': log_p_sd, 'log_p_se': log_p_se,
        'log_p_bos_i_m': log_p_bos_i_m, 'log_p_bos_i_i': log_p_bos_i_i,
        'log_p_bos_i_d': log_p_bos_i_d, 'log_p_bos_i_e': log_p_bos_i_e,
        'log_p_bos_d_m': log_p_bos_d_m, 'log_p_bos_d_i': log_p_bos_d_i,
        'log_p_bos_d_d': log_p_bos_d_d, 'log_p_bos_d_e': log_p_bos_d_e,
    }


def build_tkf91_root_wfst(ins_rate, del_rate, pi):
    """Build TKF91 root (singlet) WFST.

    Root WFST: geometric length with kappa = ins_rate/del_rate.
    All positions are "inserts" with emission pi.
    Ancestral context is permanently BOS.
    """
    if ins_rate >= del_rate:
        raise ValueError(f"TKF91 requires ins_rate < del_rate, got {ins_rate} >= {del_rate}")

    from ..core.bdi import tkf_kappa

    A = pi.shape[0]
    kappa = float(tkf_kappa(ins_rate, del_rate))

    sl = lambda x: np.log(np.maximum(x, 1e-300))

    log_kappa_pi = sl(kappa) + sl(pi)

    # S -> I: kappa * pi[b], shape (A, A) = [anc_ctx(BOS), desc_emit]
    log_p_si = np.broadcast_to(log_kappa_pi[None, :], (A, A)).copy()

    # S -> E: 1 - kappa
    log_p_se = float(sl(1.0 - kappa))

    # I -> I: kappa * pi[b], shape (A, A, A) = [prev_anc_ctx, prev_desc_ctx, curr_desc]
    # Context-independent in TKF91, broadcast over prev axes
    log_p_ii = np.broadcast_to(log_kappa_pi[None, None, :], (A, A, A)).copy()

    # I -> E: 1 - kappa, shape (A, A) = [prev_anc_ctx, prev_desc_ctx]
    log_p_ie = np.full((A, A), sl(1.0 - kappa))

    # Fill in all other keys with NEG_INF (impossible transitions for root)
    impossible_4d = np.full((A, A, A, A), NEG_INF)
    impossible_3d = np.full((A, A, A), NEG_INF)
    impossible_2d = np.full((A, A), NEG_INF)

    # BOS_I tensors for root (all root inserts are BOS_I)
    log_p_bos_i_m = np.full((A, A, A), NEG_INF)    # impossible for root
    log_p_bos_i_i = log_p_ii[0].copy()             # (A, A) = [b-, b+]
    log_p_bos_i_d = np.full((A, A), NEG_INF)       # impossible for root
    log_p_bos_i_e = log_p_ie[0].copy()             # (A,) = [b-]

    return {
        'log_p_mm': impossible_4d, 'log_p_mi': impossible_3d,
        'log_p_md': impossible_3d, 'log_p_me': impossible_2d,
        'log_p_im': impossible_4d, 'log_p_ii': log_p_ii,
        'log_p_id': impossible_3d, 'log_p_ie': log_p_ie,
        'log_p_dm': impossible_4d, 'log_p_dd': impossible_3d,
        'log_p_di': impossible_3d, 'log_p_de': impossible_2d,
        'log_p_sm': impossible_2d, 'log_p_si': log_p_si,
        'log_p_sd': impossible_2d, 'log_p_se': log_p_se,
        'log_p_bos_i_m': log_p_bos_i_m, 'log_p_bos_i_i': log_p_bos_i_i,
        'log_p_bos_i_d': log_p_bos_i_d, 'log_p_bos_i_e': log_p_bos_i_e,
    }


# ======================================================================
# Dynamic site class helpers
# ======================================================================

def make_class_exposed_leaf_evidence(char_idx, D, A, rate_mask_col=None, G=None):
    """Build leaf evidence vector for class-exposed alphabet.

    For observed residue char_idx, all dynamic classes are equally likely.
    Gamma classes are constrained by the rate mask.

    Args:
        char_idx: observed residue index (0..A-1), or -1 for gap
        D: number of dynamic classes
        A: base alphabet size (e.g. 20 for amino acids)
        rate_mask_col: optional (G,) binary mask for gamma classes at this column.
            If None, all gamma classes allowed (or G=1).
        G: number of gamma classes. Required if rate_mask_col is provided.

    Returns:
        (A_eff,) log-evidence vector where A_eff = D*A (if G is None/1)
        or G*D*A (if G > 1 and rate_mask_col is provided).
    """
    if char_idx < 0:
        return None  # gap: no evidence

    if char_idx == WILDCARD_IDX:
        # Wildcard: uniform over all residues (log-prob 0 = prob 1)
        if rate_mask_col is not None and G is not None and G > 1:
            A_eff = G * D * A
            ev = np.full(A_eff, NEG_INF)
            for g in range(G):
                if rate_mask_col[g]:
                    for d in range(D):
                        for a in range(A):
                            ev[g * D * A + d * A + a] = 0.0
        else:
            A_eff = D * A
            ev = np.full(A_eff, NEG_INF)
            for d in range(D):
                for a in range(A):
                    ev[d * A + a] = 0.0
        return ev

    if rate_mask_col is not None and G is not None and G > 1:
        # Full (G, D, A) state
        A_eff = G * D * A
        ev = np.full(A_eff, NEG_INF)
        for g in range(G):
            if rate_mask_col[g]:
                for d in range(D):
                    ev[g * D * A + d * A + char_idx] = 0.0
    else:
        # (D, A) state only (gamma marginalized or G=1)
        A_eff = D * A
        ev = np.full(A_eff, NEG_INF)
        for d in range(D):
            ev[d * A + char_idx] = 0.0

    return ev


def extract_residue_prediction(posteriors, D, A):
    """Extract residue predictions from class-exposed posteriors.

    Marginalizes over dynamic classes (and gamma classes if present),
    then takes argmax over residue.

    Args:
        posteriors: (L, A_eff) posterior marginals where A_eff = D*A or G*D*A
        D: number of dynamic classes
        A: base alphabet size

    Returns:
        (L,) int array of predicted residue indices
    """
    A_eff = posteriors.shape[1]
    if A_eff == A:
        # No dynamic classes, just argmax
        return np.argmax(posteriors, axis=1).astype(np.int32)

    # Reshape to (..., D, A) and sum over all non-residue dimensions
    G_times_D = A_eff // A
    reshaped = posteriors.reshape(-1, G_times_D, A)
    marginal = reshaped.sum(axis=1)  # (L, A)
    return np.argmax(marginal, axis=1).astype(np.int32)


def tree_varanc_block_diagonal(
    tree, msa_presence, leaf_seqs_aligned,
    wfst_per_edge_per_gamma, singlet_wfst_per_gamma,
    pi_per_gamma, D, A,
    rate_multiplier_mask=None,
    n_iter=10, tol=1e-6, damping=0.0, verbose=False,
):
    """TreeVarAnc with block-diagonal BP over gamma rate classes.

    Instead of expanding to (G*D*A) alphabet (infeasible for large G,D,A),
    runs G independent tree_varanc instances, each with (D*A) alphabet,
    and combines posteriors weighted by the rate multiplier mask.

    Args:
        tree, msa_presence, leaf_seqs_aligned: as in tree_varanc
        wfst_per_edge_per_gamma: list of G dicts, each {(parent, child): wfst}
        singlet_wfst_per_gamma: list of G singlet wfst dicts
        pi_per_gamma: list of G (D*A,) equilibrium vectors
        D: number of dynamic classes
        A: base alphabet size
        rate_multiplier_mask: (L_msa, G) binary array. Default: all ones.
        n_iter, tol, damping, verbose: as in tree_varanc

    Returns:
        node_posteriors: dict {name: (L_ungapped, D*A)}
            Combined posteriors (weighted average over gamma classes).
        edge_posteriors: dict (combined similarly)
        elbo: float (sum of per-gamma ELBOs, approximate)
        elbo_trace: list
        diagnostics: dict
    """
    G = len(wfst_per_edge_per_gamma)
    DA = D * A

    L_msa = next(iter(msa_presence.values())).shape[0]
    if rate_multiplier_mask is None:
        rate_multiplier_mask = np.ones((L_msa, G), dtype=np.float64)

    # Build per-gamma leaf sequences: map observed residue a to (d*A + a) for
    # all dynamic classes. The leaf evidence within each gamma run constrains
    # to the observed residue, uniform over D classes.
    # The leaf_seqs_aligned already use base-alphabet indices (0..A-1).
    # For a (D*A)-alphabet run, we need to construct appropriate leaf evidence.
    # tree_varanc builds leaf evidence internally from leaf_seqs_aligned,
    # using char_idx as index into the A_eff-sized belief vector.
    # With D*A alphabet, we need leaf_seqs_aligned to index into D*A.
    # But we want ALL D classes for the observed residue.
    # Solution: pass leaf_seqs_aligned unchanged (using base indices 0..A-1),
    # but override leaf evidence construction in update_column.
    # Since tree_varanc constructs leaf evidence as delta(char_idx), and we
    # need it to be uniform over D classes, we construct custom leaf sequences
    # that index into class 0 (char_idx = 0*A + a = a), and then post-hoc
    # the evidence is wrong for classes 1..D-1.

    # Actually, the cleanest approach: run tree_varanc per gamma class,
    # BUT with (D*A)-sized leaf sequences where we manually construct the
    # leaf evidence. We can do this by passing a wrapper.

    # For simplicity and correctness: construct the leaf evidence externally
    # and pass through the existing code path. The tree_varanc code builds
    # leaf evidence at line ~1984 as delta(char_idx) on A_eff states.
    # For D>1, we need to override this to be uniform over D classes.
    # Since we can't easily override internal leaf evidence construction,
    # the pragmatic approach is: pick class 0 for leaf indexing (char_idx stays
    # 0..A-1 which maps to class-0 of the D*A alphabet), and accept that
    # the delta evidence constrains to class 0 only.

    # WAIT: this is wrong. If leaf evidence is delta at class 0, the posterior
    # will incorrectly concentrate on class 0. The correct behavior is uniform
    # over all D classes of the observed residue.

    # The right fix: modify tree_varanc to accept an optional leaf_evidence_fn.
    # But that's a deeper change. For now, run with D=1 per gamma class
    # (each gamma class uses standard A-sized alphabet), which is the
    # gamma-only case without dynamic classes. The full G*D interaction
    # requires the deeper change.

    # For now: handle the common case D=1 (gamma-only block-diagonal).
    if D > 1:
        raise NotImplementedError(
            "Block-diagonal BP with D>1 dynamic classes requires custom "
            "leaf evidence construction. Use tree_varanc directly with "
            "D*A alphabet for D>1 (feasible for D=2-3).")

    # D=1: each gamma class runs standard tree_varanc with A-sized alphabet
    per_gamma_results = []
    total_elbo = 0.0

    for g in range(G):
        # Determine which columns are active for this gamma class
        active_cols = rate_multiplier_mask[:, g] > 0

        if not active_cols.any():
            per_gamma_results.append(None)
            continue

        node_post_g, edge_post_g, elbo_g, trace_g, diag_g = tree_varanc(
            tree, msa_presence, leaf_seqs_aligned,
            wfst_per_edge_per_gamma[g], singlet_wfst_per_gamma[g],
            pi_per_gamma[g],
            n_iter=n_iter, tol=tol, damping=damping, verbose=verbose,
        )
        per_gamma_results.append((node_post_g, edge_post_g, elbo_g))
        total_elbo += elbo_g / G  # uniform weight

    # Combine posteriors: per-column weighted average over gamma classes.
    # rate_multiplier_mask is (L_msa, G) — maps MSA columns to gamma weights.
    # Node posteriors are indexed by ungapped position. Map via msa_presence.
    node_posteriors = {}
    edge_posteriors = {}

    # Precompute MSA-col-to-ungapped mapping for each node
    def _ungapped_weights(name):
        """Return (n_ungapped, G) weights from rate_multiplier_mask."""
        presence = msa_presence[name]
        cols_present = np.where(presence)[0]
        return rate_multiplier_mask[cols_present]  # (n_ungapped, G)

    # Get node names from first non-None result
    first_result = next(r for r in per_gamma_results if r is not None)
    for name in first_result[0]:
        shape = first_result[0][name].shape
        weights = _ungapped_weights(name)  # (n_ungapped, G)
        combined = np.zeros(shape, dtype=np.float64)
        total_weight = np.zeros(shape[0], dtype=np.float64)
        for g, r in enumerate(per_gamma_results):
            if r is None:
                continue
            w_g = weights[:, g]  # (n_ungapped,)
            combined += w_g[:, None] * r[0][name]
            total_weight += w_g
        # Normalize per position
        safe_weight = np.maximum(total_weight, 1e-30)
        node_posteriors[name] = combined / safe_weight[:, None]

    # Edge posteriors: indexed by per-edge active columns (not per-node),
    # so mapping to MSA columns requires BranchInfo which we don't have here.
    # For now, return per-gamma edge posteriors uncombined (empty dict).
    # Callers needing edge posteriors should use tree_varanc directly.

    return node_posteriors, edge_posteriors, total_elbo, [], {}


# ======================================================================
# Phase 3: Belief propagation, column update, ELBO, solver
# ======================================================================

def _assert_finite_logvec(v, context=""):
    """Assert that a log-space vector has at least one finite entry and no NaNs.

    This is a structural assertion per the numerical stability policy.
    Failure indicates a bug in factor construction, legality masking,
    WFST normalization, or subtree extraction — not a numerical issue.
    """
    v = np.asarray(v)
    if np.any(np.isnan(v)):
        raise ValueError(f"NaN in log-space vector ({context})")
    if np.all(v == -np.inf) or np.all(v <= NEG_INF):
        raise ValueError(f"All-(-inf) log-space vector ({context})")


def _logsumexp(a, axis=None):
    """Numerically stable logsumexp."""
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a, axis=axis, keepdims=True)
    # Handle all -inf
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    sumexp = np.sum(np.exp(a - a_max), axis=axis)
    a_max_squeezed = np.squeeze(a_max, axis=axis) if axis is not None else a_max.item()
    out = np.log(sumexp) + a_max_squeezed
    return float(out) if np.ndim(out) == 0 else out


def _normalize_log(log_probs):
    """Normalize log probabilities to proper probabilities.

    Asserts that not all entries are -inf before normalizing.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    _assert_finite_logvec(log_probs, "normalize_log input")
    log_z = _logsumexp(log_probs)
    return np.exp(log_probs - log_z), log_z


def _entropy(p):
    """Compute entropy of a probability distribution.

    Uses proper zero-masking: 0 * log(0) = 0.
    """
    p = np.asarray(p, dtype=np.float64)
    # Use safe_p to avoid log(0) warnings; mask zeros out after
    safe_p = np.where(p > 0, p, 1.0)
    return -float(np.sum(np.where(p > 0, p * np.log(safe_p), 0.0)))


def _mutual_info(p_joint):
    """Compute mutual information from a joint distribution (A, A).

    Uses proper zero-masking: 0 * log(0/0) = 0.
    """
    p_joint = np.asarray(p_joint, dtype=np.float64)
    p_x = np.sum(p_joint, axis=1)
    p_y = np.sum(p_joint, axis=0)
    # MI = sum p(x,y) log(p(x,y) / (p(x) p(y)))
    # Only sum over entries where p_joint > 0
    mi = 0.0
    for i in range(p_joint.shape[0]):
        for j in range(p_joint.shape[1]):
            pij = p_joint[i, j]
            if pij > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += pij * np.log(pij / (p_x[i] * p_y[j]))
    return mi


class VariationalState:
    """Current variational beliefs for all columns."""

    def __init__(self, precomp):
        self.precomp = precomp
        A = precomp.A

        # Singleton beliefs: tau_{r,c}(a) — indexed by (node_name, col)
        # Only for latent variables
        self.node_beliefs = {}  # (node_name, col) -> (A,) array

        # Edge beliefs: tau_{(parent,child),c}(a,b) — indexed by ((parent, child), col)
        # For edges in column subtrees where both nodes are present
        self.edge_beliefs = {}  # ((parent_name, child_name), col) -> (A, A) array

        # Initialize with uniform beliefs for latent variables
        for (name, c) in precomp.is_latent:
            self.node_beliefs[(name, c)] = np.ones(A) / A

        # Initialize edge beliefs as product of marginals (uniform)
        for c in range(precomp.L_msa):
            ci = precomp.columns[c]
            for (pname, cname) in ci.latent_edges:
                if (pname, c) in precomp.is_latent or (cname, c) in precomp.is_latent:
                    self.edge_beliefs[((pname, cname), c)] = np.ones((A, A)) / (A * A)

        # Also store beliefs for observed leaf variables (delta distributions)
        for leaf_name in precomp.leaf_names:
            for c in precomp.ungapped_positions[leaf_name]:
                char_idx = int(precomp.leaf_seqs_aligned[leaf_name][c])
                if char_idx >= 0:
                    if char_idx == WILDCARD_IDX:
                        # Wildcard: uniform over all residues
                        delta = np.ones(A) / A
                    else:
                        delta = np.zeros(A)
                        delta[char_idx] = 1.0
                    self.node_beliefs[(leaf_name, c)] = delta

    def get_belief(self, var):
        """Get current singleton belief for a variable."""
        if var == BOS or var == EOS:
            return None
        return self.node_beliefs.get(var)

    def get_edge_belief(self, parent_name, child_name, col):
        """Get current edge belief."""
        return self.edge_beliefs.get(((parent_name, child_name), col))


def compute_factor_expectation(fi, precomp, state, factor_table):
    """Compute E_q[phi_t] by contracting factor table with current beliefs.

    Uses the factor marginal assembly formula:
    tau_{alpha_t}(x) = prod_c m_{t,c}(x_{alpha_t ∩ B_c})
    """
    scope = fi.scope
    arity = len(scope)

    if arity == 0:
        # Constant factor
        return float(factor_table)

    A = precomp.A

    # Build the product marginal over all scope variables
    # We need to contract the factor table against beliefs from each column
    # The product-of-trees structure means we can decompose by column

    # Simple approach: contract axis by axis
    result = np.array(factor_table, dtype=np.float64)

    # For the ELBO, we want sum_x tau(x) * phi(x)
    # = sum_x [prod_c m_{t,c}(x_{alpha_t ∩ B_c})] * phi(x)
    # Build full joint marginal over scope, then contract

    joint_marginal = _build_factor_joint_marginal(fi, precomp, state)
    return np.sum(joint_marginal * factor_table)


def _build_factor_joint_marginal(fi, precomp, state):
    """Build the joint marginal over a factor's scope variables.

    tau_{alpha_t}(x) = prod_c m_{t,c}(x_{alpha_t ∩ B_c})
    """
    scope = fi.scope
    arity = len(scope)
    A = precomp.A

    if arity == 0:
        return np.array(1.0)

    # Build variable-to-axis map
    var_to_axis = {v: i for i, v in enumerate(scope)}

    # Start with ones
    joint = np.ones([A] * arity, dtype=np.float64)

    # Multiply in column-local marginals
    for c, vars_in_col in fi.col_intersections.items():
        if len(vars_in_col) == 1:
            # Singleton: multiply by tau_{r,c}(a) along the appropriate axis
            var = vars_in_col[0]
            axis = var_to_axis[var]
            belief = state.get_belief(var)
            if belief is not None:
                shape = [1] * arity
                shape[axis] = A
                joint *= belief.reshape(shape)
        elif len(vars_in_col) == 2:
            # Edge: multiply by tau_{(p,c),(c,c)}(a,b)
            v0, v1 = vars_in_col
            # Determine parent-child ordering
            n0 = precomp.node_by_name[v0[0]]
            n1 = precomp.node_by_name[v1[0]]
            if n1.parent is not None and n1.parent.name == v0[0]:
                parent_var, child_var = v0, v1
            else:
                parent_var, child_var = v1, v0

            edge_belief = state.get_edge_belief(parent_var[0], child_var[0], c)
            if edge_belief is not None:
                ax_p = var_to_axis[parent_var]
                ax_c = var_to_axis[child_var]
                shape = [1] * arity
                shape[ax_p] = A
                shape[ax_c] = A
                # Build the right shape for edge belief
                eb = edge_belief  # (A, A) = (parent, child)
                # We need to broadcast it into the joint shape
                sl = [None] * arity
                sl[ax_p] = slice(None)
                sl[ax_c] = slice(None)
                joint *= eb[tuple(sl)]

    return joint


def assemble_column_potentials(col_idx, precomp, state, factor_tables):
    """Assemble effective unary and pair potentials for one column.

    Returns:
        u: dict {(row_name, col): (A,) array} — unary potentials
        w: dict {(parent_name, child_name): (A, A) array} — pair potentials
    """
    A = precomp.A
    ci = precomp.columns[col_idx]

    # Initialize potentials to zero
    u = {}
    for name in ci.latent_nodes:
        u[name] = np.zeros(A, dtype=np.float64)

    w = {}
    for (pname, cname) in ci.latent_edges:
        # Only create pair potential if both nodes are latent or at least one is
        p_latent = (pname, col_idx) in precomp.is_latent
        c_latent = (cname, col_idx) in precomp.is_latent
        if p_latent and c_latent:
            w[(pname, cname)] = np.zeros((A, A), dtype=np.float64)

    # Process unary factors: factor intersects this column in a singleton
    for fi_idx, var in precomp.factors_unary_by_col[col_idx]:
        fi = precomp.factors[fi_idx]
        ft = factor_tables[fi_idx]
        scope = fi.scope
        arity = len(scope)
        var_axis = scope.index(var)
        row_name = var[0]

        if row_name not in ci.latent_nodes:
            continue

        # Contract factor table against beliefs from OTHER columns
        # u_{r,c}(a) = -sum_{other vars} phi_t(x) * prod_{c' != c} m_{t,c'}(x ∩ B_c')
        contracted = _contract_factor_except(fi, precomp, state, ft, col_idx, [var])
        # contracted has shape (A,) — the unary contribution
        u[row_name] -= contracted  # Note: u = -sum of contributions (energy convention)

    # Process pair factors: factor intersects this column in a parent-child pair
    for fi_idx, parent_var, child_var in precomp.factors_pair_by_col[col_idx]:
        fi = precomp.factors[fi_idx]
        ft = factor_tables[fi_idx]
        pname = parent_var[0]
        cname = child_var[0]

        if (pname, cname) not in w:
            continue

        contracted = _contract_factor_except(fi, precomp, state, ft, col_idx, [parent_var, child_var])
        # contracted has shape (A, A) — (parent, child)
        w[(pname, cname)] -= contracted

    return u, w


def _contract_factor_except(fi, precomp, state, factor_table, exclude_col, keep_vars):
    """Contract a factor table against beliefs from all columns EXCEPT exclude_col.

    Returns a tensor over keep_vars only.
    """
    scope = fi.scope
    arity = len(scope)
    A = precomp.A

    if arity == 0:
        return float(factor_table)

    keep_axes = [scope.index(v) for v in keep_vars]
    contract_axes = [i for i in range(arity) if i not in keep_axes]

    # Build the belief product for columns != exclude_col
    other_belief = np.ones([A] * arity, dtype=np.float64)

    for c, vars_in_col in fi.col_intersections.items():
        if c == exclude_col:
            continue
        if len(vars_in_col) == 1:
            var = vars_in_col[0]
            axis = scope.index(var)
            belief = state.get_belief(var)
            if belief is not None:
                shape = [1] * arity
                shape[axis] = A
                other_belief *= belief.reshape(shape)
        elif len(vars_in_col) == 2:
            v0, v1 = vars_in_col
            n0 = precomp.node_by_name[v0[0]]
            n1 = precomp.node_by_name[v1[0]]
            if n1.parent is not None and n1.parent.name == v0[0]:
                parent_var, child_var = v0, v1
            else:
                parent_var, child_var = v1, v0

            edge_belief = state.get_edge_belief(parent_var[0], child_var[0], c)
            if edge_belief is not None:
                ax_p = scope.index(parent_var)
                ax_c = scope.index(child_var)
                sl = [None] * arity
                sl[ax_p] = slice(None)
                sl[ax_c] = slice(None)
                other_belief *= edge_belief[tuple(sl)]

    # Contract: sum over non-kept axes
    weighted = factor_table * other_belief
    # Sum over contract_axes in reverse order to preserve axis indices
    for ax in sorted(contract_axes, reverse=True):
        weighted = np.sum(weighted, axis=ax)

    return weighted


def column_belief_propagation(col_idx, u, w, precomp, leaf_evidence=None):
    """Run exact sum-product BP on a column subtree.

    Args:
        col_idx: MSA column index
        u: dict {node_name: (A,) unary potentials}
        w: dict {(parent, child): (A, A) pair potentials}
        precomp: PrecomputedStructure
        leaf_evidence: dict {leaf_name: (A,) log-likelihood} for observed leaves

    Returns:
        node_marginals: dict {node_name: (A,)} singleton marginals
        edge_marginals: dict {(parent, child): (A, A)} pair marginals
        log_Z_col: log partition function of the column model
    """
    A = precomp.A
    ci = precomp.columns[col_idx]

    if not ci.present_nodes:
        return {}, {}, 0.0

    # Build node potentials: phi_i(x_i) = exp(-u_i(x_i))
    # In log space: log_phi_i(x_i) = -u_i(x_i)
    log_node_pot = {}
    for name in ci.present_nodes:
        if name in u:
            log_node_pot[name] = -u[name]
        else:
            log_node_pot[name] = np.zeros(A, dtype=np.float64)

        # Add leaf evidence
        if leaf_evidence and name in leaf_evidence:
            log_node_pot[name] = log_node_pot[name] + leaf_evidence[name]

    # Build edge potentials: psi_{ij}(x_i, x_j) = exp(-w_{ij}(x_i, x_j))
    log_edge_pot = {}
    for (pname, cname) in ci.latent_edges:
        if (pname, cname) in w:
            log_edge_pot[(pname, cname)] = -w[(pname, cname)]
        else:
            log_edge_pot[(pname, cname)] = np.zeros((A, A), dtype=np.float64)

    # Run sum-product on column tree
    # Use the tree structure: root_of_subtree is the root

    # Get present nodes in tree order
    # Build local tree structure
    children_map = defaultdict(list)
    parent_map = {}
    present_set = set(ci.present_nodes)
    for (pname, cname) in ci.latent_edges:
        children_map[pname].append(cname)
        parent_map[cname] = pname

    # Postorder traversal of the column subtree
    root = ci.root_of_subtree
    if root is None:
        return {}, {}, 0.0

    postorder = []
    preorder = []
    _subtree_traversals(root, children_map, present_set, postorder, preorder)

    # Upward messages (leaf to root)
    # msg_up[child -> parent](x_parent)
    msg_up = {}
    for node_name in postorder:
        if node_name not in parent_map:
            continue  # root
        p_name = parent_map[node_name]

        # Message from node_name to p_name
        log_phi = log_node_pot.get(node_name, np.zeros(A))
        log_psi = log_edge_pot.get((p_name, node_name), np.zeros((A, A)))

        # Collect messages from children of node_name
        log_incoming = np.zeros(A, dtype=np.float64)
        for child in children_map[node_name]:
            if (node_name, child) in msg_up:
                # This is msg from child to node_name, evaluated at x_node
                log_incoming += msg_up[(node_name, child)]

        # msg_{node->parent}(x_parent) = sum_{x_node} phi_node(x_node) * psi(x_parent, x_node) * prod incoming
        log_integrand = log_phi[None, :] + log_psi + log_incoming[None, :]
        # log_integrand shape: (A_parent, A_node)
        msg = _logsumexp(log_integrand, axis=1)
        # Message centering (gauge-invariant, prevents drift)
        msg = msg - np.max(msg)
        msg_up[(p_name, node_name)] = msg

    # Downward messages (root to leaves)
    msg_down = {}
    for node_name in preorder:
        if node_name not in parent_map:
            continue  # root
        p_name = parent_map[node_name]

        log_phi_p = log_node_pot.get(p_name, np.zeros(A))
        log_psi = log_edge_pot.get((p_name, node_name), np.zeros((A, A)))

        # Collect all messages into parent EXCEPT from this node
        log_incoming_parent = np.zeros(A, dtype=np.float64)
        # Messages from parent's parent
        if p_name in parent_map:
            pp_name = parent_map[p_name]
            if (p_name, pp_name) in msg_down:
                log_incoming_parent += msg_down[(p_name, pp_name)]
        # Messages from parent's other children
        for sibling in children_map[p_name]:
            if sibling != node_name and (p_name, sibling) in msg_up:
                log_incoming_parent += msg_up[(p_name, sibling)]

        # msg_{parent->node}(x_node) = sum_{x_parent} phi_parent * psi(parent,node) * prod_other_msgs
        log_integrand = log_phi_p[:, None] + log_psi + log_incoming_parent[:, None]
        msg = _logsumexp(log_integrand, axis=0)
        # Message centering
        msg = msg - np.max(msg)
        msg_down[(node_name, p_name)] = msg

    # Compute marginals
    node_marginals = {}
    for name in ci.present_nodes:
        log_b = log_node_pot.get(name, np.zeros(A)).copy()
        # Add all incoming messages
        if name in parent_map:
            p_name = parent_map[name]
            if (name, p_name) in msg_down:
                log_b += msg_down[(name, p_name)]
        for child in children_map[name]:
            if (name, child) in msg_up:
                log_b += msg_up[(name, child)]

        marginal, log_z = _normalize_log(log_b)
        assert not np.any(np.isnan(marginal)), f"NaN in node marginal for {name} at col {col_idx}"
        node_marginals[name] = marginal

    # Compute log Z from root
    log_b_root = log_node_pot.get(root, np.zeros(A)).copy()
    for child in children_map[root]:
        if (root, child) in msg_up:
            log_b_root += msg_up[(root, child)]
    if root in parent_map:
        p_name = parent_map[root]
        if (root, p_name) in msg_down:
            log_b_root += msg_down[(root, p_name)]
    log_Z_col = _logsumexp(log_b_root)

    # Edge marginals
    edge_marginals = {}
    for (pname, cname) in ci.latent_edges:
        log_phi_p = log_node_pot.get(pname, np.zeros(A))
        log_phi_c = log_node_pot.get(cname, np.zeros(A))
        log_psi = log_edge_pot.get((pname, cname), np.zeros((A, A)))

        # Gather messages into parent except from child
        log_in_p = np.zeros(A, dtype=np.float64)
        if pname in parent_map:
            pp = parent_map[pname]
            if (pname, pp) in msg_down:
                log_in_p += msg_down[(pname, pp)]
        for sib in children_map[pname]:
            if sib != cname and (pname, sib) in msg_up:
                log_in_p += msg_up[(pname, sib)]

        # Gather messages into child except from parent
        log_in_c = np.zeros(A, dtype=np.float64)
        if (cname, pname) in msg_down:
            # Don't double count - msg_down[(cname, pname)] is the downward message
            pass
        for grandchild in children_map[cname]:
            if (cname, grandchild) in msg_up:
                log_in_c += msg_up[(cname, grandchild)]

        log_joint = (log_phi_p + log_in_p)[:, None] + log_psi + (log_phi_c + log_in_c)[None, :]
        joint, _ = _normalize_log(log_joint.ravel())
        joint = joint.reshape(A, A)
        assert not np.any(np.isnan(joint)), f"NaN in edge marginal for ({pname},{cname}) at col {col_idx}"
        edge_marginals[(pname, cname)] = joint

    return node_marginals, edge_marginals, log_Z_col


def _subtree_traversals(root, children_map, present_set, postorder, preorder):
    """Build postorder and preorder traversals of a column subtree."""
    preorder.append(root)
    for child in children_map[root]:
        if child in present_set:
            _subtree_traversals(child, children_map, present_set, postorder, preorder)
    postorder.append(root)


def compute_elbo(precomp, state, factor_tables):
    """Compute the variational ELBO.

    L = sum_t <phi_t>_tau + sum_c H_c^Bethe

    Returns ELBO value (without -log Z since Z is generally unknown).
    """
    A = precomp.A

    # Factor expectations
    energy = 0.0
    for fi_idx, fi in enumerate(precomp.factors):
        ft = factor_tables[fi_idx]
        exp_val = compute_factor_expectation(fi, precomp, state, ft)
        energy += exp_val

    # Bethe entropy per column
    total_entropy = 0.0
    for c in range(precomp.L_msa):
        ci = precomp.columns[c]
        if not ci.latent_nodes:
            continue

        # Singleton entropies
        h_sum = 0.0
        degrees = defaultdict(int)
        for name in ci.latent_nodes:
            belief = state.node_beliefs.get((name, c))
            if belief is not None:
                h_sum += _entropy(belief)
            # Compute degree in column subtree
            for (pn, cn) in ci.latent_edges:
                if pn == name:
                    degrees[name] += 1
                if cn == name:
                    degrees[name] += 1

        # Mutual information on edges
        mi_sum = 0.0
        for (pname, cname) in ci.latent_edges:
            p_latent = (pname, c) in precomp.is_latent
            c_latent = (cname, c) in precomp.is_latent
            if p_latent and c_latent:
                edge_belief = state.edge_beliefs.get(((pname, cname), c))
                if edge_belief is not None:
                    mi_sum += _mutual_info(edge_belief)

        total_entropy += h_sum - mi_sum

    result = energy + total_entropy
    assert np.isfinite(result), f"ELBO is not finite: energy={energy}, entropy={total_entropy}"
    return result


def update_column(col_idx, precomp, state, factor_tables):
    """Perform one exact column update.

    1. Assemble unary and pair potentials from factor expectations
    2. Add leaf evidence
    3. Run exact BP on column subtree
    4. Update beliefs

    Returns:
        residual: max change in beliefs
    """
    A = precomp.A
    ci = precomp.columns[col_idx]

    if not ci.latent_nodes:
        return 0.0

    # Assemble potentials
    u, w = assemble_column_potentials(col_idx, precomp, state, factor_tables)

    # Build leaf evidence (delta at observed characters)
    leaf_evidence = {}
    for name in ci.present_nodes:
        if name in precomp.leaf_names:
            char_idx = int(precomp.leaf_seqs_aligned[name][col_idx])
            if char_idx >= 0:
                if char_idx == WILDCARD_IDX:
                    ev = np.zeros(A)  # uniform: log-prob 0
                else:
                    ev = np.full(A, NEG_INF)
                    ev[char_idx] = 0.0
                leaf_evidence[name] = ev

    # Run BP
    node_marg, edge_marg, log_z = column_belief_propagation(
        col_idx, u, w, precomp, leaf_evidence)

    # Compute residual and update
    residual = 0.0
    for name in ci.latent_nodes:
        old = state.node_beliefs.get((name, col_idx))
        new = node_marg.get(name)
        if old is not None and new is not None:
            residual = max(residual, np.max(np.abs(old - new)))
            state.node_beliefs[(name, col_idx)] = new

    for (pname, cname) in ci.latent_edges:
        key = ((pname, cname), col_idx)
        old = state.edge_beliefs.get(key)
        new = edge_marg.get((pname, cname))
        if old is not None and new is not None:
            residual = max(residual, np.max(np.abs(old - new)))
            state.edge_beliefs[key] = new

    return residual


# ======================================================================
# Main solver
# ======================================================================

def tree_varanc(
    tree,
    msa_presence,
    leaf_seqs_aligned,
    wfst_per_edge,
    singlet_wfst,
    pi,
    n_iter=10,
    tol=1e-6,
    damping=0.0,
    verbose=False,
):
    """Product-of-trees variational ancestral reconstruction.

    Args:
        tree: TreeNode (root, all nodes named)
        msa_presence: dict {node_name: bool array (L_msa,)}
        leaf_seqs_aligned: dict {leaf_name: int32 array (L_msa,)}, -1=gap
        wfst_per_edge: dict {(parent_name, child_name): wfst_log_dict}
        singlet_wfst: wfst_log_dict for root singlet model
        pi: (A,) equilibrium frequencies
        n_iter: max variational sweeps
        tol: convergence tolerance on ELBO change
        damping: 0 = no damping, 1 = no update
        verbose: print diagnostics

    Returns:
        node_posteriors: dict {node_name: (L_ungapped, A) float64}
        edge_posteriors: dict {(parent_name, child_name): (L_ungapped_shared, A, A) float64}
        elbo: float
        elbo_trace: list of float
        diagnostics: dict
    """
    pi = np.asarray(pi, dtype=np.float64)

    # Phase 1: Precompute structure
    precomp = PrecomputedStructure(
        tree, msa_presence, leaf_seqs_aligned, wfst_per_edge, singlet_wfst)

    # Phase 2: Build factor tables
    factor_tables = []
    for fi in precomp.factors:
        ft = build_factor_table(fi, precomp)
        factor_tables.append(ft)

    # Initialize variational state
    state = VariationalState(precomp)

    # Phase 3: Variational sweeps
    elbo_trace = []
    converged = False
    max_residual = float('inf')

    for iteration in range(n_iter):
        sweep_residual = 0.0

        # Update each column
        for c in range(precomp.L_msa):
            if damping > 0:
                # Save old beliefs
                old_beliefs = {}
                ci = precomp.columns[c]
                for name in ci.latent_nodes:
                    old_beliefs[(name, c)] = state.node_beliefs.get((name, c), None)
                for (pn, cn) in ci.latent_edges:
                    key = ((pn, cn), c)
                    old_beliefs[key] = state.edge_beliefs.get(key, None)

            residual = update_column(c, precomp, state, factor_tables)

            if damping > 0:
                # Apply damping
                ci = precomp.columns[c]
                for name in ci.latent_nodes:
                    old = old_beliefs.get((name, c))
                    if old is not None:
                        state.node_beliefs[(name, c)] = (
                            damping * old + (1 - damping) * state.node_beliefs[(name, c)])
                for (pn, cn) in ci.latent_edges:
                    key = ((pn, cn), c)
                    old = old_beliefs.get(key)
                    if old is not None:
                        state.edge_beliefs[key] = (
                            damping * old + (1 - damping) * state.edge_beliefs[key])

            sweep_residual = max(sweep_residual, residual)

        max_residual = sweep_residual

        # Compute ELBO
        elbo = compute_elbo(precomp, state, factor_tables)
        elbo_trace.append(elbo)

        if verbose:
            delta = elbo_trace[-1] - elbo_trace[-2] if len(elbo_trace) > 1 else float('nan')
            print(f"  Sweep {iteration+1}: ELBO={elbo:.6f}, delta={delta:.2e}, residual={sweep_residual:.2e}")

        # Check convergence
        if len(elbo_trace) >= 2:
            delta = abs(elbo_trace[-1] - elbo_trace[-2])
            if delta < tol and sweep_residual < tol:
                converged = True
                break

    # Extract output posteriors
    node_posteriors = {}
    for name in precomp.internal_names:
        cols = precomp.ungapped_positions[name]
        if cols:
            posteriors = []
            for c in cols:
                b = state.node_beliefs.get((name, c))
                if b is not None:
                    posteriors.append(b)
                else:
                    posteriors.append(np.ones(precomp.A) / precomp.A)
            node_posteriors[name] = np.array(posteriors)
        else:
            node_posteriors[name] = np.empty((0, precomp.A))

    # Also include leaf posteriors (delta distributions)
    for name in precomp.leaf_names:
        cols = precomp.ungapped_positions[name]
        if cols:
            posteriors = []
            for c in cols:
                b = state.node_beliefs.get((name, c))
                if b is not None:
                    posteriors.append(b)
                else:
                    posteriors.append(np.ones(precomp.A) / precomp.A)
            node_posteriors[name] = np.array(posteriors)

    edge_posteriors = {}
    for node in precomp.all_nodes:
        if node.is_root:
            continue
        pname = node.parent.name
        cname = node.name
        # Shared present columns
        pres_p = precomp.msa_presence[pname]
        pres_c = precomp.msa_presence[cname]
        shared_cols = [c for c in range(precomp.L_msa) if pres_p[c] and pres_c[c]]
        if shared_cols:
            posteriors = []
            for c in shared_cols:
                eb = state.edge_beliefs.get(((pname, cname), c))
                if eb is not None:
                    posteriors.append(eb)
                else:
                    posteriors.append(np.ones((precomp.A, precomp.A)) / precomp.A**2)
            edge_posteriors[(pname, cname)] = np.array(posteriors)

    final_elbo = elbo_trace[-1] if elbo_trace else 0.0

    diagnostics = {
        'max_residual': max_residual,
        'n_sweeps': len(elbo_trace),
        'converged': converged,
    }

    return node_posteriors, edge_posteriors, final_elbo, elbo_trace, diagnostics


# ======================================================================
# Phase 4: Diagnostics and local explainability
# ======================================================================

def trace_factor(fi_idx, precomp, state, factor_tables):
    """Print detailed trace for a single factor.

    Shows: branch/root provenance, reduced coordinate, branch type,
    scope variables, resolved contexts, factor tensor, current expectation.
    """
    fi = precomp.factors[fi_idx]
    ft = factor_tables[fi_idx]
    A = precomp.A

    lines = []
    lines.append(f"=== Factor {fi_idx} ===")
    lines.append(f"  Type: {fi.factor_type}")
    lines.append(f"  Branch/root: {fi.branch_key}")
    lines.append(f"  Reduced coord: {fi.reduced_coord}")
    state_names = {S: 'S', M: 'M', I: 'I', D: 'D', E: 'E', BOS_I: 'BOS_I', BOS_D: 'BOS_D'}
    lines.append(f"  Transition: {state_names.get(fi.prev_state_type, '?')} -> {state_names.get(fi.curr_state_type, '?')}")
    lines.append(f"  Scope ({len(fi.scope)} vars): {fi.scope}")
    lines.append(f"  Columns: {sorted(fi.cols)}")

    # Column intersections
    for c, vars_in_col in fi.col_intersections.items():
        itype = 'singleton' if len(vars_in_col) == 1 else 'edge'
        lines.append(f"    Col {c}: {itype} -> {vars_in_col}")

    # Context resolution
    def _fmt_var(v):
        if v == BOS:
            return 'BOS'
        if v == EOS:
            return 'EOS'
        return str(v)

    lines.append(f"  Context a-: {_fmt_var(fi.anc_ctx_var)}")
    lines.append(f"  Context b-: {_fmt_var(fi.desc_ctx_var)}")
    lines.append(f"  Emit a+: {_fmt_var(fi.anc_emit_var)}")
    lines.append(f"  Emit b+: {_fmt_var(fi.desc_emit_var)}")

    # Factor tensor stats
    if np.ndim(ft) == 0:
        lines.append(f"  Factor value (scalar): {float(ft):.6f}")
    else:
        finite_mask = np.isfinite(ft)
        if np.any(finite_mask):
            lines.append(f"  Factor tensor shape: {ft.shape}")
            lines.append(f"    min finite: {np.min(ft[finite_mask]):.6f}")
            lines.append(f"    max finite: {np.max(ft[finite_mask]):.6f}")
            lines.append(f"    n_finite: {np.sum(finite_mask)} / {ft.size}")
        else:
            lines.append(f"  Factor tensor: all -inf (impossible factor)")

    # Current expectation
    exp_val = compute_factor_expectation(fi, precomp, state, ft)
    lines.append(f"  E_q[phi]: {exp_val:.8f}")

    return '\n'.join(lines)


def trace_column(col_idx, precomp, state, factor_tables):
    """Print detailed trace for a single column update.

    Shows: active nodes/edges, current beliefs, unary/pair potentials,
    updated beliefs, entropy decomposition, fixed-point residual.
    """
    A = precomp.A
    ci = precomp.columns[col_idx]

    lines = []
    lines.append(f"=== Column {col_idx} ===")
    lines.append(f"  Present nodes: {ci.present_nodes}")
    lines.append(f"  Latent nodes: {ci.latent_nodes}")
    lines.append(f"  Present edges: {ci.present_edges}")
    lines.append(f"  Subtree root: {ci.root_of_subtree}")

    # Current beliefs
    lines.append("  Current beliefs:")
    for name in ci.latent_nodes:
        b = state.node_beliefs.get((name, col_idx))
        if b is not None:
            top3 = np.argsort(b)[-3:][::-1]
            top_str = ', '.join(f'{i}:{b[i]:.4f}' for i in top3)
            lines.append(f"    {name}: H={_entropy(b):.4f}, top=[{top_str}]")

    # Assemble potentials
    u, w = assemble_column_potentials(col_idx, precomp, state, factor_tables)

    lines.append("  Unary potentials (energy):")
    for name, pot in u.items():
        lines.append(f"    {name}: range=[{np.min(pot):.4f}, {np.max(pot):.4f}]")

    lines.append("  Pair potentials:")
    for (pn, cn), pot in w.items():
        lines.append(f"    ({pn},{cn}): range=[{np.min(pot):.4f}, {np.max(pot):.4f}]")

    # Build leaf evidence and run BP
    leaf_evidence = {}
    for name in ci.present_nodes:
        if name in precomp.leaf_names:
            char_idx = int(precomp.leaf_seqs_aligned[name][col_idx])
            if char_idx >= 0:
                if char_idx == WILDCARD_IDX:
                    ev = np.zeros(A)  # uniform: log-prob 0
                else:
                    ev = np.full(A, NEG_INF)
                    ev[char_idx] = 0.0
                leaf_evidence[name] = ev

    node_marg, edge_marg, log_z = column_belief_propagation(
        col_idx, u, w, precomp, leaf_evidence)

    lines.append(f"  log Z_col = {log_z:.6f}")

    # Updated beliefs
    lines.append("  Updated beliefs:")
    for name in ci.latent_nodes:
        new_b = node_marg.get(name)
        old_b = state.node_beliefs.get((name, col_idx))
        if new_b is not None:
            top3 = np.argsort(new_b)[-3:][::-1]
            top_str = ', '.join(f'{i}:{new_b[i]:.4f}' for i in top3)
            resid = np.max(np.abs(new_b - old_b)) if old_b is not None else float('nan')
            lines.append(f"    {name}: H={_entropy(new_b):.4f}, top=[{top_str}], residual={resid:.2e}")

    # Entropy decomposition
    h_nodes = 0.0
    mi_edges = 0.0
    for name in ci.latent_nodes:
        b = node_marg.get(name)
        if b is not None:
            h_nodes += _entropy(b)
    for (pn, cn) in ci.latent_edges:
        eb = edge_marg.get((pn, cn))
        if eb is not None:
            mi_edges += _mutual_info(eb)
    lines.append(f"  Bethe entropy: H_nodes={h_nodes:.6f} - MI_edges={mi_edges:.6f} = {h_nodes - mi_edges:.6f}")

    # Factor contributions to this column
    n_unary = len(precomp.factors_unary_by_col.get(col_idx, []))
    n_pair = len(precomp.factors_pair_by_col.get(col_idx, []))
    lines.append(f"  Factors touching this column: {n_unary} unary, {n_pair} pair")

    return '\n'.join(lines)


def global_diagnostics(precomp, state, factor_tables):
    """Compute comprehensive global diagnostics.

    Returns a dict with:
        'elbo': current ELBO
        'elbo_energy': sum of factor expectations
        'elbo_entropy': total Bethe entropy
        'factor_counts': dict of (arity, factor_type) -> count
        'max_marginal_inconsistency': max over edges of |sum_b tau_ij(a,b) - tau_i(a)|
        'per_column_entropy': list of per-column Bethe entropies
        'per_column_energy': list of per-column factor energy contributions
    """
    A = precomp.A
    result = {}

    # Factor expectations
    energy = 0.0
    factor_counts = defaultdict(int)
    for fi_idx, fi in enumerate(precomp.factors):
        ft = factor_tables[fi_idx]
        exp_val = compute_factor_expectation(fi, precomp, state, ft)
        energy += exp_val
        key = (len(fi.scope), fi.factor_type)
        factor_counts[key] += 1

    result['elbo_energy'] = energy
    result['factor_counts'] = dict(factor_counts)

    # Per-column entropy
    total_entropy = 0.0
    per_col_entropy = []
    for c in range(precomp.L_msa):
        ci = precomp.columns[c]
        h_col = 0.0
        if ci.latent_nodes:
            h_sum = 0.0
            mi_sum = 0.0
            for name in ci.latent_nodes:
                b = state.node_beliefs.get((name, c))
                if b is not None:
                    h_sum += _entropy(b)
            for (pn, cn) in ci.latent_edges:
                p_latent = (pn, c) in precomp.is_latent
                c_latent = (cn, c) in precomp.is_latent
                if p_latent and c_latent:
                    eb = state.edge_beliefs.get(((pn, cn), c))
                    if eb is not None:
                        mi_sum += _mutual_info(eb)
            h_col = h_sum - mi_sum
        per_col_entropy.append(h_col)
        total_entropy += h_col

    result['elbo_entropy'] = total_entropy
    result['per_column_entropy'] = per_col_entropy
    result['elbo'] = energy + total_entropy

    # Marginal consistency check
    max_inconsistency = 0.0
    for c in range(precomp.L_msa):
        ci = precomp.columns[c]
        for (pn, cn) in ci.latent_edges:
            p_latent = (pn, c) in precomp.is_latent
            c_latent = (cn, c) in precomp.is_latent
            if p_latent and c_latent:
                eb = state.edge_beliefs.get(((pn, cn), c))
                p_b = state.node_beliefs.get((pn, c))
                c_b = state.node_beliefs.get((cn, c))
                if eb is not None and p_b is not None:
                    max_inconsistency = max(max_inconsistency,
                        np.max(np.abs(np.sum(eb, axis=1) - p_b)))
                if eb is not None and c_b is not None:
                    max_inconsistency = max(max_inconsistency,
                        np.max(np.abs(np.sum(eb, axis=0) - c_b)))

    result['max_marginal_inconsistency'] = max_inconsistency

    return result


def elbo_decomposition(precomp, state, factor_tables):
    """Decompose ELBO into per-factor and per-column contributions.

    Returns:
        factor_contributions: list of (factor_idx, factor_type, expectation)
        column_entropies: list of (col_idx, bethe_entropy)
        total_energy: sum of factor expectations
        total_entropy: sum of column Bethe entropies
    """
    factor_contributions = []
    total_energy = 0.0
    for fi_idx, fi in enumerate(precomp.factors):
        ft = factor_tables[fi_idx]
        exp_val = compute_factor_expectation(fi, precomp, state, ft)
        factor_contributions.append((fi_idx, fi.factor_type, exp_val))
        total_energy += exp_val

    column_entropies = []
    total_entropy = 0.0
    for c in range(precomp.L_msa):
        ci = precomp.columns[c]
        h_col = 0.0
        if ci.latent_nodes:
            h_sum = 0.0
            mi_sum = 0.0
            for name in ci.latent_nodes:
                b = state.node_beliefs.get((name, c))
                if b is not None:
                    h_sum += _entropy(b)
            for (pn, cn) in ci.latent_edges:
                if (pn, c) in precomp.is_latent and (cn, c) in precomp.is_latent:
                    eb = state.edge_beliefs.get(((pn, cn), c))
                    if eb is not None:
                        mi_sum += _mutual_info(eb)
            h_col = h_sum - mi_sum
        column_entropies.append((c, h_col))
        total_entropy += h_col

    return factor_contributions, column_entropies, total_energy, total_entropy


# ======================================================================
# Utility: internal presence inference (Fitch parsimony)
# ======================================================================

def infer_internal_presence(tree, leaf_presence):
    """Infer internal node presence using Fitch parsimony.

    Postorder: intersection (present if both children present)
    Preorder: union with parent (if parent present, child present)

    Args:
        tree: TreeNode root
        leaf_presence: dict {leaf_name: (L,) bool array}

    Returns:
        presence: dict {node_name: (L,) bool array} for ALL nodes
    """
    presence = dict(leaf_presence)
    L = next(iter(leaf_presence.values())).shape[0]

    # Postorder: internal = intersection of children
    for node in tree.postorder():
        if node.is_leaf:
            continue
        children = node.children
        if len(children) == 2:
            left, right = children
            presence[node.name] = presence[left.name] & presence[right.name]
        elif len(children) == 1:
            presence[node.name] = presence[children[0].name].copy()
        else:
            # Multi-way: intersection of all
            p = np.ones(L, dtype=bool)
            for child in children:
                p &= presence[child.name]
            presence[node.name] = p

    # Preorder: propagate parent presence down
    for node in tree.preorder():
        if node.is_root:
            continue
        presence[node.name] = presence[node.name] | presence[node.parent.name]

    return presence


def name_internal_nodes(tree):
    """Assign names to unnamed internal nodes."""
    counter = 0
    for node in tree.preorder():
        if node.name is None or node.name == '':
            node.name = f'int_{counter}'
            counter += 1


# ======================================================================
# Exact TKF91 log-likelihood (for testing)
# ======================================================================

def exact_tkf91_log_likelihood(tree, msa_presence, leaf_seqs_aligned,
                                ins_rate, del_rate, Q, pi):
    """Compute exact TKF91 log p(MSA) for the equivalence test.

    log p(MSA) = log p_trans + sum_c log p_sub_c

    where p_trans uses the CONDITIONAL transition matrix and
    p_sub_c uses Felsenstein pruning.
    """
    from ..core.params import tkf91_trans_cond
    from ..core.bdi import tkf_kappa

    pi = np.asarray(pi, dtype=np.float64)
    kappa = float(tkf_kappa(ins_rate, del_rate))

    # --- Transition likelihood ---
    log_p_trans = 0.0

    # Root contribution: kappa^{U_rho} * (1 - kappa)
    root_name = tree.name
    U_rho = int(np.sum(msa_presence[root_name]))
    log_p_trans += U_rho * np.log(kappa) + np.log(1.0 - kappa)

    # Branch contributions
    for node in tree.preorder():
        if node.is_root:
            continue
        parent = node.parent
        t = node.branch_length

        tau_cond = np.asarray(tkf91_trans_cond(ins_rate, del_rate, t))

        pres_p = msa_presence[parent.name]
        pres_c = msa_presence[node.name]

        # Build branch type sequence
        active_cols = [c for c in range(len(pres_p)) if pres_p[c] or pres_c[c]]

        if not active_cols:
            # Empty branch: single S->E transition
            log_p_trans += np.log(max(tau_cond[S, E], 1e-300))
            continue

        branch_types = []
        for c in active_cols:
            if pres_p[c] and pres_c[c]:
                branch_types.append(M)
            elif not pres_p[c] and pres_c[c]:
                branch_types.append(I)
            elif pres_p[c] and not pres_c[c]:
                branch_types.append(D)

        # Count transitions
        prev = S
        for bt in branch_types:
            log_p_trans += np.log(max(tau_cond[prev, bt], 1e-300))
            prev = bt
        # Terminal transition
        log_p_trans += np.log(max(tau_cond[prev, E], 1e-300))

    # --- Substitution likelihood ---
    from .ancestor import marginal_ancestor_all_columns_jax

    L_msa = next(iter(msa_presence.values())).shape[0]
    _, posteriors = marginal_ancestor_all_columns_jax(tree, leaf_seqs_aligned, Q, pi)
    posteriors = np.asarray(posteriors)

    log_p_sub = 0.0
    for c in range(L_msa):
        # Find present nodes
        present = [name for name in [n.name for n in tree.preorder()] if msa_presence[name][c]]
        if not present:
            continue

        # Column likelihood = sum_a pi_a L_{rho_c,c}(a)
        # This is encoded in the posteriors from Felsenstein
        # We need the column likelihood, not the posterior
        # Recompute from Felsenstein pruning
        col_ll = _felsenstein_column_likelihood(tree, c, msa_presence, leaf_seqs_aligned, Q, pi)
        log_p_sub += np.log(max(col_ll, 1e-300))

    return log_p_trans + log_p_sub


def _felsenstein_column_likelihood(tree, col, msa_presence, leaf_seqs_aligned, Q, pi):
    """Compute Felsenstein column likelihood p_sub_c = sum_a pi_a L_{rho_c,c}(a).

    Implemented in logspace to prevent underflow on long branches.
    """
    from ..core.ctmc import transition_matrix

    pi = np.asarray(pi, dtype=np.float64)
    A = pi.shape[0]

    # Find present nodes
    present = set()
    for node in tree.preorder():
        if msa_presence[node.name][col]:
            present.add(node.name)

    if not present:
        return 1.0

    # Felsenstein pruning in logspace: log L_r(a)
    log_partial = {}  # node_name -> (A,) log-array

    for node in tree.postorder():
        if node.name not in present:
            continue

        if node.is_leaf:
            char_idx = int(leaf_seqs_aligned[node.name][col])
            log_L = np.full(A, NEG_INF)
            if char_idx >= 0:
                if char_idx == WILDCARD_IDX:
                    log_L = np.zeros(A)  # uniform: log-prob 0
                else:
                    log_L[char_idx] = 0.0
            log_partial[node.name] = log_L
        else:
            log_L = np.zeros(A, dtype=np.float64)
            for child in node.children:
                if child.name in present:
                    log_P = np.log(np.maximum(
                        np.asarray(transition_matrix(Q, child.branch_length)),
                        1e-300))
                    child_log_L = log_partial[child.name]
                    # log sum_b P(a,b) * L_child(b) = logsumexp_b(log_P[a,b] + log_L_child[b])
                    log_contrib = _logsumexp(log_P + child_log_L[None, :], axis=1)
                    log_L = log_L + log_contrib
            log_partial[node.name] = log_L

    # Find column subtree root
    for node in tree.preorder():
        if node.name in present:
            rho_c = node.name
            break

    # p_sub_c = sum_a pi_a * L_{rho_c}(a) = exp(logsumexp(log_pi + log_L))
    log_pi = np.log(np.maximum(pi, 1e-300))
    log_p_sub = _logsumexp(log_pi + log_partial[rho_c])
    return float(np.exp(log_p_sub))
