"""Partition-conditioned ancestral reconstruction for MixDom with vanishing top-level indels.

WARNING: This is the REFERENCE (plain Python/NumPy) implementation.
It uses nested Python loops and is too slow for production use.
For real workloads, use the JAX-vectorized version in
``partition_recon_jax.py`` which uses jax.lax.scan and jax.vmap.

Implements the algorithm specified in tkf/partition-recon.tex.

Summary
-------

For a restricted MixDom model where the top-level TKF91 rates
lambda_0, mu_0 tend to zero at fixed ratio kappa_0 = lambda_0/mu_0, and
where each domain has a single fragment class (n_frag = 1), every
top-level domain at the root is preserved (no births/deaths) on every
branch. The MSA therefore factorises into a contiguous partition of
columns into blocks, each block labelled with a domain class, and each
block's sub-MSA evolving independently as a standalone TKF92.

This module computes, via a partition-conditioned Forward-Backward,
(i) the total likelihood log P(MSA | tree, model),
(ii) the per-column posterior over domain class, and
(iii) the marginal posterior over root residues per column (obtained
by mixing per-class Felsenstein posteriors with the per-column class
posterior).

The algorithm requires as input a per-node gap/residue annotation
(typically from Fitch parsimony on gaps); it does not reconstruct
gap patterns itself.

Inputs are MSA columns annotated with presence/absence at every tree
node (leaves observed, internal nodes via Fitch). Outputs include
per-column domain class posteriors and per-column root residue
posteriors (gaps at columns where the root is absent).

This file provides the reference Python implementation. A fully
vectorised JAX version lives in `partition_recon_jax.py`. The two are
expected to agree to machine precision on small inputs.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Sequence

from ..core.params import tkf92_trans, tkf91_trans_cond, tkf91_trans, S, M, I, D, E
from ..core.bdi import tkf_kappa
from ..core.ctmc import transition_matrix
from ..util.io import TreeNode
from .tree_varanc import infer_internal_presence, name_internal_nodes


NEG_INF = -1e30


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _logsumexp_np(x, axis=None):
    x = np.asarray(x)
    if axis is None:
        m = np.max(x)
        if not np.isfinite(m):
            return float(-np.inf)
        return float(m + np.log(np.sum(np.exp(x - m))))
    m = np.max(x, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    out = np.squeeze(m_safe, axis=axis) + np.log(
        np.sum(np.exp(x - m_safe), axis=axis)
    )
    # Propagate -inf where max was -inf (column all -inf).
    mask = ~np.isfinite(np.squeeze(m, axis=axis))
    out = np.where(mask, -np.inf, out)
    return out


def _safe_log(x, eps=1e-300):
    x = np.asarray(x)
    return np.where(x > 0, np.log(np.maximum(x, eps)), NEG_INF)



# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PartitionReconInputs:
    """Structural inputs for the reconstruction algorithm.

    All per-node arrays are keyed by node name.

    Args:
        tree: TreeNode (root). Internal nodes must be named
            (call util.name_internal_nodes first if needed).
        leaf_seqs_aligned: dict {leaf_name: (L,) int array, -1 = gap}
        presence: dict {node_name: (L,) bool array}. If None, will be
            computed from leaf_seqs_aligned via Fitch parsimony on gaps
            in `build_inputs`.
    """
    tree: TreeNode
    leaf_seqs_aligned: Dict[str, np.ndarray]
    presence: Dict[str, np.ndarray]

    @property
    def L(self) -> int:
        return int(next(iter(self.leaf_seqs_aligned.values())).shape[0])


@dataclass
class PartitionReconModel:
    """Restricted MixDom model parameters.

    The vanishing top-level rate limit is already applied: only the
    ratio kappa_top = lambda_0 / mu_0 matters at the top level.

    Args:
        kappa_top: top-level rate ratio in (0, 1).
        dom_weights: (N,) top-level class weights (sum to 1).
        dom_ins_rates: (N,) per-domain lambda_n.
        dom_del_rates: (N,) per-domain mu_n.
        ext_rates: (N,) per-domain scalar fragment extension r_n
            (used when n_frag=1).
        Q: (N, A, A) per-domain rate matrices (used when n_frag=1 or
            when class_Q/class_pi/class_dist are not provided).
        pi: (N, A) per-domain equilibrium distributions.
        ext_matrix: optional (N, F, F) per-domain fragment extension
            matrix. Entry [n, f, g] is the probability of transitioning
            from fragment state f to g while extending the current
            fragment (same presence/absence profile). When provided,
            n_frag = F and ext_rates is ignored.
        frag_weights: optional (N, F) initial fragment distribution.
            Entry [n, f] is the probability of starting a new fragment
            in state f within domain n. Must be provided when ext_matrix
            is provided.
        class_Q: optional (C, A, A) per-class rate matrices.
        class_pi: optional (C, A) per-class equilibrium distributions.
        class_dist: optional (N, F, C) per-domain-per-fragment class
            distribution. Entry [n, f, c] is the probability of class c
            given fragment state f in domain n. If not provided with
            ext_matrix, each fragment state uses the per-domain (Q, pi).
    """
    kappa_top: float
    dom_weights: np.ndarray
    dom_ins_rates: np.ndarray
    dom_del_rates: np.ndarray
    ext_rates: np.ndarray
    Q: np.ndarray
    pi: np.ndarray
    ext_matrix: Optional[np.ndarray] = None
    frag_weights: Optional[np.ndarray] = None
    class_Q: Optional[np.ndarray] = None
    class_pi: Optional[np.ndarray] = None
    class_dist: Optional[np.ndarray] = None

    @property
    def n_dom(self) -> int:
        return int(self.dom_weights.shape[0])

    @property
    def n_frag(self) -> int:
        if self.ext_matrix is not None:
            return int(self.ext_matrix.shape[1])
        return 1

    @property
    def n_class(self) -> int:
        if self.class_dist is not None:
            return int(self.class_dist.shape[2])
        return 1

    @property
    def A(self) -> int:
        return int(self.pi.shape[1])

    def get_ext_matrix(self) -> np.ndarray:
        """Return (N, F, F) extension matrix.

        For F=1 (no ext_matrix provided), returns (N, 1, 1) with the
        scalar ext_rates.
        """
        if self.ext_matrix is not None:
            return np.asarray(self.ext_matrix, dtype=np.float64)
        # F=1: scalar ext rate as 1x1 matrix
        N = self.n_dom
        ext = np.zeros((N, 1, 1), dtype=np.float64)
        for n in range(N):
            ext[n, 0, 0] = float(self.ext_rates[n])
        return ext

    def get_frag_weights(self) -> np.ndarray:
        """Return (N, F) fragment weights.

        For F=1, returns (N, 1) of ones.
        """
        if self.frag_weights is not None:
            return np.asarray(self.frag_weights, dtype=np.float64)
        return np.ones((self.n_dom, 1), dtype=np.float64)

    def get_notext(self) -> np.ndarray:
        """Return (N, F) notext probabilities = 1 - sum_g ext[n,f,g]."""
        ext = self.get_ext_matrix()
        return 1.0 - ext.sum(axis=2)

    @classmethod
    def from_mixdom_params(cls,
                           params: dict,
                           kappa_top: Optional[float] = None,
                           ) -> "PartitionReconModel":
        """Build a PartitionReconModel from an already-constrained MixDom
        parameter dict (the output of
        `tkfmixdom.jax.distill.maraschino.load_params`).

        Use this when the caller has already loaded the `.npz` checkpoint
        (as `experiments/unified_reconstruction_benchmark.py` does) and
        wants to reuse the same parameters without a second load.

        For nFrag=1 checkpoints, uses the scalar ext_rates path.
        For nFrag>1 checkpoints with an ``ext_matrix`` key (N, F, F),
        populates the full Markovian fragment parameters (ext_matrix,
        frag_weights).  If ``ext_matrix`` is absent but ``r`` has
        shape (N, F), falls back to the effective scalar ext rate for
        backward compatibility.

        Args:
            params: dict with keys ``lam0, mu0, lam, mu, r, v, S_exch, pi``
                (as returned by ``load_params``). ``S_exch`` may be (A, A) or
                (N, A, A); ``pi`` is (N, A); ``r`` is (N,) (nFrag=1) or
                (N, F) (nFrag>1).  Optional key ``ext_matrix`` (N, F, F)
                enables full Markovian fragment tracking.
            kappa_top: optional override for $\\kappa_0$; defaults to
                $\\lambda_0 / \\mu_0$.

        Returns:
            PartitionReconModel.
        """
        from ..core.ctmc import build_Q_from_S_pi

        lam0 = float(params['lam0'])
        mu0 = float(params['mu0'])
        if kappa_top is None:
            kappa_top = lam0 / mu0
        kappa_top = float(kappa_top)
        kappa_top = min(max(kappa_top, 1e-12), 1.0 - 1e-12)

        dom_ins_rates = np.asarray(params['lam'], dtype=np.float64)
        dom_del_rates = np.asarray(params['mu'], dtype=np.float64)
        dom_weights = np.asarray(params['v'], dtype=np.float64)
        dom_weights = dom_weights / dom_weights.sum()

        ext_rates_raw = np.asarray(params['r'], dtype=np.float64)
        ext_matrix = None
        frag_weights_out = None
        class_dist_out = None

        if 'ext_matrix' in params:
            # Full Markovian fragment parameters (MixDom2).
            ext_matrix = np.asarray(params['ext_matrix'], dtype=np.float64)
            frag_weights_out = np.asarray(params['frag_weights'],
                                          dtype=np.float64)
            # ext_rates is unused by the D×F code path (which uses
            # ext_matrix directly). Set to zeros as a placeholder.
            ext_rates = np.zeros(len(dom_weights), dtype=np.float64)
        elif ext_rates_raw.ndim > 1:
            # MixDom1 nFrag>1: (N, F) per-fragment scalar rates.
            # Convert to diagonal ext_matrix for the unified D×F path.
            r_frags = np.asarray(params['r_frags'], dtype=np.float64)
            frag_weights_out = np.asarray(params['frag_weights'],
                                          dtype=np.float64)
            F = r_frags.shape[1]
            ext_matrix = np.zeros((len(dom_weights), F, F), dtype=np.float64)
            for n in range(len(dom_weights)):
                ext_matrix[n] = np.diag(r_frags[n])
            ext_rates = np.zeros(len(dom_weights), dtype=np.float64)
        else:
            # F=1 scalar ext rate. Wrap as (N,1,1) ext_matrix.
            ext_rates = ext_rates_raw
            ext_matrix = ext_rates_raw[:, None, None].copy()
            frag_weights_out = np.ones((len(dom_weights), 1), dtype=np.float64)

        pi = np.asarray(params['pi'], dtype=np.float64)
        pi = pi / pi.sum(axis=1, keepdims=True)

        S_exch = np.asarray(params['S_exch'], dtype=np.float64)
        N = int(dom_weights.shape[0])
        A = int(pi.shape[1])
        Q = np.zeros((N, A, A), dtype=np.float64)
        for n in range(N):
            S_n = S_exch[n] if S_exch.ndim == 3 else S_exch
            Q[n] = np.asarray(build_Q_from_S_pi(S_n, pi[n]))

        class_Q = None
        class_pi = None
        if 'class_pi' in params:
            class_pi = np.asarray(params['class_pi'], dtype=np.float64)
            # Renormalise per class for safety (stochastic rows over A).
            class_pi = class_pi / class_pi.sum(axis=-1, keepdims=True)
        if 'class_Q' in params:
            class_Q = np.asarray(params['class_Q'], dtype=np.float64)
        elif 'class_S_exch' in params and class_pi is not None:
            # MixDom2 (Annabel-style): build per-class rate matrices from
            # exchangeability + per-class equilibrium. (C, A, A).
            import jax.numpy as _jnp
            class_S = np.asarray(params['class_S_exch'], dtype=np.float64)
            C = class_S.shape[0]
            A = class_pi.shape[1]
            class_Q = np.zeros((C, A, A), dtype=np.float64)
            for c in range(C):
                class_Q[c] = np.asarray(
                    build_Q_from_S_pi(_jnp.asarray(class_S[c]),
                                       _jnp.asarray(class_pi[c])))
        if 'class_dist' in params:
            class_dist_out = np.asarray(params['class_dist'],
                                        dtype=np.float64)

        return cls(
            kappa_top=kappa_top,
            dom_weights=dom_weights,
            dom_ins_rates=dom_ins_rates,
            dom_del_rates=dom_del_rates,
            ext_rates=ext_rates,
            Q=Q,
            pi=pi,
            ext_matrix=ext_matrix,
            frag_weights=frag_weights_out,
            class_Q=class_Q,
            class_pi=class_pi,
            class_dist=class_dist_out,
        )

    @classmethod
    def from_mixdom_checkpoint(cls,
                               path: str,
                               kappa_top: Optional[float] = None,
                               ) -> "PartitionReconModel":
        """Load a trained MixDom checkpoint and build a PartitionReconModel.

        Reads `.npz` checkpoints produced by the MixDom trainers
        (e.g. `pfam/svi_bw_d3f1_full_best_val.npz`, `pfam/svi_bw_d5f1_full_best_val.npz`)
        via `tkfmixdom.jax.distill.maraschino.load_params`, which handles
        both the SVI/BW format and the older BW checkpoint format.

        The vanishing-top-level-rate limit is taken *implicitly*: the
        top-level insertion and deletion rates are collapsed into a
        single ratio $\\kappa_0 = \\lambda_0/\\mu_0$, which is the only
        top-level parameter retained by the partition-conditioned
        algorithm. Any dependence of the per-branch TKF91 structure on
        $\\lambda_0$ and $\\mu_0$ individually (beyond the ratio) is
        dropped — this is exactly the limit $\\lambda_0, \\mu_0 \\to 0$
        with $\\lambda_0/\\mu_0$ held fixed.

        Multi-fragment checkpoints (logit_r shape (N, F) with F > 1)
        are converted to a diagonal (N, F, F) ext_matrix and tracked
        via the D x F partition reconstruction algorithm.

        Multi-site (site-class mixture) checkpoints are supported by
        using the class-weighted marginal equilibrium `pi` and a
        correspondingly weighted rate matrix. For pure multi-site
        mixture models the partition-recon is an approximation: the
        algorithm does not marginalise over within-block site classes.

        Args:
            path: path to a `.npz` checkpoint.
            kappa_top: if provided, override the $\\kappa_0$ read from
                the checkpoint. Useful to probe sensitivity or to fit
                $\\kappa_0$ on a held-out set independently of the
                trained model.

        Returns:
            PartitionReconModel ready to pass to
            `partition_recon_forward_backward` (or the JAX variant).
        """
        from ..distill.maraschino import load_params
        params, _n_domains, _n_classes = load_params(path)
        return cls.from_mixdom_params(params, kappa_top=kappa_top)


@dataclass
class PartitionReconResult:
    """Output of `partition_recon_forward_backward`.

    Args:
        log_Z_forward: log P(MSA | tree, model) from Forward.
        log_Z_backward: log P(MSA | tree, model) from Backward (sanity check).
        class_posterior: (L, N) per-column posterior over domain classes.
        root_residue_posterior: (L, A) per-column posterior at root (ignore
            columns where the root is absent: for those we return a uniform
            distribution as a safe default; caller should use presence to
            mask those out).
        root_residue_map: (L,) int array of per-column argmax root residue,
            -1 at columns where the root is absent.
        root_is_present: (L,) bool array.
        F: (L+1, N) forward table (log-space).
        beta: (L+1,) backward table (log-space) — note beta is class-free.
        G_closed: (L, L, N) block log-likelihoods (log G(k+1, l, n)) for
            0 <= k < l <= L-1. Entries with k >= l are -inf.
        frag_posterior: optional (L, F) per-column marginal fragment state
            posterior. Present when n_frag > 0 and intra-block backward
            was computed.
        site_class_posterior: optional (L, C) per-column marginal site
            class posterior, incorporating fragment-dependent class mixing.
            Present when class_dist is provided.
    """
    log_Z_forward: float
    log_Z_backward: float
    class_posterior: np.ndarray
    root_residue_posterior: np.ndarray
    root_residue_map: np.ndarray
    root_is_present: np.ndarray
    F: np.ndarray
    beta: np.ndarray
    G_closed: np.ndarray
    frag_posterior: Optional[np.ndarray] = None
    site_class_posterior: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def build_inputs(tree: TreeNode,
                 leaf_seqs_aligned: Dict[str, np.ndarray],
                 presence: Optional[Dict[str, np.ndarray]] = None,
                 ) -> PartitionReconInputs:
    """Build a `PartitionReconInputs` given a tree and aligned leaf sequences.

    If `presence` is not supplied, it is computed by Fitch parsimony on
    gaps (leaves present iff non-gap; internals inferred via
    `tree_varanc.infer_internal_presence`).

    Internal nodes of the tree are named in-place if they lack names.
    """
    name_internal_nodes(tree)
    if presence is None:
        leaf_presence = {name: (seq >= 0)
                         for name, seq in leaf_seqs_aligned.items()}
        presence = infer_internal_presence(tree, leaf_presence)
    return PartitionReconInputs(tree=tree,
                                leaf_seqs_aligned=leaf_seqs_aligned,
                                presence=presence)


def _enumerate_edges(tree: TreeNode) -> List[Tuple[Optional[str], str,
                                                    Optional[float],
                                                    TreeNode]]:
    """List edges. Entry 0 is the virtual above-root edge.

    Returns:
        list of (parent_name or None, child_name, branch_length or None, child_node)
    """
    edges = [(None, tree.name, None, tree)]  # virtual above-root
    for node in tree.preorder():
        if node.is_root:
            continue
        edges.append((node.parent.name, node.name, float(node.branch_length), node))
    return edges


def _compute_branch_types(edges, presence, L: int) -> np.ndarray:
    """Per-edge per-column TKF92 branch type in {-1, M, I, D}.

    -1 means untouched (no WFST transition at that column).
    """
    n_b = len(edges)
    types = np.full((n_b, L), -1, dtype=np.int32)
    for bi, (pname, cname, _bl, _cnode) in enumerate(edges):
        cpres = presence[cname]
        if pname is None:
            # Virtual above-root edge: I at columns where root is present.
            idx = np.where(cpres)[0]
            types[bi, idx] = I
        else:
            ppres = presence[pname]
            both = ppres & cpres
            only_p = ppres & (~cpres)
            only_c = (~ppres) & cpres
            types[bi, np.where(both)[0]] = M
            types[bi, np.where(only_p)[0]] = D
            types[bi, np.where(only_c)[0]] = I
    return types


# ---------------------------------------------------------------------------
# Per-edge transition matrices
# ---------------------------------------------------------------------------

def _tkf92_singlet_log_trans(ins_rate: float, del_rate: float,
                              ext: float) -> np.ndarray:
    """Log-TKF92-singlet transition matrix in the 5-state {S,M,I,D,E} layout.

    States M and D are unreachable for a singlet (no input sequence),
    so their rows/columns have NEG_INF except for the diagonal.

    The I->I self-loop is r + (1-r)*kappa, and I->E is (1-r)(1-kappa),
    which together form the TKF92 equilibrium length distribution.
    """
    kappa = float(ins_rate) / float(del_rate)
    r = float(ext)
    tau = np.full((5, 5), 0.0)
    tau[S, I] = kappa
    tau[S, E] = 1.0 - kappa
    tau[I, I] = r + (1.0 - r) * kappa
    tau[I, E] = (1.0 - r) * (1.0 - kappa)
    return _safe_log(tau)



# ---------------------------------------------------------------------------
# Per-column per-domain Felsenstein likelihoods and root posteriors
# ---------------------------------------------------------------------------

def _felsenstein_column(tree: TreeNode, col: int,
                        presence: Dict[str, np.ndarray],
                        leaf_seqs_aligned: Dict[str, np.ndarray],
                        Q: np.ndarray, pi: np.ndarray,
                        ) -> Tuple[float, Optional[np.ndarray]]:
    """Compute (log column likelihood, log root posterior or None).

    The "root posterior" is the log posterior over residues at the
    overall tree root, IF the overall root is present at this column;
    else None. The column likelihood is computed over the present
    subtree (the connected subtree of present nodes).

    Returns:
        log_L_col: float, log of sum_a pi[a] * L_root-of-present-subtree(a),
            or 0.0 if the column is completely absent (no present nodes).
        log_root_post: (A,) log posterior at the overall tree root, or None
            if the root is absent.
    """
    A = int(pi.shape[0])
    present = {name for name in presence if presence[name][col]}

    if not present:
        return 0.0, None

    log_pi = _safe_log(pi)
    log_partial: Dict[str, np.ndarray] = {}

    # Postorder pruning restricted to present nodes.
    for node in tree.postorder():
        if node.name not in present:
            continue
        if node.is_leaf:
            char = int(leaf_seqs_aligned.get(node.name,
                                             np.full(col + 1, -1))[col])
            log_L = np.full(A, NEG_INF)
            if char < 0:
                # Should not happen (leaf is present), but fall back to uniform.
                log_L = np.zeros(A)
            elif char >= A:
                # Wildcard.
                log_L = np.zeros(A)
            else:
                log_L[char] = 0.0
            log_partial[node.name] = log_L
        else:
            log_L = np.zeros(A)
            for child in node.children:
                if child.name not in present:
                    continue
                t = max(float(child.branch_length), 1e-8)
                Pt = np.asarray(transition_matrix(Q, t))
                log_Pt = _safe_log(Pt)
                # log L_parent(a) += logsumexp_b( log Pt[a,b] + log L_child(b) )
                child_log_L = log_partial[child.name]
                contrib = _logsumexp_np(log_Pt + child_log_L[None, :], axis=1)
                log_L = log_L + contrib
            log_partial[node.name] = log_L

    # Root of present subtree = shallowest (in preorder) present node.
    root_of_present = None
    for node in tree.preorder():
        if node.name in present:
            root_of_present = node
            break

    log_L_present_root = log_partial[root_of_present.name]
    log_joint_at_pres_root = log_pi + log_L_present_root
    log_col = _logsumexp_np(log_joint_at_pres_root)

    # Log posterior at the overall tree root, iff the overall root is present.
    if tree.name in present:
        log_root_joint = log_pi + log_partial[tree.name]
        log_root_Z = _logsumexp_np(log_root_joint)
        log_root_post = log_root_joint - log_root_Z
    else:
        log_root_post = None

    return float(log_col), log_root_post


def _precompute_felsenstein(inputs: PartitionReconInputs,
                            model: PartitionReconModel,
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                       Optional[np.ndarray]]:
    """Precompute per-column per-domain Felsenstein log-likelihoods and
    per-column per-domain log root posteriors.

    Returns:
        fels_logliks: (L, N) log column likelihood.
        root_log_post: (L, N, A) log posterior at the overall root
            (meaningful only where root_present[col] is True; else zeros).
        root_present: (L,) bool array.
        class_logliks: (L, C) per-column per-class log-likelihoods, or
            None if the model has no class_Q/class_pi.
    """
    L = inputs.L
    N = model.n_dom
    A = model.A
    C = model.n_class
    fels_logliks = np.zeros((L, N), dtype=np.float64)
    root_log_post = np.zeros((L, N, A), dtype=np.float64)
    root_present = np.asarray(inputs.presence[inputs.tree.name], dtype=bool)
    for col in range(L):
        for n in range(N):
            ll, rp = _felsenstein_column(
                inputs.tree, col, inputs.presence,
                inputs.leaf_seqs_aligned,
                np.asarray(model.Q[n]), np.asarray(model.pi[n]))
            fels_logliks[col, n] = ll
            if rp is not None:
                root_log_post[col, n] = rp
            else:
                root_log_post[col, n] = -np.log(A)

    # Per-class column likelihoods U(j, c) and root posteriors.
    class_logliks = None
    class_root_log_post = None
    if model.class_Q is not None and model.class_pi is not None:
        class_logliks = np.zeros((L, C), dtype=np.float64)
        class_root_log_post = np.zeros((L, C, A), dtype=np.float64)
        for col in range(L):
            for c in range(C):
                ll, rp = _felsenstein_column(
                    inputs.tree, col, inputs.presence,
                    inputs.leaf_seqs_aligned,
                    np.asarray(model.class_Q[c]),
                    np.asarray(model.class_pi[c]))
                class_logliks[col, c] = ll
                if rp is not None:
                    class_root_log_post[col, c] = rp
                else:
                    class_root_log_post[col, c] = -np.log(A)

    return (fels_logliks, root_log_post, root_present,
            class_logliks, class_root_log_post)


# ---------------------------------------------------------------------------
# G(k+1, l, n): block log-likelihoods
# ---------------------------------------------------------------------------

def _build_log_trans_per_edge_tkf91(edges, model: PartitionReconModel) -> np.ndarray:
    """(n_edges, N, 5, 5) log-transition matrices using TKF91 (ext=0).

    Used by the D x F algorithm where fragment extension is handled at
    the column level.  The per-branch transitions use ext=0 so that the
    fragment extension factor is not double-counted.
    """
    n_b = len(edges)
    N = model.n_dom
    out = np.full((n_b, N, 5, 5), NEG_INF, dtype=np.float64)
    for n in range(N):
        lam = float(model.dom_ins_rates[n])
        mu = float(model.dom_del_rates[n])
        # Virtual edge: TKF92 singlet with ext=0 (pure TKF91 geometric).
        out[0, n] = _tkf92_singlet_log_trans(lam, mu, 0.0)
        for bi in range(1, n_b):
            t = float(edges[bi][2])
            if t <= 0.0:
                t = 1e-8
            tau = np.asarray(tkf91_trans_cond(lam, mu, t))
            out[bi, n] = _safe_log(tau)
    return out


def _compute_presence_profile(branch_types: np.ndarray, L: int) -> np.ndarray:
    """Compute per-column presence profile hash for same-profile detection.

    Returns:
        profile_ids: (L,) int array where columns with the same
            presence/absence profile across all branches get the same id.
    """
    n_b = L  # unused
    # Use a tuple of branch types as the profile key.
    profile_map = {}
    profile_ids = np.zeros(L, dtype=np.int32)
    next_id = 0
    for col in range(L):
        key = tuple(branch_types[:, col].tolist())
        if key not in profile_map:
            profile_map[key] = next_id
            next_id += 1
        profile_ids[col] = profile_map[key]
    return profile_ids


def _compute_log_emission(model: PartitionReconModel,
                          fels_logliks: np.ndarray,
                          class_logliks: Optional[np.ndarray],
                          ) -> np.ndarray:
    """Compute log E(d, j, g) = log sum_c classdist_{d,g,c} * U(j,c).

    When class_dist and class_logliks are available, computes the full
    class-dependent emission. Otherwise falls back to E(d,j,g) = fels(j,d).

    Returns:
        log_emission: (L, N, F) array of log emission weights.
    """
    L = fels_logliks.shape[0]
    N = model.n_dom
    F = model.n_frag

    if model.class_dist is not None and class_logliks is not None:
        # E(d,j,g) = sum_c classdist_{d,g,c} * U(j,c)
        # In log-space: logsumexp_c(log classdist_{d,g,c} + log U(j,c))
        log_cd = _safe_log(np.asarray(model.class_dist))  # (N, F, C)
        # class_logliks: (L, C)
        # Vectorized: broadcast (1,N,F,C) + (L,1,1,C) -> (L,N,F,C)
        log_terms = log_cd[None, :, :, :] + class_logliks[:, None, None, :]
        # logsumexp over C axis
        m = np.max(log_terms, axis=3, keepdims=True)
        m_safe = np.where(np.isfinite(m), m, 0.0)
        log_emission = np.squeeze(m_safe, axis=3) + np.log(
            np.sum(np.exp(log_terms - m_safe), axis=3))
        mask = ~np.isfinite(np.squeeze(m, axis=3))
        log_emission = np.where(mask, NEG_INF, log_emission)
        return log_emission
    else:
        # No class_dist: E(d,j,g) = fels(j,d) for all g.
        return np.broadcast_to(fels_logliks[:, :, None],
                               (L, N, F)).copy()


def _compute_block_logliks(inputs: PartitionReconInputs,
                           model: PartitionReconModel,
                           fels_logliks: np.ndarray,
                           class_logliks: Optional[np.ndarray] = None,
                           ) -> np.ndarray:
    """Compute log G(k+1, l, n) for every valid (k, l, n).

    G[k, l, n] is the TKF92 block likelihood on columns k+1..l (using
    1-based math, so array indices k in [0, L-1] and l in [k, L-1])
    under domain n. Entries with k > l or k == l are left at -inf.

    Precisely, `G_closed[k, l, n]` in this function's output
    corresponds to log G(k+1, l+1, n) in the paper, for k in [0..L-1]
    and l in [k..L-1].

    Uses the intra-block forward recurrence over D x F column states
    (domain d, fragment state f), with TKF91 (ext=0) per-branch
    transitions and column-level fragment extension via ext_matrix.

    Complexity: O(L^2 * N * F * T) where T = number of tree edges + 1.
    """
    L = inputs.L
    N = model.n_dom
    F = model.n_frag
    edges = _enumerate_edges(inputs.tree)
    n_b = len(edges)

    # Per-branch TKF91 transitions (ext=0).
    log_trans_tkf91 = _build_log_trans_per_edge_tkf91(edges, model)
    branch_types = _compute_branch_types(edges, inputs.presence, L)

    # Fragment parameters.
    ext_mat = model.get_ext_matrix()      # (N, F, F)
    frag_wts = model.get_frag_weights()   # (N, F)
    notext = model.get_notext()           # (N, F)

    log_ext = _safe_log(ext_mat)          # (N, F, F)
    log_frag_wts = _safe_log(frag_wts)    # (N, F)
    log_notext = _safe_log(notext)        # (N, F)

    # Close-to-E factors from TKF91 transitions.
    close_lp = log_trans_tkf91[:, :, :, E]  # (n_b, N, 5)

    # Presence profile ids for same-profile detection.
    profile_ids = _compute_presence_profile(branch_types, L)

    # Emission: E(d, j, g) = sum_c classdist_{d,g,c} * U(j,c).
    log_emission = _compute_log_emission(model, fels_logliks, class_logliks)
    # (L, N, F)

    G_closed = np.full((L, L, N), NEG_INF, dtype=np.float64)

    for k in range(L):
        last_state = np.full((N, n_b), S, dtype=np.int32)
        F_intra = np.full((N, F), NEG_INF, dtype=np.float64)
        prev_profile = -1

        for l in range(k, L):
            types_l = branch_types[:, l]
            log_branch_inc = np.zeros(N, dtype=np.float64)
            touched_mask = types_l >= 0
            if np.any(touched_mask):
                touched_b = np.where(touched_mask)[0]
                for bi in touched_b:
                    new_st = int(types_l[bi])
                    for n in range(N):
                        ls = int(last_state[n, bi])
                        log_branch_inc[n] += log_trans_tkf91[bi, n, ls, new_st]
                        last_state[n, bi] = new_st

            cur_profile = int(profile_ids[l])
            same_profile = (l > k) and (cur_profile == prev_profile)

            if l == k:
                for n in range(N):
                    for g in range(F):
                        F_intra[n, g] = (log_branch_inc[n]
                                         + log_frag_wts[n, g]
                                         + log_emission[l, n, g])
            else:
                new_F_intra = np.full((N, F), NEG_INF, dtype=np.float64)
                for n in range(N):
                    for g in range(F):
                        log_terms = np.full(F, NEG_INF, dtype=np.float64)
                        for f in range(F):
                            if same_profile:
                                t_ext = log_ext[n, f, g]
                                t_notext = (log_notext[n, f]
                                            + log_branch_inc[n]
                                            + log_frag_wts[n, g])
                                t_col = _logsumexp_np(
                                    np.array([t_ext, t_notext]))
                            else:
                                t_col = (log_notext[n, f]
                                         + log_branch_inc[n]
                                         + log_frag_wts[n, g])
                            log_terms[f] = F_intra[n, f] + t_col

                        new_F_intra[n, g] = (_logsumexp_np(log_terms)
                                             + log_emission[l, n, g])
                F_intra = new_F_intra

            # Block close: G(k+1, l+1, d) = sum_f F_intra[d,f] * notext_f * T_end
            for n in range(N):
                log_t_end = 0.0
                for bi in range(n_b):
                    log_t_end += close_lp[bi, n, int(last_state[n, bi])]

                log_terms = np.full(F, NEG_INF, dtype=np.float64)
                for f in range(F):
                    log_terms[f] = F_intra[n, f] + log_notext[n, f]
                G_closed[k, l, n] = _logsumexp_np(log_terms) + log_t_end

            prev_profile = cur_profile

    return G_closed


# ---------------------------------------------------------------------------
# Intra-block forward and backward over fragment states
# ---------------------------------------------------------------------------

def _compute_intra_block_forward(inputs: PartitionReconInputs,
                                 model: PartitionReconModel,
                                 log_emission: np.ndarray,
                                 ) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
    """Compute the full intra-block forward table F_{i,j,d,f}.

    Returns:
        F_table: (L, L, N, F) where F_table[k, l, n, f] = log F_{k,l,n,f}.
            Only entries with k <= l are meaningful.
        log_branch_inc_table: (L, L, N) incremental per-column TKF log
            transition (needed by the backward).
        last_state_table: (L, L, N, n_b) int per-branch state after column l
            given block start k (needed by backward for T_end).
        profile_ids: (L,) presence profile ids.
    """
    L = inputs.L
    N = model.n_dom
    F = model.n_frag
    edges = _enumerate_edges(inputs.tree)
    n_b = len(edges)

    log_trans_tkf91 = _build_log_trans_per_edge_tkf91(edges, model)
    branch_types = _compute_branch_types(edges, inputs.presence, L)

    ext_mat = model.get_ext_matrix()
    frag_wts = model.get_frag_weights()
    notext = model.get_notext()

    log_ext = _safe_log(ext_mat)
    log_frag_wts = _safe_log(frag_wts)
    log_notext = _safe_log(notext)

    profile_ids = _compute_presence_profile(branch_types, L)

    F_table = np.full((L, L, N, F), NEG_INF, dtype=np.float64)
    log_branch_inc_table = np.zeros((L, L, N), dtype=np.float64)
    last_state_table = np.full((L, L, N, n_b), S, dtype=np.int32)

    for k in range(L):
        last_state = np.full((N, n_b), S, dtype=np.int32)
        F_intra = np.full((N, F), NEG_INF, dtype=np.float64)
        prev_profile = -1

        for l in range(k, L):
            types_l = branch_types[:, l]
            log_branch_inc = np.zeros(N, dtype=np.float64)
            touched_mask = types_l >= 0
            if np.any(touched_mask):
                touched_b = np.where(touched_mask)[0]
                for bi in touched_b:
                    new_st = int(types_l[bi])
                    for n in range(N):
                        ls = int(last_state[n, bi])
                        log_branch_inc[n] += log_trans_tkf91[bi, n, ls, new_st]
                        last_state[n, bi] = new_st

            log_branch_inc_table[k, l] = log_branch_inc
            last_state_table[k, l] = last_state.copy()  # (N, n_b) snapshot

            cur_profile = int(profile_ids[l])
            same_profile = (l > k) and (cur_profile == prev_profile)

            if l == k:
                for n in range(N):
                    for g in range(F):
                        F_intra[n, g] = (log_branch_inc[n]
                                         + log_frag_wts[n, g]
                                         + log_emission[l, n, g])
            else:
                new_F_intra = np.full((N, F), NEG_INF, dtype=np.float64)
                for n in range(N):
                    for g in range(F):
                        log_terms = np.full(F, NEG_INF, dtype=np.float64)
                        for f in range(F):
                            if same_profile:
                                t_ext = log_ext[n, f, g]
                                t_notext = (log_notext[n, f]
                                            + log_branch_inc[n]
                                            + log_frag_wts[n, g])
                                t_col = _logsumexp_np(
                                    np.array([t_ext, t_notext]))
                            else:
                                t_col = (log_notext[n, f]
                                         + log_branch_inc[n]
                                         + log_frag_wts[n, g])
                            log_terms[f] = F_intra[n, f] + t_col
                        new_F_intra[n, g] = (_logsumexp_np(log_terms)
                                             + log_emission[l, n, g])
                F_intra = new_F_intra

            F_table[k, l] = F_intra
            prev_profile = cur_profile

    return F_table, log_branch_inc_table, last_state_table, profile_ids


def _compute_intra_block_backward(inputs: PartitionReconInputs,
                                  model: PartitionReconModel,
                                  log_emission: np.ndarray,
                                  log_branch_inc_table: np.ndarray,
                                  last_state_table: np.ndarray,
                                  profile_ids: np.ndarray,
                                  ) -> np.ndarray:
    """Compute the intra-block backward table B_{i,k,j,d,f}.

    B_{i,k,j,d,f} is the probability of columns k..j given a block i..j
    in domain d, with column k-1 in fragment state f.

    Boundary (eq:intra-backward-boundary):
        B_{i,j+1,j,d,f} = notext^{(d)}_f * T_{tkf,end}(d,i,j)

    Recursion (eq:intra-backward-recurse):
        B_{i,k,j,d,f} = sum_g [delta(A(k-1)=A(k)) * ext_{fg}
                               + notext_f * T_tkf(d,i,k) * fragdist_{d,g}]
                         * E(d,k,g) * B_{i,k+1,j,d,g}

    Returns:
        B_table: (L, L, L, N, F) where B_table[i, k, j, n, f] =
            log B_{i, k+1, j, n, f} (note: k+1 offset — B_table[i,k,j,n,f]
            gives the backward probability for columns k+1..j given frag
            state f at column k).
            For k=j, B_table[i, j, j, n, f] = boundary = notext_f * T_end.
            For k<j, B_table[i, k, j, n, f] = recursive step.
    """
    L = inputs.L
    N = model.n_dom
    F = model.n_frag
    edges = _enumerate_edges(inputs.tree)
    n_b = len(edges)

    log_trans_tkf91 = _build_log_trans_per_edge_tkf91(edges, model)
    close_lp = log_trans_tkf91[:, :, :, E]  # (n_b, N, 5)

    ext_mat = model.get_ext_matrix()
    frag_wts = model.get_frag_weights()
    notext = model.get_notext()

    log_ext = _safe_log(ext_mat)
    log_frag_wts = _safe_log(frag_wts)
    log_notext = _safe_log(notext)

    # B_table[i, k, j, n, f] = log B_{i, k+1, j, n, f}
    # Index convention: B_table[i, k, j, n, f] represents the backward
    # from column k+1 through j, given fragment state f at column k.
    # So B_table[i, j, j, n, f] = notext_f * T_end (boundary).
    B_table = np.full((L, L, L, N, F), NEG_INF, dtype=np.float64)

    for i in range(L):
        for j in range(i, L):
            # Boundary: B_{i,j+1,j,d,f} = notext_f * T_end(d,i,j)
            # Stored at B_table[i, j, j, n, f].
            for n in range(N):
                log_t_end = 0.0
                for bi in range(n_b):
                    log_t_end += close_lp[bi, n, int(last_state_table[i, j, n, bi])]
                for f in range(F):
                    B_table[i, j, j, n, f] = log_notext[n, f] + log_t_end

            # Backward from k = j-1 down to i.
            for k in range(j - 1, i - 1, -1):
                # B_{i,k+1,j,d,f} = sum_g T_col(f,g,k+1) * E(d,k+1,g) * B_{i,k+2,j,d,g}
                # where T_col uses profile match between col k and k+1.
                same_profile = (profile_ids[k] == profile_ids[k + 1])
                for n in range(N):
                    for f in range(F):
                        log_terms = np.full(F, NEG_INF, dtype=np.float64)
                        for g in range(F):
                            if same_profile:
                                t_ext = log_ext[n, f, g]
                                t_notext = (log_notext[n, f]
                                            + log_branch_inc_table[i, k + 1, n]
                                            + log_frag_wts[n, g])
                                t_col = _logsumexp_np(
                                    np.array([t_ext, t_notext]))
                            else:
                                t_col = (log_notext[n, f]
                                         + log_branch_inc_table[i, k + 1, n]
                                         + log_frag_wts[n, g])
                            log_terms[g] = (t_col
                                            + log_emission[k + 1, n, g]
                                            + B_table[i, k + 1, j, n, g])
                        B_table[i, k, j, n, f] = _logsumexp_np(log_terms)

    return B_table


def _compute_frag_class_posteriors(
        F_table: np.ndarray,
        B_table: np.ndarray,
        G_closed: np.ndarray,
        bar_F: np.ndarray,
        beta: np.ndarray,
        model: PartitionReconModel,
        log_Z: float,
        class_logliks: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-column fragment state and site class posteriors.

    Uses eqs (post-frag-full) and (post-class) from partition-recon.tex.

    Returns:
        frag_posterior: (L, F) per-column marginal fragment state posterior.
        class_posterior_full: (L, C) per-column marginal site class posterior,
            incorporating fragment-dependent class mixing.
    """
    L = G_closed.shape[0]
    N = model.n_dom
    F = model.n_frag
    C = model.n_class
    log_kappa = np.log(max(float(model.kappa_top), 1e-300))
    log_v = _safe_log(np.asarray(model.dom_weights))

    frag_posterior = np.zeros((L, F), dtype=np.float64)
    class_posterior_full = np.zeros((L, C), dtype=np.float64)

    # For class posterior: need classdist and per-class column likelihoods.
    has_classes = (model.class_dist is not None and class_logliks is not None)
    if has_classes:
        log_cd = _safe_log(np.asarray(model.class_dist))  # (N, F, C)
        # Precompute log E(n,col,f) = logsumexp_cc(log_cd[n,f,cc] + class_logliks[col,cc])
        log_E = np.full((L, N, F), NEG_INF, dtype=np.float64)
        for col in range(L):
            for nn in range(N):
                for ff in range(F):
                    log_E[col, nn, ff] = _logsumexp_np(
                        log_cd[nn, ff, :] + class_logliks[col, :])

    for c in range(L):
        for n in range(N):
            for i in range(c + 1):
                for j in range(c, L):
                    if G_closed[i, j, n] <= NEG_INF + 1:
                        continue
                    log_block_w = (bar_F[i] + log_kappa + log_v[n]
                                   + G_closed[i, j, n] + beta[j + 1]
                                   - log_Z)
                    block_w = np.exp(log_block_w)
                    if block_w < 1e-300:
                        continue

                    for f in range(F):
                        # P(col c is frag f | block i..j, dom n)
                        # = F_{i,c,n,f} * B_{i,c+1,j,n,f} / G(i,j,n)
                        log_frag_p = (F_table[i, c, n, f]
                                      + B_table[i, c, j, n, f]
                                      - G_closed[i, j, n])
                        frag_p = np.exp(log_frag_p)
                        frag_posterior[c, f] += block_w * frag_p

                        # Site class posterior (eq:post-class):
                        # P(class cc | col c, frag f, dom n)
                        #   = classdist_{n,f,cc} * U(c,cc) / E(n,c,f)
                        # where E(n,c,f) = sum_cc' classdist_{n,f,cc'} * U(c,cc')
                        if has_classes:
                            for cc in range(C):
                                log_class_given_frag = (
                                    log_cd[n, f, cc]
                                    + class_logliks[c, cc]
                                    - log_E[c, n, f])
                                class_posterior_full[c, cc] += (
                                    block_w * frag_p
                                    * np.exp(log_class_given_frag))

    # Normalise.
    frag_row_sum = frag_posterior.sum(axis=1, keepdims=True)
    frag_posterior = np.where(frag_row_sum > 0,
                             frag_posterior / frag_row_sum, 1.0 / F)

    if has_classes:
        class_row_sum = class_posterior_full.sum(axis=1, keepdims=True)
        class_posterior_full = np.where(class_row_sum > 0,
                                       class_posterior_full / class_row_sum,
                                       1.0 / C)

    return frag_posterior, class_posterior_full


# ---------------------------------------------------------------------------
# Forward / Backward over blocks
# ---------------------------------------------------------------------------

def _forward(G_closed: np.ndarray,
             dom_weights: np.ndarray,
             kappa_top: float,
             ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Forward recursion in log-space over the block HMM.

    F[l, n] (for l in 1..L, n in 0..N-1) = log P(columns 1..l, last block
    ends at l in class n), including the kappa_top * v_n prior factors
    for all blocks up to and including the one ending at l.

    bar_F[k] = log sum_n F[k, n] for k in 1..L; bar_F[0] = 0 (i.e. prior 1).

    Returns:
        F: (L+1, N) log-table. F[0, :] is not used directly and is set
           to -inf.
        bar_F: (L+1,) log-array with bar_F[0] = 0.
        log_Z: log P(MSA | tree, model) = log (1-kappa_top) + log sum_n F[L, n].
    """
    L = G_closed.shape[0]
    N = G_closed.shape[2]
    log_v = _safe_log(np.asarray(dom_weights))
    log_kappa = np.log(max(float(kappa_top), 1e-300))
    log_1m_kappa = np.log(max(1.0 - float(kappa_top), 1e-300))

    F = np.full((L + 1, N), NEG_INF, dtype=np.float64)
    bar_F = np.full((L + 1,), NEG_INF, dtype=np.float64)
    bar_F[0] = 0.0  # "prior 1" before any block

    # F[l, n] = log_kappa + log_v[n] + logsumexp_{k=0..l-1}(bar_F[k] + G[k, l-1, n])
    # Here l is 1-indexed (1..L), so the block ends at array column l-1.
    for l in range(1, L + 1):
        # k ranges 0..l-1 (bar_F indexed at k), G indexed at [k, l-1, n].
        # bar_F[0..l-1] shape (l,); G shape (l, N).
        log_terms = bar_F[:l, None] + G_closed[:l, l - 1, :]  # (l, N)
        F[l, :] = log_kappa + log_v + _logsumexp_np(log_terms, axis=0)
        bar_F[l] = _logsumexp_np(F[l, :])

    log_Z = log_1m_kappa + _logsumexp_np(F[L, :])
    return F, bar_F, float(log_Z)


def _backward(G_closed: np.ndarray,
              dom_weights: np.ndarray,
              kappa_top: float,
              ) -> Tuple[np.ndarray, float]:
    """Backward recursion in log-space over the block HMM.

    Let beta[l] (for l in 0..L) = log P(columns l+1..L | last block ended
    at column l). Note this is class-free because beta depends only on
    what comes after the boundary.

    Recursion:
        beta[L] = log(1 - kappa_top)
        beta[l] = log_kappa + logsumexp_{n'} [ log v_{n'}
                     + logsumexp_{l'=l+1..L} (G[l, l'-1, n'] + beta[l']) ]
                  for l < L.

    Returns:
        beta: (L+1,) log-array.
        log_Z: log P(MSA | tree, model) recovered from the backward pass
            via beta[0] + 0 (since bar_F[0] = 0).
    """
    L = G_closed.shape[0]
    N = G_closed.shape[2]
    log_v = _safe_log(np.asarray(dom_weights))
    log_kappa = np.log(max(float(kappa_top), 1e-300))
    log_1m_kappa = np.log(max(1.0 - float(kappa_top), 1e-300))

    beta = np.full((L + 1,), NEG_INF, dtype=np.float64)
    beta[L] = log_1m_kappa

    for l in range(L - 1, -1, -1):
        # For each n': inner logsumexp over l' in l+1..L.
        # G[l, l'-1, n'] + beta[l'], where l' = l+1..L so l'-1 = l..L-1.
        # G indexing: [k=l, l_end=l..L-1, n'] has shape (L-l, N).
        # Add beta[l'=l+1..L] shape (L-l,).
        log_inner = G_closed[l, l:L, :] + beta[l + 1:L + 1, None]  # (L-l, N)
        # Sum over l' per n'.
        lse_per_n = _logsumexp_np(log_inner, axis=0)  # (N,)
        beta[l] = log_kappa + _logsumexp_np(log_v + lse_per_n)

    # log Z from backward = bar_F[0] + beta[0] = 0 + beta[0] = beta[0]
    log_Z = float(beta[0])
    return beta, log_Z


# ---------------------------------------------------------------------------
# Posterior extraction
# ---------------------------------------------------------------------------

def _posterior_class_per_column(G_closed: np.ndarray,
                                bar_F: np.ndarray,
                                beta: np.ndarray,
                                dom_weights: np.ndarray,
                                kappa_top: float,
                                log_Z: float,
                                ) -> np.ndarray:
    """Compute P(column c in class n | MSA) for all (c, n).

    P(c in n, data) = kappa_top * v_n * sum_{k<c, l>=c} bar_F[k] * G[k, l-1, n] * beta[l]

    In log-space, summed over (k in 0..c-1) and (l in c..L), with
    G_closed[k, l-1, n] and beta[l].

    Returns (L, N) array summing to 1 along the class axis.
    """
    L = G_closed.shape[0]
    N = G_closed.shape[2]
    log_v = _safe_log(np.asarray(dom_weights))
    log_kappa = np.log(max(float(kappa_top), 1e-300))

    posterior = np.zeros((L, N), dtype=np.float64)

    # Work out: for each column c (1-indexed), iterate k=0..c-1 and l=c..L.
    # bar_F[0..c-1] has shape (c,); G[k, l-1, n] needs k index and
    # l_end = l - 1 = c-1..L-1. The sum is over (k, l_end) pairs with
    # k <= l_end (always true here since k <= c-1 <= l_end).
    for c in range(1, L + 1):
        ks = np.arange(c)  # 0..c-1
        l_ends = np.arange(c - 1, L)  # l_end indices from c-1..L-1

        # For each (k, l_end): bar_F[k] + G[k, l_end, n] + beta[l_end + 1]
        # Build log_terms shape (len(ks), len(l_ends), N).
        log_terms = (bar_F[ks][:, None, None]
                     + G_closed[np.ix_(ks, l_ends)]
                     + beta[l_ends + 1][None, :, None])

        # logsumexp over (k, l_end) per n.
        flat = log_terms.reshape(-1, N)
        lse_per_n = _logsumexp_np(flat, axis=0)  # (N,)

        log_post_un = log_kappa + log_v + lse_per_n - log_Z
        posterior[c - 1, :] = np.exp(log_post_un)

    # Numerical safety: renormalise rows.
    row_sum = posterior.sum(axis=1, keepdims=True)
    posterior = np.where(row_sum > 0, posterior / row_sum, 1.0 / N)
    return posterior


def _mix_root_posterior(class_posterior: np.ndarray,
                        root_log_post: np.ndarray,
                        root_present: np.ndarray,
                        site_class_posterior: Optional[np.ndarray] = None,
                        class_root_log_post: Optional[np.ndarray] = None,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Mix Felsenstein root posteriors to get marginal root residue posteriors.

    When site class info is available (eq:root-post in partition-recon.tex):
        P(root=a|MSA) = sum_dom P(c in dom|MSA) *
                         sum_class P(class at c|dom) *
                         P(root=a|MSA, c in class)

    Without site classes, falls back to mixing over domains only.

    Args:
        class_posterior: (L, N) per-column domain posterior.
        root_log_post: (L, N, A) per-domain log posterior at root.
        root_present: (L,) bool.
        site_class_posterior: optional (L, C) per-column site class posterior.
        class_root_log_post: optional (L, C, A) per-class log root posterior.

    Returns:
        root_post: (L, A) marginal root residue posterior.
        root_map: (L,) argmax root residue; -1 at root-absent columns.
    """
    L, N, A = root_log_post.shape

    if site_class_posterior is not None and class_root_log_post is not None:
        # eq:root-post: mix over site classes using per-class posteriors.
        C = site_class_posterior.shape[1]
        class_root_prob = np.exp(class_root_log_post)  # (L, C, A)
        # P(root=a|MSA) = sum_c P(class=c|col) * P(root=a|MSA, class=c)
        root_post = np.einsum('lc,lca->la',
                              site_class_posterior, class_root_prob)
    else:
        # Fallback: mix over domains (no site class info).
        root_prob = np.exp(root_log_post)
        root_post = np.einsum('cn,cna->ca', class_posterior, root_prob)

    row_sum = root_post.sum(axis=1, keepdims=True)
    root_post = np.where(row_sum > 0, root_post / row_sum, 1.0 / A)
    root_map = np.argmax(root_post, axis=1).astype(np.int32)
    root_map[~root_present] = -1
    return root_post, root_map


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def partition_recon_forward_backward(inputs: PartitionReconInputs,
                                     model: PartitionReconModel,
                                     ) -> PartitionReconResult:
    """Run the full partition-conditioned Forward-Backward reconstruction.

    WARNING: This is the REFERENCE implementation (plain Python loops).
    It is correct but slow. For production use, call
    ``partition_recon_forward_backward_jax()`` from partition_recon_jax.py.

    See this module's docstring for a summary of the algorithm.

    Args:
        inputs: PartitionReconInputs (tree, leaf sequences, presence).
        model: PartitionReconModel (the restricted MixDom parameters).

    Returns:
        PartitionReconResult with log-likelihoods, posteriors, and
        reconstruction.
    """
    # 1. Precompute per-column per-domain Felsenstein log-likelihoods
    #    and root posteriors.
    (fels_logliks, root_log_post, root_present,
     class_logliks, class_root_log_post) = \
        _precompute_felsenstein(inputs, model)

    # 2. Compute class-dependent emission E(d, j, g).
    log_emission = _compute_log_emission(model, fels_logliks, class_logliks)

    # 3. Compute G(k+1, l, n) for all (k, l, n).
    G_closed = _compute_block_logliks(inputs, model, fels_logliks,
                                      class_logliks)

    # 4. Inter-block Forward.
    F, bar_F, log_Z_forward = _forward(G_closed, model.dom_weights,
                                       model.kappa_top)

    # 5. Inter-block Backward.
    beta, log_Z_backward = _backward(G_closed, model.dom_weights,
                                     model.kappa_top)

    # 6. Per-column domain class posterior.
    class_post = _posterior_class_per_column(
        G_closed, bar_F, beta, model.dom_weights, model.kappa_top,
        log_Z_forward)

    # 7. Intra-block forward/backward for fragment and site class posteriors.
    frag_post = None
    site_class_post = None
    F_fwd = model.n_frag
    if F_fwd >= 1:
        F_table, log_branch_inc_table, last_state_table, profile_ids = \
            _compute_intra_block_forward(inputs, model, log_emission)
        B_table = _compute_intra_block_backward(
            inputs, model, log_emission,
            log_branch_inc_table, last_state_table, profile_ids)
        frag_post, site_class_post = _compute_frag_class_posteriors(
            F_table, B_table, G_closed, bar_F, beta, model,
            log_Z_forward, class_logliks)

    # 8. Mix to get root residue posteriors (eq:root-post).
    #    Uses per-class posteriors when site classes are available.
    root_post, root_map = _mix_root_posterior(
        class_post, root_log_post, root_present,
        site_class_posterior=site_class_post,
        class_root_log_post=class_root_log_post)

    return PartitionReconResult(
        log_Z_forward=float(log_Z_forward),
        log_Z_backward=float(log_Z_backward),
        class_posterior=class_post,
        root_residue_posterior=root_post,
        root_residue_map=root_map,
        root_is_present=root_present,
        F=F,
        beta=beta,
        G_closed=G_closed,
        frag_posterior=frag_post,
        site_class_posterior=site_class_post,
    )