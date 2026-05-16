"""Hierarchical Gillespie simulator for the MixDom2 generative model.

A *fully labeled* simulator that uses Gillespie BDI events at every level
of the MixDom hierarchy:

  Step 1 (this file): TKF91-on-tree.  Evolve a sequence of "links" via
      Gillespie BDI on each edge of a phylogenetic tree.  Each link gets a
      unique lineage id; the per-edge event counts (births, immigrations,
      deaths) and the time-integrated population are recorded for use as
      sufficient-statistics ground truth.  The resulting MSA is an indicator
      matrix [leaf, lineage] = 1 iff the lineage survived to that leaf.

  Step 2 (decorator, in this file): given a labeled top-level run, build
      one subtree per surviving lineage, pruned to the leaves where it is
      present.  The subtree edge lengths are the original edge lengths
      restricted to the lineage's lifetime in each branch.

  Step 3 (TODO, follow-up): for each domain link's subtree run TKF91-on-tree
      again to evolve fragments.  For each fragment link's subtree, sample
      a fragchar Markov chain at the root (first fragchar from \\fragdist,
      successive ones via \\ext / \\notext), assign each fragchar a site
      class, then evolve substitutions on the subtree via continuous-time
      Gillespie under the class-specific rate matrix.

Why all this?  The MixDom2 chi Pair HMM and its collapsed FB suff-stats are
*marginalised over latent state*.  Parameter recovery from the chi tensor
exercises the chain-restoration code (mixdom_stats_from_collapsed_counts)
but is silent on whether the chain restoration agrees with the *generative*
model when the latent state is fully observed.  This simulator produces
fully labeled traces: every event is attributable to a level (top-level
domain BDI, per-domain fragment BDI, per-fragment fragchar Markov chain,
per-fragchar substitution Gillespie) and a parameter group.  Parameter
recovery from the labeled traces is the strongest possible sanity check.

The Gillespie inner loop used here is the same as in
``simulate_bdi_gillespie`` (simulate.py), but extended to track per-event
parent-of-new-lineage and time-integrated population so the per-edge stats
sum exactly to the BDI sufficient statistics under the generative process.
"""

from __future__ import annotations

import numpy as np

from ..util.io import TreeNode


# ---------------------------------------------------------------------------
# Stationary-distribution helpers (TKF91 / TKF92 root sampling).
# ---------------------------------------------------------------------------


def sample_tkf91_stationary_length(np_rng, ins_rate, del_rate):
    """Sample chain length L from the TKF91 stationary distribution.

    L ~ Geometric(1 − λ/μ) on the support {0, 1, 2, ...}:
        P(L = k) = (λ/μ)^k * (1 − λ/μ).
    Mean E[L] = λ / (μ − λ).

    Requires μ > λ; otherwise the chain has no stationary distribution.
    """
    if del_rate <= ins_rate:
        raise ValueError(
            f'TKF91 has no stationary distribution unless del_rate > '
            f'ins_rate; got ins={ins_rate}, del={del_rate}')
    beta = float(ins_rate) / float(del_rate)
    # numpy.random.geometric(p) returns L ≥ 1 with P(L=k) = (1-p)^(k-1) p.
    # We want L ≥ 0 with P(L=k) = beta^k * (1 - beta), which is the same
    # distribution shifted down by one.
    return int(np_rng.geometric(1.0 - beta)) - 1


def sample_tkf92_stationary_root(np_rng, ins_rate, del_rate, ext):
    """Sample the (n_fragments, fragment_lengths) for a TKF92-stationary
    root chain.

    Fragments are at TKF91 stationary at the FRAGMENT level (geometric
    count) and each fragment has shifted-geometric(1 − ext) length, per
    the standard TKF92 derivation.  This yields a site-chain length
    distribution that is the convolution of these two geometrics.

    Returns (n_fragments, list[int] of fragment lengths).
    """
    n = sample_tkf91_stationary_length(np_rng, ins_rate, del_rate)
    lengths = []
    for _ in range(n):
        n_extend = 0
        while np_rng.random() < ext:
            n_extend += 1
        lengths.append(1 + n_extend)
    return n, lengths


# ---------------------------------------------------------------------------
# Step 1: TKF91 BDI Gillespie on one edge.
# ---------------------------------------------------------------------------


def gillespie_bdi_edge(np_rng, alive_lineages, ins_rate, del_rate, t,
                         next_lineage_id):
    """Run a single TKF91 BDI Gillespie trajectory on one edge of length t.

    The process has continuous-time rates per state:
        - per-lineage birth rate     : λ
        - per-lineage death rate     : μ
        - immortal-link immigration  : λ  (a single "immortal" lineage,
                                            not represented in
                                            alive_lineages, that emits
                                            insertions throughout)

    Args:
        np_rng:           numpy.random.RandomState (mutable).
        alive_lineages:   iterable[int] — lineage ids alive at the parent
                          end of this edge.  Treated as a list; the input
                          is NOT mutated.
        ins_rate:         λ.
        del_rate:         μ.
        t:                edge length.
        next_lineage_id:  first available lineage id for this run.  The
                          returned dict reports the updated counter so
                          callers can reuse it across edges.

    Returns:
        dict with keys:
            'alive_at_end'    : list[int] of lineages alive at child end,
                                in creation order (preserves chain order
                                modulo deaths).
            'next_lineage_id' : int — first unused lineage id.
            'n_births'        : int — total births from existing lineages.
            'n_imm'           : int — total immigrations from immortal link.
            'n_deaths'        : int — total deaths.
            'sojourn'         : float — ∫_0^t |alive(s)| ds.
            'parent_of_new'   : dict[int -> int] — for each newly created
                                lineage, its parent lineage id.  -1 means
                                the immortal link.
            'death_lineages'  : list[int] — lineages that died on this edge,
                                in chronological order.
            'birth_time_in_edge'   : dict[lineage_id -> float] — for each
                                NEW lineage created on this edge, its
                                birth time τ ∈ (0, t).  Used downstream to
                                tally the "blind interval" (0, τ) during
                                which the genealogical-correction approxi-
                                mation falsely places the lineage.
            'lifetime_in_edge'     : dict[lineage_id -> float] — for EVERY
                                lineage that was alive at any point during
                                this edge (whether at-start, newly born, or
                                died), its time alive within (0, t).  The
                                values sum to `sojourn`.

    Notes:
        - We use a list (not a set) for `alive` to keep deterministic
          iteration order; population dynamics depend only on |alive|, so
          uniform sampling among lineages does the right thing.
        - The sojourn integral is computed exactly (no discretisation):
          we add |alive| * dt before each event, and |alive| * (t - s) for
          the residual interval after the last event.
    """
    alive = list(alive_lineages)
    s = 0.0
    n_births = 0
    n_imm = 0
    n_deaths = 0
    sojourn = 0.0
    parent_of_new = {}
    death_lineages = []
    # Per-lineage tracking for the blind-period tally.  birth_time[L] is
    # the time within this edge at which L came into existence (0 for
    # at-start lineages, τ for new births).  When L exits (either by
    # death or by edge end) we accumulate (exit_time - birth_time) into
    # lifetime_in_edge[L].
    birth_time = {int(L): 0.0 for L in alive}
    birth_time_in_edge = {}
    lifetime_in_edge = {}

    while True:
        n = len(alive)
        birth_rate = ins_rate * n
        death_rate = del_rate * n
        imm_rate = ins_rate
        total_rate = birth_rate + death_rate + imm_rate

        if total_rate < 1e-30:
            sojourn += n * (t - s)
            s = t
            break

        dt = np_rng.exponential(1.0 / total_rate)
        if s + dt > t:
            sojourn += n * (t - s)
            s = t
            break

        sojourn += n * dt
        s += dt

        u = np_rng.random() * total_rate
        if u < birth_rate:
            # Pick a parent uniformly from alive lineages.
            parent = alive[np_rng.randint(0, n)]
            new_id = next_lineage_id
            alive.append(new_id)
            parent_of_new[new_id] = int(parent)
            birth_time[new_id] = s
            birth_time_in_edge[new_id] = s
            next_lineage_id += 1
            n_births += 1
        elif u < birth_rate + death_rate:
            # Pick a victim uniformly from alive lineages.
            victim_idx = np_rng.randint(0, n)
            victim = alive.pop(victim_idx)
            lifetime_in_edge[int(victim)] = s - birth_time.pop(int(victim))
            death_lineages.append(int(victim))
            n_deaths += 1
        else:
            # Immortal-link immigration.
            new_id = next_lineage_id
            alive.append(new_id)
            parent_of_new[new_id] = -1
            birth_time[new_id] = s
            birth_time_in_edge[new_id] = s
            next_lineage_id += 1
            n_imm += 1

    # Edge-end: every still-alive lineage contributes (t - birth_time) to
    # its in-edge lifetime.
    for L in alive:
        lifetime_in_edge[int(L)] = t - birth_time[int(L)]

    return {
        'alive_at_end': alive,
        'next_lineage_id': int(next_lineage_id),
        'n_births': int(n_births),
        'n_imm': int(n_imm),
        'n_deaths': int(n_deaths),
        'sojourn': float(sojourn),
        'parent_of_new': parent_of_new,
        'death_lineages': death_lineages,
        'birth_time_in_edge': birth_time_in_edge,
        'lifetime_in_edge': lifetime_in_edge,
    }


# ---------------------------------------------------------------------------
# Step 1: TKF91 BDI Gillespie on a phylogenetic tree.
# ---------------------------------------------------------------------------


def _stable_node_id(node):
    """Stable string label for a TreeNode.  Uses .name when available,
    falls back to the python id() in hex (stable within one process)."""
    if node.name:
        return str(node.name)
    return f'<int_{id(node) & 0xffff:04x}>'


def simulate_tkf91_tree_gillespie(np_rng, tree_root, ins_rate, del_rate,
                                    root_length=None, root_length_mean=10):
    """Top-level TKF91 Gillespie BDI on a phylogenetic tree.

    Each edge runs an independent BDI Gillespie trajectory; lineage ids
    are assigned globally so columns of the output MSA correspond to
    lineages.  All counts and integrated quantities are recorded per
    edge AND globally so the simulator's output can be used as
    ground-truth sufficient statistics for testing.

    Args:
        np_rng:           numpy.random.RandomState.
        tree_root:        TreeNode (from parse_newick).  Branch lengths
                          are read from .branch_length on each non-root
                          node; the root's branch length is ignored.
        ins_rate:         λ.
        del_rate:         μ.
        root_length:      if not None, deterministic number of root
                          lineages.  If None, sampled Poisson with mean
                          ``root_length_mean``.  In either case the root
                          lineages are id'd 0 ... L0 - 1.
        root_length_mean: Poisson mean for the root chain length (used
                          only when root_length is None).

    Returns:
        dict with keys:
            'leaf_presence'  : dict[leaf_name -> set[int]] — surviving
                                lineages at each leaf.
            'msa'            : dict[leaf_name -> np.ndarray(int8)] —
                                indicator MSA, columns ordered by global
                                lineage id (creation order).
            'lineage_ids'    : list[int] — sorted union of leaf-surviving
                                lineages; column j of the MSA is
                                lineage_ids[j].
            'edge_stats'     : list[dict] — one entry per non-root edge,
                                in tree-preorder.  Each entry has keys:
                                  'parent_label', 'child_label', 't',
                                  'n_alive_start', 'n_alive_end',
                                  'n_births', 'n_imm', 'n_deaths',
                                  'sojourn', 'parent_of_new',
                                  'death_lineages'.
            'total_stats'    : dict — sums across all edges, keys:
                                  'n_births', 'n_imm', 'n_deaths',
                                  'sojourn', 'T_imm' (= Σ_e t_e, total
                                   immortal-link observation time).
                                Plus blind-period tallies that quantify
                                the genealogical-correction bias (cf.
                                tkf.tex):
                                  'survivor_birth_blind_time' — Σ over
                                    GLOBALLY-SURVIVING new lineages of
                                    their birth time τ within the edge
                                    where they appear.  Multiplied by an
                                    inner rate this gives the expected
                                    number of inner events the
                                    approximation OVER-counts on its
                                    first edge by treating the lineage
                                    as alive for the whole edge.
                                  'dead_lineage_lifetime' — Σ over
                                    NON-SURVIVING lineages of their
                                    total lifetime across all edges
                                    they touched.  Multiplied by an
                                    inner rate this gives the expected
                                    number of inner events the approxi-
                                    mation MISSES by skipping non-
                                    surviving lineages entirely.
                                  'survivor_total_lifetime' — Σ over
                                    surviving lineages of their total
                                    lifetime.  Equals Σ_e (t_e ·
                                    n_surviving_at_edge_e) up to inter-
                                    edge bookkeeping.
            'lineage_at_node': dict[node_label -> list[int]] — lineages
                                alive at the END of the edge ABOVE this
                                node (or at the root for the root node).
            'root_lineage_ids': list[int] — initial lineages at the root.
    """
    # 1. Sample root chain.
    if root_length is None:
        L0 = int(max(0, np_rng.poisson(root_length_mean)))
    else:
        L0 = int(root_length)
    root_lineages = list(range(L0))
    next_lineage = L0

    lineage_at_node = {}
    edge_stats = []
    total = {'n_births': 0, 'n_imm': 0, 'n_deaths': 0,
             'sojourn': 0.0, 'T_imm': 0.0}

    root_label = _stable_node_id(tree_root)
    lineage_at_node[root_label] = list(root_lineages)

    # 2. Walk the tree in preorder, running Gillespie BDI on each edge.
    #    We use an explicit recursive walk over (parent, child) pairs to
    #    keep edge order deterministic and tree-preorder.
    def walk(parent):
        for child in parent.children:
            t = float(child.branch_length or 0.0)
            parent_label = _stable_node_id(parent)
            child_label = _stable_node_id(child)
            alive_start = lineage_at_node[parent_label]

            r = gillespie_bdi_edge(
                np_rng, alive_start, ins_rate, del_rate, t,
                next_lineage_id=_walk_state['next_lineage'])
            _walk_state['next_lineage'] = r['next_lineage_id']

            lineage_at_node[child_label] = r['alive_at_end']
            edge_stats.append({
                'parent_label': parent_label,
                'child_label': child_label,
                't': t,
                'n_alive_start': len(alive_start),
                'n_alive_end': len(r['alive_at_end']),
                'n_births': r['n_births'],
                'n_imm': r['n_imm'],
                'n_deaths': r['n_deaths'],
                'sojourn': r['sojourn'],
                'parent_of_new': dict(r['parent_of_new']),
                'death_lineages': list(r['death_lineages']),
                'birth_time_in_edge': dict(r['birth_time_in_edge']),
                'lifetime_in_edge': dict(r['lifetime_in_edge']),
            })
            total['n_births'] += r['n_births']
            total['n_imm'] += r['n_imm']
            total['n_deaths'] += r['n_deaths']
            total['sojourn'] += r['sojourn']
            total['T_imm'] += t
            walk(child)

    _walk_state = {'next_lineage': next_lineage}
    walk(tree_root)

    # 3. Assemble the indicator MSA.
    leaves = tree_root.leaves()
    leaf_presence = {}
    for leaf in leaves:
        label = _stable_node_id(leaf)
        leaf_presence[leaf.name or label] = set(int(x) for x in lineage_at_node[label])

    all_lineages = sorted(set().union(*leaf_presence.values())) \
                     if leaf_presence else []
    msa = {}
    for leaf in leaves:
        leaf_key = leaf.name or _stable_node_id(leaf)
        present = leaf_presence[leaf_key]
        row = np.array([1 if lin in present else 0 for lin in all_lineages],
                        dtype=np.int8)
        msa[leaf_key] = row

    # 4. Blind-period tally (genealogical-correction bias).
    #    A lineage is "globally surviving" iff it appears at any leaf.
    #    A NEW lineage's "blind interval" on its birth edge is its τ_birth
    #    — the inner-sim approximation places the lineage as alive for the
    #    full edge, but reality only has it from τ_birth onwards.
    #    A NON-SURVIVING lineage contributes its entire across-edge
    #    lifetime to the "missed" tally — the inner sim never runs for it.
    surviving = set(all_lineages)
    survivor_birth_blind_time = 0.0
    dead_lineage_lifetime = 0.0
    survivor_total_lifetime = 0.0
    for e in edge_stats:
        for L, t_b in e['birth_time_in_edge'].items():
            if L in surviving:
                survivor_birth_blind_time += float(t_b)
        for L, lt in e['lifetime_in_edge'].items():
            if L in surviving:
                survivor_total_lifetime += float(lt)
            else:
                dead_lineage_lifetime += float(lt)
    total['survivor_birth_blind_time'] = float(survivor_birth_blind_time)
    total['dead_lineage_lifetime'] = float(dead_lineage_lifetime)
    total['survivor_total_lifetime'] = float(survivor_total_lifetime)

    return {
        'leaf_presence': leaf_presence,
        'msa': msa,
        'lineage_ids': all_lineages,
        'edge_stats': edge_stats,
        'total_stats': total,
        'lineage_at_node': lineage_at_node,
        'root_lineage_ids': list(root_lineages),
    }


# ---------------------------------------------------------------------------
# Step 2 (decorator): build per-lineage subtrees from a labeled run.
# ---------------------------------------------------------------------------


def _present_at_subtree(node, lineage_id, lineage_at_node):
    """Returns True iff `lineage_id` is alive at `node` OR at any
    descendant of `node`."""
    if node.is_leaf:
        return lineage_id in lineage_at_node[_stable_node_id(node)]
    if lineage_id in lineage_at_node[_stable_node_id(node)]:
        return True
    return any(_present_at_subtree(c, lineage_id, lineage_at_node)
                for c in node.children)


def extract_lineage_subtree(tree_root, lineage_id, lineage_at_node,
                              edge_stats):
    """Build a pruned subtree containing the lineage's lifetime.

    The subtree mirrors the original topology, but:
      - leaves are restricted to leaves where the lineage is present;
      - internal nodes with only one surviving descendant are
        contracted (branch lengths summed) so the result is a proper
        binary-or-higher tree;
      - branch lengths reflect the ORIGINAL edge lengths.  We do NOT
        attempt to compute "lifetime within edge" — for the canonical
        MixDom decorator the lineage is assumed alive for the entirety
        of any edge on its surviving path (which holds when births occur
        only at the START of an edge in the labeled simulator; the
        Gillespie simulator above has continuous-time births so this
        is an APPROXIMATION).  TODO Step 3a: refine to use the
        per-edge first-appearance / last-disappearance times so the
        subtree edge lengths are exact.

    Args:
        tree_root:        original TreeNode.
        lineage_id:       int — the lineage to extract.
        lineage_at_node:  dict from simulate_tkf91_tree_gillespie.
        edge_stats:       list[dict] from simulate_tkf91_tree_gillespie
                          (currently unused; included so future Step 3a
                          can derive exact lifetime intervals).

    Returns:
        TreeNode (new) or None if the lineage doesn't survive to any leaf.
    """
    if not _present_at_subtree(tree_root, lineage_id, lineage_at_node):
        return None

    def build(node):
        kept_children = []
        for c in node.children:
            if _present_at_subtree(c, lineage_id, lineage_at_node):
                kept_children.append(build(c))
        if node.is_leaf:
            new = TreeNode(name=node.name,
                            branch_length=float(node.branch_length or 0.0))
            return new
        if len(kept_children) == 1:
            # Contract: pull the single child's subtree up, summing the
            # branch length on top of this edge.
            sub = kept_children[0]
            sub.branch_length = float((node.branch_length or 0.0) +
                                       (sub.branch_length or 0.0))
            sub.parent = None
            return sub
        new = TreeNode(name=node.name,
                        branch_length=float(node.branch_length or 0.0))
        for c in kept_children:
            new.add_child(c)
        return new

    sub = build(tree_root)
    sub.parent = None
    sub.branch_length = 0.0  # root has no branch above
    return sub


# ---------------------------------------------------------------------------
# Step 3a: substitution Gillespie + fragchar chain sampler.
# ---------------------------------------------------------------------------


def gillespie_subst_edge(np_rng, root_state, Q, t):
    """Continuous-time substitution Gillespie on one edge.

    Args:
        np_rng:     numpy.random.RandomState.
        root_state: int — starting residue at the parent end of the edge.
        Q:          (A, A) rate matrix.  Diagonals are -Σ off-diagonals;
                    off-diagonals must be non-negative.
        t:          edge length.

    Returns:
        dict with keys:
            'end_state'           : int — residue at child end.
            'n_substitutions'     : int — total off-diagonal events.
            'sojourn_per_state'   : (A,) float — total time spent in each
                                                 state.  Sums to t.
            'subst_counts'        : (A, A) int — number of i→j events
                                                 (off-diagonal only).
    """
    state = int(root_state)
    s = 0.0
    n_subs = 0
    A = Q.shape[0]
    soj = np.zeros(A)
    counts = np.zeros((A, A), dtype=np.int64)
    while True:
        rate_out = -float(Q[state, state])
        if rate_out < 1e-30:
            soj[state] += t - s
            break
        dt = np_rng.exponential(1.0 / rate_out)
        if s + dt > t:
            soj[state] += t - s
            break
        soj[state] += dt
        s += dt
        row = np.maximum(np.asarray(Q[state, :], dtype=np.float64), 0.0)
        row[state] = 0.0
        total = row.sum()
        if total < 1e-30:
            soj[state] += t - s
            break
        probs = row / total
        next_state = int(np_rng.choice(A, p=probs))
        counts[state, next_state] += 1
        state = next_state
        n_subs += 1
    return {
        'end_state': int(state),
        'n_substitutions': int(n_subs),
        'sojourn_per_state': soj,
        'subst_counts': counts,
    }


def simulate_subst_on_subtree(np_rng, root_residue, Q, subtree_root):
    """Continuous-time substitution Gillespie on a full subtree (preorder).

    Args:
        np_rng:        numpy.random.RandomState.
        root_residue:  int — residue at the subtree root.
        Q:             (A, A) rate matrix.
        subtree_root:  TreeNode.

    Returns:
        dict with keys:
            'leaf_residues'           : dict[leaf_label -> int].
            'total_n_substitutions'   : int.
            'total_sojourn_per_state' : (A,) float — sums to the subtree's
                                                     total branch length.
            'total_subst_counts'      : (A, A) int — off-diagonal i→j sums.
            'edge_stats'              : list[dict] in preorder, each with
                                        ('parent_label', 'child_label',
                                        't', 'n_substitutions', 'subst_counts',
                                        'sojourn_per_state').
    """
    A = Q.shape[0]
    state_at = {id(subtree_root): int(root_residue)}
    total_n_subs = 0
    total_soj = np.zeros(A)
    total_counts = np.zeros((A, A), dtype=np.int64)
    edge_stats = []

    def walk(node):
        nonlocal total_n_subs
        for c in node.children:
            t = float(c.branch_length or 0.0)
            r_par = state_at[id(node)]
            r = gillespie_subst_edge(np_rng, r_par, Q, t)
            state_at[id(c)] = r['end_state']
            total_n_subs += r['n_substitutions']
            total_soj[:] += r['sojourn_per_state']
            total_counts[:] += r['subst_counts']
            edge_stats.append({
                'parent_label': _stable_node_id(node),
                'child_label': _stable_node_id(c),
                't': t,
                'n_substitutions': int(r['n_substitutions']),
                'subst_counts': r['subst_counts'].copy(),
                'sojourn_per_state': r['sojourn_per_state'].copy(),
            })
            walk(c)
    walk(subtree_root)

    leaves = subtree_root.leaves()
    leaf_residues = {(leaf.name or _stable_node_id(leaf)): state_at[id(leaf)]
                       for leaf in leaves}
    return {
        'leaf_residues': leaf_residues,
        'total_n_substitutions': int(total_n_subs),
        'total_sojourn_per_state': total_soj,
        'total_subst_counts': total_counts,
        'edge_stats': edge_stats,
    }


def sample_fragchar_chain(np_rng, fragdist, ext, max_length=10000):
    """Sample a fragchar chain (sequence of class indices) at a fragment
    subtree's root.

    Models the fragment-level Markov chain:
        - first fragchar's class drawn from `fragdist`;
        - with probability ``ext`` the chain extends by one more fragchar
          whose class is drawn fresh from `fragdist`;
        - with probability ``1 - ext`` the chain stops.

    The length L has shifted-geometric distribution
        P(L=k) = ext^(k-1) · (1 - ext),  k ≥ 1,
    so E[L] = 1 / (1 - ext).  Class indices are i.i.d. given the chain.

    Args:
        np_rng:     numpy.random.RandomState.
        fragdist:   (C,) array — class probabilities, ≥ 0 summing to 1.
        ext:        scalar in [0, 1) — extension probability.
        max_length: hard safety cap (not normally hit).

    Returns:
        list[int] of length ≥ 1.
    """
    fragdist = np.asarray(fragdist, dtype=np.float64)
    C = len(fragdist)
    if not (0.0 <= ext < 1.0):
        raise ValueError(f'ext must be in [0, 1); got {ext}')
    chain = [int(np_rng.choice(C, p=fragdist))]
    while len(chain) < max_length:
        if np_rng.random() < ext:
            chain.append(int(np_rng.choice(C, p=fragdist)))
        else:
            break
    return chain


def extract_lineage_subtrees(tree_root, run):
    """Convenience: build a subtree per surviving lineage.

    Args:
        tree_root: TreeNode (original full tree).
        run:       dict returned by simulate_tkf91_tree_gillespie.

    Returns:
        list[(lineage_id, subtree_root)] in lineage-id order.  Lineages
        that survive to no leaves are omitted.
    """
    out = []
    for lin in run['lineage_ids']:
        sub = extract_lineage_subtree(
            tree_root, lin, run['lineage_at_node'], run['edge_stats'])
        if sub is not None:
            out.append((lin, sub))
    return out


# ---------------------------------------------------------------------------
# Step 3b: full hierarchical MixDom Gillespie orchestrator.
# ---------------------------------------------------------------------------


def _assign_categories(np_rng, run, weights):
    """Assign a categorical index to every surviving lineage.

    Root lineages and newly-born lineages each get an i.i.d. draw from
    `weights`.  No information propagates from parent to child of an
    insertion event (each insertion is a fresh categorical draw); root
    lineages and newborns are treated identically.

    Args:
        np_rng:  numpy.random.RandomState.
        run:     dict from simulate_tkf91_tree_gillespie.
        weights: (K,) categorical probabilities (≥ 0, sum to 1).

    Returns:
        dict[lineage_id -> int] for every globally-surviving lineage.
    """
    weights = np.asarray(weights, dtype=np.float64)
    K = len(weights)
    surviving = set(run['lineage_ids'])
    assignment = {}
    for L in run['root_lineage_ids']:
        if L in surviving:
            assignment[int(L)] = int(np_rng.choice(K, p=weights))
    for e in run['edge_stats']:
        for L in e['birth_time_in_edge']:
            if L in surviving and L not in assignment:
                assignment[int(L)] = int(np_rng.choice(K, p=weights))
    return assignment


def simulate_mixdom_tree_gillespie(np_rng, tree_root, params,
                                     root_dom_count=None,
                                     root_dom_count_mean=3):
    """Full hierarchical MixDom2 Gillespie simulator (3 layers).

    Layer 0  — Top-level TKF91 over DOMAIN links on the full tree.
                Each surviving domain lineage is assigned a domain index
                ``d ∈ {0..D-1}`` (i.i.d. ``params['dom_weights']``).

    Layer 1  — For each surviving domain lineage, run TKF91 again on its
                pruned subtree (using the parent's edge lengths — see
                ``extract_lineage_subtree`` for the genealogical-correction
                approximation).  Use ``params['dom_ins'][d]`` /
                ``params['dom_del'][d]`` as the per-domain BDI rates.
                Each surviving fragment lineage is assigned a fragment
                index ``f ∈ {0..F-1}`` (i.i.d. ``params['frag_weights'][d]``).

    Layer 2  — For each surviving fragment lineage:
                  • sample a fragchar chain at the SUBTREE ROOT: length
                    geometric(1 - ext[d, f]); each fragchar's class
                    drawn i.i.d. from ``params['class_pis'][d, f]``.
                  • for each fragchar, sample its root residue from the
                    class-conditioned stationary distribution
                    ``params['class_pi'][κ]``, then run substitution
                    Gillespie under ``params['class_Q'][κ]`` on the
                    fragment subtree.

    Args:
        np_rng:               numpy.random.RandomState.
        tree_root:            TreeNode for the full phylogenetic tree.
        params:               dict with keys
                                'main_ins', 'main_del',
                                'dom_weights' (D,),
                                'dom_ins' (D,), 'dom_del' (D,),
                                'frag_weights' (D, F),
                                'ext_rates' (D, F) — extension probability
                                  for the fragchar chain.
                                'class_pis' (D, F, C) — fragdist per (d, f).
                                'class_Q' (C, A, A) — rate matrices.
                                'class_pi' (C, A) — stationary distribution
                                  for sampling root residues.
        root_dom_count:       deterministic count of root domain lineages.
                              If None, sampled Poisson(root_dom_count_mean).
        root_dom_count_mean:  Poisson mean for root domains.

    Returns:
        dict with keys:
            'outer_run'              : layer-0 simulator output.
            'dom_assignments'        : dict[lineage_id -> domain_idx].
            'inner_runs'             : dict[lineage_id -> layer-1 sim out].
            'inner_subtrees'         : dict[lineage_id -> TreeNode].
            'frag_assignments'       : dict[(outer_lineage_id,
                                              inner_lineage_id) -> frag_idx].
            'fragment_subtrees'      : dict[(o_lin, i_lin) -> TreeNode].
            'fragchar_chains'        : dict[(o_lin, i_lin) -> list[int]]
                                       (class indices per fragchar).
            'fragchar_root_residues' : dict[(o_lin, i_lin) -> list[int]]
                                       (root residues per fragchar).
            'subst_runs'             : dict[(o_lin, i_lin, frag_pos) ->
                                       layer-2 substitution sim output].
            'total_stats'            : dict — aggregated counts at every
                                       level for parameter recovery tests.
    """
    main_ins = float(params['main_ins'])
    main_del = float(params['main_del'])
    dom_weights = np.asarray(params['dom_weights'], dtype=np.float64)
    dom_ins = np.asarray(params['dom_ins'], dtype=np.float64)
    dom_del = np.asarray(params['dom_del'], dtype=np.float64)
    frag_weights = np.asarray(params['frag_weights'], dtype=np.float64)
    ext_rates = np.asarray(params['ext_rates'], dtype=np.float64)
    class_pis = np.asarray(params['class_pis'], dtype=np.float64)
    class_Q = np.asarray(params['class_Q'], dtype=np.float64)
    class_pi = np.asarray(params['class_pi'], dtype=np.float64)
    D = dom_weights.shape[0]
    F = frag_weights.shape[1]
    C = class_pis.shape[2]
    A = class_pi.shape[1]

    # ----- Layer 0: top-level TKF91 over domains. -----
    outer_run = simulate_tkf91_tree_gillespie(
        np_rng, tree_root, main_ins, main_del,
        root_length=root_dom_count, root_length_mean=root_dom_count_mean)
    dom_assignments = _assign_categories(np_rng, outer_run, dom_weights)

    # ----- Layer 1: per-domain TKF91 over fragments. -----
    inner_runs = {}
    inner_subtrees = {}
    frag_assignments = {}
    fragment_subtrees = {}
    for o_lin, sub in extract_lineage_subtrees(tree_root, outer_run):
        d = dom_assignments[o_lin]
        inner_subtrees[o_lin] = sub
        inner = simulate_tkf91_tree_gillespie(
            np_rng, sub, float(dom_ins[d]), float(dom_del[d]),
            root_length_mean=2.0)  # small per-domain root counts.
        inner_runs[o_lin] = inner
        f_assign = _assign_categories(np_rng, inner, frag_weights[d])
        for i_lin, f in f_assign.items():
            frag_assignments[(o_lin, i_lin)] = int(f)
        # Build per-fragment subtrees once.
        for i_lin, frag_sub in extract_lineage_subtrees(sub, inner):
            fragment_subtrees[(o_lin, i_lin)] = frag_sub

    # ----- Layer 2: fragchar chain + per-class substitution. -----
    fragchar_chains = {}
    fragchar_root_residues = {}
    subst_runs = {}
    # Per-class aggregates for parameter-recovery tests.
    class_total_subst = np.zeros((C, A, A), dtype=np.int64)
    class_total_soj = np.zeros((C, A))
    class_fragchar_counts = np.zeros(C, dtype=np.int64)
    fragchar_chain_lengths_per_df = {}
    for (o_lin, i_lin), frag_sub in fragment_subtrees.items():
        d = dom_assignments[o_lin]
        f = frag_assignments[(o_lin, i_lin)]
        fragdist = class_pis[d, f]
        ext = float(ext_rates[d, f])
        chain = sample_fragchar_chain(np_rng, fragdist, ext)
        fragchar_chains[(o_lin, i_lin)] = chain
        fragchar_chain_lengths_per_df.setdefault((d, f), []).append(len(chain))
        root_residues = []
        for pos, kappa in enumerate(chain):
            class_fragchar_counts[kappa] += 1
            r0 = int(np_rng.choice(A, p=class_pi[kappa]))
            root_residues.append(r0)
            r = simulate_subst_on_subtree(np_rng, r0, class_Q[kappa], frag_sub)
            subst_runs[(o_lin, i_lin, pos)] = {
                'class_idx': int(kappa),
                'root_residue': r0,
                'leaf_residues': r['leaf_residues'],
                'total_n_substitutions': r['total_n_substitutions'],
                'total_subst_counts': r['total_subst_counts'],
                'total_sojourn_per_state': r['total_sojourn_per_state'],
            }
            class_total_subst[kappa] += r['total_subst_counts']
            class_total_soj[kappa] += r['total_sojourn_per_state']
        fragchar_root_residues[(o_lin, i_lin)] = root_residues

    # ----- Aggregate everything for downstream tests. -----
    # Inner BDI per domain.
    inner_bdi = {d: {'B': 0, 'D': 0, 'S': 0.0, 'T_imm': 0.0,
                      'survivor_birth_blind_time': 0.0,
                      'dead_lineage_lifetime': 0.0}
                 for d in range(D)}
    for o_lin, inner in inner_runs.items():
        d = dom_assignments[o_lin]
        t = inner['total_stats']
        inner_bdi[d]['B'] += t['n_births'] + t['n_imm']
        inner_bdi[d]['D'] += t['n_deaths']
        inner_bdi[d]['S'] += t['sojourn']
        inner_bdi[d]['T_imm'] += t['T_imm']
        inner_bdi[d]['survivor_birth_blind_time'] += t['survivor_birth_blind_time']
        inner_bdi[d]['dead_lineage_lifetime'] += t['dead_lineage_lifetime']

    # Outer BDI summary (already computed in outer_run; expose for
    # convenience).
    outer_bdi = {
        'B': outer_run['total_stats']['n_births']
              + outer_run['total_stats']['n_imm'],
        'D': outer_run['total_stats']['n_deaths'],
        'S': outer_run['total_stats']['sojourn'],
        'T_imm': outer_run['total_stats']['T_imm'],
        'survivor_birth_blind_time':
            outer_run['total_stats']['survivor_birth_blind_time'],
        'dead_lineage_lifetime':
            outer_run['total_stats']['dead_lineage_lifetime'],
    }

    return {
        'outer_run': outer_run,
        'dom_assignments': dom_assignments,
        'inner_runs': inner_runs,
        'inner_subtrees': inner_subtrees,
        'frag_assignments': frag_assignments,
        'fragment_subtrees': fragment_subtrees,
        'fragchar_chains': fragchar_chains,
        'fragchar_root_residues': fragchar_root_residues,
        'subst_runs': subst_runs,
        'fragchar_chain_lengths_per_df': fragchar_chain_lengths_per_df,
        'total_stats': {
            'outer_bdi': outer_bdi,
            'inner_bdi_per_dom': inner_bdi,
            'class_total_subst': class_total_subst,
            'class_total_sojourn': class_total_soj,
            'class_fragchar_counts': class_fragchar_counts,
        },
    }


# ---------------------------------------------------------------------------
# Step C: TKF92-on-tree Gillespie (with chain-order tracking).
#
# TKF92 = TKF91 BDI on FRAGMENTS + geometric-length fragchar chain per
# fragment.  Sites within a fragment are static (no internal BDI).  The
# site-level chi distribution at any tree node is exactly tau92 because
# of this 2-level composition (Knudsen & Miyamoto 2003, Thorne et al
# 1992).
#
# Chain order is required so that ``oracle_n_trans_for_branch`` (which
# walks the parent's chain in chain order) can be applied at the SITE
# level.  We add a chain-aware Gillespie that inserts new lineages right
# after their parent in the chain (and at the chain start for immortal
# births), and keeps dead lineages in the chain (marked dead) so that
# inserts that fall in a deleted lineage's slot still anchor to it for
# child_after computation.
# ---------------------------------------------------------------------------


def gillespie_bdi_edge_chain_aware(np_rng, parent_chain, ins_rate, del_rate, t,
                                      next_lineage_id):
    """Like ``gillespie_bdi_edge`` but maintains chain order.

    Args:
        parent_chain:    list[int] of lineages alive at the parent end of
                         this edge, in chain order.  Must have no
                         duplicates.
        ins_rate:        λ.
        del_rate:        μ.
        t:               edge length.
        next_lineage_id: first available lineage id for births on this
                         edge.

    Returns:
        dict with all keys of ``gillespie_bdi_edge`` plus:
            'chain_full'      : list[int] — every lineage that touched
                                 this edge (alive at start, born during,
                                 etc), in chain order.  Dead lineages
                                 are KEPT so that inserts in their slot
                                 can anchor to them via child_after.
            'is_alive_at_end' : dict[lineage_id -> bool].
            'chain_at_end'    : list[int] — lineages alive at end, in
                                chain order (= chain_full filtered by
                                is_alive_at_end).
            'parent_set'      : set[int] — lineages that were in the
                                input parent_chain (used downstream to
                                determine match vs insert).
    """
    parent_chain = [int(L) for L in parent_chain]
    parent_set = set(parent_chain)
    if len(parent_set) != len(parent_chain):
        raise ValueError('parent_chain has duplicate lineage ids')

    chain_full = list(parent_chain)
    is_alive = {int(L): True for L in parent_chain}
    s = 0.0
    n_births = 0
    n_imm = 0
    n_deaths = 0
    sojourn = 0.0
    parent_of_new = {}
    death_lineages = []
    birth_time = {int(L): 0.0 for L in parent_chain}
    birth_time_in_edge = {}
    lifetime_in_edge = {}

    while True:
        n = sum(1 for L in chain_full if is_alive[L])
        birth_rate = ins_rate * n
        death_rate = del_rate * n
        imm_rate = ins_rate
        total_rate = birth_rate + death_rate + imm_rate

        if total_rate < 1e-30:
            sojourn += n * (t - s)
            s = t
            break
        dt = np_rng.exponential(1.0 / total_rate)
        if s + dt > t:
            sojourn += n * (t - s)
            s = t
            break

        sojourn += n * dt
        s += dt

        u = np_rng.random() * total_rate
        if u < birth_rate:
            # Pick parent uniformly from currently-alive lineages.
            alive_indices = [i for i, L in enumerate(chain_full)
                                if is_alive[L]]
            parent_idx = alive_indices[np_rng.randint(0, len(alive_indices))]
            parent_id = chain_full[parent_idx]
            new_id = next_lineage_id
            chain_full.insert(parent_idx + 1, new_id)
            is_alive[new_id] = True
            parent_of_new[new_id] = int(parent_id)
            birth_time[new_id] = s
            birth_time_in_edge[new_id] = s
            next_lineage_id += 1
            n_births += 1
        elif u < birth_rate + death_rate:
            alive_indices = [i for i, L in enumerate(chain_full)
                                if is_alive[L]]
            victim_idx = alive_indices[np_rng.randint(0, len(alive_indices))]
            victim = chain_full[victim_idx]
            is_alive[victim] = False
            lifetime_in_edge[victim] = s - birth_time[victim]
            death_lineages.append(int(victim))
            n_deaths += 1
        else:
            new_id = next_lineage_id
            chain_full.insert(0, new_id)
            is_alive[new_id] = True
            parent_of_new[new_id] = -1
            birth_time[new_id] = s
            birth_time_in_edge[new_id] = s
            next_lineage_id += 1
            n_imm += 1

    for L in chain_full:
        if is_alive[L]:
            lifetime_in_edge[L] = t - birth_time[L]

    chain_at_end = [L for L in chain_full if is_alive[L]]

    return {
        'alive_at_end': chain_at_end,
        'chain_at_end': chain_at_end,
        'chain_full': chain_full,
        'is_alive_at_end': dict(is_alive),
        'parent_set': set(parent_set),
        'next_lineage_id': int(next_lineage_id),
        'n_births': int(n_births),
        'n_imm': int(n_imm),
        'n_deaths': int(n_deaths),
        'sojourn': float(sojourn),
        'parent_of_new': parent_of_new,
        'death_lineages': death_lineages,
        'birth_time_in_edge': birth_time_in_edge,
        'lifetime_in_edge': lifetime_in_edge,
    }


def _child_after_from_chain_full(chain_full, is_alive, parent_set):
    """Walk chain_full in order; for each ALIVE lineage L, set
    child_after[L] = the most recent lineage in chain_full[:idx_of_L]
    that was in parent_set (or -1 if none).

    Matched lineages (alive AND in parent_set) get child_after[L] = L.
    """
    child_after = {}
    last_parent_id = -1
    for L in chain_full:
        if L in parent_set:
            last_parent_id = int(L)
            if is_alive.get(L, False):
                child_after[int(L)] = int(L)
        else:
            if is_alive.get(L, False):
                child_after[int(L)] = int(last_parent_id)
    return child_after


def simulate_tkf91_tree_gillespie_chain_aware(np_rng, tree_root, ins_rate,
                                                  del_rate, root_chain=None,
                                                  root_length=None,
                                                  root_length_mean=10,
                                                  root_at_stationary=False):
    """TKF91 BDI Gillespie on a tree, with chain order tracked at every
    node.

    Same return shape as ``simulate_tkf91_tree_gillespie`` plus:
        'chain_at_node' : dict[node_label -> list[int]] in chain order.
        'edge_chain_data': list[dict] in tree-preorder, one per edge,
                            with keys
                              'parent_label', 'child_label', 't',
                              'parent_chain'    (list[int] in order),
                              'child_chain'     (list[int] in order),
                              'child_after'     (dict[L -> parent_id|-1]),
                              'chain_full'      (full chain incl dead).

    The chain at each node depends on the path from the root, so this
    function walks the tree in preorder and propagates chain order down.
    Births during an edge are inserted right after their parent; deaths
    are kept in chain_full but filtered out for chain_at_end.
    """
    if root_chain is not None:
        root_lineages = list(int(L) for L in root_chain)
        next_lineage = max(root_lineages) + 1 if root_lineages else 0
    else:
        if root_at_stationary:
            L0 = sample_tkf91_stationary_length(np_rng, ins_rate, del_rate)
        elif root_length is None:
            L0 = int(max(0, np_rng.poisson(root_length_mean)))
        else:
            L0 = int(root_length)
        root_lineages = list(range(L0))
        next_lineage = L0

    chain_at_node = {}
    edge_chain_data = []
    edge_stats = []
    total = {'n_births': 0, 'n_imm': 0, 'n_deaths': 0,
             'sojourn': 0.0, 'T_imm': 0.0}

    root_label = _stable_node_id(tree_root)
    chain_at_node[root_label] = list(root_lineages)
    lineage_at_node = {root_label: list(root_lineages)}

    _walk_state = {'next_lineage': next_lineage}

    def walk(parent):
        for child in parent.children:
            t = float(child.branch_length or 0.0)
            parent_label = _stable_node_id(parent)
            child_label = _stable_node_id(child)
            parent_chain = chain_at_node[parent_label]

            r = gillespie_bdi_edge_chain_aware(
                np_rng, parent_chain, ins_rate, del_rate, t,
                next_lineage_id=_walk_state['next_lineage'])
            _walk_state['next_lineage'] = r['next_lineage_id']

            chain_at_node[child_label] = r['chain_at_end']
            lineage_at_node[child_label] = r['chain_at_end']
            child_after = _child_after_from_chain_full(
                r['chain_full'], r['is_alive_at_end'], r['parent_set'])
            edge_chain_data.append({
                'parent_label': parent_label,
                'child_label': child_label,
                't': t,
                'parent_chain': list(parent_chain),
                'child_chain': list(r['chain_at_end']),
                'child_after': child_after,
                'chain_full': list(r['chain_full']),
            })
            edge_stats.append({
                'parent_label': parent_label,
                'child_label': child_label,
                't': t,
                'n_alive_start': len(parent_chain),
                'n_alive_end': len(r['chain_at_end']),
                'n_births': r['n_births'],
                'n_imm': r['n_imm'],
                'n_deaths': r['n_deaths'],
                'sojourn': r['sojourn'],
                'parent_of_new': dict(r['parent_of_new']),
                'death_lineages': list(r['death_lineages']),
                'birth_time_in_edge': dict(r['birth_time_in_edge']),
                'lifetime_in_edge': dict(r['lifetime_in_edge']),
            })
            total['n_births'] += r['n_births']
            total['n_imm'] += r['n_imm']
            total['n_deaths'] += r['n_deaths']
            total['sojourn'] += r['sojourn']
            total['T_imm'] += t
            walk(child)
    walk(tree_root)

    leaves = tree_root.leaves()
    leaf_presence = {}
    for leaf in leaves:
        label = _stable_node_id(leaf)
        leaf_presence[leaf.name or label] = set(int(x)
                                                  for x in chain_at_node[label])

    all_lineages = sorted(set().union(*leaf_presence.values())) \
                     if leaf_presence else []
    msa = {}
    for leaf in leaves:
        leaf_key = leaf.name or _stable_node_id(leaf)
        present = leaf_presence[leaf_key]
        row = np.array([1 if lin in present else 0 for lin in all_lineages],
                        dtype=np.int8)
        msa[leaf_key] = row

    surviving = set(all_lineages)
    survivor_birth_blind_time = 0.0
    dead_lineage_lifetime = 0.0
    survivor_total_lifetime = 0.0
    for e in edge_stats:
        for L, t_b in e['birth_time_in_edge'].items():
            if L in surviving:
                survivor_birth_blind_time += float(t_b)
        for L, lt in e['lifetime_in_edge'].items():
            if L in surviving:
                survivor_total_lifetime += float(lt)
            else:
                dead_lineage_lifetime += float(lt)
    total['survivor_birth_blind_time'] = float(survivor_birth_blind_time)
    total['dead_lineage_lifetime'] = float(dead_lineage_lifetime)
    total['survivor_total_lifetime'] = float(survivor_total_lifetime)

    return {
        'leaf_presence': leaf_presence,
        'msa': msa,
        'lineage_ids': all_lineages,
        'edge_stats': edge_stats,
        'edge_chain_data': edge_chain_data,
        'chain_at_node': chain_at_node,
        'total_stats': total,
        'lineage_at_node': lineage_at_node,
        'root_lineage_ids': list(root_lineages),
    }


def simulate_tkf92_tree_gillespie(np_rng, tree_root, ins_rate, del_rate, ext,
                                     root_length=None, root_length_mean=5,
                                     root_at_stationary=False):
    """TKF92 Gillespie on a tree.

    Composes the standard 2-level TKF92 generative process:

      1. Fragments evolve TKF91 BDI Gillespie at rates (ins_rate, del_rate).
      2. Each fragment has a length L drawn from shifted-geometric(1 - ext)
         (P(L=k) = ext^(k-1) (1-ext) for k ≥ 1) at the time of birth.
         Sites within a fragment are static (no internal BDI), only their
         residues evolve via substitution downstream of this simulator.

    The site-level chi distribution at any tree node is then exactly
    tau92 = ext + (1 - ext) tau91 (the textbook TKF92 derivation).

    Args:
        np_rng:           numpy.random.RandomState.
        tree_root:        TreeNode.
        ins_rate:         λ (fragment-level birth rate, per the standard
                          TKF92 derivation).
        del_rate:         μ (fragment-level death rate).
        ext:              extension probability ∈ [0, 1).  ext = 0 reduces
                          to TKF91 (every fragment has length 1).
        root_length:      deterministic # of root fragments (else Poisson).
        root_length_mean: Poisson mean for root fragments (used iff
                          root_length is None).

    Returns:
        dict with:
            'fragment_run'        : output of
                                     simulate_tkf91_tree_gillespie_chain_aware.
            'fragment_lengths'    : dict[fragment_lineage -> int] for
                                     EVERY fragment that ever appeared
                                     (alive at any point on any edge).
            'site_chain_at_node'  : dict[node_label -> list[(frag, pos)]]
                                     in chain order, expanded to the
                                     SITE level.
            'edge_site_alignments': list[dict] in tree-preorder with keys
                                     'parent_label', 'child_label', 't',
                                     'parent_site_chain', 'child_site_chain',
                                     'child_site_after'.  Each site lineage
                                     is the tuple (frag_lineage_id, site_pos).
            'total_stats'         : aggregated counts (BDI + survivor /
                                     dead-lineage tallies + ext events).
    """
    if not (0.0 <= ext < 1.0):
        raise ValueError(f'ext must be in [0, 1); got {ext}')

    fragment_run = simulate_tkf91_tree_gillespie_chain_aware(
        np_rng, tree_root, ins_rate, del_rate,
        root_length=root_length, root_length_mean=root_length_mean,
        root_at_stationary=root_at_stationary)

    # Collect the universe of fragment ids that touched any edge — these
    # need lengths assigned even if they don't survive to a leaf.
    all_fragment_ids = set()
    for e in fragment_run['edge_chain_data']:
        all_fragment_ids.update(e['chain_full'])
    all_fragment_ids.update(fragment_run['root_lineage_ids'])

    # Sample fragment length per fragment lineage: shifted-geometric.
    fragment_lengths = {}
    for L in sorted(all_fragment_ids):
        n_extend = 0
        # Draw geometrically how many EXT events fired at fragment birth.
        while np_rng.random() < ext:
            n_extend += 1
        fragment_lengths[int(L)] = 1 + n_extend
    total_n_extensions = sum(v - 1 for v in fragment_lengths.values())
    total_n_fragments = len(fragment_lengths)

    # Build the SITE chain at every node by expanding each fragment in
    # chain order to (fragment_id, site_pos) tuples.  Two fragments
    # have non-overlapping site lineages even if they share a position
    # because the lineage is identified by (frag_id, site_pos).
    site_chain_at_node = {}
    for node_label, frag_chain in fragment_run['chain_at_node'].items():
        sites = []
        for f in frag_chain:
            for p in range(fragment_lengths[int(f)]):
                sites.append((int(f), int(p)))
        site_chain_at_node[node_label] = sites

    # Build site-level alignments per edge.
    edge_site_alignments = []
    for e in fragment_run['edge_chain_data']:
        # Parent and child site chains.
        parent_site_chain = []
        for f in e['parent_chain']:
            for p in range(fragment_lengths[int(f)]):
                parent_site_chain.append((int(f), int(p)))
        child_site_chain = []
        for f in e['child_chain']:
            for p in range(fragment_lengths[int(f)]):
                child_site_chain.append((int(f), int(p)))
        # Build the FULL site chain (alive + dead) by walking chain_full
        # and expanding fragments to their full sites.  child_after at the
        # site level: walk full site chain in order and propagate
        # last_parent_site_id (sites whose fragment was in parent's
        # fragment chain).
        parent_frag_set = set(int(f) for f in e['parent_chain'])
        full_site_chain = []
        for f in e['chain_full']:
            for p in range(fragment_lengths[int(f)]):
                full_site_chain.append((int(f), int(p)))
        # is_alive at site level = is_alive at fragment level
        # (sites within a fragment share the fragment's life).
        # We don't have per-fragment is_alive in edge_chain_data, so derive:
        alive_frag_set = set(int(f) for f in e['child_chain'])
        site_after = {}
        last_parent_site = -1
        for site in full_site_chain:
            f = site[0]
            if f in parent_frag_set:
                last_parent_site = site  # tuple
                if f in alive_frag_set:
                    site_after[site] = site
            else:
                if f in alive_frag_set:
                    site_after[site] = last_parent_site
        edge_site_alignments.append({
            'parent_label': e['parent_label'],
            'child_label': e['child_label'],
            't': e['t'],
            'parent_site_chain': parent_site_chain,
            'child_site_chain': child_site_chain,
            'child_site_after': site_after,
        })

    total_stats = dict(fragment_run['total_stats'])
    total_stats['n_extension_events'] = int(total_n_extensions)
    total_stats['n_fragments_total'] = int(total_n_fragments)

    return {
        'fragment_run': fragment_run,
        'fragment_lengths': fragment_lengths,
        'site_chain_at_node': site_chain_at_node,
        'edge_site_alignments': edge_site_alignments,
        'total_stats': total_stats,
    }


def gillespie_pair_tkf92(np_rng, ins_rate, del_rate, ext, t, Q, pi,
                            root_at_stationary=True, fixed_n_fragments=None):
    """Simulate a single TKF92 pair (anc, desc) at branch length ``t`` via
    fully-labeled Gillespie events.

    This is the single-edge analogue of ``simulate_tkf92_tree_gillespie``,
    composed end-to-end with per-fragchar substitution Gillespie so the
    output includes residue sequences.

    Steps:
      1. Sample ancestor: ``sample_tkf92_stationary_root`` for fragment
         count + per-fragment lengths.  Residues at each fragchar drawn
         i.i.d. from ``pi``.
      2. Fragment-level Gillespie BDI on a single edge: births either
         extend the parent's fragment (prob ``ext``) or start a new
         fragment (prob ``1 - ext``); deaths remove fragments.
      3. For each surviving fragment, the descendant inherits the
         fragchar count from the parent (matched fragments keep their
         residues with substitution Gillespie).  Newly-born fragments
         (during the edge) get a fresh geometric(1-ext) fragchar count
         and root residues drawn from ``pi``.
      4. Per-fragchar substitution Gillespie under ``Q`` from parent
         residue to child residue.

    Args:
        np_rng:               numpy.random.RandomState.
        ins_rate, del_rate:   fragment-level (= TKF92) λ, μ.
        ext:                  fragment-extension probability.
        t:                    branch length.
        Q:                    (A, A) substitution rate matrix.
        pi:                   (A,) stationary distribution.
        root_at_stationary:   if True, draw # fragments from
                              TKF91-stationary geometric(1 - λ/μ);
                              else require ``fixed_n_fragments``.
        fixed_n_fragments:    if not None, use this many root fragments.

    Returns:
        dict with:
          'anc_residues':     (Lx,) np.int32.
          'desc_residues':    (Ly,) np.int32.
          't':                float.
          'labeled':          dict — ground truth from Gillespie:
            'n_fragment_births_per_lineage':  int — non-imm body births.
            'n_fragment_imm':                 int — immortal-link inserts.
            'n_fragment_deaths':              int — fragment deletions.
            'fragment_sojourn':               float — ∫ |alive(s)| ds.
            'n_extensions':                    int — births decided ext.
            'n_new_fragment_decisions':        int — births decided new-frag.
            'n_substitutions':                 int — total subst events.
            'subst_counts':                    (A, A) int — i→j off-diag.
            'subst_sojourn_per_state':         (A,) float — per-state dwell.
            'fragment_lengths_anc':            list[int] — per-frag lengths
                                               at ancestor end.
            'fragment_lengths_desc':           list[int] — per-frag lengths
                                               at descendant end.
    """
    import numpy as _np
    if root_at_stationary:
        n_frag, lengths = sample_tkf92_stationary_root(np_rng, ins_rate,
                                                          del_rate, ext)
    else:
        if fixed_n_fragments is None:
            raise ValueError('Must give fixed_n_fragments when '
                              'root_at_stationary=False')
        n_frag = int(fixed_n_fragments)
        lengths = []
        for _ in range(n_frag):
            n_extend = 0
            while np_rng.random() < ext:
                n_extend += 1
            lengths.append(1 + n_extend)

    pi_np = _np.asarray(pi, dtype=_np.float64)
    A = pi_np.shape[0]

    # Ancestor residues per (frag, pos).
    anc_res = []  # flat list of residues, in chain order
    frag_residues_anc = {}  # fragment_id -> list[int] residues
    for f_id in range(n_frag):
        L_f = lengths[f_id]
        fragres = list(int(x) for x in
                          np_rng.choice(A, size=L_f, p=pi_np))
        frag_residues_anc[f_id] = fragres
        anc_res.extend(fragres)
    anc_residues = _np.asarray(anc_res, dtype=_np.int32)

    # Fragment-level chain-aware Gillespie BDI on the edge.  We track
    # the extension/new-fragment decision per birth ourselves.
    parent_chain_ids = list(range(n_frag))
    is_alive = {fid: True for fid in parent_chain_ids}
    parent_set = set(parent_chain_ids)
    chain_full = list(parent_chain_ids)
    s = 0.0
    n_births = 0
    n_imm = 0
    n_deaths = 0
    n_extensions = 0
    n_new_fragment_decisions = 0
    sojourn = 0.0
    next_lineage_id = n_frag
    next_fragment_decisions = {}  # new_lineage_id -> 'extend' or 'new'
    parent_of_new = {}
    fragment_lengths = dict.fromkeys(parent_chain_ids)
    for fid in parent_chain_ids:
        fragment_lengths[fid] = lengths[fid]

    while True:
        n = sum(1 for fid in chain_full if is_alive[fid])
        birth_rate = ins_rate * n
        death_rate = del_rate * n
        imm_rate = ins_rate
        total_rate = birth_rate + death_rate + imm_rate
        if total_rate < 1e-30:
            sojourn += n * (t - s)
            s = t
            break
        dt = np_rng.exponential(1.0 / total_rate)
        if s + dt > t:
            sojourn += n * (t - s)
            s = t
            break
        sojourn += n * dt
        s += dt

        u = np_rng.random() * total_rate
        if u < birth_rate:
            alive_indices = [i for i, fid in enumerate(chain_full)
                                if is_alive[fid]]
            parent_idx = alive_indices[np_rng.randint(0, len(alive_indices))]
            parent_id = chain_full[parent_idx]
            new_id = next_lineage_id
            chain_full.insert(parent_idx + 1, new_id)
            is_alive[new_id] = True
            parent_of_new[new_id] = int(parent_id)
            # Per-birth extend / new-frag decision.
            if np_rng.random() < ext:
                # Extend parent's fragment: same fragment_id (= parent's).
                # In TKF92, extension means the new lineage shares the
                # parent's fragment-length geometric; for the simulator
                # purposes we track this as a separate event count.
                n_extensions += 1
                next_fragment_decisions[new_id] = 'extend'
            else:
                n_new_fragment_decisions += 1
                next_fragment_decisions[new_id] = 'new'
            # Newly-born fragments (new_id) need their own length.
            n_extend = 0
            while np_rng.random() < ext:
                n_extend += 1
            fragment_lengths[new_id] = 1 + n_extend
            next_lineage_id += 1
            n_births += 1
        elif u < birth_rate + death_rate:
            alive_indices = [i for i, fid in enumerate(chain_full)
                                if is_alive[fid]]
            victim_idx = alive_indices[np_rng.randint(0, len(alive_indices))]
            victim = chain_full[victim_idx]
            is_alive[victim] = False
            n_deaths += 1
        else:
            new_id = next_lineage_id
            chain_full.insert(0, new_id)
            is_alive[new_id] = True
            parent_of_new[new_id] = -1
            n_extend = 0
            while np_rng.random() < ext:
                n_extend += 1
            fragment_lengths[new_id] = 1 + n_extend
            next_lineage_id += 1
            n_imm += 1

    # Build descendant residues.  For every fragment in chain_full that's
    # alive at edge end:
    #   - if it was in parent_set: matched — substitute parent residues.
    #   - if it's new: insert — sample fresh root residues from pi.
    # For deleted fragments (in parent_set but not alive): contribute no
    # descendant residues.
    desc_res = []
    n_subs_total = 0
    subst_counts = _np.zeros((A, A), dtype=_np.int64)
    subst_sojourn_per_state = _np.zeros(A)
    fragment_lengths_desc = {}
    for fid in chain_full:
        if not is_alive[fid]:
            continue
        if fid in parent_set:
            # Matched fragment: substitute each parent residue.
            for parent_res in frag_residues_anc[fid]:
                end_state, n_subs, soj_per, counts = _subst_run_one(
                    np_rng, parent_res, Q, t)
                desc_res.append(end_state)
                n_subs_total += n_subs
                subst_counts += counts
                subst_sojourn_per_state += soj_per
            fragment_lengths_desc[fid] = len(frag_residues_anc[fid])
        else:
            # Inserted fragment (born during edge): fresh residues.
            L_f = fragment_lengths[fid]
            for _ in range(L_f):
                # Root residue from pi; no substitution (just emitted).
                root_res = int(np_rng.choice(A, p=pi_np))
                desc_res.append(root_res)
            fragment_lengths_desc[fid] = L_f

    return {
        'anc_residues': anc_residues,
        'desc_residues': _np.asarray(desc_res, dtype=_np.int32),
        't': float(t),
        'labeled': {
            'n_fragment_births_per_lineage': int(n_births),
            'n_fragment_imm': int(n_imm),
            'n_fragment_deaths': int(n_deaths),
            'fragment_sojourn': float(sojourn),
            'n_extensions': int(n_extensions),
            'n_new_fragment_decisions': int(n_new_fragment_decisions),
            'n_substitutions': int(n_subs_total),
            'subst_counts': subst_counts,
            'subst_sojourn_per_state': subst_sojourn_per_state,
            'fragment_lengths_anc': [fragment_lengths[fid]
                                       for fid in parent_chain_ids],
            'fragment_lengths_desc': fragment_lengths_desc,
        },
    }


def _subst_run_one(np_rng, root_state, Q, t):
    """Helper: continuous-time substitution Gillespie on a single residue
    over time t.  Returns (end_state, n_subs, sojourn_per_state, counts)."""
    state = int(root_state)
    s = 0.0
    n_subs = 0
    A = Q.shape[0]
    soj = np.zeros(A)
    counts = np.zeros((A, A), dtype=np.int64)
    while True:
        rate_out = -float(Q[state, state])
        if rate_out < 1e-30:
            soj[state] += t - s
            break
        dt = np_rng.exponential(1.0 / rate_out)
        if s + dt > t:
            soj[state] += t - s
            break
        soj[state] += dt
        s += dt
        row = np.maximum(np.asarray(Q[state, :], dtype=np.float64), 0.0)
        row[state] = 0.0
        total = row.sum()
        if total < 1e-30:
            soj[state] += t - s
            break
        probs = row / total
        next_state = int(np_rng.choice(A, p=probs))
        counts[state, next_state] += 1
        state = next_state
        n_subs += 1
    return state, n_subs, soj, counts


def n_trans_from_site_alignment(parent_site_chain, child_site_chain,
                                  child_site_after):
    """Build a 5×5 WFST n_trans matrix from a per-edge site-level alignment.

    Args:
        parent_site_chain: list of site lineage ids (any hashable) at parent
                           end, in chain order.
        child_site_chain:  list of site lineage ids at child end, in chain
                           order.
        child_site_after:  dict[site_lineage -> parent_site_lineage_or_-1]
                           for each child site lineage.  Matched sites have
                           after = themselves; inserts have after = the
                           preceding parent site (or -1).

    Returns:
        (5, 5) np.float64 — n_trans matrix in WFST state semantics
        (S=0, M=1, I=2, D=3, E=4), as built by ``oracle_n_trans_for_branch``
        but operating on site lineages.
    """
    from ..core.params import S as _S, M as _M, I as _I, D as _D, E as _E
    parent_set = set(parent_site_chain)
    matched = set()
    inserts_after = {}
    for L in child_site_chain:
        if L in parent_set:
            matched.add(L)
        else:
            after = child_site_after[L]
            inserts_after.setdefault(after, []).append(L)
    events = []
    for L in inserts_after.get(-1, []):
        events.append('I')
    for p in parent_site_chain:
        events.append('M' if p in matched else 'D')
        for L in inserts_after.get(p, []):
            events.append('I')
    state_idx = {'S': _S, 'M': _M, 'I': _I, 'D': _D, 'E': _E}
    n = np.zeros((5, 5), dtype=np.float64)
    prev = _S
    for ev in events:
        nxt = state_idx[ev]
        n[prev, nxt] += 1.0
        prev = nxt
    n[prev, _E] += 1.0
    return n
