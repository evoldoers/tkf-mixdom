"""Profile SCFG compression utilities.

Implements three compression techniques for profile SCFGs extracted from
sampled parse trees, plus a compression pipeline and grammar statistics.

Compression techniques:
1. Sub-tree sharing (hash-consing) -- non-lossy
2. Nonterminal minimization (partition refinement) -- non-lossy
3. Span merging -- controlled approximation

Reference: Section 7.8 (Profile Compression) of the paper.
"""

import numpy as np
from collections import defaultdict
from ..grammar.scfg import WCFG, Production


def _production_signature(p, nt_map=None):
    """Compute a hashable signature for a production.

    Args:
        p: Production object
        nt_map: optional dict mapping old NT index -> canonical NT index.
                If provided, RHS nonterminal indices are remapped.

    Returns:
        A hashable tuple representing the production.
    """
    if nt_map is not None:
        rhs = tuple(
            nt_map.get(r, r) if t == 'N' else r
            for r, t in zip(p.rhs, p.rhs_types)
        )
    else:
        rhs = p.rhs
    return (p.rhs_types, rhs, p.weight)


def _nonterminal_signature(grammar, nt_idx, nt_map=None):
    """Compute a hashable signature for a nonterminal based on its productions.

    Args:
        grammar: WCFG
        nt_idx: nonterminal index
        nt_map: optional dict mapping old NT index -> canonical NT index

    Returns:
        A frozenset of production signatures.
    """
    prods = grammar.productions_for(nt_idx)
    sigs = []
    for p in prods:
        sigs.append(_production_signature(p, nt_map))
    return frozenset(sigs)


def hash_cons_grammar(grammar):
    """Identify and merge nonterminals with identical production sets.

    This is hash-consing for parse sub-trees: nonterminals that have
    exactly the same set of productions (same RHS symbols, same weights)
    are merged into a single representative. This is non-lossy.

    The algorithm works bottom-up. First, it groups nonterminals by their
    production signatures. Nonterminals with identical signatures are
    merged. The process iterates until stable, since merging children
    may reveal new equivalences among parents.

    Args:
        grammar: WCFG to compress

    Returns:
        compressed_grammar: WCFG with merged nonterminals
        merge_map: dict mapping old nonterminal index -> new nonterminal index
    """
    n = grammar.n_nonterminals
    # Initialize: each NT maps to itself
    nt_map = {i: i for i in range(n)}

    for _ in range(n):
        # Compute signature for each NT under current mapping
        sig_to_nts = defaultdict(list)
        for i in range(n):
            sig = _nonterminal_signature(grammar, i, nt_map)
            sig_to_nts[sig].append(i)

        # Build new merge map: map all NTs with same signature to the
        # lowest-indexed representative
        new_nt_map = {}
        for sig, group in sig_to_nts.items():
            rep = min(nt_map[g] for g in group)
            for g in group:
                new_nt_map[g] = rep

        if new_nt_map == nt_map:
            break
        nt_map = new_nt_map

    # Determine the set of surviving nonterminals (representatives)
    surviving = sorted(set(nt_map.values()))
    reindex = {old: new for new, old in enumerate(surviving)}

    # Build final merge_map: original index -> new index
    merge_map = {i: reindex[nt_map[i]] for i in range(n)}

    # Build new nonterminal list
    new_nts = [grammar.nonterminals[s] for s in surviving]

    # Build new productions: remap LHS and RHS.
    # When multiple original NTs merge, we keep only the productions from
    # the representative NT (the one that survives), since merged NTs had
    # identical production sets by construction.
    # For productions whose LHS is the representative, we remap their RHS
    # and deduplicate. For references to merged NTs in RHS, the remapping
    # handles it. Productions from non-representative NTs are skipped.
    rep_nts = set(surviving)
    prod_weights = {}
    for p in grammar.productions:
        if p.lhs not in rep_nts:
            continue  # skip productions from non-representative NTs
        new_lhs = merge_map[p.lhs]
        new_rhs = tuple(
            merge_map[r] if t == 'N' else r
            for r, t in zip(p.rhs, p.rhs_types)
        )
        key = (new_lhs, new_rhs, p.rhs_types)
        if key in prod_weights:
            prod_weights[key] += p.weight
        else:
            prod_weights[key] = p.weight

    new_prods = []
    for (new_lhs, new_rhs, rhs_types), weight in prod_weights.items():
        new_prods.append(Production(new_lhs, new_rhs, weight, rhs_types))

    new_start = merge_map[grammar.start]
    compressed = WCFG(new_nts, grammar.n_terminals, new_prods, new_start)
    return compressed, merge_map


def minimize_grammar(grammar):
    """Minimize the number of nonterminals using partition refinement.

    Two nonterminals are equivalent if they generate the same weighted
    language. Since profile SCFGs are finite and acyclic, equivalence
    is decidable by bottom-up partition refinement (analogous to DFA
    minimization):

    1. Initialize: partition nonterminals by emission type (the frozenset
       of production rhs_types they use).
    2. Refine: two NTs remain in the same class iff for each production
       in one, there exists a production in the other with the same weight
       and rhs_types whose children are pairwise in the same partition class.
    3. Iterate until stable.

    This is non-lossy and produces the smallest grammar (fewest
    nonterminals) generating the same weighted language.

    Args:
        grammar: WCFG to minimize

    Returns:
        minimized_grammar: WCFG with minimized nonterminals
        merge_map: dict mapping old nonterminal index -> new nonterminal index
    """
    n = grammar.n_nonterminals

    # Build per-nonterminal production list
    nt_prods = defaultdict(list)
    for p in grammar.productions:
        nt_prods[p.lhs].append(p)

    # Initial partition: group by the set of production type signatures
    # (rhs_types only, ignoring specific children)
    def emission_key(nt_idx):
        prods = nt_prods[nt_idx]
        return frozenset((p.rhs_types, p.weight) for p in prods)

    # Assign initial partition IDs
    key_to_id = {}
    partition = {}
    next_id = 0
    for i in range(n):
        ek = emission_key(i)
        if ek not in key_to_id:
            key_to_id[ek] = next_id
            next_id += 1
        partition[i] = key_to_id[ek]

    # Refine until stable
    for _ in range(n):
        # Compute refined signature for each NT
        def refined_sig(nt_idx):
            prods = nt_prods[nt_idx]
            sigs = []
            for p in prods:
                child_classes = tuple(
                    partition[r] if t == 'N' else r
                    for r, t in zip(p.rhs, p.rhs_types)
                )
                sigs.append((p.rhs_types, child_classes, p.weight))
            return frozenset(sigs)

        sig_to_id = {}
        new_partition = {}
        new_next_id = 0
        for i in range(n):
            sig = refined_sig(i)
            if sig not in sig_to_id:
                sig_to_id[sig] = new_next_id
                new_next_id += 1
            new_partition[i] = sig_to_id[sig]

        if new_partition == partition:
            break
        partition = new_partition

    # Build merge_map from partition: map each NT to the smallest-indexed
    # member of its partition class
    class_to_rep = {}
    for i in range(n):
        cls = partition[i]
        if cls not in class_to_rep or i < class_to_rep[cls]:
            class_to_rep[cls] = i

    surviving = sorted(set(class_to_rep.values()))
    reindex = {old: new for new, old in enumerate(surviving)}

    merge_map = {i: reindex[class_to_rep[partition[i]]] for i in range(n)}

    # Build minimized grammar: keep only productions from representative NTs
    new_nts = [grammar.nonterminals[s] for s in surviving]
    rep_nts = set(surviving)

    prod_weights = {}
    for p in grammar.productions:
        if p.lhs not in rep_nts:
            continue
        new_lhs = merge_map[p.lhs]
        new_rhs = tuple(
            merge_map[r] if t == 'N' else r
            for r, t in zip(p.rhs, p.rhs_types)
        )
        key = (new_lhs, new_rhs, p.rhs_types)
        if key in prod_weights:
            prod_weights[key] += p.weight
        else:
            prod_weights[key] = p.weight

    new_prods = []
    for (new_lhs, new_rhs, rhs_types), weight in prod_weights.items():
        new_prods.append(Production(new_lhs, new_rhs, weight, rhs_types))

    new_start = merge_map[grammar.start]
    minimized = WCFG(new_nts, grammar.n_terminals, new_prods, new_start)
    return minimized, merge_map


def merge_spans(grammar, delta, span_info):
    """Merge nonterminals that are delta-compatible in span coordinates.

    Two nonterminals are delta-compatible if they have the same emission
    type (same set of rhs_types across their productions), and their
    spans [i1,j1] and [i2,j2] satisfy |i1-i2| <= delta and |j1-j2| <= delta.

    The merged nonterminal gets span [min(i), max(j)]. Production weights
    are averaged (uniformly, or weighted by Inside probability if available).

    This is a controlled approximation: it forgets which specific child
    alignment each sampled tree implied, replacing exact span coordinates
    with a small range. Setting delta=0 is a no-op.

    Args:
        grammar: WCFG to compress
        delta: merge tolerance (non-negative integer)
        span_info: dict mapping nonterminal index -> (min_i, max_j)

    Returns:
        merged_grammar: WCFG with merged nonterminals
        merge_map: dict mapping old nonterminal index -> new nonterminal index
        new_span_info: dict mapping new nonterminal index -> (min_i, max_j)
    """
    if delta <= 0:
        merge_map = {i: i for i in range(grammar.n_nonterminals)}
        return grammar, merge_map, dict(span_info)

    n = grammar.n_nonterminals

    # Build per-nonterminal production list for emission type
    nt_prods = defaultdict(list)
    for p in grammar.productions:
        nt_prods[p.lhs].append(p)

    def emission_type(nt_idx):
        """Frozenset of rhs_types used by this nonterminal's productions."""
        return frozenset(p.rhs_types for p in nt_prods[nt_idx])

    # Group nonterminals by emission type
    type_groups = defaultdict(list)
    for i in range(n):
        type_groups[emission_type(i)].append(i)

    # Within each emission-type group, merge nonterminals whose spans
    # are within delta of each other using a greedy clustering approach
    merge_map = {i: i for i in range(n)}

    for etype, group in type_groups.items():
        # Sort by span start
        group_with_span = [(i, span_info.get(i, (0, 0))) for i in group]
        group_with_span.sort(key=lambda x: (x[1][0], x[1][1]))

        # Greedy: for each NT, try to merge with an existing cluster
        clusters = []  # list of (representative, min_i, max_j, members)
        for nt_idx, (si, sj) in group_with_span:
            merged = False
            for ci, (rep, cmin_i, cmax_j, members) in enumerate(clusters):
                if abs(si - cmin_i) <= delta and abs(sj - cmax_j) <= delta:
                    # Can merge
                    new_min_i = min(cmin_i, si)
                    new_max_j = max(cmax_j, sj)
                    members.append(nt_idx)
                    clusters[ci] = (rep, new_min_i, new_max_j, members)
                    merge_map[nt_idx] = rep
                    merged = True
                    break
            if not merged:
                clusters.append((nt_idx, si, sj, [nt_idx]))

    # Determine surviving nonterminals and reindex
    surviving = sorted(set(merge_map.values()))
    reindex = {old: new for new, old in enumerate(surviving)}
    final_merge_map = {i: reindex[merge_map[i]] for i in range(n)}

    # Compute new span info from merged clusters
    new_span_info = {}
    # Aggregate spans for each representative
    rep_spans = defaultdict(lambda: (float('inf'), float('-inf')))
    for i in range(n):
        rep = merge_map[i]
        if i in span_info:
            si, sj = span_info[i]
            old_min, old_max = rep_spans[rep]
            rep_spans[rep] = (min(old_min, si), max(old_max, sj))

    for rep, (mi, mj) in rep_spans.items():
        new_span_info[reindex[rep]] = (mi, mj)

    # Build new grammar with merged nonterminals
    new_nts = [grammar.nonterminals[s] for s in surviving]

    # Collect and merge productions: for merged NTs, take union of
    # productions with weights averaged
    # Group productions by (new_lhs, new_rhs, rhs_types)
    prod_groups = defaultdict(list)
    for p in grammar.productions:
        new_lhs = final_merge_map[p.lhs]
        new_rhs = tuple(
            final_merge_map[r] if t == 'N' else r
            for r, t in zip(p.rhs, p.rhs_types)
        )
        key = (new_lhs, new_rhs, p.rhs_types)
        prod_groups[key].append(p.weight)

    new_prods = []
    for (new_lhs, new_rhs, rhs_types), weights in prod_groups.items():
        avg_weight = sum(weights) / len(weights)
        new_prods.append(Production(new_lhs, new_rhs, avg_weight, rhs_types))

    new_start = final_merge_map[grammar.start]
    merged = WCFG(new_nts, grammar.n_terminals, new_prods, new_start)
    return merged, final_merge_map, new_span_info


def compress_profile(grammar, span_info=None, delta=0):
    """Full compression pipeline for a profile SCFG.

    Applies the three compression techniques in order:
    1. Hash-consing (sub-tree sharing) -- non-lossy
    2. Nonterminal minimization (partition refinement) -- non-lossy
    3. Span merging (if delta > 0) -- controlled approximation

    Args:
        grammar: WCFG to compress
        span_info: optional dict mapping nonterminal index -> (min_i, max_j).
                   Required if delta > 0.
        delta: span merge tolerance. 0 means no span merging.

    Returns:
        compressed_grammar: the compressed WCFG
        total_merge_map: dict mapping original NT index -> final NT index
        compression_stats: dict with keys:
            'original_nonterminals', 'after_hashcons',
            'after_minimize', 'after_spanmerge'
    """
    n_original = grammar.n_nonterminals
    stats = {'original_nonterminals': n_original}

    # Step 1: Hash-consing
    g1, map1 = hash_cons_grammar(grammar)
    stats['after_hashcons'] = g1.n_nonterminals

    # Step 2: Minimization
    g2, map2 = minimize_grammar(g1)
    stats['after_minimize'] = g2.n_nonterminals

    # Compose merge maps so far
    composed_map = {i: map2[map1[i]] for i in range(n_original)}

    # Step 3: Span merging (if delta > 0 and span_info provided)
    if delta > 0 and span_info is not None:
        # Remap span_info to post-minimization indices
        # For each new NT, aggregate span info from all original NTs that mapped to it
        remapped_span = {}
        for orig_idx, (si, sj) in span_info.items():
            new_idx = composed_map.get(orig_idx, orig_idx)
            if new_idx in remapped_span:
                old_si, old_sj = remapped_span[new_idx]
                remapped_span[new_idx] = (min(old_si, si), max(old_sj, sj))
            else:
                remapped_span[new_idx] = (si, sj)

        g3, map3, new_span_info = merge_spans(g2, delta, remapped_span)
        stats['after_spanmerge'] = g3.n_nonterminals

        # Compose all three maps
        total_merge_map = {i: map3[composed_map[i]] for i in range(n_original)}
        return g3, total_merge_map, stats
    else:
        stats['after_spanmerge'] = g2.n_nonterminals
        return g2, composed_map, stats


def grammar_stats(grammar):
    """Compute descriptive statistics for a WCFG.

    Args:
        grammar: WCFG

    Returns:
        dict with keys:
            n_nonterminals: number of nonterminals
            n_productions: total number of productions
            n_terminals: number of distinct terminal symbols referenced
            n_binary: number of binary productions (A -> B C)
            n_unary: number of unary productions (A -> B)
            n_terminal_prods: number of terminal productions (A -> a)
            n_empty: number of empty/epsilon productions (A -> eps)
            is_regular: True if grammar is regular (right-linear)
            is_cnf: True if grammar is in Chomsky Normal Form
    """
    terminals_used = set()
    n_binary = 0
    n_unary = 0
    n_terminal_prods = 0
    n_empty = 0

    for p in grammar.productions:
        if p.is_binary:
            n_binary += 1
        elif p.is_unary:
            n_unary += 1
        elif p.is_terminal:
            n_terminal_prods += 1
            terminals_used.add(p.rhs[0])
        elif p.is_right_linear:
            terminals_used.add(p.rhs[0])
        elif p.is_empty:
            n_empty += 1

    return {
        'n_nonterminals': grammar.n_nonterminals,
        'n_productions': len(grammar.productions),
        'n_terminals': len(terminals_used),
        'n_binary': n_binary,
        'n_unary': n_unary,
        'n_terminal_prods': n_terminal_prods,
        'n_empty': n_empty,
        'is_regular': grammar.is_regular(),
        'is_cnf': grammar.is_cnf(),
    }


if __name__ == '__main__':
    from ..grammar.scfg import build_grammar, inside_logprob

    print("=== Profile SCFG Compression Tests ===\n")

    # Build a simple grammar with redundant nonterminals.
    #
    # The grammar has:
    #   S -> A B   (weight 1.0)
    #   A -> 0     (weight 0.6)
    #   A -> 1     (weight 0.4)
    #   B -> 0     (weight 0.6)   <-- identical to A
    #   B -> 1     (weight 0.4)   <-- identical to A
    #   C -> A     (weight 1.0)   <-- unary alias for A
    #   D -> 0     (weight 0.6)   <-- another copy of A
    #   D -> 1     (weight 0.4)
    #
    # Also add S -> C D (weight 0.5) and adjust S -> A B to weight 0.5
    # so that S has two binary productions pointing to equivalent NTs.

    nts = ['S', 'A', 'B', 'C', 'D']
    rules = [
        ('S', [('A', 'N'), ('B', 'N')], 0.5),
        ('S', [('C', 'N'), ('D', 'N')], 0.5),
        ('A', [(0, 'T')], 0.6),
        ('A', [(1, 'T')], 0.4),
        ('B', [(0, 'T')], 0.6),
        ('B', [(1, 'T')], 0.4),
        ('C', [('A', 'N')], 1.0),
        ('D', [(0, 'T')], 0.6),
        ('D', [(1, 'T')], 0.4),
    ]
    g = build_grammar(nts, 2, rules, start='S')

    seq = np.array([0, 1])
    logp_orig = inside_logprob(g, seq)
    print(f"Original grammar: {g.n_nonterminals} nonterminals, "
          f"{len(g.productions)} productions")
    print(f"  log P([0,1] | S) = {logp_orig:.6f}")

    # --- Test 1: Hash-consing ---
    g_hc, map_hc = hash_cons_grammar(g)
    logp_hc = inside_logprob(g_hc, seq)
    print(f"\nAfter hash-consing: {g_hc.n_nonterminals} nonterminals, "
          f"{len(g_hc.productions)} productions")
    print(f"  log P([0,1] | S) = {logp_hc:.6f}")
    assert g_hc.n_nonterminals < g.n_nonterminals, \
        "Hash-consing should reduce nonterminal count"
    assert abs(logp_hc - logp_orig) < 1e-8, \
        f"Hash-consing changed log-prob: {logp_orig} -> {logp_hc}"
    print("  PASS: hash-consing is non-lossy and reduces size")

    # --- Test 2: Minimization ---
    g_min, map_min = minimize_grammar(g)
    logp_min = inside_logprob(g_min, seq)
    print(f"\nAfter minimization: {g_min.n_nonterminals} nonterminals, "
          f"{len(g_min.productions)} productions")
    print(f"  log P([0,1] | S) = {logp_min:.6f}")
    assert g_min.n_nonterminals <= g_hc.n_nonterminals, \
        "Minimization should not increase nonterminal count vs hash-consing"
    assert abs(logp_min - logp_orig) < 1e-8, \
        f"Minimization changed log-prob: {logp_orig} -> {logp_min}"
    print("  PASS: minimization is non-lossy and reduces size")

    # --- Test 3: Span merging ---
    span_info = {0: (0, 2), 1: (0, 1), 2: (1, 2), 3: (0, 1), 4: (1, 2)}
    g_sm, map_sm, new_si = merge_spans(g, delta=1, span_info=span_info)
    logp_sm = inside_logprob(g_sm, seq)
    print(f"\nAfter span merging (delta=1): {g_sm.n_nonterminals} nonterminals, "
          f"{len(g_sm.productions)} productions")
    print(f"  log P([0,1] | S) = {logp_sm:.6f}")
    print(f"  New span info: {new_si}")
    # Span merging is approximate, so we check it does not diverge wildly
    assert abs(logp_sm - logp_orig) < 1.0, \
        f"Span merging changed log-prob too much: {logp_orig} -> {logp_sm}"
    print("  PASS: span merging produces valid grammar")

    # --- Test 4: Full pipeline ---
    g_full, map_full, stats = compress_profile(g, span_info=span_info, delta=1)
    logp_full = inside_logprob(g_full, seq)
    print(f"\nFull pipeline (delta=1): {g_full.n_nonterminals} nonterminals, "
          f"{len(g_full.productions)} productions")
    print(f"  log P([0,1] | S) = {logp_full:.6f}")
    print(f"  Stats: {stats}")
    assert stats['after_hashcons'] <= stats['original_nonterminals']
    assert stats['after_minimize'] <= stats['after_hashcons']
    assert stats['after_spanmerge'] <= stats['after_minimize']
    print("  PASS: pipeline monotonically reduces size")

    # --- Test 5: Pipeline with delta=0 (no span merging) ---
    g_exact, map_exact, stats_exact = compress_profile(g)
    logp_exact = inside_logprob(g_exact, seq)
    print(f"\nFull pipeline (delta=0): {g_exact.n_nonterminals} nonterminals, "
          f"{len(g_exact.productions)} productions")
    print(f"  log P([0,1] | S) = {logp_exact:.6f}")
    assert abs(logp_exact - logp_orig) < 1e-8, \
        f"Exact pipeline changed log-prob: {logp_orig} -> {logp_exact}"
    print("  PASS: exact pipeline preserves log-prob")

    # --- Test 6: Grammar statistics ---
    s = grammar_stats(g)
    print(f"\nGrammar stats (original): {s}")
    assert s['n_nonterminals'] == 5
    assert s['n_binary'] == 2
    assert s['n_unary'] == 1
    assert s['n_terminal_prods'] == 6
    assert s['n_empty'] == 0
    assert s['is_regular'] is False  # has binary rules
    assert s['is_cnf'] is False  # has unary rule
    print("  PASS: grammar stats correct")

    # --- Test 7: Already-minimal grammar ---
    nts2 = ['S', 'X']
    rules2 = [
        ('S', [('X', 'N'), ('X', 'N')], 1.0),
        ('X', [(0, 'T')], 0.5),
        ('X', [(1, 'T')], 0.5),
    ]
    g2 = build_grammar(nts2, 2, rules2, start='S')
    g2c, _, stats2 = compress_profile(g2)
    assert g2c.n_nonterminals == g2.n_nonterminals, \
        "Already-minimal grammar should not change"
    logp2_orig = inside_logprob(g2, seq)
    logp2_comp = inside_logprob(g2c, seq)
    assert abs(logp2_orig - logp2_comp) < 1e-8
    print(f"\nAlready-minimal grammar: no change ({stats2})")
    print("  PASS")

    print("\n=== All tests passed ===")
