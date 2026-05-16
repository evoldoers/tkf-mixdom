"""Composition operations for SCFGs and WPTTs.

Implements three composition operations described in ParseTreeTransducers.md:

1. compose_generator_transducer: SCFG (generator) x WPTT (transducer)
   The generator produces tokens that the transducer consumes as input.

2. compose_transducer_recognizer: WPTT (transducer) x Recognizer
   The transducer produces tokens that the recognizer consumes as input.

3. intersect_siblings: Two recognizers at sibling branches
   Both must accept the same input tokens simultaneously.

Synchronization rules (for generator x transducer):
  - If transducer is not in ready state, generator cannot update
  - If generator rule doesn't produce a token, transducer cannot update
  - When generator produces a token, transducer must consume it as input
  - Epsilon and bifurcation must happen simultaneously
  - Composite rule weight = product of component weights
"""

import numpy as np
from ..grammar.scfg import WCFG, Production, build_grammar
from ..distill.wptt import (
    WPTT, WPTTRule, is_ready_state, IDX_START as WPTT_START,
    N_WPTT_STATES, decode_wptt_state, ALN_R,
)
from ..models.order1_scfg import N_SINGLET_STATES, IDX_START as SCFG_START
from ..models.rna_grammar import decode_terminal, N_TOTAL_TERMINALS


def compose_generator_transducer(generator, transducer):
    """Compose an order-1 singlet SCFG (generator) with a WPTT (transducer).

    Creates a composite grammar where nonterminals are (scfg_state, wptt_state)
    tuples. The generator produces tokens, the transducer consumes them as
    input and produces output tokens.

    Args:
        generator: WCFG (the order-1 singlet SCFG)
        transducer: WPTT object

    Returns:
        WCFG: composite grammar with generator.n_nonterminals * transducer.n_states
              nonterminals. Terminals are the transducer's output tokens (0-23).
    """
    n_gen = generator.n_nonterminals
    n_trans = transducer.n_states

    # Build composite nonterminal names and index mapping
    def comp_idx(gen_nt, trans_st):
        return gen_nt * n_trans + trans_st

    def comp_name(gen_nt, trans_st):
        return f'({generator.nonterminals[gen_nt]},{transducer.state_names[trans_st]})'

    n_comp = n_gen * n_trans
    comp_names = []
    for g in range(n_gen):
        for t in range(n_trans):
            comp_names.append(comp_name(g, t))

    rules = []
    start = comp_idx(generator.start, WPTT_START)

    # Pre-index transducer rules by (src, input_token) for fast lookup
    trans_by_input = {}  # (src, input_token) -> list of rules
    trans_insert_ready = {}  # src -> list of insert/ready rules
    trans_eps = {}  # src -> list of epsilon rules
    trans_bif = {}  # src -> list of bifurcation rules
    for trans_st in range(n_trans):
        for rule in transducer.rules_for(trans_st):
            if rule.input_token is not None:
                trans_by_input.setdefault(
                    (trans_st, rule.input_token), []).append(rule)
            elif rule.rule_type in ('insert', 'ready'):
                trans_insert_ready.setdefault(trans_st, []).append(rule)
            elif rule.rule_type == 'epsilon':
                trans_eps.setdefault(trans_st, []).append(rule)
            elif rule.rule_type == 'bifurcation':
                trans_bif.setdefault(trans_st, []).append(rule)

    from ..models.rna_grammar import left_terminal, right_terminal, pair_terminal

    for gen_nt in range(n_gen):
        for trans_st in range(n_trans):
            src = comp_idx(gen_nt, trans_st)
            src_name = comp_names[src]
            trans_ready = is_ready_state(trans_st)

            for gen_prod in generator.productions_for(gen_nt):
                # Case 1: Generator emits a token (right-linear, left-linear, lr-linear)
                # Transducer must be ready and must consume the token
                if gen_prod.is_right_linear or gen_prod.is_left_linear or gen_prod.is_lr_linear:
                    if not trans_ready:
                        continue  # transducer must be ready to accept input

                    # Determine the input token(s) from the generator production
                    if gen_prod.is_right_linear:
                        input_tok = gen_prod.rhs[0]
                        gen_next = gen_prod.rhs[1]
                    elif gen_prod.is_left_linear:
                        gen_next = gen_prod.rhs[0]
                        input_tok = gen_prod.rhs[1]
                    else:  # lr_linear
                        input_tok_l = gen_prod.rhs[0]
                        gen_next = gen_prod.rhs[1]
                        input_tok_r = gen_prod.rhs[2]
                        lt_type, lt_nucs = decode_terminal(input_tok_l)
                        rt_type, rt_nucs = decode_terminal(input_tok_r)
                        if lt_type == 'L' and rt_type == 'R':
                            compound_tok = pair_terminal(lt_nucs[0], rt_nucs[0])
                        else:
                            continue
                        input_tok = compound_tok

                    # Find matching transducer rules via pre-built index
                    for trans_rule in trans_by_input.get(
                            (trans_st, input_tok), []):

                        w = gen_prod.weight * trans_rule.weight
                        if w < 1e-300:
                            continue

                        trans_next = trans_rule.dst
                        comp_next_name = comp_names[comp_idx(gen_next, trans_next)]

                        if trans_rule.rule_type == 'match':
                            # Match: transducer also produces an output token
                            out_tok = trans_rule.output_token
                            out_type, out_nucs = decode_terminal(out_tok)

                            if gen_prod.is_right_linear:
                                # Composite: (A,T) -> out_tok (B,T')
                                rules.append((src_name,
                                              [(out_tok, 'T'), (comp_next_name, 'N')],
                                              w))
                            elif gen_prod.is_left_linear:
                                rules.append((src_name,
                                              [(comp_next_name, 'N'), (out_tok, 'T')],
                                              w))
                            elif gen_prod.is_lr_linear:
                                # LR match: output also has LR structure
                                if out_type == 'LR':
                                    out_l = left_terminal(out_nucs[0])
                                    out_r = right_terminal(out_nucs[1])
                                    rules.append((src_name,
                                                  [(out_l, 'T'), (comp_next_name, 'N'), (out_r, 'T')],
                                                  w))
                                else:
                                    # Shouldn't happen: LR input matched with non-LR output
                                    continue

                        elif trans_rule.rule_type == 'delete':
                            # Delete: transducer consumes input but produces nothing
                            # The generator's token is consumed but not emitted
                            if gen_prod.is_right_linear:
                                # Effectively becomes unary: (A,T) -> (B,T')
                                rules.append((src_name,
                                              [(comp_next_name, 'N')],
                                              w))
                            elif gen_prod.is_left_linear:
                                rules.append((src_name,
                                              [(comp_next_name, 'N')],
                                              w))
                            elif gen_prod.is_lr_linear:
                                rules.append((src_name,
                                              [(comp_next_name, 'N')],
                                              w))

                # Case 1b: Transducer inserts (generator doesn't advance)
                # Only when transducer is NOT ready
                if not trans_ready:
                    for trans_rule in trans_insert_ready.get(trans_st, []):
                        if trans_rule.rule_type == 'insert':
                            out_tok = trans_rule.output_token
                            trans_next = trans_rule.dst
                            w = trans_rule.weight  # only transducer weight
                            if w < 1e-300:
                                continue
                            comp_next_name = comp_names[comp_idx(gen_nt, trans_next)]
                            out_type, out_nucs = decode_terminal(out_tok)
                            if out_type == 'L':
                                rules.append((src_name,
                                              [(out_tok, 'T'), (comp_next_name, 'N')],
                                              w))
                            elif out_type == 'R':
                                rules.append((src_name,
                                              [(comp_next_name, 'N'), (out_tok, 'T')],
                                              w))
                        elif trans_rule.rule_type == 'ready':
                            trans_next = trans_rule.dst
                            w = trans_rule.weight
                            if w < 1e-300:
                                continue
                            comp_next_name = comp_names[comp_idx(gen_nt, trans_next)]
                            rules.append((src_name,
                                          [(comp_next_name, 'N')],
                                          w))

                # Case 2: Generator epsilon — both must go to epsilon simultaneously
                if gen_prod.is_empty and trans_ready:
                    for trans_rule in trans_eps.get(trans_st, []):
                        w = gen_prod.weight * trans_rule.weight
                        if w < 1e-300:
                            continue
                        rules.append((src_name, [], w))

                # Case 3: Generator bifurcation — transducer must bifurcate too
                if gen_prod.is_binary and trans_ready:
                    gen_left, gen_right = gen_prod.rhs
                    for trans_rule in trans_bif.get(trans_st, []):
                        w = gen_prod.weight * trans_rule.weight
                        if w < 1e-300:
                            continue
                        left_name = comp_names[comp_idx(gen_left, trans_rule.dst_left)]
                        right_name = comp_names[comp_idx(gen_right, trans_rule.dst_right)]
                        rules.append((src_name,
                                      [(left_name, 'N'), (right_name, 'N')],
                                      w))

                # Case 4: Generator unary — transducer doesn't advance
                if gen_prod.is_unary:
                    gen_next = gen_prod.rhs[0]
                    comp_next_name = comp_names[comp_idx(gen_next, trans_st)]
                    rules.append((src_name,
                                  [(comp_next_name, 'N')],
                                  gen_prod.weight))

                # Case 5: Generator terminal (bare terminal, no nonterminal)
                if gen_prod.is_terminal:
                    if not trans_ready:
                        continue
                    input_tok = gen_prod.rhs[0]
                    for trans_rule in trans_by_input.get(
                            (trans_st, input_tok), []):
                        if trans_rule.rule_type == 'match':
                            w = gen_prod.weight * trans_rule.weight
                            if w < 1e-300:
                                continue
                            # Terminal match: no generator continuation
                            # This would leave a dangling transducer state
                            # which needs to go to epsilon. Skip for now.
                            pass
                        elif trans_rule.rule_type == 'delete':
                            w = gen_prod.weight * trans_rule.weight
                            if w < 1e-300:
                                continue
                            # Terminal delete: consumed and discarded
                            # Also dangling. Skip.
                            pass

    if not rules:
        # Return empty grammar
        return WCFG(comp_names, N_TOTAL_TERMINALS, [], start)

    return build_grammar(comp_names, N_TOTAL_TERMINALS, rules, start=comp_names[start])
