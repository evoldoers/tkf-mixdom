#!/usr/bin/env python3
"""Compute and report expected statistics from a MixDom model checkpoint.

Pure numpy, no JAX needed.
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Report expected statistics from a MixDom checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        default="params/best/bw_d3f2_pfam100_20iter.npz",
        help="Path to .npz checkpoint file",
    )
    args = parser.parse_args()

    d = np.load(args.checkpoint, allow_pickle=True)

    main_ins = float(d["main_ins"])
    main_del = float(d["main_del"])
    dom_ins = d["dom_ins"]        # (n_dom,)
    dom_del = d["dom_del"]        # (n_dom,)
    dom_weights = d["dom_weights"]  # (n_dom,)
    ext_rates = d["ext_rates"]    # (n_dom, n_frag)
    frag_weights = d["frag_weights"]  # (n_dom, n_frag)
    t_rep = float(d["t_rep"]) if "t_rep" in d else None
    em_iter = int(d["em_iter"]) if "em_iter" in d else None

    n_dom = len(dom_ins)
    n_frag = ext_rates.shape[1]

    # Top-level TKF91
    kappa_top = main_ins / main_del
    e_links = kappa_top / (1 - kappa_top)

    print("=" * 70)
    print(f"MixDom Expected Statistics: {args.checkpoint}")
    if em_iter is not None:
        print(f"  EM iterations: {em_iter}")
    if t_rep is not None:
        print(f"  Representative time: {t_rep:.6f}")
    print("=" * 70)

    print()
    print("TOP-LEVEL TKF91 (domain birth-death)")
    print(f"  lambda (main_ins) = {main_ins:.8f}")
    print(f"  mu     (main_del) = {main_del:.8f}")
    print(f"  kappa = lambda/mu = {kappa_top:.6f}")
    print(f"  E[#domains]       = kappa/(1-kappa) = {e_links:.2f}")
    print()

    print("-" * 70)
    print(f"DOMAIN TYPES ({n_dom} domains, {n_frag} fragments each)")
    print("-" * 70)

    e_len_by_dom = np.zeros(n_dom)

    for dd in range(n_dom):
        kappa_d = dom_ins[dd] / dom_del[dd]
        # In TKF92: the domain generates a geometric number of "links"
        # (each link picks a fragment type and extends geometrically).
        # E[#links in domain] = kappa_d / (1 - kappa_d)
        e_domain_links = kappa_d / (1 - kappa_d)
        p_domain_empty = 1 - kappa_d  # P(zero links)

        print(f"\n  Domain {dd}  (weight = {dom_weights[dd]:.4f})")
        print(f"    lambda_d = {dom_ins[dd]:.8f}")
        print(f"    mu_d     = {dom_del[dd]:.8f}")
        print(f"    kappa_d  = {kappa_d:.6f}")
        print(f"    P(domain empty)  = {p_domain_empty:.6f}")
        print(f"    E[#frag links]   = {e_domain_links:.2f}")
        print()

        # Each link picks a fragment type with probability frag_weights[dd, f]
        # and then extends geometrically with rate ext_rates[dd, f].
        # The expected number of characters per link = sum_f w_f * E[chars | frag f]
        # where E[chars | frag f] = 1 / (1 - r_f)  (geometric starting at 1)
        e_chars_per_link = 0.0
        for ff in range(n_frag):
            r = ext_rates[dd, ff]
            w = frag_weights[dd, ff]
            e_frag_len = 1.0 / (1.0 - r)
            e_chars_per_link += w * e_frag_len
            print(f"    Fragment {ff}:  weight = {w:.4f},  ext_rate = {r:.4f},  E[frag len] = {e_frag_len:.2f}")

        e_domain_len = e_domain_links * e_chars_per_link
        e_len_by_dom[dd] = e_domain_len
        print(f"    E[chars/link]    = {e_chars_per_link:.2f}")
        print(f"    E[domain length] = E[#links] * E[chars/link] = {e_domain_len:.2f}")

    # Overall expected sequence length
    e_len_per_domain = np.sum(dom_weights * e_len_by_dom)
    e_total = e_links * e_len_per_domain

    print()
    print("=" * 70)
    print("OVERALL EXPECTED SEQUENCE LENGTH")
    print(f"  E[#domains]                   = {e_links:.2f}")
    print(f"  E[domain length | domain type]:")
    for dd in range(n_dom):
        print(f"    type {dd} (w={dom_weights[dd]:.4f}): {e_len_by_dom[dd]:.2f}")
    print(f"  E[chars per domain] (weighted) = {e_len_per_domain:.2f}")
    print(f"  E[sequence length]             = {e_total:.2f}")
    print("=" * 70)

    # Summary table
    print()
    print("SUMMARY TABLE")
    print(f"{'Dom':>4} {'Weight':>8} {'kappa_d':>8} {'E[links]':>9} "
          f"{'Frag':>5} {'FW':>6} {'ExtR':>6} {'E[flen]':>8} "
          f"{'E[domlen]':>10}")
    print("-" * 80)
    for dd in range(n_dom):
        kappa_d = dom_ins[dd] / dom_del[dd]
        e_domain_links = kappa_d / (1 - kappa_d)
        for ff in range(n_frag):
            r = ext_rates[dd, ff]
            w = frag_weights[dd, ff]
            e_frag_len = 1.0 / (1.0 - r)
            if ff == 0:
                print(f"{dd:>4} {dom_weights[dd]:>8.4f} {kappa_d:>8.5f} {e_domain_links:>9.2f} "
                      f"{ff:>5} {w:>6.3f} {r:>6.4f} {e_frag_len:>8.2f} "
                      f"{e_len_by_dom[dd]:>10.2f}")
            else:
                print(f"{'':>4} {'':>8} {'':>8} {'':>9} "
                      f"{ff:>5} {w:>6.3f} {r:>6.4f} {e_frag_len:>8.2f}")


if __name__ == "__main__":
    main()
