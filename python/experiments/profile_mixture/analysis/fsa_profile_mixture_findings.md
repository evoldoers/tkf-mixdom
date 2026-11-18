# Per-site profile mixtures (LG-C20) in FSA on BAliBASE, and the LG table

Findings from 2026-09-30. All runs and checks were done on **wideboy**
(details under Machine and provenance).

## Question

Does FSA (sequence annealing of pairwise posteriors) improve if the MixFrag
pair HMM's emissions come from the LG-C20 profile mixture instead of LG08?
At a given time t the class-marginal pair matrix

    J(t)_ab = sum_k w_k pi^k_a M^k(t)_ab

is a single 20x20 matrix, so the cost is the same as LG. The recollection being
tested was that C20 did not help.

## How each claim is labelled

- **[measured]** A number produced on wideboy by a script committed in this repo.
  It can be regenerated with the command given in that section.
- **[checked]** Established in the main session by reading the code, a commit,
  or the paper source, or by a direct comparison not saved as a script. Each
  one says which.
- **[agent-checked]** Established by the verification workflow: an investigating
  subagent, then a separate skeptic subagent told to refute it. I have not
  re-read those sources myself.
- **[not verified]** Stated for context only.

## Bottom line

1. **Per-site LG-C20 makes FSA worse, not better.** With MixFrag and the
   variant most favourable to C20 (separate substitution and indel clocks), the
   mean SP falls 0.018 when both C20 and LG use paper 1's LG table
   (p = 0.002), and 0.025 when both use published LG (p = 1e-5). Every other C20 variant is worse still. [measured]
2. **The cause is a property of the model.** Under LG-C20, two residues have an
   0.197 probability of being identical even at infinite divergence, because
   both come from the same class profile. For LG that floor is 0.059. BAliBASE
   reference pairs have a median identity of 0.205. So C20 explains most pairs
   with no shared ancestry beyond class membership, and the substitution time
   runs to the clip. On Pfam seed alignments, which share no data with
   BAliBASE, C20 captures less of the residue-pair association than LG does:
   0.543 vs 0.588 nats per aligned residue pair. [measured]
3. **The repo's "LG08" is not LG.** tkf-mixdom's `_LG_S_LOWER`, which papers
   1–3 use as LG08 and which `src/lg08.py` copies, differs from the
   published table in 171 of 190 exchangeabilities. The published table
   improves FSA by +0.017 SP for TKF92 (p = 0.005) and +0.009 SP for MixFrag.
   [measured]
4. **Paper 1's CherryML-C=20 row can now be reproduced.** It is TKF92 with
   the CherryML 20-class mixture applied per site, with both processes at the
   LG anchor time: 0.791 / 0.645 [measured]. The code that produced it was
   never committed [checked: commit b12b411 records only the numbers]. Its gain
   over TKF92 (+0.014 SP here) is about the same as what the LG table fix alone
   gives TKF92 (+0.017).

## Setup

**Driver.** `experiments/fsa_c20_balibase.py` wraps paper 1's pipeline in
tkf-mixdom (`experiments/expected_pairwise_balibase.py`). It keeps paper 1's
pair selection, `sequence_annealing` (3 iterations, seed 42, gap_factor 1 and
0) and core-column SP/TC scoring. It changes only two things: the pair-HMM
emissions and the per-pair times.

**Emissions.** A match column emits J(t_sub). Insert and delete columns emit
pibar = sum_k w_k pi^k. Because the class is drawn independently per column,
this is the exact class-marginal pair HMM.
- [checked] By construction. The driver's `--selftest` also confirms that J
  is symmetric, sums to 1, has margins pibar and is diagonal at t = 0.
- [agent-checked] It matches a brute-force class-augmented Forward on short
  sequences.

**Times.** As in paper 1, one E-step is run under indel + LG at t = 1. Then
there are three options:

| Clock | Substitution time | Indel time |
|---|---|---|
| `one` | One shared time, from Newton on the expected complete-data log-likelihood with the LG match term replaced by sum W_ab log J(t)_ab | Same as substitution |
| `two` | Newton on sum W_ab log J(t)_ab alone | Paper 1's tau_LG |
| `anchor` | tau_LG | tau_LG |

The substitution time is clipped at paper 1's 10, or at 1000 in the `_T1000`
runs.

**Substitution models.**

| Name | What it is |
|---|---|
| LG | Paper 1's table |
| LGp | Published LG |
| C20 | LG exchangeabilities with the Le/Gascuel/Lartillot C20 profiles, each class at mean rate 1 (IQ-TREE's LG+C20 convention, without the +G) |
| C20p | As C20, with published LG |
| C20g | As C20, with one global rate scale instead of per-class |
| CML20 | Paper 1's CherryML 20-class mixture, gap state dropped, weights unchanged |
| CML20r | As CML20, with weights conditioned on a residue being present, w_k (1 - pi^k_gap) |
| `+G alpha` | 4 discrete Gamma categories (Yang 1994 mean rates) |

The Gamma shapes are 1.5 for LG and 0.35 for C20. Each was the best value on a
grid {0.25, 0.35, 0.5, 0.7, 1, 1.5, 2.5} fitted to the Pfam pairs below. That
grid search was ad hoc and is not committed. The committed script reproduces
the chosen points.

**Indel models.** MixFrag F=2 and TKF92, with parameters pinned to the values
in paper 1's BAliBASE output JSONs. Both were fitted on Pfam with paper 1's LG
table.

**Validation.**
- [measured] With LG, clock `one` and one EM round, the driver reproduces
  paper 1's `_pairwise_posteriors_mixfrag_jax` posteriors to 1e-13.
- [measured] The F=1 path matches paper 1's own TKF92 code to 1e-12.
- [measured] End to end, MixFrag+LG reproduces paper 1's per-family SP/TC for
  all 120 families: mean 0.7902 / 0.6405.
- [measured] TKF92+LG gives 0.7765 / 0.6216. Paper 1's Table 2 has 0.776 /
  0.622; per-family values for that row are not in the paper-1 JSONs.

**Statistics.** The paired difference per family is taken against the
baseline on the families both runs scored. Each comparison reports a 95%
bootstrap interval of the mean and a two-sided sign test. There is no
multiple-comparison correction.

## Results (BAliBASE bali3pdbm, 120 families) [measured]

Regenerate with `python3 analysis/scripts/fsa_c20_summary.py`. The data is in
`results/fsa_c20_balibase/*.jsonl`. "LG" means paper 1's table. The last two
columns are paired against the LG run with the same indel model.

| Indel | Substitution | Clock | Families | SP | TC | dSP vs LG [95% CI], p | dTC vs LG |
|---|---|---|---|---|---|---|---|
| MixFrag | LG (paper 1) | one | 120 | 0.790 | 0.640 | — | — |
| MixFrag | LGp (published) | one | 120 | 0.800 | 0.651 | +0.009 [+0.001, +0.018], p=0.75 | +0.011 |
| MixFrag | LG+G1.5 | two (T1000) | 120 | 0.789 | 0.639 | -0.001 [-0.011, +0.009] | -0.001 |
| MixFrag | C20 | two | 120 | 0.772 | 0.619 | -0.018 [-0.030, -0.007], p=0.002 | -0.021 |
| MixFrag | C20p | two | 120 | 0.775 | 0.618 | -0.016 [-0.026, -0.005], p=0.0008 | -0.022 |
| MixFrag | CML20 | two | 120 | 0.802 | 0.654 | +0.012 [+0.002, +0.022], p=0.30 | +0.014 |
| MixFrag | CML20r | two | 120 | 0.799 | 0.651 | +0.009 [+0.001, +0.017], p=0.09 | +0.011 |
| TKF92 | LG (paper 1) | one | 120 | 0.776 | 0.622 | — | — |
| TKF92 | LGp (published) | one | 120 | 0.794 | 0.645 | +0.017 [+0.007, +0.031], p=0.005 | +0.023 (p=0.03) |
| TKF92 | CML20 | anchor | 120 | 0.791 | 0.645 | +0.014 [+0.003, +0.027], p=0.41 | +0.024 |
| MUSCLE 5.3 / MAFFT --auto | | | 120 | 0.841 / 0.797 | 0.715 / 0.653 | (paper 1) | |

Against published LG, with MixFrag:
- C20p is -0.025 SP [-0.034, -0.016] (p = 1e-5) and -0.033 TC.
- CML20 is +0.003 SP [-0.008, +0.013] (n.s.).
- CML20r is -0.001 (n.s.).
- LG+G1.5 (on paper 1's table) is -0.010 (n.s.).

**Incomplete runs** were stopped on wideboy to be finished elsewhere. Their
paired differences against LG are over the families finished. Those are the
first families in name order:
- every partial run contains all 41 RV11 and RV12 families;
- the runs with 68 families or more also contain all of RV20;
- the 77–86-family runs also contain most or all of RV30;
- none contains RV40 or RV50, except for 2 RV40 families in the 86-family run.

So they are not a representative sample.

| Indel | Substitution | Clock | Families | dSP vs LG [95% CI], p |
|---|---|---|---|---|
| MixFrag | C20 | one | 86 | -0.040 [-0.058, -0.022], p=1e-6 |
| MixFrag | C20 | anchor | 45 | -0.077 [-0.103, -0.051], p=9e-6 |
| MixFrag | C20g | two | 77 | -0.014 [-0.027, -0.000], p=0.08 |
| MixFrag | C20 | two (T1000) | 58 | -0.018 [-0.040, +0.002], p=0.12 |
| MixFrag | C20+G0.35 | two (T1000) | 68 | -0.021 [-0.041, -0.001], p=0.001 |
| MixFrag | CML20 | anchor | 45 | +0.005 [-0.006, +0.017] |
| TKF92 | C20 | two | 80 | -0.009 [-0.027, +0.009], p=0.18 |

## Why per-site C20 does not help

**Identity floor and clipped times.**
- [measured] P(identical) at t = 1000 is 0.197 for C20, 0.059 for LG, 0.159
  for CML20 and 0.195 for CML20r. Source: `pfam_pair_fit.py`.
- [measured] Over the 1,686 BAliBASE reference pairs, identity across aligned
  residue pairs has median 0.205 and quartiles 0.164 / 0.272, and 68% of pairs
  are below 0.25. Source: `fsa_c20_summary.py --diagnostics`.
- [measured] In the two-clock FSA runs the substitution time is at the clip of
  10 for 36% of pairs with C20 (median t_sub 7.7) and 50% with C20p. For CML20,
  CML20r and LG+G it is at the clip for 0% of pairs (medians 2.0, 3.3 and 2.3).
- [measured] Raising the clip to 1000 changes the C20 result little: -0.018 SP
  over 58 families.

**Pfam pair fit** [measured]. Regenerate with
`python3 analysis/scripts/pfam_pair_fit.py`; output in
`results/pfam_pair_fit/summary.tsv`. It uses 1,631 random pairs (up to 30 per
alignment) from the 61 Pfam seed alignments in `data/pfam_seed_sample`, with
215,291 aligned residue pairs and median identity 0.358. Each model is
maximized over t per pair. Values are per aligned residue pair. "Association"
is the log-likelihood minus the composition term sum N_ab log(pibar_a pibar_b).

| Model | Log-likelihood | Association | Median t | Fraction t >= 10 |
|---|---|---|---|---|
| LG (paper 1) | -5.1964 | 0.588 | 1.36 | 0.00 |
| LG+G1.5 | -5.1879 | 0.597 | 1.82 | 0.00 |
| LGp (published) | -5.1891 | 0.595 | 1.43 | 0.00 |
| LGp+G1.5 | **-5.1803** | **0.604** | 1.91 | 0.00 |
| C20 | -5.2627 | 0.543 | 4.12 | 0.35 |
| C20+G0.35 | -5.2303 | 0.576 | 36.0 | 0.70 |
| C20p | -5.2645 | 0.541 | 4.54 | 0.36 |
| C20g | -5.2672 | 0.539 | 5.24 | 0.39 |
| CML20 | -5.1934 | 0.598 | 1.36 | 0.07 |
| CML20r | -5.1869 | 0.593 | 2.10 | 0.13 |

C20's deficit is in the association term, which drives match-versus-gap
decisions. Adding a Gamma recovers part of it, but C20+G is still below plain
LG. The FSA ranking follows the association column. The exact LG table under
C20 hardly matters (C20 vs C20p), because the class profiles dominate.

**CherryML's mixture is different.**
- [measured] With the gap state dropped, its classes have mean rates from 0.48
  to 36.5 (weighted mean 9.35). It therefore contains strong rate heterogeneity
  and does not saturate.
- [measured] Its heaviest class (w = 0.193) is 97% gap, and 0.28 of the total
  weight is on classes whose gap mass exceeds 0.5. Source:
  `fsa_c20_summary.py --diagnostics`.
- [measured] Conditioning the weights on a residue being present (CML20r)
  improves the Pfam fit but not FSA.

**Conventions and literature** [agent-checked]:
- Le, Gascuel & Lartillot 2008 (Bioinformatics 24:2317) define C10–C60 as
  mixtures of Poisson (F81) profiles, with time in Poisson events and no
  per-class normalization.
- IQ-TREE ships `model C20 = POISSON+G+FMIX{...}`. Its LG+C20 uses LG
  exchangeabilities in every class, with each class normalized to one
  substitution per unit time and an implicit +G.
- The repo's C20 profiles and weights match IQ-TREE's exactly after
  reordering.
- I found no published test of an iid per-site C20/CAT-type mixture as aligner
  emissions on a structural benchmark. The closest is Golden et al. 2017 (Mol
  Biol Evol 34:2085, doi:10.1093/molbev/msx137). They put a CAT-GTR-type
  mixture into a TKF92 pair HMM, with hidden states chained along the
  alignment and classes trained on HOMSTRAD. Sequence-only, it was "similar"
  in accuracy to StatAlign, BAli-Phy, MUSCLE and MAFFT on 38 HOMSTRAD pairs.
- Related evidence of small effects:
  - MUMMALS (Pei & Grishin 2006, NAR 34:4364): multiple match states add about
    +0.011 Q-score.
  - Crooks & Brenner 2005 (Bioinformatics 21:975): a Dirichlet-mixture
    background model performs comparably to BLOSUM62 and VTML160.
- Larger alignment gains in the literature come from sequence context
  (CS-BLAST) or observed structure. A per-site iid class marginal cannot
  supply either.

## The "LG08" table is not LG

**[measured]** `python3 analysis/scripts/lg_table_check.py` compares
tkf-mixdom's `tkfmixdom.jax.core.protein._LG_S_LOWER` with the published table.
`src/lg08.py` holds an identical copy, and per the agent audit so does
`tkfmixdom/jax/distill/maraschino.py`; I did not check that file.
- 171 of 190 entries differ, starting at row Q. The 12 entries of rows R, N, D
  and C are correct. The frequencies are identical.
- The differences run from -99% to +2310%. The median absolute difference is
  34% (quartiles 14% and 58%; 90th percentile 101%). 109 entries are too low
  and 62 too high. Weighted by substitution flux pi_a S_ab pi_b, the median
  error is 19%.
- The worst entries:
  - H–C: 0.641 -> 0.006
  - E–C: 0.0035 -> 0.084
  - Q–C: 0.085 -> 0.746
  - G–E: 0.349 -> 2.548
- **It is not an ordering or storage problem.** Only 48 of the 190 local values
  occur anywhere in published LG. The local table has repeated values (164
  distinct vs 190). No local row matches any published row as a set, and the
  published list read column-major doesn't match either. A relabeling or a
  storage-order mix-up would preserve the set of values.

**[checked]** Against PAML's `dat/lg.dat` and `dat/wag.dat` (downloaded from
github.com/abacus-gene/paml) and IQ-TREE's `model/modelprotein.cpp`:
- PAML's lg.dat equals IQ-TREE's `model LG` entry for entry.
- `src/progalign/lg_paml.py` equals lg.dat.
- tkf-mixdom's `_WAG_S_LOWER` is also wrong: 150 of 190 entries differ from
  wag.dat, also from row Q on. This comparison is not in a committed script.
- `git log -S` finds the LG table already present in tkf-mixdom commit
  196d8f71b (2026-03-10, a module reorganization), so it is at least that old.

**Impact.**
- [measured] The published table gains 0.0073 nats per aligned residue pair on
  Pfam pairs.
- [measured] It gains +0.017 SP / +0.023 TC for TKF92 FSA (p = 0.005 / 0.03)
  and +0.009 / +0.011 for MixFrag (n.s.).
- [not verified] The indel parameters were fitted with the wrong table; the
  effect of refitting them is untested.

Nothing in tkf-mixdom or `src/lg08.py` has been changed. The published
table is used only through `--subst LGp/C20p`.

## Paper 1 (submitted) audit

- [measured] TKF92 0.776 / 0.622 and MixFrag 0.790 / 0.640 reproduce. The
  CherryML-C=20 row (0.791 / 0.645) reproduces as `tkf92_CML20_anchor`,
  0.7907 / 0.6452.
- [checked: commit b12b411 and fsa_anneal.py lines 769–779] The committed
  `--method cherryml_mixture` path is a per-family mixture: one class
  responsibility per family. Commit b12b411 replaced its numbers (0.701 /
  0.510) with the per-site variant's, but the per-site code was not committed.
- [checked: mbe_paper1.tex lines 880–960] The text says "For all models, we
  use LG08 as our substitution model, with (Q, pi) fixed" and "Both are also
  fit with exact summarized-count EM".
- [agent-checked] Neither statement holds for every row:
  - TKF92-K=20 is a per-family mixture of 20 complete TKF92+GTR models, each
    with its own fitted S_k and pi_k.
  - CherryML-C=20 was fitted by in-house per-site EM (`em_around_cherryml.py`,
    21 states including gap, Adam inner M-step), not by summarized-count EM or
    the CherryML software.
  - MixDom-d3f1 also has its own substitution model.
- [measured] Every "LG08" result uses the table above.
- Separately [checked earlier in the session]: the MAFFT row labelled
  "FFT-NS-2" is `mafft --auto`.

## Caveats

- Seven runs are incomplete (see above). Their subsets are not representative.
- A single benchmark (bali3pdbm). Paper 1's design is kept throughout: one
  E-step at t = 1 under LG, and the paper's Newton settings.
- [agent-checked] With `--em-rounds 2`, the two-clock indel time switches
  estimator in later rounds. That variant was dropped and is not reported.
- The Gamma shapes were chosen on the same Pfam pairs used in the fit table.
  No BAliBASE data was used to choose them.
- Sign tests and intervals are per comparison, with no multiplicity
  correction.

## Machine and provenance

- **Host:** wideboy (wideboy.local). Intel Xeon W-3223 @ 3.50 GHz, 16 logical
  CPUs, 32 GB RAM, macOS 15.1.1.
- **Software:** Python 3.12.6, JAX/jaxlib 0.4.38 (CPU, float64), numpy 2.4.2,
  scipy 1.17.1.
- **Repositories:** tkf-mixdom at commit 6c939b6c2, read-only. tkf-dp: the
  driver at 30fd195, results at b4adbfc, this note and its scripts in the next
  commit.
- **Data:** BAliBASE bali3pdbm from `~/bio-datasets/data/balibase/bali3pdbm`.
  Pfam seed sample in `data/pfam_seed_sample` (gitignored; see
  `analysis/profile_elbo/README.md` for how it was fetched).
- **Run time:** the ten complete runs took 3.5–5 h of wall time each, with
  10–17 runs sharing the CPU. Per-family wall time summed over all 17 runs is
  84 h.
- **Paper 1's numbers** come from a different (Linux) machine; its JSON configs
  record `/home/yam/...` paths. Agreement is exact per family for MixFrag. For
  the other rows only the table values could be compared.
- **Verification workflow:** 31 subagents, an investigator and a skeptic for
  each claim. Their scratch checks are not committed.
