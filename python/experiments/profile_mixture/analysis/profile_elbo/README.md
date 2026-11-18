# Profile-mixture bounds and alignment-free distances: notes, code, results

Three LaTeX notes live here. None of them is part of a submitted paper, and
none contains file paths or script names; this README maps each to the code
and results behind it.

| Note | Subject |
|---|---|
| `profile_classes.tex` | Column-pair scoring under a K-class profile mixture (e.g. LG-C20): the O(K+A) soft-Fitch bound, why it is too loose for alignment, the sparse-class bound and posterior-pair / log-expectation scores, decoding objectives, pseudocode, and the BAliBASE benchmark of the JAX aligner |
| `profile_elbo.tex` | The collapsed-Dirichlet ELBO (per-column pi ~ Dirichlet, rate ~ Gamma) via secret-destination augmentation, and the discrete-class special case |
| `kgram_distances.tex` | Alignment-free bigram and trigram distances under the MixFrag Pair HMM, and a (negative) test of guide trees built from them |

Build any of them with `latexmk -pdf <note>.tex` in this directory
(`latexmk -c` removes the intermediates). PDFs are not committed (`*.pdf` is
gitignored).

## Data

The Pfam tests use 60 families drawn at random (Python `random.Random(0)`)
from the test split in `~/bio-datasets/data/pfam/seed/splits/v1.json`,
downloaded as Stockholm seed alignments from InterPro:

```
https://www.ebi.ac.uk/interpro/api/entry/pfam/<PFxxxxx>/?annotation=alignment:seed&download
```

into `data/pfam_seed_sample/<PFxxxxx>.sto.gz` (gitignored). LG exchangeabilities
come from `src/lg08.py`; the C20 profiles from
`~/tkf-mixdom/python/tkfmixdom/jax/core/site_class_profiles.py`; the Pfam-trained
TKF92 indel parameters from `~/tkf-mixdom/python/params/best/cem_tkf92_pfamTrain.json`.
The scripts need only numpy and scipy (the JAX import in `lg08.py` is stubbed).

## `profile_classes.tex`

| Section | Script | Results |
|---|---|---|
| Test on Pfam (bound vs exact LG-C20, re-estimated clades, rankings) | `analysis/scripts/profile_classes_pfam.py` | `results/profile_classes_pfam/` |
| Tightness, choosing a score (clade-pair harness: bias, recall, proposal KL) | `analysis/scripts/bound_harness.py`, `analysis/scripts/pp_scorers.py` | printed; design-study code was in a scratch directory |
| Results on BAliBASE | `experiments/bench_progalign_balibase.py` (aligner: `src/progalign/`) | `results/progalign_balibase/results_g*.csv` (the script also writes each alignment to `aln/`, not committed) |
| Discrete-class separable score (toy checks, also used by `profile_elbo.tex`) | `analysis/scripts/profile_elbo_classes_check.py` | printed |

```
python3 analysis/scripts/profile_classes_pfam.py --data-dir data/pfam_seed_sample \
    --out results/profile_classes_pfam --cols-per-family 30
JAX_ENABLE_X64=1 python3.12 experiments/bench_progalign_balibase.py \
    --methods muscle mafft mafft_auto tkf92_lg_viterbi mixfrag_lg_mea mixfrag_c20_mea_pp ...
```
BAliBASE `bali3pdbm` is extracted from https://drive5.com/bench/bench.tar.gz into
`~/bio-datasets/data/balibase/bali3pdbm`; MUSCLE 5.3 is the release binary with its
gcc libraries re-pointed to Homebrew's; MAFFT 7.526 is from Homebrew. Method names are
`<indel>_<subst>_<decode>[_<scoring>][_r<rounds>]`, see the script's docstring.

## `profile_elbo.tex`

| Section | Script |
|---|---|
| Numerical check (collapsed-Dirichlet and Gamma-rate bounds vs Monte Carlo) | `analysis/scripts/profile_elbo_check.py` |
| Discrete profile classes | `analysis/scripts/profile_elbo_classes_check.py` |

## `kgram_distances.tex`

| Section | Script | Results |
|---|---|---|
| Simulation (closed forms for E[D2], E[D3]; estimator spread) | `analysis/scripts/bigram_distance_check.py` (`--c20` for the C20 mixture) | printed |
| Pfam distances vs TKF92+LG-C20 maximum-likelihood t | `analysis/scripts/bigram_distance_pfam.py` | `results/bigram_distance_pfam/` |
| Guide trees (NJ, BioNJ, variance-weighted BioNJ) vs reference | `analysis/scripts/guide_tree_variance.py` | `results/guide_tree_variance/` |
