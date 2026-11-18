# Profile-mixture alignment (paper 1 comparison rows and related notes)

Moved here from the tkf-dp repository on 2026-11-18; the layout inside this folder is the one it had there.

- `experiments/fsa_c20_balibase.py` -- paper 1's FSA pipeline (`../expected_pairwise_balibase.py`) with the
  pair-HMM emissions and per-pair times changed: per-site profile mixtures (LG-C20, CherryML C=20), the
  published LG, a GTR refit, and MixFrag-S. Its runs are in `results/fsa_c20_balibase/` (see the README
  there); the ones the paper-1 tables read are in the paper-1 results release (`../figs/README.md`).
  The launch scripts in `../benchmarks/`, `../balibase_exclusion/` and `../mixfrag_F/` call it.
- `src/progalign/` -- statistical progressive alignment under a K-class profile mixture and MixFrag
  (`analysis/profile_elbo/profile_classes.tex`); `fsa_c20_balibase.py` uses its substitution models.
  `src/lg08.py` is a copy of tkf-dp's LG08 table module, used by two scripts here.
- `analysis/` -- `fsa_profile_mixture_findings.md` (the C20-in-FSA study) and the `profile_elbo/` notes
  (internal; not part of a paper), with their scripts in `analysis/scripts/`.
- `experiments/bench_progalign_balibase.py`, `results/` -- the progressive-aligner benchmark and small result
  files of the notes.
- `tests/` -- `pytest tests` (from this folder).
