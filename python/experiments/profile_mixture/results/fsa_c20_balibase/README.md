# FSA on BAliBASE with per-site profile-mixture substitution models

Produced by `experiments/fsa_c20_balibase.py` (paper 1's pipeline with only the
pair-HMM emissions and per-pair times changed); summarize with
`python3 analysis/scripts/fsa_c20_summary.py`. One JSONL row per family; the
`.json` summary is written when a run finishes all 120 families. Rerunning a
command skips families already in its JSONL.

Method name = `<indel>_<subst>[G<alpha>]_<clock>[_T<tmax>]`, e.g.
`mixfrag_C20p_two`. LG = tkf-mixdom's table (paper 1's "LG08", which differs
from published LG in 171/190 exchangeabilities); LGp/C20p use published LG.

Complete (120 families): mixfrag_{LG,LGp}_one, mixfrag_{C20,C20p,CML20,CML20r}_two,
mixfrag_LGG1.5_two_T1000, tkf92_{LG,LGp}_one, tkf92_CML20_anchor.
Partial (stopped 2026-09-30, to be finished elsewhere): the seven below.

```
export JAX_ENABLE_X64=1 JAX_CACHE_DIR=$HOME/.cache/jax-fsa-c20   # shared compile cache
D=results/fsa_c20_balibase; E=experiments/fsa_c20_balibase.py
python3 $E --indel mixfrag --subst C20  --clock one                          --out $D/mixfrag_C20_one.json
python3 $E --indel mixfrag --subst C20g --clock two                          --out $D/mixfrag_C20g_two.json
python3 $E --indel mixfrag --subst C20  --clock anchor                       --out $D/mixfrag_C20_anchor.json
python3 $E --indel mixfrag --subst CML20 --clock anchor                      --out $D/mixfrag_CML20_anchor.json
python3 $E --indel mixfrag --subst C20  --clock two --tmax-sub 1000 --newton-sub 12 --out $D/mixfrag_C20_two_T1000.json
python3 $E --indel mixfrag --subst C20 --gamma 0.35 --clock two --tmax-sub 1000 --newton-sub 12 --out $D/mixfrag_C20G0.35_two_T1000.json
python3 $E --indel tkf92   --subst C20  --clock two                          --out $D/tkf92_C20_two.json
```
Needs `~/tkf-mixdom/python` and `~/bio-datasets/data/balibase/bali3pdbm`. On
CPU, XLA's compile threads ignore single-thread flags: run at most about
(cores - 4) of these at once.
