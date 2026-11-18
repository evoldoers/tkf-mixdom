# Paper-1 tables

The generated tables of the paper-1 manuscript come from three scripts, run from `python/`:

| script | writes |
|---|---|
| `experiments/figs/paper_tables.py` | `models_table.tex`, `decoding_table.tex`, `decoding_table_body.tex`, `tails_table.tex`, `paired_tests.json` |
| `experiments/em_vs_gradient/convergence_figure.py` | `em_vs_gradient_table.tex` and the figure `em_vs_gradient.pdf` |
| `experiments/figs/benchmark_stats.py` | `benchmarks_table.tex` |

They write to the directory given by `paper1_dir.py`: `$PAPER1_DIR` if set, else the manuscript
directory of a development checkout, else `experiments/figs/paper1_out/`.

## Inputs

The result files they read are listed in `paper_tables_inputs.txt` (paths relative to the repository root)
and are attached to the release `results/paper1-tables-2026-11-18` of this repository. To regenerate the tables
from a clone:

```bash
python/experiments/figs/fetch_table_inputs.sh      # downloads and unpacks the result files (75 MB)
cd python
python experiments/figs/paper_tables.py
python experiments/em_vs_gradient/convergence_figure.py
BIO_DATASETS=/path/to/bio-datasets python experiments/figs/benchmark_stats.py
```

`benchmark_stats.py` reads the benchmark alignments themselves (BAliBASE 3 `bali3pdbm`, SABmark `sabrem`,
OXBench `oxm`, from the MUSCLE 5 benchmark collection), fetched by the scripts of
https://github.com/evoldoers/bio-datasets (`fetch/balibase`, `fetch/oxbench`).

`table_inputs.py` rebuilds `paper_tables_inputs.txt` by running the three scripts under an audit hook that
records every file they open.
