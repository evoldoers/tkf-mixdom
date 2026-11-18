#!/usr/bin/env python3
"""List every result file the paper-1 table generators read, one repository-relative path per line, into
paper_tables_inputs.txt. Each generator runs in its own process under an audit hook, writing to a temporary
directory (the manuscript's tables and figures are not touched). These are the files of the paper-1 results
release (see README.md); files read outside this repository (the bio-datasets benchmarks) are only reported.
Usage: table_inputs.py"""
import os, runpy, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
GENERATORS = ['figs/paper_tables.py', 'figs/benchmark_stats.py', 'em_vs_gradient/convergence_figure.py']

if len(sys.argv) == 3:                        # child: run one generator, log the files it opens for reading
    gen, log = sys.argv[1], sys.argv[2]
    opened = set()
    def hook(ev, args):
        if ev == 'open' and isinstance(args[0], str) and not (len(args) > 1 and isinstance(args[1], str) and set(args[1]) & set('wax+')):
            opened.add(os.path.abspath(args[0]))
    sys.argv = [gen]
    sys.addaudithook(hook)
    try:
        runpy.run_path(gen, run_name='__main__')
    finally:
        with open(log, 'w') as fh:
            fh.write(''.join(f + '\n' for f in sorted(opened)))
    sys.exit(0)

tmp = tempfile.mkdtemp(prefix='paper1_tables_')
env = dict(os.environ, PAPER1_DIR=tmp, PAPER_TABLES_OUT=os.path.join(tmp, 'tables'))
inside, outside = set(), set()
for gen in GENERATORS:
    log = os.path.join(tmp, os.path.basename(gen) + '.opened')
    path = os.path.join(REPO, 'python', 'experiments', gen)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), path, log], env=env, cwd=os.path.dirname(path),
                       stdout=subprocess.DEVNULL)
    if r.returncode: sys.exit(f'{gen} failed (exit {r.returncode})')
    for f in open(log).read().split():
        if not os.path.isfile(f) or f.endswith(('.py', '.pyc')) or f.startswith(tmp): continue
        rel = os.path.relpath(f, REPO)
        if rel.startswith('python/experiments/'): inside.add(rel)
        elif not rel.startswith('..'): continue                     # interpreter / venv files
        elif not f.startswith(('/usr', '/proc', '/etc', '/dev')) and 'site-packages' not in f and '/uv/python' not in f: outside.add(f)
with open(os.path.join(HERE, 'paper_tables_inputs.txt'), 'w') as fh:
    fh.write(''.join(l + '\n' for l in sorted(inside)))
print(f'{len(inside)} inputs -> paper_tables_inputs.txt; tables in {tmp}', file=sys.stderr)
if outside:
    dirs = sorted({os.path.dirname(f) for f in outside})
    print(f'{len(outside)} files read outside the repository (not in the release), in {len(dirs)} directories, e.g. {dirs[:3]}', file=sys.stderr)
