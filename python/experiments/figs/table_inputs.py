#!/usr/bin/env python3
"""Run paper_tables.py with an audit hook and list every result file it reads, one per line, as
<repo>/<path> (repo = tkf-mixdom or tkf-dp), into paper_tables_inputs.txt. The publish drop ships
exactly these files, so the public repos can regenerate the paper-1 tables.
Usage: table_inputs.py   (tables go to a temporary directory; the committed ones are not touched)"""
import os, runpy, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
MIXDOM = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
opened = set()
def hook(ev, args):
    if ev == 'open' and isinstance(args[0], str) and not (len(args) > 1 and isinstance(args[1], str) and set(args[1]) & set('wax+')):
        opened.add(os.path.abspath(args[0]))
os.environ['PAPER_TABLES_OUT'] = tempfile.mkdtemp(prefix='paper_tables_')
sys.argv = ['paper_tables.py']
sys.addaudithook(hook)
g = runpy.run_path(os.path.join(HERE, 'paper_tables.py'), run_name='__main__')
roots = {'tkf-mixdom': MIXDOM, 'tkf-dp': os.path.abspath(g['TKFDP'])}
out = []
for f in sorted(opened):
    if not os.path.isfile(f) or f.endswith('.py'): continue
    for name, r in roots.items():
        if f.startswith(r + os.sep):
            rel = os.path.relpath(f, r)
            if rel.startswith(('python/experiments/', 'results/')):   # result files only, not the venv
                out.append(f'{name}/{rel}')
            break
with open(os.path.join(HERE, 'paper_tables_inputs.txt'), 'w') as fh:
    fh.write(''.join(l + '\n' for l in out))
print(f'{len(out)} inputs -> paper_tables_inputs.txt', file=sys.stderr)
