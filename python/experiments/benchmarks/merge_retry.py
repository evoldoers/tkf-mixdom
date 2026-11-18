#!/usr/bin/env python3
"""Merge a retry run of expected_pairwise_balibase.py (--families ...) into the main result file: per_family rows
of the retry replace (or add to) the main file's rows of the same family. The main file is backed up to
<main>.premerge.json. Usage: merge_retry.py <main.json> <retry.json>"""
import json, shutil, sys
main, retry = sys.argv[1], sys.argv[2]
d = json.load(open(main)); r = json.load(open(retry))
rows = {x['family']: x for x in d['per_family']}
new = 0; replaced = 0
for x in r['per_family']:
    if x['family'] in rows: replaced += 1
    else: new += 1
    rows[x['family']] = x
shutil.copy(main, main.replace('.json', '.premerge.json'))
d['per_family'] = list(rows.values()); d['n_families'] = len(d['per_family'])
json.dump(d, open(main, 'w'))
print(f'{main}: {replaced} rows replaced, {new} added -> {len(d["per_family"])} families')
