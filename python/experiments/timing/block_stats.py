"""Per-iteration costs of the timed blocks, computed post hoc over iterations 21-80 (steps 21-124 for Adam) so that
first-time executable loads in the first iterations never enter the number. Usage: python3 block_stats.py [scratch dir]"""
import re, sys, statistics as st, glob, os, json
T = sys.argv[1] if len(sys.argv) > 1 else 'experiments/timing/scratch'
PAT = {'svi': r'\[\s*([\d.]+)s\] --- SVI-BW iteration (\d+)/', 'mfs': r'\[\s*([\d.]+)s\]\s+iter\s+(\d+)/', 'adam': r'\[\s*([\d.]+)s\]\s+Step\s+(\d+)/'}
def stamps(path, pat):
    ts, loads, cur = {}, {}, 0
    for line in open(path, errors='replace'):
        m = re.match(pat, line)
        if m: cur = int(m.group(2)); ts[cur] = float(m.group(1)); loads[cur] = 0; continue
        if 'Assume version compatibility' in line: loads[cur] = loads.get(cur, 0) + 1
    return ts, loads
def block(tag, path, kind, first, last, total):
    if not os.path.exists(path): return None
    ts, loads = stamps(path, PAT[kind])
    iv = [ts[i + 1] - ts[i] for i in range(first, last) if i in ts and i + 1 in ts]
    if len(iv) < 5: return None
    r = dict(tag=tag, startup_s=ts.get(1, float('nan')), n=len(iv), median=st.median(iv), mean=st.mean(iv), max=max(iv),
             loads=sum(loads.get(i, 0) for i in range(first + 1, last + 1)), total_h=(ts.get(1, 0) + total * st.median(iv)) / 3600, total_n=total)
    unit = 'step' if kind == 'adam' else 'it'
    print(f"{tag:12s} median {r['median']:6.2f} s/{unit} (mean {r['mean']:6.2f}, max {r['max']:6.1f}, n={r['n']}, cache loads {r['loads']}); startup {r['startup_s']:5.0f} s; {total:,} {unit}s ~ {r['total_h']:5.2f} h")
    return r
out = {}
for tag in ['d1f1', 'd2f1', 'd3f1', 'd4f1', 'd5f1', 'd1f2', 'd2f2']:
    r = block(tag, f'{T}/T2_{tag}_iter80_timed.out', 'svi', 20, 80, 1620); out[tag] = r
for F in (2, 3):
    r = block(f'mixfrags_F{F}', f'{T}/mfs{F}_80.log', 'mfs', 20, 80, 1620); out[f'mixfrags_F{F}'] = r
for n in (2, 3):
    r = block(f'adam_d{n}f1', f'{T}/T3_adam_d{n}f1_steps125_timed.out', 'adam', 20, 124, 2000); out[f'adam_d{n}f1'] = r
json.dump({k: v for k, v in out.items() if v}, open(f'{T}/block_stats.json', 'w'), indent=1)
