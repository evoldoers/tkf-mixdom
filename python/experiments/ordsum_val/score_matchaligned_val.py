"""Ordering-summed (match-aligned 2D forward) val LL/pair on v3_val for all
three models. Matches are FIXED to the observed alignment (i,j); the 2D DP
sums over ALL indel orderings between match anchors (NOT a fixed path).

Models:
  MODEL=d3f1     : MixDom d3f1(c3), per-class emissions + chi from checkpoint
  MODEL=mixfrag  : MixFrag F2 pair HMM (make_mixfrag_pair_hmm)
  MODEL=tkf92    : TKF92 pair HMM  (make_tkf92_pair_hmm)

Same 116,640 v3_val pairs, same t per pair, LG08 base, 1e-30 floor, x64.
"""
import os, sys, re, io, glob, json, time
import numpy as np

sys.path.insert(0, '/home/yam/tkf-mixdom/python')
# The fast linear-space / associative-scan 2D-forward fixes are on main
# (hmm.py, commit 20768cb23), so no worktree override is needed. WT_PY can
# still be set to point tkfmixdom at an alternate python tree if desired.
_WT = os.environ.get('WT_PY', '')
if _WT and os.path.isdir(_WT):
    sys.path.insert(0, _WT)
os.chdir('/home/yam/tkf-mixdom/python')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import zstandard

MODEL = os.environ.get("MODEL", "d3f1")

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.simulate.msa import alignment_to_states
from tkfmixdom.jax.util.padding import pad_to_bin as _pad_to_bin
from tkfmixdom.jax.dp.hmm import (
    NEG_INF, mask_emissions_match_aligned, forward_backward_2d,
    pair_hmm_emissions, pair_hmm_emissions_per_class, pair_hmm_emissions_per_domain)
from tkfmixdom.jax.core.params import M as _M, I as _I, D as _D

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}
AA = len(AMINO_ACIDS)
VAL_DIR = '/home/yam/tkf-mixdom/python/pfam/precompiled_v3_val'
LOG_FLOOR = float(np.log(1e-30))

Q_lg, pi_lg = rate_matrix_lg()
Q_lg = np.asarray(Q_lg); pi_lg = np.asarray(pi_lg)

CIGAR_RE = re.compile(r'([MDI])(\d+)')
def expand_cigar(x, a, y):
    xi = yi = 0; anc = []; desc = []
    matched = CIGAR_RE.findall(a)
    assert sum(len(o) + len(c) for o, c in matched) == len(a), ("bad cigar", a)
    for op, n in matched:
        n = int(n)
        if op == 'M':
            for _ in range(n):
                anc.append(AA_TO_IDX.get(x[xi], -1)); desc.append(AA_TO_IDX.get(y[yi], -1)); xi += 1; yi += 1
        elif op == 'D':
            for _ in range(n):
                anc.append(AA_TO_IDX.get(x[xi], -1)); desc.append(-1); xi += 1
        else:
            for _ in range(n):
                anc.append(-1); desc.append(AA_TO_IDX.get(y[yi], -1)); yi += 1
    assert xi == len(x) and yi == len(y)
    a_arr = np.array(anc, dtype=np.int32); b_arr = np.array(desc, dtype=np.int32)
    mask = (a_arr >= 0) | (b_arr >= 0)
    return a_arr[mask], b_arr[mask]

def extract_match_positions(states):
    i = j = 0; mi = []; mj = []
    for s in states:
        if s == _M: mi.append(i); mj.append(j); i += 1; j += 1
        elif s == _I: j += 1
        elif s == _D: i += 1
    return np.array(mi, dtype=np.int32), np.array(mj, dtype=np.int32)

# ---- read shards: x_int, y_int, match_i, match_j, t ----
t0 = time.monotonic()
pairs = []  # (x_int, y_int, mi, mj, t)
n_records = 0
dctx = zstandard.ZstdDecompressor()
for shard in sorted(glob.glob(os.path.join(VAL_DIR, 'shard_*.jsonl.zst'))):
    with open(shard, 'rb') as f:
        for line in io.TextIOWrapper(dctx.stream_reader(f), encoding='utf-8'):
            line = line.strip()
            if not line: continue
            r = json.loads(line); n_records += 1
            anc_aln, desc_aln = expand_cigar(r['x'], r['a'], r['y'])
            states, _, _ = alignment_to_states(anc_aln, desc_aln)
            if not states: continue
            x_int = np.array([int(c) for c in anc_aln if c >= 0], dtype=np.int32)
            y_int = np.array([int(c) for c in desc_aln if c >= 0], dtype=np.int32)
            mi, mj = extract_match_positions(np.array(states, dtype=np.int32))
            pairs.append((x_int, y_int, mi, mj, float(r['t'])))
print(f"[{MODEL}] read {n_records} records, {len(pairs)} scorable pairs, {time.monotonic()-t0:.1f}s", flush=True)

# Optional length cap: score only pairs with max(Lx,Ly) <= LMAX. This is a
# deterministic, well-defined subset (identical across models) that keeps the
# 2D DP tractable — the full-grid ordering-summed DP is O(Lx*Ly*ns) per pair
# and blows up on the long tail. Reported honestly as an "L<=LMAX" subset.
LMAX = int(os.environ.get("LMAX", "0"))
if LMAX > 0:
    kept = [p for p in pairs if len(p[0]) <= LMAX and len(p[1]) <= LMAX]
    print(f"[{MODEL}] LMAX={LMAX}: scoring {len(kept)}/{len(pairs)} pairs "
          f"({100*len(kept)/len(pairs):.1f}%) with max(Lx,Ly)<={LMAX}", flush=True)
    pairs = kept

# ---- model setup ----
if MODEL == 'd3f1':
    from train_pfam import _load_checkpoint, _build_log_chi_stack
    from tkfmixdom.jax.models.mixdom import (
        n_states as mixdom_n_states, state_types as mixdom_state_types)
    CKPT = '/tmp/claude-1003/-home-yam-tkf-dp/e986c820-957e-43e6-8345-cb50bd010f86/scratchpad/sharedexp/svibw_d3f1_v3_FROZEN_iter45.npz'
    (params, *_rest) = _load_checkpoint(CKPT)
    cfg = json.loads(str(np.load(CKPT, allow_pickle=True)['_config']))
    n_dom = int(cfg['n_dom']); n_frag = int(cfg['n_frag'])
    st = np.asarray(mixdom_state_types(n_dom, n_frag))
    ns = len(st)
    n_cls = int(params['n_classes'])
    S_per_class = np.asarray(params['class_S_exch'])
    class_pis_np = np.asarray(params['class_pis'])
    classdist_np = np.asarray(params['classdist'])
    class_Qs = np.zeros((n_cls, AA, AA))
    for cc in range(n_cls):
        Q_c = S_per_class[cc] * class_pis_np[cc, None, :]
        np.fill_diagonal(Q_c, 0.0); Q_c[np.diag_indices(AA)] = -Q_c.sum(axis=1)
        class_Qs[cc] = Q_c
    print(f"[{MODEL}] n_dom={n_dom} n_frag={n_frag} ns={ns} n_cls={n_cls}", flush=True)

elif MODEL in ('mixfrag', 'tkf92'):
    from tkfmixdom.jax.models.left_regular import (
        make_mixfrag_pair_hmm, make_tkf92_pair_hmm, S, M, I, D, E)
    if MODEL == 'mixfrag':
        P = dict(ins_rate=0.04608, del_rate=0.04709,
                 exts=[0.37466, 0.83325], weights=[0.62848, 0.37152])
        def make_hmm(t):
            return make_mixfrag_pair_hmm(P['ins_rate'], P['del_rate'], t,
                jnp.asarray(P['exts']), jnp.asarray(P['weights']),
                jnp.asarray(Q_lg), jnp.asarray(pi_lg))
    else:
        P = dict(ins_rate=0.04428, del_rate=0.04521, ext=0.677)
        def make_hmm(t):
            return make_tkf92_pair_hmm(P['ins_rate'], P['del_rate'], t, P['ext'],
                jnp.asarray(Q_lg), jnp.asarray(pi_lg))
    lt0, st0, _, _ = make_hmm(pairs[0][4]); st = np.asarray(st0); ns = len(st)
    print(f"[{MODEL}] ns={ns} state_types={st}", flush=True)

    def build_log_trans_jax(t):
        lt, _, _, _ = make_hmm(t)
        return jnp.maximum(lt, LOG_FLOOR)
else:
    raise SystemExit(f"unknown MODEL {MODEL}")

st_j = jnp.asarray(st)

# ---------------------------------------------------------------------------
# Precompute per-pair log_trans (and, for the mixfrag/tkf92 LG path, sub
# matrices; for d3f1 the per-class sub matrices) in BATCHED vmaps over t.
# Then build emit-table + mask + 2D-forward inside ONE jitted vmapped kernel
# per (Lxp, Lyp) grid bucket, so per-pair host recompiles vanish.
# ---------------------------------------------------------------------------
ts_all = np.array([p[4] for p in pairs], dtype=np.float64)

if MODEL == 'd3f1':
    from train_pfam import _build_log_chi_stack as _blcs
    # batched log_chi via the repo helper (already vmapped internally)
    def batched_log_trans(tarr):
        lc = np.asarray(_blcs(params, jnp.asarray(tarr.astype(np.float32))))  # (B,ns,ns)
        return np.maximum(lc, LOG_FLOOR)
    # batched per-class sub matrices: (B, C, A, A)
    _csub_v = jax.jit(jax.vmap(lambda tt: jax.vmap(lambda Qc: transition_matrix(Qc, tt))(jnp.asarray(class_Qs))))
    def batched_subs(tarr):
        return np.asarray(_csub_v(jnp.asarray(tarr)))  # (B,C,A,A)
    classdist_j = jnp.asarray(classdist_np); class_pis_j = jnp.asarray(class_pis_np)

    def _emit_and_fwd(log_chi, x, y, rlx, rly, csub, mi, mj):
        et = pair_hmm_emissions_per_class(
            st_j, x, y, csub, class_pis_j, classdist_j, n_dom, n_frag)
        et = mask_emissions_match_aligned(et, st_j, mi, mj)
        return forward_backward_2d(log_chi, st_j, x, y, None, None,
                                   log_emit_table=et, real_Lx=rlx, real_Ly=rly,
                                   forward_only=True)
    _kernel = jax.jit(jax.vmap(_emit_and_fwd, in_axes=(0, 0, 0, 0, 0, 0, 0, 0)))
    HAS_SUBS = True
else:
    _lt_v = jax.jit(jax.vmap(build_log_trans_jax))
    def batched_log_trans(tarr):
        return np.asarray(_lt_v(jnp.asarray(tarr)))
    _sub_v = jax.jit(jax.vmap(lambda tt: transition_matrix(jnp.asarray(Q_lg), tt)))
    def batched_subs(tarr):
        return np.asarray(_sub_v(jnp.asarray(tarr)))  # (B,A,A)
    pi_lg_j = jnp.asarray(pi_lg)

    def _emit_and_fwd(log_chi, x, y, rlx, rly, sub, mi, mj):
        et = pair_hmm_emissions(st_j, x, y, sub, pi_lg_j)
        et = mask_emissions_match_aligned(et, st_j, mi, mj)
        return forward_backward_2d(log_chi, st_j, x, y, None, None,
                                   log_emit_table=et, real_Lx=rlx, real_Ly=rly,
                                   forward_only=True)
    _kernel = jax.jit(jax.vmap(_emit_and_fwd, in_axes=(0, 0, 0, 0, 0, 0, 0, 0)))
    HAS_SUBS = True

# Precompute all log_trans and subs in chunks (batched over t).
PRE = 2048
log_trans_all = np.zeros((len(pairs), ns, ns), dtype=np.float64)
if MODEL == 'd3f1':
    subs_all = np.zeros((len(pairs), n_cls, AA, AA), dtype=np.float64)
else:
    subs_all = np.zeros((len(pairs), AA, AA), dtype=np.float64)
for i in range(0, len(pairs), PRE):
    tc = ts_all[i:i+PRE]
    log_trans_all[i:i+PRE] = batched_log_trans(tc)
    subs_all[i:i+PRE] = batched_subs(tc)
print(f"[{MODEL}] precomputed log_trans+subs for {len(pairs)} pairs, {time.monotonic()-t0:.1f}s", flush=True)

from collections import defaultdict
buckets = defaultdict(list)  # (Lxp,Lyp) -> list of idx
max_nm = {}
for idx, (x_int, y_int, mi, mj, t) in enumerate(pairs):
    Lxp = _pad_to_bin(len(x_int)); Lyp = _pad_to_bin(len(y_int))
    buckets[(Lxp, Lyp)].append(idx)

# Batch size: cap by (a) a memory budget on the (B, Lxp+1, Lyp+1, ns) forward
# table (~2 GB fp64) and (b) a large cell budget so small grids get big batches
# (fewer kernel launches → far less dispatch overhead than the earlier 6M cap).
MEM_BUDGET_BYTES = 700_000_000   # ~700MB output table; DP intermediates ~5-8x
MAX_B = 2048                      # absolute batch cap (bounds wavefront vmap mem)
total_ll = 0.0; n_pairs = 0; n_bad = 0
bad_examples = []; n_done = 0
for (Lxp, Lyp), idxs in sorted(buckets.items(), key=lambda kv: kv[0][0]*kv[0][1]):
    cells = (Lxp + 1) * (Lyp + 1) * ns
    B = max(1, min(len(idxs), MAX_B, MEM_BUDGET_BYTES // (max(cells, 1) * 8)))
    # pad match arrays to the bucket's max match count
    for cs in range(0, len(idxs), B):
        chunk = idxs[cs:cs+B]; bb = len(chunk)
        nm = max(1, max(len(pairs[i][2]) for i in chunk))
        xs = np.zeros((bb, Lxp), dtype=np.int32)
        ys = np.zeros((bb, Lyp), dtype=np.int32)
        rlx = np.zeros(bb, dtype=np.int32); rly = np.zeros(bb, dtype=np.int32)
        mis = np.zeros((bb, nm), dtype=np.int32); mjs = np.zeros((bb, nm), dtype=np.int32)
        for bi, i in enumerate(chunk):
            x_int, y_int, mi, mj, t = pairs[i]
            Lx = len(x_int); Ly = len(y_int)
            xs[bi, :Lx] = x_int; ys[bi, :Ly] = y_int
            rlx[bi] = Lx; rly[bi] = Ly
            # duplicate first match into padding slots (safe: sets True at an
            # already-True grid cell; never enables an extra match position)
            if len(mi) > 0:
                mis[bi, :len(mi)] = mi; mjs[bi, :len(mj)] = mj
                mis[bi, len(mi):] = mi[0]; mjs[bi, len(mj):] = mj[0]
        lp = np.asarray(_kernel(
            jnp.asarray(log_trans_all[chunk]), jnp.asarray(xs), jnp.asarray(ys),
            jnp.asarray(rlx), jnp.asarray(rly), jnp.asarray(subs_all[chunk]),
            jnp.asarray(mis), jnp.asarray(mjs)))
        for bi, v in enumerate(lp):
            if np.isfinite(v) and v > -1e10:
                total_ll += float(v); n_pairs += 1
            else:
                n_bad += 1
                if len(bad_examples) < 8:
                    i = chunk[bi]
                    bad_examples.append((i, len(pairs[i][0]), len(pairs[i][1]), pairs[i][4], float(v)))
        n_done += bb
    print(f"  [{MODEL}] grid ({Lxp},{Lyp}) x{len(idxs)} done  n_done={n_done}/{len(pairs)}  {time.monotonic()-t0:.1f}s", flush=True)

ll_per_pair = total_ll / max(n_pairs, 1)
print("=" * 60)
print(f"[{MODEL}] MATCH-ALIGNED (ordering-summed) RESULT")
print(f"  n_pairs={n_pairs}  n_bad={n_bad}")
print(f"  total_ll = {total_ll:.4f}")
print(f"  LL/pair  = {ll_per_pair:.6f}")
for e in bad_examples:
    print(f"  BAD idx={e[0]} Lx={e[1]} Ly={e[2]} t={e[3]:.4f} lp={e[4]:.3e}")
print("=" * 60)
