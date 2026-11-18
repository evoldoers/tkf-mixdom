"""Ordering-summed (match-aligned 2D forward) val LL/pair on v3_val for all
three models. Matches are FIXED to the observed alignment (i,j); the 2D DP
sums over ALL indel orderings between match anchors (NOT a fixed path).

Models:
  MODEL=d3f1     : MixDom d3f1(c3), per-class emissions + chi from checkpoint
  MODEL=mixfrag  : MixFrag F2 pair HMM (make_mixfrag_pair_hmm)
  MODEL=tkf92    : TKF92 pair HMM  (make_tkf92_pair_hmm)
  MODEL=mixfrag_s: MixFrag-S: MixFrag pair HMM in which fragtype f has its own
                   substitution model (Q_f, pi_f), from the checkpoint npz named
                   by MIXFRAG_S_NPZ (keys ins_rate, del_rate, exts, weights, Q
                   (F,A,A), pi (F,A)); emissions via pair_hmm_emissions_per_fragtype

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
PATHCOND = os.environ.get("PATHCOND", "0") == "1"   # 1 = condition on the full path (indel ordering fixed), not only the matches
PINNED = os.environ.get("PINNED", "0") == "1"       # 1 = every reference match is forced (no insert+delete bypass of a match); 0 = the
                                                     #     historical convention: M may fire only at reference matches but a match may be skipped

from tkfmixdom.jax.core.protein import rate_matrix_lg
from tkfmixdom.jax.core.ctmc import transition_matrix
from tkfmixdom.jax.simulate.msa import alignment_to_states
from tkfmixdom.jax.util.padding import pad_to_bin as _pad_to_bin
from tkfmixdom.jax.dp.hmm import (
    NEG_INF, mask_emissions_match_aligned, mask_emissions_match_pinned, forward_backward_2d,
    pair_hmm_emissions, pair_hmm_emissions_per_class, pair_hmm_emissions_per_domain,
    pair_hmm_emissions_per_fragtype)
from tkfmixdom.jax.core.params import M as _M, I as _I, D as _D

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}
AA = len(AMINO_ACIDS)
VAL_DIR = os.environ.get('VAL_DIR', '/home/yam/tkf-mixdom/python/pfam/precompiled_v3_val')
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
def extract_path_cells(states):
    """Grid cells (1-indexed, as in the 2D DP) visited by I and D states along the path."""
    i = j = 0; ii = []; ij = []; di = []; dj = []
    for s in states:
        if s == _M: i += 1; j += 1
        elif s == _I: j += 1; ii.append(i); ij.append(j)
        elif s == _D: i += 1; di.append(i); dj.append(j)
    return (np.array(ii, dtype=np.int32), np.array(ij, dtype=np.int32), np.array(di, dtype=np.int32), np.array(dj, dtype=np.int32))
def mask_emissions_path(emit, state_types, ii, ij, di, dj):
    """On top of the match mask: I-type states may fire only at the path's insert cells and D-type
    states only at its delete cells, so the forward sums over the single observed path (fragtype /
    domain assignments still summed). Cells are 1-indexed grid coordinates; padding slots repeat a
    real cell (or (0,0), which no I/D state can reach)."""
    Lx1, Ly1, ns = emit.shape
    imask = jnp.zeros((Lx1, Ly1), dtype=bool).at[ii, ij].set(True)
    dmask = jnp.zeros((Lx1, Ly1), dtype=bool).at[di, dj].set(True)
    pen = jnp.where((state_types == _I)[None, None, :] & ~imask[:, :, None], NEG_INF, 0.0) \
        + jnp.where((state_types == _D)[None, None, :] & ~dmask[:, :, None], NEG_INF, 0.0)
    return emit + pen

# ---- read shards: x_int, y_int, match_i, match_j, t ----
t0 = time.monotonic()
pairs = []  # (x_int, y_int, mi, mj, t)
fams = []   # family id per kept pair (for family-level mixture scoring: combine_k20.py)
MAX_PAIRS = int(os.environ.get("MAX_PAIRS", "0"))   # >0: smoke test on the first MAX_PAIRS scorable pairs
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
            pairs.append((x_int, y_int, mi, mj, float(r['t']), extract_path_cells(np.array(states, dtype=np.int32))))
            fams.append(r.get('fam', ''))
            if MAX_PAIRS and len(pairs) >= MAX_PAIRS: break
    if MAX_PAIRS and len(pairs) >= MAX_PAIRS: break
print(f"[{MODEL}] read {n_records} records, {len(pairs)} scorable pairs, {time.monotonic()-t0:.1f}s", flush=True)

# Optional length cap: score only pairs with max(Lx,Ly) <= LMAX. This is a
# deterministic, well-defined subset (identical across models) that keeps the
# 2D DP tractable — the full-grid ordering-summed DP is O(Lx*Ly*ns) per pair
# and blows up on the long tail. Reported honestly as an "L<=LMAX" subset.
LMAX = int(os.environ.get("LMAX", "0"))
if LMAX > 0:
    keep_idx = [i for i, p in enumerate(pairs) if len(p[0]) <= LMAX and len(p[1]) <= LMAX]
    print(f"[{MODEL}] LMAX={LMAX}: scoring {len(keep_idx)}/{len(pairs)} pairs "
          f"({100*len(keep_idx)/len(pairs):.1f}%) with max(Lx,Ly)<={LMAX}", flush=True)
    pairs = [pairs[i] for i in keep_idx]; fams = [fams[i] for i in keep_idx]

# ---- model setup ----
if MODEL == 'd3f1':
    from train_pfam import _load_checkpoint, _build_log_chi_stack
    from tkfmixdom.jax.models.mixdom import (
        n_states as mixdom_n_states, state_types as mixdom_state_types)
    CKPT = os.environ.get('CKPT', '/tmp/claude-1003/-home-yam-tkf-dp/e986c820-957e-43e6-8345-cb50bd010f86/scratchpad/sharedexp/svibw_d3f1_v3_FROZEN_iter45.npz')
    (params, *_rest) = _load_checkpoint(CKPT)
    cfg = json.loads(str(np.load(CKPT, allow_pickle=True)['_config']))
    n_dom = int(cfg['n_dom']); n_frag = int(cfg['n_frag'])
    st = np.asarray(mixdom_state_types(n_dom, n_frag))
    ns = len(st)
    if 'n_classes' not in params or 'class_S_exch' not in params:
        # MixDom1-style checkpoint (e.g. n_classes=1 / d1f*): the per-domain (S, pi) are the live
        # substitution parameters; express them as one class per domain with an identity class map.
        params['class_S_exch'] = np.asarray(params['dom_S_exch']); params['class_pis'] = np.asarray(params['dom_pis'])
        params['classdist'] = np.broadcast_to(np.eye(n_dom)[:, None, :], (n_dom, n_frag, n_dom)).copy(); params['n_classes'] = n_dom
        print(f"[{MODEL}] no class keys in checkpoint: using per-domain (S, pi) as {n_dom} class(es)", flush=True)
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

elif MODEL in ('mixfrag', 'tkf92', 'cherryml_C20'):
    from tkfmixdom.jax.models.left_regular import (
        make_mixfrag_pair_hmm, make_tkf92_pair_hmm, S, M, I, D, E)
    C20 = None
    if MODEL == 'cherryml_C20':
        # TKF92 (anchor indel parameters of the K=20 / C=20 baselines) with the CherryML 20-class site mixture as
        # emission model.  A site mixture is a per-column mixture, so the pair HMM's match emission is
        # sum_c w_c pi_c(a) P_c(a->b | t) and its insert/delete emission sum_c w_c pi_c(a); both are reproduced by a
        # single effective (pi', sub'(t)) with pi' = sum_c w_c pi_c and sub'(t)[a,b] = sum_c w_c pi_c(a) P_c(a->b|t) / pi'(a).
        _c = np.load(os.environ.get('CHERRYML_NPZ', 'pfam/cherryml_mixture_C20_n5000.npz'), allow_pickle=True)
        _S = np.asarray(_c['S'], float)[:, :AA, :AA]; _pi21 = np.asarray(_c['pi'], float); _pi21 = _pi21 / _pi21.sum(1, keepdims=True)
        _pic = _pi21[:, :AA] / _pi21[:, :AA].sum(1, keepdims=True); _w = np.asarray(_c['weights'], float); _w = _w / _w.sum()
        _Qc = _S * _pic[:, None, :]; _Qc[:, np.arange(AA), np.arange(AA)] = 0.0; _Qc[:, np.arange(AA), np.arange(AA)] = -_Qc.sum(2)
        C20 = dict(Q=jnp.asarray(_Qc), pi=jnp.asarray(_pic), w=jnp.asarray(_w))
        pi_lg = np.asarray(_w @ _pic)           # pi' (used for the insert / delete emissions and as the HMM's pi)
        _anchor = json.load(open(os.environ.get('TKF92_PARAMS', 'experiments/tkf92_anchor_mixtures.json')))
        print(f"[cherryml_C20] {len(_w)} classes from {os.environ.get('CHERRYML_NPZ', 'pfam/cherryml_mixture_C20_n5000.npz')}; indel params {_anchor}", flush=True)
    if MODEL in ('tkf92', 'mixfrag') and os.environ.get('SUBST_NPZ'):
        # (Q, pi) replacing LG08: a component of the K=20 TKF92 mixture (k20_components.py) or the refitted GTR
        _z = np.load(os.environ['SUBST_NPZ'], allow_pickle=True); Q_lg = np.asarray(_z['Q'], float); pi_lg = np.asarray(_z['pi'], float)
        print(f"[tkf92] substitution model from {os.environ['SUBST_NPZ']}", flush=True)
    if MODEL == 'mixfrag':
        # default: the fitted F=2 log-space Pfam fit (the set the paper describes). Until 2026-11-13 the default was a
        # hard-coded older set (0.04608/0.04709, exts 0.37466/0.83325, weights 0.62848/0.37152); runs made with it are
        # archived as *_scorerdefault.log.
        _mpf = os.environ.get('MIXFRAG_PARAMS') or 'experiments/mixfrag_F/mixfrag_F2_logspace.json'
        if True:      # JSON with lam/mu/exts/weights (any number of fragtypes)
            os.environ['MIXFRAG_PARAMS'] = _mpf
            _mp = json.load(open(_mpf))
            P = dict(ins_rate=float(_mp['lam']), del_rate=float(_mp['mu']), exts=[float(v) for v in _mp['exts']], weights=[float(v) for v in _mp['weights']])
            print(f"[mixfrag] params from {os.environ['MIXFRAG_PARAMS']}: {P}", flush=True)
        def make_hmm(t):
            return make_mixfrag_pair_hmm(P['ins_rate'], P['del_rate'], t,
                jnp.asarray(P['exts']), jnp.asarray(P['weights']),
                jnp.asarray(Q_lg), jnp.asarray(pi_lg))
    else:
        P = dict(ins_rate=0.04428, del_rate=0.04521, ext=0.677)
        _tpf = os.environ.get('TKF92_PARAMS') or ('experiments/tkf92_anchor_mixtures.json' if MODEL == 'cherryml_C20' else None)
        if _tpf:     # JSON with ins_rate / del_rate / ext (K=20 component) or ext_rate (the K=20 / C=20 anchor TKF92)
            _tp = json.load(open(_tpf)); P = dict(ins_rate=float(_tp['ins_rate']), del_rate=float(_tp['del_rate']), ext=float(_tp.get('ext', _tp.get('ext_rate'))))
            print(f"[{MODEL}] TKF92 params from {_tpf}: {P}", flush=True)
        def make_hmm(t):
            return make_tkf92_pair_hmm(P['ins_rate'], P['del_rate'], t, P['ext'],
                jnp.asarray(Q_lg), jnp.asarray(pi_lg))
    lt0, st0, _, _ = make_hmm(pairs[0][4]); st = np.asarray(st0); ns = len(st)
    print(f"[{MODEL}] ns={ns} state_types={st}", flush=True)

    def build_log_trans_jax(t):
        lt, _, _, _ = make_hmm(t)
        return jnp.maximum(lt, LOG_FLOOR)
elif MODEL == 'mixfrag_s':
    # MixFrag-S: the MixFrag pair HMM (shared lam, mu; per-fragtype r_f, w_f) with a substitution model
    # (Q_f, pi_f) per fragtype. Transitions come from make_mixfrag_pair_hmm, whose log_trans does not depend
    # on the (Q, pi) it is handed; the per-fragtype emissions are built in the kernel from P_f(t) = exp(Q_f t).
    from tkfmixdom.jax.models.left_regular import make_mixfrag_pair_hmm
    MIXFRAG_S_NPZ = os.environ.get('MIXFRAG_S_NPZ')
    if not MIXFRAG_S_NPZ: raise SystemExit("MODEL=mixfrag_s needs MIXFRAG_S_NPZ (checkpoint npz)")
    _z = np.load(MIXFRAG_S_NPZ, allow_pickle=True)
    P = dict(ins_rate=float(_z['ins_rate']), del_rate=float(_z['del_rate']),
             exts=[float(v) for v in np.asarray(_z['exts'], float)], weights=[float(v) for v in np.asarray(_z['weights'], float)])
    frag_Qs = np.asarray(_z['Q'], float); frag_pis = np.asarray(_z['pi'], float)
    n_fragtypes = len(P['exts'])
    assert frag_Qs.shape == (n_fragtypes, AA, AA) and frag_pis.shape == (n_fragtypes, AA), (frag_Qs.shape, frag_pis.shape, n_fragtypes)
    print(f"[{MODEL}] params from {MIXFRAG_S_NPZ}: F={n_fragtypes} {P} em_iter={_z['em_iter'] if 'em_iter' in _z.files else '?'}", flush=True)
    def make_hmm(t):
        return make_mixfrag_pair_hmm(P['ins_rate'], P['del_rate'], t,
            jnp.asarray(P['exts']), jnp.asarray(P['weights']),
            jnp.asarray(frag_Qs[0]), jnp.asarray(frag_pis[0]))
    lt0, st0, _, _ = make_hmm(pairs[0][4]); st = np.asarray(st0); ns = len(st)
    assert ns == 3 * n_fragtypes + 2, (ns, n_fragtypes)
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

    def _emit_and_fwd(log_chi, x, y, rlx, rly, csub, mi, mj, ii, ij, di, dj):
        et = pair_hmm_emissions_per_class(
            st_j, x, y, csub, class_pis_j, classdist_j, n_dom, n_frag)
        et = (mask_emissions_match_pinned if PINNED else mask_emissions_match_aligned)(et, st_j, mi, mj)
        if PATHCOND: et = mask_emissions_path(et, st_j, ii, ij, di, dj)
        return forward_backward_2d(log_chi, st_j, x, y, None, None,
                                   log_emit_table=et, real_Lx=rlx, real_Ly=rly,
                                   forward_only=True)
    _kernel = jax.jit(jax.vmap(_emit_and_fwd, in_axes=(0,) * 12))
    HAS_SUBS = True
elif MODEL == 'mixfrag_s':
    _lt_v = jax.jit(jax.vmap(build_log_trans_jax))
    def batched_log_trans(tarr):
        return np.asarray(_lt_v(jnp.asarray(tarr)))
    # batched per-fragtype sub matrices: (B, F, A, A)
    _fsub_v = jax.jit(jax.vmap(lambda tt: jax.vmap(lambda Qf: transition_matrix(Qf, tt))(jnp.asarray(frag_Qs))))
    def batched_subs(tarr):
        return np.asarray(_fsub_v(jnp.asarray(tarr)))  # (B,F,A,A)
    frag_pis_j = jnp.asarray(frag_pis)

    def _emit_and_fwd(log_chi, x, y, rlx, rly, fsub, mi, mj, ii, ij, di, dj):
        et = pair_hmm_emissions_per_fragtype(st_j, x, y, fsub, frag_pis_j)
        et = (mask_emissions_match_pinned if PINNED else mask_emissions_match_aligned)(et, st_j, mi, mj)
        if PATHCOND: et = mask_emissions_path(et, st_j, ii, ij, di, dj)
        return forward_backward_2d(log_chi, st_j, x, y, None, None,
                                   log_emit_table=et, real_Lx=rlx, real_Ly=rly,
                                   forward_only=True)
    _kernel = jax.jit(jax.vmap(_emit_and_fwd, in_axes=(0,) * 12))
    HAS_SUBS = True
else:
    _lt_v = jax.jit(jax.vmap(build_log_trans_jax))
    def batched_log_trans(tarr):
        return np.asarray(_lt_v(jnp.asarray(tarr)))
    if MODEL == 'cherryml_C20':
        def _mix_sub(tt):
            Pc = jax.vmap(lambda Qc: transition_matrix(Qc, tt))(C20['Q'])            # (C,A,A)
            joint = jnp.einsum('c,ca,cab->ab', C20['w'], C20['pi'], Pc)             # sum_c w_c pi_c(a) P_c(a->b)
            return joint / jnp.asarray(pi_lg)[:, None]                               # sub'(t): rows sum to 1
        _sub_v = jax.jit(jax.vmap(_mix_sub))
    else:
        _sub_v = jax.jit(jax.vmap(lambda tt: transition_matrix(jnp.asarray(Q_lg), tt)))
    def batched_subs(tarr):
        return np.asarray(_sub_v(jnp.asarray(tarr)))  # (B,A,A)
    pi_lg_j = jnp.asarray(pi_lg)

    def _emit_and_fwd(log_chi, x, y, rlx, rly, sub, mi, mj, ii, ij, di, dj):
        et = pair_hmm_emissions(st_j, x, y, sub, pi_lg_j)
        et = (mask_emissions_match_pinned if PINNED else mask_emissions_match_aligned)(et, st_j, mi, mj)
        if PATHCOND: et = mask_emissions_path(et, st_j, ii, ij, di, dj)
        return forward_backward_2d(log_chi, st_j, x, y, None, None,
                                   log_emit_table=et, real_Lx=rlx, real_Ly=rly,
                                   forward_only=True)
    _kernel = jax.jit(jax.vmap(_emit_and_fwd, in_axes=(0,) * 12))
    HAS_SUBS = True

# Precompute all log_trans and subs in chunks (batched over t).
PRE = 2048
log_trans_all = np.zeros((len(pairs), ns, ns), dtype=np.float64)
if MODEL == 'd3f1':
    subs_all = np.zeros((len(pairs), n_cls, AA, AA), dtype=np.float64)
elif MODEL == 'mixfrag_s':
    subs_all = np.zeros((len(pairs), n_fragtypes, AA, AA), dtype=np.float64)
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
for idx, (x_int, y_int, mi, mj, t, _cells) in enumerate(pairs):
    Lxp = _pad_to_bin(len(x_int)); Lyp = _pad_to_bin(len(y_int))
    buckets[(Lxp, Lyp)].append(idx)

# Batch size: cap by (a) a memory budget on the (B, Lxp+1, Lyp+1, ns) forward
# table (~2 GB fp64) and (b) a large cell budget so small grids get big batches
# (fewer kernel launches → far less dispatch overhead than the earlier 6M cap).
MEM_BUDGET_BYTES = int(float(os.environ.get('MEM_BUDGET_MB', '400')) * 1e6)   # output-table budget (DP intermediates ~5-8x); was 700 MB, lowered so d3f1 PATHCOND=1 runs beside other GPU jobs
MAX_B = 2048                      # absolute batch cap (bounds wavefront vmap mem)
total_ll = 0.0; n_pairs = 0; n_bad = 0
bad_examples = []; n_done = 0
ll_arr = np.full(len(pairs), np.nan)   # per-pair log-likelihoods (PERPAIR_OUT)
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
        ni = max(1, max(len(pairs[i][5][0]) for i in chunk)); nd = max(1, max(len(pairs[i][5][2]) for i in chunk))
        iis = np.zeros((bb, ni), dtype=np.int32); ijs = np.zeros((bb, ni), dtype=np.int32)
        dis = np.zeros((bb, nd), dtype=np.int32); djs = np.zeros((bb, nd), dtype=np.int32)
        for bi, i in enumerate(chunk):
            x_int, y_int, mi, mj, t, (ii, ij, di, dj) = pairs[i]
            if len(ii): iis[bi, :len(ii)] = ii; ijs[bi, :len(ij)] = ij; iis[bi, len(ii):] = ii[0]; ijs[bi, len(ij):] = ij[0]
            if len(di): dis[bi, :len(di)] = di; djs[bi, :len(dj)] = dj; dis[bi, len(di):] = di[0]; djs[bi, len(dj):] = dj[0]
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
            jnp.asarray(mis), jnp.asarray(mjs), jnp.asarray(iis), jnp.asarray(ijs), jnp.asarray(dis), jnp.asarray(djs)))
        for bi, v in enumerate(lp):
            if np.isfinite(v) and v > -1e10:
                total_ll += float(v); n_pairs += 1; ll_arr[chunk[bi]] = float(v)
            else:
                n_bad += 1
                if len(bad_examples) < 8:
                    i = chunk[bi]
                    bad_examples.append((i, len(pairs[i][0]), len(pairs[i][1]), pairs[i][4], float(v)))
        n_done += bb
    print(f"  [{MODEL}] grid ({Lxp},{Lyp}) x{len(idxs)} done  n_done={n_done}/{len(pairs)}  {time.monotonic()-t0:.1f}s", flush=True)

ll_per_pair = total_ll / max(n_pairs, 1)
if os.environ.get('PERPAIR_OUT'):
    np.savez(os.environ['PERPAIR_OUT'], ll=ll_arr, fam=np.array(fams), t=ts_all)
    print(f"[{MODEL}] per-pair log-likelihoods -> {os.environ['PERPAIR_OUT']}", flush=True)
print("=" * 60)
print(f"[{MODEL}] " + ("PATH-CONDITIONED (full path fixed) RESULT" if PATHCOND else ("MATCH-PINNED (ordering-summed, matches forced) RESULT" if PINNED else "MATCH-ALIGNED (ordering-summed, matches permitted not forced) RESULT")))
print(f"  n_pairs={n_pairs}  n_bad={n_bad}")
print(f"  total_ll = {total_ll:.4f}")
print(f"  LL/pair  = {ll_per_pair:.6f}")
for e in bad_examples:
    print(f"  BAD idx={e[0]} Lx={e[1]} Ly={e[2]} t={e[3]:.4f} lp={e[4]:.3e}")
print("=" * 60)
