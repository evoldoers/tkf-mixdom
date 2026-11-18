#!/usr/bin/env python3
"""Convert between train_pfam MixDom checkpoints for the late-stage switch.

  to-maraschino <ckpt.npz> <out.npz>
      Identity-classdist MixDom2 checkpoint -> Maraschino init file whose
      per-domain (S, pi) are the LIVE per-class matrices (class_S_exch,
      class_pis); dom_S_exch in such checkpoints is stale.
  to-scoring <adam_ckpt.npz> <template_ckpt.npz> <out.npz>
      MixDom1 per-domain Adam checkpoint -> MixDom2-identity-format checkpoint
      (class_* = dom_*, classdist = identity, _config copied from template) so
      score_pathcond_val.py, score_matchaligned_val.py and the BAliBASE
      harness read it like the svi-BW checkpoints.
"""
import json, os, sys
import numpy as np
os.environ.setdefault("JAX_ENABLE_X64", "1"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
mode = sys.argv[1]; _a = sys.argv; sys.argv = ['x']
import train_pfam
sys.argv = _a

if mode == 'to-maraschino':
    src, out = sys.argv[2], sys.argv[3]
    d = np.load(src, allow_pickle=True)
    n_dom = int(np.asarray(d['dom_ins']).shape[0])
    params = {k: (float(d[k]) if d[k].shape == () else d[k].copy()) for k in
              ('main_ins', 'main_del', 'dom_ins', 'dom_del', 'dom_weights', 'frag_weights', 'ext_rates')}
    if 'class_S_exch' in d.files:
        cd = np.asarray(d['classdist']).reshape(n_dom, -1, np.asarray(d['class_S_exch']).shape[0])
        assert np.allclose(cd[:, 0, :], np.eye(n_dom)), "classdist is not identity; cannot map classes to domains"
        params['dom_S_exch'] = np.asarray(d['class_S_exch']).copy()
        params['dom_pis'] = np.asarray(d['class_pis']).copy()
    else:
        params['dom_S_exch'] = np.asarray(d['dom_S_exch']).copy(); params['dom_pis'] = np.asarray(d['dom_pis']).copy()
    n_frag = int(np.asarray(params['frag_weights']).shape[1]) if np.asarray(params['frag_weights']).ndim > 1 else 1
    train_pfam._save_params_as_maraschino(out, params, n_dom, n_frag)
    print('wrote', out)
elif mode == 'to-scoring':
    src, tpl, out = sys.argv[2], sys.argv[3], sys.argv[4]
    d = dict(np.load(src, allow_pickle=True)); t = np.load(tpl, allow_pickle=True)
    n_dom = int(np.asarray(d['dom_ins']).shape[0]); AA = np.asarray(d['dom_pis']).shape[1]
    S = np.asarray(d['dom_S_exch']); pis = np.asarray(d['dom_pis'])
    Qs = np.zeros((n_dom, AA, AA))
    for i in range(n_dom):
        Q = S[i] * pis[i][None, :]; np.fill_diagonal(Q, 0.0); Q[np.diag_indices(AA)] = -Q.sum(axis=1); Qs[i] = Q
    d['dom_Qs'] = Qs; d['class_S_exch'] = S.copy(); d['class_pis'] = pis.copy()
    n_frag = int(np.asarray(d['frag_weights']).shape[1]) if np.asarray(d['frag_weights']).ndim > 1 else 1
    d['classdist'] = np.broadcast_to(np.eye(n_dom)[:, None, :], (n_dom, n_frag, n_dom)).copy()
    d['n_classes_frag'] = np.int64(n_dom)
    for k in ('_config', 'em_iter', 't_rep', 'log_probs', 'step_file_idx'):
        if k in t.files and k not in d: d[k] = t[k]
    np.savez_compressed(out, **d); print('wrote', out)
else:
    raise SystemExit('unknown mode')
