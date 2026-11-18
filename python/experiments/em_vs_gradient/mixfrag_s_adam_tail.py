#!/usr/bin/env python3
"""Adam tail for a MixFrag-S svi-BW fit (the MixFrag-S analogue of the
``train_pfam.py --adam`` switch in balibase_exclusion/launch_retrain_nobali.sh).

Starting from a contract checkpoint (train_mixfrag_s.py; e.g. the 1,620-
iteration svi-BW fit), runs Adam at a fixed learning rate on minibatches of
the training corpus on the recipe's MAP objective

    [ sum_{pairs in batch} log P(alignment, x, y | theta) ] / n_batch
    + log_prior(theta) / N_total

(the per-pair stochastic estimate of (observed 1D path-conditioned LL +
log-prior) / N_total; see mixfrag_s_tail_common for the prior).  Gradients
come from the closed-form (custom-VJP) route: one 1D E-step per step, then
the score formulas on the expected sufficient statistics.

Every step logs the minibatch objective, the parameter move, lam, mu, exts,
w and each fragtype's pi.  Output: the contract npz (``--out``) plus
``<out>_history.json``; ``--resume`` continues from the sidecar
``<out>.adam_state.npz`` (Adam moments + step), written every --ckpt-every
steps together with the contract npz.

Recipe (launch_retrain_nobali.sh mirror): --adam-steps 2000 --adam-lr 0.01
--adam-batch 2000 --seed 42 on pfam/precompiled_v3_train_nobali.

Usage:
  mixfrag_s_adam_tail.py --init pfam/svibw_mixfrags_F2_nobali_1620.npz \
      --precompiled pfam/precompiled_v3_train_nobali \
      --out pfam/adam_mixfrags_F2_nobali_from1620.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mixfrag_s_tail_common as C                                   # noqa: E402
import jax                                                          # noqa: E402
import jax.numpy as jnp                                             # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--init', required=True, help='contract npz to start from')
ap.add_argument('--out', required=True, help='output contract npz')
ap.add_argument('--precompiled', default='pfam/precompiled_v3_train_nobali')
ap.add_argument('--adam-steps', type=int, default=2000)
ap.add_argument('--adam-lr', type=float, default=0.01)
ap.add_argument('--adam-batch', type=int, default=2000)
ap.add_argument('--seed', type=int, default=42,
                help='minibatch k is drawn with default_rng([seed, k])')
ap.add_argument('--n-total-pairs', type=int, default=0,
                help='N_total weighting the prior (default: #pairs loaded)')
ap.add_argument('--n-pairs-subset', type=int, default=0,
                help='train on a random subset of this many pairs (tests)')
ap.add_argument('--subset-seed', type=int, default=0)
ap.add_argument('--select', choices=('final', 'best'), default='final',
                help="parameters written to --out: the last step's (default) or "
                     "the step with the best minibatch objective (the "
                     "train_pfam --adam rule)")
ap.add_argument('--ckpt-every', type=int, default=50)
ap.add_argument('--resume', action='store_true')
ap.add_argument('--log-every', type=int, default=1)
ap.add_argument('--log-file', default=None)
C.add_prior_args(ap)
C.add_estep_args(ap)
args = ap.parse_args()

C.setup_jax(args)
t_start = time.time()
log = C.make_logger(args.log_file, t_start)
log(f'MixFrag-S Adam tail: backend={jax.default_backend()} args={vars(args)}')

p0, F, A, ck0 = C.load_contract(args.init)
em_iter0 = int(ck0['em_iter'])
parent_config = json.loads(str(ck0['_config'])) if '_config' in ck0 else {}
log(f'  init {args.init}: F={F} A={A} em_iter={em_iter0} lam={p0["lam"]:.6f} '
    f'mu={p0["mu"]:.6f} exts={C.fmt_vec(p0["exts"])} w={C.fmt_vec(p0["weights"])}')
packer = C.Packer(F, A)
priors = C.priors_from_args(args)
vg, lp_fn = C.make_value_grad(packer, priors)
ek = C.estep_kwargs(args)

pairs = C.load_path_pairs(args.precompiled, log)
if args.n_pairs_subset and args.n_pairs_subset < len(pairs):
    rng = np.random.default_rng(args.subset_seed)
    ix = np.sort(rng.choice(len(pairs), args.n_pairs_subset, replace=False))
    pairs = [pairs[i] for i in ix]
    log(f'  subset: {len(pairs)} pairs (seed {args.subset_seed})')
n_total = int(args.n_total_pairs) if args.n_total_pairs else len(pairs)
prior_weight = 1.0 / float(n_total)
batch = min(args.adam_batch, len(pairs))

import optax                                                        # noqa: E402
opt = optax.adam(args.adam_lr)
v = jnp.asarray(packer.pack(p0))
opt_state = opt.init(v)
step0 = 0
history = []
best = dict(obj=-np.inf, step=-1, v=np.asarray(v))
sidecar = args.out + '.adam_state.npz'
hist_path = args.out.replace('.npz', '_history.json')
if args.resume and os.path.exists(sidecar):
    sd = np.load(sidecar, allow_pickle=False)
    v = jnp.asarray(sd['v'])
    cnt, mu_s, nu_s = sd['count'], sd['mu'], sd['nu']
    opt_state = (optax.ScaleByAdamState(count=jnp.asarray(cnt), mu=jnp.asarray(mu_s),
                                        nu=jnp.asarray(nu_s)),) + tuple(opt_state[1:])
    step0 = int(sd['step'])
    best = dict(obj=float(sd['best_obj']), step=int(sd['best_step']), v=np.asarray(sd['best_v']))
    if os.path.exists(hist_path):
        history = json.load(open(hist_path))['history']
    log(f'Resuming from {sidecar} at step {step0} (history {len(history)} records)')
elif args.resume:
    log(f'--resume: no sidecar {sidecar}; starting fresh.')

config = dict(tail='adam', model='MixFrag-S', dp_mode='path', from_checkpoint=args.init,
              from_em_iter=em_iter0, adam_steps=args.adam_steps, adam_lr=args.adam_lr,
              adam_batch=batch, seed=args.seed, n_total_pairs=n_total,
              n_pairs_loaded=len(pairs), precompiled=args.precompiled, select=args.select,
              priors=priors, parent_config=parent_config)


def save(step, final=False):
    v_sel = best['v'] if args.select == 'best' else np.asarray(v)
    p_sel = packer.natural(v_sel)
    cfg = dict(config, steps_done=int(step), final=bool(final),
               selected_step=int(best['step']) if args.select == 'best' else int(step))
    C.save_contract(args.out, p_sel, F, em_iter0, cfg, history)
    st = opt_state[0]
    tmp = sidecar + f'.tmp.{os.getpid()}.npz'   # explicit .npz: np.savez appends it otherwise
    np.savez(tmp, v=np.asarray(v), count=np.asarray(st.count), mu=np.asarray(st.mu),
             nu=np.asarray(st.nu), step=np.int64(step), best_obj=np.float64(best['obj']),
             best_step=np.int64(best['step']), best_v=np.asarray(best['v']))
    os.replace(tmp, sidecar)
    C.dump_history(hist_path, dict(args=vars(args), config=cfg), history)
    log(f'  [checkpoint step={step} -> {args.out}; sidecar {sidecar}]')


log(f'Adam: {args.adam_steps} steps, lr={args.adam_lr}, batch={batch}, N_total={n_total} '
    f'(prior weight 1/N_total), {packer.n_params} unconstrained parameters; start step {step0}')
t0 = time.time()
for step in range(step0, args.adam_steps):
    rng = np.random.default_rng([int(args.seed), int(step)])
    ix = rng.choice(len(pairs), batch, replace=len(pairs) < batch)
    mb = [pairs[i] for i in ix]
    p = packer.natural(v)
    t_e = time.time()
    ll, suff, n_mb = C.estep_pairs(mb, p, **ek)
    t_e = time.time() - t_e
    sj = C.suff_to_jnp(suff)
    scale = 1.0 / float(n_mb)
    sj = {k: sj[k] * scale for k in sj}                              # per-pair Q
    _q, g = vg(v, sj, prior_weight)
    lp = float(lp_fn(v))
    obj = ll / n_mb + prior_weight * lp                              # per-pair MAP estimate
    upd, opt_state = opt.update(-g, opt_state, v)                    # ascent
    v_new = optax.apply_updates(v, upd)
    dv = float(jnp.max(jnp.abs(v_new - v)))
    dv2 = float(jnp.linalg.norm(v_new - v))
    v = v_new
    if obj > best['obj']:
        best = dict(obj=obj, step=step, v=np.asarray(v))
    rec = dict(step=step, wall=time.time() - t0, obj_per_pair=obj, obs_ll_per_pair=ll / n_mb,
               obs_ll=ll, log_prior=lp, n_pairs=n_mb, grad_norm=float(jnp.linalg.norm(g)),
               dv_max=dv, dv_norm=dv2, estep_sec=t_e, lam=p['lam'], mu=p['mu'],
               exts=p['exts'], weights=p['weights'], pi=p['pi'])
    history.append(rec)
    if step % args.log_every == 0 or step == args.adam_steps - 1:
        log(f'  step {step:4d}/{args.adam_steps}: MAP/pair={obj:.4f} obs_LL/pair={ll / n_mb:.4f} '
            f'(obs_LL={ll:.1f} n={n_mb} logprior={lp:.2f}) |g|={rec["grad_norm"]:.3e} '
            f'|dv|max={dv:.2e} lam={p["lam"]:.6f} mu={p["mu"]:.6f} exts={C.fmt_vec(p["exts"])} '
            f'w={C.fmt_vec(p["weights"])} (estep {t_e:.1f}s; {time.time() - t0:.0f}s)')
        for f in range(F):
            log(f'      pi[{f}]={C.fmt_vec(p["pi"][f])}')
    if args.ckpt_every and (step + 1) % args.ckpt_every == 0 and step + 1 < args.adam_steps:
        save(step + 1)

p_end = packer.natural(v)
log(f'Adam done: {len(history)} steps; best minibatch MAP/pair={best["obj"]:.4f} at step '
    f'{best["step"]}; final lam={p_end["lam"]:.6f} mu={p_end["mu"]:.6f} '
    f'exts={C.fmt_vec(p_end["exts"])} w={C.fmt_vec(p_end["weights"])}')
for f in range(F):
    log(f'  final pi[{f}]={C.fmt_vec(p_end["pi"][f])}')
save(args.adam_steps, final=True)
log(f'saved {args.out}')
