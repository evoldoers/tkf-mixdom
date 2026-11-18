#!/usr/bin/env python3
"""Site-class mixture of 20-state GTR substitution models, fitted by EM-around-CherryML on the precompiled Pfam
training pairs: one latent class per aligned column, the model that the pair-HMM emission mixture represents
(sum_c w_c pi_c(i) P_c(i -> j | t)), at the same Poisson-corrected pair times t that every other model in the
paper is trained and scored with (em_around_cherryml.py bins its counts by FastTree tree distances instead, a
different clock, which is why its classes score below LG08 on the precompiled pairs).

Counts: N[b, i, j] over the matched residue columns of all pairs, t binned geometrically.  Because each column is
its own site with a single (b, i, j) entry, the E-step is closed-form on the count table,
    r[b, i, j, c] = w_c pi_c(i) P_c(i, j | t_b) / sum_c' (...),
and the M-step for class c maximizes sum_{b,i,j} r[b,i,j,c] N[b,i,j] (log pi_c(i) + log P_c(i, j | t_b)) over the
exchangeabilities and equilibrium of class c (Adam on log S and pi logits), with w_c = the responsibility mass.

Usage (from ~/tkf-mixdom/python):
  JAX_ENABLE_X64=1 .venv/bin/python experiments/fit_paircolumn_sitemix.py --precompiled pfam/precompiled_v3_train \
      --n-classes 20 --output pfam/cherryml_mixture_C20_paircol.npz
Output npz: S (C,20,20) symmetric exchangeabilities, pi (C,20), weights (C,), t_centers, n_states=20, history.
"""
import argparse, glob, io, json, os, re, sys, time
import numpy as np
os.environ.setdefault('JAX_ENABLE_X64', '1')
import jax, jax.numpy as jnp, jax.scipy.linalg
import zstandard
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tkfmixdom.jax.core.protein import rate_matrix_lg

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"; AA = 20
AA_TO_IDX = {a: i for i, a in enumerate(AMINO_ACIDS)}
CIGAR_RE = re.compile(r'([MDI])(\d+)')
LUT = np.full(256, -1, dtype=np.int64)
for a, i in AA_TO_IDX.items(): LUT[ord(a)] = i


def match_columns(x, a, y):
    """Residue index pairs (i, j) of the M columns of a CIGAR-encoded alignment (non-standard residues dropped)."""
    xi = yi = 0; out_i = []; out_j = []
    for op, n in CIGAR_RE.findall(a):
        n = int(n)
        if op == 'M':
            out_i.append(LUT[np.frombuffer(x[xi:xi + n].encode(), dtype=np.uint8)])
            out_j.append(LUT[np.frombuffer(y[yi:yi + n].encode(), dtype=np.uint8)])
            xi += n; yi += n
        elif op == 'I': yi += n
        else: xi += n
    if not out_i: return np.zeros(0, np.int64), np.zeros(0, np.int64)
    i = np.concatenate(out_i); j = np.concatenate(out_j); ok = (i >= 0) & (j >= 0)
    return i[ok], j[ok]


def build_counts(precompiled, t_edges, max_pairs=0):
    n_bins = len(t_edges) - 1; keys = []; n = 0; t0 = time.time()
    dctx = zstandard.ZstdDecompressor()
    for shard in sorted(glob.glob(os.path.join(precompiled, 'shard_*.jsonl.zst'))):
        with open(shard, 'rb') as fh:
            for line in io.TextIOWrapper(dctx.stream_reader(fh), encoding='utf-8'):
                line = line.strip()
                if not line: continue
                r = json.loads(line); i, j = match_columns(r['x'], r['a'], r['y'])
                if len(i) == 0: continue
                b = min(max(int(np.searchsorted(t_edges, float(r['t']), side='right') - 1), 0), n_bins - 1)
                keys.append(b * AA * AA + i * AA + j); n += 1
                if n % 200000 == 0: print(f'  {n} pairs, {time.time()-t0:.0f}s', flush=True)
                if max_pairs and n >= max_pairs: break
        if max_pairs and n >= max_pairs: break
    N = np.bincount(np.concatenate(keys), minlength=n_bins * AA * AA).reshape(n_bins, AA, AA).astype(np.float64)
    print(f'counts from {n} pairs: {N.sum():.0f} matched columns, {time.time()-t0:.0f}s', flush=True)
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--precompiled', default='pfam/precompiled_v3_train')
    ap.add_argument('--n-classes', type=int, default=20)
    ap.add_argument('--n-bins', type=int, default=20)
    ap.add_argument('--t-min', type=float, default=0.01); ap.add_argument('--t-max', type=float, default=5.0)
    ap.add_argument('--n-outer', type=int, default=60); ap.add_argument('--n-inner', type=int, default=60)
    ap.add_argument('--lr', type=float, default=0.05); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-pairs', type=int, default=0, help='>0: use only the first N pairs (smoke test)')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    C = args.n_classes
    t_edges = np.geomspace(args.t_min, args.t_max, args.n_bins + 1); t_centers = np.sqrt(t_edges[:-1] * t_edges[1:])
    N = build_counts(args.precompiled, t_edges, args.max_pairs)
    N_j = jnp.asarray(N); t_j = jnp.asarray(t_centers)

    # --- parameters: log S (upper triangle) and pi logits per class; LG08 init with perturbation
    Q_lg, pi_lg = rate_matrix_lg(); Q_lg = np.asarray(Q_lg); pi_lg = np.asarray(pi_lg)
    S_lg = Q_lg / pi_lg[None, :]; np.fill_diagonal(S_lg, 0.0)
    rows, cols = np.triu_indices(AA, k=1); rows_j = jnp.asarray(rows); cols_j = jnp.asarray(cols)
    rng = np.random.default_rng(args.seed)
    logS = np.tile(np.log(np.maximum(S_lg[rows, cols], 1e-6)), (C, 1)) + 0.3 * rng.standard_normal((C, len(rows)))
    logpi = np.tile(np.log(pi_lg), (C, 1)) + 0.2 * rng.standard_normal((C, AA))
    logw = np.log(np.full(C, 1.0 / C))

    def class_joint(logS_c, logpi_c):
        """log [pi_c(i) P_c(i, j | t_b)] for every bin: (n_bins, A, A)."""
        S = jnp.zeros((AA, AA)).at[rows_j, cols_j].set(jnp.exp(logS_c)); S = S + S.T
        pi = jax.nn.softmax(logpi_c)
        Q = S * pi[None, :]; Q = Q.at[jnp.diag_indices(AA)].set(0.0); Q = Q.at[jnp.diag_indices(AA)].set(-Q.sum(1))
        P = jax.vmap(lambda t: jax.scipy.linalg.expm(Q * t))(t_j)                     # (B, A, A)
        return jnp.log(pi)[None, :, None] + jnp.log(jnp.maximum(P, 1e-300))

    all_joint = jax.jit(jax.vmap(class_joint))                                          # (C, B, A, A)

    @jax.jit
    def e_step(logS, logpi, logw):
        lj = all_joint(logS, logpi) + logw[:, None, None, None]
        lse = jax.scipy.special.logsumexp(lj, axis=0)                                    # (B, A, A)
        resp = jnp.exp(lj - lse[None]); total_ll = jnp.sum(N_j * lse)
        return resp, total_ll

    def class_loss(logS_c, logpi_c, Nc):
        return -jnp.sum(Nc * class_joint(logS_c, logpi_c))

    grad_fn = jax.jit(jax.vmap(jax.value_and_grad(class_loss, argnums=(0, 1))))       # over classes

    def adam_m_step(logS, logpi, Nc, n_steps, lr):
        m = [np.zeros_like(logS), np.zeros_like(logpi)]; v = [np.zeros_like(logS), np.zeros_like(logpi)]
        params = [logS, logpi]
        for k in range(1, n_steps + 1):
            loss, grads = grad_fn(jnp.asarray(params[0]), jnp.asarray(params[1]), Nc)
            for q in range(2):
                g = np.asarray(grads[q]); m[q] = 0.9 * m[q] + 0.1 * g; v[q] = 0.999 * v[q] + 0.001 * g * g
                mh = m[q] / (1 - 0.9 ** k); vh = v[q] / (1 - 0.999 ** k)
                params[q] = params[q] - lr * mh / (np.sqrt(vh) + 1e-8)
        return params[0], params[1], float(np.sum(np.asarray(loss)))

    history = []; t0 = time.time()
    for it in range(1, args.n_outer + 1):
        resp, total_ll = e_step(jnp.asarray(logS), jnp.asarray(logpi), jnp.asarray(logw))
        Nc = resp * N_j[None]                                                              # (C, B, A, A)
        mass = np.asarray(Nc.sum(axis=(1, 2, 3))); logw = np.log(np.maximum(mass / mass.sum(), 1e-12))
        logS, logpi, loss = adam_m_step(logS, logpi, Nc, args.n_inner, args.lr)
        history.append(dict(iter=it, ll=float(total_ll), weights=np.exp(logw).tolist()))
        print(f'  EM iter {it}: total log-lik = {float(total_ll):.2f}  (per column {float(total_ll)/N.sum():.5f})  '
              f'w max {np.exp(logw).max():.3f}  {time.time()-t0:.0f}s', flush=True)
    resp, total_ll = e_step(jnp.asarray(logS), jnp.asarray(logpi), jnp.asarray(logw))
    S_out = np.zeros((C, AA, AA)); pi_out = np.zeros((C, AA))
    for c in range(C):
        S = np.zeros((AA, AA)); S[rows, cols] = np.exp(logS[c]); S_out[c] = S + S.T
        pi_out[c] = np.exp(logpi[c] - logpi[c].max()); pi_out[c] /= pi_out[c].sum()
    np.savez(args.output, S=S_out, pi=pi_out, weights=np.exp(logw), t_centers=t_centers, n_states=20,
             n_classes=C, n_pairs_source=args.precompiled, history=np.array(history, dtype=object))
    print(f'final total log-lik {float(total_ll):.2f}; saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
