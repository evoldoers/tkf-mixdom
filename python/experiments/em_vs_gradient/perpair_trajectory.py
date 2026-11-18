#!/usr/bin/env python3
"""Per-pair log-likelihoods at every recorded pass of a compare_optimizers.py run, for error bars on the
convergence figure (the runs record only the mean LL per pass; their parameters per pass are recorded, so the
per-pair values can be recomputed on the same corpus).

Usage: perpair_trajectory.py <run.json> [--out <npz>] [--n-pairs N] [--chunk C]
  Rebuilds the run's corpus from its recorded args (corpus, n_pairs, max_len, seed, subst_npz), evaluates the
  per-pair forward LL at each pass's parameters for every method, and writes
    <out>.npz: for each method m, ll_<m> (n_pass, N) and pass_<m> (n_pass,); plus 'order' (the pair order used)
  --n-pairs / --chunk override the recorded values (smoke tests only; the corpus then differs from the run's).
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_optimizers as C
import jax, jax.numpy as jnp


def make_perpair(model, Q, pi):
    Qj, pij = jnp.asarray(Q), jnp.asarray(pi)

    def per_pair(v, x, y, t, Lx, Ly):
        lam, mu, exts, weights = model.unpack(v)
        if model.name == "mixfrag_s":
            _Ss, pis, Qs = model.unpack_subst(v)
            return C._estep_pair_mixfrag_s_forward_only_core(x, y, t, Lx, Ly, lam, mu, exts, weights, Qs, pis)
        if model.name == "mixfrag":
            return C._estep_pair_mixfrag_forward_only_core(x, y, t, Lx, Ly, lam, mu, exts, weights, Qj, pij)
        return C._estep_pair_tkf92_forward_only_core(x, y, t, Lx, Ly, lam, mu, exts[0], Qj, pij)

    return jax.jit(jax.vmap(per_pair, in_axes=(None, 0, 0, 0, 0, 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-pairs", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--max-passes", type=int, default=None, help="smoke tests: only the first k passes per method")
    a = ap.parse_args()
    d = json.load(open(a.run))
    ra = argparse.Namespace(**d["args"])
    if a.n_pairs: ra.n_pairs = a.n_pairs
    if a.chunk: ra.chunk = a.chunk
    out = a.out or os.path.splitext(a.run)[0] + "_perpair.npz"
    Q, pi = C.load_subst(ra.subst_npz)
    model = C.Model(ra.model, n_fragtypes=getattr(ra, "n_fragtypes", 2), A=pi.shape[0])
    if model.name == "mixfrag_s":
        sys.exit("mixfrag_s runs do not record their per-pass matrices; not supported")
    t0 = time.time()
    pairs = C.load_corpus(ra, Q, pi)
    chunks, n_buckets = C.make_chunks(pairs, ra.chunk)
    N = len(pairs)
    print(f"corpus {ra.corpus}: N={N}, {len(chunks)} chunks of {ra.chunk}; loaded in {time.time()-t0:.1f}s", flush=True)
    f = make_perpair(model, Q, pi)

    def evaluate(v):
        outl = []
        for xs, ys, ts, Lx, Ly, w in chunks:
            lp = np.asarray(f(v, xs, ys, ts, Lx, Ly))
            outl.append(lp[np.asarray(w) > 0])
        return np.concatenate(outl)

    res = {}
    for m, traj in d["methods"].items():
        if a.max_passes: traj = traj[:a.max_passes]
        rows, passes, t1 = [], [], time.time()
        for k, rec in enumerate(traj):
            v = model.pack(rec["params"])
            ll = evaluate(v)
            rows.append(ll); passes.append(rec["pass"])
            if k == 0:
                print(f"  {m}: first pass {time.time()-t1:.1f}s (compile); recorded mean {rec['ll']:.6f}, "
                      f"recomputed {ll.mean():.6f}", flush=True)
        rows = np.stack(rows)
        rec_means = np.array([r["ll"] for r in traj])
        print(f"  {m}: {len(traj)} passes in {time.time()-t1:.0f}s; max |recomputed mean - recorded| = "
              f"{np.max(np.abs(rows.mean(1) - rec_means)):.2e}", flush=True)
        res[f"ll_{m}"] = rows; res[f"pass_{m}"] = np.array(passes)
    np.savez_compressed(out, N=N, methods=np.array(list(d["methods"]), dtype=object), **res)
    print(f"saved {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
