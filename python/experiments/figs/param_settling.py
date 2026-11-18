"""How early do a MixDom fit's domain types settle?  For ONE run (the released fit's own seed, rerun with the
compositions logged), the earth-mover distance between each domain type's equilibrium amino-acid composition at
iteration k and its composition at the end of that run (its own checkpoint), one curve per domain type.  Ground distance between amino acids: -log of the symmetrised one-substitution exchange probability
under the refitted GTR (pfam/gtr_refit_Qpi.npz) at one expected substitution per site, scaled to mean 1 off the
diagonal; the same curves with a 0/1 ground distance (= total variation) are drawn dotted.  The basis for stopping
the reproducibility restarts at 540 iterations.
Usage: python param_settling.py [d2f1 d3f1 ...] -> experiments/figs/param_settling.pdf"""
import os, re, sys, numpy as np
from scipy.optimize import linprog
from scipy.linalg import expm
HERE = os.path.dirname(os.path.abspath(__file__)); PY = os.path.dirname(os.path.dirname(HERE))
BLOCK = re.compile(r'--- SVI-BW iteration (\d+)/\d+ ---(.*?)(?=--- SVI-BW iteration|\Z)', re.S)
PI = re.compile(r'dom (\d+): pi=\[([^\]]*)\]')


def ground_distance():
    g = np.load(f'{PY}/pfam/gtr_refit_Qpi.npz'); Q = np.asarray(g['Q'], float); pi = np.asarray(g['pi'], float); pi = pi / pi.sum()
    Q = Q / (-(pi * np.diag(Q)).sum())                      # one expected substitution per site at t = 1
    P = expm(Q); D = -np.log(np.maximum(0.5 * (P + P.T), 1e-300)); np.fill_diagonal(D, 0.0)
    off = D[~np.eye(20, dtype=bool)]; return D / off.mean()


def emd(p, q, D):
    n = len(p); A = np.zeros((2 * n, n * n)); b = np.concatenate([p, q])
    for i in range(n): A[i, i * n:(i + 1) * n] = 1
    for j in range(n): A[n + j, j::n] = 1
    r = linprog(D.ravel(), A_eq=A, b_eq=b, bounds=(0, None), method='highs'); return float(r.fun)


def parse_pis(path):
    """{iteration: (D, 20) compositions} from a training log with per-iteration 'dom d: pi=[...]' lines."""
    out = {}
    for m in BLOCK.finditer(open(path, errors='replace').read()):
        rows = PI.findall(m.group(2))
        if rows:   # a block may report the parameters more than once (before and after the M-step): keep the last line per domain
            last = {int(d): np.array(v.split(), float) for d, v in rows}
            out[int(m.group(1))] = np.array([last[d] for d in sorted(last)])
    return out


def final_pis(npz):
    d = np.load(npz, allow_pickle=True); D = d['dom_ins'].shape[0]
    if 'class_pis' in d.files and 'classdist' in d.files:
        cd = np.asarray(d['classdist']); fw = np.asarray(d['frag_weights']); cp = np.asarray(d['class_pis'])
        return np.array([(fw[k] / fw[k].sum()) @ cd[k] @ cp for k in range(D)])
    return np.asarray(d['dom_pis'])


def curves(model, niter=540, seed=42):
    traj = parse_pis(f'{PY}/pfam/svibw_{model}_nobali_{niter}_seed{seed}.log')
    ref = final_pis(f'{PY}/pfam/svibw_{model}_nobali_{niter}_seed{seed}.npz'); G = ground_distance(); TV = 1.0 - np.eye(20)
    its = np.array([k for k in sorted(traj) if k < niter]); D = ref.shape[0]   # the last iteration is the reference itself
    e = np.array([[emd(traj[k][dd] / traj[k][dd].sum(), ref[dd] / ref[dd].sum(), G) for dd in range(D)] for k in its])
    tv = np.array([[0.5 * np.abs(traj[k][dd] / traj[k][dd].sum() - ref[dd] / ref[dd].sum()).sum() for dd in range(D)] for k in its])
    return its, e, tv, ref


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    models = sys.argv[1:] or ['d2f1']
    fig, axs = plt.subplots(1, len(models), figsize=(3.6 * len(models), 2.8), squeeze=False)
    for ax, model in zip(axs[0], models):
        its, e, tv, ref = curves(model)
        for dd in range(e.shape[1]):
            ln, = ax.plot(its, e[:, dd], lw=1.2, label=f'domain type {dd + 1}'); ax.plot(its, tv[:, dd], lw=0.8, ls=':', color=ln.get_color())
        ax.set_yscale('log'); ax.set_xlabel('svi-BW iteration'); ax.set_ylabel('EMD to final composition')
        ax.set_title(f'MixDom-{model}, {len(its) + 1} iterations'); ax.legend(frameon=False, fontsize=7)
        for k in (20, 50, 100, 200, 400, 520):
            if (its <= k).any(): print(f'{model} it {k}: EMD ' + ' '.join(f'{v:.3f}' for v in e[its <= k][-1]) + ' | TV ' + ' '.join(f'{v:.3f}' for v in tv[its <= k][-1]))
    fig.tight_layout(); fig.savefig(f'{HERE}/param_settling.pdf'); fig.savefig(f'{HERE}/param_settling.png', dpi=130); print('wrote', f'{HERE}/param_settling.pdf')
