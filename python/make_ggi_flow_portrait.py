"""Phase portrait of the GGI->TKF92 instantaneous KL-projection flow in the
(mu, r) plane.  For several GGI shapes we trace theta_direct(t)=argmin D(GGI(t)||TKF92)
via Gillespie+M-step (the validated direct fit), showing:
  * the flow runs almost horizontally (mu ~ fixed) downward in r,
  * from the generator-match r0 (t->0) to the GGI attractor r_inf (t->inf),
  * the static moment-match sitting far to the NE (high mu, r=x),
  * TKF91 (r=0) as the *self-composition* flow's attractor (the other flow).
Output: python/experiments/figures/ggi_flow_portrait.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scratch_ggi_flow as G
import scratch_ggi_flow_traj as TR

M = 1
rng = G.rng


def direct_fit(x, y, mu0, t, nrep=40000):
    lam0 = mu0 * x * (1 - y) / (y * (1 - x))
    Nsum = np.zeros((5, 5))
    for _ in range(nrep):
        n = TR.ggi_equilibrium_len(x, y)
        Nsum += G.count_transitions(G.gillespie_leg2([M] * n, lam0, mu0, x, y, t))
    return np.array(G.kar_to_lmr(*G.kl_fit(Nsum / nrep), t))


shapes = [
    # (x, y, colour, label)
    (0.3, 0.50, "#1b9e77", r"$\ell_{\rm GGI}=1.5$"),
    (0.4, 0.55, "#d95f02", r"$\ell_{\rm GGI}=2.7$"),
    (0.6, 0.70, "#7570b3", r"$\ell_{\rm GGI}=6$"),
]
mu0 = 1.0
tgrid = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 4.0])
NREP = 40000

fig, ax = plt.subplots(figsize=(6.4, 4.6))
for (x, y, col, lab) in shapes:
    traj = np.array([direct_fit(x, y, mu0, t) for t in tgrid])  # columns lam,mu,r
    mu, r = traj[:, 1], traj[:, 2]
    ax.plot(mu, r, "-", color=col, lw=1.6, zorder=3)
    # direction arrows along the trajectory
    for i in (1, 4):
        ax.annotate("", xy=(mu[i + 1], r[i + 1]), xytext=(mu[i], r[i]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6), zorder=4)
    ax.plot(mu[0], r[0], "o", mfc="white", mec=col, mew=1.6, ms=8, zorder=5)   # t->0 generator match
    ax.plot(mu[-1], r[-1], "o", color=col, ms=8, zorder=5)                     # attractor
    # moment match (static)
    lam_mm, mu_mm, r_mm = TR.moment_match(mu0 * x * (1 - y) / (y * (1 - x)), mu0, x, y)
    ax.plot(mu_mm, r_mm, "*", color=col, ms=15, mec="k", mew=0.5, zorder=6)
    ax.annotate(lab, (mu[-1], r[-1]), textcoords="offset points",
                xytext=(6, -2), fontsize=9, color=col)

# TKF91 axis = self-composition attractor
ax.axhline(0, color="0.4", ls="--", lw=1)
ax.annotate("TKF91 line ($r=0$): attractor of the\nself-composition flow",
            (1.0, 0.0), textcoords="offset points", xytext=(0, 6),
            fontsize=8.5, color="0.35")

# legend proxies
from matplotlib.lines import Line2D
handles = [
    Line2D([], [], marker="o", mfc="white", mec="k", ls="", ms=8, label=r"generator match ($t\!\to\!0$)"),
    Line2D([], [], marker="o", color="k", ls="", ms=8, label=r"GGI attractor ($t\!\to\!\infty$)"),
    Line2D([], [], marker="*", color="0.5", mec="k", ls="", ms=14, label="static moment match"),
    Line2D([], [], marker=r"$\rightarrow$", color="k", ls="", ms=12, label="flow direction (increasing $t$)"),
]
ax.legend(handles=handles, fontsize=8.5, loc="upper center", frameon=True, ncol=2)

ax.set_xlabel(r"deletion rate $\mu$  (per-fragment)")
ax.set_ylabel(r"fragment-extension probability $r$")
ax.set_title(r"GGI$\to$TKF92 flow in the $(\mu,r)$ plane")
ax.set_xlim(0.7, 3.6)
ax.set_ylim(-0.03, 0.62)
fig.tight_layout()
out = "experiments/figures/ggi_flow_portrait.pdf"
fig.savefig(out)
print("wrote", out)
