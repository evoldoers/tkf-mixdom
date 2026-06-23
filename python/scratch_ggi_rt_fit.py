"""Try simple closed-form fits to r(t) from the matched flow."""
import sys, time
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/Users/yam/tkf-mixdom/python')

from scratch_ggi_triad_eliminated import run_flow, boundary_condition

# Candidate functional forms:
def f_stretched_exp(t, r0, k, alpha):
    """r(t) = r0 * exp(-k * t^alpha)"""
    return r0 * np.exp(-k * np.power(t, alpha))

def f_power_law(t, r0, c, p):
    """r(t) = r0 / (1 + c*t)^p"""
    return r0 / np.power(1 + c * t, p)

def f_exp_sqrt(t, r0, k):
    """r(t) = r0 * exp(-k * sqrt(t))  [restricted alpha=1/2]"""
    return r0 * np.exp(-k * np.sqrt(t))

def f_invquad(t, r0, k):
    """r(t) = r0 / (1 + k * r0 * t)  [from dr/dt = -k * r^2]"""
    return r0 / (1 + k * r0 * t)


def fit_and_report(t, r, label):
    print(f"\n  -- {label} --")
    r0 = r[0]
    # exp_sqrt (1 fit param)
    try:
        (k,), _ = curve_fit(lambda t, k: f_exp_sqrt(t, r0, k), t, r, p0=[1.0],
                            bounds=([0], [100]))
        pred = f_exp_sqrt(t, r0, k)
        rmse = np.sqrt(np.mean((pred - r)**2))
        max_rel = np.max(np.abs((pred - r) / r))
        print(f"    r(t) = r0*exp(-k*sqrt(t)):     k = {k:.4f}  | RMSE = {rmse:.4f}  max-rel = {max_rel*100:.1f}%")
    except Exception as e:
        print(f"    exp_sqrt fit failed: {e}")
    # stretched exp (2 fit params)
    try:
        (k, alpha), _ = curve_fit(lambda t, k, alpha: f_stretched_exp(t, r0, k, alpha), t, r,
                                    p0=[1.0, 0.5], bounds=([0, 0.1], [100, 2]))
        pred = f_stretched_exp(t, r0, k, alpha)
        rmse = np.sqrt(np.mean((pred - r)**2))
        max_rel = np.max(np.abs((pred - r) / r))
        print(f"    r(t) = r0*exp(-k*t^alpha):    k = {k:.4f}, alpha = {alpha:.3f}  | RMSE = {rmse:.4f}  max-rel = {max_rel*100:.1f}%")
    except Exception as e:
        print(f"    stretched_exp fit failed: {e}")
    # inverse quadratic
    try:
        (k,), _ = curve_fit(lambda t, k: f_invquad(t, r0, k), t, r, p0=[1.0],
                            bounds=([0], [100]))
        pred = f_invquad(t, r0, k)
        rmse = np.sqrt(np.mean((pred - r)**2))
        max_rel = np.max(np.abs((pred - r) / r))
        print(f"    r(t) = r0/(1+k*r0*t)  [dr=-k*r^2]: k = {k:.4f}  | RMSE = {rmse:.4f}  max-rel = {max_rel*100:.1f}%")
    except Exception as e:
        print(f"    invquad fit failed: {e}")
    # power-law
    try:
        (c, p), _ = curve_fit(lambda t, c, p: f_power_law(t, r0, c, p), t, r,
                                p0=[1.0, 1.0], bounds=([0, 0.1], [100, 5]))
        pred = f_power_law(t, r0, c, p)
        rmse = np.sqrt(np.mean((pred - r)**2))
        max_rel = np.max(np.abs((pred - r) / r))
        print(f"    r(t) = r0/(1+c*t)^p:          c = {c:.4f}, p = {p:.3f}  | RMSE = {rmse:.4f}  max-rel = {max_rel*100:.1f}%")
    except Exception as e:
        print(f"    power_law fit failed: {e}")


# Run flows at multiple parameter combos
cases = [
    ("(0.4, 0.55)", 1.0 * 0.4 * (1 - 0.55) / (0.55 * (1 - 0.4)), 1.0, 0.4, 0.55, 5e-3, 5.0),
    ("Pfam (x=0.65, y=0.71)", 0.0145, 0.0148, 0.65, 0.7098, 0.01, 10.0),
    ("Pfam (x=0.5, y=0.77)",  0.0145, 0.0148, 0.5,  0.7666, 0.01, 10.0),
    ("(0.2, 0.4)",            1.0 * 0.2 * 0.6 / (0.4 * 0.8), 1.0, 0.2, 0.4, 5e-3, 5.0),
    ("(0.3, 0.6)",            1.0 * 0.3 * 0.4 / (0.6 * 0.7), 1.0, 0.3, 0.6, 5e-3, 5.0),
]

print("="*72)
print("Fitting simple forms to r(t) from the matched flow")
print("="*72)
results = []
for label, lam0, mu0, x, y, t_eps, t_max in cases:
    bc_l, bc_m, bc_r = boundary_condition(lam0, mu0, x, y)
    t_eval = np.geomspace(t_eps, t_max, 40)
    sol, _ = run_flow(lam0, mu0, x, y, t_eps=t_eps, t_max=t_max, t_eval=t_eval)
    if sol.status != 0:
        print(f"\n{label}: flow failed")
        continue
    t = sol.t
    r = sol.y[2]
    print(f"\n{label}: lam0={lam0:.4f}, mu0={mu0:.4f}, r0_boundary={bc_r:.4f}")
    fit_and_report(t, r, label)
    results.append((label, t, r, bc_r))

# Plot best-form fits
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for i, (label, t, r, r0) in enumerate(results[:6]):
    ax = axes[i//3, i%3]
    ax.semilogx(t, r, 'ko', ms=4, label='full flow', zorder=10)
    # Fit stretched exp
    try:
        (k, alpha), _ = curve_fit(lambda t, k, alpha: f_stretched_exp(t, r0, k, alpha), t, r,
                                    p0=[1.0, 0.5], bounds=([0, 0.1], [100, 2]))
        t_fine = np.geomspace(t[0], t[-1], 200)
        ax.semilogx(t_fine, f_stretched_exp(t_fine, r0, k, alpha), 'b-', lw=2,
                    label=fr'$r_0\, e^{{-k\,t^\alpha}}$, $k={k:.3f}, \alpha={alpha:.2f}$')
    except Exception:
        pass
    # exp_sqrt restricted
    try:
        (k,), _ = curve_fit(lambda t, k: f_exp_sqrt(t, r0, k), t, r, p0=[1.0],
                            bounds=([0], [100]))
        t_fine = np.geomspace(t[0], t[-1], 200)
        ax.semilogx(t_fine, f_exp_sqrt(t_fine, r0, k), 'r--', lw=1.5,
                    label=fr'$r_0\, e^{{-k\sqrt{{t}}}}$, $k={k:.3f}$')
    except Exception:
        pass
    ax.set_xlabel('t'); ax.set_ylabel('r')
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=0.3)
for j in range(len(results), 6):
    axes[j//3, j%3].axis('off')
plt.tight_layout()
plt.savefig('/tmp/rt_fits.png', dpi=110, bbox_inches='tight')
print("\nSaved /tmp/rt_fits.png")
