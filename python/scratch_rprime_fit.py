"""
Algebraic fit of TKF92 to the boundary-forgetting self-composition
TKF92(t) o TKF92(u), in the gauge t' = t+u.

Rates are pinned exactly: kappa'=kappa (equilibrium), mu'=mu (per-residue
survival composes as alpha_t*alpha_u = alpha_{t+u}), hence lambda'=lambda.
The only running coordinate is r'. We fix it by matching the adjacent-
residue survival probability M = P(two adjacent ancestral residues both
survive), a 2-site fragment-structure moment.
"""
import sympy as sp

kap, mu, t, u, r, eps = sp.symbols('kappa mu t u r epsilon', positive=True)

alpha = sp.exp(-mu*t)        # branch-1 per-fragment survival
b     = sp.exp(-mu*u)        # branch-2 per-fragment survival
s     = alpha*b              # = alpha_{t+u}, single-residue survival over both
p     = r/(r + (1-r)*kap)    # equilibrium intra-fragment bond prob

# TKF91 insertion prob on branch 1, in (kappa, alpha):
al1 = alpha**(1-kap)
bet = kap*(1 - al1)/(1 - kap*al1)         # beta_t
nu  = 1 - bet                              # P(no insertion splits the bond on branch 1)

# --- composite adjacent-survival moment M_comp ---
A_tot = p*alpha + (1-p)*alpha**2           # P(both survive branch 1)
A     = p*alpha + (1-p)*alpha**2*nu        # ... and still adjacent in Y
M_comp = A_tot*b**2 + A*p*b*(1-b)

# --- TKF92(mu, r', t+u) moment:  M = p'*s + (1-p')*s^2 ---
# invert: p' = (M_comp - s^2) / (s (1-s))
p_eff = sp.simplify((M_comp - s**2)/(s*(1-s)))
print("p'(t,u) =")
sp.pprint(p_eff)

# r' from p' = r'/(r' + (1-r') kappa)  =>  r' = kappa p'/(1 - p'(1-kappa))
r_eff = sp.simplify(kap*p_eff/(1 - p_eff*(1-kap)))

print("\n--- limit checks ---")
print("r=0   -> p' =", sp.simplify(p_eff.subs(r, 0)))
print("u=0   -> p' =", sp.simplify(p_eff.subs(u, 0)), " (expect p)")
print("t=0   -> p' =", sp.simplify(p_eff.subs(t, 0)), " (expect p)")
print("p (for reference) =", sp.simplify(p))

print("\n--- leading order in time (t->eps t, u->eps u) ---")
pe = p_eff.subs({t: eps*t, u: eps*u})
lead = sp.series(pe, eps, 0, 2).removeO()
lead = sp.simplify(lead)
print("p'  ~ ", lead)
# predicted:  p' - p ~ -p(1-p)(lambda+mu) * t u/(t+u),  lambda=kappa*mu
pred = p - p*(1-p)*(kap*mu+mu)*(eps*t*eps*u)/(eps*t+eps*u)
pred = sp.simplify(sp.series(pred, eps, 0, 2).removeO())
print("pred~ ", pred)
print("difference (should be 0):", sp.simplify(lead - pred))

print("\n--- r' leading order ---")
re_lead = sp.series(r_eff.subs({t: eps*t, u: eps*u}), eps, 0, 2).removeO()
re_lead = sp.simplify(re_lead)
print("r' ~ ", re_lead)
pred_r = r - (kap*mu+mu)*r*(1-r)*(eps*t*eps*u)/(eps*t+eps*u)
pred_r = sp.simplify(sp.series(pred_r, eps, 0, 2).removeO())
print("pred r' ~ ", pred_r)
print("difference (should be 0):", sp.simplify(re_lead - pred_r))
