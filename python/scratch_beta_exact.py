"""Exact RG beta-function beta(t,r) = dr'/du|_{u=0} (2-site survival fit)."""
import sympy as sp
kap, mu, t, u, r = sp.symbols('kappa mu t u r', positive=True)
al = sp.exp(-mu*t); b = sp.exp(-mu*u); s = al*b
p = r/(r + (1-r)*kap)
al1 = al**(1-kap); bet = kap*(1-al1)/(1-kap*al1); nu = 1-bet
f = lambda q: q + (1-q)*b
M2 = p*al*b*f(p) + (1-p)*al**2*b*f(nu*p)
p2 = (M2/s - s)/(1-s)
r2 = kap*p2/(1 - p2*(1-kap))
beta = sp.simplify(sp.diff(r2, u).subs(u, 0))
print("beta(t,r) ="); sp.pprint(beta)
print("\nt->0 :", sp.simplify(sp.limit(beta, t, 0)),
      "   [-(lam+mu)r(1-r) =", sp.expand(-(kap*mu+mu)*r*(1-r)), "]")
print("t->oo:", sp.simplify(sp.limit(beta, t, sp.oo)),
      "   [-mu r(1-r) =", sp.expand(-mu*r*(1-r)), "]")
# factor out -mu r(1-r):
g = sp.simplify(beta/(-mu*r*(1-r)))
print("\nbeta / (-mu r(1-r)) ="); sp.pprint(sp.simplify(g))
