"""
Test of moment-independence of the leading-order r' fit.

If the boundary-forgetting composite is on the TKF92 manifold to leading
order (just at shifted r'), then its survival statistics satisfy the TKF92
identity  M3 * s == M2**2  (since M2 = s[p'+(1-p')s], M3 = s[p'+(1-p')s]**2),
and the 3-site-fitted r' equals the 2-site one. We test both.

Framework (unified): k adjacent ancestral residues; equilibrium fragment
bonds i.i.d. intra w.p. p. Branch 1 survival per fragment alpha, insertions
split inter-bonds w.p. beta (so adjacent-in-Y w.p. nu=1-beta). Y boundaries
re-imputed i.i.d. intra w.p. p among adjacent survivors. Branch 2 survival
per Y-fragment b. P(all survive br2 | #Y-frags=f) = b**f.
"""
import sympy as sp

kap, mu, t, u, r, eps = sp.symbols('kappa mu t u r epsilon', positive=True)
alpha = sp.exp(-mu*t); b = sp.exp(-mu*u); s = alpha*b
p = r/(r + (1-r)*kap)
al1 = alpha**(1-kap); bet = kap*(1-al1)/(1-kap*al1); nu = 1-bet

# per Y-bond factor E[b^{I}] for a bond intra w.p. q: = q + (1-q)b
def f(q): return q + (1-q)*b

# 2-site survival moment (both adjacent residues survive to Z)
M2 = ( p   * alpha    * b * f(p)
     + (1-p)* alpha**2 * b * f(nu*p) )

# 3-site survival moment (all three survive to Z); X-partitions of 2 bonds
M3 = ( p**2     * alpha    * b * f(p)   * f(p)      # 111  (both intra)
     + p*(1-p)  * alpha**2 * b * f(p)   * f(nu*p)   # 11|3
     + (1-p)*p  * alpha**2 * b * f(nu*p)* f(p)      # 1|23
     + (1-p)**2 * alpha**3 * b * f(nu*p)* f(nu*p) ) # 1|2|3

print("limit checks (M2):")
print("  r=0:", sp.simplify(M2.subs(r,0)), " (expect s^2 =", sp.simplify(s**2),")")
print("  u=0:", sp.simplify(M2.subs(u,0)))
print("  t=0:", sp.simplify(M2.subs(t,0)))
print("limit checks (M3):")
print("  r=0:", sp.simplify(M3.subs(r,0)), " (expect s^3)")

# ---- on-manifold test: D = M3*s - M2^2  (==0 iff TKF92-like in survival) ----
D = M3*s - M2**2
print("\nD = M3*s - M2^2, leading order in time:")
Dlead = sp.series(D.subs({t:eps*t, u:eps*u}), eps, 0, 4).removeO()
Dlead = sp.simplify(Dlead)
print("  D ~", Dlead)

# ---- fit p' from each moment, compare leading orders ----
# M2 = s[p2'+(1-p2')s] -> p2' = (M2/s - s)/(1-s)
p2 = sp.simplify((M2/s - s)/(1-s))
# M3 = s[p3'+(1-p3')s]^2 -> p3' = (sqrt(M3/s)-s)/(1-s)
p3 = sp.simplify((sp.sqrt(M3/s) - s)/(1-s))

def lead(expr):
    return sp.simplify(sp.series(expr.subs({t:eps*t,u:eps*u}), eps,0,2).removeO())

p2L = lead(p2); p3L = lead(p3)
print("\np2' leading:", p2L)
print("p3' leading:", p3L)
print("p3' - p2' (leading):", sp.simplify(p3L - p2L))

# predicted moment-independent shift: p' - p = -p(1-p)(lambda+mu) t u/(t+u)
pred = sp.simplify(lead(p) - p*(1-p)*(kap*mu+mu)*(eps*t*eps*u)/(eps*t+eps*u))
print("predicted p' leading:", sp.simplify(pred))
print("p2' - pred:", sp.simplify(p2L - pred))

# ================= beta-function / RG ODE: u -> 0, t finite =================
print("\n================ beta-function (u->0, finite t) ================")
r2 = sp.simplify(kap*p2/(1 - p2*(1-kap)))           # effective r' from 2-site fit
beta = sp.simplify(sp.diff(r2, u).subs(u, 0))        # dr'/du at u=0  = dr/dt flow
print("beta(t,r) = dr'/du|_{u=0} =")
sp.pprint(beta)
print("\nsmall-t limit of beta:")
beta_small = sp.simplify(sp.series(beta.subs(t, eps*t), eps, 0, 1).removeO())
print("  beta ~", beta_small, "  (expect -(lambda+mu) r(1-r) = ",
      sp.expand(-(kap*mu+mu)*r*(1-r)), ")")
print("\nis beta t-independent?  beta - beta(t->0 form):",
      sp.simplify(beta - (-(kap*mu+mu)*r*(1-r))))
