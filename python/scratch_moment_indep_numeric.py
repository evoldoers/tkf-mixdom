"""
Numerical test (fast) of:
  (1) moment-independence of the r' fit: does the 3-site survival moment
      give the same r' as the 2-site one?  (exactly, and at leading order)
  (2) the RG beta-function beta(t,r)=dr'/du|_{u=0}: is it t-independent?
"""
import math, random

def pieces(kap, mu, r, t, u):
    al = math.exp(-mu*t); b = math.exp(-mu*u); s = al*b
    p = r/(r + (1-r)*kap)
    al1 = al**(1-kap); bet = kap*(1-al1)/(1-kap*al1); nu = 1-bet
    f = lambda q: q + (1-q)*b
    M2 = p*al*b*f(p) + (1-p)*al**2*b*f(nu*p)
    M3 = (p**2*al*b*f(p)**2 + 2*p*(1-p)*al**2*b*f(p)*f(nu*p)
          + (1-p)**2*al**3*b*f(nu*p)**2)
    return s, p, M2, M3

def p_from_M2(s, M2): return (M2/s - s)/(1-s)
def p_from_M3(s, M3): return (math.sqrt(M3/s) - s)/(1-s)
def r_from_p(kap, pp): return kap*pp/(1 - pp*(1-kap))

print("=== (1) moment-independence: r' from 2-site vs 3-site survival ===")
print(f"{'kap':>5}{'mu':>5}{'r':>5}{'t':>6}{'u':>6}  {'r2prime':>11}{'r3prime':>11}{'r3-r2':>12}{'M3*s-M2^2':>13}")
random.seed(1)
for _ in range(8):
    kap=round(random.uniform(0.2,0.8),2); mu=round(random.uniform(0.3,1.5),2)
    r=round(random.uniform(0.2,0.8),2); t=round(random.uniform(0.2,1.2),2)
    u=round(random.uniform(0.2,1.2),2)
    s,p,M2,M3 = pieces(kap,mu,r,t,u)
    r2=r_from_p(kap,p_from_M2(s,M2)); r3=r_from_p(kap,p_from_M3(s,M3))
    print(f"{kap:5.2f}{mu:5.2f}{r:5.2f}{t:6.2f}{u:6.2f}  {r2:11.6f}{r3:11.6f}{r3-r2:12.2e}{M3*s-M2**2:13.2e}")

print("\n=== leading order: does (r3'-r2') vanish faster than (r2'-r)? ===")
print("scale t,u by eps; report (r2'-r)/eps and (r3'-r2')/eps as eps->0")
kap,mu,r,t0,u0 = 0.5,1.0,0.5,1.0,0.8
for eps in [0.1,0.03,0.01,0.003,0.001]:
    s,p,M2,M3 = pieces(kap,mu,r,eps*t0,eps*u0)
    r2=r_from_p(kap,p_from_M2(s,M2)); r3=r_from_p(kap,p_from_M3(s,M3))
    print(f"  eps={eps:7.4f}  (r2'-r)/eps={ (r2-r)/eps:10.6f}  (r3'-r2')/eps={(r3-r2)/eps:11.3e}")

print("\n=== (2) beta-function beta(t,r)=dr'/du|_{u=0}, finite t ===")
print("compare to small-t prediction -(lambda+mu) r(1-r) = -(kap+1)*mu*r*(1-r)")
kap,mu,r = 0.5,1.0,0.5
pred0 = -(kap*mu+mu)*r*(1-r)
h=1e-6
print(f"  prediction (t->0):  {pred0:.6f}")
for t in [0.01,0.1,0.5,1.0,2.0,4.0,8.0]:
    s,p,M2,M3 = pieces(kap,mu,r,t,h)
    r2=r_from_p(kap,p_from_M2(s,M2))
    beta = (r2 - r)/h
    print(f"  t={t:5.2f}  beta(t,r)={beta:10.6f}")

print("\n=== check candidate autonomous solution r/(1-r)=r0/(1-r0) e^{-(lam+mu)t} ===")
print("if beta were t-indep & = -(lam+mu)r(1-r), the odds decay exponentially.")
print("integrate ODE dr/dt=beta(t,r) numerically (RK4) vs that formula:")
kap,mu,r0 = 0.5,1.0,0.6
def beta_fn(t,r):
    h=1e-6; s,p,M2,M3=pieces(kap,mu,r,t,h); return (r_from_p(kap,p_from_M2(s,M2))-r)/h
def odds(rr): return rr/(1-rr)
r_ode=r0; dt=0.001; T=3.0; n=int(T/dt)
for i in range(n):
    t=i*dt
    k1=beta_fn(t,r_ode); k2=beta_fn(t+dt/2,r_ode+dt/2*k1)
    k3=beta_fn(t+dt/2,r_ode+dt/2*k2); k4=beta_fn(t+dt,r_ode+dt*k3)
    r_ode+=dt/6*(k1+2*k2+2*k3+k4)
auto = r0/(1-r0)*math.exp(-(kap*mu+mu)*T)
auto_r = auto/(1+auto)
print(f"  ODE-integrated r(T={T}) = {r_ode:.6f}   odds={odds(r_ode):.6f}")
print(f"  autonomous-formula r(T) = {auto_r:.6f}   odds={auto:.6f}")
