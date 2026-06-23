"""
TKF92 WFST self-composition, done right: 6-state WFST (S,M,J,I,D,E),
J=Ins0 reachable only from S.  Compose two (branch1 Y|X, branch2 Z|Y),
report the reachable product state count (should be ~14, not 300), and
check r=0 reduces to TKF91(t+u).

State order: S=0, M=1, J=2(Ins0), I=3(Ins1), D=4, E=5.
"""
import sympy as sp

S,M,J,I,D,E = range(6)
NAME = {S:'S',M:'M',J:'J',I:'I',D:'D',E:'E'}

kap, al, r = sp.symbols('kappa alpha r', positive=True)

def beta_gamma(kap, al):
    a1 = al**(1-kap)
    b = kap*(1-a1)/(1-kap*a1)
    g = 1 - b/(kap*(1-al))
    return sp.simplify(b), sp.simplify(g)

def wfst6(kap, al, r):
    """6x6 TKF92 conditional WFST (eq:tkf92-wfst).  Rows/cols S,M,J,I,D,E."""
    b,g = beta_gamma(kap,al); ob,og = 1-b,1-g; p = r+(1-r)*kap
    T = sp.zeros(6,6)
    T[S,M]=ob*al;          T[S,J]=b;            T[S,D]=ob*(1-al);          T[S,E]=ob
    T[M,M]=(r+(1-r)*ob*kap*al)/p; T[M,I]=(1-r)*b; T[M,D]=(1-r)*ob*kap*(1-al)/p; T[M,E]=ob
    T[J,M]=(1-r)*ob*al;    T[J,J]=r+(1-r)*b;    T[J,D]=(1-r)*ob*(1-al);    T[J,E]=(1-r)*ob
    T[I,M]=(1-r)*ob*kap*al/p; T[I,I]=r+(1-r)*b; T[I,D]=(1-r)*ob*kap*(1-al)/p; T[I,E]=ob
    T[D,M]=(1-r)*og*kap*al/p; T[D,I]=(1-r)*g;   T[D,D]=(r+(1-r)*og*kap*(1-al))/p; T[D,E]=og
    return T

# Which columns of a WFST state produce / consume the shared (Y) tape?
# As a transducer X->Y: M,J,I produce a Y symbol; D produces no Y (output-eps).
# As a transducer Y->Z: M,D consume a Y symbol; J,I consume no Y (input-eps).
Yprod = {M,J,I}      # branch1 states that emit a Y character
Ycons = {M,D}        # branch2 states that read a Y character

# Build reachable product states under standard composition (canonical
# eps-order: branch2 input-eps (insertions) before branch1 output-eps).
# Product state = (q1,q2); transitions: sync on a Y char, branch1-D (out-eps),
# branch2-J/I (in-eps).  We enumerate reachability from (S,S).
from collections import deque
start=(S,S); seen={start}; order=[start]; dq=deque([start])
edges=[]   # (src, dst, kind)  kind in {'sync','del1','ins2'}
T1=wfst6(kap,al,r); T2=wfst6(kap,al,r)  # symbolic structure only for reachability
def nz(T,i,j): return T[i,j]!=0
while dq:
    q1,q2=dq.popleft()
    if q1==E and q2==E: continue
    # sync: branch1 emits Y (a in Yprod), branch2 reads Y (c in Ycons)
    for a in Yprod:
        if not nz(T1,q1,a): continue
        for c in Ycons:
            if not nz(T2,q2,c): continue
            d=(a,c); edges.append(((q1,q2),d,'sync'))
            if d not in seen: seen.add(d); order.append(d); dq.append(d)
    # branch1 output-eps (delete X): q1->D, q2 fixed
    if nz(T1,q1,D):
        d=(D,q2); edges.append(((q1,q2),d,'del1'))
        if d not in seen: seen.add(d); order.append(d); dq.append(d)
    # branch2 input-eps (insert Z): q2->J or I, q1 fixed
    for c in (J,I):
        if not nz(T2,q2,c): continue
        d=(q1,c); edges.append(((q1,q2),d,'ins2'))
        if d not in seen: seen.add(d); order.append(d); dq.append(d)
    # End: both ->E
    if nz(T1,q1,E) and nz(T2,q2,E):
        d=(E,E); edges.append(((q1,q2),d,'end'))
        if d not in seen: seen.add(d); order.append(d); dq.append(d)

print(f"reachable product states: {len(seen)}")
for q in order: print("  ", (NAME[q[0]],NAME[q[1]]))
print(f"\nedges: {len(edges)}  (sync/del1/ins2/end = "
      f"{sum(e[2]=='sync' for e in edges)}/{sum(e[2]=='del1' for e in edges)}/"
      f"{sum(e[2]=='ins2' for e in edges)}/{sum(e[2]=='end' for e in edges)})")
