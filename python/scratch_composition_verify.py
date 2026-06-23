"""
SymPy verification for tkf/composition-renormalization.tex.

Checks, against the appendix:
  (1) Lemma 5.1  : beta, gamma depend on (lambda,mu,t) only through (kappa,alpha);
                   the (kappa,alpha) closed forms equal the standard TKF91 ones;
                   identity  exp((lambda-mu) t) = alpha**(1-kappa).
  (2) Prop 5.3   : r=0 (TKF91) is an exact semigroup -- the link birth-death
                   pgf composes:  F(F(z,u),t) = F(z,t+u)  (Moebius semigroup).
  (3) Theorem 5.1: minimal latent-fragment deletion model. Explicit defect
                   operator  Delta_2 = Tddot'(0) - Tdot'(0)^2  in closed form,
                   its vanishing iff r=0, and the leading self-composition
                   identity  T(t)T(u) - T(t+u) = -t*u*Delta_2 + O(s^3).
"""
import sympy as sp

lam, mu, t, u, z, r, kappa, alpha = sp.symbols(
    'lambda mu t u z r kappa alpha', positive=True)

print("=" * 70)
print("(1) Lemma 5.1: (kappa, alpha) reparametrisation of beta, gamma")
print("=" * 70)

# Standard TKF91 forms (body-tkf91.tex, eq tkf:beta / tkf:gamma):
beta_std  = lam * (1 - sp.exp((lam - mu) * t)) / (mu - lam * sp.exp((lam - mu) * t))
gamma_std = 1 - mu * beta_std / (lam * (1 - sp.exp(-mu * t)))

# identity  exp((lambda-mu) t) = alpha**(1-kappa)   with alpha=exp(-mu t), kappa=lam/mu
id_check = sp.simplify(
    sp.exp((lam - mu) * t) - sp.exp(-mu * t) ** (1 - lam / mu))
print("exp((lam-mu)t) - alpha**(1-kappa)  simplifies to:", id_check,
      " -> OK" if id_check == 0 else " -> MISMATCH")

# appendix (kappa, alpha) closed forms
beta_ka  = kappa * (1 - alpha ** (1 - kappa)) / (1 - kappa * alpha ** (1 - kappa))
gamma_ka = 1 - beta_ka / (kappa * (1 - alpha))

# substitute kappa=lam/mu, alpha=exp(-mu t) into the (kappa,alpha) forms and
# compare to the standard forms
subs = {kappa: lam / mu, alpha: sp.exp(-mu * t)}
beta_diff  = sp.simplify(beta_ka.subs(subs)  - beta_std)
gamma_diff = sp.simplify(gamma_ka.subs(subs) - gamma_std)
print("beta_(kappa,alpha)  - beta_std  =", beta_diff,
      " -> OK" if beta_diff == 0 else " -> MISMATCH")
print("gamma_(kappa,alpha) - gamma_std =", gamma_diff,
      " -> OK" if gamma_diff == 0 else " -> MISMATCH")

print()
print("=" * 70)
print("(2) Prop 5.3: TKF91 link birth-death pgf is an exact semigroup")
print("=" * 70)

xi = lam - mu
# linear birth-death (birth lam, death mu), X(0)=1, pgf F(z,t):
F = (mu * (1 - z) - (mu - lam * z) * sp.exp(-xi * t)) \
    / (lam * (1 - z) - (mu - lam * z) * sp.exp(-xi * t))
print("F(z,0) =", sp.simplify(F.subs(t, 0)), " (should be z)")
print("F(1,t) =", sp.simplify(F.subs(z, 1)), " (should be 1)")

# semigroup:  F(F(z,u), t) ?= F(z, t+u)
F_t = F                                   # F(z, t) but we will compose times
F_inner = F.subs(t, u)                    # F(z, u)
F_compose = F.subs({z: F_inner})          # F( F(z,u), t )
F_direct = F.subs(t, t + u)               # F(z, t+u)
semigroup_diff = sp.simplify(F_compose - F_direct)
print("F(F(z,u),t) - F(z,t+u) =", semigroup_diff,
      " -> SEMIGROUP OK" if semigroup_diff == 0 else " -> NOT A SEMIGROUP")

print()
print("=" * 70)
print("(3) Theorem 5.1: explicit defect operator in the minimal model")
print("=" * 70)
# Minimal faithful model of the deletion-side non-Markovianity:
# two adjacent ancestral residues; with prob p they are ONE fragment
# (die together at rate mu); else TWO independent fragments (each dies at mu).
# Observable state = (presence_1, presence_2) in {11,10,01,00}.
# p = equilibrium prob that an adjacent bond is intra-fragment.
p = sp.symbols('p', positive=True)
a = sp.exp(-mu * t)                         # single-residue survival

# kernel rows (states ordered 11, 10, 01, 00)
def row11(tt):
    aa = sp.exp(-mu * tt)
    return [p * aa + (1 - p) * aa**2,
            (1 - p) * aa * (1 - aa),
            (1 - p) * (1 - aa) * aa,
            p * (1 - aa) + (1 - p) * (1 - aa)**2]
def row_single(tt, alive_idx):
    aa = sp.exp(-mu * tt)
    row = [0, 0, 0, 0]
    row[alive_idx] = aa
    row[3] = 1 - aa
    return row

def kernel(tt):
    return sp.Matrix([
        row11(tt),
        row_single(tt, 1),   # from 10
        row_single(tt, 2),   # from 01
        [0, 0, 0, 1],        # from 00 (absorbing)
    ])

T = kernel(t)
# row sums = 1 ?
print("row sums of T(t):", [sp.simplify(s) for s in (T * sp.ones(4, 1))])

G = sp.simplify(T.diff(t).subs(t, 0))     # generator  = Tdot'(0)
B = sp.simplify(T.diff(t, 2).subs(t, 0))  # Tddot'(0)
Delta2 = sp.simplify(B - G * G)           # defect operator
print("\nG = Tdot'(0) =")
sp.pprint(G)
print("\nDelta_2 = Tddot'(0) - Tdot'(0)^2 =")
sp.pprint(Delta2)

# expected closed form on the 11-row:  mu^2 p(1-p) [ +1, -1, -1, +1 ]
expected_row11 = sp.Matrix([[mu**2 * p * (1 - p) * c for c in (1, -1, -1, 1)]])
row11_diff = sp.simplify(Delta2.row(0) - expected_row11)
print("\nDelta_2 row(11) - mu^2 p(1-p)[+,-,-,+] =", row11_diff.T.T,
      " -> OK" if row11_diff.is_zero_matrix else " -> MISMATCH")
print("Delta_2 == 0  iff  p(1-p)=0  iff  p in {0,1}.")

# tie p to TKF92:  p = r / (r + (1-r) kappa)  (the body-tkf92 latent correction)
p_tkf = r / (r + (1 - r) * kappa)
print("\np(r=0) =", sp.simplify(p_tkf.subs(r, 0)), " -> Delta_2(r=0)=0 (CTMC).")
print("p in (0,1) for r in (0,1), kappa>0  ->  Delta_2 != 0  iff  r>0.")
mag = sp.simplify((mu**2 * p * (1 - p)).subs(p, p_tkf))
print("leading defect magnitude  mu^2 p(1-p) =", mag)

print()
print("-- self-composition defect  T(t)T(u) - T(t+u) = -t u Delta_2 + O(s^3) --")
Tt, Tu, Ttu = kernel(t), kernel(u), kernel(t + u)
Dself = Tt * Tu - Ttu
# leading bilinear term: take mixed second derivative d/dt d/du at 0
mixed = sp.simplify(Dself.diff(t).diff(u).subs({t: 0, u: 0}))
print("d^2/dt du [T(t)T(u)-T(t+u)] at 0  =")
sp.pprint(mixed)
print("\n-Delta_2 =")
sp.pprint(sp.simplify(-Delta2))
print("\nmixed - (-Delta_2) =")
sp.pprint(sp.simplify(mixed - (-Delta2)),
          )
ok = sp.simplify(mixed - (-Delta2)).is_zero_matrix
print(" -> Delta_self = -t u Delta_2 to leading order:",
      "OK" if ok else "MISMATCH")
