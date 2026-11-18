"""verify_all.py -- symbolic checks for every identity used in proof.md / proof.tex.

Each check prints the reduced form of (LHS - RHS) [must be 0] or a boolean [must be True].

Representation.  A function of a time T is a rational function of
    lam, mu, T, eta = exp(-lam T), al = exp(-mu T)
with eta, al treated as independent indeterminates.  The total T-derivative is the
derivation  D = d_T - lam*eta*d_eta - mu*al*d_al ;  lam-, mu-derivatives at fixed T are
d_lam - T*eta*d_eta and d_mu - T*al*d_al.  If a rational expression in these indeterminates
cancels to the zero rational function, it is zero after substituting the exponentials, so
every reported 0 is an identity in (lam, mu, T) wherever both sides are defined.

Sections
  1  pair-HMM zone bookkeeping (count relations, factorisation of the path weight, affine rank)
  2  closed forms from the score identities + conservation; equality of the three routes' forms
  3  Holmes's (8), (13), (9)-(16), corrected (17'), and the excess of printed (17)
  4  equal rates lam = mu
  5  generating-function (tilted Riccati / Volterra) route: lemmas (a)-(g) and the 8 log-derivatives
  6  remarks (lam <-> mu symmetry)
"""
import itertools
import time
import sympy as sp

T0 = time.time()
lam, mu, T = sp.symbols('lambda mu T', positive=True)
eta, al = sp.symbols('eta alpha', positive=True)      # e^{-lam T}, e^{-mu T}

FAIL = []


def red(e):
    """cancel to lowest terms and return the numerator (0 iff identity)."""
    num, _ = sp.fraction(sp.cancel(sp.together(e)))
    return sp.expand(num)


def check(name, expr):
    r = red(expr)
    ok = (r == 0)
    print(f"[{'ok' if ok else 'FAIL'}] {name}:  reduced = {r if ok else '(nonzero)'}", flush=True)
    if not ok:
        FAIL.append(name)


def check_true(name, b):
    ok = bool(b)
    print(f"[{'ok' if ok else 'FAIL'}] {name}:  {ok}", flush=True)
    if not ok:
        FAIL.append(name)


def DT(f):
    return sp.diff(f, T) - lam * eta * sp.diff(f, eta) - mu * al * sp.diff(f, al)


def Dlam(f):
    return sp.diff(f, lam) - T * eta * sp.diff(f, eta)


def Dmu(f):
    return sp.diff(f, mu) - T * al * sp.diff(f, al)


EXPS = {eta: sp.exp(-lam * T), al: sp.exp(-mu * T)}


def lim0(e, var=T):
    """lim_{T->0+} by series (exponentials substituted)."""
    ee = e.subs(EXPS) if var is T else e
    ser = sp.series(ee, var, 0, 2).removeO()
    return sp.simplify(ser.subs(var, 0))


# TKF91 factors, eq:abg (= Holmes's alpha, beta, gamma)
alpha = al
delta = mu * eta - lam * al
beta = lam * (eta - al) / delta
gamma = 1 - mu * beta / (lam * (1 - alpha))
lv = lam * alpha
lw = lam * (1 - alpha) * gamma
lx = lam * (1 - alpha) * (1 - gamma)
q = (1 - alpha) * (1 - gamma)          # inferred q_0

print("=" * 78)
print("1. Pair-HMM zone bookkeeping")
print("=" * 78)
# zone type of each transition (Holmes sec 5.4)
ZONE = {}
for a in 'SMI':
    for b in 'MDE':
        ZONE[(a, b)] = '*'
    ZONE[(a, 'I')] = '**'
for b in 'MDE':
    ZONE[('D', b)] = 'o'
ZONE[('D', 'I')] = 'o*'
A_, B_, G_ = sp.symbols('A B G', positive=True)   # generic alpha, beta, gamma
TAU = {}
for a in 'SMI':
    TAU[(a, 'M')] = (1 - B_) * A_; TAU[(a, 'I')] = B_
    TAU[(a, 'D')] = (1 - B_) * (1 - A_); TAU[(a, 'E')] = 1 - B_
TAU[('D', 'M')] = (1 - G_) * A_; TAU[('D', 'I')] = G_
TAU[('D', 'D')] = (1 - G_) * (1 - A_); TAU[('D', 'E')] = 1 - G_
PZg = {'o': (1 - A_) * (1 - G_), '*': A_ * (1 - B_), 'o*': (1 - A_) * G_ / A_, '**': B_}
cz = {'o': -1, '*': 0, 'o*': 0, '**': 1}
ok_counts, ok_fact, npaths = True, True, 0
vecs = set()
for L in range(0, 8):
    for states in itertools.product('MID', repeat=L):
        path = ('S',) + states + ('E',)
        N = {k: 0 for k in PZg}
        w = sp.Integer(1)
        for a, b in zip(path[:-1], path[1:]):
            N[ZONE[(a, b)]] += 1
            w *= TAU[(a, b)]
        nM, nI, nD = states.count('M'), states.count('I'), states.count('D')
        i_len, j_len = nM + nD, nM + nI
        ok_counts &= (nI == N['o*'] + N['**'] and nD == N['o'] + N['o*']
                      and nM == N['*'] - N['o*'] - 1 and i_len == N['o'] + N['*'] - 1
                      and j_len - i_len == N['**'] - N['o'])
        prod = sp.Integer(1) / A_
        for k in PZg:
            prod *= PZg[k] ** N[k]
        ok_fact &= (sp.cancel(w - prod) == 0)
        vecs.add((1, N['o'], N['*'], N['o*'], N['**']))
        npaths += 1
check_true(f"count relations #I=N_o*+N_**, #D=N_o+N_o*, #M=N_*-N_o*-1, i=N_o+N_*-1, "
           f"j-i=N_**-N_o on all {npaths} paths of length<=7", ok_counts)
check_true("path weight == prod_z P_z^{N_z} / alpha on all those paths "
           "(P_o=(1-a)(1-g), P_*=a(1-b), P_o*=(1-a)g/a, P_**=b)", ok_fact)
Mrank = sp.Matrix(sorted(vecs)).rank()
check_true("affine rank: vectors (1,N_o,N_*,N_o*,N_**) over alignments span R^5 (rank==5)",
           Mrank == 5)
# explicit witness set: '', 'M', 'D', 'I', 'DI'
W5 = sp.Matrix([[1, 0, 1, 0, 0], [1, 0, 2, 0, 0], [1, 1, 1, 0, 0], [1, 0, 1, 0, 1], [1, 0, 2, 1, 0]])
check_true("witness alignments '', M, D, I, DI: det != 0", W5.det() != 0)

print("=" * 78)
print("2. Closed forms from the score identities and conservation")
print("=" * 78)
check("Lemma: lam_x == mu*beta  [q_0 = (1-alpha)(1-gamma) = mu*beta/lam]", lx - mu * beta)
check("Lemma: lam_v + lam_w + lam_x == lam", lv + lw + lx - lam)
check("Riccati for beta: D beta == (1-beta)(lam - mu beta)", DT(beta) - (1 - beta) * (lam - mu * beta))
check("(1-beta)(lam_v+lam_w) == D beta", (1 - beta) * (lv + lw) - DT(beta))

ell = {'o': sp.log((1 - alpha) * (1 - gamma)), '*': sp.log(alpha * (1 - beta)),
       'o*': sp.log((1 - alpha) * gamma / alpha), '**': sp.log(beta)}
Lz, Mz, Z, I = {}, {}, {}, {}
for k in ell:
    Lz[k] = sp.cancel(sp.together(lam * Dlam(ell[k])))
    Mz[k] = sp.cancel(sp.together(mu * Dmu(ell[k])))
    Z[k] = sp.cancel(sp.together((cz[k] + Mz[k] - Lz[k]) / (lam - mu)))
    I[k] = sp.cancel(sp.together(Lz[k] + lam * Z[k]))
# Theorem 1 algebra: with N_z symbolic, score identities + conservation => totals
No, Ns, Nos, Nss, EB, ED, ES = sp.symbols('N_o N_s N_os N_ss EB ED ES')
Nv = {'o': No, '*': Ns, 'o*': Nos, '**': Nss}
sumL = sum(Nv[k] * Lz[k] for k in ell)
sumM = sum(Nv[k] * Mz[k] for k in ell)
# lam d_lam log P = sumL ; mu d_mu log P = sumM + mu T   (constant C = -log alpha = mu T)
check("constant: lam d_lam(-log alpha) == 0", Dlam(-sp.log(alpha)) * lam)
check("constant: mu d_mu(-log alpha) == mu T", Dmu(-sp.log(alpha)) * mu - mu * T)
sol = sp.solve([sumL - (EB - lam * (ES + T)), sumM + mu * T - (ED - mu * ES),
                EB - ED - sum(Nv[k] * cz[k] for k in ell)], [EB, ED, ES], dict=True)[0]
check("Theorem 1: E[S|pi] + T == sum_z N_z Z_z", sol[ES] + T - sum(Nv[k] * Z[k] for k in ell))
check("Theorem 1: E[B|pi] == sum_z N_z I_z", sol[EB] - sum(Nv[k] * I[k] for k in ell))
check("Theorem 1: E[D|pi] == sum_z N_z (I_z - c_z)", sol[ED] - sum(Nv[k] * (I[k] - cz[k]) for k in ell))

# explicit (factored) forms used in the note, k = lam - mu, eps = lam(1-al) - mu(1-eta)
kk = lam - mu
eps = lam * (1 - al) - mu * (1 - eta)
FORM = {
    ('I', 'o'): lam * mu * (eta**2 - al**2 + 2 * kk * T * eta * al) / (kk * (al - eta) * delta),
    ('Z', 'o'): (mu * eta**2 - lam * al**2 + kk * eta * al + kk * (lam + mu) * T * eta * al) / (kk * (al - eta) * delta),
    ('I', '*'): lam * mu * ((lam + mu) * (al - eta) - kk * T * (lam * al + mu * eta)) / (kk**2 * delta),
    ('Z', '*'): (2 * lam * mu * (al - eta) - kk * T * (lam**2 * al + mu**2 * eta)) / (kk**2 * delta),
    ('I', 'o*'): lam * mu * ((lam + mu) * (al - eta) - kk * T * (lam * al + mu * eta)) / (kk * delta * eps),
    ('Z', 'o*'): lam * mu * (2 * (al - eta) - kk * T * (al + eta)) / (kk * delta * eps),
}
FORM[('I', '**')] = 1 + FORM[('I', 'o')]
FORM[('Z', '**')] = FORM[('Z', 'o')]
for (nm, k), f in FORM.items():
    src = I if nm == 'I' else Z
    check(f"explicit form {nm}_{k} == score-identity definition", src[k] - f)
check("I_o* == (k/eps) I_*", I['o*'] - kk / eps * I['*'])
check("P_o / P_** == mu/lam (T-free)  => I_** = I_o + 1, Z_** = Z_o", (1 - alpha) * (1 - gamma) / beta - mu / lam)
check("I_** - I_o - 1 == 0", I['**'] - I['o'] - 1)
check("Z_** - Z_o == 0", Z['**'] - Z['o'])
# Route A's forms (d = mu-lam, E = e^{dT} = eta/al) and route C's forms (x = e^{(lam-mu)T} = al/eta)
d = mu - lam
E = eta / al
Poc = 1 - al - mu * beta / lam
A_FORM = {
    ('Z', 'o'): (mu * E + lam) / (d * (mu * E - lam)) - (lam + mu) * T * E / ((E - 1) * (mu * E - lam)),
    ('I', 'o'): lam * mu * (E + 1) / (d * (mu * E - lam)) - 2 * lam * mu * T * E / ((E - 1) * (mu * E - lam)),
    ('Z', '*'): T * (mu**2 * E + lam**2) / (d * (mu * E - lam)) - 2 * lam * mu * (E - 1) / (d**2 * (mu * E - lam)),
    ('I', '*'): lam * mu * (T * (mu * E + lam) / (d * (mu * E - lam)) - (lam + mu) * (E - 1) / (d**2 * (mu * E - lam))),
    ('Z', 'o*'): lam * mu * (d * T * (E + 1) - 2 * (E - 1)) / (d * (mu * E - lam)**2 * Poc),
}
for (nm, k), f in A_FORM.items():
    check(f"route-A form {nm}_{k} == definition", (I if nm == 'I' else Z)[k] - f)
x_, y_ = al / eta, 1 / al
ep = lam - mu
C_FORM = {
    ('I', 'o'): lam * mu * ((x_**2 - 1) - 2 * ep * T * x_) / (ep * (x_ - 1) * (lam * x_ - mu)),
    ('Z', 'o'): ((x_ - 1) * (lam * x_ + mu) - ep * T * (lam + mu) * x_) / (ep * (x_ - 1) * (lam * x_ - mu)),
    ('I', '*'): lam * mu * (ep * T * (lam * x_ + mu) - (lam + mu) * (x_ - 1)) / (ep**2 * (lam * x_ - mu)),
    ('Z', '*'): (ep * T * (lam**2 * x_ + mu**2) - 2 * lam * mu * (x_ - 1)) / (ep**2 * (lam * x_ - mu)),
    ('I', 'o*'): lam * mu * x_ * y_ * (ep * T * (lam * x_ + mu) - (lam + mu) * (x_ - 1)) / (ep * (lam * x_ - mu) * (ep * x_ * y_ - lam * x_ + mu)),
    ('Z', 'o*'): lam * mu * x_ * y_ * (ep * T * (x_ + 1) - 2 * (x_ - 1)) / (ep * (lam * x_ - mu) * (ep * x_ * y_ - lam * x_ + mu)),
}
for (nm, k), f in C_FORM.items():
    check(f"route-C form {nm}_{k} == definition", (I if nm == 'I' else Z)[k] - f)

print("=" * 78)
print("3. Holmes's equations (8)-(17)")
print("=" * 78)
# two-time representation: (eta, al) at T ; (eta2, al2) at s = T - t ; alpha(t) = al/al2
eta2, al2 = sp.symbols('eta2 alpha2', positive=True)
SUB_s = {eta: eta2, al: al2}
beta_s, q_s, lv_s, lw_s = [e.subs(SUB_s) for e in (beta, q, lv, lw)]
alpha_t = al / al2
v = alpha_t * (1 - beta) / (1 - beta_s)
# (8): d/dt log v(t|T) at fixed T equals -mu - d/dt log(1-beta(T-t)) = -mu - beta'(s)/(1-beta(s))
check("(8): beta'/(1-beta) == lam_v + lam_w  [so d_t log v = -(mu + lam_v + lam_w)(T-t)]",
      DT(beta) / (1 - beta) - (lv + lw))
check("(8): v(0|T) == 1", v.subs({eta2: eta, al2: al}) - 1)
# (13) separable forms
a_k = v * q_s / q
b_k = alpha_t * q_s / ((1 - alpha) * gamma)
c_k = (1 - beta_s) / beta
Fk = {'a': alpha * (1 - beta) / q, 'b': alpha / ((1 - alpha) * gamma), 'c': 1 / beta}
Gk = {'a': q / (alpha * (1 - beta)), 'b': q / alpha, 'c': 1 - beta}
for nm, kern in (('a', a_k), ('b', b_k), ('c', c_k)):
    check(f"(13) {nm}(t|T) == F_{nm}(T) G_{nm}(T-t)", kern - Fk[nm] * Gk[nm].subs(SUB_s))
# semigroup (S): int_t^T mu v(l|T) dl = v(t|T) q(T-t)  <=>  d_t[v q(T-t)] = -mu v, and q(0)=0
# d_t = -D on functions of s (alpha(t) = al/al2: d_t log alpha(t) = -mu)
# compute directly: log[v q(s)] = log alpha(t) + log(1-beta(T)) - log(1-beta(s)) + log q(s)
#   d_t = -mu + [D log(1-beta)](s) - [D log q](s)
dlog = -mu + (DT(sp.log(1 - beta)) - DT(sp.log(q)))
check("(S): d_t[v(t|T) q(T-t)] == -mu v(t|T)   (i.e. q*dlog == -mu)", q * dlog + mu)
check("(S) endpoint: q(0) == 0", lim0(q))
# normalisations of (13)
# (the evidence that a(t|T) is the posterior survival function of the marker given o is the
#  (S) check above: int_t^T mu v(l|T) dl = v(t|T) q(T-t).)  The next identity is equivalent to
#  int_0^T lam v(t|T) dt = beta(T); it is a side identity, not used in proof.md.
check("side identity: D[beta/(alpha(1-beta))] == lam/(alpha(1-beta))  [<=> int_0^T lam v(t|T) dt = beta(T)]",
      DT(beta / (alpha * (1 - beta))) - lam / (alpha * (1 - beta)))
check("side identity: beta/(alpha(1-beta)) -> 0 at T=0", lim0(beta / (alpha * (1 - beta))))
check("(13) b: D[1/F_b] == G_b (lam_v+lam_w)   [int b (lam_v+lam_w) = 1]", DT(1 / Fk['b']) - Gk['b'] * (lv + lw))
check("(13) b: 1/F_b -> 0 at T=0", lim0(1 / Fk['b']))
check("(13) c: D[1/F_c] == G_c (lam_v+lam_w)   [int c (lam_v+lam_w) = 1]", DT(1 / Fk['c']) - Gk['c'] * (lv + lw))
check("(13) c: 1/F_c -> 0 at T=0", lim0(1 / Fk['c']))
check("q nondecreasing: D q == mu (1-beta)(lam - mu beta)/lam", DT(q) - mu * (1 - beta) * (lam - mu * beta) / lam)

# bound used in the uniqueness proof for (11), (16): (1-alpha)gamma ~ lam mu T^2/2 as T -> 0
check("(1-alpha)gamma / T^2 -> lam mu / 2 at T=0", sp.limit(((1 - alpha) * gamma / T**2).subs(EXPS), T, 0, '+') - lam * mu / 2)
check("q / T -> mu at T=0", sp.limit((q / T).subs(EXPS), T, 0, '+') - mu)
# uniqueness (Theorem 3): with u = d/F, the homogeneous equations become u(T) = int_0^T k(s) u(s) ds
check("Theorem 3 kernel for (9),(14): G_a F_a lam_x == lam_x (G_a F_a == 1)", Gk['a'] * Fk['a'] - 1)
check("Theorem 3 kernel for (11),(16): G_b F_b lam_w == lam q", Gk['b'] * Fk['b'] * lw - lam * q)
# nonvanishing denominators (remark after Theorem 2)
eps_ = lam * (1 - al) - mu * (1 - eta)
check("(1-alpha) gamma == -alpha eps / delta", (1 - alpha) * gamma + alpha * eps_ / delta)

EQ = {
    '(9)  I_o ': ('a', I['o'], lx * (1 + I['o'])),
    '(10) I_* ': (None, I['*'], lx * (1 + I['o'])),
    '(11) I_o*': ('b', I['o*'], lv * (1 + I['o']) + lw * (1 + I['o'] + I['o*'])),
    '(12) I_**': ('c', I['**'], lv * (1 + I['*']) + lw * (1 + I['*'] + I['o*'])),
    '(14) Z_o ': ('a', Z['o'], 1 + lx * Z['o']),
    '(15) Z_* ': (None, Z['*'], 1 + lx * Z['o']),
    '(16) Z_o*': ('b', Z['o*'], lv * Z['o'] + lw * (Z['o'] + Z['o*'])),
    "(17') Z_** (no 't+')": ('c', Z['**'], lv * Z['*'] + lw * (Z['*'] + Z['o*'])),
}
for name, (kn, X, Rr) in EQ.items():
    t1 = time.time()
    if kn is None:
        check(f"{name}: D X == R", DT(X) - Rr)
        check(f"{name}: X(0) == 0", lim0(X))
    else:
        check(f"{name}: D[X/F_{kn}] == G_{kn} R", DT(X / Fk[kn]) - Gk[kn] * Rr)
        check(f"{name}: lim_(T->0) X/F_{kn} == 0", lim0(X / Fk[kn]))
# T -> 0 limits of the closed forms (remark after Theorem 2)
LIM0 = {('I', 'o'): 0, ('I', '*'): 0, ('I', 'o*'): 1, ('I', '**'): 1,
        ('Z', 'o'): 0, ('Z', '*'): 0, ('Z', 'o*'): 0, ('Z', '**'): 0}
for (nm, k), val in LIM0.items():
    src = (I if nm == 'I' else Z)[k]
    check(f"lim_(T->0) {nm}_{k} == {val}", sp.limit(src.subs(EXPS), T, 0, '+') - val)
# printed (17): extra = int_0^T c(t|T)(lam_v+lam_w)(T-t) t dt.  With s=T-t:
#   = (1/beta(T)) int_0^T beta'(s)(T-s) ds = (1/beta(T)) int_0^T beta(s) ds   (parts, beta(0)=0)
Jb = (lam * T + sp.log(1 - beta)) / mu          # claimed antiderivative of beta
check("(17) excess: D[(lam T + log(1-beta))/mu] == beta", DT(Jb) - beta)
check("(17) excess: (lam T + log(1-beta))/mu -> 0 at T=0", lim0(Jb))
check("(17) excess integrand: c(t|T)(lam_v+lam_w)(s) == beta'(s)/beta(T)",
      c_k * (lv_s + lw_s) - DT(beta).subs(SUB_s) / beta)

print("=" * 78)
print("4. Equal rates lam = mu")
print("=" * 78)
k_ = sp.symbols('k')
sig = mu * T
EQF = {
    ('I', 'o'): sig**2 / (3 * (1 + sig)),
    ('Z', 'o'): T * (2 * sig + 3) / (6 * (1 + sig)),
    ('I', '*'): sig**2 * (sig + 3) / (6 * (1 + sig)),
    ('Z', '*'): T * (sig**2 + 6 * sig + 6) / (6 * (1 + sig)),
    ('I', 'o*'): sig**2 * (sig + 3) / (6 * (1 + sig) * (1 - (1 + sig) * al)),
    ('Z', 'o*'): sig**2 * T / (6 * (1 + sig) * (1 - (1 + sig) * al)),
}
EQF[('I', '**')] = 1 + EQF[('I', 'o')]
EQF[('Z', '**')] = EQF[('Z', 'o')]
for (nm, k), f in EQF.items():
    src = (I if nm == 'I' else Z)[k]
    e = src.subs({lam: mu + k_, eta: al * sp.exp(-k_ * T)})
    ser = sp.series(e, k_, 0, 1).removeO()
    has_pole = any(t.as_coeff_exponent(k_)[1].is_negative for t in sp.Add.make_args(sp.expand(ser)))
    check_true(f"lam->mu: no pole in k for {nm}_{k}", not has_pole)
    lim = ser.subs(k_, 0)
    check(f"lam->mu: limit of {nm}_{k} == stated equal-rate form", sp.expand(lim) - f)
# direct verification at lam = mu
DTe = lambda f: sp.diff(f, T) - mu * al * sp.diff(f, al)
be = mu * T / (1 + mu * T)
ge = 1 - be / (1 - al)
lve, lwe, lxe = mu * al, mu * (1 - al) * ge, mu * (1 - al) * (1 - ge)
qe = (1 - al) * (1 - ge)
check("lam=mu: beta = muT/(1+muT) is the lam->mu limit of beta",
      sp.series(beta.subs({lam: mu + k_, eta: al * sp.exp(-k_ * T)}), k_, 0, 1).removeO().subs(k_, 0) - be)
check("lam=mu: D beta == (1-beta)(mu - mu beta)", DTe(be) - (1 - be) * (mu - mu * be))
check("lam=mu: lam_x == mu beta", lxe - mu * be)
Fe = {'a': al * (1 - be) / qe, 'b': al / ((1 - al) * ge), 'c': 1 / be}
Ge = {'a': qe / (al * (1 - be)), 'b': qe / al, 'c': 1 - be}
Ie = {k: EQF[('I', k)] for k in ell}
Ze = {k: EQF[('Z', k)] for k in ell}
EQE = {
    '(9)': ('a', Ie['o'], lxe * (1 + Ie['o'])),
    '(10)': (None, Ie['*'], lxe * (1 + Ie['o'])),
    '(11)': ('b', Ie['o*'], lve * (1 + Ie['o']) + lwe * (1 + Ie['o'] + Ie['o*'])),
    '(12)': ('c', Ie['**'], lve * (1 + Ie['*']) + lwe * (1 + Ie['*'] + Ie['o*'])),
    '(14)': ('a', Ze['o'], 1 + lxe * Ze['o']),
    '(15)': (None, Ze['*'], 1 + lxe * Ze['o']),
    '(16)': ('b', Ze['o*'], lve * Ze['o'] + lwe * (Ze['o'] + Ze['o*'])),
    "(17')": ('c', Ze['**'], lve * Ze['*'] + lwe * (Ze['*'] + Ze['o*'])),
}
EXPe = {al: sp.exp(-mu * T)}
for name, (kn, X, Rr) in EQE.items():
    if kn is None:
        check(f"lam=mu {name}: D X == R", DTe(X) - Rr)
        check(f"lam=mu {name}: X(0) == 0", sp.limit(X.subs(EXPe), T, 0, '+'))
    else:
        check(f"lam=mu {name}: D[X/F] == G R", DTe(X / Fe[kn]) - Ge[kn] * Rr)
        check(f"lam=mu {name}: lim X/F == 0", sp.limit((X / Fe[kn]).subs(EXPe), T, 0, '+'))
check("lam=mu (13) b normalisation", DTe(1 / Fe['b']) - Ge['b'] * (lve + lwe))
check("lam=mu (13) c normalisation", DTe(1 / Fe['c']) - Ge['c'] * (lve + lwe))

print("=" * 78)
print("5. Generating-function route (tilted Riccati / Volterra)")
print("=" * 78)
# Parametrise by the roots p = z_+, m = z_- of lam x^2 - (lam+mu+th) x + mu w:
#   lam (p+m) = lam + mu + th,  lam p m = mu w.
p, m, z, th, w, tt = sp.symbols('p m z theta w t')
Ex = sp.symbols('E', positive=True)        # E = e^{Delta t}, Delta = lam (p - m)
Dl = lam * (p - m)
mu_w = lam * p * m                       # = mu*w
lmt = lam * (p + m)                      # = lam + mu + th
f = (p * (z - m) - m * (z - p) * Ex) / ((z - m) - (z - p) * Ex)     # eq:pgf-closed with tilted roots
Dt_E = lambda e: sp.diff(e, Ex) * Dl * Ex
check("tilted Riccati: d_t f == lam f^2 - (lam+mu+th) f + mu w", Dt_E(f) - (lam * f**2 - lmt * f + mu_w))
check("tilted Riccati: f(t=0) == z", f.subs(Ex, 1) - z)
# I_D(t,Tw) = e^{lam(p-1)Tw} (1 - rho_{t-Tw})/(1 - rho_t),  rho_s = (z-p)/(z-m) e^{Delta s}
Et, ETw, Ep1 = sp.symbols('E_t E_Tw Ep1', positive=True)    # e^{Delta t}, e^{Delta Tw}, e^{lam(p-1)Tw}
rho = lambda Es: (z - p) / (z - m) * Es
ID = Ep1 * (1 - rho(Et / ETw)) / (1 - rho(Et))
dTw_logID = (sp.diff(ID, ETw) * Dl * ETw + sp.diff(ID, Ep1) * lam * (p - 1) * Ep1) / ID
f_tmTw = f.subs(Ex, Et / ETw)
check("I_D: d_Tw log I_D(t,Tw) == lam (f_{t-Tw} - 1)   [I_D = exp(lam int (f-1))]", dTw_logID - lam * (f_tmTw - 1))
check("I_D: I_D(t,0) == 1", ID.subs({ETw: 1, Ep1: 1}) - 1)
# the three functions B, W, R at time T (E = e^{Delta T})
Bf = (Ex - 1) / (p * Ex - m)
f0 = f.subs(z, 0)
check("Lemma: F_T(0;th,w) == (mu w/lam) B", f0 - (mu_w / lam) * Bf)
F1 = sp.diff(f, z).subs(z, 0)
check("Lemma: F^(1) := [z^1]F == (1 - p B)(1 - m B)", F1 - (1 - p * Bf) * (1 - m * Bf))
check("Lemma: d_T B == lam F^(1)", Dt_E(Bf) - lam * F1)
check("Lemma: d_T B == lam - (lam+mu+th) B + mu w B^2 (same Riccati as F(0) after scaling)",
      Dt_E(Bf) - (lam - lmt * Bf + mu_w * Bf**2))
check("Lemma: B(0) == 0", Bf.subs(Ex, 1))
# I_D(T,T;z) = I_D(T,T;0)/(1 - B z): take t = Tw (E_t = E_Tw = E)
IDTT = ID.subs({Et: Ex, ETw: Ex})
check("Lemma: I_D(T,T;z) == I_D(T,T;0)/(1 - B z)", IDTT - IDTT.subs(z, 0) / (1 - Bf * z))
# W = e^{-th T} I_D(T,T;0) = e^{(lam p - lam - th)T}(p-m)/(pE-m) ; Ep1 = e^{lam(p-1)T}
EthT = sp.symbols('Eth', positive=True)                     # e^{-th T}
Wf = EthT * IDTT.subs(z, 0)
check("W == e^{-th T} e^{lam(p-1)T} (p-m)/(pE-m)", Wf - EthT * Ep1 * (p - m) / (p * Ex - m))
# F^(1)/W == (p-m) E e^{-lam(p-1)T} e^{th T}/(pE - m)
check("F^(1)/W == (p-m) E /(Ep1 Eth (pE-m))   [= (p-m)e^{(lam+th-lam m)T}/(pE-m)]",
      F1 / Wf - (p - m) * Ex / (Ep1 * EthT * (p * Ex - m)))

# Log-derivatives at (th, w) = (0, 1).  Implicit derivatives of the roots:
#   d p = (-mu dw + p dth) / (2 lam p - lam - mu - th), same for m.
p0, m0 = mu / lam, sp.Integer(1)
den_p, den_m = 2 * lam * p0 - lam - mu, 2 * lam * m0 - lam - mu
dp = {'w': -mu / den_p, 'th': p0 / den_p}
dm = {'w': -mu / den_m, 'th': m0 / den_m}
check("root derivative consistency: d(p+m)/dth == 1/lam", dp['th'] + dm['th'] - 1 / lam)
check("root derivative consistency: d(pm)/dw == mu/lam", p0 * dm['w'] + m0 * dp['w'] - mu / lam)
# explicit exponentials, as functions of (p, m, th, T):
P_, M_, TH_ = sp.symbols('P M TH')
EXPL = {
    'E': sp.exp(lam * (P_ - M_) * T),          # e^{Delta T}
    'Ep1': sp.exp(lam * (P_ - 1) * T),         # e^{lam(p-1)T}
    'Eth': sp.exp(-TH_ * T),                   # e^{-th T}
}
Bx = (EXPL['E'] - 1) / (P_ * EXPL['E'] - M_)
Wx = EXPL['Eth'] * EXPL['Ep1'] * (P_ - M_) / (P_ * EXPL['E'] - M_)
Rx = (P_ - M_) * EXPL['E'] / (EXPL['Ep1'] * EXPL['Eth'] * (P_ * EXPL['E'] - M_)) - al
BASE = {P_: p0, M_: m0, TH_: 0}


def to_eta_al(e):
    def rep(arg):
        arg = sp.expand(arg)
        cl = sp.simplify(sp.diff(arg, lam) / T)
        cm = sp.simplify(sp.diff(arg, mu) / T)
        assert sp.simplify(arg - cl * lam * T - cm * mu * T) == 0 and cl.is_number and cm.is_number, arg
        return eta ** (-cl) * al ** (-cm)
    return e.replace(sp.exp, rep)


def dlog(F, var):
    g = sp.diff(F, P_) * dp[var] + sp.diff(F, M_) * dm[var]
    if var == 'th':
        g += sp.diff(F, TH_)
    return to_eta_al(sp.simplify((g / F).subs(BASE)))


check("base point: B == beta", to_eta_al(Bx.subs(BASE)) - beta)
check("base point: W == 1 - beta", to_eta_al(Wx.subs(BASE)) - (1 - beta))
check("base point: R == (1-alpha) gamma", to_eta_al(Rx.subs(BASE)) - (1 - alpha) * gamma)
check("base point: F_T(0) == q", (mu / lam) * beta - q)
GF = {
    ('I', 'o'): dlog(Bx, 'w'), ('Z', 'o'): -dlog(Bx, 'th'),
    ('I', '*'): dlog(Wx, 'w'), ('Z', '*'): -dlog(Wx, 'th'),
    ('I', 'o*'): dlog(Rx, 'w'), ('Z', 'o*'): -dlog(Rx, 'th'),
    ('I', '**'): 1 + dlog(Bx, 'w'), ('Z', '**'): -dlog(Bx, 'th'),
}
for (nm, k), g in GF.items():
    check(f"GF route == score route: {nm}_{k}", g - (I if nm == 'I' else Z)[k])
# family-event check of the 'what must equal what' table via score identities applied to
# family probabilities P(surv,n)=al(1-b)b^n, P(die,0)=q, P(die,n)=(1-al)g(1-b)b^(n-1)
def fam_stats(logP, c_fam):
    Lf = lam * Dlam(logP); Mf = mu * Dmu(logP)
    # lam d_lam log P = E[B] - lam E[S_lam], mu d_mu log P = E[D] - mu E[S_mu], E[B]-E[D] = c
    S = (c_fam + Mf - Lf) / (lam - mu)
    return sp.cancel(sp.together(Lf + lam * S)), sp.cancel(sp.together(S))
n_ = 3
Bs = {}
for n in range(0, n_):
    Bs[('surv', n)] = fam_stats(sp.log(al * (1 - beta) * beta**n), n)
Bs[('die', 0)] = fam_stats(sp.log(q), -1)
for n in range(1, n_):
    Bs[('die', n)] = fam_stats(sp.log((1 - al) * gamma * (1 - beta) * beta**(n - 1)), n - 1)
check("family: E[X|surv,0] == X_* (births)", Bs[('surv', 0)][0] - I['*'])
check("family: E[X|surv,0] == X_* (exposure)", Bs[('surv', 0)][1] - Z['*'])
check("family: E[X|die,0] == X_o (births)", Bs[('die', 0)][0] - I['o'])
check("family: E[X|die,0] == X_o (exposure)", Bs[('die', 0)][1] - Z['o'])
for n in range(0, n_ - 1):
    check(f"family: E[B|surv,{n+1}]-E[B|surv,{n}] == I_**", Bs[('surv', n + 1)][0] - Bs[('surv', n)][0] - I['**'])
    check(f"family: E[S|surv,{n+1}]-E[S|surv,{n}] == Z_**", Bs[('surv', n + 1)][1] - Bs[('surv', n)][1] - Z['**'])
check("family: E[B|die,2]-E[B|die,1] == I_**", Bs[('die', 2)][0] - Bs[('die', 1)][0] - I['**'])
check("family: E[S|die,2]-E[S|die,1] == Z_**", Bs[('die', 2)][1] - Bs[('die', 1)][1] - Z['**'])
check("family: E[B|die,1] - I_* == I_o*", Bs[('die', 1)][0] - I['*'] - I['o*'])
check("family: E[S|die,1] - Z_* == Z_o*", Bs[('die', 1)][1] - Z['*'] - Z['o*'])
# (star): R e^{mu T} = int_0^T lam (1 + e^{mu s} R(s)) F_s(0) ds, in the tilted (th, w) world
#   <=> d_T[R e^{mu T}] == lam (1 + e^{mu T} R) F_T(0) and R(0) = 0.
#   Exponentials: E (rate Delta), Ep1 (rate lam(p-1)), Eth (rate -th), Emu = e^{mu T} (rate mu)
Emu = sp.symbols('Emu', positive=True)
mu_ = lmt - lam - th                       # keep th free; mu w = lam p m fixes w
Rg = (p - m) * Ex / (Ep1 * EthT * (p * Ex - m)) - 1 / Emu
def DTg(e):
    return (sp.diff(e, Ex) * Dl * Ex + sp.diff(e, Ep1) * lam * (p - 1) * Ep1
            + sp.diff(e, EthT) * (-th) * EthT + sp.diff(e, Emu) * mu_ * Emu)
check("(star): d_T[R e^{mu T}] == lam (1 + e^{mu T} R) F_T(0)   [general th, w]",
      DTg(Rg * Emu) - lam * (1 + Emu * Rg) * f0)
check("(star): R(0) == 0", Rg.subs({Ex: 1, Ep1: 1, EthT: 1, Emu: 1}))
check("Lemma: B == lam int_0^T W (e^{-mu s} + R) ds  <=> d_T B == lam W (1/Emu + R)",
      Dt_E(Bf) - lam * Wf.subs(z, 0) * (1 / Emu + Rg))

# Lemmas (d) additivity and (g) semigroup (S) at general (th, w), with explicit exponentials.
# Here mu is eliminated through lam(p+m) = lam + mu + th, and mu w = lam p m.
Tg, tg, thg = sp.symbols('T_g t_g theta_g', positive=True)
mu_g = lam * (p + m) - lam - thg
Dg = lam * (p - m)
F0g = lambda s_: (sp.exp(Dg * s_) - 1) / (p * sp.exp(Dg * s_) - m) * p * m        # F_s(0) = (mu w/lam) B(s)
IDg = lambda TT, tt_: (sp.exp(lam * (p - 1) * tt_) * (1 - (p / m) * sp.exp(Dg * (TT - tt_)))
                       / (1 - (p / m) * sp.exp(Dg * TT)))                       # I_D(TT, tt; 0), eq:ID-closed at z=0
Wg = lambda s_: sp.exp(-thg * s_) * IDg(s_, s_)
check("(g)/(d) setup: d_t log I_D(T,t;0) == lam (F_{T-t}(0) - 1)",
      sp.simplify(sp.diff(sp.log(IDg(Tg, tg)), tg) - lam * (F0g(Tg - tg) - 1)))
RHSg = sp.exp(-(mu_g + thg) * tg) * IDg(Tg, tg) * F0g(Tg - tg)
check("(g) semigroup: d_t[e^{-(mu+th)t} I_D(T,t;0) F_{T-t}(0)] == -mu w e^{-(mu+th)t} I_D(T,t;0)",
      sp.simplify(sp.diff(RHSg, tg) + lam * p * m * sp.exp(-(mu_g + thg) * tg) * IDg(Tg, tg)))
check("(g) semigroup: right-hand side vanishes at t = T", sp.simplify(RHSg.subs(tg, Tg)))
check("(d) additivity: I_D(T,t;0) e^{-th t} W(T-t) == W(T)",
      sp.simplify(IDg(Tg, tg) * sp.exp(-thg * tg) * Wg(Tg - tg) - Wg(Tg)))

print("=" * 78)
print("6. Remark: lam <-> mu symmetry of all zone coefficients")
print("=" * 78)
SW = {lam: mu, mu: lam, eta: al, al: eta}
for k in ell:
    check(f"I_{k}(lam,mu) == I_{k}(mu,lam)", I[k] - I[k].subs(SW, simultaneous=True))
    check(f"Z_{k}(lam,mu) == Z_{k}(mu,lam)", Z[k] - Z[k].subs(SW, simultaneous=True))

print("=" * 78)
print(f"SUMMARY: {'ALL CHECKS PASSED' if not FAIL else 'FAILED: ' + str(FAIL)}   [{time.time()-T0:.0f}s]")
