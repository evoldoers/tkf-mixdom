"""
GGI -> TKF92 instantaneous KL-projection flow (the user's TriadHMM object).

We track an INHOMOGENEOUS TKF92 (lambda(t),mu(t),r(t)) that best approximates a
TRUE GGI(lambda0,mu0,x,y) process as a function of branch length t, by the
relative-entropy (m-projection) fit of the appendix.  The flow is

    theta(t+dt) = argmin_theta D( TKF92(theta(t),t) o GGI(dt)  ||  TKF92(theta,t+dt) )
                = M-step (argmax sum Nbar log tau') on the composite's expected
                  X->Z transition counts Nbar(t,dt).

The dt->0 limit is an ODE  dtheta/dt = beta(theta,t; GGI params).

This script establishes GROUND TRUTH for the flow's RHS two independent ways and
checks they agree, BEFORE we trust any closed form:

  (G) Gillespie:  simulate the true composite TKF92(theta,t) o GGI(u) for small
      finite u, M-step, finite-difference in u.
  (A) Generator:  for each leg-1 X->Y alignment sample, sum the GGI generator's
      single-event perturbations of the X->Z transition counts EXACTLY (no u
      noise), giving Ndot = d/du Nbar|_{u=0}; then dtheta/du via M-step Jacobian.

Validation gates:
  (0) u=0 self-consistency: M-step on leg-1 counts returns theta(t).
  (1) TKF91 fixed line: x=y=0 GGI with r=0 surrogate  =>  flow == 0 (exact
      semigroup, no running).
  (2) (A) vs (G): generator increment matches finite-u Gillespie.

Indels only (one-symbol alphabet), per the appendix.  Alignment columns are
M (X&Z), I (Z only / insert), D (X only / delete).  Composite ordering uses the
MB/TKF "insert-before-delete" convention in interior gaps; edge inserts are
O(1/length) and vanish for the equilibrium ancestor used here.
"""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# TKF92 pair HMM (5-state S,M,I,D,E), in (kappa, alpha, r) coordinates.
#   alpha = e^{-mu t},  kappa = lambda/mu.
# ----------------------------------------------------------------------------
S, M, I, D, E = 0, 1, 2, 3, 4

def beta_gamma(kap, alpha):
    # L'Hopital limit at kap = 1 (lambda = mu): beta = s/(1+s) where s = -log(alpha) = mu*t.
    # Derived by Taylor-expanding alpha^(1-kap) = 1 + (1-kap) log(alpha) + ... around kap=1
    # and noting both numerator and denominator are O(1-kap).
    if abs(1.0 - kap) < 1e-8:
        s = -np.log(alpha)
        beta = s / (1.0 + s)
        # gamma's formula 1 - beta/(kap (1-alpha)) is regular at kap=1
        gamma = 1.0 - beta / (1.0 - alpha)
        return beta, gamma
    a1 = alpha ** (1.0 - kap)
    beta = kap * (1.0 - a1) / (1.0 - kap * a1)
    gamma = 1.0 - beta / (kap * (1.0 - alpha))
    return beta, gamma

def tau5(kap, alpha, r):
    b, g = beta_gamma(kap, alpha)
    ob, og, ok = 1 - b, 1 - g, 1 - kap
    T = np.zeros((5, 5))
    T[S] = [0, ob * kap * alpha, b, ob * kap * (1 - alpha), ob * ok]
    for src, sc in [(M, M), (I, I)]:
        row = (1 - r) * np.array([0, ob*kap*alpha, b, ob*kap*(1-alpha), ob*ok])
        row[sc] += r
        T[src] = row
    rowd = (1 - r) * np.array([0, og*kap*alpha, g, og*kap*(1-alpha), og*ok])
    rowd[D] += r
    T[D] = rowd
    return T

def expected_counts5(kap, alpha, r):
    """Exact expected X->Y transition counts for ONE pair (one immortal link)."""
    T = tau5(kap, alpha, r)
    Q = T[np.ix_([M, I, D], [M, I, D])]
    p0 = T[S, [M, I, D]]
    v = p0 @ np.linalg.inv(np.eye(3) - Q)
    vis = {S: 1.0, M: v[0], I: v[1], D: v[2]}
    N = np.zeros((5, 5))
    for s in [S, M, I, D]:
        N[s] = vis[s] * T[s]
    return N

def ml_fit(Nbar, x0=(0.4, 0.5, 0.2)):
    """Joint ML / KL m-projection = argmax sum Nbar log tau' over (kappa, alpha, r)
    using the 5x5 TKF92 Pair HMM."""
    def nll(x):
        T = tau5(*x)
        with np.errstate(divide='ignore', invalid='ignore'):
            lt = np.where(T > 0, np.log(np.clip(T, 1e-300, None)), 0.0)
        return -np.sum(Nbar * lt)
    res = minimize(nll, x0, bounds=[(1e-5, 1 - 1e-7)] * 3, method='L-BFGS-B',
                   options=dict(ftol=1e-16, gtol=1e-14, maxiter=20000))
    return res.x

# Backward-compat alias (older scripts call kl_fit).
kl_fit = ml_fit

# ----------------------------------------------------------------------------
# 6x6 TKF92 WFST (conditional on ancestor).
# State order: S=0, M=1, I0=2, I1=3, D=4, E=5.
# I0: first-emit insertion (no ancestor character consumed before this run);
#     reachable from S, I0 only.
# I1: subsequent insertion (at least one ancestor character before this run);
#     reachable from M, I1, D.
# eq:tkf92-wfst in tkf/tkf92-wfst-derivation.tex.
# ----------------------------------------------------------------------------
S6, M6, I0_6, I1_6, D6, E6 = 0, 1, 2, 3, 4, 5

def tau6_wfst(kap, alpha, r):
    b, g = beta_gamma(kap, alpha)
    ob, og = 1 - b, 1 - g
    p = r + (1 - r) * kap                # singlet emit-state outgoing weight
    T = np.zeros((6, 6))
    # Row S: TKF91-like (immortal-link descendant fragment, no extension self-loop).
    T[S6, M6]   = ob * alpha
    T[S6, I0_6] = b
    T[S6, D6]   = ob * (1 - alpha)
    T[S6, E6]   = ob
    # Row M (emit-source: divisor p on ancestor exits; 1-beta on E).
    T[M6, M6]   = (r + (1 - r) * ob * kap * alpha) / p
    T[M6, I1_6] = (1 - r) * b
    T[M6, D6]   = (1 - r) * ob * kap * (1 - alpha) / p
    T[M6, E6]   = ob
    # Row I0 (S-source: divisor kappa absorbed; (1-r) factors from fragment exit).
    T[I0_6, M6]   = (1 - r) * ob * alpha
    T[I0_6, I0_6] = r + (1 - r) * b
    T[I0_6, D6]   = (1 - r) * ob * (1 - alpha)
    T[I0_6, E6]   = (1 - r) * ob
    # Row I1 (emit-source).
    T[I1_6, M6]   = (1 - r) * ob * kap * alpha / p
    T[I1_6, I1_6] = r + (1 - r) * b
    T[I1_6, D6]   = (1 - r) * ob * kap * (1 - alpha) / p
    T[I1_6, E6]   = ob
    # Row D (emit-source).
    T[D6, M6]   = (1 - r) * og * kap * alpha / p
    T[D6, I1_6] = (1 - r) * g
    T[D6, D6]   = (r + (1 - r) * og * kap * (1 - alpha)) / p
    T[D6, E6]   = og
    return T


def ml_fit_wfst(Nbar6, x0=(0.4, 0.5, 0.2)):
    """Conditional ML fit via the 6x6 TKF92 WFST.

    Args:
        Nbar6: 6x6 transition counts in {S, M, I0, I1, D, E} from a recoded
            SMIDE alignment (insertions before the first M/D become I0; insertions
            from the first M/D onwards become I1).
        x0: initial (kap, alpha, r) guess.

    Returns:
        (kap, alpha, r) maximising sum Nbar6 * log tau6_wfst.
    """
    def nll(x):
        T = tau6_wfst(*x)
        with np.errstate(divide='ignore', invalid='ignore'):
            lt = np.where(T > 0, np.log(np.clip(T, 1e-300, None)), 0.0)
        return -np.sum(Nbar6 * lt)
    res = minimize(nll, x0, bounds=[(1e-5, 1 - 1e-7)] * 3, method='L-BFGS-B',
                   options=dict(ftol=1e-16, gtol=1e-14, maxiter=20000))
    return res.x

# ----------------------------------------------------------------------------
# Leg-1 sampler: sample one X->Y alignment from the TKF92 pair HMM with
# X ~ equilibrium.  Returns the alignment as a list of column types in
# {M,I,D}, walking S -> ... -> E.  X~equilibrium is built into the pair HMM
# stationary alignment chain started from S.
# ----------------------------------------------------------------------------
def sample_alignment(kap, alpha, r, max_len=100000):
    T = tau5(kap, alpha, r)
    cols = []
    s = S
    while True:
        probs = T[s]
        s2 = rng.choice(5, p=probs / probs.sum())
        if s2 == E:
            break
        cols.append(s2)            # 1=M,2=I,3=D
        s = s2
        if len(cols) > max_len:
            raise RuntimeError("runaway alignment")
    return cols   # list over {M,I,D}

def count_transitions(cols):
    """Expected/observed transition-count matrix for one S->...->E column list."""
    N = np.zeros((5, 5))
    prev = S
    for c in cols:
        N[prev, c] += 1
        prev = c
    N[prev, E] += 1
    return N

# ----------------------------------------------------------------------------
# GGI generator acting on Y (the M/I columns of the leg-1 alignment).
#   per-gap insertion rate (k residues): lam0 x^{k-1}(1-x)   -> total per gap lam0
#   per-run deletion rate (k residues):  mu0  y^{k-1}(1-y)
# Apply a single event to the leg-1 column list, return the new X->Z column list.
# Convention: insert-before-delete in interior gaps.
# ----------------------------------------------------------------------------
def y_positions(cols):
    """Indices (into cols) of the Y residues (M or I columns), in order."""
    return [i for i, c in enumerate(cols) if c in (M, I)]

def apply_deletion(cols, ypos, a, b):
    """Delete Y-residues with Y-index a..b (inclusive). M->D, I->ghost(remove)."""
    out = []
    del_set = set(ypos[a:b+1])
    for i, c in enumerate(cols):
        if i in del_set:
            if c == M:
                out.append(D)        # match -> delete
            # I -> ghost: drop entirely
        else:
            out.append(c)
    return out

def apply_insertion(cols, ypos, g, k):
    """Insert k new I columns at Y-gap g (0..len(ypos)).  Insert-before-delete:
    place immediately AFTER the column of Y-residue g-1 (so before any D's that
    follow it in the gap); g=0 places at the front."""
    if g == 0:
        anchor = -1          # before everything
    else:
        anchor = ypos[g - 1]  # column index of the left Y neighbour
    out = []
    for i, c in enumerate(cols):
        out.append(c)
        if i == anchor:
            out.extend([I] * k)
    if anchor == -1:
        out = [I] * k + out
    return out

# ---- exact generator increment Ndot = d/du Nbar|_{u=0} for ONE leg-1 sample --
def ggi_Ndot_for_sample(cols, lam0, mu0, x, y, kmax=None):
    """Sum over all single GGI events e: rate(e) * (N(cols after e) - N(cols)).
    Returns the 5x5 increment matrix for this leg-1 alignment."""
    ypos = y_positions(cols)
    nY = len(ypos)
    N0 = count_transitions(cols)
    Ndot = np.zeros((5, 5))
    if kmax is None:                     # geometric tail cutoff (rates < 1e-13)
        mxy = max(x, y, 1e-6)
        kmax = min(600, max(2, int(np.log(1e-13) / np.log(mxy)) + 3))
    # geometric tails
    kk = np.arange(1, kmax + 1)
    ins_rate_k = lam0 * x ** (kk - 1) * (1 - x)      # rate of inserting exactly k
    del_rate_k = mu0 * y ** (kk - 1) * (1 - y)       # rate of deleting exactly k

    # --- insertions: each of (nY+1) gaps, each k ---
    for g in range(nY + 1):
        for kidx, k in enumerate(kk):
            rate = ins_rate_k[kidx]
            if rate < 1e-18:
                break
            newcols = apply_insertion(cols, ypos, g, int(k))
            Ndot += rate * (count_transitions(newcols) - N0)
    # --- deletions: each contiguous Y-run [a..a+k-1] within [0,nY) ---
    for a in range(nY):
        for kidx, k in enumerate(kk):
            b = a + int(k) - 1
            if b >= nY:
                break
            rate = del_rate_k[kidx]
            if rate < 1e-18:
                break
            newcols = apply_deletion(cols, ypos, a, b)
            Ndot += rate * (count_transitions(newcols) - N0)
    return Ndot, N0

# ----------------------------------------------------------------------------
# Coordinate conversions.  Surrogate is TKF92(lambda,mu,r) observed at time t;
# the pair-HMM coords are kappa=lambda/mu, alpha=e^{-mu t}.  The M-step returns
# (kappa',alpha',r') at TOTAL time t+u, so mu'=-ln(alpha')/(t+u), lambda'=kappa'mu'.
# ----------------------------------------------------------------------------
def lmr_to_kar(lam, mu, r, t):
    return lam / mu, np.exp(-mu * t), r

def kar_to_lmr(kap, alpha, r, t):
    mu = -np.log(alpha) / t
    return kap * mu, mu, r

# ----------------------------------------------------------------------------
# Generator-route flow in (lambda,mu,r):  Nbar(t,u) = N0 + u*Ndot, M-step on the
# perturbed counts at total time t+du, converted back to (lambda,mu,r).
# ----------------------------------------------------------------------------
def flow_generator(lam, mu, r, t, lam0, mu0, x, y, nsamp=4000, du=1e-4,
                   shared_cols=None):
    kap, alpha, _ = lmr_to_kar(lam, mu, r, t)
    N0_sum = np.zeros((5, 5)); Ndot_sum = np.zeros((5, 5))
    cols_list = shared_cols if shared_cols is not None else \
        [sample_alignment(kap, alpha, r) for _ in range(nsamp)]
    for cols in cols_list:
        Nd, N0 = ggi_Ndot_for_sample(cols, lam0, mu0, x, y)
        Ndot_sum += Nd; N0_sum += N0
    n = len(cols_list)
    N0bar = N0_sum / n; Ndot = Ndot_sum / n
    kar0 = kl_fit(N0bar, x0=(kap, alpha, r))
    karp = kl_fit(N0bar + du * Ndot, x0=tuple(kar0))
    # back to (lambda,mu,r): kar0 lives at time t, karp at time t+du
    lmr0 = np.array(kar_to_lmr(*kar0, t))
    lmrp = np.array(kar_to_lmr(*karp, t + du))
    dlmr = (lmrp - lmr0) / du
    return lmr0, dlmr, N0bar, Ndot

# ----------------------------------------------------------------------------
# Finite-u Gillespie of leg-2 GGI on Y, using the SAME apply_ functions (so the
# ordering convention matches route A exactly).  Cross-checks that route A's
# Ndot is the genuine d/du of the composite counts.
# ----------------------------------------------------------------------------
def _sample_del_run(L, y):
    """Sample (start i, length k) of a deletion run on length-L seq, weights
    mu0 y^{k-1}(1-y) (mu0 cancels).  P(i) ~ 1 - y^{L-i}; k|i truncated-geom."""
    w = 1.0 - y ** (L - np.arange(L))          # P(start=i) up to norm
    i = rng.choice(L, p=w / w.sum())
    m = L - i                                   # max run length
    # truncated geometric on 1..m: P(k) ~ y^{k-1}(1-y)
    pk = y ** (np.arange(1, m + 1) - 1) * (1 - y)
    k = 1 + rng.choice(m, p=pk / pk.sum())
    return i, k

def gillespie_leg2(cols0, lam0, mu0, x, y, u):
    cols = list(cols0)
    t = 0.0
    while True:
        ypos = y_positions(cols)
        L = len(ypos)
        ins_total = lam0 * (L + 1)
        # total deletion rate = mu0 * sum_i (1 - y^{L-i})
        del_total = mu0 * (L - y * (1 - y ** L) / (1 - y)) if L > 0 else 0.0
        R = ins_total + del_total
        if R <= 0:
            break
        t += rng.exponential(1.0 / R)
        if t > u:
            break
        if rng.random() < ins_total / R:
            g = rng.integers(0, L + 1)
            k = 1 + rng.geometric(1 - x) - 1   # geometric(1-x): P(k)=x^{k-1}(1-x)
            cols = apply_insertion(cols, ypos, int(g), int(k))
        else:
            i, k = _sample_del_run(L, y)
            cols = apply_deletion(cols, ypos, int(i), int(i + k - 1))
    return cols

def flow_gillespie(kap, alpha, r, lam0, mu0, x, y, nsamp=20000, u=0.02):
    """Finite-u composite counts via Gillespie; (Nbar(u)-Nbar(0))/u ~ Ndot."""
    N0_sum = np.zeros((5, 5)); Nu_sum = np.zeros((5, 5))
    for _ in range(nsamp):
        cols = sample_alignment(kap, alpha, r)
        N0_sum += count_transitions(cols)
        colsu = gillespie_leg2(cols, lam0, mu0, x, y, u)
        Nu_sum += count_transitions(colsu)
    N0bar = N0_sum / nsamp; Nubar = Nu_sum / nsamp
    Ndot_fd = (Nubar - N0bar) / u
    return N0bar, Nubar, Ndot_fd

# ============================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True, linewidth=140)
    LBL = ['S', 'M', 'I', 'D', 'E']

    print("=" * 76)
    print("GATE 0: u=0 self-consistency  (M-step on leg-1 counts returns theta)")
    print("=" * 76)
    for (kap, alpha, r) in [(0.5, 0.6, 0.3), (0.4, 0.7, 0.5), (0.6, 0.5, 0.1)]:
        Nx = expected_counts5(kap, alpha, r)
        fit = kl_fit(Nx, x0=(0.4, 0.5, 0.2))
        print(f"  exact  (k,a,r)=({kap},{alpha},{r}) -> fit={np.round(fit,5)} "
              f"err={np.max(np.abs(np.array([kap,alpha,r])-fit)):.1e}")
    # sampler self-consistency
    kap, alpha, r = 0.5, 0.6, 0.3
    Nsum = np.zeros((5, 5)); ns = 60000
    for _ in range(ns):
        Nsum += count_transitions(sample_alignment(kap, alpha, r))
    print(f"  sampled (k,a,r)=({kap},{alpha},{r}) -> fit={np.round(kl_fit(Nsum/ns),5)} "
          f"(MC, n={ns})   [exact N vs sampled N maxerr "
          f"{np.max(np.abs(Nsum/ns - expected_counts5(kap,alpha,r))):.1e}]")

    print("\n" + "=" * 76)
    print("GATE 1: TKF91 fixed line.  GGI x=y=0 (single-residue) == TKF91(lam0,mu0).")
    print("  Surrogate set EQUAL to GGI (lam=lam0, mu=mu0, r=0): composite is the")
    print("  exact TKF91 semigroup  =>  d(lam,mu,r)/dt == 0.")
    print("=" * 76)
    for (lam0, mu0, t) in [(0.5, 1.0, 0.5), (0.8, 1.2, 1.0), (0.3, 0.6, 0.3)]:
        lmr0, dlmr, N0, Nd = flow_generator(lam0, mu0, 0.0, t, lam0, mu0, 0.0, 0.0,
                                            nsamp=12000)
        print(f"  GGI=surrogate (lam,mu,r,t)=({lam0},{mu0},0,{t}): "
              f"d(lam,mu,r)/dt={np.round(dlmr,4)}  (expect ~0)")

    print("\n" + "=" * 76)
    print("GATE 2: generator route (A) vs finite-u Gillespie (G):  Ndot agree?")
    print("=" * 76)
    # GGI params (reversible): pick lam0,mu0,x,y with lam0 y(1-x)=mu0 x(1-y)
    x, y = 0.3, 0.35
    mu0 = 1.0
    lam0 = mu0 * x * (1 - y) / (y * (1 - x))   # reversibility
    print(f"  GGI: lam0={lam0:.4f} mu0={mu0} x={x} y={y}   (rev: lam0 y(1-x)={lam0*y*(1-x):.4f} = mu0 x(1-y)={mu0*x*(1-y):.4f})")
    kap, alpha, r = 0.5, 0.6, 0.3
    # route A  (paired samples shared with G)
    na = 60000
    shared = [sample_alignment(kap, alpha, r) for _ in range(na)]
    N0a_sum = np.zeros((5, 5)); Nd_sum = np.zeros((5, 5))
    for cols in shared:
        nd, n0 = ggi_Ndot_for_sample(cols, lam0, mu0, x, y)
        Nd_sum += nd; N0a_sum += n0
    Ndot_A = Nd_sum / na
    # route G: same leg-1 samples, finite u; (counts(u)-counts(0))/u -> Ndot
    for u in [0.02, 0.01, 0.005]:
        Nu_sum = np.zeros((5, 5)); N0_sum = np.zeros((5, 5))
        for cols in shared:
            N0_sum += count_transitions(cols)
            Nu_sum += count_transitions(gillespie_leg2(cols, lam0, mu0, x, y, u))
        Ndot_G = (Nu_sum - N0_sum) / na / u
        diff = np.max(np.abs(Ndot_A - Ndot_G))
        print(f"  u={u:.3f}: max|Ndot_A - Ndot_G| = {diff:.4f}")
    print("  Ndot_A (generator, exact in u) =")
    for s in range(4):
        print("    " + LBL[s] + ": " + "  ".join(f"{LBL[c]}:{Ndot_A[s,c]:+.4f}" for c in range(1,5)))
