"""Alignment-conditioned indel-history sampler for one TKF91 branch.

Implements Algorithm 1 of the MBE paper ("Closed-form Baum-Welch for
TKF-based evolutionary models", Section 2.4.2): stochastic traceback of the
time-indexed gravestone grammar

    L(t)   -> A   . D(t,t)      weight  exp(-mu t)
    L(t)   -> G_l . D(t,l)      density mu exp(-mu l) dl,         0 < l < t
    D(t,T) -> L(t-u) . D(t,u)   density lam exp(-lam (T-u)) du,   0 < u < T
    D(t,T) -> eps               weight  exp(-lam T)

conditioned on the alignment of the ancestral and descendant sequences.

Under TKF91 an ancestral link and all of its descendants occupy a
contiguous block of the descendant sequence, so the alignment fixes, for
each ancestral link, whether it survived (M) or was deleted (D) and the
number n of inserted residues that follow it. Blocks are independent. The
leftmost block belongs to the immortal link.

The traceback uses two families of coefficients (gravestones summed out,
i.e. w = 1):

    a_m(s)   = [z^m] f_s(z,1)         P(link with s remaining has exactly m
                                         alive descendants, itself included)
    d_n(t,T) = [z^n] I_D(t,T; z, 1)   P(children born in a window of length
                                         T have exactly n alive descendants)

Both generating functions are ratios of linear functions of z (paper
Eqs. 23-24), so their coefficients are geometric beyond the constant term.

Times are measured from the start of the branch. Pure numpy; the
one-dimensional densities over birth and death times are sampled by
inverting a trapezoid-rule CDF on a fine grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_GRID = 2049  # grid points for 1-D inverse-CDF sampling


# --------------------------------------------------------------------------
# Closed-form coefficients
# --------------------------------------------------------------------------

def _equal_rates(lam: float, mu: float) -> bool:
    return abs(lam - mu) <= 1e-12 * max(lam, mu)


def tkf91_beta(lam: float, mu: float, s):
    """TKF91 insertion-continuation probability beta(s)."""
    s = np.asarray(s, dtype=float)
    if _equal_rates(lam, mu):
        return lam * s / (1.0 + lam * s)
    e = np.exp((lam - mu) * s)
    return lam * (1.0 - e) / (mu - lam * e)


def link_coeffs(lam: float, mu: float, s, nmax: int) -> np.ndarray:
    """a_m(s) for m = 0..nmax.  Shape (..., nmax+1) for array s.

    Linear birth-death from one individual: P(0) = (mu/lam) beta and
    P(m) = (1 - P(0)) (1 - beta) beta^(m-1) for m >= 1.
    """
    s = np.asarray(s, dtype=float)
    b = tkf91_beta(lam, mu, s)[..., None]
    p0 = (mu / lam) * b
    m = np.arange(nmax + 1)
    out = (1.0 - p0) * (1.0 - b) * b ** np.maximum(m - 1, 0)
    out = np.where(m == 0, p0, out)
    return out


def _lf_coeffs(K, a, b, c, d, nmax):
    """Coefficients of K (a + b z) / (c + d z), broadcasting over arrays."""
    K, a, b, c, d = (np.asarray(x, dtype=float)[..., None] for x in (K, a, b, c, d))
    r = -d / c
    m = np.arange(nmax + 1)
    out = K * r ** np.maximum(m - 1, 0) * (a * r + b) / c
    return np.where(m == 0, K * a / c, out)


def window_coeffs(lam: float, mu: float, t: float, T, nmax: int) -> np.ndarray:
    """d_n(t,T) for n = 0..nmax: [z^n] I_D(t,T; z, w=1).  Shape (..., nmax+1)."""
    T = np.asarray(T, dtype=float)
    if _equal_rates(lam, mu):
        # I_D = (1 + lam (t-T) (1-z)) / (1 + lam t (1-z))
        return _lf_coeffs(1.0, 1.0 + lam * (t - T), -lam * (t - T),
                          1.0 + lam * t, -lam * t, nmax)
    zp, zm = max(1.0, mu / lam), min(1.0, mu / lam)
    delta = abs(lam - mu)
    E1 = np.exp(delta * (t - T))
    E2 = np.exp(delta * t)
    K = np.exp(lam * (zp - 1.0) * T)
    # (1 - rho_{t-T}) / (1 - rho_t), rho_s = (z - zp)/(z - zm) e^{delta s}
    #   = [(z - zm) - (z - zp) E1] / [(z - zm) - (z - zp) E2]
    return _lf_coeffs(K, zp * E1 - zm, 1.0 - E1, zp * E2 - zm, 1.0 - E2, nmax)


# --------------------------------------------------------------------------
# History representation
# --------------------------------------------------------------------------

@dataclass
class Link:
    birth: float                 # 0.0 for ancestral links
    death: float | None          # None if alive at the end of the branch
    ancestral: bool = False
    immortal: bool = False
    children: list["Link"] = field(default_factory=list)  # latest-born first

    @property
    def alive(self) -> bool:
        return self.death is None

    def descendants_in_order(self):
        """Yield this link's subtree in sequence order (self, then each
        child's subtree, children latest-born first)."""
        yield self
        for c in self.children:
            yield from c.descendants_in_order()


@dataclass
class BranchSummary:
    births: int           # inserted links, ghosts included (B)
    deaths: int           # deleted links, ancestral or inserted (D)
    exposure: float       # time-integrated sequence length (S), immortal excluded
    ghosts: int           # inserted links that died on the branch


def summarize(history: list[Link], t: float) -> BranchSummary:
    births = deaths = ghosts = 0
    exposure = 0.0
    for root in history:
        for x in root.descendants_in_order():
            if x.immortal:
                continue
            if not x.ancestral:
                births += 1
            if not x.alive:
                deaths += 1
                if not x.ancestral:
                    ghosts += 1
            end = t if x.alive else x.death
            exposure += end - x.birth
    return BranchSummary(births, deaths, exposure, ghosts)


# --------------------------------------------------------------------------
# Traceback
# --------------------------------------------------------------------------

def _sample_grid(rng, x, g):
    """Sample from density proportional to g on grid x (trapezoid CDF)."""
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x))])
    if cdf[-1] <= 0.0:
        raise ValueError("density has zero mass")
    return float(np.interp(rng.uniform(0.0, cdf[-1]), cdf, x))


class HistorySampler:
    def __init__(self, lam: float, mu: float, rng: np.random.Generator,
                 grid: int = _GRID):
        self.lam, self.mu, self.rng, self.grid = lam, mu, rng, grid

    # SampleD(t, T, n, tau0): children of a link born at tau0, alive for T,
    # with n alive descendants in total.  Returned latest-born first.
    def sample_window(self, t: float, T: float, n: int, tau0: float) -> list[Link]:
        lam, rng = self.lam, self.rng
        children: list[Link] = []
        while True:
            dn = window_coeffs(lam, self.mu, t, T, n)[n]
            if n == 0 and rng.uniform() < np.exp(-lam * T) / dn:
                return children
            u = np.linspace(0.0, T, self.grid)
            a = link_coeffs(lam, self.mu, t - u, n)               # (G, n+1): a_m(t-u)
            d = window_coeffs(lam, self.mu, t, u, n)              # (G, n+1): d_k(t,u)
            # joint weight over (u, m): lam e^{-lam(T-u)} a_m(t-u) d_{n-m}(t,u)
            w = lam * np.exp(-lam * (T - u))[:, None] * a * d[:, ::-1]
            mass = np.sum(0.5 * (w[1:] + w[:-1]) * np.diff(u)[:, None], axis=0)
            m = int(rng.choice(n + 1, p=mass / mass.sum()))
            uu = _sample_grid(rng, u, w[:, m])
            children.append(self.sample_link(t - uu, m, tau0 + uu))
            T, n = uu, n - m

    # SampleL(s, m, tau): a link born at tau with s remaining and m alive
    # descendants (itself included).
    def sample_link(self, s: float, m: int, tau: float) -> Link:
        lam, mu, rng = self.lam, self.mu, self.rng
        am = link_coeffs(lam, mu, s, m)[m]
        p_alive = 0.0
        if m >= 1:
            p_alive = np.exp(-mu * s) * window_coeffs(lam, mu, s, s, m - 1)[m - 1] / am
        if rng.uniform() < p_alive:
            node = Link(birth=tau, death=None)
            node.children = self.sample_window(s, s, m - 1, tau)
            return node
        ell = self._sample_death(s, m)
        node = Link(birth=tau, death=tau + ell)
        node.children = self.sample_window(s, ell, m, tau)
        return node

    def _sample_death(self, s: float, n: int) -> float:
        """Sample l in (0,s) with density prop. to mu e^{-mu l} d_n(s, l)."""
        x = np.linspace(0.0, s, self.grid)
        g = self.mu * np.exp(-self.mu * x) * window_coeffs(self.lam, self.mu, s, x, n)[:, n]
        return _sample_grid(self.rng, x, g)

    def sample_branch(self, blocks: list[tuple[str, int]], t: float) -> list[Link]:
        """SampleBranch.  `blocks` lists the alignment as (kind, n) pairs:
        the first is ('I', n0) for the immortal link's leading insertions,
        then ('M', n) or ('D', n) for each ancestral residue, n being the
        number of inserted residues that follow it."""
        kind0, n0 = blocks[0]
        assert kind0 == 'I'
        immortal = Link(birth=0.0, death=None, ancestral=True, immortal=True)
        immortal.children = self.sample_window(t, t, n0, 0.0)
        history = [immortal]
        for kind, n in blocks[1:]:
            if kind == 'M':
                node = Link(birth=0.0, death=None, ancestral=True)
                node.children = self.sample_window(t, t, n, 0.0)
            elif kind == 'D':
                ell = self._sample_death(t, n)
                node = Link(birth=0.0, death=ell, ancestral=True)
                node.children = self.sample_window(t, ell, n, 0.0)
            else:
                raise ValueError(kind)
            history.append(node)
        return history


# --------------------------------------------------------------------------
# Alignment helpers
# --------------------------------------------------------------------------

def blocks_from_path(path: str) -> list[tuple[str, int]]:
    """Convert a pair-HMM state path over {M, I, D} (no S/E) into blocks."""
    blocks: list[list] = [['I', 0]]
    for c in path:
        if c == 'I':
            blocks[-1][1] += 1
        elif c in 'MD':
            blocks.append([c, 0])
        else:
            raise ValueError(c)
    return [tuple(b) for b in blocks]


def path_log_likelihood(path: str, lam: float, mu: float, t: float) -> float:
    """log P(path | ancestor) under the conditionally-normalized TKF91
    transition matrix (paper Eq. 2), emissions excluded."""
    alpha = np.exp(-mu * t)
    beta = float(tkf91_beta(lam, mu, t))
    gamma = 1.0 - mu * beta / (lam * (1.0 - alpha))

    def row(prev):
        x = gamma if prev == 'D' else beta
        return {'M': (1 - x) * alpha, 'I': x, 'D': (1 - x) * (1 - alpha), 'E': 1 - x}

    lp, prev = 0.0, 'S'
    for c in path + 'E':
        lp += np.log(row(prev)[c])
        prev = c
    return lp


def expected_stats_from_path(path: str, lam: float, mu: float, t: float,
                             h: float = 1e-5) -> tuple[float, float, float]:
    """E[B], E[D], E[S] given the alignment path, from the score-function
    identity (paper Eq. 4) plus the conservation law E[B] - E[D] = j - i.
    Derivatives by central finite differences in log rates."""
    f = lambda la, m: path_log_likelihood(path, la, m, t)
    g_lam = (f(lam * np.exp(h), mu) - f(lam * np.exp(-h), mu)) / (2 * h)
    g_mu = (f(lam, mu * np.exp(h)) - f(lam, mu * np.exp(-h))) / (2 * h)
    i = sum(c in 'MD' for c in path)
    j = sum(c in 'MI' for c in path)
    ES = ((j - i) - lam * t - g_lam + g_mu) / (lam - mu)
    ED = g_mu + mu * ES
    EB = ED + (j - i)
    return EB, ED, ES
