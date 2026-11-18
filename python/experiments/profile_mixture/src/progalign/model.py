"""Substitution (profile mixture over LG exchangeabilities) and indel (MixFrag) models."""
import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tkfmixdom.jax.core.params import mixfrag_trans
from tkfmixdom.jax.core.protein import lg_exchangeability
from tkfmixdom.jax.core.site_class_profiles import le_gascuel_c20

from . import wavefront as wf

A = 20
PARAMS_DIR = Path(__file__).resolve().parents[4] / "params" / "best"   # tkf-mixdom/python/params/best


@dataclass(frozen=True)
class Subst:
    """K reversible chains Q^k_xy = r_k S_xy pi^k_y sharing exchangeabilities S."""
    S: jnp.ndarray       # (A, A), zero diagonal
    P: jnp.ndarray       # (K, A) profiles pi^k
    w: jnp.ndarray       # (K,) class weights
    R: jnp.ndarray       # (K,) class rates (each class has mean rate 1)
    lam: jnp.ndarray     # (K, A) eigenvalues of B^k = Pi^{1/2} Q^k Pi^{-1/2}
    Psi: jnp.ndarray     # (K, A, A) orthonormal eigenvectors of B^k
    name: str = ""

    @property
    def K(self):
        return self.P.shape[0]

    @property
    def G(self):          # (K, A): (S pi^k)_x
        return self.P @ self.S

    @property
    def pibar(self):
        return self.w @ self.P

    @property
    def B(self):          # (K, A, A) symmetrized rate matrices
        sq = jnp.sqrt(self.P)
        Bk = self.R[:, None, None] * sq[:, :, None] * self.S[None] * sq[:, None, :]
        return Bk.at[:, jnp.arange(A), jnp.arange(A)].set(-self.R[:, None] * self.G)

    def M(self, t):
        """(K, A, A) transition matrices e^{t Q^k}."""
        sq = jnp.sqrt(self.P)
        core = jnp.einsum("kxi,ki,kyi->kxy", self.Psi, jnp.exp(self.lam * t), self.Psi)
        return core * (sq[:, None, :] / sq[:, :, None])

    def joint(self, t):
        """(A, A) class-marginal joint J_ab(t) = sum_k w_k pi^k_a M^k_ab(t)."""
        return jnp.einsum("k,ka,kab->ab", self.w, self.P, self.M(t))


def make_subst(kind="C20"):
    """kind: LG | C20 (each class mean rate 1) | C20g (mixture mean rate 1)."""
    S, pi_lg = lg_exchangeability()
    S = np.array(S, float)
    np.fill_diagonal(S, 0.0)
    if kind == "LG":
        P, w = np.asarray(pi_lg, float)[None], np.ones(1)
    elif kind in ("C20", "C20g"):
        P, w, _ = le_gascuel_c20()
        P, w = np.asarray(P, float), np.asarray(w, float)
    else:
        raise ValueError(kind)
    P = P / P.sum(1, keepdims=True)
    w = w / w.sum()
    rates = np.einsum("kx,xy,ky->k", P, S, P)       # mean rate of each class with r_k = 1
    if kind == "C20g":
        # one global scale: the mixture has mean rate 1, so concentrated profiles evolve slowly
        R = np.full(len(w), 1.0 / float(w @ rates))
    else:
        # each class normalized to mean rate 1 (IQ-TREE-style per-class normalization)
        R = 1.0 / rates
    sq = np.sqrt(P)
    B = R[:, None, None] * sq[:, :, None] * S[None] * sq[:, None, :]
    B[:, np.arange(A), np.arange(A)] = -R[:, None] * (P @ S)
    lam, Psi = np.linalg.eigh(B)
    return Subst(jnp.asarray(S), jnp.asarray(P), jnp.asarray(w), jnp.asarray(R),
                 jnp.asarray(lam), jnp.asarray(Psi), name=kind)


@dataclass(frozen=True)
class Indel:
    """MixFrag: TKF91 links with rates lam < mu; fragtype f has extension r_f, weight w_f."""
    lam: float
    mu: float
    r: tuple
    w: tuple
    name: str = ""

    @property
    def F(self):
        return len(self.r)

    @property
    def state_types(self):
        F = self.F
        return jnp.array([wf.S] + [wf.M] * F + [wf.I] * F + [wf.D] * F + [wf.E])

    def trans(self, t):
        return mixfrag_trans(self.lam, self.mu, t, jnp.asarray(self.r), jnp.asarray(self.w))

    def log_trans(self, t):
        tau = self.trans(t)
        return jnp.where(tau > 0, jnp.log(jnp.where(tau > 0, tau, 1.0)), wf.NEG_INF)


def load_indel(kind="mixfrag_F2"):
    """Pfam-trained parameters of paper 1 (summarized-count EM, LG08 substitution)."""
    fname = {"mixfrag_F2": "cem_mixfrag_F2_pfamTrain.json", "tkf92": "cem_tkf92_pfamTrain.json"}[kind]
    d = json.loads((PARAMS_DIR / fname).read_text())
    return Indel(float(d["lam"]), float(d["mu"]), tuple(float(x) for x in d["exts"]),
                 tuple(float(x) for x in d["weights"]), name=kind)
