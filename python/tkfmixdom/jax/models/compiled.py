"""Concrete CompiledModel implementations for TKF91, TKF92, MixDom, and TKFST.

Each implementation wraps existing model-building and training logic behind
the CompiledModel interface, enabling generic EM/Adam training loops.

Configuration:
    fixed_params: set[str] — parameter names to hold fixed during M-step.

All models use the joint quadratic M-step (eq:kappa-quadratic in tkf.tex)
which includes L and M counts and enforces κ = λ/μ < 1. BDI observation
time T is accumulated in BDISuffStats.T across pairs via _add_stats.
        e.g. {'Q', 'pi', 't'} to hold substitution model and time fixed.

MAP-EM monotonicity: The BDI M-step guarantees the PENALIZED log-likelihood
(ℓ + log Gamma prior) is monotone. Raw ℓ can decrease with non-flat priors.
For exact ML-EM without priors, use _maximize_q_rates() below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ..core.types import (
    GrammarClass, NullInfo,
    RateParam, SimplexParam, BernoulliParam,
    BDISuffStats, SimplexSuffStats, BernoulliSuffStats,
)
from ..core.params import S, M, I, D, E, tkf91_trans

from ..dp.hmm import forward_backward_2d, safe_log
from ..models.left_regular import make_tkf91_pair_hmm, make_tkf92_pair_hmm


def init_mixdom_params(init_ins, init_del, n_dom, n_frag, seed=None):
    """Initialize MixDom parameters with random perturbation.

    Random perturbation breaks symmetry between domains and prevents
    collapse to a single dominant domain.
    """
    init_ins = max(init_ins, 0.001)
    init_del = max(init_del, 0.002)

    rng = np.random.RandomState(seed)

    log_noise_ins = rng.randn(n_dom) * 0.5
    log_noise_del = rng.randn(n_dom) * 0.5
    dom_ins = np.array([init_ins * np.exp(log_noise_ins[d]) for d in range(n_dom)])
    dom_del = np.array([max(init_del * np.exp(log_noise_del[d]),
                            dom_ins[d] * 1.5) for d in range(n_dom)])
    dom_ins = np.clip(dom_ins, 1e-5, 7.0)
    dom_del = np.clip(dom_del, 1e-5, 7.0)

    dom_weights = rng.dirichlet(np.ones(n_dom) * 5.0)
    frag_weights = np.array([rng.dirichlet(np.ones(n_frag) * 5.0)
                             for _ in range(n_dom)])
    # MixDom2: ext_rates is (D, F, F) - initialize as diagonal matrices
    # with random extension rates on the diagonal
    ext_diag = rng.uniform(0.15, 0.45, size=(n_dom, n_frag))
    ext_rates = np.zeros((n_dom, n_frag, n_frag))
    for d in range(n_dom):
        ext_rates[d] = np.diag(ext_diag[d])

    return init_ins, init_del, dom_ins, dom_del, dom_weights, frag_weights, ext_rates


# ---------------------------------------------------------------------------
# TKF91
# ---------------------------------------------------------------------------

@dataclass
class TKF91Model:
    """CompiledModel for TKF91 Pair HMM.

    No null states, no null cycles — the simplest case.

    Params dict:
        ins_rate: float
        del_rate: float
        t: float
        Q: (A, A) rate matrix
        pi: (A,) equilibrium distribution

    Aligned pytrees (extract_stats ↔ m_step):
        counts = {'indel': BDISuffStats(E_B, E_D, E_S, n_kappa, n_1mkappa)}
        priors = {'ins': RateParam, 'del': RateParam}
    """
    grammar_class: GrammarClass = GrammarClass.LEFT_REGULAR
    null_info: NullInfo = field(default_factory=NullInfo)
    ins_prior: tuple = (2.0, 10.0)   # Gamma(shape, rate)
    del_prior: tuple = (2.0, 10.0)
    fixed_params: frozenset = frozenset({'Q', 'pi', 't'})  # params to hold fixed

    def build_trans(self, params: dict) -> tuple[jax.Array, jax.Array]:
        """Build TKF91 transition matrix and state types."""
        log_trans, state_types, sub_matrix, pi = make_tkf91_pair_hmm(
            params['ins_rate'], params['del_rate'], params['t'],
            params['Q'], params['pi'])
        return jnp.exp(log_trans), state_types

    def log_trans_and_emissions(self, params: dict) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Build log-transition matrix, state types, substitution matrix, pi."""
        return make_tkf91_pair_hmm(
            params['ins_rate'], params['del_rate'], params['t'],
            params['Q'], params['pi'])

    def e_step(self, params: dict, x: jax.Array, y: jax.Array
               ) -> tuple[float, jax.Array, jax.Array]:
        """Run Forward-Backward, return (log_prob, n_trans, posteriors)."""
        log_trans, state_types, sub_matrix, pi = self.log_trans_and_emissions(params)
        log_prob, posteriors, n_trans = forward_backward_2d(
            log_trans, state_types, x, y, sub_matrix, pi)
        return float(log_prob), n_trans, posteriors

    def extract_stats(self, n_trans: jax.Array, params: dict) -> dict:
        """Extract BDI sufficient statistics from transition counts.

        Returns aligned counts pytree:
            {'indel': BDISuffStats, 'n_trans': array}

        Always conditions on ancestor length for the BDI decomposition
        (E_B, E_D, E_S). The ancestor length counts (n_kappa, n_1mkappa)
        and observation time T are populated for the quadratic M-step.
        """
        from ..core.bdi import bdi_stats_from_counts, check_count_sanity
        check_count_sanity(n_trans, label="TKF91 extract_stats")

        ins_rate = params['ins_rate']
        del_rate = params['del_rate']
        t = params['t']

        # BDI stats directly from FB counts (eq:exposure-tkf -- eq:deaths-tkf)
        E_B, E_D, E_S = bdi_stats_from_counts(n_trans, ins_rate, del_rate, t)

        # L = n_kappa (ancestor link count), M = n_{1-kappa} (sec:bw-tkf91)
        n_kap = float(jnp.sum(n_trans[:, M]) + jnp.sum(n_trans[:, D]))
        n_1mkap = float(jnp.sum(n_trans[:, E]))

        # T = t × M (one BDI process per pair, M = n_{1-κ} terminations)
        T_pair = t * max(n_1mkap, 1e-15)

        return {
            'indel': BDISuffStats(E_B=float(E_B), E_D=float(E_D),
                                  E_S=float(E_S),
                                  n_kappa=n_kap, n_1mkappa=n_1mkap,
                                  T=T_pair),
            'n_trans': n_trans,
        }

    def m_step(self, stats: dict, params: dict) -> dict:
        """MAP update from BDI sufficient statistics.

        Uses the joint quadratic M-step (eq:kappa-quadratic) which includes
        L and M counts and enforces κ = λ/μ < 1. T is accumulated in
        BDISuffStats across pairs.
        """
        new_params = dict(params)

        if 'ins_rate' not in self.fixed_params:
            bdi = stats['indel']
            ins_param = RateParam(params['ins_rate'], *self.ins_prior)
            del_param = RateParam(params['del_rate'], *self.del_prior)
            new_ins, new_del = bdi.map_update(ins_param, del_param)
            new_ins = float(new_ins) if np.isfinite(new_ins) \
                else float(params['ins_rate'])
            new_del = float(new_del) if np.isfinite(new_del) \
                else float(params['del_rate'])
            assert new_del > new_ins, \
                f"M-step produced ins={new_ins} >= del={new_del} (kappa >= 1)"
            new_params['ins_rate'] = new_ins
            new_params['del_rate'] = new_del

        return new_params


# ---------------------------------------------------------------------------
# TKF92
# ---------------------------------------------------------------------------

@dataclass
class TKF92Model:
    """CompiledModel for TKF92 Pair HMM.

    Adds fragment extension (BernoulliParam) to TKF91.
    Grammar elaboration: TKF91 → 8-state with null boundary states
    (M_end, I_end, D_end) → null elimination → TKF92 5×5.

    Params dict:
        ins_rate, del_rate, t, ext: float
        Q: (A, A) rate matrix
        pi: (A,) equilibrium distribution

    Aligned pytrees:
        counts = {'indel': BDISuffStats, 'ext': BernoulliSuffStats}

    Substitution parameter estimation:
        When 'Q' and/or 'pi' are NOT in fixed_params, the M-step will update
        them using the GTR coordinate-ascent M-step (m_step_subst_option1).
        The caller must provide 'match_pairs' (A,A), 'insert_chars' (A,),
        and 'delete_chars' (A,) arrays in the stats dict. These are NOT
        produced by extract_stats (which only sees transition counts), but
        must be accumulated externally from the alignment character data.
        See train.constrained.observed_counts for the accumulation pattern.
    """
    grammar_class: GrammarClass = GrammarClass.LEFT_REGULAR
    null_info: NullInfo = field(default_factory=NullInfo)
    ins_prior: tuple = (2.0, 10.0)   # Gamma(shape, rate)
    del_prior: tuple = (2.0, 10.0)
    ext_prior: tuple = (1.0, 1.0)    # Beta(a, b), uniform by default
    fixed_params: frozenset = frozenset({'Q', 'pi', 't'})
    # Substitution prior pseudocount weights (for m_step_subst_option1)
    pi_pseudo: float = 1.0           # Dirichlet concentration for pi prior
    S_pseudo: float = 0.1            # pseudocount weight for S exchangeability prior

    def build_trans(self, params: dict) -> tuple[jax.Array, jax.Array]:
        log_trans, state_types, sub_matrix, pi = make_tkf92_pair_hmm(
            params['ins_rate'], params['del_rate'], params['t'],
            params['ext'], params['Q'], params['pi'])
        return jnp.exp(log_trans), state_types

    def log_trans_and_emissions(self, params: dict):
        return make_tkf92_pair_hmm(
            params['ins_rate'], params['del_rate'], params['t'],
            params['ext'], params['Q'], params['pi'])

    def e_step(self, params: dict, x: jax.Array, y: jax.Array
               ) -> tuple[float, jax.Array, jax.Array]:
        log_trans, state_types, sub_matrix, pi = self.log_trans_and_emissions(params)
        log_prob, posteriors, n_trans = forward_backward_2d(
            log_trans, state_types, x, y, sub_matrix, pi)
        return float(log_prob), n_trans, posteriors

    def extract_stats(self, n_trans: jax.Array, params: dict) -> dict:
        """Extract BDI + extension sufficient statistics.

        Decomposes TKF92 5×5 counts into TKF91 component (after removing
        fragment extension self-loops) and extension component.

        Returns: {'indel': BDISuffStats, 'ext': BernoulliSuffStats, 'n_trans': array}
        """
        from ..core.bdi import check_count_sanity
        check_count_sanity(n_trans, label="TKF92 extract_stats")

        ins_rate = params['ins_rate']
        del_rate = params['del_rate']
        t = params['t']
        ext = params['ext']

        # Decompose TKF92 counts: separate fragment-extension self-loops
        # from TKF91 self-loops. For each state s ∈ {M,I,D}:
        #   τ92[s,s] = ext + (1-ext)·τ91[s,s]
        # So the TKF91 portion of n[s,s] is n[s,s] · (1-ext)·τ91[s,s] / τ92[s,s]
        #
        # In the elaborated 8-state model, each visit to state s produces
        # ONE (ext, 1-ext) choice: extend fragment or exit. Therefore:
        #   n(ext at s) = n[s,s] × ext / τ92[s,s]      (fragment self-loop)
        #   n(exit at s) = n_s − n(ext at s)             (fragment exit)
        # The exit count includes BOTH off-diagonal transitions AND the
        # TKF91-self-loop portion of n[s,s] (exit → re-enter same state).
        tau91 = tkf91_trans(ins_rate, del_rate, t)
        n91_trans = np.array(n_trans)
        ext_self_total = 0.0
        ext_exit_total = 0.0

        for s_idx in [M, I, D]:
            p_s = float(tau91[s_idx, s_idx])
            tau92_ss = ext + (1.0 - ext) * p_s
            n_ss = float(n_trans[s_idx, s_idx])
            n_s = float(n_trans[s_idx].sum())

            if tau92_ss > 1e-30:
                frac_ext = ext / tau92_ss
                ext_self_s = n_ss * frac_ext
                ext_self_total += ext_self_s
                ext_exit_total += n_s - ext_self_s
                n91_trans[s_idx, s_idx] = n_ss * (1.0 - frac_ext)
            else:
                ext_exit_total += n_s

        # BDI stats from resolved TKF91 counts (eq:exposure-tkf -- eq:deaths-tkf)
        # Note: L counts fragments (links), not residues (sec:bw-tkf92)
        from ..core.bdi import bdi_stats_from_counts
        n91_jax = jnp.array(n91_trans)
        E_B, E_D, E_S = bdi_stats_from_counts(n91_jax, ins_rate, del_rate, t)

        n_kap = float(jnp.sum(n91_jax[:, M]) + jnp.sum(n91_jax[:, D]))
        n_1mkap = float(jnp.sum(n91_jax[:, E]))

        # T = t × M (one BDI process per pair)
        T_pair = t * max(n_1mkap, 1e-15)

        return {
            'indel': BDISuffStats(E_B=float(E_B), E_D=float(E_D),
                                  E_S=float(E_S),
                                  n_kappa=n_kap, n_1mkappa=n_1mkap,
                                  T=T_pair),
            'ext': BernoulliSuffStats(n_success=ext_self_total,
                                      n_failure=ext_exit_total),
            'n_trans': n_trans,
        }

    def m_step(self, stats: dict, params: dict) -> dict:
        """MAP update from BDI + extension + substitution sufficient statistics.

        Uses the joint quadratic M-step which enforces κ < 1.
        T is accumulated in BDISuffStats across pairs.

        If 'Q'/'pi' are not in fixed_params AND 'match_pairs' is present in
        stats, updates Q and pi via the GTR coordinate-ascent M-step
        (m_step_subst_option1) using Holmes-Rubin CTMC sufficient statistics.
        """
        new_params = dict(params)

        if 'ins_rate' not in self.fixed_params:
            bdi = stats['indel']
            ins_param = RateParam(params['ins_rate'], *self.ins_prior)
            del_param = RateParam(params['del_rate'], *self.del_prior)
            new_ins, new_del = bdi.map_update(ins_param, del_param)
            new_ins = float(new_ins) if np.isfinite(new_ins) \
                else float(params['ins_rate'])
            new_del = float(new_del) if np.isfinite(new_del) \
                else float(params['del_rate'])
            assert new_del > new_ins, \
                f"M-step produced ins={new_ins} >= del={new_del} (kappa >= 1)"
            new_params['ins_rate'] = new_ins
            new_params['del_rate'] = new_del

        if 'ext' not in self.fixed_params:
            ext_stats = stats['ext']
            ext_param = BernoulliParam(params['ext'], *self.ext_prior)
            new_ext = float(ext_stats.map_update(ext_param))
            new_params['ext'] = new_ext

        # Substitution parameter update (Q, pi) via Holmes-Rubin + GTR M-step
        if ('Q' not in self.fixed_params or 'pi' not in self.fixed_params) \
                and 'match_pairs' in stats:
            new_params = self._m_step_substitution(stats, new_params)

        return new_params

    def _m_step_substitution(self, stats: dict, params: dict) -> dict:
        """Update Q and pi from character-level sufficient statistics.

        Requires stats to contain:
            'match_pairs': (A, A) array of weighted (anc, desc) pair counts
            'insert_chars': (A,) array of insert character counts
            'delete_chars': (A,) array of delete (ancestor) character counts

        Uses Holmes-Rubin expected dwell times W and transition counts U
        accumulated over all match pairs, then applies the GTR coordinate-
        ascent M-step (m_step_subst_option1) with weak LG08 priors.
        """
        from ..core.ctmc import holmes_rubin_integrals, m_step_subst_option1

        match_pairs = np.asarray(stats['match_pairs'])
        A = match_pairs.shape[0]
        total_matches = float(match_pairs.sum())

        # Need enough data for a meaningful update
        if total_matches < A:
            return params

        Q = np.asarray(params['Q'])
        pi = np.asarray(params['pi'])
        t = float(params['t'])

        # Compute Holmes-Rubin integrals once for (Q, pi, t)
        I_hr, M_hr = holmes_rubin_integrals(jnp.array(Q), jnp.array(pi), t)
        I_hr = np.asarray(I_hr)
        M_hr = np.asarray(M_hr)

        # Accumulate W (dwell times) and U (transition counts) over match pairs
        W_total = np.zeros(A)
        U_total = np.zeros((A, A))

        for a in range(A):
            for b in range(A):
                weight = match_pairs[a, b]
                if weight < 1e-30:
                    continue
                M_ab = M_hr[a, b]
                if M_ab < 1e-30:
                    continue
                # w_hat_i = I[a,b,i,i] / M[a,b]
                w_hat = I_hr[a, b, np.arange(A), np.arange(A)] / M_ab
                # u_hat_ij = Q_ij * I[a,b,i,j] / M[a,b]
                u_hat = Q * I_hr[a, b] / M_ab
                np.fill_diagonal(u_hat, 0.0)

                W_total += weight * w_hat
                U_total += weight * u_hat

        # Composition counts V: ancestor + insert + delete characters
        insert_chars = np.asarray(stats.get('insert_chars', np.zeros(A)))
        delete_chars = np.asarray(stats.get('delete_chars', np.zeros(A)))
        # V_i = count of character i as ancestor in matches + inserts + deletes
        # Ancestor counts from match_pairs: sum over desc dimension
        anc_counts = match_pairs.sum(axis=1)  # (A,)
        V = anc_counts + insert_chars + delete_chars

        # Get LG08 prior (only for protein-sized alphabets)
        S_prior = None
        pi_prior = None
        if A == 20:
            try:
                from ..core.protein import lg_exchangeability
                S_prior, pi_prior = lg_exchangeability()
                S_prior = np.asarray(S_prior)
                pi_prior = np.asarray(pi_prior)
            except Exception:
                pass

        # GTR coordinate-ascent M-step
        S_new, pi_new, Q_new = m_step_subst_option1(
            W_total, U_total, V,
            S_prior=S_prior, pi_prior=pi_prior,
            pi_pseudo=self.pi_pseudo, S_pseudo=self.S_pseudo,
        )

        new_params = dict(params)
        if 'Q' not in self.fixed_params:
            new_params['Q'] = jnp.array(Q_new)
        if 'pi' not in self.fixed_params:
            new_params['pi'] = jnp.array(pi_new)

        return new_params


# ---------------------------------------------------------------------------
# MixDom
# ---------------------------------------------------------------------------

@dataclass
class MixDomModel:
    """CompiledModel for MixDom nested Pair HMM.

    Has null cycles from domain-level null states. The compile step
    removes these via (I - T_ZZ)^{-1} closure, and extract_stats
    restores the phantom counts with M_null mixture decomposition.

    Params dict:
        main_ins, main_del, t: float
        dom_ins, dom_del: (n_dom,) arrays
        dom_weights: (n_dom,) simplex
        frag_weights: (n_dom, n_frag) simplices
        ext_rates: (n_dom, n_frag) in (0, 1)
        Q: (A, A) rate matrix
        pi: (A,) equilibrium distribution

    Aligned pytrees (extract_stats → m_step):
        counts = {
            'top_indel': BDISuffStats,          # from top_counts_restored
            'dom_indel': [BDISuffStats, ...],    # per-domain from dom_counts
            'dom_weights': SimplexSuffStats,     # from dom_occupancy
            'frag_weights': [SimplexSuffStats],  # per-domain from frag_occupancy
            'ext': [[BernoulliSuffStats]],       # per (dom, frag)
        }
    """
    n_dom: int = 2
    n_frag: int = 2
    grammar_class: GrammarClass = GrammarClass.LEFT_REGULAR
    null_info: NullInfo = field(default_factory=NullInfo)
    ins_prior: tuple = (2.0, 10.0)
    del_prior: tuple = (2.0, 10.0)
    ext_prior: tuple = (2.0, 3.0)       # Beta(2,3), mode at 0.25
    dom_dirichlet: float = 1.5          # symmetric Dirichlet alpha per domain
    frag_dirichlet: float = 1.5         # symmetric Dirichlet alpha per fragment
    fixed_params: frozenset = frozenset({'Q', 'pi', 't'})  # params to hold fixed

    def build_trans(self, params: dict) -> tuple[jax.Array, jax.Array]:
        from ..models.mixdom import build_nested_trans, state_types as mixdom_state_types
        chi, _ = build_nested_trans(
            params['main_ins'], params['main_del'], params['t'],
            jnp.array(params['dom_ins']), jnp.array(params['dom_del']),
            jnp.array(params['dom_weights']),
            jnp.array(params['frag_weights']),
            jnp.array(params['ext_rates']))
        st = mixdom_state_types(self.n_dom, self.n_frag)
        return chi, st

    def e_step(self, params: dict, x: jax.Array, y: jax.Array
               ) -> tuple[float, jax.Array, jax.Array]:
        from ..models.mixdom import build_nested_trans, state_types as mixdom_state_types
        from ..core.ctmc import transition_matrix
        from ..dp.hmm import pair_hmm_emissions_per_domain
        chi, _ = build_nested_trans(
            params['main_ins'], params['main_del'], params['t'],
            jnp.array(params['dom_ins']), jnp.array(params['dom_del']),
            jnp.array(params['dom_weights']),
            jnp.array(params['frag_weights']),
            jnp.array(params['ext_rates']))
        st = mixdom_state_types(self.n_dom, self.n_frag)
        log_chi = safe_log(chi)

        # Build per-domain substitution matrices
        t = params['t']

        if 'S_exch' in params:
            from ..tree.fsa_anneal import build_per_domain_sub_matrices
            sub_matrices, pis = build_per_domain_sub_matrices(
                params, t, self.n_dom)
            log_emit = pair_hmm_emissions_per_domain(
                st, x, y, sub_matrices, pis, self.n_dom, self.n_frag)
            log_prob, posteriors, n_trans = forward_backward_2d(
                log_chi, st, x, y, None, None, log_emit_table=log_emit)
        else:
            # Legacy path: single Q + pi
            sub_matrix = transition_matrix(params['Q'], t)
            pi = params['pi']
            log_prob, posteriors, n_trans = forward_backward_2d(
                log_chi, st, x, y, sub_matrix, pi)

        return float(log_prob), n_trans, posteriors

    def extract_stats(self, n_trans: jax.Array, params: dict) -> dict:
        """Extract aligned sufficient statistics via exact chain restoration.

        Uses exact_suffstats (6-step chain restoration, verified to match
        autodiff on build_transnest to 1e-14 for all parameter groups).

        Returns counts pytree parallel to the parameter structure.
        """
        from ..core.bdi import check_count_sanity
        from ..models.exact_suffstats import exact_suffstats
        check_count_sanity(n_trans, label="MixDom extract_stats")
        ss = exact_suffstats(
            np.asarray(n_trans), params['main_ins'], params['main_del'],
            params['t'], params['dom_ins'], params['dom_del'],
            params['dom_weights'], params['frag_weights'],
            params['ext_rates'])
        return self._extract_stats_from_exact(ss, params)

    def _extract_stats_from_exact(self, ss, params):
        """Convert exact_suffstats output to the stats dict format expected by m_step.

        Uses bdi_stats_from_counts (eq:exposure-tkf -- eq:deaths-tkf) for BDI
        recovery. L and M from count matrix column sums (sec:bw-mixdom).
        T accumulated as T_pair × n_{1-κ} per domain (sec:bw-mixdom).
        """
        from ..core.bdi import bdi_stats_from_counts
        t = params['t']

        # Top-level BDI (eq:exposure-tkf -- eq:deaths-tkf)
        top_n = jnp.array(ss['top_5x5'])
        E_B, E_D, E_S = bdi_stats_from_counts(
            top_n, params['main_ins'], params['main_del'], t)
        # L = n_kappa (M+D column sums), M = n_{1-kappa} (E column sum)
        top_nk = float(jnp.sum(top_n[:, M]) + jnp.sum(top_n[:, D]))
        top_n1k = float(jnp.sum(top_n[:, E]))
        # T = t × M for top-level BDI
        top_T = t * max(top_n1k, 1e-15)
        top_bdi = BDISuffStats(E_B=float(E_B), E_D=float(E_D), E_S=float(E_S),
                               n_kappa=top_nk, n_1mkappa=top_n1k, T=top_T)

        # Per-domain BDI (eq:exposure-tkf -- eq:deaths-tkf with domain params)
        dom_bdi = []
        for d in range(self.n_dom):
            dom_n = jnp.array(ss['dom_M_5x5'][d])
            nk = ss['dom_kappa'][d]
            n1k = ss['dom_1mkappa'][d]
            if float(dom_n.sum()) < 0.01 and nk < 0.01:
                dom_bdi.append(BDISuffStats(E_B=0, E_D=0, E_S=0,
                                            n_kappa=nk, n_1mkappa=n1k, T=0.0))
                continue
            # T^(k) = t × (M-type entries + I/D-type entries)
            n_entries_M = float(jnp.sum(dom_n[S, :]))
            T_d = t * (n_entries_M + n1k)
            eb, ed, es = bdi_stats_from_counts(
                dom_n, params['dom_ins'][d], params['dom_del'][d], t, T=T_d)
            # L = ALL kappa events: M-type (M+D col sums) + I/D-type
            nk_M = float(jnp.sum(dom_n[:, M]) + jnp.sum(dom_n[:, D]))
            n1k_M = float(jnp.sum(dom_n[:, E]))
            dom_bdi.append(BDISuffStats(E_B=float(eb), E_D=float(ed), E_S=float(es),
                                        n_kappa=nk_M + nk, n_1mkappa=n1k_M + n1k,
                                        T=T_d))

        # Weights (sec:bw-mixdom, mixture selectors)
        dom_weights_stats = SimplexSuffStats(counts=jnp.array(ss['dom_w']))
        frag_weights_stats = [
            SimplexSuffStats(counts=jnp.array(ss['frag_w'][d]))
            for d in range(self.n_dom)]

        # Extension (sec:bw-mixdom, fragment transition/termination)
        # MixDom2: ext_counts is (D, F, F), term_counts is (D, F)
        # Each row f of the fragment transition matrix is a simplex over
        # {frag_0, ..., frag_{F-1}, terminate}. We store the counts for
        # row-normalization in the M-step.
        ext_stats = {
            'ext_counts': jnp.array(ss['ext']),    # (D, F, F)
            'term_counts': jnp.array(ss['term']),  # (D, F)
        }

        # T^(k) = t × n_{1-κ}^(k) (sec:bw-mixdom, T accumulation)
        dom_bdi_time = []
        for d in range(self.n_dom):
            n_1mkap_d = ss['dom_1mkappa'][d]
            # For M-type domains, n_{1-κ} comes from S-row sum of dom_M_5x5
            n_entries_M = float(jnp.sum(jnp.array(ss['dom_M_5x5'][d])[S, :]))
            dom_bdi_time.append(t * max(n_entries_M + n_1mkap_d, 1e-15))

        return {
            'top_indel': top_bdi,
            'dom_indel': dom_bdi,
            'dom_bdi_time': dom_bdi_time,
            'dom_weights': dom_weights_stats,
            'frag_weights': frag_weights_stats,
            'ext': ext_stats,
        }

    def m_step(self, stats: dict, params: dict) -> dict:
        """Exact MAP update from aligned sufficient statistics.

        Each parameter group updated independently from its own counts + prior.
        Parameters in self.fixed_params are not updated.
        """
        new_params = dict(params)
        t = params['t']

        # Top-level indel rates from BDI (eq:kappa-quadratic -- eq:insrate-root)
        if 'main_ins' not in self.fixed_params:
            from ..core.bdi import m_step_indel_quadratic
            top_bdi = stats['top_indel']
            # T is accumulated in BDISuffStats.T across pairs
            new_main_ins, new_main_del = m_step_indel_quadratic(
                top_bdi.E_B, top_bdi.E_D, top_bdi.E_S,
                L=top_bdi.n_kappa, M=top_bdi.n_1mkappa, T=top_bdi.T,
                prior_alpha_lam=self.ins_prior[0], prior_alpha_mu=self.del_prior[0],
                prior_beta=self.ins_prior[1])
            new_params['main_ins'] = float(new_main_ins) \
                if np.isfinite(new_main_ins) else float(params['main_ins'])
            new_params['main_del'] = float(new_main_del) \
                if np.isfinite(new_main_del) else float(params['main_del'])
            assert new_params['main_del'] > new_params['main_ins'], (
                f"top-level M-step produced ins={new_params['main_ins']} >= "
                f"del={new_params['main_del']} (kappa >= 1)")

        # Per-domain indel rates from BDI (eq:kappa-quadratic with domain params)
        if 'dom_ins' not in self.fixed_params:
            from ..core.bdi import m_step_indel_quadratic
            new_dom_ins = np.array(params['dom_ins'], dtype=float)
            new_dom_del = np.array(params['dom_del'], dtype=float)
            for d in range(self.n_dom):
                bdi_d = stats['dom_indel'][d]
                if (bdi_d.E_B == 0.0 and bdi_d.E_D == 0.0
                        and bdi_d.E_S == 0.0
                        and bdi_d.n_kappa == 0.0):
                    continue
                # T is accumulated in BDISuffStats.T across pairs
                ni, nd = m_step_indel_quadratic(
                    bdi_d.E_B, bdi_d.E_D, bdi_d.E_S,
                    L=bdi_d.n_kappa, M=bdi_d.n_1mkappa, T=bdi_d.T,
                    prior_alpha_lam=self.ins_prior[0], prior_alpha_mu=self.del_prior[0],
                    prior_beta=self.ins_prior[1])
                ni = float(ni) if np.isfinite(ni) else float(params['dom_ins'][d])
                nd = float(nd) if np.isfinite(nd) else float(params['dom_del'][d])
                assert nd > ni, (
                    f"per-domain M-step produced ins={ni} >= del={nd} "
                    f"(kappa >= 1) at d={d}")
                new_dom_ins[d] = ni
                new_dom_del[d] = nd
            new_params['dom_ins'] = new_dom_ins
            new_params['dom_del'] = new_dom_del

        # Domain weights from Dirichlet MAP
        if 'dom_weights' not in self.fixed_params:
            dom_w_param = SimplexParam(
                value=jnp.array(params['dom_weights']),
                prior_alpha=jnp.full(self.n_dom, self.dom_dirichlet))
            new_params['dom_weights'] = np.array(
                stats['dom_weights'].map_update(dom_w_param))

        # Fragment weights from Dirichlet MAP
        if 'frag_weights' not in self.fixed_params:
            new_frag_w = np.zeros((self.n_dom, self.n_frag))
            for d in range(self.n_dom):
                fw_param = SimplexParam(
                    value=jnp.array(params['frag_weights'][d]),
                    prior_alpha=jnp.full(self.n_frag, self.frag_dirichlet))
                new_frag_w[d] = np.array(
                    stats['frag_weights'][d].map_update(fw_param))
            new_params['frag_weights'] = new_frag_w

        # Extension rates: MixDom2 row-normalization with Dirichlet prior
        # Each row f of ext[d] is a simplex over {frag_0, ..., frag_{F-1}, terminate}
        if 'ext_rates' not in self.fixed_params:
            ext_counts = np.asarray(stats['ext']['ext_counts'])    # (D, F, F)
            term_counts = np.asarray(stats['ext']['term_counts'])  # (D, F)
            new_ext = np.zeros((self.n_dom, self.n_frag, self.n_frag))
            for d in range(self.n_dom):
                for f in range(self.n_frag):
                    # Counts for row f: [ext[d,f,0], ..., ext[d,f,F-1], term[d,f]]
                    row_counts = np.append(ext_counts[d, f, :], term_counts[d, f])
                    # Dirichlet MAP: add pseudocounts (ext_prior[0]-1 for frag transitions,
                    # ext_prior[1]-1 for termination)
                    pseudocounts = np.append(
                        np.full(self.n_frag, self.ext_prior[0] - 1.0),
                        self.ext_prior[1] - 1.0)
                    posterior = row_counts + pseudocounts
                    posterior = np.maximum(posterior, 0.0)
                    row_sum = posterior.sum()
                    if row_sum < 1e-15:
                        # No data: keep current params
                        new_ext[d, f, :] = np.asarray(params['ext_rates'])[d, f, :]
                    else:
                        normalized = posterior / row_sum
                        # Termination prob must remain > 0 (else degenerate
                        # geometric fragment-length distribution); raise loud
                        # rather than silently clip. Set ext_prior[1] > 1 to
                        # add a Dirichlet pseudocount on termination.
                        assert normalized[self.n_frag] > 1e-12, (
                            f"ext_rates M-step at (d={d}, f={f}) produced "
                            f"termination prob = {normalized[self.n_frag]:.3e} "
                            f"(ext_prior={self.ext_prior}); set "
                            f"ext_prior[1] > 1 or check input data.")
                        # ext[d,f,:] = first F entries (the frag transition probs)
                        new_ext[d, f, :] = normalized[:self.n_frag]
            new_params['ext_rates'] = new_ext

        return new_params


# ---------------------------------------------------------------------------
# TKFST (TKF Structure Tree)
# ---------------------------------------------------------------------------

@dataclass
class TKFSTModel:
    """CompiledModel for TKFST Structure Tree pair SCFG.

    Context-free grammar with stem and loop TKF processes. Uses
    alignment-constrained Inside-Outside for E-step.

    Unlike HMM models, TKFST uses SCFG-native parametrization where
    alpha, beta, gamma, kappa are free parameters (not derived from rates).
    The SCFG factors kappa separately from M/I/D transitions, which differs
    structurally from the TKF91 5x5 matrix.

    Params dict:
        alpha_S, beta_S, gamma_S, kappa_S: float  (stem TKF params)
        alpha_L, beta_L, gamma_L, kappa_L: float  (loop TKF params)
        p_bp, p_st, p_bu: float  (stem type simplex)
        p_lf, p_rf, p_sl: float  (loop link type simplex)
        ext_K: float  (stacked pair extension)
        ext_B: float  (bulge fragment extension)
        (optional: pi, pi_bp, pi_stack, sub_matrix, sub_bp, sub_stack)

    Aligned pytrees (extract_stats -> m_step):
        counts = {
            'stem_alpha': BernoulliSuffStats,
            'stem_beta': BernoulliSuffStats,
            'stem_gamma': BernoulliSuffStats,
            'stem_kappa': BernoulliSuffStats,
            'loop_alpha': BernoulliSuffStats,
            'loop_beta': BernoulliSuffStats,
            'loop_gamma': BernoulliSuffStats,
            'loop_kappa': BernoulliSuffStats,
            'stem_type': SimplexSuffStats,    # (p_bp, p_st, p_bu)
            'loop_type': SimplexSuffStats,    # (p_lf, p_rf, p_sl)
            'ext_K': BernoulliSuffStats,
            'ext_B': BernoulliSuffStats,
        }
    """
    grammar_class: GrammarClass = GrammarClass.CONTEXT_FREE
    null_info: NullInfo = field(default_factory=NullInfo)
    # Priors for Bernoulli params: Beta(a, b)
    tkf_prior: tuple = (1.0, 1.0)        # uniform prior for alpha/beta/gamma
    kappa_prior: tuple = (1.0, 1.0)       # uniform prior for kappa
    ext_prior: tuple = (1.0, 1.0)         # uniform prior for ext
    simplex_alpha: float = 1.0            # symmetric Dirichlet (uniform)
    fixed_params: frozenset = frozenset()

    def e_step(self, params: dict, x: jax.Array, y: jax.Array,
               alignment: list | None = None
               ) -> tuple[float, dict, None]:
        """Run Inside-Outside on fixed alignment, return factor counts.

        Args:
            params: parameter dict matching build_tkfst_pair_grammar kwargs
            x, y: integer nucleotide sequences
            alignment: list of (x_idx_or_None, y_idx_or_None) pairs.
                Required for TKFST.

        Returns:
            (log_prob, factor_counts_dict, None)
        """
        from ..models.context_free import (
            build_tkfst_pair_grammar, alignment_to_columns,
            tkfst_pair_inside_aligned, tkfst_pair_outside_aligned,
            tkfst_pair_expected_counts_aligned, _classify_rule_factors,
            _FACTOR_TAGS,
        )

        if alignment is None:
            raise ValueError("TKFSTModel.e_step requires alignment argument")

        col_types, col_x, col_y = alignment_to_columns(x, y, alignment)
        grammar = build_tkfst_pair_grammar(**params)

        log_prob, log_I = tkfst_pair_inside_aligned(
            grammar, col_types, col_x, col_y, return_table=True)
        if log_prob <= -1e29:
            return log_prob, None, None

        log_O = tkfst_pair_outside_aligned(
            grammar, col_types, col_x, col_y, log_I)
        counts = tkfst_pair_expected_counts_aligned(
            grammar, col_types, col_x, col_y, log_I, log_O)

        annotations = _classify_rule_factors(grammar)

        factor_counts = {}
        for tag in _FACTOR_TAGS:
            factor_counts[tag] = 0.0
        for pi_idx, (count, tags) in enumerate(zip(counts, annotations)):
            for tag in tags:
                if tag in factor_counts:
                    factor_counts[tag] += count

        return float(log_prob), factor_counts, None

    def extract_stats(self, factor_counts: dict, params: dict) -> dict:
        """Map factor counts to typed sufficient statistics.

        Args:
            factor_counts: dict of factor tag -> expected count
            params: current parameters (unused for TKFST)

        Returns:
            Typed pytree of BernoulliSuffStats and SimplexSuffStats.
        """
        if factor_counts is None:
            return None

        fc = factor_counts
        return {
            'stem_alpha': BernoulliSuffStats(
                n_success=fc['alpha_S'], n_failure=fc['1-alpha_S']),
            'stem_beta': BernoulliSuffStats(
                n_success=fc['beta_S'], n_failure=fc['1-beta_S']),
            'stem_gamma': BernoulliSuffStats(
                n_success=fc['gamma_S'], n_failure=fc['1-gamma_S']),
            'stem_kappa': BernoulliSuffStats(
                n_success=fc['kappa_S'], n_failure=fc['1-kappa_S']),
            'loop_alpha': BernoulliSuffStats(
                n_success=fc['alpha_L'], n_failure=fc['1-alpha_L']),
            'loop_beta': BernoulliSuffStats(
                n_success=fc['beta_L'], n_failure=fc['1-beta_L']),
            'loop_gamma': BernoulliSuffStats(
                n_success=fc['gamma_L'], n_failure=fc['1-gamma_L']),
            'loop_kappa': BernoulliSuffStats(
                n_success=fc['kappa_L'], n_failure=fc['1-kappa_L']),
            'stem_type': SimplexSuffStats(
                counts=jnp.array([fc['p_bp'], fc['p_st'], fc['p_bu']])),
            'loop_type': SimplexSuffStats(
                counts=jnp.array([fc['p_lf'], fc['p_rf'], fc['p_sl']])),
            'ext_K': BernoulliSuffStats(
                n_success=fc['ext_K'], n_failure=fc['1-ext_K']),
            'ext_B': BernoulliSuffStats(
                n_success=fc['ext_B'], n_failure=fc['1-ext_B']),
        }

    def m_step(self, stats: dict, params: dict) -> dict:
        """MAP update from typed sufficient statistics.

        Each parameter updated independently via conjugate map_update.
        """
        if stats is None:
            return params

        new_params = dict(params)
        _EPS = 1e-10

        # Bernoulli parameters: alpha, beta, gamma, kappa, ext
        _bernoulli_map = [
            ('stem_alpha', 'alpha_S', self.tkf_prior),
            ('stem_beta', 'beta_S', self.tkf_prior),
            ('stem_gamma', 'gamma_S', self.tkf_prior),
            ('stem_kappa', 'kappa_S', self.kappa_prior),
            ('loop_alpha', 'alpha_L', self.tkf_prior),
            ('loop_beta', 'beta_L', self.tkf_prior),
            ('loop_gamma', 'gamma_L', self.tkf_prior),
            ('loop_kappa', 'kappa_L', self.kappa_prior),
            ('ext_K', 'ext_K', self.ext_prior),
            ('ext_B', 'ext_B', self.ext_prior),
        ]
        for stat_key, param_key, prior in _bernoulli_map:
            if param_key not in self.fixed_params:
                bs = stats[stat_key]
                # Skip update if no data (keep original param)
                if bs.n_success + bs.n_failure < _EPS:
                    continue
                bp = BernoulliParam(params[param_key], *prior)
                new_params[param_key] = float(np.clip(
                    bs.map_update(bp), _EPS, 1.0 - _EPS))

        # Simplex parameters: stem type, loop link type
        if 'p_bp' not in self.fixed_params:
            st_counts = stats['stem_type'].counts
            if float(jnp.sum(st_counts)) > _EPS:
                sp = SimplexParam(
                    value=jnp.array([params['p_bp'], params['p_st'], params['p_bu']]),
                    prior_alpha=jnp.full(3, self.simplex_alpha))
                new_vals = np.array(stats['stem_type'].map_update(sp))
                new_params['p_bp'] = float(new_vals[0])
                new_params['p_st'] = float(new_vals[1])
                new_params['p_bu'] = float(new_vals[2])

        if 'p_lf' not in self.fixed_params:
            lt_counts = stats['loop_type'].counts
            if float(jnp.sum(lt_counts)) > _EPS:
                sp = SimplexParam(
                    value=jnp.array([params['p_lf'], params['p_rf'], params['p_sl']]),
                    prior_alpha=jnp.full(3, self.simplex_alpha))
                new_vals = np.array(stats['loop_type'].map_update(sp))
                new_params['p_lf'] = float(new_vals[0])
                new_params['p_rf'] = float(new_vals[1])
                new_params['p_sl'] = float(new_vals[2])

        return new_params
