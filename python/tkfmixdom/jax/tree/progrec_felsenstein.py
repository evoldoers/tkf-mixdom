"""Profile-based progressive reconstruction with Felsenstein optimization.

For TKF92 (or TKF91), substitutions are independent of indels. This means
ancestral amino acid states can be marginalized exactly at each column using
Felsenstein pruning. The alignment DP tracks only M/I/D states (3 alignment
states), carrying a 20-dimensional conditional likelihood vector at each
position instead of a discrete amino acid.

For MixDom, the full order-1 WFST is used where transitions depend on
previously emitted characters. At each DP cell, the context is tracked:
- Match: (a, b) for AA x AA = 400 context entries
- Insert: b for AA = 20 context entries
- Delete: a for AA = 20 context entries
The WFST transition tensors (p_mm, p_mi, etc.) encode joint transition+emission
probabilities. Profile CLs act as Felsenstein emission factors.

Key data structure: a "profile" is an (L, A) array of Felsenstein conditional
likelihoods, where profile[i, a] = P(all data below position i | character
at this position = a). For leaves, this is just one-hot encoding of the
observed sequence.

Match emission at (i,j):
    sum_a pi[a] * (P_left @ cond_left[i])[a] * (P_right @ cond_right[j])[a]

Insert emission at j:
    sum_a pi[a] * (P @ cond[j])[a]

Delete emission at i:
    sum_a pi[a] * (P @ cond[i])[a]
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..core.params import tkf91_trans, tkf92_trans, S, M, I, D, E
from ..core.ctmc import transition_matrix
from ..dp.hmm import _forward_2d_core, _pad_to_bin, _pad_seq, _emit_mask, NEG_INF, _find_e_idx, safe_log
from ..util.io import TreeNode


def seq_to_profile(seq, alphabet_size):
    """Convert an integer sequence to a one-hot profile.

    Args:
        seq: (L,) integer array of character indices
        alphabet_size: number of characters (e.g. 20 for amino acids)

    Returns:
        profile: (L, A) one-hot array
    """
    L = seq.shape[0]
    return jnp.eye(alphabet_size)[seq]  # (L, A)


def profile_emissions(state_types, profile_x, profile_y, sub_left, sub_right,
                      pi, is_root=True):
    """Precompute emission log-probabilities for profile-based pair HMM.

    Instead of using discrete sequences, uses Felsenstein conditional
    likelihood profiles. The substitution matrices P_left and P_right
    transform child profiles to parent-conditional likelihoods.

    At root (is_root=True), weights by equilibrium distribution pi.
    At intermediate nodes (is_root=False), weights uniformly (1/A) to
    avoid double-counting the equilibrium distribution (Bug 4 fix).

    Args:
        state_types: (ns,) state type codes
        profile_x: (Lx, A) left child conditional likelihoods
        profile_y: (Ly, A) right child conditional likelihoods
        sub_left: (A, A) substitution matrix for left branch: P[a,b] = P(b|a,t_left)
        sub_right: (A, A) substitution matrix for right branch
        pi: (A,) equilibrium distribution
        is_root: if True, weight by pi; if False, weight uniformly

    Returns:
        emit: (Lx+1, Ly+1, ns) log emission table
    """
    Lx = profile_x.shape[0]
    Ly = profile_y.shape[0]
    A = pi.shape[0]
    ns = state_types.shape[0]

    # At root, weight by equilibrium; at intermediate nodes, weight uniformly
    weights = pi if is_root else jnp.ones(A) / A

    # Transform profiles through substitution matrices:
    # left_contrib[i, a] = sum_b P(b|a, t_left) * profile_x[i, b]
    # = (sub_left @ profile_x^T)^T = profile_x @ sub_left^T
    # But sub_left[a,b] = P(desc_char=b | anc_char=a), so
    # left_contrib[i, a] = sum_b sub_left[a, b] * profile_x[i, b]
    left_contrib = profile_x @ sub_left.T  # (Lx, A): contrib[i,a] = sum_b P(b|a) * cond_x[i,b]
    right_contrib = profile_y @ sub_right.T  # (Ly, A)

    # Match emission at (i,j): sum_a w[a] * left_contrib[i,a] * right_contrib[j,a]
    weighted_left = weights[None, :] * left_contrib  # (Lx, A)
    match_emit = jnp.log(jnp.maximum(weighted_left @ right_contrib.T, 1e-300))  # (Lx, Ly)

    # Insert emission at j: sum_a w[a] * right_contrib[j,a]
    ins_emit = jnp.log(jnp.maximum(jnp.sum(weights[None, :] * right_contrib, axis=1), 1e-300))  # (Ly,)

    # Delete emission at i: sum_a w[a] * left_contrib[i,a]
    del_emit = jnp.log(jnp.maximum(jnp.sum(weighted_left, axis=1), 1e-300))  # (Lx,)

    # Pad to (Lx+1, Ly+1) grid: position 0 is dummy (before sequence start)
    # Use 0 log-emission for dummy positions
    match_full = jnp.zeros((Lx + 1, Ly + 1))
    match_full = match_full.at[1:, 1:].set(match_emit)

    ins_full = jnp.zeros(Ly + 1)
    ins_full = ins_full.at[1:].set(ins_emit)

    del_full = jnp.zeros(Lx + 1)
    del_full = del_full.at[1:].set(del_emit)

    # Build (Lx+1, Ly+1, ns) emission table
    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    emit = (is_M[None, None, :] * match_full[:, :, None] +
            is_I[None, None, :] * ins_full[None, :, None] +
            is_D[None, None, :] * del_full[:, None, None])

    is_emit = is_M | is_I | is_D
    emit = jnp.where(is_emit[None, None, :], emit, 0.0)

    return emit


def profile_emissions_mixdom(state_types, profile_x, profile_y,
                             P_left_domains, P_right_domains,
                             pis, domain_weights, is_root=True):
    """Compute emission log-probs using domain-mixture Felsenstein model.

    Instead of a single substitution matrix, uses a mixture over N domains,
    each with its own equilibrium distribution and substitution matrix.

    At root (is_root=True), weights by per-domain equilibrium pi_n.
    At intermediate nodes (is_root=False), weights uniformly (1/A) to
    avoid double-counting the equilibrium distribution (Bug 4 fix).

    Args:
        state_types: (ns,) state type codes
        profile_x: (Lx, A) left child conditional likelihoods
        profile_y: (Ly, A) right child conditional likelihoods
        P_left_domains: (N, A, A) per-domain substitution matrices for left branch
        P_right_domains: (N, A, A) per-domain substitution matrices for right branch
        pis: (N, A) per-domain equilibrium distributions
        domain_weights: (N,) domain weights v_n
        is_root: if True, weight by pi_n; if False, weight uniformly

    Returns:
        emit: (Lx+1, Ly+1, ns) log emission table
    """
    Lx = profile_x.shape[0]
    Ly = profile_y.shape[0]
    A = pis.shape[1]
    N = pis.shape[0]
    ns = state_types.shape[0]

    # Per-domain Felsenstein contributions:
    # left_contribs[n, i, a] = sum_b P_n[a,b] * profile_x[i,b]
    left_contribs = jnp.einsum('nab,ib->nia', P_left_domains, profile_x)   # (N, Lx, A)
    right_contribs = jnp.einsum('nab,jb->nja', P_right_domains, profile_y) # (N, Ly, A)

    # At root: w_n * pi_n(a); at intermediate: w_n * (1/A)
    if is_root:
        weights = domain_weights[:, None] * pis  # (N, A)
    else:
        weights = domain_weights[:, None] * jnp.ones((N, A)) / A  # (N, A)

    # Match: sum_n sum_a weights[n,a] * left_contribs[n,i,a] * right_contribs[n,j,a]
    weighted_left = weights[:, None, :] * left_contribs  # (N, Lx, A)
    match_emit = jnp.log(jnp.maximum(
        jnp.einsum('nia,nja->ij', weighted_left, right_contribs), 1e-300))  # (Lx, Ly)

    # Insert: sum_n sum_a weights[n,a] * right_contribs[n,j,a]
    ins_emit = jnp.log(jnp.maximum(
        jnp.einsum('na,nja->j', weights, right_contribs), 1e-300))  # (Ly,)

    # Delete: sum_n sum_a weights[n,a] * left_contribs[n,i,a]
    del_emit = jnp.log(jnp.maximum(
        jnp.einsum('na,nia->i', weights, left_contribs), 1e-300))  # (Lx,)

    # Pad to (Lx+1, Ly+1) grid
    match_full = jnp.zeros((Lx + 1, Ly + 1))
    match_full = match_full.at[1:, 1:].set(match_emit)

    ins_full = jnp.zeros(Ly + 1)
    ins_full = ins_full.at[1:].set(ins_emit)

    del_full = jnp.zeros(Lx + 1)
    del_full = del_full.at[1:].set(del_emit)

    is_M = (state_types == M)
    is_I = (state_types == I)
    is_D = (state_types == D)

    emit = (is_M[None, None, :] * match_full[:, :, None] +
            is_I[None, None, :] * ins_full[None, :, None] +
            is_D[None, None, :] * del_full[:, None, None])

    is_emit = is_M | is_I | is_D
    emit = jnp.where(is_emit[None, None, :], emit, 0.0)

    return emit


def effective_sub_matrix(P_domains, pis, domain_weights):
    """Compute effective domain-mixture substitution matrix.

    P_eff[a, b] = sum_n [v_n * pi_n(a) / pi_mix(a)] * P_n[a, b]

    The weights are the posterior P(domain=n | ancestor char=a), so each
    row of P_eff uses domain weights conditioned on the character.

    Args:
        P_domains: (N, A, A) per-domain substitution matrices
        pis: (N, A) per-domain equilibrium distributions
        domain_weights: (N,) domain weights

    Returns:
        P_eff: (A, A) effective substitution matrix
        pi_mix: (A,) mixture equilibrium distribution
    """
    pi_mix = jnp.sum(domain_weights[:, None] * pis, axis=0)  # (A,)
    # Posterior P(domain=n | char=a) = v_n * pi_n(a) / pi_mix(a)
    w_per_char = domain_weights[:, None] * pis / jnp.maximum(pi_mix[None, :], 1e-30)  # (N, A)
    P_eff = jnp.einsum('na,nab->ab', w_per_char, P_domains)  # (A, A)
    return P_eff, pi_mix


# ============================================================
# Order-1 WFST DP: context-dependent transitions
# ============================================================

def compute_wfst_log_tensors(params, n_classes, tau, precomp=None):
    """Distill MixDom to order-1 WFST and return log transition tensors.

    Uses WFST normalization: sync transitions (M/D consuming ancestor a')
    are normalized per (a, b, a') giving P(b', state' | a', state, a, b).
    Non-sync transitions (insert, end) are normalized per (a, b).

    The singlet transition log_singlet[a, a'] is included for use at
    sync steps in compose_intersect (applied ONCE per sync step, shared
    between left and right branches).

    Args:
        params: constrained MixDom parameter dict
        n_classes: number of gamma rate classes
        tau: total evolutionary distance for the pair
        precomp: optional precomputed eigendecompositions

    Returns:
        dict with log transition tensors, P_domains, and log_singlet
    """
    from ..distill.maraschino import distill_mixdom, normalize_freqs_wfst
    dist = distill_mixdom(params, tau, n_classes, precomp)
    probs = normalize_freqs_wfst(dist)

    sl = lambda x: np.log(np.maximum(np.asarray(x), 1e-300))

    return {
        'log_p_mm': sl(probs['p_mm']),  # (AA,AA,AA,AA)
        'log_p_mi': sl(probs['p_mi']),  # (AA,AA,AA)
        'log_p_md': sl(probs['p_md']),  # (AA,AA,AA)
        'log_p_me': sl(probs['p_me']),  # (AA,AA)
        'log_p_im': sl(probs['p_im']),  # (AA,AA,AA,AA)
        'log_p_ii': sl(probs['p_ii']),  # (AA,AA,AA)
        'log_p_id': sl(probs['p_id']),  # (AA,AA,AA)
        'log_p_ie': sl(probs['p_ie']),  # (AA,AA)
        'log_p_dm': sl(probs['p_dm']),  # (AA,AA,AA,AA)
        'log_p_dd': sl(probs['p_dd']),  # (AA,AA,AA)
        'log_p_di': sl(probs['p_di']),  # (AA,AA,AA)
        'log_p_de': sl(probs['p_de']),  # (AA,AA)
        'log_p_sm': sl(probs['p_sm']),  # (AA,AA)
        'log_p_si': sl(probs['p_si']),  # (AA,AA)
        'log_p_sd': sl(probs['p_sd']),  # (AA,AA)
        'log_p_se': float(sl(np.asarray(probs['p_se']))),
        'log_singlet': np.asarray(probs['log_singlet']),  # (AA, AA)
        'P_domains': np.asarray(dist['P_domains']),
    }


def compute_hmm_log_tensors(params, n_classes, tau, precomp=None):
    """Distill MixDom to order-1 HMM and return log transition tensors.

    Uses HMM normalization: all transitions normalized per (a, b) giving
    P(next_state, a', b' | current_state, a, b). Correct for pairwise
    scoring but NOT for tree composition (use compute_wfst_log_tensors
    for compose_intersect).
    """
    from ..distill.maraschino import distill_mixdom, normalize_freqs_hmm
    dist = distill_mixdom(params, tau, n_classes, precomp)
    probs = normalize_freqs_hmm(dist)

    sl = lambda x: np.log(np.maximum(np.asarray(x), 1e-300))

    return {
        'log_p_mm': sl(probs['p_mm']),
        'log_p_mi': sl(probs['p_mi']),
        'log_p_md': sl(probs['p_md']),
        'log_p_me': sl(probs['p_me']),
        'log_p_im': sl(probs['p_im']),
        'log_p_ii': sl(probs['p_ii']),
        'log_p_id': sl(probs['p_id']),
        'log_p_ie': sl(probs['p_ie']),
        'log_p_dm': sl(probs['p_dm']),
        'log_p_dd': sl(probs['p_dd']),
        'log_p_di': sl(probs['p_di']),
        'log_p_de': sl(probs['p_de']),
        'log_p_sm': sl(probs['p_sm']),
        'log_p_si': sl(probs['p_si']),
        'log_p_sd': sl(probs['p_sd']),
        'log_p_se': float(sl(np.asarray(probs['p_se']))),
        'P_domains': np.asarray(dist['P_domains']),
    }


def viterbi_profile_wfst(wfst, profile_x, profile_y, sub_left, sub_right):
    """Order-1 WFST Viterbi alignment of two profiles.

    Uses full context-dependent transitions from the distilled WFST.
    At each DP cell, tracks emission context:
    - Match: (a, b) ancestor/descendant characters (AA x AA entries)
    - Insert: (x, y) ancestor_passthrough/descendant_emitted (AA x AA entries)
    - Delete: (x, y) ancestor_emitted/descendant_passthrough (AA x AA entries)

    The WFST transition tensors encode joint transition+emission probabilities.
    Profile conditional likelihoods act as Felsenstein emission factors.

    Args:
        wfst: dict of log transition tensors from compute_wfst_log_tensors
        profile_x: (Lx, A) left child conditional likelihoods
        profile_y: (Ly, A) right child conditional likelihoods
        sub_left: (A, A) substitution matrix for parent profile construction
        sub_right: (A, A) substitution matrix for parent profile construction

    Returns:
        log_prob: log probability of best alignment
        path: list of (i, j, state_type) tuples
        parent_profile: (L_parent, A) conditional likelihoods at parent
    """
    AA = profile_x.shape[1]
    Lx = profile_x.shape[0]
    Ly = profile_y.shape[0]

    with np.errstate(divide='ignore'):
        log_cl_x = np.log(np.maximum(np.asarray(profile_x), 1e-300))
        log_cl_y = np.log(np.maximum(np.asarray(profile_y), 1e-300))

    lp = {k: wfst[k] for k in wfst if k.startswith('log_p_')}

    # DP tables
    V_M = np.full((Lx + 1, Ly + 1, AA, AA), NEG_INF)
    V_I = np.full((Lx + 1, Ly + 1, AA, AA), NEG_INF)
    V_D = np.full((Lx + 1, Ly + 1, AA, AA), NEG_INF)

    # Traceback: (prev_type, prev_ctx_flat)
    # prev_type: 0=M, 1=I, 2=D, 3=S
    TB_M_type = np.zeros((Lx + 1, Ly + 1, AA, AA), dtype=np.int8)
    TB_M_ctx = np.zeros((Lx + 1, Ly + 1, AA, AA), dtype=np.int16)
    TB_I_type = np.zeros((Lx + 1, Ly + 1, AA, AA), dtype=np.int8)
    TB_I_ctx = np.zeros((Lx + 1, Ly + 1, AA, AA), dtype=np.int16)
    TB_D_type = np.zeros((Lx + 1, Ly + 1, AA, AA), dtype=np.int8)
    TB_D_ctx = np.zeros((Lx + 1, Ly + 1, AA, AA), dtype=np.int16)

    # Anti-diagonal fill
    for d in range(1, Lx + Ly + 1):
        i_min = max(0, d - Ly)
        i_max = min(d, Lx)
        for i in range(i_min, i_max + 1):
            j = d - i

            # === Match at (i,j): predecessor at (i-1, j-1) ===
            if i > 0 and j > 0:
                cl_m = log_cl_x[i - 1, :, None] + log_cl_y[j - 1, None, :]

                best_score = np.full((AA, AA), NEG_INF)
                best_type = np.zeros((AA, AA), dtype=np.int8)
                best_ctx = np.zeros((AA, AA), dtype=np.int16)

                # From M
                vm = V_M[i - 1, j - 1]
                if vm.max() > NEG_INF + 1:
                    # scores[a',b'] = max_{a,b} vm[a,b] + lp_mm[a,b,a',b']
                    combined = vm[:, :, None, None] + lp['log_p_mm']
                    flat = combined.reshape(AA * AA, AA, AA)
                    sc = flat.max(axis=0)
                    ct = flat.argmax(axis=0)
                    mask = sc > best_score
                    best_score[mask] = sc[mask]
                    best_type[mask] = 0
                    best_ctx[mask] = ct[mask]

                # From I: I(x,y) → M(a,b) via p_im[x,y,a,b]
                vi = V_I[i - 1, j - 1]
                if vi.max() > NEG_INF + 1:
                    combined = vi[:, :, None, None] + lp['log_p_im']
                    flat = combined.reshape(AA * AA, AA, AA)
                    sc = flat.max(axis=0)
                    ct = flat.argmax(axis=0)
                    mask = sc > best_score
                    best_score[mask] = sc[mask]
                    best_type[mask] = 1
                    best_ctx[mask] = ct[mask]

                # From D: D(x,y) → M(a,b) via p_dm[x,y,a,b]
                vd = V_D[i - 1, j - 1]
                if vd.max() > NEG_INF + 1:
                    combined = vd[:, :, None, None] + lp['log_p_dm']
                    flat = combined.reshape(AA * AA, AA, AA)
                    sc = flat.max(axis=0)
                    ct = flat.argmax(axis=0)
                    mask = sc > best_score
                    best_score[mask] = sc[mask]
                    best_type[mask] = 2
                    best_ctx[mask] = ct[mask]

                # From S (only at i=1, j=1)
                if i == 1 and j == 1:
                    sc = lp['log_p_sm']
                    mask = sc > best_score
                    best_score[mask] = sc[mask]
                    best_type[mask] = 3
                    best_ctx[mask] = 0

                V_M[i, j] = best_score + cl_m
                TB_M_type[i, j] = best_type
                TB_M_ctx[i, j] = best_ctx

            # === Insert at (i,j): predecessor at (i, j-1) ===
            # I context: (x=ancestor_passthrough, y=descendant_emitted)
            if j > 0:
                cl_i = log_cl_y[j - 1]  # (AA,) emission on y axis

                best_score = np.full((AA, AA), NEG_INF)
                best_type = np.zeros((AA, AA), dtype=np.int8)
                best_ctx = np.zeros((AA, AA), dtype=np.int16)

                # From M(a,b) → I(a,c): p_mi[a,b,c], max over b
                vm = V_M[i, j - 1]
                if vm.max() > NEG_INF + 1:
                    combined = vm[:, :, None] + lp['log_p_mi']  # (AA,AA,AA) [a,b,c]
                    sc = combined.max(axis=1)  # (AA,AA) [a,c]
                    ct_b = combined.argmax(axis=1)  # (AA,AA) [a,c]
                    mask = sc > best_score
                    best_score[mask] = sc[mask]
                    best_type[mask] = 0
                    # Source M ctx flat: a*AA + best_b
                    source_flat = np.arange(AA)[:, None] * AA + ct_b
                    best_ctx[mask] = source_flat[mask]

                # From I(x,y) → I(x,z): p_ii[x,y,z], max over y
                vi = V_I[i, j - 1]
                if vi.max() > NEG_INF + 1:
                    combined = vi[:, :, None] + lp['log_p_ii']  # (AA,AA,AA) [x,y,z]
                    sc = combined.max(axis=1)  # (AA,AA) [x,z]
                    ct_y = combined.argmax(axis=1)  # (AA,AA) [x,z]
                    mask = sc > best_score
                    best_score[mask] = sc[mask]
                    best_type[mask] = 1
                    source_flat = np.arange(AA)[:, None] * AA + ct_y
                    best_ctx[mask] = source_flat[mask]

                # From D(x,y) → I(x,z): p_di[x,y,z], max over y
                vd = V_D[i, j - 1]
                if vd.max() > NEG_INF + 1:
                    combined = vd[:, :, None] + lp['log_p_di']  # (AA,AA,AA) [x,y,z]
                    sc = combined.max(axis=1)  # (AA,AA) [x,z]
                    ct_y = combined.argmax(axis=1)
                    mask = sc > best_score
                    best_score[mask] = sc[mask]
                    best_type[mask] = 2
                    source_flat = np.arange(AA)[:, None] * AA + ct_y
                    best_ctx[mask] = source_flat[mask]

                # From S (only at i=0, j=1)
                if i == 0 and j == 1:
                    sc = lp['log_p_si']  # (AA,AA)
                    mask = sc > best_score
                    best_score[mask] = sc[mask]
                    best_type[mask] = 3
                    best_ctx[mask] = 0

                V_I[i, j] = best_score + cl_i[None, :]  # emission on y axis
                TB_I_type[i, j] = best_type
                TB_I_ctx[i, j] = best_ctx

            # === Delete at (i,j): predecessor at (i-1, j) ===
            # D context: (x=ancestor_emitted, y=descendant_passthrough)
            if i > 0:
                cl_d = log_cl_x[i - 1]  # (AA,) emission on x axis

                best_score = np.full((AA, AA), NEG_INF)
                best_type = np.zeros((AA, AA), dtype=np.int8)
                best_ctx = np.zeros((AA, AA), dtype=np.int16)

                # From M(a,b) → D(c,b): p_md[a,b,c], max over a
                vm = V_M[i - 1, j]
                if vm.max() > NEG_INF + 1:
                    combined = vm[:, :, None] + lp['log_p_md']  # (AA,AA,AA) [a,b,c]
                    # D(c,b): max_a combined[a,b,c]
                    sc_bc = combined.max(axis=0)  # (AA,AA) [b,c]
                    ct_a_bc = combined.argmax(axis=0)  # (AA,AA) [b,c]
                    sc = sc_bc.T  # (AA,AA) [c,b]
                    ct_a = ct_a_bc.T  # (AA,AA) [c,b]
                    mask = sc > best_score
                    best_score[mask] = sc[mask]
                    best_type[mask] = 0
                    # Source M ctx flat: best_a*AA + b
                    source_flat = ct_a * AA + np.arange(AA)[None, :]
                    best_ctx[mask] = source_flat[mask]

                # From I(x,y) → D(z,y): p_id[x,y,z], max over x
                vi = V_I[i - 1, j]
                if vi.max() > NEG_INF + 1:
                    combined = vi[:, :, None] + lp['log_p_id']  # (AA,AA,AA) [x,y,z]
                    sc_yz = combined.max(axis=0)  # (AA,AA) [y,z]
                    ct_x_yz = combined.argmax(axis=0)  # (AA,AA) [y,z]
                    sc = sc_yz.T  # (AA,AA) [z,y]
                    ct_x = ct_x_yz.T  # (AA,AA) [z,y]
                    mask = sc > best_score
                    best_score[mask] = sc[mask]
                    best_type[mask] = 1
                    source_flat = ct_x * AA + np.arange(AA)[None, :]
                    best_ctx[mask] = source_flat[mask]

                # From D(x,y) → D(z,y): p_dd[x,y,z], max over x
                vd = V_D[i - 1, j]
                if vd.max() > NEG_INF + 1:
                    combined = vd[:, :, None] + lp['log_p_dd']  # (AA,AA,AA) [x,y,z]
                    sc_yz = combined.max(axis=0)  # (AA,AA) [y,z]
                    ct_x_yz = combined.argmax(axis=0)
                    sc = sc_yz.T  # (AA,AA) [z,y]
                    ct_x = ct_x_yz.T
                    mask = sc > best_score
                    best_score[mask] = sc[mask]
                    best_type[mask] = 2
                    source_flat = ct_x * AA + np.arange(AA)[None, :]
                    best_ctx[mask] = source_flat[mask]

                # From S (only at i=1, j=0)
                if i == 1 and j == 0:
                    sc = lp['log_p_sd']  # (AA,AA)
                    mask = sc > best_score
                    best_score[mask] = sc[mask]
                    best_type[mask] = 3
                    best_ctx[mask] = 0

                V_D[i, j] = best_score + cl_d[:, None]  # emission on x axis
                TB_D_type[i, j] = best_type
                TB_D_ctx[i, j] = best_ctx

    # Terminal: best score at (Lx, Ly) → End
    end_scores = []
    end_info = []  # (state, ctx)

    if V_M[Lx, Ly].max() > NEG_INF + 1:
        sc_m = V_M[Lx, Ly] + lp['log_p_me']
        best_flat = int(sc_m.argmax())
        end_scores.append(sc_m.flat[best_flat])
        end_info.append((M, best_flat))
    else:
        end_scores.append(NEG_INF)
        end_info.append((M, 0))

    if V_I[Lx, Ly].max() > NEG_INF + 1:
        sc_i = V_I[Lx, Ly] + lp['log_p_ie']  # (AA,AA)
        best_flat = int(sc_i.argmax())
        end_scores.append(sc_i.flat[best_flat])
        end_info.append((I, best_flat))
    else:
        end_scores.append(NEG_INF)
        end_info.append((I, 0))

    if V_D[Lx, Ly].max() > NEG_INF + 1:
        sc_d = V_D[Lx, Ly] + lp['log_p_de']  # (AA,AA)
        best_flat = int(sc_d.argmax())
        end_scores.append(sc_d.flat[best_flat])
        end_info.append((D, best_flat))
    else:
        end_scores.append(NEG_INF)
        end_info.append((D, 0))

    # S→E (empty alignment)
    if Lx == 0 and Ly == 0:
        end_scores.append(lp['log_p_se'])
        end_info.append((S, 0))
    else:
        end_scores.append(NEG_INF)
        end_info.append((S, 0))

    end_scores = np.array(end_scores)
    best_end = int(end_scores.argmax())
    log_prob = float(end_scores[best_end])
    cur_state, cur_ctx_flat = end_info[best_end]

    # Decode context (all states now use (x,y) = divmod flat index)
    if cur_state == M:
        cur_ctx = divmod(cur_ctx_flat, AA)
    elif cur_state == I:
        cur_ctx = divmod(cur_ctx_flat, AA)
    elif cur_state == D:
        cur_ctx = divmod(cur_ctx_flat, AA)
    else:
        cur_ctx = ()

    # Traceback
    path = []
    ci, cj = Lx, Ly
    while ci > 0 or cj > 0:
        path.append((ci, cj, int(cur_state)))

        if cur_state == M:
            a, b = cur_ctx
            prev_type = int(TB_M_type[ci, cj, a, b])
            prev_ctx_flat = int(TB_M_ctx[ci, cj, a, b])
            ci, cj = ci - 1, cj - 1
        elif cur_state == I:
            x, y = cur_ctx
            prev_type = int(TB_I_type[ci, cj, x, y])
            prev_ctx_flat = int(TB_I_ctx[ci, cj, x, y])
            ci, cj = ci, cj - 1
        elif cur_state == D:
            x, y = cur_ctx
            prev_type = int(TB_D_type[ci, cj, x, y])
            prev_ctx_flat = int(TB_D_ctx[ci, cj, x, y])
            ci, cj = ci - 1, cj
        else:
            break

        if prev_type == 3:  # from S
            break
        elif prev_type == 0:  # from M
            cur_state = M
            cur_ctx = divmod(prev_ctx_flat, AA)
        elif prev_type == 1:  # from I
            cur_state = I
            cur_ctx = divmod(prev_ctx_flat, AA)
        elif prev_type == 2:  # from D
            cur_state = D
            cur_ctx = divmod(prev_ctx_flat, AA)

    path.append((0, 0, int(S)))
    path.reverse()

    # Build parent profile from alignment path (Felsenstein peeling)
    sub_left_np = np.asarray(sub_left)
    sub_right_np = np.asarray(sub_right)
    left_contrib = np.asarray(profile_x) @ sub_left_np.T
    right_contrib = np.asarray(profile_y) @ sub_right_np.T

    parent_conds = []
    for i_pos, j_pos, st in path:
        if st == M and i_pos > 0 and j_pos > 0:
            parent_conds.append(left_contrib[i_pos - 1] * right_contrib[j_pos - 1])
        elif st == D and i_pos > 0:
            parent_conds.append(left_contrib[i_pos - 1])

    if parent_conds:
        parent_profile = np.stack(parent_conds, axis=0)
    else:
        parent_profile = np.zeros((0, AA))

    return log_prob, path, parent_profile


def forward_profile_wfst(wfst, profile_x, profile_y):
    """Order-1 WFST Forward algorithm for profile-based pair HMM.

    Returns the total log probability and forward tables for sampling.

    Args:
        wfst: dict of log transition tensors from compute_wfst_log_tensors
        profile_x: (Lx, A) left child conditional likelihoods
        profile_y: (Ly, A) right child conditional likelihoods

    Returns:
        log_prob: total log probability
        F_M: (Lx+1, Ly+1, AA, AA) Match forward table
        F_I: (Lx+1, Ly+1, AA, AA) Insert forward table
        F_D: (Lx+1, Ly+1, AA, AA) Delete forward table
    """
    from scipy.special import logsumexp as _lse

    AA = profile_x.shape[1]
    Lx = profile_x.shape[0]
    Ly = profile_y.shape[0]

    with np.errstate(divide='ignore'):
        log_cl_x = np.log(np.maximum(np.asarray(profile_x), 1e-300))
        log_cl_y = np.log(np.maximum(np.asarray(profile_y), 1e-300))

    lp = {k: wfst[k] for k in wfst if k.startswith('log_p_')}

    F_M = np.full((Lx + 1, Ly + 1, AA, AA), NEG_INF)
    F_I = np.full((Lx + 1, Ly + 1, AA, AA), NEG_INF)
    F_D = np.full((Lx + 1, Ly + 1, AA, AA), NEG_INF)

    for d in range(1, Lx + Ly + 1):
        i_min = max(0, d - Ly)
        i_max = min(d, Lx)
        for i in range(i_min, i_max + 1):
            j = d - i

            # === Match at (i,j) ===
            if i > 0 and j > 0:
                cl_m = log_cl_x[i - 1, :, None] + log_cl_y[j - 1, None, :]
                contribs = []

                fm = F_M[i - 1, j - 1]
                if fm.max() > NEG_INF + 1:
                    c = fm[:, :, None, None] + lp['log_p_mm']
                    contribs.append(_lse(c.reshape(AA * AA, AA, AA), axis=0))

                # I(x,y) → M(a,b) via p_im[x,y,a,b]
                fi = F_I[i - 1, j - 1]
                if fi.max() > NEG_INF + 1:
                    c = fi[:, :, None, None] + lp['log_p_im']
                    contribs.append(_lse(c.reshape(AA * AA, AA, AA), axis=0))

                # D(x,y) → M(a,b) via p_dm[x,y,a,b]
                fd = F_D[i - 1, j - 1]
                if fd.max() > NEG_INF + 1:
                    c = fd[:, :, None, None] + lp['log_p_dm']
                    contribs.append(_lse(c.reshape(AA * AA, AA, AA), axis=0))

                if i == 1 and j == 1:
                    contribs.append(lp['log_p_sm'])

                if contribs:
                    F_M[i, j] = _lse(np.stack(contribs), axis=0) + cl_m

            # === Insert at (i,j): I(x,y) = (ancestor_pass, desc_emitted) ===
            if j > 0:
                cl_i = log_cl_y[j - 1]  # (AA,) emission on y axis
                contribs = []

                # M(a,b) → I(a,c): p_mi[a,b,c], sum over b
                fm = F_M[i, j - 1]
                if fm.max() > NEG_INF + 1:
                    c = fm[:, :, None] + lp['log_p_mi']  # (AA,AA,AA) [a,b,c]
                    contribs.append(_lse(c, axis=1))  # (AA,AA) [a,c]

                # I(x,y) → I(x,z): p_ii[x,y,z], sum over y
                fi = F_I[i, j - 1]
                if fi.max() > NEG_INF + 1:
                    c = fi[:, :, None] + lp['log_p_ii']  # (AA,AA,AA) [x,y,z]
                    contribs.append(_lse(c, axis=1))  # (AA,AA) [x,z]

                # D(x,y) → I(x,z): p_di[x,y,z], sum over y
                fd = F_D[i, j - 1]
                if fd.max() > NEG_INF + 1:
                    c = fd[:, :, None] + lp['log_p_di']  # (AA,AA,AA) [x,y,z]
                    contribs.append(_lse(c, axis=1))  # (AA,AA) [x,z]

                if i == 0 and j == 1:
                    contribs.append(lp['log_p_si'])  # (AA,AA)

                if contribs:
                    F_I[i, j] = _lse(np.stack(contribs), axis=0) + cl_i[None, :]

            # === Delete at (i,j): D(x,y) = (ancestor_emitted, desc_pass) ===
            if i > 0:
                cl_d = log_cl_x[i - 1]  # (AA,) emission on x axis
                contribs = []

                # M(a,b) → D(c,b): p_md[a,b,c], sum over a → (b,c) → transpose → (c,b)
                fm = F_M[i - 1, j]
                if fm.max() > NEG_INF + 1:
                    c = fm[:, :, None] + lp['log_p_md']  # (AA,AA,AA) [a,b,c]
                    contribs.append(_lse(c, axis=0).T)  # sum over a → (b,c).T → (c,b)

                # I(x,y) → D(z,y): p_id[x,y,z], sum over x → (y,z) → transpose → (z,y)
                fi = F_I[i - 1, j]
                if fi.max() > NEG_INF + 1:
                    c = fi[:, :, None] + lp['log_p_id']  # (AA,AA,AA) [x,y,z]
                    contribs.append(_lse(c, axis=0).T)  # (z,y)

                # D(x,y) → D(z,y): p_dd[x,y,z], sum over x → (y,z) → transpose → (z,y)
                fd = F_D[i - 1, j]
                if fd.max() > NEG_INF + 1:
                    c = fd[:, :, None] + lp['log_p_dd']  # (AA,AA,AA) [x,y,z]
                    contribs.append(_lse(c, axis=0).T)  # (z,y)

                if i == 1 and j == 0:
                    contribs.append(lp['log_p_sd'])  # (AA,AA)

                if contribs:
                    F_D[i, j] = _lse(np.stack(contribs), axis=0) + cl_d[:, None]

    # Terminal
    end_contribs = []
    if F_M[Lx, Ly].max() > NEG_INF + 1:
        end_contribs.append(_lse((F_M[Lx, Ly] + lp['log_p_me']).ravel()))
    if F_I[Lx, Ly].max() > NEG_INF + 1:
        end_contribs.append(_lse((F_I[Lx, Ly] + lp['log_p_ie']).ravel()))
    if F_D[Lx, Ly].max() > NEG_INF + 1:
        end_contribs.append(_lse((F_D[Lx, Ly] + lp['log_p_de']).ravel()))
    if Lx == 0 and Ly == 0:
        end_contribs.append(lp['log_p_se'])

    if end_contribs:
        log_prob = float(_lse(np.array(end_contribs)))
    else:
        log_prob = NEG_INF

    return log_prob, F_M, F_I, F_D


def sample_traceback_profile_wfst(F_M, F_I, F_D, wfst, Lx, Ly, rng_key):
    """Stochastic traceback on order-1 WFST Forward table.

    Samples a path proportional to forward probability, respecting
    context-dependent transitions.

    Args:
        F_M: (Lx+1, Ly+1, AA, AA) Match forward table
        F_I: (Lx+1, Ly+1, AA, AA) Insert forward table
        F_D: (Lx+1, Ly+1, AA, AA) Delete forward table
        wfst: dict of log transition tensors
        Lx, Ly: real sequence lengths
        rng_key: JAX random key

    Returns:
        path: list of (i, j, state_type) tuples
    """
    from scipy.special import logsumexp as _lse

    AA = F_M.shape[2]
    lp = {k: wfst[k] for k in wfst if k.startswith('log_p_')}

    # Sample terminal state+context at (Lx, Ly)
    all_terminal = []  # list of (log_score, state, ctx_flat)
    if F_M[Lx, Ly].max() > NEG_INF + 1:
        sc_m = (F_M[Lx, Ly] + lp['log_p_me']).ravel()
        for idx in range(len(sc_m)):
            if sc_m[idx] > NEG_INF + 1:
                all_terminal.append((sc_m[idx], M, idx))
    if F_I[Lx, Ly].max() > NEG_INF + 1:
        sc_i = (F_I[Lx, Ly] + lp['log_p_ie']).ravel()
        for idx in range(len(sc_i)):
            if sc_i[idx] > NEG_INF + 1:
                all_terminal.append((sc_i[idx], I, idx))
    if F_D[Lx, Ly].max() > NEG_INF + 1:
        sc_d = (F_D[Lx, Ly] + lp['log_p_de']).ravel()
        for idx in range(len(sc_d)):
            if sc_d[idx] > NEG_INF + 1:
                all_terminal.append((sc_d[idx], D, idx))

    if not all_terminal:
        return [(0, 0, int(S))]

    term_scores = np.array([t[0] for t in all_terminal])
    term_scores -= term_scores.max()
    term_probs = np.exp(term_scores)
    term_probs /= term_probs.sum() + 1e-30

    rng_key, subkey = jr.split(rng_key)
    choice = int(jr.choice(subkey, len(all_terminal), p=jnp.array(term_probs)))
    _, cur_state, cur_ctx_flat = all_terminal[choice]

    cur_ctx = divmod(cur_ctx_flat, AA)  # all states use (x,y) flat encoding

    path = []
    ci, cj = Lx, Ly

    while ci > 0 or cj > 0:
        path.append((ci, cj, int(cur_state)))

        # Compute predecessor scores
        if cur_state == M and ci > 0 and cj > 0:
            a2, b2 = cur_ctx
            pi_, pj_ = ci - 1, cj - 1
            preds = []
            # From M
            fm = F_M[pi_, pj_]
            if fm.max() > NEG_INF + 1:
                sc = fm + lp['log_p_mm'][:, :, a2, b2]
                for idx in range(AA * AA):
                    a, b = divmod(idx, AA)
                    if sc[a, b] > NEG_INF + 1:
                        preds.append((sc[a, b], 0, a * AA + b))
            # From I(x,y) → M(a2,b2) via p_im[x,y,a2,b2]
            fi = F_I[pi_, pj_]
            if fi.max() > NEG_INF + 1:
                sc = fi + lp['log_p_im'][:, :, a2, b2]
                for idx in range(AA * AA):
                    x, y = divmod(idx, AA)
                    if sc[x, y] > NEG_INF + 1:
                        preds.append((sc[x, y], 1, x * AA + y))
            # From D(x,y) → M(a2,b2) via p_dm[x,y,a2,b2]
            fd = F_D[pi_, pj_]
            if fd.max() > NEG_INF + 1:
                sc = fd + lp['log_p_dm'][:, :, a2, b2]
                for idx in range(AA * AA):
                    x, y = divmod(idx, AA)
                    if sc[x, y] > NEG_INF + 1:
                        preds.append((sc[x, y], 2, x * AA + y))
            # From S
            if ci == 1 and cj == 1:
                preds.append((lp['log_p_sm'][a2, b2], 3, 0))

        elif cur_state == I and cj > 0:
            # I context: (x2=ancestor_pass, y2=desc_emitted)
            x2, y2 = cur_ctx
            pi_, pj_ = ci, cj - 1
            preds = []
            # From M(a,b) → I(a,c): p_mi[a,b,c]. Here dest I(x2,y2) so a=x2, c=y2.
            # Source M: p_mi[x2, :, y2] selects b
            fm = F_M[pi_, pj_]
            if fm.max() > NEG_INF + 1:
                sc = fm[x2, :] + lp['log_p_mi'][x2, :, y2]  # (AA,) over b
                for b in range(AA):
                    if sc[b] > NEG_INF + 1:
                        preds.append((sc[b], 0, x2 * AA + b))
            # From I(x,y) → I(x,z): p_ii[x,y,z]. Dest I(x2,y2) so x=x2, z=y2.
            fi = F_I[pi_, pj_]
            if fi.max() > NEG_INF + 1:
                sc = fi[x2, :] + lp['log_p_ii'][x2, :, y2]  # (AA,) over y
                for y in range(AA):
                    if sc[y] > NEG_INF + 1:
                        preds.append((sc[y], 1, x2 * AA + y))
            # From D(x,y) → I(x,z): p_di[x,y,z]. Dest I(x2,y2) so x=x2, z=y2.
            fd = F_D[pi_, pj_]
            if fd.max() > NEG_INF + 1:
                sc = fd[x2, :] + lp['log_p_di'][x2, :, y2]  # (AA,) over y
                for y in range(AA):
                    if sc[y] > NEG_INF + 1:
                        preds.append((sc[y], 2, x2 * AA + y))
            if ci == 0 and cj == 1:
                preds.append((lp['log_p_si'][x2, y2], 3, 0))

        elif cur_state == D and ci > 0:
            # D context: (x2=ancestor_emitted, y2=desc_pass)
            x2, y2 = cur_ctx
            pi_, pj_ = ci - 1, cj
            preds = []
            # From M(a,b) → D(c,b): p_md[a,b,c]. Dest D(x2,y2) so c=x2, b=y2.
            fm = F_M[pi_, pj_]
            if fm.max() > NEG_INF + 1:
                sc = fm[:, y2] + lp['log_p_md'][:, y2, x2]  # (AA,) over a
                for a in range(AA):
                    if sc[a] > NEG_INF + 1:
                        preds.append((sc[a], 0, a * AA + y2))
            # From I(x,y) → D(z,y): p_id[x,y,z]. Dest D(x2,y2) so z=x2, y=y2.
            fi = F_I[pi_, pj_]
            if fi.max() > NEG_INF + 1:
                sc = fi[:, y2] + lp['log_p_id'][:, y2, x2]  # (AA,) over x
                for x in range(AA):
                    if sc[x] > NEG_INF + 1:
                        preds.append((sc[x], 1, x * AA + y2))
            # From D(x,y) → D(z,y): p_dd[x,y,z]. Dest D(x2,y2) so z=x2, y=y2.
            fd = F_D[pi_, pj_]
            if fd.max() > NEG_INF + 1:
                sc = fd[:, y2] + lp['log_p_dd'][:, y2, x2]  # (AA,) over x
                for x in range(AA):
                    if sc[x] > NEG_INF + 1:
                        preds.append((sc[x], 2, x * AA + y2))
            if ci == 1 and cj == 0:
                preds.append((lp['log_p_sd'][x2, y2], 3, 0))
        else:
            break

        if not preds:
            break

        pred_scores = np.array([p[0] for p in preds])
        pred_scores -= pred_scores.max()
        pred_probs = np.exp(pred_scores)
        pred_probs /= pred_probs.sum() + 1e-30

        rng_key, subkey = jr.split(rng_key)
        choice = int(jr.choice(subkey, len(preds), p=jnp.array(pred_probs)))
        _, prev_type, prev_ctx_flat = preds[choice]

        if cur_state == M:
            ci, cj = ci - 1, cj - 1
        elif cur_state == I:
            cj = cj - 1
        elif cur_state == D:
            ci = ci - 1

        if prev_type == 3:  # from S
            break
        elif prev_type == 0:  # from M
            cur_state = M
            cur_ctx = divmod(prev_ctx_flat, AA)
        elif prev_type == 1:  # from I
            cur_state = I
            cur_ctx = divmod(prev_ctx_flat, AA)
        elif prev_type == 2:  # from D
            cur_state = D
            cur_ctx = divmod(prev_ctx_flat, AA)

    path.append((0, 0, int(S)))
    path.reverse()
    return path


def viterbi_profile(log_trans, state_types, profile_x, profile_y,
                    sub_left, sub_right, pi, emit_override=None,
                    is_root=True):
    """Viterbi alignment of two profiles.

    Args:
        log_trans: (5, 5) log-transition matrix
        state_types: (5,) state type codes [S, M, I, D, E]
        profile_x: (Lx, A) left child profile
        profile_y: (Ly, A) right child profile
        sub_left: (A, A) substitution matrix P(b|a, t_left)
        sub_right: (A, A) substitution matrix P(b|a, t_right)
        pi: (A,) equilibrium distribution
        emit_override: optional (Lx+1, Ly+1, ns) precomputed log emission table.
            When provided, skips internal emission computation. sub_left/sub_right
            are still used for parent profile computation.
        is_root: if True, weight emissions by pi; if False, weight uniformly

    Returns:
        log_prob: log probability of best alignment
        path: list of (i, j, state_type) tuples
        parent_profile: (L_parent, A) conditional likelihoods at parent
    """
    Lx = profile_x.shape[0]
    Ly = profile_y.shape[0]
    A = pi.shape[0]
    n_states = log_trans.shape[0]

    # Precompute contributions (always needed for parent profile)
    left_contrib = profile_x @ sub_left.T   # (Lx, A)
    right_contrib = profile_y @ sub_right.T  # (Ly, A)

    # Compute emissions
    if emit_override is not None:
        emit = emit_override
    else:
        emit = profile_emissions(state_types, profile_x, profile_y,
                                 sub_left, sub_right, pi, is_root=is_root)

    # Viterbi DP (not JIT-compiled — Python loop for traceback)
    V = np.full((Lx + 1, Ly + 1, n_states), NEG_INF)
    V[0, 0, S] = 0.0
    TB = np.zeros((Lx + 1, Ly + 1, n_states, 3), dtype=np.int32)

    emit_np = np.asarray(emit)
    log_trans_np = np.asarray(log_trans)
    state_types_np = np.asarray(state_types)

    for d in range(1, Lx + Ly + 1):
        i_min = max(0, d - Ly)
        i_max = min(d, Lx)
        for i in range(i_min, i_max + 1):
            j = d - i
            if i == 0 and j == 0:
                continue
            for k in range(n_states):
                st = state_types_np[k]
                if st == M and i > 0 and j > 0:
                    scores = V[i - 1, j - 1, :] + log_trans_np[:, k]
                    best = np.argmax(scores)
                    V[i, j, k] = scores[best] + emit_np[i, j, k]
                    TB[i, j, k] = [i - 1, j - 1, best]
                elif st == I and j > 0:
                    scores = V[i, j - 1, :] + log_trans_np[:, k]
                    best = np.argmax(scores)
                    V[i, j, k] = scores[best] + emit_np[i, j, k]
                    TB[i, j, k] = [i, j - 1, best]
                elif st == D and i > 0:
                    scores = V[i - 1, j, :] + log_trans_np[:, k]
                    best = np.argmax(scores)
                    V[i, j, k] = scores[best] + emit_np[i, j, k]
                    TB[i, j, k] = [i - 1, j, best]

    # Terminal
    e_idx = int(np.argmax(state_types_np == E))
    final_scores = V[Lx, Ly, :] + log_trans_np[:, e_idx]
    best_final = int(np.argmax(final_scores))
    log_prob = float(final_scores[best_final])

    # Traceback
    path = []
    ci, cj, ck = Lx, Ly, best_final
    while ci > 0 or cj > 0:
        path.append((ci, cj, int(state_types_np[ck])))
        pi_, pj_, pk_ = TB[ci, cj, ck]
        ci, cj, ck = int(pi_), int(pj_), int(pk_)
    path.append((0, 0, int(S)))
    path.reverse()

    # Build parent profile from alignment path
    parent_conds = []
    for i_pos, j_pos, st in path:
        if st == M and i_pos > 0 and j_pos > 0:
            # Parent conditional = left_contrib * right_contrib (elementwise)
            parent_cond = np.asarray(left_contrib[i_pos - 1]) * np.asarray(right_contrib[j_pos - 1])
            parent_conds.append(parent_cond)
        elif st == D and i_pos > 0:
            # Only left child contributes
            parent_conds.append(np.asarray(left_contrib[i_pos - 1]))
        # I states (insertions in right child) do not contribute to parent

    if parent_conds:
        parent_profile = np.stack(parent_conds, axis=0)
    else:
        parent_profile = np.zeros((0, A))

    return log_prob, path, parent_profile


def forward_profile(log_trans, state_types, profile_x, profile_y,
                    sub_left, sub_right, pi, emit_override=None,
                    is_root=True):
    """Forward algorithm for profile-based pair HMM.

    Uses geometric padding and JIT-compiled core.

    Args:
        Same as viterbi_profile.
        emit_override: optional (Lx+1, Ly+1, ns) precomputed log emission table.
            When provided, skips internal emission computation and uses this
            directly (after padding and masking). Must be at real size, not padded.
        is_root: if True, weight emissions by pi; if False, weight uniformly

    Returns:
        log_prob: total log probability P(x, y | tree)
        F: (Lx+1, Ly+1, n_states) forward table
    """
    Lx = profile_x.shape[0]
    Ly = profile_y.shape[0]
    Lx_pad = _pad_to_bin(Lx)
    Ly_pad = _pad_to_bin(Ly)

    if emit_override is not None:
        # Pad the provided emissions
        ns = state_types.shape[0]
        emit = jnp.full((Lx_pad + 1, Ly_pad + 1, ns), 0.0)
        emit = emit.at[:Lx + 1, :Ly + 1, :].set(emit_override)
    else:
        # Pad profiles
        A = profile_x.shape[1]
        if Lx_pad > Lx:
            profile_x_pad = jnp.concatenate([profile_x,
                                              jnp.zeros((Lx_pad - Lx, A))], axis=0)
        else:
            profile_x_pad = profile_x
        if Ly_pad > Ly:
            profile_y_pad = jnp.concatenate([profile_y,
                                              jnp.zeros((Ly_pad - Ly, A))], axis=0)
        else:
            profile_y_pad = profile_y

        emit = profile_emissions(state_types, profile_x_pad, profile_y_pad,
                                 sub_left, sub_right, pi, is_root=is_root)

    mask = _emit_mask(Lx, Ly, Lx_pad, Ly_pad, state_types.shape[0])
    emit = jnp.where(mask, emit, NEG_INF)

    _, F_pad = _forward_2d_core(log_trans, state_types, emit, Lx_pad, Ly_pad)

    e_idx = _find_e_idx(state_types)
    log_prob = jax.nn.logsumexp(F_pad[Lx, Ly, :] + log_trans[:, e_idx])
    F = F_pad[:Lx + 1, :Ly + 1, :]
    return log_prob, F


def sample_traceback_profile(F, log_trans, state_types, Lx, Ly, rng_key):
    """Stochastic traceback on profile Forward table.

    Samples a path proportional to forward probability.

    Args:
        F: (Lx+1, Ly+1, n_states) forward table from forward_profile
        log_trans: (5, 5) log-transition matrix
        state_types: (5,) state type codes
        Lx, Ly: real sequence lengths
        rng_key: JAX random key

    Returns:
        path: list of (i, j, state_type) tuples
    """
    n_states = log_trans.shape[0]
    F_np = np.asarray(F)
    log_trans_np = np.asarray(log_trans)
    state_types_np = np.asarray(state_types)

    e_idx = int(np.argmax(state_types_np == E))

    # Sample terminal state at (Lx, Ly)
    terminal_scores = F_np[Lx, Ly, :] + log_trans_np[:, e_idx]
    terminal_scores -= np.max(terminal_scores)
    terminal_probs = np.exp(terminal_scores)
    terminal_probs /= terminal_probs.sum() + 1e-30
    rng_key, subkey = jr.split(rng_key)
    ck = int(jr.choice(subkey, n_states, p=jnp.array(terminal_probs)))

    path = []
    ci, cj = Lx, Ly

    while ci > 0 or cj > 0:
        path.append((ci, cj, int(state_types_np[ck])))
        st = state_types_np[ck]

        if st == M and ci > 0 and cj > 0:
            pi_, pj_ = ci - 1, cj - 1
        elif st == I and cj > 0:
            pi_, pj_ = ci, cj - 1
        elif st == D and ci > 0:
            pi_, pj_ = ci - 1, cj
        else:
            break

        # Sample predecessor state
        pred_scores = F_np[pi_, pj_, :] + log_trans_np[:, ck]
        pred_scores -= np.max(pred_scores)
        pred_probs = np.exp(pred_scores)
        pred_probs /= pred_probs.sum() + 1e-30
        rng_key, subkey = jr.split(rng_key)
        pk = int(jr.choice(subkey, n_states, p=jnp.array(pred_probs)))

        ci, cj, ck = pi_, pj_, pk

    path.append((0, 0, int(S)))
    path.reverse()
    return path


def build_recognizer_from_paths(paths, left_contrib, right_contrib):
    """Build a recognizer (set of profile positions) from sampled paths.

    The recognizer is built from the union of all M/D positions across
    all paths, then compressed by merging positions that reference the
    same left child position. This ensures each left child character
    appears at most once in the recognizer, which is required for
    valid MSA extraction.

    When multiple (i, j, st) tuples share the same left position (i),
    their conditional likelihoods are combined using element-wise max
    across alternatives. This captures the best right pairing from
    any sampled path while preserving the constraint that each leaf
    character maps to exactly one recognizer state.

    Args:
        paths: list of paths, each path is list of (i, j, state_type)
        left_contrib: (Lx, A) transformed left child profile
        right_contrib: (Ly, A) transformed right child profile

    Returns:
        parent_profile: (L_parent, A) conditional likelihoods
        merged_path: sorted list of all (i, j, st) tuples including S/I states
        profile_positions: sorted list of (i, j, st) tuples for profile entries
            (one per unique left child position, after compression)
    """
    # Collect all positions from all paths (M/D/I, not S)
    all_positions = set()
    profile_pos_set = set()
    for path in paths:
        for i, j, st in path:
            if st == M or st == D:
                profile_pos_set.add((i, j, st))
                all_positions.add((i, j, st))
            elif st == I:
                all_positions.add((i, j, st))

    st_order = {M: 0, D: 1, I: 2}
    merged_path = [(0, 0, int(S))] + sorted(
        all_positions,
        key=lambda x: (x[0] + x[1], x[0], st_order.get(x[2], 3))
    )

    # Compress: merge profile positions with same left child position.
    # For each unique left position (i), collect all (i, j, st) entries.
    # Keep one representative per left position; combine CLs via max.
    # Prefer M over D when available (M carries more information).
    A = left_contrib.shape[1]
    left_pos_entries = {}  # left_pos -> list of (i, j, st)
    for i, j, st in profile_pos_set:
        lp = i - 1 if i > 0 else i
        left_pos_entries.setdefault(lp, []).append((i, j, st))

    profile_positions = []
    parent_conds = []
    for lp in sorted(left_pos_entries.keys()):
        entries = left_pos_entries[lp]
        # Pick representative: prefer M entries, then D
        m_entries = [(i, j, st) for i, j, st in entries if st == M]
        d_entries = [(i, j, st) for i, j, st in entries if st == D]

        # Compute CL for each alternative, take element-wise max
        cls = []
        for i, j, st in entries:
            if st == M and i > 0 and j > 0:
                cl = np.asarray(left_contrib[i - 1]) * np.asarray(right_contrib[j - 1])
                cls.append(cl)
            elif st == D and i > 0:
                cl = np.asarray(left_contrib[i - 1])
                cls.append(cl)

        if not cls:
            continue

        # Combined CL: element-wise max across alternatives
        combined_cl = cls[0]
        for cl in cls[1:]:
            combined_cl = np.maximum(combined_cl, cl)

        # Pick representative position (prefer M, then the first entry)
        if m_entries:
            # Sort by anti-diagonal, pick first
            rep = sorted(m_entries, key=lambda x: (x[0] + x[1], x[0]))[0]
        elif d_entries:
            rep = sorted(d_entries, key=lambda x: (x[0] + x[1], x[0]))[0]
        else:
            rep = entries[0]

        profile_positions.append(rep)
        parent_conds.append(combined_cl)

    if parent_conds:
        parent_profile = np.stack(parent_conds, axis=0)
    else:
        parent_profile = np.zeros((0, A))

    return parent_profile, merged_path, profile_positions


def viterbi_and_sample_profile(log_trans, state_types, profile_x, profile_y,
                               sub_left, sub_right, pi, n_samples=100,
                               rng_key=None):
    """Viterbi + stochastic sampling for profile-based pair HMM.

    Runs Viterbi for the optimal path, then Forward + N stochastic
    tracebacks to sample additional paths. Returns the union recognizer
    (parent profile built from all M/D positions across all paths).

    Args:
        log_trans, state_types, profile_x, profile_y, sub_left, sub_right, pi:
            same as viterbi_profile
        n_samples: number of stochastic tracebacks (default 100)
        rng_key: JAX random key (required if n_samples > 0)

    Returns:
        log_prob: Viterbi log probability
        fwd_log_prob: Forward log probability (>= Viterbi)
        path: Viterbi path (canonical, for MSA extraction)
        parent_profile: (L_parent, A) recognizer profile
        n_paths: total number of paths (1 + n_samples)
        n_profile_positions: number of distinct profile positions
    """
    Lx = profile_x.shape[0]
    Ly = profile_y.shape[0]

    # Run Viterbi
    log_prob, viterbi_path, _ = viterbi_profile(
        log_trans, state_types, profile_x, profile_y,
        sub_left, sub_right, pi
    )

    # Precompute contributions for profile building
    left_contrib = np.asarray(profile_x @ np.asarray(sub_left).T)
    right_contrib = np.asarray(profile_y @ np.asarray(sub_right).T)

    if n_samples == 0 or rng_key is None:
        # Viterbi only
        parent_conds = []
        for i, j, st in viterbi_path:
            if st == M and i > 0 and j > 0:
                parent_conds.append(left_contrib[i - 1] * right_contrib[j - 1])
            elif st == D and i > 0:
                parent_conds.append(left_contrib[i - 1])
        A = pi.shape[0]
        parent_profile = np.stack(parent_conds, axis=0) if parent_conds else np.zeros((0, A))
        return log_prob, log_prob, viterbi_path, parent_profile, 1, len(parent_conds)

    # Run Forward for sampling
    fwd_log_prob, F = forward_profile(
        log_trans, state_types, profile_x, profile_y,
        sub_left, sub_right, pi
    )

    # Sample paths
    all_paths = [viterbi_path]
    for k in range(n_samples):
        rng_key, subkey = jr.split(rng_key)
        sampled_path = sample_traceback_profile(
            F, log_trans, state_types, Lx, Ly, subkey
        )
        all_paths.append(sampled_path)

    # Build recognizer from union of all paths
    parent_profile, merged_path, profile_positions = build_recognizer_from_paths(
        all_paths, left_contrib, right_contrib
    )

    return (log_prob, float(fwd_log_prob), merged_path, parent_profile,
            len(all_paths), len(profile_positions))


def reconstruct_progressive_felsenstein(tree_root, leaf_seqs, ins_rate,
                                        del_rate, t_scale, Q, pi,
                                        use_tkf92=False, ext=0.5,
                                        n_samples=0, rng_key=None,
                                        single_seq_ancestors=False):
    """Progressive reconstruction on a tree using Felsenstein profiles.

    Bottom-up: at each internal node, align two child profiles using
    Viterbi (and optionally stochastic sampling), producing a parent
    profile (conditional likelihoods).

    When n_samples > 0, the profile at each internal node is built from
    the union of the Viterbi path and n_samples stochastically sampled
    paths (recognizer construction). This captures alignment uncertainty
    and produces richer profiles for upstream alignments.

    The substitution model is exactly marginalized at each position,
    while the indel model uses the TKF91 or TKF92 pair HMM.

    Args:
        tree_root: TreeNode (root of phylogenetic tree)
        leaf_seqs: dict of {leaf_name: integer_array}
        ins_rate, del_rate: TKF91 indel rates
        t_scale: multiplier for branch lengths (1.0 = use as-is)
        Q: (A, A) substitution rate matrix
        pi: (A,) equilibrium distribution
        use_tkf92: if True, use TKF92 instead of TKF91
        ext: TKF92 fragment extension probability (ignored if use_tkf92=False)
        n_samples: number of stochastic tracebacks per node (0 = Viterbi only)
        rng_key: JAX random key (required if n_samples > 0)
        single_seq_ancestors: if True, convert internal node profiles to
            one-hot sequences (argmax) instead of keeping full CL vectors.
            This tests Viterbi ancestral reconstruction without Felsenstein
            marginalization.

    Returns:
        node_profiles: dict of {node_id: (L, A) profile}
        node_alignments: dict of {node_id: alignment_info}
        root_sequence: integer array (MAP ancestral sequence at root)
    """
    A = pi.shape[0]
    node_profiles = {}
    node_alignments = {}
    node_leaf_maps = {}  # node_id -> list of {leaf_name: seq_pos}

    # Assign leaf profiles and leaf_maps
    for node in tree_root.preorder():
        if node.is_leaf and node.name in leaf_seqs:
            seq = np.asarray(leaf_seqs[node.name])
            node_profiles[id(node)] = np.eye(A)[seq]  # one-hot
            node_leaf_maps[id(node)] = [
                {node.name: i} for i in range(len(seq))
            ]

    # Bottom-up reconstruction
    for node in tree_root.postorder():
        if node.is_leaf:
            continue

        is_root = (node is tree_root)

        children_with_profiles = [c for c in node.children
                                  if id(c) in node_profiles]
        if len(children_with_profiles) == 0:
            continue

        if len(children_with_profiles) == 1:
            child = children_with_profiles[0]
            t = max(child.branch_length * t_scale, 1e-6)
            sub_mat = np.asarray(transition_matrix(Q, t))
            profile = node_profiles[id(child)]
            # Transform through branch
            node_profiles[id(node)] = profile @ sub_mat.T
            # Propagate leaf_maps from child
            if id(child) in node_leaf_maps:
                node_leaf_maps[id(node)] = node_leaf_maps[id(child)]
            continue

        # Two children: align through their common ancestor
        left, right = children_with_profiles[0], children_with_profiles[1]
        t_left = max(left.branch_length * t_scale, 1e-6)
        t_right = max(right.branch_length * t_scale, 1e-6)
        t_total = t_left + t_right

        sub_left = np.asarray(transition_matrix(Q, t_left))
        sub_right = np.asarray(transition_matrix(Q, t_right))
        # Build transition matrix using total time
        if use_tkf92:
            tau = tkf92_trans(ins_rate, del_rate, t_total, ext)
        else:
            tau = tkf91_trans(ins_rate, del_rate, t_total)
        log_trans = safe_log(tau)
        state_types = jnp.array([S, M, I, D, E])

        profile_x = jnp.asarray(node_profiles[id(left)])
        profile_y = jnp.asarray(node_profiles[id(right)])

        if profile_x.shape[0] == 0 or profile_y.shape[0] == 0:
            # Use whichever is non-empty
            if profile_x.shape[0] > 0:
                node_profiles[id(node)] = node_profiles[id(left)]
            else:
                node_profiles[id(node)] = node_profiles[id(right)]
            continue

        # Get leaf_maps for children
        left_lmaps = node_leaf_maps.get(id(left), [])
        right_lmaps = node_leaf_maps.get(id(right), [])

        left_contrib = np.asarray(profile_x @ np.asarray(sub_left).T)
        right_contrib = np.asarray(profile_y @ np.asarray(sub_right).T)

        if n_samples > 0 and rng_key is not None:
            # Order-0 TKF91/TKF92 with sampling
            emit_override = None
            rng_key, subkey = jr.split(rng_key)
            fwd_log_prob, F = forward_profile(
                log_trans, state_types, profile_x, profile_y,
                sub_left, sub_right, pi, is_root=is_root)
            log_prob, viterbi_path, _ = viterbi_profile(
                log_trans, state_types, profile_x, profile_y,
                sub_left, sub_right, pi, is_root=is_root)
            all_paths = [viterbi_path]
            for k in range(n_samples):
                rng_key, subkey2 = jr.split(rng_key)
                sampled = sample_traceback_profile(
                    F, log_trans, state_types,
                    profile_x.shape[0], profile_y.shape[0], subkey2)
                all_paths.append(sampled)

            rec_profile, _, rec_positions = build_recognizer_from_paths(
                all_paths, left_contrib, right_contrib)
            node_profiles[id(node)] = rec_profile

            parent_lmaps = []
            for i_pos, j_pos, st in rec_positions:
                lmap = {}
                if st == M and i_pos > 0 and j_pos > 0:
                    if i_pos - 1 < len(left_lmaps):
                        lmap.update(left_lmaps[i_pos - 1])
                    if j_pos - 1 < len(right_lmaps):
                        lmap.update(right_lmaps[j_pos - 1])
                elif st == D and i_pos > 0:
                    if i_pos - 1 < len(left_lmaps):
                        lmap.update(left_lmaps[i_pos - 1])
                parent_lmaps.append(lmap)
            node_leaf_maps[id(node)] = parent_lmaps
            path = viterbi_path
        else:
            log_prob, path, parent_profile = viterbi_profile(
                log_trans, state_types, profile_x, profile_y,
                sub_left, sub_right, pi, is_root=is_root)
            fwd_log_prob = log_prob
            node_profiles[id(node)] = parent_profile
            rec_positions = []
            parent_lmaps = []
            for i_pos, j_pos, st in path:
                if st == M or st == D:
                    rec_positions.append((i_pos, j_pos, st))
                    lmap = {}
                    if st == M and i_pos > 0 and j_pos > 0:
                        if i_pos - 1 < len(left_lmaps):
                            lmap.update(left_lmaps[i_pos - 1])
                        if j_pos - 1 < len(right_lmaps):
                            lmap.update(right_lmaps[j_pos - 1])
                    elif st == D and i_pos > 0:
                        if i_pos - 1 < len(left_lmaps):
                            lmap.update(left_lmaps[i_pos - 1])
                    parent_lmaps.append(lmap)
            node_leaf_maps[id(node)] = parent_lmaps
        # Convert profile to one-hot sequence if single_seq_ancestors mode
        if single_seq_ancestors and id(node) in node_profiles:
            prof = node_profiles[id(node)]
            if prof.shape[0] > 0:
                seq_idx = np.argmax(prof, axis=1)
                node_profiles[id(node)] = np.eye(prof.shape[1])[seq_idx]

        node_alignments[id(node)] = {
            "left_child": left.name or f"node_{id(left)}",
            "right_child": right.name or f"node_{id(right)}",
            "path": path,
            "rec_positions": rec_positions,
            "log_prob": log_prob,
            "fwd_log_prob": fwd_log_prob,
            "t_left": t_left,
            "t_right": t_right,
        }

    # Extract MAP root sequence from root profile
    root_profile = node_profiles.get(id(tree_root))
    if root_profile is not None and len(root_profile) > 0:
        # Weight by pi and take argmax
        weighted = np.asarray(pi)[None, :] * root_profile
        root_sequence = np.argmax(weighted, axis=1).astype(np.int32)
    else:
        root_sequence = np.array([], dtype=np.int32)

    return node_profiles, node_alignments, root_sequence



def extract_msa(tree_root, leaf_seqs, node_alignments, node_profiles):
    """Extract a multiple sequence alignment from progressive reconstruction.

    Uses recursive merge with profile-position tracking. Each internal node
    stores rec_positions mapping each profile position to (left_child_i,
    right_child_j, state_type) and a leaf_maps list mapping each profile
    position to {leaf_name: seq_position}.

    The Viterbi path at each node determines MSA column ordering. When
    recognizer sampling adds extra profile positions beyond the Viterbi
    path, these are mapped to MSA columns via their leaf_maps, which
    provide a direct leaf-to-position mapping for each recognizer state.

    Args:
        tree_root: TreeNode
        leaf_seqs: dict of {leaf_name: integer_array}
        node_alignments: from reconstruct_progressive_felsenstein
        node_profiles: from reconstruct_progressive_felsenstein

    Returns:
        msa: dict of {leaf_name: list_of_chars_or_gaps} where gaps are -1
        msa_length: total alignment length
    """

    def _get_msa_data(node):
        """Return (columns, profile_indices, leaf_maps).

        columns: list of column dicts (full MSA for this subtree)
        profile_indices: list of indices into columns for profile positions.
            len(profile_indices) == len(node's profile/rec_positions).
        leaf_maps: list of dicts {leaf_name: seq_pos} for each profile pos.
        """
        if node.is_leaf:
            if node.name in leaf_seqs:
                seq = leaf_seqs[node.name]
                cols = [{node.name: int(seq[i])} for i in range(len(seq))]
                lmaps = [{node.name: i} for i in range(len(seq))]
                return cols, list(range(len(cols))), lmaps
            return [], [], []

        aln_info = node_alignments.get(id(node))
        if aln_info is None:
            for c in node.children:
                result = _get_msa_data(c)
                if result[0]:
                    return result
            return [], [], []

        # Find children
        left_child = right_child = None
        for c in node.children:
            name = c.name or f"node_{id(c)}"
            if name == aln_info["left_child"]:
                left_child = c
            elif name == aln_info["right_child"]:
                right_child = c

        if left_child is None or right_child is None:
            return [], [], []

        left_cols, left_prof_idx, left_lmaps = _get_msa_data(left_child)
        right_cols, right_prof_idx, right_lmaps = _get_msa_data(right_child)

        path = aln_info["path"]
        rec_positions = aln_info.get("rec_positions")

        # Walk the Viterbi path to build MSA columns in correct order.
        # This handles insertion interleaving correctly.
        merged = []
        left_col_pos = 0
        right_col_pos = 0

        # Map (i_pos, j_pos, st) -> MSA column index for path M/D states
        position_to_col = {}
        # Map left_child_prof_pos -> MSA column index
        left_pos_to_col = {}
        # Viterbi profile indices (M/D states from path)
        viterbi_prof_idx = []

        for i_pos, j_pos, st in path:
            if st == S:
                continue

            if st == M:
                lp, rp = i_pos - 1, j_pos - 1
                # Emit left insertion columns up to this position
                if lp < len(left_prof_idx):
                    target = left_prof_idx[lp]
                    while left_col_pos < target:
                        merged.append(dict(left_cols[left_col_pos]))
                        left_col_pos += 1
                # Emit right insertion columns up to this position
                if rp < len(right_prof_idx):
                    target = right_prof_idx[rp]
                    while right_col_pos < target:
                        merged.append(dict(right_cols[right_col_pos]))
                        right_col_pos += 1
                # Merge match column
                col = {}
                if lp < len(left_prof_idx):
                    col.update(left_cols[left_prof_idx[lp]])
                    left_col_pos = max(left_col_pos, left_prof_idx[lp] + 1)
                if rp < len(right_prof_idx):
                    col.update(right_cols[right_prof_idx[rp]])
                    right_col_pos = max(right_col_pos, right_prof_idx[rp] + 1)
                col_idx = len(merged)
                merged.append(col)
                viterbi_prof_idx.append(col_idx)
                position_to_col[(i_pos, j_pos, st)] = col_idx
                left_pos_to_col[lp] = col_idx

            elif st == D:
                lp = i_pos - 1
                if lp < len(left_prof_idx):
                    target = left_prof_idx[lp]
                    while left_col_pos < target:
                        merged.append(dict(left_cols[left_col_pos]))
                        left_col_pos += 1
                    col = dict(left_cols[left_prof_idx[lp]])
                    left_col_pos = max(left_col_pos, left_prof_idx[lp] + 1)
                else:
                    col = {}
                col_idx = len(merged)
                merged.append(col)
                viterbi_prof_idx.append(col_idx)
                position_to_col[(i_pos, j_pos, st)] = col_idx
                left_pos_to_col[lp] = col_idx

            elif st == I:
                rp = j_pos - 1
                if rp < len(right_prof_idx):
                    target = right_prof_idx[rp]
                    while right_col_pos < target:
                        merged.append(dict(right_cols[right_col_pos]))
                        right_col_pos += 1
                    col = dict(right_cols[right_prof_idx[rp]])
                    right_col_pos = max(right_col_pos, right_prof_idx[rp] + 1)
                else:
                    col = {}
                merged.append(col)
                # I states don't contribute to profile

        # Output remaining child columns
        while left_col_pos < len(left_cols):
            merged.append(dict(left_cols[left_col_pos]))
            left_col_pos += 1
        while right_col_pos < len(right_cols):
            merged.append(dict(right_cols[right_col_pos]))
            right_col_pos += 1

        # Build leaf_maps and profile_indices for rec_positions.
        if rec_positions is not None:
            # Build leaf_maps: each rec_position maps to a merged
            # dict of child leaf_maps.
            leaf_maps = []
            profile_idx = []
            for i_pos, j_pos, st in rec_positions:
                lmap = {}
                if st == M:
                    lp, rp = i_pos - 1, j_pos - 1
                    if lp < len(left_lmaps):
                        lmap.update(left_lmaps[lp])
                    if rp < len(right_lmaps):
                        lmap.update(right_lmaps[rp])
                elif st == D:
                    lp = i_pos - 1
                    if lp < len(left_lmaps):
                        lmap.update(left_lmaps[lp])
                leaf_maps.append(lmap)

                # Map to MSA column: try exact match first,
                # then fall back to left child position.
                col = position_to_col.get((i_pos, j_pos, st))
                if col is not None:
                    profile_idx.append(col)
                else:
                    lp = i_pos - 1
                    col = left_pos_to_col.get(lp)
                    if col is not None:
                        profile_idx.append(col)
                    elif left_pos_to_col:
                        closest = min(left_pos_to_col.keys(),
                                      key=lambda k: abs(k - lp))
                        profile_idx.append(left_pos_to_col[closest])
                    else:
                        profile_idx.append(0)
        else:
            # No rec_positions: use Viterbi path M/D states
            profile_idx = viterbi_prof_idx
            # Build leaf_maps from path
            leaf_maps = []
            for i_pos, j_pos, st in path:
                if st == M or st == D:
                    lmap = {}
                    lp = i_pos - 1
                    if st == M:
                        rp = j_pos - 1
                        if lp < len(left_lmaps):
                            lmap.update(left_lmaps[lp])
                        if rp < len(right_lmaps):
                            lmap.update(right_lmaps[rp])
                    elif st == D:
                        if lp < len(left_lmaps):
                            lmap.update(left_lmaps[lp])
                    leaf_maps.append(lmap)

        return merged, profile_idx, leaf_maps

    columns, _, _ = _get_msa_data(tree_root)
    msa_length = len(columns)

    # Convert columns to rows
    msa = {}
    for node in tree_root.preorder():
        if node.is_leaf and node.name in leaf_seqs:
            msa[node.name] = [col.get(node.name, -1) for col in columns]

    return msa, msa_length


def extract_full_msa(tree_root, leaf_seqs, node_alignments, node_profiles):
    """Extract MSA with rows and presence arrays for ALL nodes (leaves + internal).

    Like extract_msa but also tracks internal node presence and builds
    internal node MSA rows from profile argmax. Internal node presence
    is derived directly from the column merging walk — at each internal
    node, M/D states in the Viterbi path (or rec_positions if available)
    correspond to columns where that node is present.

    Args:
        tree_root: TreeNode
        leaf_seqs: dict of {leaf_name: integer_array}
        node_alignments: from reconstruct_progressive_felsenstein or
            reconstruct_with_triad
        node_profiles: from the same reconstruction (keyed by id(node))

    Returns:
        dict with keys:
            'msa': {name: np.array(int) for ALL nodes} where gaps are -1.
                For leaves, the observed sequence character or -1.
                For internal nodes, argmax of CL profile at present positions,
                -1 at absent positions.
            'presence': {name: np.array(bool) for ALL nodes}
            'length': int — MSA column count
    """
    import numpy as np

    S_ST, M_ST, I_ST, D_ST, E_ST = 0, 1, 2, 3, 4

    # node_present_cols: maps id(node) -> set of MSA column indices
    # where that node is present.
    node_present_cols = {}

    def _get_msa_data(node):
        """Return (columns, profile_indices, leaf_maps).

        Same logic as extract_msa._get_msa_data, but also populates
        node_present_cols as a side effect.

        columns: list of column dicts (full MSA for this subtree)
        profile_indices: list of indices into columns for profile positions.
            len(profile_indices) == len(node's profile/rec_positions).
        leaf_maps: list of dicts {leaf_name: seq_pos} for each profile pos.
        """
        if node.is_leaf:
            if node.name in leaf_seqs:
                seq = leaf_seqs[node.name]
                cols = [{node.name: int(seq[i])} for i in range(len(seq))]
                lmaps = [{node.name: i} for i in range(len(seq))]
                # Leaf is present at all its columns
                node_present_cols[id(node)] = set(range(len(cols)))
                return cols, list(range(len(cols))), lmaps
            node_present_cols[id(node)] = set()
            return [], [], []

        aln_info = node_alignments.get(id(node))
        if aln_info is None:
            for c in node.children:
                result = _get_msa_data(c)
                if result[0]:
                    # Pass-through: this node is present wherever the child is
                    node_present_cols[id(node)] = set(node_present_cols.get(id(c), set()))
                    return result
            node_present_cols[id(node)] = set()
            return [], [], []

        # Find children
        left_child = right_child = None
        for c in node.children:
            name = c.name or f"node_{id(c)}"
            if name == aln_info["left_child"]:
                left_child = c
            elif name == aln_info["right_child"]:
                right_child = c

        if left_child is None or right_child is None:
            node_present_cols[id(node)] = set()
            return [], [], []

        left_cols, left_prof_idx, left_lmaps = _get_msa_data(left_child)
        right_cols, right_prof_idx, right_lmaps = _get_msa_data(right_child)

        # Remap child present_cols: they are relative to child's own merged
        # columns. We need to remap them after building our merged columns.
        # We'll track the mapping from child column indices to merged indices.
        left_col_remap = {}  # child_col_idx -> merged_col_idx
        right_col_remap = {}

        path = aln_info["path"]
        rec_positions = aln_info.get("rec_positions")

        # Walk the Viterbi path to build MSA columns in correct order.
        merged = []
        left_col_pos = 0
        right_col_pos = 0

        position_to_col = {}
        left_pos_to_col = {}
        viterbi_prof_idx = []
        # Track which merged columns this internal node is present at
        this_node_present = set()

        for i_pos, j_pos, st in path:
            if st == S_ST:
                continue

            if st == M_ST:
                lp, rp = i_pos - 1, j_pos - 1
                if lp < len(left_prof_idx):
                    target = left_prof_idx[lp]
                    while left_col_pos < target:
                        idx = len(merged)
                        left_col_remap[left_col_pos] = idx
                        merged.append(dict(left_cols[left_col_pos]))
                        left_col_pos += 1
                if rp < len(right_prof_idx):
                    target = right_prof_idx[rp]
                    while right_col_pos < target:
                        idx = len(merged)
                        right_col_remap[right_col_pos] = idx
                        merged.append(dict(right_cols[right_col_pos]))
                        right_col_pos += 1
                col = {}
                if lp < len(left_prof_idx):
                    left_col_remap[left_prof_idx[lp]] = len(merged)
                    col.update(left_cols[left_prof_idx[lp]])
                    left_col_pos = max(left_col_pos, left_prof_idx[lp] + 1)
                if rp < len(right_prof_idx):
                    right_col_remap[right_prof_idx[rp]] = len(merged)
                    col.update(right_cols[right_prof_idx[rp]])
                    right_col_pos = max(right_col_pos, right_prof_idx[rp] + 1)
                col_idx = len(merged)
                merged.append(col)
                viterbi_prof_idx.append(col_idx)
                position_to_col[(i_pos, j_pos, st)] = col_idx
                left_pos_to_col[lp] = col_idx

            elif st == D_ST:
                lp = i_pos - 1
                if lp < len(left_prof_idx):
                    target = left_prof_idx[lp]
                    while left_col_pos < target:
                        idx = len(merged)
                        left_col_remap[left_col_pos] = idx
                        merged.append(dict(left_cols[left_col_pos]))
                        left_col_pos += 1
                    left_col_remap[left_prof_idx[lp]] = len(merged)
                    col = dict(left_cols[left_prof_idx[lp]])
                    left_col_pos = max(left_col_pos, left_prof_idx[lp] + 1)
                else:
                    col = {}
                col_idx = len(merged)
                merged.append(col)
                viterbi_prof_idx.append(col_idx)
                position_to_col[(i_pos, j_pos, st)] = col_idx
                left_pos_to_col[lp] = col_idx

            elif st == I_ST:
                rp = j_pos - 1
                if rp < len(right_prof_idx):
                    target = right_prof_idx[rp]
                    while right_col_pos < target:
                        idx = len(merged)
                        right_col_remap[right_col_pos] = idx
                        merged.append(dict(right_cols[right_col_pos]))
                        right_col_pos += 1
                    right_col_remap[right_prof_idx[rp]] = len(merged)
                    col = dict(right_cols[right_prof_idx[rp]])
                    right_col_pos = max(right_col_pos, right_prof_idx[rp] + 1)
                else:
                    col = {}
                merged.append(col)
                # I states don't contribute to profile

        # Output remaining child columns
        while left_col_pos < len(left_cols):
            left_col_remap[left_col_pos] = len(merged)
            merged.append(dict(left_cols[left_col_pos]))
            left_col_pos += 1
        while right_col_pos < len(right_cols):
            right_col_remap[right_col_pos] = len(merged)
            merged.append(dict(right_cols[right_col_pos]))
            right_col_pos += 1

        # Remap children's present_cols to merged column indices
        left_child_present = node_present_cols.get(id(left_child), set())
        right_child_present = node_present_cols.get(id(right_child), set())
        remapped_left = {left_col_remap[c] for c in left_child_present
                         if c in left_col_remap}
        remapped_right = {right_col_remap[c] for c in right_child_present
                          if c in right_col_remap}
        node_present_cols[id(left_child)] = remapped_left
        node_present_cols[id(right_child)] = remapped_right

        # Determine which merged columns this node is present at.
        # Use rec_positions if available (triad-filtered), otherwise M/D from path.
        if rec_positions is not None:
            for i_pos, j_pos, st in rec_positions:
                col = position_to_col.get((i_pos, j_pos, st))
                if col is not None:
                    this_node_present.add(col)
                else:
                    lp = i_pos - 1
                    col = left_pos_to_col.get(lp)
                    if col is not None:
                        this_node_present.add(col)
        else:
            # All M/D states from path = this node is present
            for i_pos, j_pos, st in path:
                if st == M_ST or st == D_ST:
                    col = position_to_col.get((i_pos, j_pos, st))
                    if col is not None:
                        this_node_present.add(col)

        node_present_cols[id(node)] = this_node_present

        # Build leaf_maps and profile_indices for rec_positions.
        if rec_positions is not None:
            leaf_maps = []
            profile_idx = []
            for i_pos, j_pos, st in rec_positions:
                lmap = {}
                if st == M_ST:
                    lp, rp = i_pos - 1, j_pos - 1
                    if lp < len(left_lmaps):
                        lmap.update(left_lmaps[lp])
                    if rp < len(right_lmaps):
                        lmap.update(right_lmaps[rp])
                elif st == D_ST:
                    lp = i_pos - 1
                    if lp < len(left_lmaps):
                        lmap.update(left_lmaps[lp])
                leaf_maps.append(lmap)

                col = position_to_col.get((i_pos, j_pos, st))
                if col is not None:
                    profile_idx.append(col)
                else:
                    lp = i_pos - 1
                    col = left_pos_to_col.get(lp)
                    if col is not None:
                        profile_idx.append(col)
                    elif left_pos_to_col:
                        closest = min(left_pos_to_col.keys(),
                                      key=lambda k: abs(k - lp))
                        profile_idx.append(left_pos_to_col[closest])
                    else:
                        profile_idx.append(0)
        else:
            profile_idx = viterbi_prof_idx
            leaf_maps = []
            for i_pos, j_pos, st in path:
                if st == M_ST or st == D_ST:
                    lmap = {}
                    lp = i_pos - 1
                    if st == M_ST:
                        rp = j_pos - 1
                        if lp < len(left_lmaps):
                            lmap.update(left_lmaps[lp])
                        if rp < len(right_lmaps):
                            lmap.update(right_lmaps[rp])
                    elif st == D_ST:
                        if lp < len(left_lmaps):
                            lmap.update(left_lmaps[lp])
                    leaf_maps.append(lmap)

        return merged, profile_idx, leaf_maps

    columns, _, _ = _get_msa_data(tree_root)
    msa_length = len(columns)

    # Build MSA rows for all nodes
    msa = {}
    presence = {}

    for node in tree_root.preorder():
        if node.name is None:
            continue

        if node.is_leaf and node.name in leaf_seqs:
            # Leaf: observed characters or gap
            msa[node.name] = np.array(
                [col.get(node.name, -1) for col in columns], dtype=np.int32)
            presence[node.name] = np.array(
                [col.get(node.name, -1) >= 0 for col in columns], dtype=bool)
        else:
            # Internal node (or leaf without sequence): use presence from walk
            present_set = node_present_cols.get(id(node), set())
            pres_arr = np.zeros(msa_length, dtype=bool)
            for c in present_set:
                if c < msa_length:
                    pres_arr[c] = True
            presence[node.name] = pres_arr

            # Build MSA row: argmax of profile at present positions, -1 elsewhere
            profile = node_profiles.get(id(node))
            row = np.full(msa_length, -1, dtype=np.int32)
            if profile is not None and len(profile) > 0:
                # Map profile positions to MSA columns.
                # The profile has one entry per rec_position (or per M/D in path).
                # We need to map profile index -> MSA column.
                # The present_set gives us which columns, but we need the order.
                present_cols_sorted = sorted(present_set)
                n_prof = len(profile)
                n_present = len(present_cols_sorted)
                # Profile length should match number of present columns
                if n_prof == n_present:
                    for pi_idx, col_idx in enumerate(present_cols_sorted):
                        if col_idx < msa_length:
                            row[col_idx] = int(np.argmax(profile[pi_idx]))
                elif n_prof > 0:
                    # Fallback: fill as many as we can
                    for pi_idx in range(min(n_prof, n_present)):
                        col_idx = present_cols_sorted[pi_idx]
                        if col_idx < msa_length:
                            row[col_idx] = int(np.argmax(profile[pi_idx]))
            msa[node.name] = row

    return {
        'msa': msa,
        'presence': presence,
        'length': msa_length,
    }
