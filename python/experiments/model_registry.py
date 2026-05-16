"""Registry of trained MixDom model variants.

Provides a single source of truth for model paths, parameters, and metadata.
Used by benchmark scripts to avoid hardcoded paths and env-var overrides.

Usage:
    from experiments.model_registry import get_model, list_models

    params, n_dom, n_frag, fsa_params = get_model('d3f1')
    params, n_dom, n_frag, fsa_params = get_model('d3f1_unaligned')
"""

import os
import numpy as np

_PYTHON_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model definitions: key → (checkpoint_path, description, format)
# format: 'mixdom1' (scalar ext, no classdist) or 'mixdom2' (F×F ext_matrix, classdist)
MODEL_DEFS = {
    # --- MixDom1 models (DEPRECATED: use converted MixDom2 versions) ---
    'd3f1': ('pfam/svi_bw_d3f1_full_best_val.npz', '3-domain 1-frag, Pfam seed aligned [MixDom1]', 'mixdom1'),
    'd5f1': ('pfam/svi_bw_d5f1_full_best_val.npz', '5-domain 1-frag, Pfam seed aligned [MixDom1]', 'mixdom1'),
    'd1f1': ('pfam/svi_bw_d1f1_full_best_val.npz', '1-domain 1-frag, Pfam seed aligned [MixDom1]', 'mixdom1'),
    'd2f1': ('pfam/svi_bw_d2f1_full_best_val.npz', '2-domain 1-frag, Pfam seed aligned [MixDom1]', 'mixdom1'),
    'd10f1': ('pfam/svi_bw_d10f1_full_best_val.npz', '10-domain 1-frag, Pfam seed aligned [MixDom1]', 'mixdom1'),
    'd3f2': ('pfam/svi_bw_d3f2_full_best_val.npz', '3-domain 2-frag, Pfam seed aligned [MixDom1]', 'mixdom1'),

    # Training data variants (MixDom1)
    'd3f1_pfam_full': ('pfam/svi_bw_d3f1_pfam_full_best_val.npz', '3-domain 1-frag, Pfam full [MixDom1]', 'mixdom1'),
    'd3f1_panther': ('pfam/svi_bw_d3f1_panther_best_val.npz', '3-domain 1-frag, PANTHER [MixDom1]', 'mixdom1'),

    # Training modality variants (MixDom1)
    'd3f1_unaligned': ('pfam/svi_bw_d3f1_unaligned_best_val.npz', '3-domain 1-frag, unaligned [MixDom1]', 'mixdom1'),
    'd3f1_matchaligned': ('pfam/svi_bw_d3f1_matchaligned_best_val.npz', '3-domain 1-frag, match-aligned [MixDom1]', 'mixdom1'),
    'd5f1_matchaligned': ('pfam/svi_bw_d5f1_matchaligned_best_val.npz', '5-domain 1-frag, match-aligned [MixDom1]', 'mixdom1'),

    # --- MixDom2 converted models (from MixDom1 via convert_to_mixdom2.py) ---
    # These use the unified D×F partition reconstruction with per-class substitution.
    'd1f1c1': ('pfam/d1f1c1.npz', '1-domain 1-frag 1-class (converted)', 'mixdom2'),
    'd2f1c2': ('pfam/d2f1c2.npz', '2-domain 1-frag 2-class (converted)', 'mixdom2'),
    'd3f1c3': ('pfam/d3f1c3.npz', '3-domain 1-frag 3-class (converted)', 'mixdom2'),
    'd5f1c5': ('pfam/d5f1c5.npz', '5-domain 1-frag 5-class (converted)', 'mixdom2'),

    # --- MixDom2 native models (trained from random seed with F>1) ---
    'd3f2_fresh': ('pfam/svi_bw_d3f2_fresh.npz', '3-domain 2-frag (fresh, Pfam seed)', 'mixdom2'),
}


def _resolve_path(rel_path):
    """Resolve a path relative to the python/ directory."""
    return os.path.join(_PYTHON_ROOT, rel_path)


def list_models(available_only=True, format_filter=None):
    """List all registered models.

    Args:
        available_only: skip missing checkpoints.
        format_filter: 'mixdom1', 'mixdom2', or None for all.
    """
    result = []
    for key, entry in sorted(MODEL_DEFS.items()):
        path, desc = entry[0], entry[1]
        fmt = entry[2] if len(entry) > 2 else 'mixdom1'
        if format_filter and fmt != format_filter:
            continue
        full_path = _resolve_path(path)
        exists = os.path.exists(full_path)
        if not available_only or exists:
            result.append((key, desc, exists, fmt))
    return result


def get_model(key):
    """Load a model by key.

    Returns:
        params: raw params dict from load_params
        n_dom: number of domains
        n_frag: number of fragments
        fsa_params: dict ready for fsa_align (with S_exch, pi, Q keys)
    """
    if key not in MODEL_DEFS:
        available = [k for k, _, _, _ in list_models()]
        raise KeyError(f"Unknown model '{key}'. Available: {available}")

    entry = MODEL_DEFS[key]
    rel_path, desc = entry[0], entry[1]
    fmt = entry[2] if len(entry) > 2 else 'mixdom1'
    if fmt == 'mixdom1':
        import warnings
        warnings.warn(
            f"Model '{key}' is MixDom1 format. Consider using the MixDom2 "
            f"converted version (run convert_to_mixdom2.py).",
            DeprecationWarning, stacklevel=2)
    full_path = _resolve_path(rel_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model '{key}' checkpoint not found: {full_path}")

    from tkfmixdom.jax.distill.maraschino import load_params, build_rate_matrix
    import jax.numpy as jnp

    params, n_dom, n_cls = load_params(full_path)
    S_exch = np.asarray(params['S_exch'])
    pis = np.asarray(params['pi'])
    pis = pis / pis.sum(axis=-1, keepdims=True)
    avg_pi = np.einsum('n,na->a', np.asarray(params['v']), pis)
    avg_pi = avg_pi / avg_pi.sum()
    avg_S = np.einsum('n,nab->ab', np.asarray(params['v']), S_exch)

    fw = np.asarray(params['frag_weights'])
    n_frag = fw.shape[1] if fw.ndim > 1 else 1

    fsa_params = {
        'main_ins': float(params['lam0']),
        'main_del': float(params['mu0']),
        'dom_ins': np.asarray(params['lam']),
        'dom_del': np.asarray(params['mu']),
        'dom_weights': np.asarray(params['v']),
        'frag_weights': fw,
        'ext_rates': np.asarray(params['r_frags']),
        'S_exch': S_exch,
        'pi': pis,
        'Q': np.asarray(build_rate_matrix(jnp.array(avg_S), jnp.array(avg_pi))),
    }

    return params, n_dom, n_frag, fsa_params


def get_partition_model(key):
    """Load a model and build a PartitionReconModel for partition reconstruction."""
    from experiments.partition_recon_adapter import mixdom_model_from_params
    params, n_dom, n_frag, _ = get_model(key)
    return mixdom_model_from_params(params)
