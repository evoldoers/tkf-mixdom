"""Statistical progressive alignment under a K-class profile mixture and MixFrag.

Implements analysis/profile_elbo/profile_classes.tex:
  model      substitution (LG exchangeabilities + profile-mixture classes) and
             indel (MixFrag / TKF92) models
  wavefront  anti-diagonal 2D Pair HMM dynamic programming (Forward, Viterbi)
  distance   pairwise times by one EM iteration (Forward gradients + Newton)
  tree       BioNJ and midpoint rooting
  profile    column summaries, the O(K+A) pair score, Merge/Extend/Reestimate
  align      the progressive aligner
"""
import os
import sys
from pathlib import Path

_TKFMIXDOM = Path(os.environ.get("TKFMIXDOM_PYTHON", Path(__file__).resolve().parents[4]))   # tkf-mixdom/python
if str(_TKFMIXDOM) not in sys.path:
    sys.path.insert(0, str(_TKFMIXDOM))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
