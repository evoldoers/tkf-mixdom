"""Where the paper-1 table and figure scripts write: $PAPER1_DIR, else the manuscript directory
molbiolevol-paper/paper1-baumwelch/ of this repository (dev checkout only; it is not published), else
python/experiments/figs/paper1_out/."""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
_MANUSCRIPT = os.path.join(_REPO, 'molbiolevol-paper', 'paper1-baumwelch')
PAPER1 = os.environ.get('PAPER1_DIR') or (_MANUSCRIPT if os.path.isdir(_MANUSCRIPT) else os.path.join(_HERE, 'paper1_out'))
