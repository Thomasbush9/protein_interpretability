"""`pi_protocol` now lives at `protein_interpretability.experiments.protocol`.

This is an alias, not a copy. It keeps the ~60 scripts in this directory
importing `pi_protocol` unchanged while the package becomes the real home, so the
promotion can be verified against the archived results before any call site is
touched.

The `src` directory is located the way `artifacts` locates the repository, and
for the same reason: jobs execute from `prot_interp_files/harness/`, a plain
COPY of this directory rather than a checkout, so a path relative to `__file__`
resolves to somewhere that does not exist. Falling back to the absolute
checkout is what makes this file work in both places.
"""

import sys
from pathlib import Path

_REPO = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/"
             "protein_interpretability")
_SRC = Path(__file__).resolve().parent.parent / "src"
if not (_SRC / "protein_interpretability").is_dir():
    _SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import protein_interpretability.experiments.protocol as _module  # noqa: E402

# Replacing the module object rather than re-exporting names: `import *` would
# drop underscore-prefixed module state, and every call site reaches these
# through the module (`pi_stats.spearman`), so the alias has to be total.
sys.modules[__name__] = _module
