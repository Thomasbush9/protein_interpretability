#!/usr/bin/env python3
"""Single-record gradient-attribution launcher (no -m / PYTHONPATH needed).

The boltz env on the cluster doesn't have this project installed, so
``python -m protein_interpretability.attribution.cli`` fails unless
PYTHONPATH includes ``src/``. This wrapper bootstraps sys.path itself and
forwards every CLI flag straight through.

Usage::

    # In the boltz env, no env vars required:
    python scripts/run_attribution_single.py path/to/seq.yaml \\
        --out_dir /tmp/grad_smoke \\
        --cache /n/holylfs06/.../boltz_db \\
        --target contact:65,202 \\
        --target mean_contact \\
        --recycling_steps 0,5,10 \\
        --no_kernels
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from protein_interpretability.attribution.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
