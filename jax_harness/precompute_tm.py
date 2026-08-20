"""Cache per-variant TM-score to wild type, so the container can read it.

`tmtools` lives in the repository venv and JAX lives in the mosaic container,
and neither has the other. Every analysis so far has needed only one of the two,
so the split never mattered; `analyze_symmetry` needs both, because it runs
batched linear algebra AND has to rebuild the published `output_rich` baseline,
whose first column is TM to wild type.

Dropping that column instead would be the wrong shortcut: TM is the strongest
single emitted quantity in the whole comparison, so removing it would weaken the
output side, and it would weaken it in the direction that favours the
conclusion the symmetry test is supposed to be able to refute.

Run this with the venv interpreter, then pass the cache to `analyze_symmetry`:

    .venv/bin/python precompute_tm.py --out runs/tm_cache.npz
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import geom  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    R = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/"
    ap.add_argument("--glob", default=R + "runs/gym2s_*.npz")
    ap.add_argument("--prefix", default="gym2s_",
                    help="filename prefix to strip when keying the cache. The "
                         "key must be the ASSAY id, because that is what the "
                         "consumer looks up -- and the cross-model family is "
                         "named xm_<model>_<run>_<assay>, not <prefix><assay>.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0, out = time.time(), {}
    for f in sorted(glob.glob(a.glob)):
        d = np.load(f, allow_pickle=True)
        # Prefer the assay the capture RECORDS over the one its name implies;
        # the xm family carries it, and a filename is not evidence.
        stem = (str(d["assay"]) if "assay" in d.files
                else Path(f).stem[len(a.prefix):])
        ca_wt = np.asarray(d["ca_wt"], float)
        tm = np.array([geom.tm_score(np.asarray(c, float), ca_wt)
                       for c in d["ca"]])
        out[stem] = tm
        print(f"   {stem.split('_')[0]:8s} n={len(tm):4d}  "
              f"TM mean {tm.mean():.4f}  min {tm.min():.4f}  "
              f"[{time.time() - t0:5.1f}s]", flush=True)

    np.savez_compressed(a.out, **out)
    print(f"\nwrote {a.out}  ({len(out)} assays, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
