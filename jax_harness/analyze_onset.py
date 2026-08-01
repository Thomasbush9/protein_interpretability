"""Does the mutation propagate through the contact graph, or appear everywhere at once?

The residue x layer matrix shows bands at the mutated residues from layer 0, and
a second set of bands appearing only after ~L45 at positions that are far in
sequence. The hypothesis is that those late bands are residues *spatially* close
to the mutation in the folded structure, reached by the triangle operations
after a few rounds of message passing. If so, a residue's onset depth should
track its 3-D distance from the nearest mutated residue.

Onset is the divergence-weighted mean layer:

    onset(r) = sum_L  L * dKL(r, L)  /  sum_L dKL(r, L)      dKL = max(diff, 0)

i.e. the average depth at which residue r's divergence actually accumulated.
Low = the signal was there early; high = it arrived late. This is preferable to
a threshold crossing, which depends on an arbitrary cut and is noisy for
residues whose divergence is small.

Controls that matter:
  - sequence distance |i - nearest mutated i| as a rival predictor. If onset
    tracks sequence distance just as well, nothing has been shown about
    3-D propagation -- spatial and sequence proximity are correlated.
  - partial correlation of onset with spatial distance, controlling for
    sequence distance, is therefore the load-bearing number.
  - the surface mutant, whose substitutions are exposed and should propagate
    less far, as a comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import parse_cif_ca  # noqa: E402


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def partial_spearman(x, y, z):
    """Spearman(x, y) controlling for z, via ranks + linear residualisation."""
    r = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731
    rx, ry, rz = r(x), r(y), r(z)

    def resid(a, b):
        b1 = np.stack([b, np.ones_like(b)], 1)
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef

    return spearman(resid(rx, rz), resid(ry, rz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cif", required=True, help="WT predicted structure")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.loads(Path(args.matrix).read_text())
    manifest = {r["id"]: r for r in csv.DictReader((Path(args.data) / "manifest.csv").open())}
    seq, coords = parse_cif_ca(Path(args.cif))
    residues = np.array(d["residues"])          # token indices, 0-based
    N, L = len(residues), d["n_layers"]

    if len(coords) < residues.max() + 1:
        raise SystemExit(f"cif has {len(coords)} CA but matrix references index {residues.max()}")
    xyz = coords[residues]

    out = {}
    print(f"{'mutant':16s} {'rho(onset,3D)':>14s} {'rho(onset,seq)':>15s} "
          f"{'partial 3D|seq':>15s} {'n_far':>6s}")
    print("-" * 72)

    for mid, rec in d["mutants"].items():
        M = np.array(rec["residue_by_layer"])          # [N, L]
        mut_rows = np.array(rec.get("mutated_rows", []), dtype=int)
        if mut_rows.size == 0:
            continue

        dK = np.diff(M, axis=1)
        dK = np.clip(dK, 0, None)                       # accumulation only
        w = dK.sum(1)
        onset = (dK * np.arange(1, L)).sum(1) / np.maximum(w, 1e-12)

        d3 = np.linalg.norm(xyz[:, None] - xyz[None, mut_rows], axis=-1).min(1)
        dseq = np.abs(residues[:, None] - residues[mut_rows][None]).min(1)

        # exclude the mutated residues themselves: their onset is trivially early
        # and they would dominate both correlations
        sel = d3 > 0
        rho3 = spearman(onset[sel], d3[sel])
        rhos = spearman(onset[sel], dseq[sel])
        par = partial_spearman(onset[sel], d3[sel], dseq[sel])

        out[mid] = {
            "rho_onset_vs_3d": rho3, "rho_onset_vs_seq": rhos,
            "partial_3d_controlling_seq": par,
            "onset": onset.tolist(), "dist_3d": d3.tolist(),
            "dist_seq": dseq.tolist(), "n_non_mutated": int(sel.sum()),
        }
        print(f"{mid:16s} {rho3:14.3f} {rhos:15.3f} {par:15.3f} {int(sel.sum()):6d}")

        # onset by distance shell, the readable version of the same thing
        shells = [(0, 8), (8, 12), (12, 16), (16, 24), (24, 100)]
        parts = []
        for lo, hi in shells:
            m = sel & (d3 >= lo) & (d3 < hi)
            if m.sum() >= 5:
                parts.append(f"{lo}-{hi}A: L{onset[m].mean():.1f} (n={int(m.sum())})")
        print("      onset by shell  " + "   ".join(parts))

    Path(args.out).write_text(json.dumps(out))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
