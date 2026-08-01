"""Add tmtools TM-scores to the benchmark table, from saved coordinates.

Runs in the analysis venv (tmtools is not in the mosaic container). TM is
computed per diffusion sample against every wild-type sample, so the
within-wild-type spread is available as an explicit noise floor rather than
assumed small -- the mistake the previous hand-rolled TM-score hid.
"""
from __future__ import annotations
import argparse, csv, itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from geom import tm_and_rmsd

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True); ap.add_argument("--coords", required=True)
ap.add_argument("--data", required=True); ap.add_argument("--wt", default="gfp_wt")
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = json.loads(Path(a.bench).read_text())
d = np.load(a.coords)
seqs = {}
for y in (Path(a.data) / "yamls").glob("*.yaml"):
    seqs[y.stem] = y.read_text().split("sequence:")[1].split()[0]

wt = d[a.wt]
w_within = [tm_and_rmsd(wt[i], wt[j], seqs[a.wt], seqs[a.wt])[0]
            for i, j in itertools.combinations(range(len(wt)), 2)]
floor = float(np.mean(w_within)) if w_within else 1.0
print(f"noise floor: within-{a.wt} mean pairwise TM = {floor:.4f} "
      f"(n={len(w_within)} pairs)\n")

for r in rows:
    cid = r["id"]
    if cid not in d:
        continue
    vals = [tm_and_rmsd(d[cid][i], wt[j], seqs[cid], seqs[a.wt])
            for i in range(len(d[cid])) for j in range(len(wt))]
    tms = [v[0] for v in vals]; rms = [v[1] for v in vals]
    r["tm_to_wt"] = float(np.mean(tms)); r["tm_sd"] = float(np.std(tms))
    r["rmsd_to_wt"] = float(np.mean(rms))
r_sorted = sorted(rows, key=lambda r: (r["mode"], r["n_mut"]))
print(f"{'id':18s} {'mode':9s} {'%mut':>6s} {'TM':>7s} {'RMSD':>6s} {'pLDDT':>7s} {'KL':>7s}")
print("-" * 68)
for r in r_sorted:
    print(f"{r['id']:18s} {r['mode']:9s} {r['pct_mut']:6.1f} "
          f"{r.get('tm_to_wt', float('nan')):7.4f} {r.get('rmsd_to_wt', float('nan')):6.2f} "
          f"{r['plddt']:7.3f} {r['kl']:7.4f}")
Path(a.out).write_text(json.dumps(rows, indent=2))
print(f"\nwrote {a.out}")
