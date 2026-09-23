"""Rebuild report_master from the adopted 2026-09-06 evidence release.

Gate 0 of the September publication audit asks for ONE authoritative release:
every manuscript number pointing at a single adopted artifact, produced from an
identifiable committed checkout. `runs/adopted_20260906/` is that release --
thirteen producers rerun clean (git_dirty: false), each reproducing its
predecessor to float noise. This script rebuilds the page against it.

It exists because the rebuild is not one command. Figures must be regenerated
before the builder runs or its staleness guard aborts, and three of the page's
figures are BORROWED from report_svd, so they have to be regenerated in
report_svd rather than here -- the guard checks their mtimes too, and
regenerating heldout_v1.json without redrawing heldout.png is exactly the drift
it was added to catch.

    sbatch checkout.sbatch rebuild_master.py
    sbatch checkout.sbatch rebuild_master.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
ADOPTED = W / "runs/adopted_20260906"
RUNS = W / "runs"


def pick(name):
    """The adopted artifact if the release carries one, else the primary run.

    Stated rather than silent: the page prints which inputs came from the
    adopted release, and a number whose artifact was never regenerated should
    not look as though it was.
    """
    a = ADOPTED / name
    return a if a.exists() else RUNS / name


INPUTS = {k: pick(v) for k, v in {
    "transfer": "transfer_full.json",
    "transfer_ind": "transfer_inductive.json",
    "heldout": "heldout_v1.json",
    "bw": "bw_v1.json",
    "depth": "depth_v1.json",
    "xmodel": "xmodel_v1.json",
    "svd": "svd_dz_v3.json",
    "svd_ds": "svd_ds_v1.json",
    "chem": "chem_v1.json",
    "scrutiny": "scrutiny_v2.json",
    "xio": "xmodel_io_vec.json",
    "layermatch": "layer_match.json",
    "steer": "steer_pooled_v2.json",
    "ablate": "ablate_v2.json",
    "jac": "jac_pooled.json",
    "gate": "gate_probe.json",
    "rotate": "rotate_pooled.json",
    "basis": "basis_depth.json",
}.items()}

# Figures the page owns, and the three it borrows from report_svd. Both lists
# are regenerated here; the builder's guard then finds nothing stale.
FIGURES = [
    ("headline.png", ["fig_headline.py", "--transfer", "{transfer}",
                      "--bw", "{bw}", "--out", "{master}/figures/headline.png"]),
    ("causal.png", ["fig_causal.py", "--steer", "{steer}",
                    "--ablate", "{ablate}",
                    "--out", "{master}/figures/causal.png"]),
    ("xmodel_io.png", ["fig_xmodel_io.py", "--xio", "{xio}",
                       "--out", "{master}/figures/xmodel_io.png"]),
]
BORROWED = [
    ("svd.png", ["fig_svd.py", "--svd", "{svd}", "--svd-ds", "{svd_ds}",
                 "--out", "{svd_dir}/figures/svd.png"]),
    ("heldout.png", ["fig_heldout.py", "--heldout", "{heldout}",
                     "--out", "{svd_dir}/figures/heldout.png"]),
    ("depth.png", ["fig_depth.py", "--depth", "{depth}",
                   "--out", "{svd_dir}/figures/depth.png"]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    fmt = {k: str(v) for k, v in INPUTS.items()}
    fmt["master"] = str(W / "report_master")
    fmt["svd_dir"] = str(W / "report_svd")

    missing = [k for k, v in INPUTS.items() if not v.exists()]
    if missing:
        print(f"missing inputs: {missing}")
        return 1
    n_adopted = sum(1 for v in INPUTS.values() if v.parent == ADOPTED)
    print(f"{n_adopted} of {len(INPUTS)} inputs from the adopted release\n")
    for k, v in sorted(INPUTS.items()):
        tag = "adopted" if v.parent == ADOPTED else "runs"
        print(f"  {k:14s} {tag:8s} {v.name}")

    cmds = [[sys.executable, str(HERE / c[0])] + [x.format(**fmt) for x in c[1:]]
            for _, c in FIGURES + BORROWED]
    cmds.append([sys.executable, str(HERE / "build_master_report.py")]
                + [x for k, v in INPUTS.items()
                   for x in (f"--{k.replace('_', '-')}", str(v))])

    for cmd in cmds:
        print(f"\n$ {' '.join(Path(c).name if '/' in c else c for c in cmd)}",
              flush=True)
        if a.dry_run:
            continue
        r = subprocess.run(cmd, cwd=str(HERE))
        if r.returncode != 0:
            print(f"FAILED: {cmd[1]}")
            return r.returncode
    print("\nreport_master rebuilt" if not a.dry_run else "\n(dry run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
