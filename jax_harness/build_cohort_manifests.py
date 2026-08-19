"""Write down which assays each result was computed over, and check it exists.

Until now a cohort has been whatever a glob matched at the time. `launch_gym2s.sh`
iterates `runs/gym2_*.npz`, so the identity of "the twelve assays" lives in
filenames left behind by a previous sweep -- delete a capture and the cohort
silently shrinks; add an unrelated one and it silently grows. Nothing records
what the set was supposed to be, which means nothing can report that it changed.

These manifests are the record. They are GENERATED from disk rather than typed,
because a hand-written list of twelve accession names is a transcription error
waiting to happen, and then verified against disk on every load -- generating
from the same place you check against is only useful if the check happens later,
on a different day, when something has moved.

Each entry carries the assay's CSV and alignment with sha256, the variant count,
and the wild-type sequence length. Those are what a run needs to fail fast: an
alignment that changed underfoot is the failure this project has already had.

    uv run python jax_harness/build_cohort_manifests.py --out configs/cohorts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
ASSAY_DIR = W / "data/gym/assays/DMS_ProteinGym_substitutions"
PANELS = ["panel2", "panel", "panel3", "panel4"]

# Cohort -> (the capture glob that has been standing in for it, description)
COHORTS = {
    "basis_assays": (
        "gym2s_*.npz",
        "The twelve stability assays the shared PC basis is fitted on. Read by "
        "analyze_svd, analyze_transfer, analyze_chem, analyze_scrutiny and "
        "analyze_attrib."),
    "heldout_assays": (
        "gym3p3_*.npz",
        "Disjoint from basis_assays. Projected onto the FROZEN basis, never "
        "used to fit it -- that disjointness is the whole claim of heldout_v1."),
    "cross_model_assays": (
        "xm_boltz2_r1_*.npz",
        "The four assays run through Boltz-2, OpenFold3 and Protenix on "
        "identical inputs, twice each. Behind xmodel_v1, xmodel_io_vec and "
        "depth_v1."),
    "intervention_assays": (
        "steerall_*.npz",
        "Proteins used for PC2 steering; behind steer_pooled."),
}

STRIP = {"basis_assays": "gym2s_", "heldout_assays": "gym3p3_",
         "cross_model_assays": "xm_boltz2_r1_", "intervention_assays": "steerall_"}

# Cohorts named by an EXPLICIT list rather than by a glob over existing
# captures. Every cohort above is "the assays we already have captures for",
# which is the right definition for describing archived work and the wrong one
# for planning new work: a cohort that can only name what has already been run
# cannot express the run you want next.
#
# `length_ladder` exists because the length question could not be asked. All 217
# ProteinGym substitution assays are on disk, 31 have alignments, and 26 of
# those are Tsuboyama mini-domains under 100 residues -- so every cohort here
# spans 40-118 aa while the median ProteinGym assay is 245. These five are every
# assay with an alignment above 100 residues, and they run 101 -> 403.
#
# COST, MEASURED RATHER THAN ARGUED. The obvious argument -- the Pairformer's
# triangle operations are O(N^3), so a 403-residue protein must be hundreds of
# times a 65-residue one -- is wrong at the sizes we can check. Measured
# 2026-08-19 at 100 variants: protenix costs 8.3 s/variant at 40 aa and
# 10.1 s/variant at 118 aa, a 1.2x increase for a 3x longer protein. Per-variant
# time is dominated by re-parsing the alignment and by the sampler's fixed 200
# steps, both length-independent.
#
# That makes this cohort far cheaper than a cubic estimate suggests, but 403 aa
# is still 3.4x beyond anything measured, and the pair tensor does grow as N^2:
# at 403 residues boltz2's per-layer z stack is ~5.3 GB and the per-layer
# distogram ~2.7 GB, which fits an 80 GB H100 but is where the real constraint
# will show up first. Run PTEN alone before running the cohort.
EXPLICIT = {
    "length_ladder": (
        ["CCDB_ECOLI_Tripathi_2016",        # 101 aa
         "PHOT_CHLRE_Chen_2023",            # 118 aa
         "ESTA_BACSU_Nutschel_2020",        # 212 aa
         "TPMT_HUMAN_Matreyek_2018",        # 245 aa
         "PTEN_HUMAN_Matreyek_2021"],       # 403 aa
        "Every ProteinGym assay with an alignment on disk above 100 residues, "
        "101 to 403 aa. Exists to ask whether the internal-versus-output result "
        "holds on proteins larger than the Tsuboyama mini-domains every other "
        "cohort here is made of. Overlaps heldout_assays by two assays, so use "
        "Cohort.assert_disjoint before making a held-out claim from it."),
}

# The smoke cohort is one assay, chosen for being the smallest thing that is
# still a real assay: ARGR is 69 residues with a 7056-row alignment, so a slice
# over a handful of its variants exercises the whole path in minutes rather
# than hours. It is deliberately a SUBSET of basis_assays, so anything it
# produces can be compared against an archive that already exists.
SMOKE_ASSAY = "ARGR_ECOLI_Tsuboyama_2023_1AOY"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_a3m(assay: str) -> Path | None:
    for panel in PANELS:
        cand = W / "data/gym" / panel / "colabfold_output" / f"{assay}.a3m"
        if cand.exists():
            return cand
    return None


def wt_of(rows: list[dict]) -> str | None:
    """The wild type, recovered by undoing the first variant's own mutation."""
    m = re.match(r"([A-Z])(\d+)([A-Z])", rows[0]["mutant"])
    if not m:
        return None
    seq = list(rows[0]["mutated_sequence"])
    seq[int(m.group(2)) - 1] = m.group(1)
    return "".join(seq)


def a3m_depth(p: Path) -> int:
    with open(p, errors="ignore") as fh:
        return sum(1 for line in fh if line.startswith(">"))


def describe(assay: str) -> dict:
    entry: dict = {"id": assay}
    csv_p = ASSAY_DIR / f"{assay}.csv"
    if csv_p.exists():
        rows = [r for r in csv.DictReader(open(csv_p)) if ":" not in r["mutant"]]
        wt = wt_of(rows) if rows else None
        entry["assay_csv"] = {"path": str(csv_p), "sha256": sha256(csv_p)}
        entry["n_single_variants"] = len(rows)
        if wt:
            entry["wt_length"] = len(wt)
    else:
        entry["assay_csv"] = None
        entry["MISSING"] = "assay csv not found"
    a3m = find_a3m(assay)
    if a3m:
        entry["msa"] = {"path": str(a3m), "sha256": sha256(a3m),
                        "rows": a3m_depth(a3m), "panel": a3m.parts[-3]}
    else:
        entry["msa"] = None
        entry["MISSING"] = "alignment not found in any panel"
    return entry


def yaml_dump(doc: dict) -> str:
    """Minimal YAML writer: no dependency, and the shape here is fixed."""
    def val(v):
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        s = str(v)
        return f'"{s}"' if re.search(r"[:#\-{}\[\]]|^\s|\s$", s) else s

    out = []
    for k, v in doc.items():
        if k == "assays":
            out.append("assays:")
            for entry in v:
                first = True
                for ek, ev in entry.items():
                    lead = "  - " if first else "    "
                    first = False
                    if isinstance(ev, dict):
                        out.append(f"{lead}{ek}:")
                        for sk, sv in ev.items():
                            out.append(f"      {sk}: {val(sv)}")
                    else:
                        out.append(f"{lead}{ek}: {val(ev)}")
        elif isinstance(v, str) and "\n" in v:
            out.append(f"{k}: >-")
            for line in v.strip().split("\n"):
                out.append(f"  {line.strip()}")
        else:
            out.append(f"{k}: {val(v)}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(W / "runs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", action="append",
                    help="write only these cohorts. Regenerating every "
                         "manifest rewrites checksums for cohorts nobody "
                         "meant to touch, so a targeted addition says so.")
    a = ap.parse_args()

    runs = Path(a.runs)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(a.only) if a.only else None

    for name, (ids, description) in EXPLICIT.items():
        if only and name not in only:
            continue
        doc = {
            "cohort": name,
            "description": description,
            "derived_from": "an explicit list in build_cohort_manifests.EXPLICIT",
            "n_assays": len(ids),
            "assays": [describe(x) for x in ids],
        }
        missing = [e["id"] for e in doc["assays"] if "MISSING" in e]
        path = out_dir / f"{name}.yaml"
        path.write_text(yaml_dump(doc))
        flag = f"  MISSING INPUTS: {missing}" if missing else ""
        print(f"{name:22s} {len(ids):3d} assays -> {path}{flag}")

    for name, (glob, description) in COHORTS.items():
        if only and name not in only:
            continue
        prefix = STRIP[name]
        assays = sorted(
            p.name[len(prefix):-len(".npz")] for p in runs.glob(glob))
        doc = {
            "cohort": name,
            "description": description,
            "derived_from": f"{glob} in {runs}",
            "n_assays": len(assays),
            "assays": [describe(x) for x in assays],
        }
        missing = [e["id"] for e in doc["assays"] if "MISSING" in e]
        path = out_dir / f"{name}.yaml"
        path.write_text(yaml_dump(doc))
        flag = f"  MISSING INPUTS: {missing}" if missing else ""
        print(f"{name:22s} {len(assays):3d} assays -> {path}{flag}")

    if only and "smoke_pairformer" not in only:
        return

    smoke = {
        "cohort": "smoke_pairformer",
        "description": ("One assay for the pair-layer vertical slice. A subset "
                        "of basis_assays on purpose, so anything collected "
                        "against it can be compared with an archive that "
                        "already exists."),
        "derived_from": f"{SMOKE_ASSAY}, chosen as the smallest real assay",
        "n_assays": 1,
        "assays": [describe(SMOKE_ASSAY)],
    }
    smoke_path = out_dir / "smoke_pairformer.yaml"
    smoke_path.write_text(yaml_dump(smoke))
    print(f"{'smoke_pairformer':22s}   1 assays -> {smoke_path}")

    # The disjointness heldout_v1's claim rests on, checked rather than assumed.
    basis = {p.name[len("gym2s_"):-4] for p in runs.glob("gym2s_*.npz")}
    held = {p.name[len("gym3p3_"):-4] for p in runs.glob("gym3p3_*.npz")}
    overlap = basis & held
    print(f"\nbasis n={len(basis)}  heldout n={len(held)}  overlap={len(overlap)}"
          + (f"  !! {sorted(overlap)}" if overlap else "  (disjoint, as required)"))


if __name__ == "__main__":
    main()
