"""Build the 'physics vs memory' mutant cohort.

Idea: mutations that should be structurally catastrophic on physical grounds --
buried hydrophobic core residues replaced by charged ones -- paired with a
surface-mutation control matched for count. Every mutant keeps the *wild-type*
MSA, so the alignment says "wild-type fold" while the sequence says "this
cannot fold". Whatever the model predicts is the arbitration we want to
localise.

MSA grafting (reusing the WT a3m verbatim, only rewriting the query row) is
sound here and ~100x cheaper than recomputing: for point mutants the ColabFold
search returns essentially the WT alignment anyway. It also makes the MSA an
exactly controlled variable rather than a confound, which matters more for this
experiment than realism -- and it is the condition the mechanistic question is
actually about.

Outputs a directory of Boltz yamls + one shared a3m per parent, plus a
manifest.csv describing every mutant.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

HYDROPHOBIC = set("AVLIMFWCY")
CHARGED = "DEKR"
POLAR_SURFACE = "STNQ"

# <= 5 chars: Boltz truncates chain names to MAX_CHAIN_NAME.
CHAIN_ID = "A"

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def parse_cif_ca(cif: Path):
    """Return (seq, coords) for CA atoms of the first chain in a Boltz cif."""
    res = {}
    chain0 = None
    with cif.open() as fh:
        cols, in_loop = [], False
        for line in fh:
            s = line.strip()
            if s.startswith("_atom_site."):
                cols.append(s.split(".")[1])
                in_loop = True
                continue
            if in_loop and (s.startswith("#") or not s):
                if cols:
                    break
                continue
            if in_loop and cols and not s.startswith("_"):
                f = s.split()
                if len(f) < len(cols):
                    continue
                rec = dict(zip(cols, f))
                if rec.get("group_PDB") != "ATOM" or rec.get("label_atom_id") != "CA":
                    continue
                ch = rec.get("label_asym_id")
                if chain0 is None:
                    chain0 = ch
                if ch != chain0:
                    continue
                idx = int(rec["label_seq_id"])
                res[idx] = (
                    THREE_TO_ONE.get(rec["label_comp_id"], "X"),
                    (float(rec["Cartn_x"]), float(rec["Cartn_y"]), float(rec["Cartn_z"])),
                )
    order = sorted(res)
    seq = "".join(res[i][0] for i in order)
    coords = np.array([res[i][1] for i in order], dtype=float)
    return seq, coords


def burial(coords: np.ndarray, radius: float = 10.0) -> np.ndarray:
    """CA neighbour count within `radius` -- a cheap, robust burial proxy."""
    d = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    return ((d < radius).sum(1) - 1).astype(int)


def pick_sites(seq: str, nb: np.ndarray, n: int, mode: str, rng) -> list[int]:
    """Pick n 0-based positions. mode='core' = buried hydrophobic,
    'surface' = exposed, non-hydrophobic."""
    hi, lo = np.percentile(nb, 75), np.percentile(nb, 25)
    if mode == "core":
        cand = [i for i, a in enumerate(seq) if a in HYDROPHOBIC and nb[i] >= hi]
        # most buried first -- these are the least ambiguous physics cases
        cand.sort(key=lambda i: -nb[i])
    else:
        cand = [i for i, a in enumerate(seq) if a not in HYDROPHOBIC and nb[i] <= lo]
        cand.sort(key=lambda i: nb[i])
    if len(cand) < n:
        raise ValueError(f"only {len(cand)} {mode} sites available, need {n}")
    # deterministic nested subsets: the n=4 set is a subset of the n=8 set, so
    # dose-response curves are not confounded by which sites were drawn
    return cand[:n]


def mutate(seq: str, sites: list[int], mode: str, seed: int) -> tuple[str, list[str]]:
    """Substitute each site. The replacement is a deterministic function of the
    site, not of the draw order, so the n=4 mutant is exactly the n=8 mutant
    restricted to its first four sites -- otherwise the dose-response curve
    would confound 'more mutations' with 'different substitutions'."""
    out = list(seq)
    labels = []
    for i in sites:
        wt = seq[i]
        pool = [a for a in (CHARGED if mode == "core" else POLAR_SURFACE) if a != wt]
        new = pool[(i * 2654435761 + seed) % len(pool)]
        out[i] = new
        labels.append(f"{wt}{i+1}{new}")
    return "".join(out), labels


def _core(a3m_seq: str) -> str:
    """Aligned core of an a3m row: drop lowercase insertions and gaps."""
    return "".join(c for c in a3m_seq if not c.islower()).replace("-", "")


def write_a3m(dst: Path, src_a3m: Path, query: str, name: str, wt: str):
    """Copy the parent a3m, replacing only the query row.

    Boltz's featurizer silently substitutes a dummy MSA if row 0 does not match
    the yaml sequence, which would make the run single-sequence without saying
    so. Rewriting row 0 is what keeps the graft honest.

    Rows whose aligned core equals the *wild-type* sequence are dropped. The
    WT's own self-hit is normally present in a ColabFold alignment, and Boltz
    deduplicates it against row 0 -- but only for the WT, whose query matches
    it. Mutants keep that row, so the WT ends up with one fewer MSA row than
    every mutant, and row-wise patching between them silently misaligns (it
    surfaces as a shape mismatch, 731 vs 732). Removing the duplicate up front
    makes every variant in the cohort share an identical alignment, which is
    what the path-patching experiment assumes.
    """
    lines = src_a3m.read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(">"))
    body, drop = [], 0
    for h, s in zip(lines[i + 2 :: 2], lines[i + 3 :: 2]):
        if _core(s) == wt:
            drop += 1
            continue
        body += [h, s]
    dst.write_text("\n".join([f">{name}", query, *body]) + "\n")
    return drop


YAML_TMPL = """version: 1

sequences:
  - protein:
      id: "{cid}"
      sequence: {seq}
      msa: {msa}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", required=True, help="WT predicted structure (for burial)")
    ap.add_argument("--a3m", required=True, help="WT alignment to graft")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="gfp")
    ap.add_argument("--counts", default="1,2,4,8,16,32")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "yamls").mkdir(parents=True, exist_ok=True)
    (out / "msa").mkdir(parents=True, exist_ok=True)

    seq, coords = parse_cif_ca(Path(args.cif))
    nb = burial(coords)
    counts = [int(c) for c in args.counts.split(",")]
    rng = np.random.default_rng(args.seed)

    print(f"WT length {len(seq)}, burial range {nb.min()}-{nb.max()}")

    rows = []
    wt_seq = seq

    def emit(cid, s, mode, n, labels):
        # Boltz truncates chain names to MAX_CHAIN_NAME (5) chars while building
        # its chain->msa map from the *untruncated* yaml id, so any id longer
        # than 5 chars raises KeyError deep in parse_boltz_schema and the input
        # is silently skipped. Use a fixed short chain id; the file name carries
        # the identity instead.
        a3m = out / "msa" / f"{cid}.a3m"
        n_drop = write_a3m(a3m, Path(args.a3m), s, CHAIN_ID, wt_seq)
        (out / "yamls" / f"{cid}.yaml").write_text(
            YAML_TMPL.format(cid=CHAIN_ID, seq=s, msa=a3m.resolve())
        )
        rows.append(
            {
                "id": cid, "mode": mode, "n_mut": n,
                "mutations": ";".join(labels), "seq_len": len(s),
                "identity_to_wt": f"{1 - n / len(s):.4f}",
            }
        )

    emit(f"{args.name}_wt", seq, "wt", 0, [])
    for mode in ("core", "surface"):
        for n in counts:
            sites = pick_sites(seq, nb, n, mode, rng)
            mut, labels = mutate(seq, sites, mode, args.seed)
            emit(f"{args.name}_{mode}_{n:02d}", mut, mode, n, labels)
            print(f"  {mode:7s} n={n:2d}  {','.join(labels[:6])}{'...' if n > 6 else ''}")

    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} sequences to {out}")


if __name__ == "__main__":
    main()
