"""Build the random-mutation dose ladder with replicate draws.

The original gfp_rand_* arm (one draw per load, generator never committed,
seed unrecorded) is superseded by this cohort: a denser ladder of mutation
loads with several independent random draws per load, every draw's seed
recorded in the manifest. Conventions match the reconstructed original arm:
positions uniform over all residues, replacement drawn from the 19 non-WT
standard amino acids, draws independent across loads and replicates (NOT
nested -- unlike the core/surface arm, the point here is sampling variation,
not a controlled dose path).

MSA grafting as in build_dataset.py: every variant reuses the WT alignment
with only the query row rewritten.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from build_dataset import CHAIN_ID, YAML_TMPL, write_a3m

AMINO = "ACDEFGHIKLMNPQRSTVWY"


def read_wt(yaml_path: Path) -> str:
    for line in yaml_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("sequence:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"no sequence: line in {yaml_path}")


def draw(seq: str, n: int, rng) -> tuple[str, list[str]]:
    sites = sorted(rng.choice(len(seq), size=n, replace=False).tolist())
    out = list(seq)
    labels = []
    for i in sites:
        wt = seq[i]
        pool = [a for a in AMINO if a != wt]
        new = pool[rng.integers(len(pool))]
        out[i] = new
        labels.append(f"{wt}{i+1}{new}")
    return "".join(out), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt-yaml", required=True, help="WT yaml (sequence source)")
    ap.add_argument("--a3m", required=True, help="WT alignment to graft")
    ap.add_argument("--out", required=True)
    ap.add_argument("--loads", default="1,2,4,8,12,24,48,95,167",
                    help="mutation counts (n_mut)")
    ap.add_argument("--reps", type=int, default=8, help="independent draws per load")
    ap.add_argument("--master-seed", type=int, default=20260909)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "yamls").mkdir(parents=True, exist_ok=True)
    (out / "msa").mkdir(parents=True, exist_ok=True)
    wt = read_wt(Path(args.wt_yaml))
    src_a3m = Path(args.a3m)
    loads = [int(c) for c in args.loads.split(",")]

    rows = []

    def emit(cid, s, mode, n, labels, seed):
        a3m = out / "msa" / f"{cid}.a3m"
        write_a3m(a3m, src_a3m, s, CHAIN_ID, wt)
        (out / "yamls" / f"{cid}.yaml").write_text(
            YAML_TMPL.format(cid=CHAIN_ID, seq=s, msa=a3m.resolve())
        )
        rows.append({"id": cid, "mode": mode, "n_mut": n,
                     "mutations": ";".join(labels), "seq_len": len(s),
                     "identity_to_wt": f"{1 - n / len(s):.4f}", "seed": seed})

    emit("gfp_wt", wt, "wt", 0, [], "")
    for n in loads:
        for r in range(args.reps):
            # child seed is a pure function of (master, load, rep): the cohort
            # is reproducible and extensible without re-drawing existing rows
            rng = np.random.default_rng(np.random.SeedSequence(
                [args.master_seed, n, r]))
            s, labels = draw(wt, n, rng)
            emit(f"gfp_r{n:03d}_s{r}", s, "random", n, labels,
                 f"{args.master_seed}.{n}.{r}")

    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows ({len(loads)} loads x {args.reps} reps + WT) -> {out}")


if __name__ == "__main__":
    main()
