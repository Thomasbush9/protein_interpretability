"""Choose the next panel of ProteinGym assays, and say why each one is in it.

The expansion this is for: the internal-versus-output result was established on
twelve stability assays of 61-72 residues, and a generality claim needs more
proteins, more lengths and more phenotypes. Which proteins is a DESIGN question,
not a matter of taking the next N alphabetically, because of one structural fact
about ProteinGym:

    phenotype    <100  100-200  200-400  400-800  >800
    stability      64        0        2        1      0
    fitness         3       12       21       25     14
    abundance       2        5       12        9      4

STABILITY IS SMALL PROTEINS. Sixty-four of the sixty-seven stability assays are
Tsuboyama's cDNA-display proteolysis on mini-domains, all under 100 residues.
There is no length axis within stability to be had; the experiments were never
done. So "does this hold on bigger proteins" cannot be asked while holding the
phenotype at stability, and any panel that simply picks longer proteins is
silently also changing what the DMS score measures.

FITNESS IS THE LENGTH AXIS. It is the only class with real spread across every
band, so it carries the length contrast at a fixed phenotype. Abundance and
binding then supply a phenotype contrast at matched length. That gives two
separable comparisons instead of one confounded one:

    length effect     within fitness, across bands
    phenotype effect  within a band, across fitness / abundance / binding

Neither is a controlled experiment -- these are different labs, organisms and
readouts, and nothing here changes that. What the design buys is that the two
factors are no longer perfectly collinear, so a difference can be attributed to
one of them rather than to "bigger proteins, which are also a different assay".

    uv run python jax_harness/select_panel.py --out-dir .../panel5 --per-cell 6
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
GYM = W / "data" / "gym"
PANELS = ("panel", "panel2", "panel3", "panel4")

BANDS = [(0, 100, "<100"), (100, 200, "100-200"), (200, 400, "200-400"),
         (400, 800, "400-800"), (800, 100000, ">800")]

# The cells the panel is built to fill. Length contrast runs down the fitness
# column; phenotype contrast runs across a row.
TARGET_CELLS = [
    ("fitness", "100-200"), ("fitness", "200-400"),
    ("fitness", "400-800"), ("fitness", ">800"),
    ("abundance", "200-400"), ("abundance", "400-800"),
    ("binding", "400-800"),
    ("activity", "400-800"),
]


def coarse_phenotype(s: str) -> str:
    """Collapse 36 free-text `selection_type` values into usable classes.

    The column is not controlled vocabulary -- it holds both "Activity" and
    "activity", both "Binding" and "binding", and singletons like "Survival
    (dosed with trametinib)". Grouping is therefore a judgement, and it is made
    here in one place so the panel can state which judgement it used.
    """
    s = str(s).lower()
    if "proteolysis" in s or "stability" in s or "thermostab" in s:
        return "stability"
    if any(k in s for k in ("growth", "survival", "resistance", "fitness",
                            "complementation", "phage")):
        return "fitness"
    if any(k in s for k in ("facs", "flow", "abundance", "vamp",
                            "fluorescence", "rna-seq", "rna-sequencing")):
        return "abundance"
    if "binding" in s:
        return "binding"
    if any(k in s for k in ("activity", "enzymatic", "toxin", "voltage",
                            "phosphatase")):
        return "activity"
    return "other"


# Keyword heuristic for integral-membrane and membrane-anchored proteins.
# Imperfect by construction -- it reads a free-text `molecule_name`, not UniProt
# subcellular annotation -- so it is used to BALANCE the panel and is recorded
# per assay, never to make a claim.
MEMBRANE_WORDS = ("receptor", "channel", "transporter", "rhodopsin", "opsin",
                  "gpcr", "abc ", "hemagglutinin", "kir", "integrin",
                  "adrb", "ccr", "permease", "porin", "atpase")


def is_membrane(row) -> bool:
    """Whether this is likely a membrane or membrane-anchored protein.

    WHY THE PANEL HAS TO KNOW. Large deep-mutational-scanned human proteins are
    disproportionately receptors, channels and transporters, so a panel built by
    taking the longest available assays picks up membrane proteins as a side
    effect -- in the first draft of this selection, five of the nine assays at
    500 aa and above. That matters here more than it would elsewhere: these
    models predict a single chain with no bilayer, and pLDDT on a membrane
    protein predicted out of its environment is not the same quantity as pLDDT
    on a soluble one. pLDDT is the OUTPUT side of the comparison, so letting
    membrane-ness ride along with length would put a difference in the output
    baseline inside the length axis.
    """
    text = f"{row.get('molecule_name', '')} {row.get('DMS_id', '')}".lower()
    return any(w in text for w in MEMBRANE_WORDS)


def band_of(n: int) -> str:
    for lo, hi, name in BANDS:
        if lo < n <= hi:
            return name
    return ">800"


def existing_alignments() -> set:
    return {Path(p).stem
            for pn in PANELS
            for p in glob.glob(str(GYM / pn / "colabfold_output" / "*.a3m"))}


def load() -> pd.DataFrame:
    d = pd.read_csv(GYM / "ref.csv")
    d["pheno"] = d.selection_type.map(coarse_phenotype)
    d["band"] = d.seq_len.map(band_of)
    d["has_msa"] = d.DMS_id.isin(existing_alignments())
    d["membrane"] = d.apply(is_membrane, axis=1)
    return d


def select(d: pd.DataFrame, per_cell: int, max_len: int,
           min_neff: float = 1000.0, one_per_protein: bool = True) -> pd.DataFrame:
    """Fill each target cell, deterministically and with reasons recorded.

    Ordering is by alignment depth descending. A shallow alignment is a
    different operating point for these models -- `exp_msa_regime` measured that
    directly -- so taking the deepest available in each cell keeps MSA depth
    from becoming a third thing that varies with length by accident.

    ONE ASSAY PER PROTEIN, by default. ProteinGym holds four separate BLAT_ECOLX
    assays from four labs, two DYR_ECOLI, and several proteins measured under
    two readouts. They are different experiments but they are not different
    proteins: same sequence, same alignment, same structure. Every interval in
    this project is a cluster bootstrap over assays, so admitting four BLAT rows
    would count one protein four times and report an interval narrower than the
    evidence supports. Deduplicating here makes "assay" and "protein" the same
    unit, which is what the bootstrap already assumes.

    A shallow alignment is excluded outright rather than ranked last:
    SPG1_STRSG_Olson_2014 has N_eff = 2, so the model would be reading a single
    sequence, and a single-sequence operating point was measured ~4.4x more
    mutation-sensitive. Mixing it in would put that difference inside the
    length axis.
    """
    pool = d[(~d.has_msa)
             & (d.DMS_number_single_mutants >= 100)
             & (d.seq_len <= max_len)
             & (d.MSA_N_eff >= min_neff)].copy()
    picks, claimed = [], set()
    for pheno, band in TARGET_CELLS:
        cell = pool[(pool.pheno == pheno) & (pool.band == band)]
        if one_per_protein:
            cell = cell[~cell.UniProt_ID.isin(claimed)]
        # Soluble first, then by alignment depth. Ranked rather than filtered:
        # a cell short of soluble candidates takes membrane ones rather than
        # coming up empty, and the flag travels with every row either way.
        cell = cell.sort_values(["membrane", "MSA_N_eff",
                                 "DMS_number_single_mutants"],
                                ascending=[True, False, False])
        if one_per_protein:
            cell = cell.drop_duplicates(subset="UniProt_ID")
        take = cell.head(per_cell).copy()
        take["cell"] = f"{pheno}/{band}"
        claimed.update(take.UniProt_ID)
        picks.append(take)
        if len(take) < per_cell:
            print(f"  NOTE {pheno}/{band}: only {len(take)} of {per_cell} "
                  f"available after filters -- the cell is thin, not full")
    return pd.concat(picks).drop_duplicates(subset="DMS_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=800,
                    help="above this the capture is unproven: at 403 residues "
                         "the 16-layer model already sits at 1.4e-3 of a 2e-3 "
                         "drift tolerance, and the per-layer distogram is "
                         "O(L*N^2*bins) in host memory. Probe before raising.")
    ap.add_argument("--min-neff", type=float, default=1000.0,
                    help="minimum alignment N_eff. A near-empty alignment is a "
                         "different operating point, not a thin version of the "
                         "same one")
    ap.add_argument("--allow-repeat-proteins", action="store_true",
                    help="permit several assays of the same protein. Off by "
                         "default: the cluster bootstrap treats an assay as an "
                         "independent unit and four BLAT_ECOLX rows are not")
    ap.add_argument("--out-dir", help="write targets.fasta here for "
                                      "colabfold_search")
    ap.add_argument("--csv", help="write the panel table here")
    a = ap.parse_args()

    d = load()
    print(f"{len(d)} ProteinGym assays, {int(d.has_msa.sum())} already aligned")
    panel = select(d, a.per_cell, a.max_len, min_neff=a.min_neff,
                   one_per_protein=not a.allow_repeat_proteins)

    print(f"\nselected {len(panel)} assays:\n")
    for cell, grp in panel.groupby("cell", sort=False):
        print(f"  {cell}")
        for _, r in grp.iterrows():
            print(f"    {r.seq_len:5d} aa  {r.DMS_number_single_mutants:6d} muts  "
                  f"Neff {r.MSA_N_eff:9.0f}  {r.DMS_id}")

    print(f"\nlength range {panel.seq_len.min()}-{panel.seq_len.max()} aa; "
          f"{panel.pheno.nunique()} phenotype classes; "
          f"{panel.UniProt_ID.nunique()} distinct proteins for {len(panel)} assays")
    print("per phenotype:", panel.pheno.value_counts().to_dict())
    mem = int(panel.membrane.sum())
    print(f"membrane/anchored: {mem} of {len(panel)}"
          + (f"  ({', '.join(panel[panel.membrane].DMS_id)})" if mem else ""))
    big = panel[panel.seq_len >= 500]
    if len(big):
        print(f"  of the {len(big)} assays >=500 aa, {int(big.membrane.sum())} "
              f"are membrane -- this is the ratio that must not track length")

    if a.out_dir:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fasta = out / "targets.fasta"
        with open(fasta, "w") as fh:
            for _, r in panel.iterrows():
                fh.write(f">{r.DMS_id}\n{r.target_seq}\n")
        print(f"\nwrote {fasta}  ({len(panel)} sequences)")
        print("  next: sbatch a copy of jax_harness/msa_panel3.sbatch pointed "
              "at this directory")
    if a.csv:
        panel.to_csv(a.csv, index=False)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
