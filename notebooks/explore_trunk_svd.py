"""Inspect a trunk capture and the shared basis built from it.

    uv run --group notebook marimo edit notebooks/explore_trunk_svd.py
    uv run --group notebook marimo run  notebooks/explore_trunk_svd.py   # read-only

This notebook READS captures; it never loads a model. That is not a limitation
to work around -- it is what lets it run on a login node and open in seconds.
The representations it explores were produced on GPU nodes by
`exp_gym2.py`, and every capture records the exact command that made it, which
the notebook shows rather than paraphrases.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Inside the trunk

        Boltz-2's internal state predicts mutational stability far better than
        anything it emits. This notebook walks the evidence for that, from a raw
        capture to the shared direction the report is built on.

        Nothing here loads a model. Every representation was collected on a GPU
        node; this reads the archives.
        """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import numpy as np

    REPO = Path(__file__).resolve().parents[1]
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))

    from protein_interpretability import artifacts
    from protein_interpretability.analysis import statistics as st
    from protein_interpretability.collection import Cohort

    W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
    RUNS = W / "runs"
    return Cohort, RUNS, artifacts, np, st


@app.cell
def _(Cohort, mo):
    cohort = Cohort.load("basis_assays")
    cohort.verify(checksums=False)          # existence only; hashing 44 files is slow here

    mo.md(
        f"""
        ## The cohort

        **{cohort.name}** — {len(cohort)} assays. {cohort.description}

        `verify()` would refuse to go on if an assay table or alignment had been
        rewritten since the manifest was generated. Checksums are skipped here
        only because hashing every input is slow for an interactive notebook;
        the collection scripts do the full check.
        """
    )
    return (cohort,)


@app.cell
def _(cohort, mo):
    assay_pick = mo.ui.dropdown(
        options={a.id.split("_")[0]: a.id for a in cohort},
        value=cohort.assays[0].id.split("_")[0],
        label="assay",
    )
    assay_pick
    return (assay_pick,)


@app.cell
def _(RUNS, artifacts, assay_pick, mo, np):
    cap = artifacts.load_capture(RUNS / f"gym2s_{assay_pick.value}.npz")
    dz = np.asarray(cap.field("dz_site"), float)      # [variants, layers, channels]
    score = np.asarray(cap.field("score"), float)
    mutants = [str(m) for m in cap.field("mutant")]
    n_var, n_layers, width = dz.shape

    mo.md(
        f"""
        ## What one capture holds

        `gym2s_{assay_pick.value}.npz` — **{n_var} variants × {n_layers} layers ×
        {width} channels**.

        `dz_site` is the pair representation row at the mutated residue, minus
        the wild type's, at every Pairformer layer. It is a **direction**, not a
        magnitude — and that distinction is load-bearing: the same field name
        holds a per-layer norm in the cross-model archives, which is why
        `Capture` checks the array's rank rather than trusting the name.

        `|dz|` ranges {np.abs(dz).mean():.2f} on average, up to
        {np.abs(dz).max():.1f}.
        """
    )
    return cap, dz, mutants, n_layers, n_var, score, width


@app.cell
def _(cap, mo):
    fields = "\n".join(
        f"| `{k}` | {tuple(cap._arr(k).shape)} | {cap._arr(k).dtype} |"
        for k in sorted(cap.files) if not k.startswith("_")
    )
    mo.md(
        f"""
        ### Every field in the archive

        | field | shape | dtype |
        |---|---|---|
        {fields}
        """
    )
    return


@app.cell
def _(RUNS, mo):
    import json

    prov = json.loads((RUNS / "svd_dz_v3.json").read_text())["provenance"]
    argv = " ".join(a.replace(str(RUNS.parent), "$W") for a in prov["argv"])

    mo.md(
        f"""
        ### Where these numbers came from

        Every result file records the command that produced it, so a capture is
        never an orphan. The SVD study below was produced by:

        ```
        {argv}
        ```

        written {prov['written_utc']} on `{prov.get('host', '?')}`, git
        `{(prov.get('git_commit') or '?')[:12]}`. `pi reproduce` replays exactly
        this rather than asking anyone to reconstruct it.
        """
    )
    return


if __name__ == "__main__":
    app.run()
