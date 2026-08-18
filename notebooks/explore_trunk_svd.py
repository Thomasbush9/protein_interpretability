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


@app.cell
def _():
    import matplotlib
    import matplotlib.pyplot as plt

    # This matplotlib wheel ships no bundled fonts -- its `fonts/ttf` directory
    # is absent -- so the default DejaVu Sans raises rather than falling back,
    # and every figure fails. Name the families this machine actually has, in
    # order, and re-enable the fallback that the failure said was disabled.
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "Nimbus Sans", "Droid Sans", "Cantarell", "sans-serif",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    # One palette for every chart here, so they read as one system. These are
    # the reference palette's first three categorical slots, used unchanged --
    # documented as clearing the all-pairs colour-vision gates in both modes.
    INK = "#0b0b0b"
    MUTED = "#52514e"
    SERIES_1 = "#2a78d6"      # blue
    SERIES_2 = "#eb6834"      # orange
    GRID = "#e4e3df"

    def axes(width=7.2, height=3.4, xlabel="", ylabel="", title=""):
        """A chart with recessive furniture: the data should be the only thing
        with weight on the page."""
        fig, ax = plt.subplots(figsize=(width, height))
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=12)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        return fig, ax

    return INK, MUTED, SERIES_1, SERIES_2, axes, plt


@app.cell
def _(mo):
    mo.md(
        """
        ## Where in the trunk does the signal live?

        At each Pairformer layer, how well does the size of the representation's
        movement track the measured stability effect? One number per layer.
        """
    )
    return


@app.cell
def _(INK, MUTED, SERIES_1, axes, dz, n_layers, np, score, st):
    trace = np.array([
        st.spearman(np.linalg.norm(dz[:, i, :], axis=-1), score)
        for i in range(n_layers)
    ])
    peak = int(np.nanargmax(np.abs(trace)))

    fig_trace, ax_trace = axes(
        xlabel="Pairformer layer",
        ylabel="Spearman  |dz| vs stability",
        title="The signal strengthens through the last quarter of the trunk",
    )
    ax_trace.axhline(0, color=MUTED, linewidth=1, zorder=1)
    ax_trace.plot(np.arange(n_layers), trace, color=SERIES_1, linewidth=2,
                  zorder=3)
    # Direct-label the peak only; a number on every point is noise. Flip the
    # label inward when the peak sits near an edge -- for this trace it lands on
    # the final layer, where a right-hand label runs outside the axes.
    ax_trace.plot([peak], [trace[peak]], "o", color=SERIES_1, markersize=9,
                  markeredgecolor="white", markeredgewidth=1.5, zorder=4)
    near_right = peak > 0.75 * n_layers
    ax_trace.annotate(
        f"strongest  layer {peak}   {trace[peak]:+.2f}",
        xy=(peak, trace[peak]),
        xytext=(-10 if near_right else 10, 12), textcoords="offset points",
        ha="right" if near_right else "left",
        color=INK, fontsize=9,
    )
    fig_trace.tight_layout()
    fig_trace
    return peak, trace


@app.cell
def _(assay_pick, mo, n_layers, peak, trace):
    mo.md(
        f"""
        For **{assay_pick.value}** the correlation is strongest at layer
        **{peak}** of {n_layers}, at {trace[peak]:+.3f}. It sits near
        {trace[:len(trace) // 2].mean():+.2f} through the first half and
        strengthens late — which is why the report reads the pair row at the
        **final** layer rather than averaging the trunk.

        The sign is negative because a larger movement means a more destabilising
        mutation, and the assay scores destabilisation as negative.

        Note this is a **magnitude** — it asks *where* the signal is, not *which
        direction* carries it. The direction is the next section, and it is where
        the result actually comes from.
        """
    )
    return


if __name__ == "__main__":
    app.run()
