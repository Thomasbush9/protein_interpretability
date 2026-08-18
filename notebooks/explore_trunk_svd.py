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


@app.cell
def _(mo):
    mo.md(
        """
        ## The shared basis

        Each assay is a different protein, yet the 128 pair channels mean the
        same thing in all of them. That is the claim worth testing, and the test
        is whether one decomposition fitted across every assay carries signal in
        each — not whether the reconstruction error is small.

        Per-assay z-score, pool, subtract the pooled mean, SVD. The sign of each
        leading component is then fixed against a reference quantity, because an
        SVD determines a direction only up to sign, and "PC2" means nothing until
        that is pinned.
        """
    )
    return


@app.cell
def _(RUNS, artifacts, cohort, np):
    from protein_interpretability.analysis import basis as basis_mod

    blocks, orient_ref = {}, {}
    for _assay in cohort:
        _c = artifacts.load_capture(RUNS / f"gym2s_{_assay.id}.npz")
        _k = _assay.id.split("_")[0]
        blocks[_k] = np.asarray(_c.field("dz_site"), float)
        orient_ref[_k] = np.asarray(_c.field("kl_glob"), float)[:, -1]

    shared = basis_mod.fit(
        blocks, layer=-1,                 # the final layer, measured not asserted
        orient_on="kl_glob", orient_ref=orient_ref, orient_k=2,
        n_boot=2000, seed=0,
    )
    return basis_mod, blocks, orient_ref, shared


@app.cell
def _(MUTED, SERIES_1, axes, np, shared):
    n_show = 10
    ev = np.asarray(shared.ev)[:n_show]

    fig_ev, ax_ev = axes(
        height=3.0,
        xlabel="component", ylabel="share of variance",
        # Not "most of the variance" -- PC1 and PC2 together are 39%, and the
        # tail is long. The interesting claim is not that the basis compresses
        # well; it is that one of these components transfers across proteins.
        title="PC1 and PC2 lead, ahead of a long tail",
    )
    _bars = ax_ev.bar(np.arange(1, n_show + 1), ev, color=SERIES_1,
                      width=0.7, zorder=3)
    # Label only the two the report actually uses.
    for _i in (0, 1):
        ax_ev.text(_i + 1, ev[_i] + 0.008, f"{ev[_i]:.1%}", ha="center",
                   color=MUTED, fontsize=9)
    ax_ev.set_xticks(np.arange(1, n_show + 1))
    ax_ev.set_ylim(0, max(ev) * 1.18)
    fig_ev.tight_layout()
    fig_ev
    return (ev,)


@app.cell
def _(ev, mo, np):
    mo.md(
        f"""
        PC1 takes {ev[0]:.1%} and PC2 {ev[1]:.1%}; the first eight together
        reach {np.cumsum(ev)[7]:.1%}. **PC2 is the one the report is about** —
        PC1 tracks substitution volume and PC3 hydropathy, neither of which is
        the stability axis. That is why the causal experiment steers PC2 and
        uses PC1 and PC3 as controls: if all components behaved alike, the
        effect would be about perturbation size rather than about this
        direction.
        """
    )
    return


@app.cell
def _(INK, MUTED, SERIES_1, artifacts, assay_pick, axes, np, RUNS, shared, st):
    _cap = artifacts.load_capture(RUNS / f"gym2s_{assay_pick.value}.npz")
    _dz = np.asarray(_cap.field("dz_site"), float)
    _y = np.asarray(_cap.field("score"), float)

    pc = shared.project(_dz[:, -1, :], layer=-1)
    pc2 = pc[:, 1]
    rho_pc2 = float(st.spearman(pc2, _y))

    fig_pc, ax_pc = axes(
        height=3.6,
        xlabel="PC2 score", ylabel="measured stability",
        title=f"PC2 against the assay, {assay_pick.value}",
    )
    ax_pc.axhline(0, color=MUTED, linewidth=1, zorder=1)
    ax_pc.scatter(pc2, _y, s=26, color=SERIES_1, alpha=0.75,
                  edgecolor="white", linewidth=0.6, zorder=3)
    ax_pc.text(0.02, 0.06, f"Spearman {rho_pc2:+.3f}   n = {len(_y)}",
               transform=ax_pc.transAxes, color=INK, fontsize=9)
    fig_pc.tight_layout()
    fig_pc
    return pc, pc2, rho_pc2


@app.cell
def _(mo):
    mo.md(
        """
        ## The headline

        Everything above looked at one assay at a time. The claim in the report
        is stronger: fit on eleven proteins, test on the twelfth, and the probe
        still works — so the direction is not a per-protein artefact.

        This is the +0.758 the report opens with, recomputed here from the same
        captures.
        """
    )
    return


@app.cell
def _():
    from protein_interpretability.analysis.probes import leave_one_group_out
    return (leave_one_group_out,)


@app.cell
def _(RUNS, artifacts, blocks, cohort, leave_one_group_out, np):
    # The assay score travels with each capture, so it is read from the same
    # archives the blocks came from rather than re-derived from the CSVs and
    # re-aligned by mutant name.
    probe_blocks = {}
    for _assay in cohort:
        _k = _assay.id.split("_")[0]
        _c = artifacts.load_capture(RUNS / f"gym2s_{_assay.id}.npz")
        probe_blocks[_k] = {
            "X": blocks[_k][:, -1, :],
            "y": np.asarray(_c.field("score"), float),
        }

    per_assay = leave_one_group_out(probe_blocks, lam=10.0)
    pooled = float(np.mean(list(per_assay.values())))
    return per_assay, pooled, probe_blocks


@app.cell
def _(INK, MUTED, SERIES_1, SERIES_2, axes, np, per_assay, pooled):
    names = sorted(per_assay, key=lambda k: per_assay[k])
    vals = [per_assay[n] for n in names]

    fig_hl, ax_hl = axes(
        height=3.8,
        xlabel="Spearman on the held-out assay", ylabel="",
        title="Fit on eleven proteins, tested on the twelfth",
    )
    ax_hl.barh(np.arange(len(names)), vals, color=SERIES_1, height=0.66,
               zorder=3)
    ax_hl.axvline(pooled, color=SERIES_2, linewidth=2, zorder=4)
    ax_hl.text(pooled, len(names) - 0.35, f"  mean {pooled:+.3f}",
               color=SERIES_2, fontsize=9, va="center")
    ax_hl.set_yticks(np.arange(len(names)))
    ax_hl.set_yticklabels(names, color=MUTED, fontsize=9)
    ax_hl.set_xlim(0, max(vals) * 1.12)
    # The extremes only. A number on every bar collided with the mean line at
    # the four assays nearest it, and bar length already carries the comparison
    # -- the two ends are what a reader needs in figures.
    for _i in (0, len(vals) - 1):
        ax_hl.text(vals[_i] + 0.01, _i, f"{vals[_i]:.2f}", va="center",
                   color=INK, fontsize=9)
    fig_hl.tight_layout()
    fig_hl
    return names, vals


@app.cell
def _(mo, per_assay, pooled):
    mo.md(
        f"""
        **{pooled:+.6f}** across {len(per_assay)} held-out assays — the figure
        the report quotes as +0.758, from the recipe in
        `experiments/analysis/reproduce_headline_transfer.py`: final-layer
        `dz_site`, z-scored within assay, leave-one-assay-out ridge at λ=10.

        The spread matters as much as the mean. The weakest assay is
        {min(per_assay, key=per_assay.get)} at
        {min(per_assay.values()):+.3f} and the strongest
        {max(per_assay, key=per_assay.get)} at {max(per_assay.values()):+.3f};
        a mean over twelve proteins with that spread is a different claim from
        a single number on one.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Going further

        Everything here reads artifacts. To collect new ones you need a GPU node,
        and the job is rendered before it is queued:

        ```bash
        # what would run, resolved from the site profile — loads nothing
        uv run pi render --checkout collect_pairformer_layers.py

        # the capture itself
        sbatch jax_harness/checkout.sbatch \\
            ../experiments/collection/collect_pairformer_layers.py \\
            --n-variants 8 --out $W/runs/mine.npz
        ```

        Two things worth knowing before comparing anything you collect against
        an archive:

        - `dz_site` agrees across jobs to about **1%**, not exactly, and some
          variants are far more sensitive than others — one sampled assay ranged
          from 0.2% to 5.4% between two runs of identical code. Comparing
          captures against zero will always fail.
        - The same field name means different things in different archives:
          a 128-channel **vector** here, a per-layer **norm** in the cross-model
          captures. `load_capture` checks the array's rank rather than trusting
          the name, and `CaptureSpec` states which one a run promised.

        `docs/API.md` is the guide; `pi reproduce` replays any archived result
        from the command it recorded.
        """
    )
    return


if __name__ == "__main__":
    app.run()
