"""Figure: is the L37-45 transition_z band real, or is any nine layers as good?

Five panels answering the three objections the audit raised against the original
single-control ablation:

  A/B  the per-layer divergence curve under each intervention, on the protein the
       band was DISCOVERED on (GFP) and on a HELD-OUT protein (DIO3)
  C    graded scaling of transition_z through the band -- a dose-response, not a
       single out-of-distribution deletion
  D    the frozen band against every width-matched sliding band, which is the
       null the original "one control band elsewhere" could not provide
  E    what each intervention did to the WILD-TYPE prediction, because an
       intervention that raises divergence by degrading the model into noise is
       not evidence about mutation representation

Colours follow the report's house categorical order; the pairs used here were
checked for CVD separation (worst adjacent OKLab dE = 9.5, above the floor of 8).
`ablate_all` is drawn in neutral ink rather than a series colour: panel E shows it
destroys the prediction, so it is a scale reference, not a comparable condition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SURF = "#fcfcfb"
_have = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_sans = next((f for f in ("Nimbus Sans", "DejaVu Sans", "Helvetica", "Arial")
              if f in _have), None)
plt.rcParams.update({
    **({"font.family": "sans-serif", "font.sans-serif": [_sans]} if _sans else {}),
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "font.size": 9, "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
})
C_INTACT, C_BAND, C_CTRL = "#2a78d6", "#eb6834", "#1baf7a"
C_ALL = "#8a8885"


def band_key(d):
    lo, hi = d["band"]
    return f"ablate_band_{lo}_{hi}"


def ctrl_key(d):
    lo, hi = d["control_band"]
    return f"ablate_control_{lo}_{hi}"


def curve_panel(ax, d, mid, title, letter):
    lo, hi = d["band"]
    cond = d["conditions"]
    ax.axvspan(lo, hi, color=C_BAND, alpha=0.10, zorder=0, lw=0)
    series = [("intact", C_INTACT, f"intact"),
              (band_key(d), C_BAND, f"transition_z deleted, L{lo}–{hi}"),
              (ctrl_key(d), C_CTRL,
               f"control band L{d['control_band'][0]}–{d['control_band'][1]}")]
    if "ablate_all" in cond:
        series.append(("ablate_all", C_ALL, "all 64 layers (model destroyed)"))
    for name, col, lab in series:
        if name not in cond or mid not in cond[name]:
            continue
        kl = np.array(cond[name][mid]["kl"])
        ax.plot(np.arange(len(kl)), kl, color=col, label=lab, zorder=3,
                ls="--" if name == "ablate_all" else "-",
                lw=1.6 if name == "ablate_all" else 2.0)
    ax.set_xlabel("Pairformer layer")
    ax.set_ylabel("mutant–WT distogram divergence\n(mean symmetric KL)")
    ax.set_title(f"{letter}  {title}", color=INK, fontsize=10, loc="left", pad=8)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.annotate(f"L{lo}–{hi}", xy=((lo + hi) / 2, ax.get_ylim()[1]),
                xytext=(0, -10), textcoords="offset points", ha="center",
                fontsize=7.5, color=C_BAND)


def dose_panel(ax, datasets, letter):
    """Graded scaling: alpha = 1 is intact, alpha = 0 is full deletion."""
    for (label, d, mid), col in zip(datasets, (C_INTACT, C_BAND)):
        pts = []
        for name, rec in d["conditions"].items():
            if mid not in rec:
                continue
            if name == "intact":
                pts.append((1.0, rec[mid]["kl_change_frozen_band"]))
            elif name.startswith("scale_band_") or name == band_key(d):
                pts.append((rec["alpha"], rec[mid]["kl_change_frozen_band"]))
        if not pts:
            continue
        pts.sort()
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        ax.plot(x, y, color=col, marker="o", ms=7, mec=SURF, mew=2,
                label=label, zorder=3)
    ax.axhline(0, color=INK2, lw=0.8, zorder=2)
    ax.set_xlabel("transition_z output scale α through the frozen band\n"
                  "(1 = intact, 0 = deleted)")
    ax.set_ylabel("change in divergence\nacross L37–45")
    ax.set_title(f"{letter}  Dose-response, not an all-or-nothing shock",
                 color=INK, fontsize=10, loc="left", pad=8)
    ax.invert_xaxis()
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)


def band_effect(d, mid, name, start, end):
    """Effect of ablating a band, corrected for what the INTACT model does there.

    The raw change across a band is not comparable between bands, because the
    intact model's own divergence profile is not flat: a band sitting on a steep
    stretch shows a large change with no intervention at all. Subtracting the
    intact change over the same layers is what makes the 56 bands a null
    distribution the frozen band can be ranked against.
    """
    kl = np.array(d["conditions"][name][mid]["kl"])
    it = np.array(d["conditions"]["intact"][mid]["kl"])
    s = max(start - 1, 0)
    return (kl[end] - kl[s]) - (it[end] - it[s])


def null_panel(ax, datasets, letter):
    """The frozen band against EVERY width-matched band, by band position."""
    drew = False
    for (label, d, mid), col in zip(datasets, (C_INTACT, C_BAND)):
        names = d.get("sliding_names") or []
        if not names:
            continue
        starts, eff = [], []
        for nm in names:
            s, e = (int(x) for x in nm.split("_")[1:3])
            starts.append(s)
            eff.append(band_effect(d, mid, nm, s, e))
        starts, eff = np.array(starts), np.array(eff)
        o = np.argsort(starts)
        lo, hi = d["band"]
        frozen = band_effect(d, mid, band_key(d), lo, hi)
        pct = float((eff <= frozen).mean())
        ax.plot(starts[o], eff[o], color=col, lw=1.8, alpha=0.85,
                label=f"{label} — all {len(eff)} bands", zorder=3)
        ax.scatter([lo], [frozen], s=70, color=col, edgecolor=SURF, lw=2,
                   zorder=6)
        ax.annotate(f"L{lo}–{hi}: {pct:.0%} pctile", xy=(lo, frozen),
                    xytext=(6, 8), textcoords="offset points", fontsize=8,
                    color=col, fontweight="bold")
        drew = True
    if not drew:
        ax.text(0.5, 0.5, "sliding-band sweep pending", ha="center", va="center",
                transform=ax.transAxes, color=INK2, fontsize=9)
    ax.axhline(0, color=INK2, lw=0.8, zorder=2)
    ax.set_xlabel("first layer of the ablated nine-layer band")
    ax.set_ylabel("effect on divergence across the band\n(ablated minus intact)")
    ax.set_title(f"{letter}  Responsive region, not a special band",
                 color=INK, fontsize=10, loc="left", pad=8)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    if drew:
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")


def quality_panel(ax, datasets, letter):
    """WT-quality diagnostics: did the intervention keep the model working?"""
    labels, plddt, rmsd, cols = [], [], [], []
    label_of = {"intact": "intact", "ablate_all": "all 64"}
    for (name_p, d, _mid) in datasets:
        for name, rec in d["conditions"].items():
            q = rec.get("wt_quality", {})
            if "plddt" not in q or name.startswith("scale_"):
                continue
            short = label_of.get(name)
            if short is None:
                short = ("L37–45" if name == band_key(d)
                         else "control" if name == ctrl_key(d) else None)
            if short is None:
                continue
            labels.append(f"{name_p}\n{short}")
            plddt.append(q["plddt"])
            rmsd.append(q["rmsd_to_intact_wt"])
            cols.append({"intact": C_INTACT, "L37–45": C_BAND,
                         "control": C_CTRL, "all 64": C_ALL}[short])
    x = np.arange(len(labels))
    ax.bar(x, plddt, 0.62, color=cols, zorder=3)
    for xi, (p, r) in enumerate(zip(plddt, rmsd)):
        ax.annotate(f"{r:.1f}Å", xy=(xi, p), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("wild-type pLDDT")
    ax.set_title(f"{letter}  Did the wild-type prediction survive?  "
                 "(bar labels: Cα RMSD to the intact model)",
                 color=INK, fontsize=10, loc="left", pad=8)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfp-primary", default="runs/ablate2_gfp_primary.json")
    ap.add_argument("--dio3-primary", default="runs/ablate2_dio3_primary.json")
    ap.add_argument("--gfp-sliding", default="runs/ablate2_gfp_sliding.json")
    ap.add_argument("--dio3-sliding", default="runs/ablate2_dio3_sliding.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    def load(p):
        p = Path(p)
        return json.loads(p.read_text()) if p.exists() else None

    gfp, dio3 = load(a.gfp_primary), load(a.dio3_primary)
    gfp_s, dio3_s = load(a.gfp_sliding), load(a.dio3_sliding)
    if gfp is None or dio3 is None:
        raise SystemExit("need both primary runs")

    gm = gfp["mutants"][0]
    dm = dio3["mutants"][0]

    fig = plt.figure(figsize=(14.5, 8.8))
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.30)

    curve_panel(fig.add_subplot(gs[0, 0]), gfp, gm,
                "GFP core×32 — where the band was found", "A")
    curve_panel(fig.add_subplot(gs[0, 1]), dio3, dm,
                "DIO3 core×32 — HELD OUT, band frozen", "B")
    dose_panel(fig.add_subplot(gs[0, 2]),
               [("GFP", gfp, gm), ("DIO3 (held out)", dio3, dm)], "C")
    null_panel(fig.add_subplot(gs[1, 0]),
               [(k, v, m) for k, v, m in
                (("GFP", gfp_s, gm), ("DIO3 (held out)", dio3_s, dm)) if v], "D")
    quality_panel(fig.add_subplot(gs[1, 1:]),
                  [("GFP", gfp, gm), ("DIO3", dio3, dm)], "E")

    fig.suptitle("Deleting transition_z reverses the local divergence dip on a "
                 "held-out protein — but the effect belongs to the mid-to-late "
                 "trunk, not specifically to L37–45",
                 x=0.008, ha="left", fontsize=11.5, color=INK, y=0.985)
    fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
