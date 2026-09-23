"""Paper figure: emitted metrics are mutation-insensitive.

Main figure, two rows:
  A. Per-assay Spearman rho between each emitted metric (TM-to-WT, CA RMSD,
     mean pLDDT) and the DMS assay score, panel5 paired cohort (25 ProteinGym
     assays x 100 single mutants, all three models). Insensitivity = the
     metric carries no functional-mutation signal, so rho sits at zero.
     Boltz-2's structure-module sampling noise widens its raw metric
     distributions but carries no signal either -- which is why the raw
     violins live in the supplement, not here.
  B. GFP random-mutation dose ladder (0/5/10/20/40/70% of residues), same
     three metrics vs mutation load: flat through the realistic regime.

Supplement: the raw metric distributions (violins) behind row A.

TM from the per-model caches, pLDDT from the captures, RMSD is Kabsch on the
stored CA coordinates; dose-ladder TM/RMSD computed from runs/dose_{model}.npz.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent))
import geom  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
_h = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_s = next((f for f in ("Nimbus Sans", "DejaVu Sans", "Helvetica", "Arial") if f in _h), None)
plt.rcParams.update({**({"font.family": "sans-serif", "font.sans-serif": [_s]} if _s else {}),
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
                     "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})
PALETTE = ["#2a78d6", "#eb6834", "#159a8c"]
MODELS = ["boltz2", "of3", "protenix"]
NICE = {"boltz2": "Boltz-2", "of3": "OpenFold3", "protenix": "Protenix"}
METRICS = [("tm", "TM-score to WT"), ("rmsd", u"CA RMSD to WT (Å)"), ("plddt", "mean pLDDT")]

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
GFP_LEN = 238


def collect_panel5():
    assays = json.load(open(W / "runs/geometry_panel5.json"))["assays"]
    out = {}
    for m in MODELS:
        tmz = np.load(W / f"runs/tm_panel5_{m}.npz")
        vals = {k: [] for k, _ in METRICS}
        rho = {k: [] for k, _ in METRICS}
        for a in assays:
            d = np.load(W / f"runs/xmodel_panel5/xm_{m}_r1_{a}.npz")
            caw = d["ca_wt"].astype(float)
            y = d["score"]
            per = {"tm": tmz[a],
                   "rmsd": np.array([geom.kabsch_rmsd(c.astype(float), caw) for c in d["ca"]]),
                   "plddt": d["plddt_mean"]}
            for k, _ in METRICS:
                vals[k].append(per[k])
                rho[k].append(spearmanr(per[k], y).correlation)
        out[m] = dict(vals={k: np.concatenate(v) for k, v in vals.items()},
                      rho={k: np.array(v) for k, v in rho.items()})
    return assays, out


def collect_dose():
    """gfp_dose2 replicated ladder: per model, per load, per-seed metric arrays."""
    man = list(csv.DictReader(open(W / "data/gfp_dose2/manifest.csv")))
    loads = sorted({int(r["n_mut"]) for r in man if r["mode"] == "random"})
    out = {}
    for m in MODELS:
        z = np.load(W / f"runs/dose2_{m}.npz", allow_pickle=True)
        caw = z["gfp_wt__ca"].astype(float)
        per = {k: {n: [] for n in loads} for k, _ in METRICS}
        for r in man:
            if r["mode"] != "random":
                continue
            n = int(r["n_mut"])
            ca = z[f"{r['id']}__ca"].astype(float)
            per["tm"][n].append(geom.tm_score(ca, caw))
            per["rmsd"][n].append(geom.kabsch_rmsd(ca, caw))
            per["plddt"][n].append(float(z[f"{r['id']}__plddt"].mean()))
        out[m] = {k: np.array([per[k][n] for n in loads]) for k, _ in METRICS}
        out[m]["wt_plddt"] = float(z["gfp_wt__plddt"].mean())
    return loads, out


def fig_main(assays, panel, loads, dose, out):
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.6))

    # Row A -- per-assay rho(metric, assay score): no functional-mutation signal.
    rng = np.random.default_rng(0)
    for ax, (key, label) in zip(axes[0], METRICS):
        ax.axhline(0, color=INK2, lw=0.8, zorder=1)
        for i, m in enumerate(MODELS):
            r = panel[m]["rho"][key]
            ax.scatter(i + rng.uniform(-0.16, 0.16, r.size), r,
                       s=9, color=PALETTE[i], alpha=0.65, lw=0, zorder=3)
            q1, med, q3 = np.percentile(r, [25, 50, 75])
            ax.vlines(i + 0.30, q1, q3, color=PALETTE[i], lw=2.5)
            ax.scatter([i + 0.30], [med], s=16, color=INK, zorder=5)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([NICE[m] for m in MODELS])
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlim(-0.55, len(MODELS) - 0.30)
        ax.set_title(label, fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Spearman rho vs assay score", color=INK)

    # Row B -- GFP random-mutation dose ladder: flat through the realistic
    # regime. Median line + min-max band over the independent draws per load.
    x = np.array(loads, dtype=float)
    n_reps = dose[MODELS[0]]["tm"].shape[1]
    for ax, (key, label) in zip(axes[1], METRICS):
        for i, m in enumerate(MODELS):
            v = dose[m][key]
            ax.fill_between(x, v.min(axis=1), v.max(axis=1),
                            color=PALETTE[i], alpha=0.15, lw=0)
            ax.plot(x, np.median(v, axis=1), color=PALETTE[i], lw=1.8,
                    marker="o", ms=3.5, label=NICE[m], zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel(f"random mutations (of {GFP_LEN} residues)", color=INK)
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xticks(loads)
        ax.set_xticklabels([str(n) for n in loads], fontsize=7.5)
        ax.minorticks_off()
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        if key in ("tm", "plddt"):
            ax.set_ylim(0, 1.04)
    axes[1, 0].legend(frameon=False, loc="lower left", fontsize=8)

    fig.subplots_adjust(left=0.075, right=0.98, top=0.88, bottom=0.09,
                        hspace=0.62, wspace=0.30)
    fig.text(0.075, 0.945, "A", fontsize=11, color=INK, weight="bold")
    fig.text(0.105, 0.945, f"ProteinGym single mutants: emitted metrics do not track "
                           f"functional effect ({len(assays)} assays × 100 variants)",
             fontsize=9.5, color=INK)
    fig.text(0.075, 0.455, "B", fontsize=11, color=INK, weight="bold")
    fig.text(0.105, 0.455, f"GFP random-mutation ladder ({loads[0]}–{loads[-1]} of "
                           f"{GFP_LEN} residues, {n_reps} draws per load, "
                           f"median and min–max range)",
             fontsize=9.5, color=INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def fig_supp(assays, panel, out):
    n = panel[MODELS[0]]["vals"]["tm"].size
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.1))
    for ax, (key, label) in zip(axes, METRICS):
        logy = key == "rmsd"
        for i, m in enumerate(MODELS):
            v = panel[m]["vals"][key]
            vv = np.log10(v) if logy else v
            parts = ax.violinplot([vv], positions=[i], widths=0.72,
                                  showextrema=False, showmedians=False)
            for b in parts["bodies"]:
                b.set_facecolor(PALETTE[i]); b.set_alpha(0.35)
                b.set_edgecolor(PALETTE[i]); b.set_linewidth(1.0)
            q1, med, q3 = np.percentile(vv, [25, 50, 75])
            ax.vlines(i, q1, q3, color=PALETTE[i], lw=2.5)
            ax.scatter([i], [med], s=16, color=INK, zorder=5)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([NICE[m] for m in MODELS])
        ax.set_ylabel(label, color=INK)
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        if logy:
            ticks = [0.1, 0.3, 1, 3, 10, 30]
            ax.set_yticks(np.log10(ticks))
            ax.set_yticklabels([f"{t:g}" for t in ticks])
            ax.set_ylim(np.log10(0.04), np.log10(45))
        else:
            ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_title(f"Raw metric distributions, ProteinGym single mutants "
                      f"({len(assays)} assays × 100, n={n} per model)",
                      loc="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=W / "figures_paper")
    args = ap.parse_args()

    assays, panel = collect_panel5()
    loads, dose = collect_dose()
    fig_main(assays, panel, loads, dose, args.outdir / "metric_insensitivity.png")
    fig_supp(assays, panel, args.outdir / "metric_insensitivity_supp_distributions.png")

    stats = {"rho": {m: {k: {"median": round(float(np.median(panel[m]["rho"][k])), 4),
                             "q25": round(float(np.percentile(panel[m]["rho"][k], 25)), 4),
                             "q75": round(float(np.percentile(panel[m]["rho"][k], 75)), 4),
                             "median_abs": round(float(np.median(np.abs(panel[m]["rho"][k]))), 4)}
                         for k, _ in METRICS} for m in MODELS},
             "vals": {m: {k: {"median": round(float(np.median(panel[m]["vals"][k])), 4),
                              "q25": round(float(np.percentile(panel[m]["vals"][k], 25)), 4),
                              "q75": round(float(np.percentile(panel[m]["vals"][k], 75)), 4)}
                          for k, _ in METRICS} for m in MODELS},
             "dose": {m: {**{k: {"median": [round(float(x), 4) for x in np.median(dose[m][k], axis=1)],
                                 "min": [round(float(x), 4) for x in dose[m][k].min(axis=1)],
                                 "max": [round(float(x), 4) for x in dose[m][k].max(axis=1)]}
                             for k, _ in METRICS},
                          "wt_plddt": round(dose[m]["wt_plddt"], 4)} for m in MODELS},
             "_meta": {"cohort": "panel5 paired", "n_assays": len(assays),
                       "assays": assays, "dose_cohort": "gfp_dose2",
                       "dose_loads_n_mut": loads,
                       "dose_reps": int(dose[MODELS[0]]["tm"].shape[1])}}
    sp = args.outdir / "metric_insensitivity.json"
    sp.write_text(json.dumps(stats, indent=2))
    print(f"wrote {sp}")


if __name__ == "__main__":
    main()
