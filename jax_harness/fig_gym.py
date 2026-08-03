"""Figures for the ProteinGym stability results: what the rho actually is.

Reads the v2 analysis outputs (`compare_io12_v2.json`, `rsa_v2.json`) so the
figure and the report table cannot disagree. Three presentation requirements from
the August 2026 audit are built in:

  * assay-level dots sit behind every pooled bar, so the reader sees the spread
    rather than a bar and a whisker whose meaning is ambiguous;
  * error bars are 95 % confidence intervals from a hierarchical bootstrap over
    ASSAYS, labelled as such, and are never the standard deviation across splits;
  * the baselines that decide whether this is a claim about the model --
    substitution chemistry and residue identity -- are plotted next to the
    internal features, not omitted.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from probe_gym import grouped_split, select_k, ridge_fit, ridge_pred, spearman  # noqa: E402

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
    "axes.spines.top": False, "axes.spines.right": False,
})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
C_INT, C_CHEM, C_OUT = "#2a78d6", "#eb6834", "#8a8885"

ap = argparse.ArgumentParser()
ap.add_argument("--io", default="runs/compare_io12_v2.json")
ap.add_argument("--rsa", default="runs/rsa_v2.json")
ap.add_argument("--out", required=True)
a = ap.parse_args()

io = json.load(open(a.io))
pred = io["predictors"]
rsa = json.load(open(a.rsa))

fig = plt.figure(figsize=(14.0, 8.8))
gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.30)

# --- A: every predictor, assay dots behind pooled bars -----------------
axA = fig.add_subplot(gs[0, :2])
ORDER = ["internal", "chemistry", "identity", "output_ridge", "pLDDT_mean",
         "TM_to_WT", "pLDDT_site", "nearest_position"]
LABEL = {"internal": "internal\n(Pairformer)", "chemistry": "substitution\nchemistry",
         "identity": "residue\nidentity", "output_ridge": "output ridge\n(TM+pLDDT)",
         "pLDDT_mean": "pLDDT\nmean", "TM_to_WT": "TM to\nwild type",
         "pLDDT_site": "pLDDT at\nthe site", "nearest_position": "nearest\nposition"}
COL = {"internal": C_INT, "chemistry": C_CHEM, "identity": C_CHEM}
x = np.arange(len(ORDER))
rng = np.random.default_rng(0)
for i, k in enumerate(ORDER):
    p = pred[k]
    col = COL.get(k, C_OUT)
    axA.bar(i, p["mean"], 0.64, color=col, alpha=0.85, zorder=3)
    lo, hi = p["ci_lo"], p["ci_hi"]
    if np.isfinite(lo):
        axA.plot([i, i], [lo, hi], color=INK, lw=2.0, zorder=5,
                 solid_capstyle="butt")
    # one dot per assay, jittered, drawn over the bar
    per = [np.nanmean(v) for v in p["per_assay"].values() if len(v)]
    axA.scatter(i + rng.uniform(-0.16, 0.16, len(per)), per, s=13, zorder=6,
                color=INK, alpha=0.55, edgecolor=SURF, linewidth=0.6)
axA.axhline(0, color=INK2, lw=0.8)
axA.set_xticks(x)
axA.set_xticklabels([LABEL[k] for k in ORDER], fontsize=7.5)
axA.set_ylabel("Spearman with ProteinGym DMS_score\non held-out positions")
axA.set_title("A  The internal state beats the model's own output — and, by a much "
              "narrower margin, substitution chemistry",
              color=INK, fontsize=10, loc="left", pad=8)
axA.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
axA.set_axisbelow(True)
axA.annotate("bars = mean over 12 assays · black line = 95 % CI\n"
             "(hierarchical bootstrap over assays) · dots = one assay",
             xy=(0.99, 0.97), xycoords="axes fraction", ha="right", va="top",
             fontsize=7.5, color=INK2)

# --- B: the paired gaps that carry the claim ---------------------------
axB = fig.add_subplot(gs[0, 2])
GAPS = [("vs TM to WT", "internal - TM_to_WT"),
        ("vs output ridge", "internal - output_ridge"),
        ("vs identity", "internal - identity"),
        ("vs chemistry", "internal - chemistry")]
y = np.arange(len(GAPS))
for i, (lab, key) in enumerate(GAPS):
    g = io["gaps"][key]
    col = C_CHEM if "chem" in key or "identity" in key else C_INT
    axB.plot([g["ci_lo"], g["ci_hi"]], [i, i], color=col, lw=2.4, zorder=3,
             solid_capstyle="round")
    axB.scatter([g["gap"]], [i], s=46, color=col, zorder=4, edgecolor=SURF,
                linewidth=2)
    axB.annotate(f"{g['gap']:+.3f}  ({g['wins']}/{g['splits']})",
                 xy=(g["ci_hi"], i), xytext=(6, 0), textcoords="offset points",
                 va="center", fontsize=7.5, color=INK2)
axB.axvline(0, color=INK2, lw=1.0, zorder=2)
axB.set_yticks(y)
axB.set_yticklabels([lab for lab, _ in GAPS], fontsize=8)
axB.set_xlabel("internal minus baseline (paired)")
axB.set_title("B  Every gap clears zero,\n     but not by the same distance",
              color=INK, fontsize=10, loc="left", pad=8)
axB.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
axB.set_axisbelow(True)
axB.margins(x=0.28)

# --- C: what a median split looks like ---------------------------------
for j, assay in enumerate(["RCRO", "RS15"]):
    ax = fig.add_subplot(gs[1, 1] if j else gs[1, 0])
    f = glob.glob(f"runs/gym2_{assay}*.npz")[0]
    d = np.load(f, allow_pickle=True)
    ys, pos = d["score"], d["pos"]
    X = np.concatenate([d["kl_glob"], d["kl_site"],
                        np.linalg.norm(d["dz_site"], axis=-1),
                        np.linalg.norm(d["ds_site"], axis=-1)], axis=1)
    runs_ = []
    for s_ in range(5):
        tr, te = grouped_split(pos, 0.25, np.random.default_rng(s_))
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xs = (X - mu) / sd
        idx = select_k(Xs[tr], ys[tr], 16)
        w_ = ridge_fit(Xs[tr][:, idx],
                       (ys[tr] - ys[tr].mean()) / (ys[tr].std() + 1e-8), 1.0)
        p = ridge_pred(w_, Xs[te][:, idx])
        runs_.append((spearman(p, ys[te]), p, te))
    rhos = np.array([r[0] for r in runs_])
    med = int(np.argsort(rhos)[len(rhos) // 2])
    rho, p, te = runs_[med]
    ax.scatter(p, ys[te], s=16, color=C_INT, alpha=0.7, edgecolor="none", zorder=3)
    ax.set_xlabel("predicted (internal state)")
    ax.set_ylabel("ProteinGym DMS_score")
    ax.set_title(f"{'D' if j else 'C'}  {assay}: median split rho = {rho:+.3f}\n"
                 f"     across 5 splits {rhos.mean():+.3f} ± {rhos.std():.3f} "
                 f"(n={te.sum()} held out)", color=INK, fontsize=9.5,
                 loc="left", pad=8)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

# --- E: RSA trajectory, trunk -> output --------------------------------
axE = fig.add_subplot(gs[1, 2])
for i, (k, v) in enumerate(sorted(rsa.items())):
    axE.plot(v["descriptive_curve_cosine"], color=C[i % len(C)], lw=1.5,
             alpha=0.8, label=k.split("_")[0], zorder=3)
mean_pf = np.mean([v["descriptive_curve_cosine"] for v in rsa.values()], axis=0)
axE.plot(mean_pf, color=INK, lw=2.6, zorder=4, label="mean")
for lab, key, st in (("distogram", "rsa_disto_vs_exp", "--"),
                     ("structure", "rsa_struct_vs_exp", ":")):
    m = np.mean([v[key]["rho"] for v in rsa.values()])
    axE.axhline(m, color=INK2, ls=st, lw=1.4, zorder=2)
    axE.text(1, m, f" {lab} ({m:+.3f}, n.s.)", fontsize=7.5, color=INK2,
             va="bottom")
ho = [v["held_out_layer_cosine"]["rho_held_out_half"] for v in rsa.values()
      if v.get("held_out_layer_cosine")]
if ho:
    axE.axhline(np.mean(ho), color=C_CHEM, lw=1.8, zorder=5)
    axE.text(1, np.mean(ho), f" held-out layer ({np.mean(ho):+.3f})",
             fontsize=7.5, color=C_CHEM, va="bottom")
axE.axhline(0, color=INK2, lw=0.8)
axE.set_xlabel("Pairformer layer (0-63)")
axE.set_ylabel("partial RSA vs experiment")
axE.set_title("E  Geometry match survives in the trunk, not downstream\n"
              "     (curves are DESCRIPTIVE: selected and scored on all pairs)",
              color=INK, fontsize=9.5, loc="left", pad=8)
axE.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
axE.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
axE.set_axisbelow(True)

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
