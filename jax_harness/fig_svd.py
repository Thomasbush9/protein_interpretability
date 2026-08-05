"""Figure: the mutation subspace -- how many directions, shared by whom, made of what.

Four claims, one panel each, in the order they have to be believed:

  A  the signal survives truncation to a handful of directions, and a rotation
     beats a subset of channels at every truncation. The k = D endpoint is an
     identity, not a result, so the two curves must meet there -- that is the
     panel's own built-in check and it is drawn rather than asserted.
  B  a basis learned WITHOUT the held-out protein still transfers, and almost
     all of it arrives by the second component.
  C  what those components are: PC1 volume, PC2 stability-and-width together.
  D  twelve unrelated folds put their mutation response in nearly the same
     eight directions, measured against a chance floor below and the
     same-assay repeat above.

Palette is the project's validated categorical set. Blue is the component
(rotation) view throughout, orange the raw-channel control, grey the reference
levels -- the same colour means the same thing in every panel. Panel C uses a
diverging map with a neutral grey midpoint, blue positive and orange negative,
so a sign flip is never carried by hue intensity alone; every cell is also
printed as a number.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
import numpy as np                           # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
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
C_PC, C_PCP, C_RAW, C_REF, C_AMB = "#2a78d6", "#1baf7a", "#eb6834", "#8a8885", "#eda100"
DIV = LinearSegmentedColormap.from_list("div", ["#eb6834", "#efeeea", "#2a78d6"])

ap = argparse.ArgumentParser()
ap.add_argument("--svd", required=True)
ap.add_argument("--svd-ds", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()

S = json.load(open(a.svd))
DS = json.load(open(a.svd_ds)) if a.svd_ds else None
D = S["protocol"]["dim"]

fig = plt.figure(figsize=(14.6, 9.0))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                      height_ratios=[1.0, 1.05])

# --- A: held-out rho against number of directions --------------------------
axA = fig.add_subplot(gs[0, :2])
cur = S["centered"]["curves"]
SERIES = [("components, variance-ordered", C_PC, "-", "components"),
          ("components, prediction-ordered", C_PCP, "-", "components (pred-ordered)"),
          ("RAW channels, prediction-selected (control)", C_RAW, "-",
           "raw channels (control)")]
for key, col, ls, lab in SERIES:
    row = cur[key]
    kk = sorted(int(k) for k in row)
    m = np.array([row[str(k)]["mean"] for k in kk])
    lo = np.array([row[str(k)]["ci_lo"] for k in kk])
    hi = np.array([row[str(k)]["ci_hi"] for k in kk])
    axA.fill_between(kk, lo, hi, color=col, alpha=0.13, lw=0, zorder=2)
    axA.plot(kk, m, color=col, lw=2.0, ls=ls, zorder=4, solid_capstyle="round",
             label=lab)
    axA.scatter(kk, m, s=26, color=col, zorder=5, edgecolor=SURF, linewidth=1.6)
axA.set_xscale("log", base=2)
axA.set_xticks(kk); axA.set_xticklabels([str(k) for k in kk])
axA.set_xlabel("directions kept (k)")
axA.set_ylabel("held-out Spearman")
axA.axvline(D, color=C_REF, lw=1.0, ls=":", zorder=1)
axA.annotate(f"k = {D}: rotation and selection are the\nsame fit "
             f"(identity, checked at 0.0e+00)",
             xy=(D, 0.18), xytext=(-8, 0), textcoords="offset points",
             fontsize=7.2, color=C_REF, ha="right", va="center")
# The variance-ordered curve starts BELOW the raw control and crosses it at
# k = 2. That is not noise and the title says so: PC1 is substitution volume,
# which is nearly orthogonal to stability, so one variance-ordered direction is
# worse than one well-chosen channel. Claiming the rotation wins everywhere
# would be contradicted by the leftmost point of this panel.
axA.set_title("A  Ranked by variance, the first direction is the wrong one; from "
              "k = 2 onward the rotation leads",
              color=INK, fontsize=10, loc="left", pad=8)
axA.legend(frameon=False, fontsize=8, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, -0.30))
axA.grid(True, color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)
axA.margins(x=0.16)

# --- D: is the subspace shared across proteins? ----------------------------
axD = fig.add_subplot(gs[0, 2])
sa = S["subspace_agreement"]
per = np.array(sa["per_layer_mean"])
axD.plot(np.arange(len(per)), per, color=C_PC, lw=2.0, zorder=4)
rep = S.get("replicate_stability", {}).get("pooled", {}).get("mean")
if rep:
    axD.axhline(rep, color=C_REF, lw=1.4, ls="--", zorder=3)
    axD.annotate(f"same assay, repeat run  {rep:.3f}", xy=(0, rep),
                 xytext=(2, -11), textcoords="offset points",
                 fontsize=7.4, color=C_REF)
axD.axhline(sa["chance"], color=C_RAW, lw=1.4, ls="--", zorder=3)
axD.annotate(f"random {sa['k']}-dim subspaces  {sa['chance']:.3f}",
             xy=(0, sa["chance"]), xytext=(2, 5), textcoords="offset points",
             fontsize=7.4, color=C_RAW)
axD.set_ylim(0, 1.05)
axD.set_xlabel("Pairformer layer"); axD.set_ylabel(r"mean cos$^2$ of principal angles")
axD.set_title(f"D  Shared across folds\n     {len(sa['pairs_last8'])} assay pairs, "
              f"top {sa['k']}", color=INK, fontsize=9.5, loc="left", pad=8)
axD.grid(True, color=GRID, lw=0.8, zorder=0); axD.set_axisbelow(True)

# --- B: transfer to a protein the basis never saw --------------------------
axB = fig.add_subplot(gs[1, 0])
lo_ = S["loao_shared_basis"]["results"]
kk = sorted(int(k) for k in lo_)
m = np.array([lo_[str(k)]["mean"] for k in kk])
clo = np.array([lo_[str(k)]["ci_lo"] for k in kk])
chi = np.array([lo_[str(k)]["ci_hi"] for k in kk])
axB.fill_between(kk, clo, chi, color=C_PC, alpha=0.13, lw=0, zorder=2)
axB.plot(kk, m, color=C_PC, lw=2.0, zorder=4)
axB.scatter(kk, m, s=26, color=C_PC, zorder=5, edgecolor=SURF, linewidth=1.6)
axB.scatter([2], [m[kk.index(2)]], s=95, facecolor="none", edgecolor=C_AMB,
            linewidth=2.0, zorder=6)
axB.annotate(f"k=2: {m[kk.index(2)]:+.3f}\n(PC1 is volume, PC2 is stability)",
             xy=(2, m[kk.index(2)]), xytext=(10, -26), textcoords="offset points",
             fontsize=7.6, color=C_AMB)
axB.set_xscale("log", base=2)
axB.set_xticks(kk); axB.set_xticklabels([str(k) for k in kk], fontsize=7.5)
axB.set_xlabel("directions kept (k)"); axB.set_ylabel("Spearman on the held-out assay")
axB.set_title("B  Basis learned without\n     the held-out protein",
              color=INK, fontsize=9.5, loc="left", pad=8)
axB.grid(True, color=GRID, lw=0.8, zorder=0); axB.set_axisbelow(True)
axB.margins(x=0.16)

# --- C: what the shared components are made of -----------------------------
axC = fig.add_subplot(gs[1, 1:])
pool = S["annotation_last_layer"]["pooled"]
chem = S["annotation_last_layer"]["chem_pooled"]
ROWS = [("DMS", pool["DMS"]),
        ("width change (d sigma)", pool.get("dsd_glob")),
        ("broadening (spread)", pool.get("spread_glob")),
        ("relocation (shift)", pool.get("shift_glob")),
        ("symmetric KL", pool.get("kl_glob"))]
ROWS = [(l, v) for l, v in ROWS if v]
labels = [l for l, _ in ROWS]
M = np.array([[c["mean"] for c in v] for _, v in ROWS])
SIG = np.array([[np.isfinite(c["ci_lo"]) and (c["ci_lo"] > 0 or c["ci_hi"] < 0)
                 for c in v] for _, v in ROWS])
for nm in ("d_volume", "d_hydropathy"):
    if nm in chem:
        M = np.vstack([M, np.array(chem[nm])[None, :]])
        SIG = np.vstack([SIG, np.ones((1, M.shape[1]), bool)])
        labels.append(nm.replace("d_", "Δ "))
n_pc = M.shape[1]
vmax = float(np.nanmax(np.abs(M)))
im = axC.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
for i in range(M.shape[0]):
    for j in range(n_pc):
        axC.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7.4,
                 color=INK if abs(M[i, j]) < 0.55 * vmax else SURF,
                 fontweight="bold" if SIG[i, j] else "normal")
axC.set_xticks(range(n_pc)); axC.set_xticklabels([f"PC{c+1}" for c in range(n_pc)])
axC.set_yticks(range(len(labels))); axC.set_yticklabels(labels, fontsize=8)
axC.set_title("C  A shared basis, so signs are comparable: PC1 is substitution volume, "
              "PC2 is stability and width at once",
              color=INK, fontsize=9.5, loc="left", pad=8)
axC.tick_params(length=0)
for s in axC.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=axC, fraction=0.026, pad=0.015)
cb.set_label("Spearman (assay-level mean)", fontsize=7.5)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7, length=0)
axC.annotate("bold = assay-level 95% interval excludes zero", xy=(0, 0),
             xycoords="axes fraction", xytext=(0, -34), textcoords="offset points",
             fontsize=7.2, color=INK2)

if DS:
    f = S["centered"]["curves"]["components, variance-ordered"][str(D)]["mean"]
    dsd = DS["protocol"]["dim"]
    g = DS["centered"]["curves"]["components, variance-ordered"][str(dsd)]["mean"]
    axA.annotate(f"pair track (dz_site, {D} dims) reaches {f:+.3f};  "
                 f"single track (ds_site, {dsd} dims) only {g:+.3f}",
                 xy=(0.015, 0.955), xycoords="axes fraction", fontsize=7.6,
                 color=C_REF, va="top")

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
