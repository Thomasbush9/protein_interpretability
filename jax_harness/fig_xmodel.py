"""Figure: three structure predictors, one mutation signal.

All three use the same AF3-derived trunk design and, as it happens, the same
128-dimensional pair width -- but they are trained independently, so their
channel bases are unrelated. Panel A is the control that establishes this, and
it is what licenses every other panel to use coordinate-free statistics instead
of principal angles.

  A  principal angles between channel subspaces, against chance. Near chance
     means the bases are unrelated, which is the premise for B and C.
  B  representational agreement, each pair measured against the ceiling the
     model sets with its own repeat run. Without that ceiling a low number
     could always be one model being noisy.
  C  each model's probe, and whether combining them beats the best single one.

Note the panel B ordering: what the models share is not obviously predicted by
their architecture, which is why the ceiling and the agreement are drawn on the
same axis rather than described separately.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

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
C_SELF, C_CROSS, C_REF, C_COMB = "#8a8885", "#2a78d6", "#eb6834", "#1baf7a"
NICE = {"boltz2": "Boltz-2", "of3": "OpenFold3", "protenix": "Protenix"}

ap = argparse.ArgumentParser()
ap.add_argument("--xmodel", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
S = json.load(open(a.xmodel))


def nice(lab):
    for k, v in NICE.items():
        lab = lab.replace(k, v)
    return lab


fig = plt.figure(figsize=(14.4, 4.8))
gs = fig.add_gridspec(1, 3, wspace=0.48, width_ratios=[1.0, 1.15, 1.1])

# --- A: are the channel bases related at all? ------------------------------
axA = fig.add_subplot(gs[0, 0])
nc = S["principal_angles_negative_control"]
pairs = sorted(nc["pairs"])
vals = [np.mean(list(nc["pairs"][p].values())) for p in pairs]
x = np.arange(len(pairs))
axA.bar(x, vals, 0.6, color=C_REF, alpha=0.9, zorder=3)
axA.axhline(nc["chance"], color=INK, lw=1.6, ls="--", zorder=5)
axA.annotate(f"chance for unrelated bases  ({nc['chance']:.3f})",
             xy=(0.02, nc["chance"]), xycoords=("axes fraction", "data"),
             xytext=(0, 6), textcoords="offset points", fontsize=7.4, color=INK)
axA.set_xticks(x)
axA.set_xticklabels([nice(p).replace("|", "\nvs ") for p in pairs], fontsize=7.4)
axA.set_ylabel(r"mean cos$^2$ of principal angles")
axA.set_ylim(0, max(vals) * 1.7)
axA.set_title("A  The 128-dim spaces are unrelated\n     bases — as they must be",
              color=INK, fontsize=9.5, loc="left", pad=8)
axA.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)

# --- B: agreement against the self-repeat ceiling --------------------------
axB = fig.add_subplot(gs[0, 1])
keys = [k for k in S["cka"] if "repeat" in k] + [k for k in S["cka"] if "repeat" not in k]
y = np.arange(len(keys))[::-1]
for yy, k in zip(y, keys):
    self_ = "repeat" in k
    col = C_SELF if self_ else C_CROSS
    c, r = S["cka"][k], S["rsa"][k]
    axB.plot([c["ci_lo"], c["ci_hi"]], [yy + 0.14] * 2, color=col, lw=2.6,
             zorder=4, solid_capstyle="round")
    axB.scatter([c["mean"]], [yy + 0.14], s=48, color=col, zorder=5,
                edgecolor=SURF, linewidth=1.5)
    axB.scatter([r["mean"]], [yy - 0.16], s=40, color=col, zorder=5,
                marker="D", edgecolor=SURF, linewidth=1.3, alpha=0.75)
axB.set_yticks(y)
SHORT = {"Boltz-2": "B2", "OpenFold3": "OF3", "Protenix": "PTX"}
def _sh(k):
    k = nice(k)
    for a_, b_ in SHORT.items():
        k = k.replace(a_, b_)
    return k.replace("|", " · ").replace(" (repeat)", "  (own repeat)")
axB.set_yticklabels([_sh(k) for k in keys], fontsize=7.8)
axB.set_xlim(0, 1.06)
axB.set_xlabel("agreement (circle = CKA, diamond = RSA ρ)")
axB.set_title("B  Grey = the ceiling each model sets\n     against itself",
              color=INK, fontsize=9.5, loc="left", pad=8)
axB.grid(True, axis="x", color=GRID, lw=0.8, zorder=0); axB.set_axisbelow(True)

# --- C: complementarity ----------------------------------------------------
axC = fig.add_subplot(gs[0, 2])
cm = S["complementarity"]
ORDER = [("protenix", C_CROSS), ("of3", C_CROSS), ("boltz2", C_CROSS),
         ("concatenated features", C_COMB), ("averaged predictions", C_COMB)]
y = np.arange(len(ORDER))[::-1]
best = max(cm[m]["mean"] for m in ("boltz2", "of3", "protenix"))
for yy, (k, col) in zip(y, ORDER):
    b = cm[k]
    axC.plot([b["ci_lo"], b["ci_hi"]], [yy, yy], color=col, lw=2.8, zorder=4,
             solid_capstyle="round")
    axC.scatter([b["mean"]], [yy], s=54, color=col, zorder=5, edgecolor=SURF,
                linewidth=1.7)
axC.axvline(best, color=INK, lw=1.3, ls="--", zorder=3)
axC.annotate("best single model", xy=(best, len(ORDER) - 0.6), xytext=(5, 0),
             textcoords="offset points", fontsize=7.4, color=INK)
g = cm["averaged minus best single"]
axC.set_yticks(y)
axC.set_yticklabels([nice(k) for k, _ in ORDER], fontsize=8)
axC.set_xlabel("held-out Spearman vs DMS")
axC.set_title(f"C  All three decode it; combining buys "
              f"{g['gap']:+.3f}\n     [{g['ci_lo']:+.3f}, {g['ci_hi']:+.3f}] "
              f"— interval includes zero",
              color=INK, fontsize=9.5, loc="left", pad=8)
axC.grid(True, axis="x", color=GRID, lw=0.8, zorder=0); axC.set_axisbelow(True)
axC.margins(x=0.16, y=0.14)

fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=SURF)
print(f"wrote {a.out}")
