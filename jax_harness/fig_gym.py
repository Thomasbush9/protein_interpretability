"""Figures for the ProteinGym stability results: what the 0.48 actually is."""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from geom import tm_score
from probe_gym import grouped_split, select_k, ridge_fit, ridge_pred, spearman

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
_have = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_sans = next((f for f in ("Nimbus Sans","DejaVu Sans","Helvetica","Arial") if f in _have), None)
plt.rcParams.update({**({"font.family":"sans-serif","font.sans-serif":[_sans]} if _sans else {}),
 "figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","axes.edgecolor":GRID,
 "axes.labelcolor":INK2,"text.color":INK,"xtick.color":INK2,"ytick.color":INK2,
 "font.size":9,"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
C = ["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4"]

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
io = json.load(open("runs/compare_io.json"))
rsa = {}
for f in glob.glob("runs/rsa_*.json"):
    rsa.update(json.load(open(f)))
names = sorted(io)

fig = plt.figure(figsize=(13.5, 8.6))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

# --- A: predictor comparison, per assay + pooled -----------------------
axA = fig.add_subplot(gs[0, :2])
preds = [("internal","internal (pairformer)"),("tm","TM to WT (structure)"),
         ("pl","pLDDT mean"),("pls","pLDDT at site"),("base","position only")]
x = np.arange(len(names)+1); w = 0.16
for i,(k,lab) in enumerate(preds):
    vals = [np.mean(io[n][k]) for n in names]
    errs = [np.std(io[n][k]) for n in names]
    pooled = np.concatenate([io[n][k] for n in names])
    vals.append(pooled.mean()); errs.append(pooled.std())
    axA.bar(x + (i-2)*w, vals, w, yerr=errs, color=C[i%len(C)], label=lab,
            zorder=3, error_kw=dict(lw=0.9, ecolor=INK2, capsize=2))
axA.axhline(0, color=INK2, lw=0.8)
axA.set_xticks(x); axA.set_xticklabels([n for n in names]+["POOLED"], fontsize=8)
axA.set_ylabel("Spearman(predicted, measured dG)\non held-out positions")
axA.set_title("A  Internal state predicts stability better than the model's own output",
              color=INK, fontsize=10, loc="left", pad=8)
axA.legend(frameon=False, fontsize=7.5, ncol=2)
axA.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); axA.set_axisbelow(True)

# --- B: what rho=0.63 / 0.39 look like as scatter ----------------------
for j, assay in enumerate(["RCRO", "RS15"]):
    ax = fig.add_subplot(gs[0, 2] if j == 0 else gs[1, 2])
    f = glob.glob(f"runs/gym2_{assay}*.npz")[0]
    d = np.load(f, allow_pickle=True)
    y, pos = d["score"], d["pos"]
    X = np.concatenate([d["kl_glob"], d["kl_site"],
                        np.linalg.norm(d["dz_site"],axis=-1),
                        np.linalg.norm(d["ds_site"],axis=-1)], axis=1)
    runs_ = []
    for s_ in range(5):
        tr, te = grouped_split(pos, 0.25, np.random.default_rng(s_))
        mu, sd = X[tr].mean(0), X[tr].std(0)+1e-8; Xs = (X-mu)/sd
        idx = select_k(Xs[tr], y[tr], 16)
        w_ = ridge_fit(Xs[tr][:,idx], (y[tr]-y[tr].mean())/(y[tr].std()+1e-8), 1.0)
        p = ridge_pred(w_, Xs[te][:,idx])
        runs_.append((spearman(p, y[te]), p, te))
    rhos = np.array([r[0] for r in runs_])
    med = int(np.argsort(rhos)[len(rhos)//2])          # show the MEDIAN split
    rho, p, te = runs_[med]
    ax.scatter(p, y[te], s=16, color=C[0], alpha=0.7, edgecolor="none", zorder=3)
    ax.set_xlabel("predicted (internal state)"); ax.set_ylabel("measured dG")
    ax.set_title(f"{'B' if j==0 else 'C'}  {assay}: median split rho = {rho:+.3f}\n"
                 f"     across 5 splits {rhos.mean():+.3f} +/- {rhos.std():.3f}  "
                 f"(n={te.sum()} held-out)", color=INK, fontsize=9.5, loc="left", pad=8)
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)

# --- D: layer sweep, single feature at a time --------------------------
axD = fig.add_subplot(gs[1, 0])
blocks = ["kl_glob","kl_site","dz_site","ds_site"]
for bi, blk in enumerate(blocks):
    curves = []
    for n in names:
        f = glob.glob(f"runs/gym2_{n}*.npz")[0]; d = np.load(f, allow_pickle=True)
        L = int(d["n_layers"]); y = d["score"]
        M = d[blk] if blk.startswith("kl") else np.linalg.norm(d[blk], axis=-1)
        curves.append([abs(spearman(M[:,l], y)) for l in range(L)])
    m = np.mean(curves, axis=0)
    axD.plot(m, color=C[bi], lw=2, label=blk, zorder=3)
axD.set_xlabel("Pairformer layer (0-63)")
axD.set_ylabel("|Spearman| with measured dG\n(in-sample, descriptive)")
axD.set_title("D  Which layer, which quantity (mean of 4 assays, IN-SAMPLE:\n     not comparable to the held-out values in A)", color=INK,
              fontsize=9.5, loc="left", pad=8)
axD.legend(frameon=False, fontsize=7.5)
axD.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); axD.set_axisbelow(True)

# --- E: RSA trajectory, trunk -> output --------------------------------
axE = fig.add_subplot(gs[1, 1])
for i, (k, v) in enumerate(sorted(rsa.items())):
    axE.plot(v["rsa_pf_vs_exp"], color=C[i%len(C)], lw=1.6, alpha=0.85,
             label=k.split("_")[0], zorder=3)
mean_pf = np.mean([v["rsa_pf_vs_exp"] for v in rsa.values()], axis=0)
axE.plot(mean_pf, color=INK, lw=2.6, zorder=4, label="mean")
for lab, key, st in (("distogram","rsa_disto_vs_exp","--"),
                     ("structure","rsa_struct_vs_exp",":")):
    m = np.mean([v[key] for v in rsa.values()])
    axE.axhline(m, color=INK2, ls=st, lw=1.4, zorder=2)
    axE.text(1, m, f" {lab} ({m:+.3f})", fontsize=7.5, color=INK2, va="bottom")
axE.axhline(0, color=INK2, lw=0.8)
axE.set_xlabel("Pairformer layer (0-63)")
axE.set_ylabel("partial RSA vs experiment")
axE.set_title("E  Geometry match survives in the trunk, not downstream", color=INK,
              fontsize=10, loc="left", pad=8)
axE.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
axE.grid(True, axis="y", color=GRID, lw=0.8, zorder=0); axE.set_axisbelow(True)

fig.savefig(a.out, dpi=170, bbox_inches="tight"); print(f"wrote {a.out}")
