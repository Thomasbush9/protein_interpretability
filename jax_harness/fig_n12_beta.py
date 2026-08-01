"""N=12 predictor comparison, and the beta dose-response that refuted the hypothesis."""
from __future__ import annotations
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
INK,INK2,GRID="#0b0b0b","#52514e","#e6e5e1"
_h={f.name for f in matplotlib.font_manager.fontManager.ttflist}
_s=next((f for f in ("Nimbus Sans","DejaVu Sans","Helvetica","Arial") if f in _h),None)
plt.rcParams.update({**({"font.family":"sans-serif","font.sans-serif":[_s]} if _s else {}),
 "figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","axes.edgecolor":GRID,
 "axes.labelcolor":INK2,"text.color":INK,"xtick.color":INK2,"ytick.color":INK2,
 "font.size":9,"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
C=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4"]
d=json.load(open("runs/compare_io12.json"))
names=sorted(d, key=lambda k: -np.mean(d[k]["internal"]))
fig=plt.figure(figsize=(13.5,7.6)); gs=fig.add_gridspec(2,2,hspace=.45,wspace=.24,height_ratios=[1.15,1])

ax=fig.add_subplot(gs[0,:])
x=np.arange(len(names)+1); w=.16
for i,(k,lab) in enumerate([("internal","internal (pairformer)"),("pl","pLDDT mean"),
                            ("tm","TM to WT"),("pls","pLDDT at site"),("base","position only")]):
    v=[np.mean(d[n][k]) for n in names]; e=[np.std(d[n][k]) for n in names]
    pool=np.concatenate([d[n][k] for n in names]); v.append(pool.mean()); e.append(pool.std())
    ax.bar(x+(i-2)*w,v,w,yerr=e,color=C[i],label=lab,zorder=3,
           error_kw=dict(lw=.8,ecolor=INK2,capsize=1.6))
ax.axhline(0,color=INK2,lw=.8); ax.set_xticks(x)
ax.set_xticklabels(names+["POOLED"],fontsize=7.5,rotation=30,ha="right")
ax.set_ylabel("Spearman(predicted, measured dG)\non held-out positions")
ax.set_title("A  Twelve Tsuboyama stability assays: internal state beats the model's own output in 57/60 splits",
             color=INK,fontsize=10,loc="left",pad=8)
ax.legend(frameon=False,fontsize=7.5,ncol=5)
ax.grid(True,axis="y",color=GRID,lw=.8,zorder=0); ax.set_axisbelow(True)

ax=fig.add_subplot(gs[1,0])
ii=np.array([np.mean(d[n]["internal"]) for n in names])
tt=np.array([np.mean(d[n]["tm"]) for n in names])
ax.scatter(tt,ii,s=46,color=C[0],zorder=3,edgecolor="none")
lim=[-.05,.75]; ax.plot(lim,lim,ls=":",color=INK2,lw=1.2,zorder=2)
ax.text(.52,.56,"y = x",fontsize=8,color=INK2)
for n,a_,b_ in zip(names,tt,ii): ax.annotate(n,(a_,b_),fontsize=6.2,color=INK2,
                                             xytext=(3,3),textcoords="offset points")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("TM to WT (structure)"); ax.set_ylabel("internal (pairformer)")
ax.set_title("B  Every assay above the diagonal",color=INK,fontsize=10,loc="left",pad=8)
ax.grid(True,color=GRID,lw=.8,zorder=0); ax.set_axisbelow(True)

ax=fig.add_subplot(gs[1,1])
b=np.array([1.0,1.5,2.0,3.0])
ax.plot(b,[0.435,0.294,0.283,0.018],"-o",color=C[0],lw=2,ms=6,label="TM to WT vs dG",zorder=3)
ax.plot(b,[0.037,0.139,0.208,0.299],"-o",color=C[1],lw=2,ms=6,label="pLDDT vs dG",zorder=3)
ax.plot(b,[0.161,0.208,0.156,-0.139],"-o",color=C[2],lw=2,ms=6,label="ensemble spread vs dG",zorder=3)
ax2=ax.twinx()
ax2.plot(b,[0.990,0.803,0.614,0.303],"--",color=INK2,lw=1.4,zorder=2)
ax2.set_ylabel("WT ensemble spread (mean pairwise TM)",color=INK2,fontsize=8)
ax2.tick_params(labelsize=8); ax2.spines["top"].set_visible(False)
ax.axhline(0,color=INK2,lw=.8)
ax.set_xlabel("beta (scale on pair-derived attention biases)")
ax.set_ylabel("Spearman with measured dG")
ax.set_title("C  Widening the sampler does NOT free the signal\n     (dashed: ensemble widens 0.99 -> 0.30)",
             color=INK,fontsize=9.5,loc="left",pad=8)
ax.legend(frameon=False,fontsize=7.5,loc="lower left")
ax.grid(True,axis="y",color=GRID,lw=.8,zorder=0); ax.set_axisbelow(True)
fig.savefig("figures/n12_beta.png",dpi=170,bbox_inches="tight"); print("wrote figures/n12_beta.png")
