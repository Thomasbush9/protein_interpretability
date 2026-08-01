"""Structure vs confidence vs internal state, on the same variants."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
_have={f.name for f in matplotlib.font_manager.fontManager.ttflist}
_sans=next((f for f in ("Nimbus Sans","DejaVu Sans","Helvetica","Arial") if f in _have),None)
plt.rcParams.update({**({"font.family":"sans-serif","font.sans-serif":[_sans]} if _sans else {}),
 "figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","axes.edgecolor":GRID,
 "axes.labelcolor":INK2,"text.color":INK,"xtick.color":INK2,"ytick.color":INK2,
 "font.size":9,"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
C = {"random":"#2a78d6", "core":"#eb6834", "surface":"#1baf7a"}

ap=argparse.ArgumentParser(); ap.add_argument("--bench",required=True); ap.add_argument("--out",required=True)
a=ap.parse_args()
rows=json.loads(Path(a.bench).read_text())
wt=[r for r in rows if r["mode"]=="wt"][0]
groups={m:sorted([r for r in rows if r["mode"]==m], key=lambda r:r["pct_mut"])
        for m in ("random","core","surface")}

fig,axes=plt.subplots(1,3,figsize=(13,3.8))
specs=[("tm_to_wt","TM-score to WT (tmtools)","TM",False),
       ("plddt","pLDDT","confidence",False),
       ("kl","internal state: KL(mutant || WT)","symmetric KL at trunk output",True)]
for ax,(k,title,ylab,logy) in zip(axes,specs):
    for m,rs in groups.items():
        x=[r["pct_mut"] for r in rs]; y=[r[k] for r in rs]
        ax.plot(x,y,"-o",color=C[m],lw=2,ms=5,label=m,zorder=3)
    ax.axhline(wt[k],color=INK2,lw=0.9,ls=":",zorder=2)
    if k=="tm_to_wt":
        ax.axhspan(0.9798,1.0,color="#f0efec",zorder=0)
        ax.text(0.98,0.9798,"  within-WT reproducibility",fontsize=7,color=INK2,va="bottom")
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(title,color=INK,fontsize=10,loc="left",pad=8)
    ax.set_xlabel("% of sequence mutated"); ax.set_ylabel(ylab)
    ax.grid(True,axis="y",color=GRID,lw=0.8,zorder=0); ax.set_axisbelow(True)
axes[0].legend(frameon=False,fontsize=8)
# mark the paper's 20-vs-40 comparison
for ax,k in zip(axes,("tm_to_wt","plddt","kl")):
    # match by id, not by float percentage -- pct_mut is derived and rounds
    rs={r["id"]:r for r in groups["random"]}
    for pid in ("gfp_rand_20","gfp_rand_40"):
        if pid in rs:
            ax.plot([rs[pid]["pct_mut"]],[rs[pid][k]],marker="o",ms=10,
                    mfc="none",mec="#c0392b",mew=1.6,zorder=5)
axes[2].text(0.02,0.02,"red rings: 20% vs 40% random\nTM +4.9%   pLDDT +9.3%   KL +143%",
             transform=axes[2].transAxes,fontsize=7.5,color="#c0392b",va="bottom")
fig.tight_layout(); fig.savefig(a.out,dpi=170); print(f"wrote {a.out}")
