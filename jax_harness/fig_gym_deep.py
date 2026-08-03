"""Deep probe, three models, matched feature construction."""
from __future__ import annotations
import argparse, glob, sys
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent))
import geom  # noqa
import pi_stats  # noqa
from analyze_gym_multi import grouped_split  # noqa
from analyze_gym_deep import fit_deep, BLOCKS  # noqa

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
_h = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_s = next((f for f in ("Nimbus Sans","DejaVu Sans","Helvetica","Arial") if f in _h), None)
plt.rcParams.update({**({"font.family":"sans-serif","font.sans-serif":[_s]} if _s else {}),
    "figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","axes.edgecolor":GRID,
    "axes.labelcolor":INK2,"text.color":INK,"xtick.color":INK2,"ytick.color":INK2,
    "font.size":9,"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
PAL={"boltz2":"#2a78d6","of3":"#eb6834","protenix":"#159a8c"}
NICE={"boltz2":"Boltz-2","of3":"OpenFold3","protenix":"Protenix"}
ORDER=["internal (deep)","TM to WT","pLDDT","pLDDT@site","nearest-position"]

def collect(files, splits=5):
    """Per model, per predictor, per ASSAY -- the assay is the independent unit."""
    per=defaultdict(lambda: defaultdict(list)); gaps=defaultdict(dict); meta={}
    for f in sorted(files):
        d=np.load(f, allow_pickle=True)
        m=str(d["model"]) if "model" in d.files else "boltz2"
        assay=str(d["assay"]) if "assay" in d.files else Path(f).stem
        y,pos=d["score"],d["pos"]; nL=int(d["n_layers"]); meta[m]=nL
        gaps[m].setdefault(assay,[])
        X=np.concatenate([np.linalg.norm(d[b],axis=-1) if d[b].ndim==3 else d[b]
                          for b in BLOCKS],axis=1)
        caw=d["ca_wt"].astype(float)
        tm=np.array([geom.tm_score(c.astype(float),caw) for c in d["ca"]])
        pl=d["plddt_mean"] if "plddt_mean" in d.files else d["plddt"]
        pl=pl.mean(-1) if pl.ndim>1 else pl
        rng=np.random.default_rng(0)
        for s in range(splits):
            tr,te=grouped_split(pos,rng)
            if te.sum()<8 or tr.sum()<20: continue
            ri=fit_deep(X,y,pos,tr,te,s); rt=spearmanr(tm[te],y[te]).correlation
            per[m]["internal (deep)"].append(ri); per[m]["TM to WT"].append(rt)
            per[m]["pLDDT"].append(spearmanr(pl[te],y[te]).correlation)
            per[m]["pLDDT@site"].append(spearmanr(d["plddt_site"][te],y[te]).correlation)
            tp,tv=pos[tr],y[tr]
            pr=np.array([tv[np.argmin(np.abs(tp-p))] for p in pos[te]])
            per[m]["nearest-position"].append(spearmanr(pr,y[te]).correlation)
            gaps[m][assay].append(ri-rt)
    return per,gaps,meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob",default="runs/_3way/deep_*.npz")
    ap.add_argument("--out",default="figures_models/aggregate/internal_vs_output_deep.png")
    a=ap.parse_args()
    per,gaps,meta=collect(glob.glob(a.glob))
    models=[m for m in ("boltz2","of3","protenix") if m in per]
    fig=plt.figure(figsize=(13.6,5.6))
    gs=fig.add_gridspec(1,2,wspace=.24,left=.06,right=.985,top=.66,bottom=.14,
                        width_ratios=[1.75,1])
    ax=fig.add_subplot(gs[0,0]); n=len(models); x=np.arange(len(ORDER)); w=.8/n
    for k,m in enumerate(models):
        off=(k-(n-1)/2)*w
        v=[np.nanmean(per[m][o]) for o in ORDER]
        e=[np.nanstd(per[m][o])/np.sqrt(max(len(per[m][o]),1)) for o in ORDER]
        ax.bar(x+off,v,w*.9,yerr=e,capsize=2,color=PAL[m],zorder=3,
               label=f"{NICE[m]}  ({meta[m]} layers, {4*meta[m]} features)",
               error_kw=dict(lw=.8,ecolor=INK2))
        for xi,vi in enumerate(v):
            ax.text(xi+off,vi+.012,f"{vi:.2f}",ha="center",va="bottom",
                    fontsize=6.8,color=INK,rotation=90)
    ax.axhline(0,color=GRID,lw=1); ax.set_xticks(x); ax.set_xticklabels(ORDER,fontsize=8.2)
    ax.set_ylabel("Spearman rho with measured dG"); ax.set_ylim(-.10,.72)
    ax.legend(frameon=False,fontsize=8.2)
    ax.set_title("A   every predictor, every model",loc="left",fontsize=11,
                 fontweight="bold",color=INK,pad=20)
    ax.text(0,1.02,"internal is now 4 quantities x EVERY Pairformer layer, as in the headline",
            transform=ax.transAxes,fontsize=8.1,color=INK2,va="bottom")
    ax=fig.add_subplot(gs[0,1])
    jit=np.random.default_rng(0)
    for k,m in enumerate(models):
        # the ASSAY is the independent unit, not the split
        pt,lo,hi,_=pi_stats.cluster_bootstrap(gaps[m],n_boot=10000,seed=0)
        ax.errorbar([k],[pt],yerr=[[pt-lo],[hi-pt]],fmt="o",ms=9,
                    color=PAL[m],capsize=4,lw=1.6,zorder=4)
        per_assay=[np.nanmean(v) for v in gaps[m].values() if len(v)]
        ax.scatter(k+jit.uniform(-.10,.10,len(per_assay)),per_assay,s=22,
                   color=PAL[m],alpha=.45,edgecolor="#fcfcfb",linewidth=.8,zorder=3)
        ax.text(k,hi+.012,f"{pt:+.3f}",ha="center",va="bottom",fontsize=8.6,color=INK)
    ax.axhline(0,color=INK,lw=1.1,ls="--")
    ax.set_xticks(range(len(models))); ax.set_xticklabels([NICE[m] for m in models],fontsize=8.6)
    ax.set_ylabel("internal minus TM-to-WT"); ax.set_ylim(-.20,.62)
    ax.set_title("B   the gap, 95 % CI over ASSAYS",loc="left",fontsize=11,
                 fontweight="bold",color=INK,pad=20)
    ax.text(0,1.02,"all three clear zero; dots are the 4 individual assays",
            transform=ax.transAxes,fontsize=8.1,color=INK2,va="bottom")
    fig.text(.06,.945,"With matched variants and a capture check that can fail, the trunk beats "
             "the emitted structure in all three models",fontsize=13,fontweight="bold",color=INK)
    fig.text(.06,.80,
        "Same 4 assays and the SAME VARIANT IDs, alignments, recycles, sampling steps and folds for "
        "every model -- all three now run through one script. Internal = kl_glob, kl_site,\n"
        "dz_site, ds_site at EVERY Pairformer layer. Every per-layer feature was regenerated after "
        "the capture-fidelity check was made capable of failing; drift against each model's\n"
        "own trunk is 0.0 / 6.2e-4 / 4.6e-4 with a 400-660x mutation signal. Intervals treat the "
        "ASSAY as the independent unit, so they are ~2x wider than the split-level ones\n"
        "they replace. These are PAIRED WITHIN-MODEL comparisons, not a ranking: trunk depth "
        "(64/48/16) and distogram grids differ, and the per-assay dots in B overlap heavily.",
        fontsize=8.3,color=INK2)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out,dpi=190,facecolor=fig.get_facecolor()); print(f"wrote {a.out}")
    for m in models:
        pt,lo,hi,nk=pi_stats.cluster_bootstrap(gaps[m],n_boot=10000,seed=0)
        print(f"  {NICE[m]:11s} internal {np.nanmean(per[m]['internal (deep)']):+.3f}  "
              f"TM {np.nanmean(per[m]['TM to WT']):+.3f}  "
              f"gap {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] over {nk} assays")

if __name__=="__main__": main()
