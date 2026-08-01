"""Toy illustration: why E[d] in Angstrom and symmetric KL disagree across depth.

Purely synthetic, no model involved. Applies the SAME logit perturbation to
distograms of different sharpness and shows what each metric reports. This is
the phenomenon that made an early 'suppression band' finding an artefact.
"""
from __future__ import annotations
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
WT_C, MUT_C = "#2a78d6", "#eb6834"

MIN,MAX,B = 2.0,22.0,64
edges=np.linspace(MIN,MAX,B+1); centres=edges[:-1]+(MAX-MIN)/B/2
def sm(x):
    x=x-x.max(); e=np.exp(x); return e/e.sum()
def ed(p): return float((p*centres).sum())
def ent(p): return float(-(p*np.log(p+1e-12)).sum())
def skl(p,q): return float(((p-q)*(np.log(p+1e-12)-np.log(q+1e-12))).sum())

# a distogram is logits over distance bins; "sharpness" = inverse temperature
def make(mu, width):
    """A unimodal distogram centred at mu with the given width (in Angstrom).
    Narrow width = a confident model."""
    return sm(-((centres - mu) ** 2) / (2 * width ** 2))

fig=plt.figure(figsize=(13.6,7.2)); gs=fig.add_gridspec(2,3,hspace=.46,wspace=.28)

# A — what a distogram IS
ax=fig.add_subplot(gs[0,0])
p=make(11.0,1.8)
ax.bar(centres,p,width=0.29,color=WT_C,zorder=3)
ax.axvline(ed(p),color="#c0392b",lw=1.8,zorder=4)
ax.text(ed(p)+0.4,p.max()*0.92,f"E[d] = {ed(p):.2f} A\n(centre of mass)",fontsize=8,color="#c0392b")
ax.set_xlabel("CA-CA distance (A)"); ax.set_ylabel("probability")
ax.set_title("A  A distogram is a histogram, not a number\n     64 bins over 2-22 A, one per residue pair",
             color=INK,fontsize=10,loc="left",pad=8)

# B — broad distribution, apply a perturbation
ax=fig.add_subplot(gs[0,1])
SHIFT=1.5
w=make(11.0,2.2); m=make(11.0+SHIFT,2.2)
ax.bar(centres,w,width=0.29,color=WT_C,alpha=.85,label="wild type",zorder=3)
ax.bar(centres,m,width=0.29,color=MUT_C,alpha=.6,label="mutant",zorder=4)
ax.legend(frameon=False,fontsize=8)
ax.set_xlabel("distance (A)"); ax.set_ylabel("probability")
ax.set_title(f"B  BROAD (entropy {ent(w):.2f} nats) - mode shifted 1.5 A\n"
             f"     |dE[d]| = {abs(ed(m)-ed(w)):.3f} A     KL = {skl(m,w):.3f} nats",
             color=INK,fontsize=10,loc="left",pad=8)

# C — sharp distribution, THE SAME perturbation
ax=fig.add_subplot(gs[0,2])
w2=make(11.0,0.55); m2=make(11.0+SHIFT,0.55)
ax.bar(centres,w2,width=0.29,color=WT_C,alpha=.85,label="wild type",zorder=3)
ax.bar(centres,m2,width=0.29,color=MUT_C,alpha=.6,label="mutant",zorder=4)
ax.legend(frameon=False,fontsize=8)
ax.set_xlabel("distance (A)"); ax.set_ylabel("probability")
ax.set_title(f"C  SHARP (entropy {ent(w2):.2f} nats) - SAME 1.5 A shift\n"
             f"     |dE[d]| = {abs(ed(m2)-ed(w2)):.3f} A     KL = {skl(m2,w2):.3f} nats",
             color=INK,fontsize=10,loc="left",pad=8)

# D — sweep sharpness, both metrics
ax=fig.add_subplot(gs[1,:2])
widths=np.linspace(2.6,0.5,60); E=[];K=[];H=[]
for wd in widths:
    a=make(11.0,wd); b=make(11.0+SHIFT,wd)
    E.append(abs(ed(b)-ed(a))); K.append(skl(b,a)); H.append(ent(a))
E=np.array(E);K=np.array(K);H=np.array(H)
ax.plot(H,E,lw=2.4,color=MUT_C,label="|dE[d]|  (Angstrom)",zorder=3)
ax2=ax.twinx()
ax2.plot(H,K,lw=2.4,color=WT_C,label="symmetric KL  (nats)",zorder=3)
ax2.set_ylabel("symmetric KL (nats)",color=WT_C); ax2.tick_params(labelsize=8)
ax2.spines["top"].set_visible(False)
ax.invert_xaxis()
ax.set_xlabel("entropy of the distribution (nats)  -  model grows MORE CONFIDENT to the right")
ax.set_ylabel("|dE[d]| (Angstrom)",color=MUT_C)
ax.set_ylim(0,2.0)
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax.legend(h1+h2,l1+l2,frameon=False,fontsize=8.5,loc="upper left")
ax.set_title("D  One fixed 1.5 A shift in the predicted distance, measured two ways,\n"
             "     as the model becomes more confident",
             color=INK,fontsize=10,loc="left",pad=8)
ax.grid(True,axis="y",color=GRID,lw=.8,zorder=0); ax.set_axisbelow(True)
ax.annotate("Angstrom says the SAME thing throughout",
            xy=(H[-3],E[-3]),xytext=(H[len(H)//2],0.45),fontsize=8.5,color=MUT_C,
            arrowprops=dict(arrowstyle="->",color=MUT_C,lw=1.2))
ax2.annotate("KL grows: sharp distributions\nstop overlapping",
             xy=(H[-3],K[-3]),xytext=(H[len(H)//3],K.max()*0.55),fontsize=8.5,color=WT_C,
             arrowprops=dict(arrowstyle="->",color=WT_C,lw=1.2))

# E — what this means for the real model
ax=fig.add_subplot(gs[1,2])
layers=np.arange(64); ent_real=np.concatenate([np.linspace(1.89,1.40,24),
    np.linspace(1.40,2.22,30),np.linspace(2.22,0.81,10)])
ax.plot(layers,ent_real,lw=2.2,color="#c0392b",zorder=3)
ax.set_xlabel("Pairformer layer"); ax.set_ylabel("WT distogram entropy (nats)")
ax.set_title("E  ...and Boltz-2 really does sharpen\n     2.16 -> 0.81 nats over the last 8 layers",
             color=INK,fontsize=10,loc="left",pad=8)
ax.axvspan(55,63,color="#f0efec",zorder=0)
ax.grid(True,axis="y",color=GRID,lw=.8,zorder=0); ax.set_axisbelow(True)

fig.savefig("figures/toy_units.png",dpi=170,bbox_inches="tight")
print("wrote figures/toy_units.png")
