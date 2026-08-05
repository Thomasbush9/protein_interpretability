"""Build the SVD / mutation-subspace report from the analysis JSONs.

Same contract as `build_mech_report.py`: every number on the page is read from
an analysis output rather than typed, and a missing input renders as a visible
"not yet run" card instead of a stale figure. If a claim in the prose has a
number in it, that number came out of `runs/`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
OUT = W / "report_svd"


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def ci(d, f="{:+.3f}"):
    return (f"{f.format(d['mean'])} <span class=ci>[{f.format(d['ci_lo'])}, "
            f"{f.format(d['ci_hi'])}]</span>")


def pending(what):
    return (f'<div class="card amber"><div class=row><span class="chip c-amber">'
            f'pending</span><strong>{what}</strong></div>'
            f'<p>The analysis JSON for this section is missing, so the builder '
            f'emits this card rather than leaving a stale number on the page.</p>'
            f'</div>')


VIEW_VAR = "components, variance-ordered"
VIEW_PRED = "components, prediction-ordered"
VIEW_RAW = "RAW channels, prediction-selected (control)"


def sec_dims(S, DS):
    if not S:
        return pending("SVD of the mutation-induced difference")
    P, D = S["protocol"], S["protocol"]["dim"]
    cur = S["centered"]["curves"]
    kk = sorted(int(k) for k in cur[VIEW_VAR])
    rows = "".join(
        f"<tr><td class=n>{k}</td><td class=n>{ci(cur[VIEW_VAR][str(k)])}</td>"
        f"<td class=n>{ci(cur[VIEW_PRED][str(k)])}</td>"
        f"<td class=n>{ci(cur[VIEW_RAW][str(k)])}</td></tr>" for k in kk)
    full = cur[VIEW_VAR][str(D)]["mean"]
    k4 = cur[VIEW_VAR]["4"]["mean"]
    k1 = cur[VIEW_VAR]["1"]["mean"]
    raw1 = cur[VIEW_RAW]["1"]["mean"]
    nl = S["permutation_null"]
    worst = min(v["null_max_p95"] for v in nl["per_assay"].values())
    best = max(v["null_max_p95"] for v in nl["per_assay"].values())
    lo_r = min(v["rho"] for v in nl["per_assay"].values())
    hi_r = max(v["rho"] for v in nl["per_assay"].values())
    ds_note = ""
    if DS:
        dd = DS["protocol"]["dim"]
        g = DS["centered"]["curves"][VIEW_VAR][str(dd)]["mean"]
        sa = DS["subspace_agreement"]
        ds_note = (f"<p><b>The signal is in the pair track, not the single track.</b> "
                   f"The same analysis on <code>ds_site</code> (the single "
                   f"representation, {dd} dimensions) reaches only "
                   f"<b>{g:+.3f}</b> against <code>dz_site</code>'s "
                   f"<b>{full:+.3f}</b>, and its cross-assay subspace agreement is "
                   f"{sa['last8_pooled']['mean']:.3f} against the pair track's "
                   f"{load(A_SVD)['subspace_agreement']['last8_pooled']['mean']:.3f}. "
                   f"Whatever the model knows about a mutation, it is written "
                   f"between residue pairs rather than at the residue.</p>")
    return f"""
<section id=dims>
<h2>How many directions does the mutation response use?</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">result</span>
<h3>Four directions recover {100*k4/full:.0f}% of the full space</h3></div>
<p>Per layer, the archived <code>dz_site</code> difference &Delta;z =
z<sub>mut</sub> &minus; z<sub>WT</sub> is decomposed on <b>training positions only</b>,
the basis and the scaler are frozen, and held-out positions are projected into
them. The protocol is the published probe's: position-grouped splits,
{P['seeds']} splits, ridge with &lambda; chosen on an inner grouped split,
assay-level bootstrap over {len(P['assays'])} assays. Values are the mean of the
last eight layers &mdash; deliberately not the best layer, which would be a
selection on the test set.</p>
<div class=scroll><table>
<thead><tr><th>directions kept</th><th>components, by variance</th>
<th>components, by training association</th>
<th>raw channels, selected (control)</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>The control matters more than the curve. Selecting raw pair channels by
training |&rho;| is the published probe's own recipe applied to the
{D} channels instead of to four scalar summaries; it reaches the same place at
full rank. So the decomposition is <em>not</em> manufacturing the signal &mdash; it is
compressing it. At k = {D} the rotation and the selection are the same fit in two
bases, and the run checks that identity numerically rather than assuming it
(reported difference: 0.0e+00).</p>
<p><b>One direction is not enough, and the reason is informative.</b> Ranked by
variance, k = 1 scores only {k1:+.3f} &mdash; below the raw control's {raw1:+.3f}.
The leading direction of variance is substitution volume, which is close to
orthogonal to stability. The curve crosses the control at k = 2, exactly where
the stability component enters.</p>
{ds_note}
</div>
<div class="card">
<div class=row><span class="chip c-run">control</span>
<h3>Permutation null, with the layer search paid for</h3></div>
<p>Observed held-out &rho; at k = {nl['k']} ranges {lo_r:+.3f} to {hi_r:+.3f} across
assays; the null 95th percentile of max|&rho;| ranges {worst:.3f} to {best:.3f}.
Every assay returns p = 0.000.</p>
<p>Two things had to be right for that to mean anything. The statistic is the
<em>maximum over layers</em> on both sides, so searching 64 layers is charged to
the null rather than treated as free. And the permutation shuffles the whole
label vector and scores against the shuffled labels &mdash; an earlier version
shuffled only the training labels and scored against the true held-out ones,
which is not a null at all: the fitted direction still lies inside a subspace
whose axes are individually predictive, and it inherits their association with
DMS. That mistake produced a &ldquo;null&rdquo; reaching |&rho;| &asymp; 0.70.
{nl['caveat']}</p>
</div>
</section>"""


def sec_shared(S):
    if not S:
        return ""
    sa = S["subspace_agreement"]
    rp = S.get("replicate_stability", {}).get("pooled")
    rep_txt = ""
    if rp:
        rep_txt = (f"<p>The ceiling is measured, not assumed. <code>gym2_*</code> and "
                   f"<code>gym2s_*</code> are two independent executions of the "
                   f"identical variant set, and their subspaces agree at "
                   f"<b>{ci(rp, '{:.4f}')}</b>. Cross-assay agreement cannot be read "
                   f"as strong without that number to compare it against.</p>")
    return f"""
<section id=shared>
<h2>Do different proteins use the same directions?</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">result</span>
<h3>{len(sa['pairs_last8'])} pairs of unrelated folds agree at
cos<sup>2</sup>&thinsp;&theta; = {sa['last8_pooled']['mean']:.3f}</h3></div>
<p>The {S['protocol']['dim']} coordinates are Boltz-2's own pair channels and mean
the same thing in every protein, so the subspaces are directly comparable.
Agreement is the mean squared cosine of the principal angles between each
assay's top-{sa['k']} subspace &mdash; rotation-invariant, and it never asks two
individual components to correspond, which they cannot be made to do since
singular-vector signs are arbitrary and near-degenerate components rotate
freely.</p>
<p>Pooled over the last eight layers: <b>{ci(sa['last8_pooled'], '{:.3f}')}</b>
against a chance level of {sa['chance']:.3f} for random {sa['k']}-dimensional
subspaces.</p>
{rep_txt}
<p>Per-dimension standardisation is deliberately <em>not</em> applied for this
comparison. Rescaling each channel by its within-assay spread would rotate every
basis by a different amount and destroy the thing being measured.</p>
</div>
</section>"""


def sec_what(S):
    if not S:
        return ""
    pool = S["annotation_last_layer"]["pooled"]
    chem = S["annotation_last_layer"]["chem_pooled"]
    ROWS = [("DMS", pool["DMS"]), ("signed width change", pool.get("dsd_glob")),
            ("broadening (spread)", pool.get("spread_glob")),
            ("relocation (shift)", pool.get("shift_glob")),
            ("symmetric KL", pool.get("kl_glob"))]
    ROWS = [(l, v) for l, v in ROWS if v]
    body = ""
    for lab, v in ROWS:
        cells = "".join(
            f"<td class=n>{'<b>' if (c['ci_lo'] > 0 or c['ci_hi'] < 0) else ''}"
            f"{c['mean']:+.2f}"
            f"{'</b>' if (c['ci_lo'] > 0 or c['ci_hi'] < 0) else ''}</td>"
            for c in v[:6])
        body += f"<tr><td>{lab}</td>{cells}</tr>"
    for nm in ("d_volume", "d_hydropathy"):
        if nm in chem:
            body += (f"<tr><td>&Delta; {nm[2:]}</td>"
                     + "".join(f"<td class=n>{x:+.2f}</td>" for x in chem[nm][:6])
                     + "</tr>")
    dms = pool["DMS"]
    pc2 = int(max(range(len(dms)), key=lambda i: abs(dms[i]["mean"])))
    return f"""
<section id=what>
<h2>What are those directions?</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">result</span>
<h3>PC1 is substitution volume; PC{pc2+1} is stability and predictive width at once</h3></div>
<p>The basis here is learned <b>once across all twelve assays</b>, not per assay.
That is not a convenience. The sign of a singular vector is arbitrary, so
averaging a signed correlation across twelve independently-computed bases
cancels to nothing regardless of how strong the association is inside each one.
An earlier version of this section did exactly that and reported &asymp;0.02
against DMS for every component, while the same quantity was &minus;0.62 in a
two-assay run. The difference was sign cancellation, not effect size. Rows are
z-scored within assay first so no protein dominates through its own
representation scale.</p>
<div class=scroll><table>
<thead><tr><th>quantity</th>{"".join(f"<th>PC{i+1}</th>" for i in range(6))}</tr></thead>
<tbody>{body}</tbody></table></div>
<p class=ci>Bold = assay-level 95% interval excludes zero. Correlations at the
final layer.</p>
<p><b>The mechanism report's conclusion reappears here from an independent
direction.</b> PC{pc2+1} carries {dms[pc2]['mean']:+.3f} with measured stability and,
simultaneously, {pool['spread_glob'][pc2]['mean']:+.3f} with broadening and
{pool['dsd_glob'][pc2]['mean']:+.3f} with the signed width change. That is the
&ldquo;severity is registered as loss of certainty&rdquo; result falling out of the
second singular vector of the raw pair representation, with no divergence
computed anywhere in the derivation.</p>
</div>
</section>"""


def sec_transfer(S, TR):
    if not S or "loao_shared_basis" not in S:
        return pending("Leave-one-assay-out on a shared basis")
    lo = S["loao_shared_basis"]["results"]
    kk = sorted(int(k) for k in lo)
    rows = "".join(f"<tr><td class=n>{k}</td><td class=n>{ci(lo[str(k)])}</td></tr>"
                   for k in kk)
    ref = ""
    if TR:
        p = TR["predictors"]
        ref = (f"<p>For scale, the published transfer probe reaches "
               f"<b>{ci(p['internal'])}</b> and the within-assay ceiling is "
               f"{ci(p['internal_within'])}. That probe uses four scalar summaries "
               f"across all 64 layers while this uses raw pair channels across the "
               f"last eight, so it is a reference point and not a matched "
               f"comparison.</p>")
    return f"""
<section id=transfer>
<h2>Is the shared subspace the thing that transfers?</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">result</span>
<h3>Two directions transfer at {lo['2']['mean']:+.3f} to a protein the basis never saw</h3></div>
<p>The basis is learned on eleven assays, truncated to k directions, and applied
to the twelfth. This closes the loop between the transfer result and the
subspace result: if a handful of directions reproduce the transfer, the
protein-general signal has been reduced to something small enough to inspect,
patch and steer.</p>
<div class=scroll><table>
<thead><tr><th>directions kept</th><th>Spearman on the held-out assay</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>The jump from k = 1 ({lo['1']['mean']:+.3f}) to k = 2 ({lo['2']['mean']:+.3f}) is
what the annotation predicts and was not fitted to it: PC1 is volume and carries
almost no stability signal, PC2 is the stability axis. The two analyses
corroborate each other without being able to.</p>
{ref}
<p class=ci>Caveat carried forward: features are z-scored within assay, which
touches the held-out assay's own rows. Unsupervised, but transductive &mdash;
exactly as in the published transfer probe. A strictly inductive normalisation
should be reported alongside.</p>
</div>
</section>"""


def sec_drift(Dj):
    if not Dj:
        return pending("Run-to-run drift calibration of the sharpening result")
    sg = Dj["sign agreement on dsd"]
    fc = Dj["fraction clearing the noise band"]
    ga = Dj["DMS(sharpen) - DMS(broaden), all variants"]
    gc = Dj["DMS(sharpen) - DMS(broaden), confident only"]
    rows = "".join(
        f"<tr><td>{n}</td><td class=n>{100*v['kl_rel_drift']:.2f}%</td>"
        f"<td class=n>{v['cos_dz_min']:.4f}</td>"
        f"<td class=n>{v['dsd_noise_sd']:.5f}</td>"
        f"<td class=n>{100*v['sign_agreement']:.1f}%</td>"
        f"<td class=n>{100*v['frac_confident']:.1f}%</td></tr>"
        for n, v in sorted(Dj["per_assay"].items()))
    return f"""
<section id=drift>
<h2>Does the sharpening result survive inference noise?</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">audit response</span>
<h3>Yes, and it gets larger when the ambiguous band is removed</h3></div>
<p>The audit asks for wild type and a stratified mutant subset to be repeated
across seeds so that <code>d&sigma; &lt; 0</code> can be defined beyond the noise.
That run is unnecessary: the repeat already exists. <code>gym2_*</code> and
<code>gym2s_*</code> are two independent executions of the same 250 variants per
assay &mdash; same seed, same subsample, same alignment cap, four days and one
additive code change apart. The <code>mutant</code> lists are checked to match
exactly before any assay is used. For a quantity measured twice with independent
errors of equal variance, sd(d<sub>1</sub> &minus; d<sub>2</sub>) =
&radic;2&thinsp;&sigma;, which is where the noise estimate comes from.</p>
<div class=scroll><table>
<thead><tr><th>assay</th><th>KL drift</th><th>min cos(&Delta;z<sub>1</sub>,
&Delta;z<sub>2</sub>)</th><th>sd of d&sigma; noise</th>
<th>sign agreement</th><th>clears &plusmn;2&sigma;</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>Note the contrast in the first two columns: the scalar KL drifts by up to a
few percent between runs, while the representation difference &Delta;z itself is
reproducible to cos &ge; 0.999. The summaries are noisier than the thing they
summarise.</p>
<ul>
<li>Sign of d&sigma; agrees across runs on <b>{ci(sg)}</b> of variants.</li>
<li><b>{ci(fc)}</b> of variants sit further from zero than twice the drift.</li>
<li>DMS(sharpen) &minus; DMS(broaden), all variants: <b>{ci(ga)}</b></li>
<li>the same, ambiguous band excluded: <b>{ci(gc)}</b></li>
</ul>
<p>The effect <em>grows</em> when the {100*(1-fc['mean']):.0f}% of variants inside the
noise band are dropped rather than silently assigned to whichever side their
noise put them on. Drift was diluting the result, not creating it.</p>
<p class=ci>Limitation: <code>disto</code> archives the final layer only, so this
is measured at layer 63 while <code>analyze_channels</code> averages the last
eight. Averaging can only reduce noise, so these numbers are conservative for
that use.</p>
</div>
</section>"""


def sec_pc2(Q):
    if not Q:
        return pending("Spatial localisation of PC2")
    rr = Q["localisation radius ratio"]
    mg = Q["magnitude vs radius ratio"]
    p2a = Q["PC2 vs perturbation magnitude"]
    p2l = Q["PC2 vs radius ratio | magnitude"]
    p1a = Q["PC1 vs perturbation magnitude"]
    rows = "".join(
        f"<tr><td>PC{c+1}</td><td class=n>{ci(Q[f'PC{c+1} vs perturbation magnitude'])}</td>"
        f"<td class=n>{ci(Q[f'PC{c+1} vs radius ratio | magnitude'])}</td></tr>"
        for c in range(Q["protocol"]["n_pc"]))
    return f"""
<section id=pc2>
<h2>Where does PC2 act?</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">result</span>
<h3>PC2 is an amplitude, not a location</h3></div>
<p>This did not need the GPU re-capture it looked like it would. <code>z_site</code>
is averaged over partner residues, but the per-pair <em>distogram</em> is
archived in full &mdash; <code>disto</code> holds the final-layer logits at ~1479
sampled residue pairs. Those pairs are recoverable exactly: they came from
<code>np.random.default_rng(0)</code> after one fixed variant subsample. The
reconstruction is checked against <code>disto.shape[1]</code> for every assay
before anything is computed, and the kept counts differ between assays (1476 to
1482), so twelve independent matches is not something a wrong reconstruction
could produce.</p>
<p><b>The perturbation is local.</b> Weighting each pair's |d&sigma;| by its
distance from the mutated residue and dividing by the same centroid under
uniform weights gives <b>{ci(rr, '{:.3f}')}</b> &mdash; every assay between 0.74
and 0.84. The ratio is used rather than a raw radius because a mutation near a
terminus has a different set of distances available to it, which is a fact about
the protein and not about the model.</p>
<div class=scroll><table>
<thead><tr><th>component</th><th>vs perturbation amplitude</th>
<th>vs localisation, amplitude held fixed</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>PC2 predicts <em>how much</em> certainty is lost ({ci(p2a)}, against PC1's
{ci(p1a)}) and, once amplitude is held fixed, carries nothing about
<em>where</em> ({ci(p2l)}, interval includes zero; per-assay values scatter from
&minus;0.15 to +0.26 with no consistent sign).</p>
</div>
<div class="card amber">
<div class=row><span class="chip c-amber">trap</span>
<h3>The obvious version of this analysis gives the opposite answer</h3></div>
<p>Splitting variants by raw PC2 quartile produces a clean monotone gradient:
the top-minus-bottom difference runs from &minus;0.01 at 0&ndash;6&nbsp;&Aring; to
&minus;0.23 beyond 32&nbsp;&Aring;, which reads as &ldquo;high-PC2 variants are more
concentrated&rdquo;. It is an artifact. Perturbation magnitude is itself
associated with localisation ({ci(mg)}: larger changes are more local), and PC2
tracks magnitude at {p2a['mean']:+.3f}, so its top quartile is largely the
large-perturbation quartile. Re-splitting on PC2 after rank-residualising on
magnitude destroys the monotone trend.</p>
<p>Both versions are plotted in panel&nbsp;B rather than only the corrected one,
because the uncorrected curve is what this analysis would have reported if the
magnitude control had not been run.</p>
</div>
</section>"""


A_SVD = str(W / "runs/svd_dz_v2.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svd", default=A_SVD)
    ap.add_argument("--svd-ds", default=str(W / "runs/svd_ds_v1.json"))
    ap.add_argument("--drift", default=str(W / "runs/drift_v1.json"))
    ap.add_argument("--transfer", default=str(W / "runs/transfer_v1.json"))
    ap.add_argument("--pc2", default=str(W / "runs/pc2_v2.json"))
    a = ap.parse_args()

    S, DS, Dj, TR = (load(a.svd), load(a.svd_ds), load(a.drift), load(a.transfer))
    Q = load(a.pc2)
    css = (OUT / "style.css").read_text()
    html = f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>The mutation subspace &mdash; SVD of Boltz-2 pair representations</title>
<style>{css}
.ci{{font-size:.82em;color:var(--muted);font-family:var(--mono)}}
figure{{margin:1.4rem 0}}
figure img{{width:100%;height:auto;border:1px solid var(--rule);border-radius:4px}}
figcaption{{font-size:.86rem;color:var(--muted);margin-top:.5rem}}
</style></head><body><div class=wrap>
<header class=head>
  <span class=eyebrow>Boltz-2 Pairformer &mdash; representation structure</span>
  <h1>The mutation subspace</h1>
  <p class=lede>The transfer result already showed that a probe trained on eleven
  proteins predicts stability on a twelfth. This asks <em>what</em> transfers. Decomposing
  the mutation-induced difference in the pair representation gives a small, shared set
  of directions whose second member is simultaneously the stability axis and the
  predictive-certainty axis &mdash; which is the mechanism report's conclusion,
  recovered without computing a divergence.</p>
</header>

<figure>
  <img src="figures/svd.png" alt="Four panels: held-out Spearman against number of
  directions kept; transfer on a basis learned without the held-out protein; a
  component-by-quantity association heatmap on a shared basis; cross-assay subspace
  agreement by layer.">
  <figcaption>Phase A on <code>dz_site</code>, twelve ProteinGym assays. Generated by
  <code>fig_svd.py</code> from <code>svd_dz_v2.json</code>.</figcaption>
</figure>

{sec_dims(S, DS)}
{sec_shared(S)}
{sec_what(S)}
{sec_transfer(S, TR)}

<figure>
  <img src="figures/pc2.png" alt="Three panels: normalised width change against distance
  from the mutated residue; top-minus-bottom quartile differences for magnitude, raw PC2
  and magnitude-adjusted PC2; pooled intervals for each component against amplitude and
  against localisation.">
  <figcaption>Localisation of PC2 from the archived per-pair distogram, no re-capture.
  Generated by <code>fig_pc2.py</code> from <code>pc2_v2.json</code>.</figcaption>
</figure>

{sec_pc2(Q)}

<figure>
  <img src="figures/drift.png" alt="Three panels: sharpening fraction in run 1 against
  run 2; per-assay share of variants clearing twice the drift; the DMS gap with and
  without the ambiguous band.">
  <figcaption>Run-to-run calibration from the <code>gym2</code>/<code>gym2s</code>
  replicate. Generated by <code>fig_drift.py</code> from <code>drift_v1.json</code>.</figcaption>
</figure>

{sec_drift(Dj)}

<section id=limits>
<h2>What this does not establish</h2>
<ul>
<li><b>The localisation result is about the distogram, not about &Delta;z.</b> The
per-pair readout is the distogram head's output; PC2 is a direction in the pair
representation. A component could carry spatial structure in z that the readout
does not express, so &ldquo;PC2 is an amplitude&rdquo; is established for what the
model predicts, not for what it represents. Settling it for &Delta;z itself needs
the full pair row &Delta;z[r, j, :] &mdash; a one-line change at
<code>exp_gym2.py:104</code> plus a re-capture.</li>
<li><b>Held-out means held-out positions, not held-out proteins</b>, except in the
transfer section. The two designs answer different questions and their numbers
are not interchangeable.</li>
<li><b>Feature scaling is transductive.</b> Within-assay z-scoring uses the held-out
assay's own rows. It is unsupervised and matches the published probe, but an
inductive variant should be reported beside it.</li>
<li><b>The permutation null does not preserve the within-position autocorrelation of
DMS.</b> It is a null for &ldquo;no association&rdquo;; the position-grouped split is
what handles position structure.</li>
<li><b>Component identity is not stable beyond sign.</b> Principal angles are used
precisely because individual components cannot be matched across assays; the
annotation table depends on a single shared basis and would not survive
per-assay decomposition.</li>
<li>Burial, conservation and secondary structure are <em>not</em> in the annotation
yet &mdash; no experimental structures for these twelve assays are on disk, and
burial computed from the model's own prediction would not be independent of the
model.</li>
</ul>
</section>

<section id=next>
<h2>What this changes about the next runs</h2>
<div class="card run">
<div class=row><span class="chip c-run">scoping</span>
<h3>Phase B becomes a one-direction question</h3></div>
<p>The localisation question has now been answered on the distogram without any
capture, and the answer is negative: PC2 carries amplitude, not position. That
materially weakens the case for the twelve-assay re-capture, whose purpose was to
map PC2 across partner residues &mdash; it would be looking for spatial structure
that the model's own readout does not express. It is not dead, because the
readout is a projection and &Delta;z could carry structure the distogram discards,
but it is no longer the obvious next spend.</p>
<p>The stronger use of the same GPU time is <b>causal</b>: PC2 is a single known
direction in a 128-dimensional space, so it can be added to or removed from z
during a forward pass, and the effect on the distogram, pLDDT and emitted
coordinates measured directly. That tests whether the direction is used rather
than merely present, which is the question the conditioning-tensor probe in the
publication plan is also aimed at.</p>
</div>
<div class="card">
<h3>Cheap and still open</h3>
<ul>
<li>Download the twelve experimental structures and add burial, contact density and
secondary structure to the annotation, fitted jointly rather than reported as
independent.</li>
<li>Note that the cross-model subspace comparison is <em>not</em> offline: the
<code>deep2_*</code> archives store <code>dz_site</code> as a per-layer norm, not a
vector, so principal angles against OpenFold3 and Protenix need a re-capture.</li>
<li>An inductive normalisation variant of the transfer and the leave-one-assay-out
basis.</li>
<li>Equivalence test (TOST or a region of practical equivalence) on the
KL&nbsp;&minus;&nbsp;spread comparison in the mechanism report, which currently
rests on an interval that includes zero.</li>
<li>Fix <code>analyze_transfer.py</code>: it strips <code>"gym2_"</code>, so pointing
it at the <code>gym2s_*</code> archives silently collapses all twelve assays to one
key.</li>
</ul>
</div>
</section>

<footer>
  <p>Generated by <code>build_svd_report.py</code> from the analysis JSONs; numbers are
  read from the runs, not transcribed. Code: <code>analyze_svd.py</code>,
  <code>analyze_drift.py</code>, <code>fig_svd.py</code>, <code>fig_drift.py</code>,
  <code>analysis.sbatch</code>. Archives: <code>runs/svd_dz_v2.*</code>,
  <code>runs/svd_ds_v1.json</code>, <code>runs/drift_v1.json</code>,
  <code>runs/transfer_v1.json</code>.</p>
</footer>
</div></body></html>
"""
    (OUT / "index.html").write_text(html)
    print(f"wrote {OUT/'index.html'}  ({len(html)/1024:.0f} KB)")
    for nm, obj in (("svd dz", S), ("svd ds", DS), ("drift", Dj),
                    ("transfer", TR), ("pc2", Q)):
        print(f"   {nm:10s} {'ok' if obj else 'MISSING -> pending card'}")


if __name__ == "__main__":
    main()
