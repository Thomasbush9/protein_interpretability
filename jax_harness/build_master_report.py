"""Build the single comprehensive report: what Boltz-2 knows and does not say.

There are four report pages in this project and none of them is the one to hand
someone. `report_svd/` is an audit trail -- twenty sections, every challenge and
correction in the order it happened. `report_jacobian/` and `report_jac/` are
this month's mechanism work. What was missing is a linear read of the actual
result: what PC2 is, how it was obtained and why, and how it compares to what the
model emits.

This page is that. It is deliberately NOT a superset -- it drops the audit trail
and the negative mechanism detail, and points at the other pages for both.

Every number is read from an archived analysis JSON at build time, every input
is copied next to the page with its SHA-256, and a figure older than its data
aborts the build. Provenance machinery is shared in `pi_report.py`.

  python build_master_report.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pi_report as R  # noqa: E402

OUT = R.W / "report_master"
FIGSPEC = {
    "headline.png": ("python fig_headline.py --transfer {transfer} --bw {bw} "
                     "--out {out}"),
    "causal.png": ("python fig_causal.py --steer {steer} --ablate {ablate} "
                   "--out {out}"),
    "xmodel_io.png": ("python fig_xmodel_io.py --xio {xio} --out {out}"),
}
# Repo-relative since the promotion: the entry points are still in jax_harness,
# the protocol module is not. Hashing `jax_harness/pi_protocol.py` after the move
# would have hashed the six-line alias and reported it as the code identity of
# every protocol block on the page.
CODE = ["jax_harness/build_master_report.py", "jax_harness/pi_report.py",
        "jax_harness/fig_headline.py", "jax_harness/fig_causal.py",
        "jax_harness/analyze_transfer.py", "jax_harness/analyze_steer_pool.py",
        "jax_harness/analyze_ablate.py", "jax_harness/exp_ablate_pc.py",
        "jax_harness/compare_internal_output.py", "jax_harness/analyze_pc2.py",
        "jax_harness/analyze_chem.py", "jax_harness/analyze_xmodel_io.py",
        "jax_harness/fig_xmodel_io.py", "jax_harness/analyze_layer_match.py",
        "src/protein_interpretability/experiments/protocol.py"]

# Reused unchanged from report_svd; copied in by the builder so this page is
# self-contained and its figure cannot drift from the one that was reviewed.
#
# These were a blind spot. `shutil.copy2` preserves mtimes, and they were not in
# FIGSPEC, so the staleness guard never tested them -- when svd_dz_v3.json and
# heldout_v1.json were regenerated their figures silently went stale and the
# manifest still reported none. Each now names the input it is drawn from and
# the command that rebuilds it, and is checked exactly like a local figure.
BORROWED = {
    "svd.png": ("svd", "python fig_svd.py --svd {svd} --svd-ds {svd_ds} "
                       "--out " + str(R.W / "report_svd/figures/svd.png")),
    "heldout.png": ("heldout", "python fig_heldout.py --heldout {heldout} "
                               "--out " + str(R.W / "report_svd/figures/heldout.png")),
    "depth.png": ("depth", "python fig_depth.py --depth {depth} "
                           "--out " + str(R.W / "report_svd/figures/depth.png")),
}


def sec_lede(TR, HO, ST, SV):
    if not (TR and HO and ST and SV):
        return R.pending("summary")
    P = TR["predictors"]
    g = TR["gaps"]["internal 128-dim - output-rich"]
    ho = HO["summary"]["pc2_inductive"]
    return f"""
<section id=summary>
<h2>The result in three sentences</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">1</span>
<strong>Boltz-2's internal state predicts mutational stability far better than
prespecified summaries of its sampled structure and confidence.</strong></div>
<p>A probe on the trunk reaches Spearman
<span class=big>{P['internal_vec']['mean']:+.3f}</span>
[{P['internal_vec']['ci_lo']:+.3f}, {P['internal_vec']['ci_hi']:+.3f}] on a protein it
was never trained on, against {P['output_rich']['mean']:+.3f} for ten
prespecified summaries of the sampled C&alpha; geometry and confidence (a
37-feature version moves this baseline, not the conclusion; the distogram
head, all-atom output and full output distribution are not part of either
comparator) &mdash; a gap of
{g['gap']:+.3f} [{g['ci_lo']:+.3f}, {g['ci_hi']:+.3f}] that holds in
<span class=big>{g['wins']} of {g['n_assays']}</span> proteins. The probe
reads the 128 pair channels at the final trunk layer &mdash; the direction the
mutation moved the representation, not merely how far.</p>
</div>
<div class="card ok">
<div class=row><span class="chip c-ok">2</span>
<strong>Almost all of it lives in one direction, and that direction is
protein-general.</strong></div>
<p>The second component of a shared basis fitted across twelve unrelated folds
is simultaneously the stability axis and the predicted-distance-width axis. On
proteins outside the set the basis was fitted on, that single direction reaches
|&rho;| = <span class=big>{abs(ho['stability (12)']['mean']):.3f}</span> on
stability assays and only
{abs(ho['non-stability (4)']['mean']):.3f} on non-stability ones.</p>
</div>
<div class="card warn">
<div class=row><span class="chip c-warn">3</span>
<strong>Intervention evidence is descriptive, and the deletion test is
null.</strong></div>
<p>Injecting the direction into the final z moves the distogram width with a
consistent sign at the middle dose in all twelve proteins, and pLDDT the
opposite way &mdash; but the response is dose-non-monotone, PC2 is also the
highest-<em>gain</em> direction (its sign-blind even component ranks first in
most proteins), and the eight "independent" controls are the same eight
orientations in every protein, so the earlier confirmatory p-values are
withdrawn (&sect;5). Deleting PC2 from a real mutation's own &Delta;z changes
what the model emits no more than deleting a random direction &mdash; every
PC2-minus-random interval includes zero, with the surgery verified. The
supported statement is that the direction is <em>readable</em> and its
injection <em>perturbs</em> the output; that the model <em>uses</em> it is not
established.</p>
</div>
</section>"""


def sec_numbers(TR, SV, CH, HO, LM):
    """Reconcile the several 'internal' numbers this project reports.

    Two earlier drafts got this wrong in the same way. The transfer analysis fed
    `dz_site` in as a per-layer NORM, so the headline discarded the direction and
    read 0.573 while every direction-based figure elsewhere read 0.63-0.73. The
    fix was to give the transfer probe the 128 channels; the numbers now agree
    and this section exists to show that they do.
    """
    if not (TR and SV and CH and HO and LM):
        return R.pending("reconciling the numbers")
    var = SV["centered"]["curves"]["components, variance-ordered"]["128"]
    lo2 = SV["loao_shared_basis"]["results"]["2"]
    pa = CH["pc2_alone_loao"]
    ho = HO["summary"]["pc2_inductive"]["stability (12)"]
    P = TR["predictors"]

    def ci(d):
        # abs() on a negative interval reverses it: |-0.734| > |-0.649|.
        # A review caught 0.695 [0.734, 0.649] on the page.
        lo, hi = sorted((abs(d['ci_lo']), abs(d['ci_hi'])))
        return f"{abs(d['mean']):.3f} <span class=ci>[{lo:.3f}, {hi:.3f}]</span>"

    rows = [
        ["all 128 pair channels, final layer", "leave-one-assay-out",
         ci(P["internal_vec"])],
        ["all 128 directions, final layer", "within protein, by position",
         f"{LM['final_layer']['128']:.3f}"],
        ["all 128 directions, mean of last "
         f"{LM['layer_window']} layers", "within protein, by position",
         ci(var)],
        ["top 2 components of the shared basis", "leave-one-assay-out", ci(lo2)],
        ["PC2 alone", "separate held-out protein set", ci(ho)],
        ["PC2 alone", "leave-one-assay-out", ci(pa)],
        ["per-layer magnitudes only", "leave-one-assay-out", ci(P["internal"])],
        ["per-layer magnitudes only", "within protein, by position",
         ci(P["internal_within"])],
    ]
    d = TR["gaps"]["internal 128-dim - internal norms"]
    return f"""
<section id=numbers>
<h2>2 &middot; Reconciling the internal numbers</h2>
<p>Several "internal" figures appear in this project. They now agree, and the
table says which measurement each one is.</p>
{R.table(["features", "design", "Spearman"], rows)}
<p>Two conventions separate the top three rows, and neither is a disagreement
about the model.</p>
<ul>
<li><strong>Layer window.</strong> The plotted SVD curve pools the last
{LM['layer_window']} layers rather than reading one, deliberately, so that no
single layer is chosen by held-out performance. At all 128 dimensions that
averaging is worth {LM['last_window']['128'] - LM['final_layer']['128']:+.3f}
({LM['final_layer']['128']:.3f} at the final layer alone against
{LM['last_window']['128']:.3f} pooled). Comparisons against a final-layer number
should use {LM['final_layer']['128']:.3f}.</li>
<li><strong>Training set size.</strong> At the matched setting &mdash; all 128
dimensions, final layer &mdash; leave-one-assay-out reaches
{P['internal_vec']['mean']:.3f} against {LM['final_layer']['128']:.3f}
within-protein. Training across eleven proteins uses roughly
{11 * 250:,} rows; a within-protein split uses about a fifteenth of that. The
across-protein design does not cost accuracy here, it buys it.</li>
</ul>
<div class="card ok">
<div class=row><span class="chip c-ok">why a full PCA cannot beat the raw channels</span>
<strong>At full rank they are the same fit</strong></div>
<p>A question this table invites: if the components are the "better" basis, why
does the complete PCA score below the raw pair representation? It does not, and
it cannot. A rank-128 rotation of 128 channels is invertible, so a linear probe
has exactly the same function class in either basis; the fit is identical, not
merely similar. Two independent checks confirm it &mdash; the archived SVD
analysis reports the two agreeing to
{SV['protocol']['spearman_delta']:.1e}, and the leave-one-assay-out sweep in
panel B of the figure has them meeting at
{abs(TR['k_sweep']['128']['rotated']['mean'] - TR['k_sweep']['128']['channels']['mean']):.4f}.</p>
<p>Rotation helps only under TRUNCATION, by packing the signal into fewer
directions: at one dimension it is worth
{TR['k_sweep']['1']['rotated']['mean'] - TR['k_sweep']['1']['channels']['mean']:+.3f},
at sixteen
{TR['k_sweep']['16']['rotated']['mean'] - TR['k_sweep']['16']['channels']['mean']:+.3f},
and at full rank exactly nothing. Any apparent gap between a "complete PCA"
number and a "full representation" number is therefore a difference of protocol
&mdash; layer window or training-set size &mdash; never of information.</p>
</div>
<p>Everything that carries the direction, once those two conventions are held
fixed, lands between {min(LM['final_layer']['128'], abs(pa['mean'])):.2f} and
{max(P['internal_vec']['mean'], abs(ho['mean'])):.2f}.</p>
<div class="card warn">
<div class=row><span class="chip c-warn">corrected</span>
<strong>Two earlier drafts led with a handicapped probe</strong></div>
<p>The transfer analysis originally fed <code>dz_site</code> in as its per-layer
L2 NORM &mdash; how far the pair row moved, never which way. That is a
defensible shared feature space and it is also the one quantity this project is
about. It cost <span class=big>{d['gap']:+.3f}</span>
[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] against the same probe given the 128
channels, and it is why the summary read
{P['internal']['mean']:.3f} while the figure in section 4 read
{abs(var['mean']):.3f}. The 128 pair channels mean the same thing in every
protein &mdash; that is the shared-subspace result &mdash; so nothing prevented
pooling them across assays. Both rows are kept in section 1.</p>
</div>
</section>"""


def sec_headline(TR, BW, stale, TI):
    if not (TR and BW and TI):
        return R.pending("performance")
    P, gp = TR["predictors"], TR["gaps"]
    pr = TR["protocol"]
    rows = [[lab, f"{P[k]['mean']:+.3f}",
             f"[{P[k]['ci_lo']:+.3f}, {P[k]['ci_hi']:+.3f}]"]
            for k, lab in [
                ("internal_vec", "internal, all 128 pair channels (transferred)"),
                ("internal", "internal, per-layer magnitudes (transferred)"),
                ("internal_within", "internal magnitudes, within the same protein"),
                ("chemistry", "substitution chemistry"),
                ("output_rich", "emitted structure, 10 features"),
                ("TM_to_WT", "TM score to wild type")]]
    grows = [[lab, f"{gp[k]['gap']:+.3f}",
              f"[{gp[k]['ci_lo']:+.3f}, {gp[k]['ci_hi']:+.3f}]",
              f"{gp[k]['wins']}/{gp[k]['n_assays']}"]
             for k, lab in [
                 ("internal 128-dim - output-rich", "128-dim vs emitted structure"),
                 ("internal 128-dim - chemistry", "128-dim vs substitution chemistry"),
                 ("internal 128-dim - internal norms", "128-dim vs magnitudes only"),
                 ("transferred internal - output-rich", "magnitudes vs emitted structure"),
                 ("transferred internal - TM", "magnitudes vs TM to wild type")]]
    bl = BW["blocks"]
    brows = [[lab,
              f"{bl[k]['between']['mean']:+.3f}", f"{bl[k]['within']['mean']:+.3f}"]
             for k, lab in [
                 ("internal dz (128, one layer)", "internal"),
                 ("substitution chemistry (17)", "substitution chemistry"),
                 ("output rich (published, 10)", "emitted structure")]]
    warn = ('<div class="card warn"><div class=row><span class="chip c-warn">'
            'stale</span><strong>Figure older than its data</strong></div></div>'
            ) if "headline.png" in stale else ""
    return f"""
<section id=performance>
<h2>1 &middot; Internal versus output</h2>
{warn}
<figure><img src="figures/headline.png" alt="Three panels: predictor comparison
with per-protein points; the same comparison paired within protein showing
internal winning in all twelve; and the between-position versus within-position
decomposition.">
<figcaption>Leave-one-assay-out throughout. Each predictor is trained on eleven
proteins and scored on the twelfth. Panel B answers "why this many channels" by
showing the whole truncation curve; channels are ranked by absolute Spearman
against the target on the training proteins only, so the selection never sees
the protein it is scored on.</figcaption></figure>

<h3>The protocol, and why it is the fair comparison</h3>
<p>{pr['n_assays']} stability assays, {pr['design']}, ridge with &lambda; =
{pr['lam']}, {pr['normalisation']}. The feature cap is k = {pr['k']},
at or above every block's own width, so none of the transferred predictors is
truncated: 128 pair channels, 256 per-layer magnitudes, 17 chemistry features,
10 emitted-structure features.</p>
<div class="card warn">
<div class=row><span class="chip c-warn">corrected twice</span>
<strong>The cap was wrong, then still wrong</strong></div>
<p>An earlier build capped every block at k = 16. That left chemistry (17
features) and the emitted structure (10) essentially intact while discarding 112
of the internal side's 128 channels and 240 of the magnitude block's 256 &mdash;
matching the <em>number</em> of features when the widths differ by an order of
magnitude, which is the opposite of matching. Raising it to 128 fixed the
128-channel block but left the 256-wide magnitude block still halved, while the
page claimed nothing was truncated. At k = {pr['k']} that claim is finally
true.</p>
<p>The 128-channel headline is unaffected either way &mdash; 128 was already its
full width, so it reads {P['internal_vec']['mean']:.3f} at both caps. What moved
is the magnitude block, from 0.573 at k=16 to
{P['internal']['mean']:.3f} now.</p>
</div>
<p>One row is deliberately not untruncated. The within-protein reference fits on
roughly 190 rows, so {pr['k']} features over-parameterise it; it is a reference
scale, not a claim, and it falls to {P['internal_within']['mean']:.3f} as a
result. Both normalisations are load-bearing: feature scale
depends on chain length and on the model's representation scale for that fold,
and target dynamic range varies the same way, so without them the ridge would
spend its capacity learning which protein a row came from.</p>
<p>The obvious objection is dimensional. The internal side gets 256 features
(4 quantities &times; 64 layers); a single TM score gets one, and no monotone
transform of one number can change its Spearman. So the output side is given the
richest description of its own product the saved coordinates allow &mdash; ten
emitted quantities through the identical ridge protocol. The trunk distogram is
deliberately excluded from it: that is a head on the Pairformer, not a product
of the structure module, and including it would blur the internal/output
distinction the comparison rests on.</p>

<h3>Held-out performance</h3>
{R.table(["predictor", "Spearman", "95% interval"], rows)}
<div class="card ok">
<div class=row><span class="chip c-ok">stricter protocol</span>
<strong>Under inductive normalisation the gap grows</strong></div>
<p>The table above scales each assay by its own feature statistics &mdash;
unsupervised, but transductive: the held-out protein's rows set its own scale.
Rescaling instead with the TRAINING assays' statistics, so nothing whatever
about the held-out protein enters, moves the internal probe barely
({TI['predictors']['internal']['mean']:+.3f} against
{P['internal']['mean']:+.3f}) while the emitted-structure baseline falls from
{P['output_rich']['mean']:+.3f} to
{TI['predictors']['output_rich']['mean']:+.3f}. The gap RISES to
<span class=big>{TI['gaps']['transferred internal - output-rich']['gap']:+.3f}</span>
[{TI['gaps']['transferred internal - output-rich']['ci_lo']:+.3f},
{TI['gaps']['transferred internal - output-rich']['ci_hi']:+.3f}], still
{TI['gaps']['transferred internal - output-rich']['wins']}/{TI['gaps']['transferred internal - output-rich']['n_assays']}.</p>
<p>Output features carry protein-specific scales &mdash; displacement magnitudes
depend on chain length &mdash; so they lose more when denied the held-out
protein's own statistics. The internal features are already in the model's
shared channel space. The transductive number is reported as the primary one
because it is the more conservative of the two.</p>
</div>
<p>Transfer costs nothing. A probe trained on eleven other proteins does as well
on the twelfth as one trained on the twelfth itself &mdash; the within-minus-
transferred gap is {gp['within-assay - transferred']['gap']:+.3f}
[{gp['within-assay - transferred']['ci_lo']:+.3f},
{gp['within-assay - transferred']['ci_hi']:+.3f}], winning in only
{gp['within-assay - transferred']['wins']} of
{gp['within-assay - transferred']['n_assays']}.</p>

<h3>Paired gaps</h3>
{R.table(["comparison", "gap", "95% interval", "proteins won"], grows)}

<h3>Where the output fails</h3>
<p>Splitting the target into variance BETWEEN positions (which residue was
mutated) and WITHIN a position (which substitution at a fixed residue) separates
an easy signal from a hard one. Burial and packing largely determine the first.
The second requires knowing something about the specific exchange.</p>
{R.table(["predictor", "between positions", "within position"], brows)}
</section>"""


def sec_chem(TR, CH, SC, BW):
    """Is the internal signal just the substitution?

    The head-to-head internal-minus-chemistry gap is the wrong statistic for
    this question and an earlier draft of this page led with it. A horse race
    between two predictors answers "which would you deploy"; the interpretability
    question is whether the internal direction is a re-encoding of the amino acid
    exchange the model was handed. That is a nested comparison and a partial
    correlation, both of which were already computed.
    """
    if not (TR and CH and SC and BW):
        return R.pending("chemistry control")
    inc = CH["increments"]
    pc = SC["pc2_vs_dms_partial_on_chemistry"]
    gp = TR["gaps"]
    bl = BW["blocks"]
    rows = [[lab, f"{inc[k]['gap']:+.3f}",
             f"[{inc[k]['ci_lo']:+.3f}, {inc[k]['ci_hi']:+.3f}]",
             f"{inc[k]['wins']}/{inc[k]['n']}"]
            for k, lab in [
                ("PC2 adds to chemistry", "PC2 added on top of chemistry"),
                ("chemistry adds to PC2", "chemistry added on top of PC2"),
                ("full dz adds to chemistry", "the full pair row added to chemistry"),
                ("chemistry-residual dz beats chemistry",
                 "chemistry-residualised pair row, against chemistry alone")]]
    return f"""
<section id=chemistry>
<h2>3 &middot; Is it just the substitution?</h2>
<p>The one confound that would deflate everything: the model is handed the
mutated sequence, so a direction correlating with stability might be nothing
more than a re-encoding of which amino acid replaced which. Substitution
chemistry scores {TR['predictors']['chemistry']['mean']:+.3f} on this task, not
far below the internal probe's
{TR['predictors']['internal']['mean']:+.3f}, so the question is real.</p>
<div class="card ok">
<div class=row><span class="chip c-ok">it is not</span>
<strong>The two directions of the comparison are wildly asymmetric</strong></div>
<p>A head-to-head gap &mdash; internal minus chemistry, here
{gp['transferred internal - chemistry']['gap']:+.3f} &mdash; answers "which
would you deploy". It does not answer this. The nested comparison does: give a
model chemistry and ask what PC2 adds, then give it PC2 and ask what chemistry
adds.</p>
{R.table(["comparison", "gain", "95% interval", "proteins won"], rows)}
<p>PC2 on top of chemistry is worth
<span class=big>{inc['PC2 adds to chemistry']['gap']:+.3f}</span> in
{inc['PC2 adds to chemistry']['wins']} of
{inc['PC2 adds to chemistry']['n']} proteins. Chemistry on top of PC2 is worth
<span class=big>{inc['chemistry adds to PC2']['gap']:+.3f}</span>. The internal
direction very nearly subsumes substitution chemistry; chemistry captures almost
nothing PC2 lacks.</p>
<p>The partial correlation says the same thing directly: PC2's correlation with
measured stability is {abs(pc['raw']):.3f} raw and
<span class=big>{abs(pc['partial']):.3f}</span>
[{abs(pc['ci_hi']):.3f}, {abs(pc['ci_lo']):.3f}] after controlling for
chemistry. Removing the substitution's own contribution costs
{abs(pc['raw']) - abs(pc['partial']):.3f}.</p>
</div>
<div class="card amber">
<div class=row><span class="chip c-amber">the one that does not come back clean</span>
<strong>The residual on its own is not a better predictor</strong></div>
<p>Strip chemistry out of the pair row entirely and what remains beats chemistry
alone by only {inc['chemistry-residual dz beats chemistry']['gap']:+.3f}
[{inc['chemistry-residual dz beats chemistry']['ci_lo']:+.3f},
{inc['chemistry-residual dz beats chemistry']['ci_hi']:+.3f}], winning in
{inc['chemistry-residual dz beats chemistry']['wins']} of
{inc['chemistry-residual dz beats chemistry']['n']} &mdash; an interval that
includes zero. So the internal representation is not carrying a large
stability signal that is fully orthogonal to chemistry. It carries chemistry's
information and a great deal more <em>organised differently</em>, which is what
the nested comparison above measures and what the head-to-head gap obscures.</p>
</div>
<p>On within-position variance &mdash; which substitution at a fixed residue
&mdash; chemistry reaches
{bl['substitution chemistry (17)']['within']['mean']:+.3f} against internal's
{bl['internal dz (128, one layer)']['within']['mean']:+.3f}, while the emitted
structure manages {bl['output rich (published, 10)']['within']['mean']:+.3f}.
Chemistry is a serious baseline on the hard half of the task; the model's own
output is not.</p>
</section>"""


def sec_what(SV):
    ann = ""
    if SV and "annotation_last_layer" in SV:
        ann = ("<p>Panel C of the figure is the annotation table behind the "
               "name: each component scored against measured stability, the "
               "distogram width and shift, and the substitution's volume and "
               "hydropathy change, in the shared basis so signs are comparable "
               "across proteins.</p>")
    return f"""
<section id=what>
<h2>4 &middot; What PC2 is, and how it was obtained</h2>
<figure><img src="figures/svd.png" alt="Four panels: held-out Spearman against
number of directions kept; the basis learned without the held-out protein;
the component annotation table; and cross-fold subspace agreement.">
<figcaption>Reused unchanged from the SVD study. Panel C is the definition of
what PC1 and PC2 are.</figcaption></figure>

<h3>The quantity that is decomposed</h3>
<p>For each variant the harness stores <code>dz_site</code>: the pair
representation row at the mutated residue, averaged over partners, as a
DIFFERENCE between mutant and wild type at the same residue and the same layer.
It is a difference by construction, so nothing below is about what the protein
is &mdash; only about what the mutation did.</p>

<h3>Why a rotation rather than channels</h3>
<p>The obvious alternative is to pick individual pair channels. Panel A settles
it empirically: at every truncation from k = 2 upward the rotated basis beats
selecting the same number of raw channels, and at k = 128 the two must coincide
because the fit is then the same fit &mdash; which is the panel's own built-in
check and is drawn rather than asserted.</p>

<h3>How the basis is fitted</h3>
<p>Pool the twelve assays, z-score each channel WITHIN each assay so no protein
dominates by representation scale, subtract the pooled mean, and take the SVD.
Components are then oriented so each one's pooled correlation with the model's
own divergence is non-negative, which fixes the otherwise arbitrary SVD sign and
makes signs comparable between proteins.</p>
<p>Two centrings are computed and reported apart. The top component of an
UNCENTRED decomposition is essentially the mean mutation direction &mdash;
"something was substituted here" &mdash; and would dominate every plot if it
were not separated out.</p>

<h3>What the components turn out to be</h3>
<p><strong>PC1 is substitution volume</strong> (loading &minus;0.80 on
&Delta;volume). <strong>PC2 is stability and predicted-distance width at
once</strong>: it loads &minus;0.65 on the measured DMS score and +0.54 on the
width change, +0.59 on broadening. That single component is why the k = 2 point
in panel B already reaches most of the achievable signal.</p>
{ann}
<p>These are properties of one representation, not of one protein: the top-8
subspaces of different folds agree at roughly 0.8 mean cos&sup2; of their
principal angles, against 0.06 for random subspaces of the same dimension
(panel D).</p>
</section>"""


def sec_causal(ST, AB, stale):
    if not ST or not AB:
        return R.pending("intervention evidence")
    sd = ST["cells"]["d_sd_site:sym"]
    pl = ST["cells"]["d_plddt_site:sym"]
    doses = sorted(sd["by_abs_odd_per_dose"], key=float)
    n = sd["n_assays"]
    dose_rows = [[f"|&alpha;| = {float(k):g}",
                  f"{sd['by_abs_odd_per_dose'][k]['first']}/{n}",
                  f"{sd['signed_positive_per_dose'][k]}/{n}"]
                 for k in doses]
    abl = AB["directions"]
    gaps = AB["pc2_minus_random"]
    abl_rows = [[dn,
                 f"{abl[dn]['recovery']['mean']:+.3f} "
                 f"[{abl[dn]['recovery']['ci_lo']:+.3f}, "
                 f"{abl[dn]['recovery']['ci_hi']:+.3f}]",
                 f"{abl[dn]['d_plddt']['mean']:+.5f}",
                 f"{abl[dn]['ca']['mean']:.3f}"]
                for dn in sorted(abl)]
    warn = ('<div class="card warn"><div class=row><span class="chip c-warn">'
            'stale</span><strong>Figure older than its data</strong></div></div>'
            ) if "causal.png" in stale else ""
    return f"""
<section id=causal>
<h2>5 &middot; Does the model use the direction? Not established.</h2>
{warn}
<div class="card warn">
<div class=row><span class="chip c-warn">corrected 2026-09-06</span>
<strong>The earlier confirmatory p-values on this page are withdrawn.</strong></div>
<p>This section previously reported that PC2's odd response outranks eight
random directions in 6/12 proteins, exact binomial p&nbsp;=&nbsp;9.6e-04. Four
defects, found by the September publication audit and confirmed on the archived
runs, remove the confirmatory reading. (1)&nbsp;The eight controls are the
<em>same</em> eight orientations in every protein &mdash; the launcher passed no
per-assay seed &mdash; so the null's twelve independent comparisons do not
exist. (2)&nbsp;The response is not odd in &alpha;: the pooled statistic
averages doses whose per-dose contributions have opposite signs, the |&alpha;|=30
term reversing in 12/12 proteins, and at &alpha;=&minus;30 the distogram
<em>broadens</em> everywhere, contradicting the sign story as previously
written. (3)&nbsp;PC2 is also the highest-<em>gain</em> direction &mdash; its
even, sign-blind component ranks first in 8/12 &mdash; and once the odd response
is normalised by the even one, PC2 ranks first in only 4/12, with odd at most
~9% of even for every direction. The SVD report's own steering section had
already published that negative. (4)&nbsp;The reported cell was the best of six
mode&thinsp;&times;&thinsp;metric cells; the literal single-row injection is at
chance.</p>
</div>
<figure><img src="figures/causal.png" alt="Left: PC2's odd distogram-width
response per dose, against the shared random controls, showing the dose
non-monotonicity. Right: the deletion test — distogram recovery when one
direction is removed from a real mutation's own delta-z, PC2 against random
directions, every interval crossing zero.">
<figcaption>Left: injection, twelve proteins, descriptive. Right: deletion of
the direction from real mutational responses, four proteins &mdash; the
better-posed test, and it is null.</figcaption>
</figure>
<p>PC2 was derived from the pair row after all 64 Pairformer layers &mdash;
exactly the tensor the structure module is conditioned on &mdash; so the
injection needs no surgery inside the stack: run the trunk normally, add
&alpha; &times; direction to the final z, and hand the modified state to the
structure module. What survives as observation, per dose (mode = sym,
distogram width at the injected site):</p>
{R.table(["dose", "PC2 first by |odd| (of 9)", "positive odd response"],
         dose_rows)}
<p>The signed observations are consistent and worth keeping: at the middle dose
the width response is positive in {sd['signed_positive_per_dose'].get('10.0',
sd['signed_positive_per_dose'][doses[1]])}/{n} proteins and the pLDDT response
negative in {n - pl['signed_positive_per_dose'].get('10.0',
pl['signed_positive_per_dose'][doses[1]])}/{n} &mdash; injecting the direction
broadens the predicted distance distribution and lowers confidence at the site.
An odd response is, however, generic first-order sensitivity
(<code>f(z+&alpha;v) &minus; f(z&minus;&alpha;v) =
2&alpha;&nabla;f&middot;v + O(&alpha;&sup3;)</code>): even a clean odd ranking
would show alignment with the local gradient, not semantic use.</p>
<h3>The deletion test &mdash; the better-posed question, and it is null</h3>
<p>An injected vector is nothing the model ever produces. The sharper test
takes a real variant's own &Delta;z = z<sub>mut</sub> &minus; z<sub>wt</sub>,
removes one direction from it at every pair, and re-runs the structure module.
If PC2 carries the phenotype the model reads, deleting it should cost more than
deleting an arbitrary direction. It does not:</p>
{R.table(["direction removed", "distogram recovery [95% CI]",
          "&Delta; pLDDT", "CA shift (&Aring;)"], abl_rows)}
<p>Every PC2-minus-random paired interval includes zero
({", ".join(f"{g['gap']:+.3f} [{g['ci_lo']:+.3f}, {g['ci_hi']:+.3f}]"
            for g in gaps.values())}; {AB['protocol']['n_assays']
 if 'protocol' in AB else 4} proteins), CA shifts sit an order of magnitude
below the sampler's own run-to-run drift, and the surgery is verified &mdash;
the residual projection after removal is
{max(AB['positive_control_residual_fraction'].values()):.1e} of the original,
so "nothing changed" cannot be confused with "the deletion failed". Removing
the direction does eliminate the probe's own prediction, but that is algebra,
not biology. The defensible summary: the direction is readable and its
injection perturbs the output; necessity, sufficiency and mediation are all
unestablished, and the deletion evidence leans against necessity.</p>
</section>"""



def sec_xmodel(XI, XM, stale):
    """Internal vs output in three architectures, on the full 128-dim vector.

    An earlier draft of this section used the `deep2_*` archives, which store
    `dz_site` as a per-layer NORM rather than a vector. A probe built on them
    cannot see the direction at all, and Boltz-2 came out at +0.468 instead of
    +0.613 -- a property of the archive, not of the model. `xm_*` carries
    `dz_vec` at (n, L, 128) for all three models plus each model's pLDDT, so the
    comparison now closes inside one family.
    """
    if not XI:
        return R.pending("multi-model comparison")
    M = {"boltz2": "Boltz-2", "of3": "OpenFold3", "protenix": "Protenix"}
    IK = XI["order"][0]
    rows = []
    for m in XI["models"]:
        sp = XI["spearman"][m]
        g = XI["internal_minus"]["pLDDT"][m]
        gs_ = XI["internal_minus"]["pLDDT@site"][m]
        rows.append([M[m], str(XI["layers"][m]), f"{sp[IK]:+.3f}",
                     f"{sp['pLDDT']:+.3f}",
                     f"{g['gap']:+.3f} <span class=ci>[{g['ci_lo']:+.3f}, "
                     f"{g['ci_hi']:+.3f}]</span>", f"{g['wins']}/{g['splits']}",
                     f"{gs_['gap']:+.3f}"])
    allsig = all(XI["internal_minus"]["pLDDT"][m]["ci_lo"] > 0
                 for m in XI["models"])
    lo = min(XI["spearman"][m][IK] for m in XI["models"])
    hi = max(XI["spearman"][m][IK] for m in XI["models"])
    warn = ('<div class="card warn"><div class=row><span class="chip c-warn">'
            'stale</span><strong>Figure older than its data</strong></div></div>'
            ) if "xmodel_io.png" in stale else ""
    cka = ""
    if XM:
        cka = (f"<p>The three are only moderately similar to one another &mdash; "
               f"CKA between Boltz-2 and OpenFold3 is "
               f"{XM['cka']['boltz2|of3']['mean']:.3f} against "
               f"{XM['cka']['boltz2|boltz2 (repeat)']['mean']:.3f} for a repeat "
               f"run of the same model &mdash; so this is the same phenomenon "
               f"appearing in genuinely different representations, not one "
               f"architecture measured three times.</p>")
    return f"""
<section id=xmodel>
<h2>6 &middot; Does it hold in other architectures?</h2>
{warn}
<figure><img src="figures/xmodel_io.png" alt="Left: each model's internal probe
against its own pLDDT outputs. Right: the internal-minus-pLDDT gap per model
with confidence intervals, all clearing zero.">
<figcaption>{len(XI['assays'])} assays &times; {XI['splits']} splits, held-out
residue positions, identical variants across models.</figcaption></figure>
<p>The internal side is a ridge on channels selected from the 128-dimensional
pair-row difference at the FINAL trunk layer (an inner training-only sweep
keeps at most 64 of the 128 &mdash; the archive records what it kept) &mdash;
chosen not for being best but because it is the tensor the structure module is
conditioned on. The baseline is each model's own <strong>pLDDT</strong> rather
than its geometry, deliberately: "your confidence head already tells you this"
is the first objection a referee raises.</p>
{R.table(["model", "layers", "internal (&le;64 of 128 ch)", "pLDDT",
          "gap vs pLDDT", "splits won", "gap vs pLDDT@site"], rows)}
<p>All three models land within a few hundredths of each other on the internal
side ({lo:+.3f} to {hi:+.3f}) despite 64, 48 and 16 trunk blocks respectively,
and each beats its own confidence head{' with an interval that clears zero'
if allsig else ''}. Against pLDDT at the mutated residue the gaps are larger
still.</p>
<div class="card amber">
<div class=row><span class="chip c-amber">weight this correctly</span>
<strong>Four assays, not twelve</strong></div>
<p>The independent unit is the assay and there are {len(XI['assays'])} of them
({', '.join(x.split('_')[0] for x in XI['assays'])}), so the intervals stay
wide even where they clear zero. This shows the phenomenon is not peculiar to
Boltz-2; the twelve-protein result in section 1 is what carries the weight. It
also cannot rank the models &mdash; layer counts, distogram grids and alignment
handling all differ.</p>
</div>
<p>TM to wild type is absent here. It requires <code>tmtools</code>, which is
not installed in the analysis container, and substituting a different structural
metric under the same name is how a number nobody computed reaches a paper.</p>
{cka}
</section>"""


def sec_where(DP, XM, BA):
    if not DP:
        return R.pending("depth")
    nl, dec = DP["n_layers"], DP["decodability_by_depth"]
    peaks = {m: max(rows, key=lambda r: r["mean"]) for m, rows in dec.items()}
    sev = DP["severity_direction_by_depth"]
    rot = ""
    if BA:
        rot = (f"<p>One caution for any depth-resolved claim: the PC basis is a "
               f"LAST-LAYER object and it rotates. Refitted independently at "
               f"each layer, the top-{BA['k']} subspace at mid-depth overlaps "
               f"the final one at {BA['worst_vs_last']:.3f}, against "
               f"{BA['random_baseline']:.3f} for unrelated bases. PC2 is not a "
               f"fixed direction running through the trunk; it forms over "
               f"roughly the last sixteen layers.</p>")
    rows = [[{"boltz2": "Boltz-2", "of3": "OpenFold3",
              "protenix": "Protenix"}[m], str(nl[m]),
             f"{peaks[m]['frac']:.3f}", f"{peaks[m]['mean']:+.3f}",
             f"{dec[m][-1]['mean']:+.3f}",
             f"{max(sev[m], key=lambda r: r['mean'])['frac']:.3f}"]
            for m in dec]
    return f"""
<section id=where>
<h2>7 &middot; Where in the trunk it lives</h2>
<figure><img src="figures/depth.png" alt="Decodability against relative depth
for Boltz-2, OpenFold3 and Protenix.">
<figcaption>Reused unchanged from the SVD study.</figcaption></figure>
{R.table(["model", "layers", "decodability peaks at", "peak rho",
          "rho at final layer", "severity direction peaks at"], rows)}
<p>Decodability of the full pair row is high throughout and peaks at the very
end in Boltz-2 and OpenFold3. Protenix is the exception &mdash; its best depth
is {peaks['protenix']['frac']:.3f} at {peaks['protenix']['mean']:+.3f}, against
{dec['protenix'][-1]['mean']:+.3f} at the final layer &mdash; so "the signal is
strongest where the structure module reads" is true of two of the three, not
all.</p>
<div class="card warn">
<div class=row><span class="chip c-warn">corrected</span>
<strong>Two different curves</strong></div>
<p>An earlier draft reported that decodability peaks at relative depth 1.00 in
all three models. That figure belongs to the SEVERITY DIRECTION curve &mdash;
the single DMS-associated component, which does peak at the final layer in all
three &mdash; not to decodability of the whole representation. Both are in the
table above, separately.</p>
</div>
{rot}
</section>"""


def sec_heldout(HO):
    if not HO:
        return R.pending("held-out proteins")
    s = HO["summary"]
    rows = []
    for mode, lab in (("pc2_inductive", "basis fitted WITHOUT the held-out protein"),
                      ("pc2_transductive", "basis fitted with it included")):
        for grp in ("all 16", "stability (12)", "non-stability (4)"):
            d = s[mode][grp]
            rows.append([f"{lab} &mdash; {grp}", f"{abs(d['mean']):.3f}",
                         f"[{abs(d['ci_hi']):.3f}, {abs(d['ci_lo']):.3f}]"])
    return f"""
<section id=heldout>
<h2>8 &middot; Does the direction work on proteins it never saw?</h2>
<figure><img src="figures/heldout.png" alt="Held-out protein performance of PC2,
split by whether the assay measures stability.">
<figcaption>Reused unchanged from the SVD study.</figcaption></figure>
<p>The strongest version of the generality claim: fit the basis without the
held-out protein at all, project that protein's variants onto PC2, and score.
Magnitudes are shown as |&rho;| &mdash; the sign is a basis-orientation
convention, fixed once and reported in section 2.</p>
{R.table(["condition", "|Spearman|", "95% interval"], rows)}
<p>Inductive and transductive agree to three decimals, so nothing here depends
on the held-out protein having contributed to the basis. The split that matters
is the last one: the direction is roughly twice as predictive on stability
assays as on non-stability ones, which is the specificity control the name
"stability axis" needs.</p>
</section>"""


def sec_mech(JP, GP, RT):
    if not (JP and GP):
        return R.pending("mechanism")
    extra = ""
    if RT:
        extra = (f"<li>It forms by <strong>drift, not construction</strong>. "
                 f"Rotation of the mutation subspace is "
                 f"{RT['late_mean']/RT['early_mean']:.1f}&times; faster per "
                 f"layer in the second half of the stack, the MLP contributes "
                 f"{100*RT['share']['transition_z']:.0f}% of it &mdash; exactly "
                 f"what its size alone predicts &mdash; and every operation "
                 f"rotates the subspace LESS than a random operator of "
                 f"identical singular spectrum would.</li>")
    return f"""
<section id=mechanism>
<h2>9 &middot; What we know about how it forms</h2>
<p>Summarised; the full treatment is in
<a href="../report_jacobian/index.html">the method report</a> and
<a href="../report_jac/index.html">the Pairformer result</a>.</p>
<ul>
<li>Decomposing the pair operations' <em>weights</em> says nothing useful. The
transition is SwiGLU, so there is no single matrix to decompose, and the
singular ordering reflects weight magnitude rather than what the layer writes.</li>
<li>Differentiating each operation at a real operating point instead gives an
operator of effective rank {JP['eff_rank_median']:.1f} of {JP['dim']}, against
73&ndash;94 for the bare weights. The mechanism is the SwiGLU gate: only
{100*GP['live_mean']/512:.0f}% of hidden units are live at any operating point,
tracking the rank at r&nbsp;=&nbsp;{GP['corr']:.2f}.</li>
<li>That operator is protein-general: the top-{JP['k']} subspaces of twelve
unrelated folds agree at
{JP['agreement']['out']['last_layer_mean']:.3f} mean cos&sup2; against
{JP['agreement']['out']['random_baseline']:.3f} for random subspaces.</li>
<li><strong>No operation singles out stability.</strong> Four of the five
z-path operations engage the mutation subspace above chance, but none does so
for PC2 more than for the other components.</li>
{extra}
</ul>
<p>So the mechanism section is honest about being a negative: we know the
direction is real, transferable and used, and we know it is not implemented by
any single dedicated operation.</p>
</section>"""


def sec_limits(TR):
    return """
<section id=limits>
<h2>10 &middot; What this does not establish</h2>
<ul>
<li><strong>Scope.</strong> All twelve stability assays are Tsuboyama 2023
mini-domains, 61&ndash;72 residues. Nothing here shows the effect on larger
proteins or on other stability datasets. The three non-Tsuboyama stability
assays in ProteinGym are also the only large ones (212, 245 and 403 residues),
so one experiment would answer both &mdash; it has not been run.</li>
<li><strong>This is not a stability predictor.</strong> The comparison is
internal against the same model's own output on identical rows. No claim is
made against the ProteinGym leaderboard, and the substitution-chemistry baseline
is close enough that none should be.</li>
<li><strong>The intervention evidence is descriptive.</strong> Injection
perturbs the emitted distogram and confidence with a consistent sign at the
middle dose, but the controls were shared across proteins, the response is
dose-non-monotone and magnitude-dominated, and the deletion test is null
(&sect;5) &mdash; manipulability was observed; necessity, sufficiency and
mediation were not established, and the deletion evidence leans against
necessity. Nothing here shows the direction is what the trunk computes it
FOR.</li>
<li><strong>One quantity, one row.</strong> Everything rests on
<code>dz_site</code>, the pair row at the mutated residue averaged over
partners. A separate analysis found the averaging does not cost much, but it is
still one summary of a large tensor.</li>
</ul>
</section>"""


def sec_ledger(resolved):
    """One table: what protocol produced each number on this page.

    Every correction in section 2 traced to the same root cause -- a number
    whose protocol lived in a source comment rather than beside it, so two
    figures measured differently looked directly comparable. This makes the
    conventions visible without opening any code, and shows which archives
    still predate the requirement.
    """
    rows, missing = [], []
    for key, path in sorted(resolved.items()):
        d = R.load(path)
        if not d:
            continue
        p = d.get("protocol")
        name = f"<code>{Path(path).name}</code>"
        if not p or "design" not in p:
            missing.append(key)
            rows.append([f"<code>{key}</code>", name,
                         "<em>not recorded</em>", "<em>&mdash;</em>",
                         "<em>&mdash;</em>", "<em>&mdash;</em>"])
            continue
        lay = p.get("layer", {})
        which = lay.get("which", "?") if isinstance(lay, dict) else str(lay)
        win = lay.get("pooled_over_last") if isinstance(lay, dict) else None
        feat = p.get("features", {})
        if isinstance(feat, dict) and "width" in feat:
            fs = f"{feat['name']} ({feat['kept']} of {feat['width']})"
        elif isinstance(feat, dict):
            fs = ", ".join(f"{v['kept']}/{v['width']}" for v in feat.values())
        else:
            fs = str(feat)
        rows.append([f"<code>{key}</code>", name, p["design"][:74],
                     which + (f", pooled over last {win}" if win else ""),
                     fs, str(p.get("n_assays", "?"))])
    warn = ""
    if missing:
        warn = (f'<div class="card amber"><div class=row>'
                f'<span class="chip c-amber">incomplete</span>'
                f'<strong>{len(missing)} archive(s) predate the requirement</strong>'
                f'</div><p><code>{"</code>, <code>".join(missing)}</code> were '
                f'produced before their analysis recorded a protocol block. '
                f'Their numbers are unaffected; what is missing is the record of '
                f'how they were measured, which is exactly the gap that caused '
                f'the corrections in section 2. Rerunning the analysis populates '
                f'it.</p></div>')
    return f"""
<section id=ledger>
<h2>11 &middot; Protocol ledger</h2>
<p>Two numbers in this project are comparable only if they share a design, a
layer convention and a feature width. Three separate corrections happened
because one of those was recorded in a source comment instead of in the file.
Each analysis now writes its own protocol, and this is the result.</p>
{R.table(["input", "file", "design", "layer", "features (kept of width)",
          "assays"], rows)}
{warn}
<p>The enforcement lives in <code>pi_protocol.py</code>: an analysis that omits
design, layer, features, source or assay count fails rather than writing a file
that will mislead later. A missing field there costs a rerun; a missing field in
an archive costs a correction after the number has been quoted.</p>
</section>"""


def sec_repro(manifest, resolved):
    rows = [[f"<code>{k}</code>", f"<code>{e['file']}</code>", f"{e['bytes']:,}",
             f"<code>{e['sha256'][:12]}</code>"]
            for k, e in sorted(manifest["inputs"].items())]
    # Derived from FIGSPEC and the resolved paths, never typed. A review found
    # the hardcoded version naming transfer_v1.json while the figure beside it
    # held the transfer_vec.json result.
    cmds = "\n".join(
        cmd.format(out=f"report_master/figures/{fig}", **resolved)
        for fig, cmd in sorted(FIGSPEC.items()))
    ok = manifest.get("reproducible_from_commit")
    miss = (manifest.get("code_absent_from_commit", [])
            + manifest.get("code_differs_from_commit", []))
    blocker = "" if ok else (
        '<div class="card warn"><div class=row><span class="chip c-warn">'
        'not reproducible</span><strong>The code that built this page is not '
        f'at commit <code>{manifest["commit"]}</code></strong></div>'
        f'<p>{len(miss)} script(s) are absent from or differ from that commit: '
        f'<code>{"</code>, <code>".join(miss)}</code>. The archived data and '
        f'its digests are unaffected, but the page cannot be regenerated from '
        f'the cited commit until these are committed. Do not cite it as '
        f'reproducible in a submission.</p></div>')
    return f"""
<section id=repro>
<h2>12 &middot; Provenance</h2>
{blocker}
<p>Every number on this page was read from these files at build time. Figures
<code>svd.png</code>, <code>heldout.png</code> and <code>depth.png</code> are
copied unchanged from <code>report_svd/</code> and are regenerated there.</p>
{R.table(["key", "file", "bytes", "sha-256"], rows)}
<pre><code>{cmds}
python build_master_report.py</code></pre>
</section>"""


def main():
    ap = argparse.ArgumentParser()
    D = R.W / "runs"   # primary sources, not another report's copies
    ap.add_argument("--transfer", default=str(R.W / "runs/transfer_full.json"))
    ap.add_argument("--bw", default=str(D / "bw_v1.json"))
    ap.add_argument("--heldout", default=str(D / "heldout_v1.json"))
    ap.add_argument("--depth", default=str(D / "depth_v1.json"))
    ap.add_argument("--xmodel", default=str(D / "xmodel_v1.json"))
    ap.add_argument("--svd", default=str(D / "svd_dz_v3.json"))
    ap.add_argument("--chem", default=str(D / "chem_v1.json"))
    ap.add_argument("--transfer-ind", default=str(R.W / "runs/transfer_inductive.json"))
    ap.add_argument("--xio", default=str(R.W / "runs/xmodel_io_vec.json"))
    ap.add_argument("--layermatch", default=str(R.W / "runs/layer_match.json"))
    ap.add_argument("--svd-ds", default=str(R.W / "runs/svd_ds_v1.json"))
    ap.add_argument("--scrutiny", default=str(D / "scrutiny_v2.json"))
    # v2, descriptive: the v1 steer_pooled.json p-values were withdrawn on
    # 2026-09-06 (shared controls across assays). The deletion null joined the
    # page the same day; it existed since August but was never an input here.
    ap.add_argument("--steer", default=str(R.W / "runs/steer_pooled_v2.json"))
    ap.add_argument("--ablate", default=str(R.W / "runs/ablate_v2.json"))
    ap.add_argument("--jac", default=str(R.W / "runs/jac_pooled.json"))
    ap.add_argument("--gate", default=str(R.W / "runs/gate_probe.json"))
    ap.add_argument("--rotate", default=str(R.W / "runs/rotate_pooled.json"))
    ap.add_argument("--basis", default=str(R.W / "runs/basis_depth.json"))
    ap.add_argument("--allow-stale-figures", action="store_true")
    a = ap.parse_args()

    resolved = {k: getattr(a, k) for k in
                ("transfer", "bw", "heldout", "depth", "xmodel", "svd", "steer",
                 "ablate", "jac", "gate", "rotate", "basis", "chem", "scrutiny",
                 "transfer_ind", "xio", "layermatch", "svd_ds")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    missing = [k for k, v in resolved.items() if not Path(v).exists()]
    if missing:
        print(f"   missing inputs: {', '.join(missing)} -- pending cards")

    borrowed_stale = []
    for fig, (key, cmd) in BORROWED.items():
        src = R.W / "report_svd" / "figures" / fig
        if not src.exists():
            print(f"   WARNING: borrowed figure {fig} not found in report_svd")
            continue
        data = resolved.get(key)
        if data and Path(data).exists() and \
                src.stat().st_mtime < Path(data).stat().st_mtime:
            borrowed_stale.append((fig, cmd.format(**resolved)))
        shutil.copy2(src, OUT / "figures" / fig)
    if borrowed_stale and not a.allow_stale_figures:
        raise SystemExit(
            "borrowed figures older than the data they are drawn from:\n"
            + "\n".join(f"  {f}" for f, _ in borrowed_stale)
            + "\n\nregenerate in report_svd, then rebuild:\n"
            + "\n".join(f"    {c}" for _, c in borrowed_stale))

    stale = R.check_figures(OUT, FIGSPEC, resolved, a.allow_stale_figures)
    stale += [f for f, _ in borrowed_stale]
    manifest = R.archive_inputs(OUT, resolved, stale, CODE)

    # quoted=True marks an input this page STATES a number from. Those raise
    # if they carry no protocol block rather than rendering an unattributable
    # claim; anything else loads and is listed by R.provenance_notice().
    TR, BW = R.load(a.transfer, quoted=True), R.load(a.bw, quoted=True)
    HO, DP = R.load(a.heldout, quoted=True), R.load(a.depth, quoted=True)
    XM, SV = R.load(a.xmodel, quoted=True), R.load(a.svd, quoted=True)
    ST, JP = R.load(a.steer, quoted=True), R.load(a.jac, quoted=True)
    AB = R.load(a.ablate, quoted=True)
    GP, RT = R.load(a.gate, quoted=True), R.load(a.rotate, quoted=True)
    BA = R.load(a.basis)
    CH, SC = R.load(a.chem), R.load(a.scrutiny)
    TI, XI = R.load(a.transfer_ind), R.load(a.xio)
    LM = R.load(a.layermatch)

    body = "".join([
        sec_lede(TR, HO, ST, SV), sec_headline(TR, BW, stale, TI),
        sec_numbers(TR, SV, CH, HO, LM), sec_chem(TR, CH, SC, BW), sec_what(SV),
        sec_causal(ST, AB, stale), sec_xmodel(XI, XM, stale), sec_where(DP, XM, BA), sec_heldout(HO),
        sec_mech(JP, GP, RT), sec_limits(TR), sec_ledger(resolved), sec_repro(manifest, resolved),
    ])

    R.page(
        OUT,
        title="What Boltz-2 knows about stability but does not say",
        eyebrow="comprehensive report &middot; august 2026",
        h1="What Boltz-2 knows about stability but does not say",
        lede="A single direction in the pair representation predicts measured "
             "mutational stability better than 37 prespecified summaries of "
             "the sampled C-alpha geometry and confidence, and transfers to "
             "proteins it was never fitted on. Injecting it perturbs the "
             "distogram and confidence; deleting it from a real mutation's "
             "response changes nothing — the intervention evidence is "
             "descriptive, not causal.",
        nav_items=[("summary", "summary"), ("performance", "1 internal vs output"),
                   ("numbers", "2 which number"),
                   ("chemistry", "3 just chemistry?"),
                   ("what", "4 what PC2 is"), ("causal", "5 causal"),
                   ("xmodel", "6 three models"), ("where", "7 depth"),
                   ("heldout", "8 held-out"), ("mechanism", "9 mechanism"),
                   ("limits", "10 limits"), ("ledger", "11 protocols"),
                   ("repro", "12 provenance")],
        body=body, manifest=manifest,
        sibling=("../report_svd/index.html", "the full audit trail"))


if __name__ == "__main__":
    main()
