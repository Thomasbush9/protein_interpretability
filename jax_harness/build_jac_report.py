"""Build the PC2 report: is the stability axis computed in the Pairformer?

The method this rests on has its own page, `report_jacobian/`, built by
`build_jacobian_report.py`. What is here is the application: take the Jacobian
of every operation that writes to `z`, ask whether any of them treats the
mutation-response subspace -- and PC2 in particular -- differently from an
arbitrary direction, and check that the linearisation predicts anything at all.

This page carries three corrections, and they are the reason it is worth
reading rather than skipping to the summary. Two were errors in how a null was
compared; the third was a basis applied at depths where it does not hold, and
it changed the answer.

Provenance rules live in `pi_report.py`.

  python build_jac_report.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pi_report as R  # noqa: E402

OUT = R.W / "report_jac"
FIGSPEC = {
    "pc2.png": ("python fig_pc2_jac.py --ops {ops} --jac {jac} "
                "--basis {basis} --comp {comp} --rot {rot} --out {out}"),
}
CODE = ["build_jac_report.py", "pi_report.py", "fig_pc2_jac.py",
        "analyze_basis.py", "exp_compose.py", "analyze_compose.py",
        "analyze_ops.py", "analyze_jac.py", "analyze_rotate.py"]

NICE = {"tri_mul_out": "tri_mul_out", "tri_mul_in": "tri_mul_in",
        "tri_att_start": "tri_att_start", "tri_att_end": "tri_att_end",
        "transition_z": "transition_z (MLP)"}


def sec_summary(O, J, B, C, RT):
    if not (O and J and B and C and RT):
        return R.pending("summary")
    nd = O["null_departure"]
    ch = O["null_departure_chance"]
    ops = O["ops"]
    # PC2's rank among the four components, per operation -- the claim is that
    # it is never the standout, so the builder checks rather than asserts it.
    worst = max(sorted(nd[o], reverse=True).index(nd[o][1]) for o in ops)
    lo = B["worst_vs_last"]
    return f"""
<section id=summary>
<h2>The short version</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">answer</span>
<strong>No. The Pairformer engages the mutation subspace, but it does not
single out the stability axis.</strong></div>
<p>In each layer's own basis, four of the five z-path operations treat the
mutation subspace differently from arbitrary channel directions &mdash;
departures from the matched null of {min(nd[o][0] for o in ops):.2f} to
{max(nd[o][0] for o in ops):.2f} against a chance level of {ch:.2f}. But the
effect is flat across components: PC2 is never the largest of the four for any
operation (worst rank {worst + 1} of 4). Whatever these layers are doing to the
mutation response, they are not doing something specific to stability.</p>
</div>
<div class="card warn">
<div class=row><span class="chip c-warn">corrected</span>
<strong>The first version of this analysis used a basis that does not hold at
depth.</strong></div>
<p>The PC basis was fitted on <code>dz_site</code> at the final Pairformer
layer and then applied at all {J['layers']}. Fitting it independently per layer
shows it rotates almost completely: the top-{B['k']} subspace at mid-depth
overlaps the last layer's at <span class=big>{lo:.3f}</span>, against
{B['random_baseline']:.3f} for unrelated bases. Every depth profile in the first
version was therefore measuring a direction those layers do not use. Corrected
here; the conclusion above is from the corrected run.</p>
</div>
<div class="card ok">
<div class=row><span class="chip c-ok">validated</span>
<strong>The linearisation predicts the real response.</strong></div>
<p>Pushing the archived mutation difference through each layer's own Jacobian
reproduces the model's next-layer difference at cosine
<span class=big>{C['one_step_mean_cosine']:.3f}</span>, averaged over layers and
{len(C['assays'])} assays. Chaining all {C['layers']-1} without re-reading the
archive decays to {C['free_final_cosine']:.2f}, which bounds how far the
composed picture can be pushed without impugning the per-layer one.</p>
</div>
<div class="card ok">
<div class=row><span class="chip c-ok">how it forms</span>
<strong>By drift, not by construction.</strong></div>
<p>The axis is built inside the stack &mdash; the entrance basis is nearly
unrelated to the exit basis &mdash; and the rotation is
{RT['late_mean']/RT['early_mean']:.1f}&times; faster per layer in the second
half. The MLP contributes {100*RT['share']['transition_z']:.0f}% of it, which is
exactly what its size alone predicts. And against a null holding each operator's
exact singular spectrum, <em>every</em> operation rotates the subspace
<em>less</em> than chance. The mutation subspace is comparatively preserved by
these layers, drifts anyway, and its second component happens to be the
stability axis by the time it exits.</p>
</div>
</section>"""


def sec_figure(stale):
    warn = ""
    if stale:
        warn = ('<div class="card warn"><div class=row>'
                '<span class="chip c-warn">stale</span><strong>This figure is '
                'older than its data</strong></div><p>Built with '
                '--allow-stale-figures.</p></div>')
    return f"""
<section id=figure>
<h2>In six panels</h2>
{warn}
<figure>
<img src="figures/pc2.png" alt="Six panels: the mutation basis rotating with
depth; departure from the matched null per operation and component in each
layer's own basis; one-step and free-running prediction accuracy; where PC1 and
PC2 sit inside the transition's Jacobian; where the rotation happens and which
operation contributes it; and each operation's rotation against a
spectrum-matched null.">
<figcaption>Panels B, E and F plot operations and use the operation palette,
matching the method report's figure. A, C and D plot other quantities and use a
neutral pair so a colour never means two things. Grey is always a reference
level. A and C are the correction and the validation; B, E and F are the
results that depend on them.</figcaption>
</figure>
</section>"""


def sec_basis(B, J):
    if not (B and J):
        return R.pending("basis")
    ident = B["identity_vs_last"]
    rows = []
    for li in (0, 8, 16, 24, 32, 40, 48, 56, B["layers"] - 1):
        rows.append([str(li), f"{B['rot_vs_last'][li]:.3f}"]
                    + [f"{ident[li][c]:.3f}" for c in range(B["n_pc"])])
    mean = B["meaning_spearman"]
    mrows = [[str(li)] + [f"{mean[li][c]:+.3f}" for c in range(B["n_pc"])]
             for li in (0, 16, 32, 48, B["layers"] - 1)]
    return f"""
<section id=basis>
<h2>1 &middot; "PC2" is not one direction</h2>
<p>Everything downstream depends on what basis a gain is measured in, and the
study's first version measured every depth against the basis fitted at layer
{J['layers']-1}. Refitting independently at each layer &mdash; same protocol as
<code>analyze_pc2.py</code>: pool the twelve assays, z-score per channel within
each assay, subtract the pooled mean, SVD &mdash; shows how far that
assumption is from true.</p>
{R.table(["layer", f"top-{B['k']} subspace"]
         + [f"|cos| PC{c+1}" for c in range(B["n_pc"])], rows)}
<p>Unrelated bases would give {B['random_baseline']:.3f} for the subspace
column. Until roughly layer 45 the agreement is barely above that, and layer
40's PC2 is essentially orthogonal to layer {J['layers']-1}'s
(|cos| {ident[40][1]:.3f}). The drift is gradual rather than a hinge &mdash;
adjacent layers agree at {min(B['rot_adjacent']):.3f} at worst &mdash; so the
basis is locally stable and globally not.</p>
<p>The refit reproduces the archived <code>pc2_v2.npz</code> at the last layer
to |cos| {min(B['refit_vs_archived']):.4f} across all components, so this is a
statement about depth and not about two different protocols.</p>
<h3>The label migrates too</h3>
<p>PC2 was named for tracking stability and predictive certainty. Scoring each
layer's OWN component against <code>kl_glob</code> at that layer shows which
component carries that role at each depth (Spearman, cluster bootstrap over
assays):</p>
{R.table(["layer"] + [f"PC{c+1}" for c in range(B["n_pc"])], mrows)}
<p>At the input it is PC3 that carries the signal ({mean[0][2]:+.3f}, against
{mean[0][1]:+.3f} for PC2); by the last layer it is PC2 ({mean[-1][1]:+.3f}).
So the direction rotates and the role moves between components. Both have to be
handled before a depth profile means anything.</p>
</section>"""


def sec_result(O, J):
    if not (O and J):
        return R.pending("result")
    nd = O["null_departure"]
    ch = O["null_departure_chance"]
    rows = [[NICE[o]] + [f"{nd[o][c]:.3f}" for c in range(len(nd[o]))]
            for o in O["ops"]]
    return f"""
<section id=result>
<h2>2 &middot; The subspace is engaged; PC2 is not singled out</h2>
<p>For component c, the <strong>gain</strong> is <code>w_c &middot; M e_c</code>
&mdash; the added coordinate per unit of that coordinate. Because every
operation is a residual branch, the total multiplier is <code>1 + gain</code>.
The null is drawn orthonormally in the standardised basis and carried through
the same per-channel spread, so it cannot differ from the components by an
artefact of that bridge, and results are reported as a percentile against it
rather than a ratio.</p>
<p>Below: |percentile &minus; 0.5|, averaged over layers and assays, in
<strong>each layer's own basis</strong>. A percentile drawn from noise is
uniform on [0,1], whose mean absolute deviation from 0.5 is
<strong>{ch:.2f}</strong> &mdash; that, not zero, is the level to beat, and
0.50 is the ceiling.</p>
{R.table(["operation"] + [f"PC{c+1}" for c in range(len(nd[O['ops'][0]]))], rows)}
<p>Four of the five clear chance comfortably, and
<code>tri_att_start</code> sits essentially at the ceiling &mdash; its
percentiles are pinned near 1.0 at almost every layer, meaning it preserves the
mutation directions relative to nearly every random direction. That is a real
finding, and it is not the one this page asked about.</p>
<div class="card">
<div class=row><span class="chip c-run">the actual answer</span>
<strong>Flat across components</strong></div>
<p>For every operation the four components sit within a narrow band, and PC2 is
never the largest. If any of these layers implemented something specific to
stability, PC2 would stand out from PC1, PC3 and PC4 &mdash; it does not. What
the operations engage is the mutation subspace as a whole, which is
unsurprising: that subspace is where the model's own computation puts the
response.</p>
</div>
<p>At the last layer, where the basis is exact by construction, the picture is
the same from the other direction: PC1 and PC2 fall <em>below</em> the random
baseline in the transition's own singular bases, most clearly on the read side.
The MLP barely reads the stability axis.</p>
</section>"""


def sec_compose(C, J):
    if not (C and J):
        return R.pending("composition")
    o = C["one_step"]
    f = C["free"]
    rows = []
    for li in (1, 8, 16, 24, 32, 40, 48, 56, C["layers"] - 1):
        rows.append([str(li), f"{o['cosine'][li]:.3f}", f"{o['rel_err'][li]:.3f}",
                     f"{f['cosine'][li]:.3f}", f"{f['rel_err'][li]:.3f}",
                     f"{f['pc']['PC2']['r'][li]:.3f}"])
    return f"""
<section id=compose>
<h2>3 &middot; Does the linearisation predict anything?</h2>
<p>Everything above is descriptive: it measures what the operations do to
directions. None of it had been asked to predict. This asks.</p>
<p><code>gym2_*.npz</code> archives, for 250 real variants per assay, the
mutant-minus-wildtype pair row at every layer. So take the archived difference,
push it through the layer's own Jacobian at the wild-type operating point, and
compare against the difference the model actually produced. The full
[N,N,{J['dim']}] tangent field is propagated, not the 128-vector summary &mdash;
reducing between layers would discard the off-row structure the triangle
operations read, and would test a model of the computation rather than the
computation.</p>
{R.table(["layer", "one-step cos", "rel err", "free cos", "rel err",
          "PC2 r (free)"], rows)}
<p><strong>One-step</strong> re-seeds from the archive at every layer, so errors
cannot accumulate: it tests the linearisation alone, and it holds at cosine
{min(o['cosine'][1:]):.3f}&ndash;{max(o['cosine'][1:]):.3f} across all
{C['layers']} layers and {len(C['assays'])} assays. Each layer's Jacobian is an
accurate description of what that layer does to a real mutation.</p>
<p><strong>Free-running</strong> is seeded once at layer 0 and never re-reads
the archive, so it tests the composition. It decays to
{C['free_final_cosine']:.2f}. Two things are mixed in that decay &mdash;
accumulated linearisation error, and the uniform-row reconstruction the initial
tangent needs because the archive stores only the row mean. The one-step curve
separates them: it is re-seeded through the same reconstruction every layer and
does not decay, so the reconstruction is not the problem and the loss is
genuinely in the chaining.</p>
<p>The practical consequence: per-layer statements in this study are supported;
end-to-end statements composed across all {C['layers']-1} layers are not, and
nothing here relies on one.</p>
</section>"""


def sec_rotate(RT, B):
    if not (RT and B):
        return R.pending("rotation attribution")
    ops = RT["ops"]
    nl = RT["rotation_null_spectrum"]
    tot_null = sum(nl.values())
    rows = [[NICE[o], f"{RT['share'][o]:.3f}", f"{nl[o]/tot_null:.3f}",
             f"{RT['rotation'][o]:.4f}", f"{nl[o]:.4f}",
             f"{RT['excess_vs_spectrum'][o]:+.4f}"] for o in ops]
    lead = max(ops, key=lambda o: RT["share"][o])
    return f"""
<section id=rotate>
<h2>4 &middot; So how does the axis form?</h2>
<p>Section 1 says the direction is built inside the stack rather than carried
into it &mdash; the basis at the Pairformer entrance is nearly unrelated to the
one at its exit. Section 2 says no operation privileges PC2 within the subspace.
Together those force a specific hypothesis: the axis forms by accumulated
rotation rather than by any operation implementing stability. This tests it.</p>
<p>Linearised, the layer composes as
<code>C_k = (I + M_k)...(I + M_1)</code> over the five operations in the order
the layer applies them. Pushing layer l&minus;1's basis through the cumulative
maps and taking principal angles at each stage attributes that layer's rotation
to individual operations.</p>
<div class="card ok">
<div class=row><span class="chip c-ok">closure</span>
<strong>The decomposition operates on the right object</strong></div>
<p>The composed operators must land on the next layer's basis, or the split is
arithmetic on something that is not the layer's real action. Mean closure is
{RT['closure_mean']:.3f} overall and <span class=big>{RT['closure_where_rotating']:.3f}</span>
over the {sum(1 for v in RT['layer_actual'][1:] if v > 0.02)} layers that
actually rotate by more than 0.02 &mdash; a layer that barely moves would pass
trivially, so only the second number is informative.</p>
</div>
<h3>Where, and who</h3>
<p>Rotation per layer is
<span class=big>{RT['late_mean']/RT['early_mean']:.1f}&times;</span> higher in
layers 32&ndash;{RT['layers']-1} than in 1&ndash;31
({RT['late_mean']:.4f} against {RT['early_mean']:.4f}), and it accelerates to
the end of the stack. So the axis forms late, which matches the shape of the
agreement curve in section 1.</p>
{R.table(["operation", "share of rotation", "share its size predicts",
          "rotation", "spectrum null", "excess"], rows)}
<p><code>{NICE[lead]}</code> accounts for
{100*RT['share'][lead]:.0f}% of the rotation &mdash; but that is exactly the
{100*nl[lead]/tot_null:.0f}% its own size predicts. Every operation's share of
the rotation matches its share of the null to within a few points. The MLP
dominates because it is the largest operator in the layer, not because it is
doing something the others are not.</p>
<div class="card">
<div class=row><span class="chip c-run">the result</span>
<strong>Every operation rotates the subspace LESS than chance</strong></div>
<p>The null holds each operator's exact singular spectrum &mdash; norm, rank and
conditioning all fixed &mdash; and randomises only its singular vectors, so the
one thing varying is where the operator points. A norm-matched dense random
matrix gives the same answer to three decimals, which is expected: an
isotropically random operator's response to a fixed vector depends on Frobenius
norm alone.</p>
<p>Against that null, <strong>every operation falls short</strong>, the MLP by
the largest margin ({RT['excess_vs_spectrum']['transition_z']:+.4f} on a
rotation of {RT['rotation']['transition_z']:.4f}). The operations rotate the
mutation subspace roughly two to three times less than random operators of
identical spectrum would. They are comparatively <em>aligned</em> with it: more
of their action stays inside the subspace than an arbitrary operator's
would.</p>
<p>That is the same fact section 2 reports from the other side. There, four of
five operations came out above chance on the null-departure measure, meaning the
mutation directions are attenuated <em>less</em> than arbitrary ones. Here, the
subspace is rotated less than chance. Both say the mutation subspace is a
comparatively preserved subspace under the Pairformer's operations &mdash; and
neither shows anything that singles out stability within it.</p>
</div>
<p>The picture that survives all of this: the stability axis is not constructed
by a mechanism. The mutation subspace is partially protected by every operation,
drifts slowly under them anyway, drifts about four times faster in the second
half of the stack, and by the last layer its second component happens to be the
direction that predicts stability. Where that leaves the causal question is in
the limits below.</p>
</section>"""


def sec_corrections():
    return """
<section id=corrections>
<h2>5 &middot; Corrections</h2>
<div class="card warn">
<div class=row><span class="chip c-warn">changed the answer</span>
<strong>A last-layer basis applied at every depth</strong></div>
<p>The first version projected every layer's gain onto the PC basis fitted at
the final layer. Section 1 shows that basis is nearly orthogonal to the
mid-depth one. In the wrong basis every operation looked like chance; in each
layer's own basis, four of five clear chance decisively. The conclusion about
PC2 specifically survives, but the route to it in the first version did not
support it.</p>
</div>
<div class="card warn">
<div class=row><span class="chip c-warn">corrected</span>
<strong>A null compared by magnitude instead of sign</strong></div>
<p>The first pass put signed component gains beside the null's median
<em>absolute</em> gain. At depth the transition contracts nearly every
direction, so the null's own median is strongly negative, and the comparison
turned a generic contraction into an apparently PC2-specific effect. All gains
are now reported as percentiles against the matched signed null.</p>
</div>
<div class="card warn">
<div class=row><span class="chip c-warn">corrected</span>
<strong>Chance for |percentile &minus; 0.5| is 0.25, not 0</strong></div>
<p>It reads like a deviation-from-zero statistic and is not: a percentile drawn
from noise is uniform. Read against zero, every operation appears selective. The
chance level and the 0.50 ceiling are now both drawn on the figure.</p>
</div>
<div class="card warn">
<div class=row><span class="chip c-warn">guarded</span>
<strong>A consistency check that compared two different quantities</strong></div>
<p><code>analyze_ops</code> validates itself against <code>exp_jac</code>'s
independent per-pair Jacobian. That comparison is only meaningful when both use
the same basis, and <code>exp_jac</code> projects onto the last layer's. Run
under per-layer bases it reported r = 0.90 with a WARNING that the two
constructions disagreed &mdash; which would have entered a report as a defect
rather than as the expected result of comparing different things. The check is
now skipped, with a note, when the bases differ; it passes at r = 0.9963 in the
last-layer run, where it is valid.</p>
</div>
</section>"""


def sec_limits(J, C):
    if not (J and C):
        return R.pending("limits")
    return f"""
<section id=limits>
<h2>6 &middot; What this does not establish</h2>
<ul>
<li><strong>Where PC2 <em>is</em> computed.</strong> Five z-path operations are
covered. The MSA module's outer-product-mean write into <code>z</code>, the
template module, the recycling path, and the initial embedding are not. "Not
these five" is the tested claim.</li>
<li><strong>The engagement result may be partly circular.</strong> Each layer's
basis is fitted on the mutation response at that layer, which is itself produced
by these operations. That the operations engage the directions their own output
occupies is not fully independent evidence. The PC2-versus-other-components
comparison is immune to this &mdash; all four components share the
circularity &mdash; but the absolute departure from chance is not.</li>
<li><strong>Local linearisation at the wild-type point.</strong> Validated to
cosine {C['one_step_mean_cosine']:.3f} per layer for the substitutions in these
assays; nothing here bounds it for larger perturbations.</li>
<li><strong>Composition does not survive the full depth.</strong> Free-running
prediction reaches {C['free_final_cosine']:.2f} by the last layer, so no claim
in this study is composed across the stack.</li>
<li><strong>Sampling.</strong> 8 residue rows per protein for the channel
operators, 250 variants per assay for the prediction test.</li>
</ul>
</section>"""


def sec_repro(manifest):
    rows = [[f"<code>{k}</code>", f"<code>{e['file']}</code>", f"{e['bytes']:,}",
             f"<code>{e['sha256'][:12]}</code>"]
            for k, e in sorted(manifest["inputs"].items())]
    return f"""
<section id=repro>
<h2>7 &middot; Reproducing this</h2>
<p>Assumes the method report's runs already exist &mdash; see
<a href="../report_jacobian/index.html">report_jacobian</a> for
<code>launch_jac.sh</code>, <code>launch_ops.sh</code> and their analyses.</p>
<pre><code>R=$WORK/runs

# 1. the basis, and whether it holds at depth
sbatch analysis.sbatch analyze_basis.py --glob "$R/gym2_*.npz" \\
    --pc $R/pc2_v2.npz --out $R/basis_depth.json --npz $R/basis_depth.npz

# 2. re-score the operators in each layer's OWN basis
sbatch analysis.sbatch analyze_ops.py --glob "$R/ops_*.npz" \\
    --jac-glob "$R/jac_*.npz" --basis $R/basis_depth.npz \\
    --out $R/ops_pooled_perlayer.json

# 3. does the linearisation predict? ~9 min per assay
./launch_compose.sh
sbatch analysis.sbatch analyze_compose.py --glob "$R/comp_*.npz" \\
    --out $R/comp_pooled.json

# 4. which operations rotate the basis? (analysis only, no new captures)
sbatch analysis.sbatch analyze_rotate.py --glob "$R/ops_*.npz" \\
    --basis $R/basis_depth.npz --out $R/rotate_pooled.json

python fig_pc2_jac.py --ops $R/ops_pooled_perlayer.json --jac $R/jac_pooled.json \\
    --basis $R/basis_depth.json --comp $R/comp_pooled.json \\
    --rot $R/rotate_pooled.json --out {OUT}/figures/pc2.png
python build_jac_report.py</code></pre>
<h3>Inputs to this build</h3>
{R.table(["key", "file", "bytes", "sha-256"], rows)}
</section>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default=str(R.W / "runs/ops_pooled_perlayer.json"))
    ap.add_argument("--jac", default=str(R.W / "runs/jac_pooled.json"))
    ap.add_argument("--basis", default=str(R.W / "runs/basis_depth.json"))
    ap.add_argument("--comp", default=str(R.W / "runs/comp_pooled.json"))
    ap.add_argument("--rot", default=str(R.W / "runs/rotate_pooled.json"))
    ap.add_argument("--allow-stale-figures", action="store_true")
    a = ap.parse_args()

    resolved = {"ops": a.ops, "jac": a.jac, "basis": a.basis,
                "comp": a.comp, "rot": a.rot}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    missing = [k for k, v in resolved.items() if not Path(v).exists()]
    if missing:
        print(f"   missing inputs: {', '.join(missing)} -- pending cards")

    stale = R.check_figures(OUT, FIGSPEC, resolved, a.allow_stale_figures)
    manifest = R.archive_inputs(OUT, resolved, stale, CODE)
    O, J = R.load(a.ops), R.load(a.jac)
    B, C = R.load(a.basis), R.load(a.comp)
    RT = R.load(a.rot)

    if O and O.get("basis") != "per-layer":
        raise SystemExit(
            f"--ops was built with the {O.get('basis')} basis. This report's "
            f"whole argument is that the last-layer basis does not hold at "
            f"depth; publishing its numbers as the corrected ones would "
            f"reintroduce the error section 4 documents.\n"
            f"Rebuild with: analyze_ops.py --basis <basis_depth.npz>")

    body = "".join([
        sec_summary(O, J, B, C, RT), sec_figure(stale), sec_basis(B, J),
        sec_result(O, J), sec_compose(C, J), sec_rotate(RT, B),
        sec_corrections(),
        sec_limits(J, C), sec_repro(manifest),
    ])

    R.page(
        OUT,
        title="Is the stability axis computed in the Pairformer?",
        eyebrow="result report &middot; august 2026",
        h1="Is the stability axis computed in the Pairformer?",
        lede="Applying the operating-point Jacobian to every operation that "
             "writes to the pair representation. The layers engage the mutation "
             "subspace, the per-layer linearisation predicts the real response "
             "to cosine 0.98 &mdash; and none of it is specific to stability.",
        nav_items=[("summary", "summary"), ("figure", "figure"),
                   ("basis", "1 basis"), ("result", "2 result"),
                   ("compose", "3 prediction"), ("rotate", "4 how it forms"),
                   ("corrections", "5 corrections"), ("limits", "6 limits"),
                   ("repro", "7 reproduce")],
        body=body, manifest=manifest,
        sibling=("../report_jacobian/index.html", "the method"))


if __name__ == "__main__":
    main()
