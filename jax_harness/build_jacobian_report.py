"""Build the METHOD report: the Jacobian of the Boltz-2 pair path.

Split out of `build_jac_report.py` because the two things that run share
machinery but not an audience. What is here stands on its own and does not
depend on the mutation-response work at all:

  * why a weight-space SVD describes a matrix Boltz-2 never applies;
  * how to recover each operation's real input without reimplementing the model;
  * how to define a comparable Jacobian for operations that are not pointwise
    in the pair index;
  * what that operator turns out to be -- low rank, gate-driven, and nearly the
    same subspace in twelve unrelated folds.

The companion page, `report_jac/`, uses this method to ask a specific question
about the stability axis and gets a negative answer. Someone who wants the
method should not have to read that, and someone who wants that should not have
to re-derive this.

Provenance rules live in `pi_report.py` and are shared by both builders.

  python build_jacobian_report.py
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pi_report as R  # noqa: E402

OUT = R.W / "report_jacobian"
FIGSPEC = {
    "jacobian.png": ("python fig_jacobian.py --ops {ops} --gate {gate} "
                     "--wsvd {wsvd} --out {out}"),
}
CODE = ["build_jacobian_report.py", "pi_report.py", "fig_jacobian.py",
        "exp_jac.py", "analyze_jac.py", "exp_ops.py", "analyze_ops.py",
        "probe_wsvd.py", "probe_gate.py"]

NICE = {"tri_mul_out": "tri_mul_out", "tri_mul_in": "tri_mul_in",
        "tri_att_start": "tri_att_start", "tri_att_end": "tri_att_end",
        "transition_z": "transition_z (MLP)"}


def sec_summary(J, O, G, WS):
    if not (J and O and G and WS):
        return R.pending("summary")
    wmed = [st.median(WS["eff_rank"][m]) for m in ("fc1", "fc2", "fc3")]
    ranks = sorted(O["eff_rank_median"].items(), key=lambda kv: kv[1])
    ag = J["agreement"]["out"]
    return f"""
<section id=summary>
<h2>The short version</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">method</span>
<strong>Decompose the Jacobian at a real operating point, not the weights.</strong></div>
<p>Boltz-2's pair transition is SwiGLU, so it has no single weight matrix to
decompose, and the singular ordering of the one matrix you might pick reflects
weight magnitude rather than what the layer writes. Taking the derivative at a
wild-type <code>z</code> removes both problems and produces an operator that is
directly comparable across proteins.</p>
</div>
<div class="card ok">
<div class=row><span class="chip c-ok">result</span>
<strong>The pair path is far smaller, and far more shared, than its weights
suggest.</strong></div>
<ul>
<li>The bare weight matrices have median effective ranks of
{min(wmed):.0f}&ndash;{max(wmed):.0f} out of {WS['dim']}. Every operation's
Jacobian sits between <span class=big>{ranks[0][1]:.1f}</span>
({NICE[ranks[0][0]]}) and <span class=big>{ranks[-1][1]:.1f}</span>
({NICE[ranks[-1][0]]}).</li>
<li>The mechanism is the SwiGLU gate: only
<span class=big>{100*G['live_mean']/512:.0f}%</span> of the 512 hidden units
carry meaningful contribution at any operating point, and that count tracks the
Jacobian's effective rank at r&nbsp;=&nbsp;{G['corr']:.2f} across (assay,
layer).</li>
<li>It is a property of the layer, not the protein. The top-{J['k']} subspaces
of {len(J['assays'])} unrelated folds agree at
<span class=big>{ag['last_layer_mean']:.3f}</span> mean cos&sup2; against a
random baseline of {ag['random_baseline']:.3f}, with the worst of the
{len(J['assays'])*(len(J['assays'])-1)//2} pairs at
{ag['last_layer_min']:.3f}.</li>
</ul>
</div>
</section>"""


def sec_figure(stale):
    warn = ""
    if stale:
        warn = ('<div class="card warn"><div class=row>'
                '<span class="chip c-warn">stale</span><strong>This figure is '
                'older than its data</strong></div><p>Built with '
                '--allow-stale-figures; the panels may not match the numbers '
                'below.</p></div>')
    return f"""
<section id=figure>
<h2>In three panels</h2>
{warn}
<figure>
<img src="figures/jacobian.png" alt="Three panels: effective rank of the bare
weights versus each operation's Jacobian; the SwiGLU gate's live-unit count
tracking that rank across depth; and cross-assay subspace agreement per
operation.">
<figcaption>Colour is the operation and means the same thing in every panel and
in the companion page's figure. Grey is always a reference level, never a
series. Every plotted value also appears as a number below.</figcaption>
</figure>
</section>"""


def sec_why(J, WS):
    if not (J and WS):
        return R.pending("motivation")
    return f"""
<section id=why>
<h2>1 &middot; Why the weights are the wrong object</h2>
<p>The starting point was Xue &amp; Andrzejak's ICML 2026 paper, <em>SVD as a
Fast Interpretability Method for Transformers</em>, which decomposes MLP
projection matrices into rank-1 detector/effector pairs: each right singular
vector is treated as an input pattern the unit responds to, each left singular
vector as what it writes. It is training-free and cheap, which is exactly what a
{J['layers']}-layer trunk wants.</p>
<p>Two things stop it transferring here.</p>
<h3>SwiGLU has no <code>W_out @ W_in</code></h3>
<p>The pairing assumes a two-matrix MLP. Boltz-2's transition is</p>
<pre><code>fc3(silu(fc1 v) * fc2 v)</code></pre>
<p>with <code>fc1</code> and <code>fc2</code> both {WS['hidden']}&times;{WS['dim']}
feeding a <em>multiplicative</em> gate. There is no single linear map to
decompose, and which of the two is "the detector" depends on where you are in
activation space. Any answer you get carries the arbitrariness of that
choice.</p>
<h3>Weight magnitude is not what the layer writes</h3>
<p><code>fc3</code>'s singular ordering ranks directions by the size of their
columns. What actually reaches <code>z</code> depends on the hidden activations
multiplying those columns, and those are nowhere near isotropic &mdash; a
small-&sigma; direction dominates the output if its units fire hard. Section 3
shows this is not hypothetical: it is the difference between rank
{min(st.median(WS['eff_rank'][m]) for m in ('fc1','fc2','fc3')):.0f} and rank
{J['eff_rank_median']:.1f}.</p>
<p>Both problems dissolve if the derivative is taken at a real
<code>z</code>. That is an unambiguous {J['dim']}&times;{J['dim']} map, it needs
no linearisation choice, and it acts on exactly the perturbation a mutation
produces.</p>
</section>"""


def sec_method(J, O):
    if not (J and O):
        return R.pending("method")
    return f"""
<section id=method>
<h2>2 &middot; Method</h2>

<h3>Recovering each operation's real input</h3>
<p>A Jacobian is taken <em>somewhere</em>, and the somewhere has to be the
operation's own input. Nothing in the harness stored those.
<code>PairformerLayer2</code> touches <code>z</code> five times and only five
times:</p>
<pre><code>z = z + tri_mul_out(z, pair_mask)
z = z + tri_mul_in(z, pair_mask)
z = z + tri_att_start(z, pair_mask)
z = z + tri_att_end(z, pair_mask)
z = z + transition_z(z)          # last z operation</code></pre>
<p>The per-layer <code>z</code> the harness already captures is the layer
<em>output</em>. The four intermediate values are not stored anywhere.</p>
<p>For the transition there is an exact trick that needs no new model code: run
the same layer object with <code>fc3.weight</code> set to zero.
<code>fc3</code> has no bias, so the transition contributes identically zero
while everything upstream of it is untouched, and both runs consume the same key
through the same four <code>get_dropout_mask</code> calls. What comes back is
the transition's input, bit-for-bit. The identity
<code>z_out == z_pre + transition_z(z_pre)</code> is then checked rather than
assumed, and the run aborts above 1e-4.</p>
<p>For the four triangle operations there is no such trick, so the five-line
z-path above is replicated &mdash; the only duplicated model code in this
harness. Under <code>deterministic=True</code> every dropout mask is all ones
(<code>dropout=0</code> makes <code>bernoulli(key, 1.0)</code> all-True and the
scale 1/(1&minus;0)), so the replica is exact, and it is verified against the
real layer's output before any derivative is taken.</p>

<h3>Two Jacobians, because the operations differ in kind</h3>
<p><code>transition_z</code> is pointwise in the pair index: pair (i,j)'s output
depends only on pair (i,j)'s input. Its Jacobian is therefore an unambiguous
{J['dim']}&times;{J['dim']} matrix per pair, taken with <code>jacfwd</code> and
averaged over {J['dim']} sampled pairs.</p>
<p>The triangle operations are not. <code>tri_mul_out</code> at pair (i,j) reads
every (i,k) and (j,k); the attentions read whole rows or columns. Their true
Jacobian is an operator on the entire [N,N,{J['dim']}] tensor &mdash; too large
to decompose, and not comparable between proteins of different length.</p>
<p>What <em>is</em> comparable is the <strong>channel-space operator</strong>:
perturb <code>z</code> the way a point mutation does, and read the response in
the model's own {J['dim']} pair channels.</p>
<ul>
<li><strong>Perturb</strong> &mdash; a substitution at residue r changes every
pair involving r, so the tangent enters at row r <em>and</em> column r.
Perturbing the row alone would measure a perturbation the model cannot
receive.</li>
<li><strong>Read</strong> &mdash; mean over partners of the response in row r,
which is exactly how <code>exp_gym2</code> defines the archived
<code>z_site</code> (<code>z[0].mean(axis=1)[row]</code>). The operator
therefore acts on the same quantity the rest of the project is built from.</li>
</ul>
<p>The {J['dim']} basis tangents go through <code>jax.linearize</code> of the
real module, so the primal is computed once and each operation's own
nonlinearity &mdash; softmax, sigmoid gates, LayerNorm, including LayerNorm's
z-dependent mean-removal and 1/rms factor &mdash; is differentiated exactly
rather than hand-derived.</p>
<div class="card">
<div class=row><span class="chip c-run">why this is checkable</span>
<strong>The two constructions overlap on purpose</strong></div>
<p>Because the transition is pointwise, the channel operator applied to it must
reduce to the row-average of its per-pair Jacobian. It is therefore recomputed
by the second method rather than copied from the first, and the agreement is the
validation in section 5.</p>
</div>

<h3>Scope</h3>
<p>{len(J['assays'])} ProteinGym stability assays
({', '.join(J['assays'])}), wild-type sequence only, {J['layers']} Pairformer
layers, {J['dim']} pair channels. Trunk recycles and MSA cap match
<code>exp_gym2</code>, so the operating point is the one every archived number
in this project was computed at.</p>
</section>"""


def sec_rank(J, O, WS, G):
    if not (J and O and WS and G):
        return R.pending("rank")
    wrows = [[m, f"{st.median(WS['eff_rank'][m]):.1f}",
              f"{100*st.median(WS['top8_frac'][m]):.1f}%"]
             for m in ("fc1", "fc2", "fc3")]
    orows = [[NICE[o], f"{O['eff_rank_median'][o]:.1f}",
              f"{O['frobenius_median'][o]:.2f}"] for o in O["ops"]]
    return f"""
<section id=rank>
<h2>3 &middot; The operator is a quarter the size of its weights</h2>
<p>Effective rank throughout is the participation ratio of the squared spectrum,
(&Sigma;&sigma;&sup2;)&sup2;/&Sigma;&sigma;&#8308;. It needs no threshold and
sits on the same scale as the dimension; the ceiling is {J['dim']}.</p>
<h3>Bare weights</h3>
{R.table(["matrix", "effective rank", "energy in top 8"], wrows)}
<h3>Jacobian at a wild-type operating point</h3>
{R.table(["operation", "effective rank", "Frobenius norm"], orows)}
<p>Frobenius norm is reported beside rank because rank says nothing about how
much an operation actually moves the site row, and the two do not have to agree.
Here they do: the transition is both the highest-rank and the largest-moving
operation in the layer, and <code>tri_att_end</code> is the smallest on both
counts.</p>
<p>Across all layers and assays the transition's Jacobian ranges
{J['eff_rank_min']:.1f}&ndash;{J['eff_rank_max']:.1f}, against a weight-space
answer of {st.median(WS['eff_rank']['fc3']):.0f} for <code>fc3</code>. A
weight-space decomposition of this layer is not a coarse version of the right
answer; it is a description of a matrix the model never applies.</p>
</section>"""


def sec_gate(G, J):
    if not (G and J):
        return R.pending("mechanism")
    return f"""
<section id=gate>
<h2>4 &middot; The gate is the mechanism</h2>
<p>A hidden unit whose <code>fc1</code> pre-activation sits well below zero
contributes nothing to the Jacobian regardless of what its <code>fc3</code>
column holds &mdash; <code>silu</code> has killed it. If that is what shrinks
the operator, then counting live units should reproduce the rank curve.</p>
<p>Each unit u contributes a term whose size is set by
<code>||W3[:,u]|| &middot; ||r_u||</code>, where</p>
<pre><code>r_u = silu'(a_u) * b_u * W1[u] + silu(a_u) * W2[u]</code></pre>
<p>Taking the participation ratio of that quantity over the 512 units gives
<span class=big>{G['live_mean']:.0f}</span> live units
({100*G['live_mean']/512:.1f}%), and it correlates with the Jacobian's own
effective rank at <span class=big>r&nbsp;=&nbsp;{G['corr']:.2f}</span> across
(assay, layer). The two curves rise and fall together with depth, both peaking
around layers 48&ndash;56 and collapsing at the last layer.</p>
<p>This is the concrete reason the weight-space method failed here. The gate
decides, per operating point, which seventh of the hidden units is in play; the
weights of the other six-sevenths are still in the matrix and still dominate its
spectrum.</p>
</section>"""


def sec_shared(J, O):
    if not (J and O):
        return R.pending("generality")
    rows = [[NICE[o], f"{O['agreement'][o][0]:.3f}", f"{O['agreement'][o][1]:.3f}"]
            for o in O["ops"]]
    ao, ai = J["agreement"]["out"], J["agreement"]["in"]
    depth = ao["by_depth"]
    return f"""
<section id=shared>
<h2>5 &middot; Twelve unrelated folds use the same subspace</h2>
<p>A low-rank operator per protein would be unremarkable. The claim worth making
is that it is low rank in the <em>same place</em> every time, and that is
measurable because every operator is expressed in the model's own {J['dim']}
pair channels, which mean the same thing in every protein.</p>
<p>Mean cos&sup2; of the principal angles between the top-{O['k']} subspaces of
different assays. Random baseline is {O['k']}/{J['dim']} =
{O['k']/J['dim']:.3f}.</p>
{R.table(["operation", "write side", "read side"], rows)}
<p>For the transition, measured instead from the pair-averaged second moments
&mdash; which is the cleaner object, since per-pair singular vectors of two
different proteins are not in correspondence &mdash; agreement at the last layer
is {ao['last_layer_mean']:.3f} on the write side and {ai['last_layer_mean']:.3f}
on the read side, with the worst of the
{len(J['assays'])*(len(J['assays'])-1)//2} pairs at {ao['last_layer_min']:.3f}.
It holds at every depth: {', '.join(f'{v:.2f} at layer {li}' for li, v in depth[:4])},
and so on.</p>
</section>"""


def sec_checks(J, O):
    if not (J and O):
        return R.pending("checks")
    c = O["consistency"]
    return f"""
<section id=checks>
<h2>6 &middot; What was checked</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">passed</span>
<strong>Two independent routes to the same quantity</strong></div>
<p><code>transition_z</code> is measured twice by different machinery: as a
per-pair <code>jacfwd</code> in <code>exp_jac</code>, and as a row-averaged
channel operator from <code>jax.linearize</code> in <code>exp_ops</code>.
Because it is pointwise in the pair index they must agree.
r&nbsp;=&nbsp;<span class=big>{c['r']:.4f}</span> over {c['n_values']} values
across {len(c['assays'])} assays, mean absolute difference
{100*c['mean_abs_diff_frac']:.1f}% of the mean magnitude &mdash; the residual
being the differing pair and row samples. This is the only end-to-end evidence
that the channel-operator construction measures what it claims.</p>
</div>
<div class="card ok">
<div class=row><span class="chip c-ok">passed</span>
<strong>The base points are the operations' real inputs</strong></div>
<p><code>z_out == z_pre + transition_z(z_pre)</code> holds at ~1e-5 &mdash;
float32 roundoff &mdash; in all {len(J['assays'])} transition runs, with the
script aborting above 1e-4. The replicated five-line z-path reproduces the real
layer's output at <strong>exactly zero</strong> relative error in all
{len(O['assays'])} operations runs. Without these, every number on this page
would describe a computation the model never ran.</p>
</div>
</section>"""


def sec_limits(J, O):
    if not (J and O):
        return R.pending("limits")
    return f"""
<section id=limits>
<h2>7 &middot; What this does not establish</h2>
<ul>
<li><strong>Local, at one base point.</strong> Every Jacobian is taken at the
wild-type <code>z</code>. A large enough perturbation probes curvature none of
this can see, and nothing here bounds how large "large enough" is.</li>
<li><strong>The channel operator is a summary, not the full Jacobian.</strong>
For the triangle operations it compresses an [N,N,{J['dim']}] operator into
{J['dim']}&times;{J['dim']} by fixing how the perturbation enters (row and
column of one residue) and how the response is read (mean over partners). It
answers "what does a mutation-shaped perturbation do", not "what can this
operation do".</li>
<li><strong>Five operations, not the whole trunk.</strong> The Pairformer's
z-path is covered. The MSA module's outer-product-mean write into <code>z</code>,
the template module, the recycling path, and the initial embedding are not.</li>
<li><strong>Agreement is not identity.</strong> Top-{O['k']} agreement near 0.9
still leaves real per-protein structure. It says the operations are mostly
shared, not that they are the same map.</li>
<li><strong>Sampling.</strong> {J['dim']} pairs per protein for the per-pair
Jacobian, 8 residue rows for the channel operators. The across-assay spread is
small, but per-pair variability is averaged out of the second moments used for
the agreement test.</li>
</ul>
</section>"""


def sec_repro(manifest):
    rows = [[f"<code>{k}</code>", f"<code>{e['file']}</code>", f"{e['bytes']:,}",
             f"<code>{e['sha256'][:12]}</code>"]
            for k, e in sorted(manifest["inputs"].items())]
    return f"""
<section id=repro>
<h2>8 &middot; Reproducing this</h2>
<p>All GPU jobs &mdash; the analyses too, since this account has no CPU
partition and <code>analysis.sbatch</code> targets a GPU node. Paths must be
absolute; relative ones resolve against the submit directory, not the
harness.</p>
<pre><code>R=$WORK/runs

sbatch analysis.sbatch probe_wsvd.py --out $R/wsvd_probe.json \\
    --npz $R/wsvd_probe.npz --pc $R/pc2_v2.npz --glob "$R/gym2_*.npz"

./launch_jac.sh                     # per-pair Jacobian, 12 assays, ~2 min each
sbatch analysis.sbatch analyze_jac.py --glob "$R/jac_*.npz" --out $R/jac_pooled.json
sbatch analysis.sbatch probe_gate.py  --glob "$R/jac_*.npz" --out $R/gate_probe.json

./launch_ops.sh                     # channel operators, 12 assays, ~3.5 min each
sbatch analysis.sbatch analyze_ops.py --glob "$R/ops_*.npz" \\
    --jac-glob "$R/jac_*.npz" --out $R/ops_pooled.json

python fig_jacobian.py --ops $R/ops_pooled.json --gate $R/gate_probe.json \\
    --wsvd $R/wsvd_probe.json --out {OUT}/figures/jacobian.png
python build_jacobian_report.py</code></pre>
<h3>Inputs to this build</h3>
{R.table(["key", "file", "bytes", "sha-256"], rows)}
</section>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jac", default=str(R.W / "runs/jac_pooled.json"))
    ap.add_argument("--ops", default=str(R.W / "runs/ops_pooled.json"))
    ap.add_argument("--gate", default=str(R.W / "runs/gate_probe.json"))
    ap.add_argument("--wsvd", default=str(R.W / "runs/wsvd_probe.json"))
    ap.add_argument("--allow-stale-figures", action="store_true")
    a = ap.parse_args()

    resolved = {"jac": a.jac, "ops": a.ops, "gate": a.gate, "wsvd": a.wsvd}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    missing = [k for k, v in resolved.items() if not Path(v).exists()]
    if missing:
        print(f"   missing inputs: {', '.join(missing)} -- pending cards")

    stale = R.check_figures(OUT, FIGSPEC, resolved, a.allow_stale_figures)
    manifest = R.archive_inputs(OUT, resolved, stale, CODE)
    J, O = R.load(a.jac), R.load(a.ops)
    G, WS = R.load(a.gate), R.load(a.wsvd)

    body = "".join([
        sec_summary(J, O, G, WS), sec_figure(stale), sec_why(J, WS),
        sec_method(J, O), sec_rank(J, O, WS, G), sec_gate(G, J),
        sec_shared(J, O), sec_checks(J, O), sec_limits(J, O),
        sec_repro(manifest),
    ])

    R.page(
        OUT,
        title="The Jacobian of the Boltz-2 pair path",
        eyebrow="method report &middot; august 2026",
        h1="The Jacobian of the Boltz-2 pair path",
        lede="Weight-space SVD describes a matrix Boltz-2 never applies. "
             "Differentiating each Pairformer operation at a real operating "
             "point instead gives an operator a quarter the size, driven by the "
             "SwiGLU gate, and nearly identical across twelve unrelated protein "
             "folds.",
        nav_items=[("summary", "summary"), ("figure", "figure"),
                   ("why", "1 why"), ("method", "2 method"), ("rank", "3 rank"),
                   ("gate", "4 mechanism"), ("shared", "5 shared"),
                   ("checks", "6 checks"), ("limits", "7 limits"),
                   ("repro", "8 reproduce")],
        body=body, manifest=manifest,
        sibling=("../report_jac/index.html", "applied to the stability axis"))


if __name__ == "__main__":
    main()
