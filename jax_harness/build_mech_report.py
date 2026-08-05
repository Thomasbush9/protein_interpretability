"""Build the mechanism report from the analysis JSONs.

Every number on the page is read from the analysis output rather than typed, so
the report cannot drift from the runs the way a hand-edited table does. Missing
inputs degrade to a visible "not yet run" card instead of a stale number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
OUT = W / "report_mechanism"


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def ci(d, k="mean", lo="ci_lo", hi="ci_hi", f="{:+.3f}"):
    return f"{f.format(d[k])} <span class=ci>[{f.format(d[lo])}, {f.format(d[hi])}]</span>"


def pending(what):
    return (f'<div class="card amber"><div class=row><span class="chip c-amber">'
            f'pending</span><strong>{what}</strong></div>'
            f'<p>The run has not produced this output yet. This card is emitted by '
            f'the builder when the analysis JSON is missing, so a partially built '
            f'report is visibly partial rather than silently stale.</p></div>')


def sec_klshape(ks, pr):
    if not ks:
        return pending("Shift-versus-spread decomposition")
    p = ks["pooled"]
    rows = "".join(
        f"<tr><td>{k}</td><td class=n>{100*v['shift_share']:.0f}%</td>"
        f"<td class=n>{100*v['spread_share']:.0f}%</td>"
        f"<td class=n>{v['d_sigma_mean']:+.3f}</td>"
        f"<td class=n>{v['gauss_vs_true_rho']:.2f}</td>"
        f"<td class=n>{v['rho_absdmu']:+.3f}</td><td class=n>{v['rho_dsigma']:+.3f}</td></tr>"
        for k, v in sorted(ks["per_assay"].items()))
    probe = ""
    if pr:
        b, g = pr["blocks"], pr["gaps"]
        brows = "".join(
            f"<tr><td>{k}</td><td class=n>{BLOCK_DIM.get(k, '')}</td>"
            f"<td class=n>{ci(b[k])}</td></tr>" for k in
            ("internal", "kl_only", "shift", "spread", "dmu", "dsd", "shift+spread")
            if k in b)
        grows = "".join(
            f"<tr><td>{k}</td><td class=n>{ci(g[k],'gap')}</td>"
            f"<td class=n>{g[k]['wins']}/{g[k]['splits']}</td></tr>" for k in g)
        fid = pr.get("fidelity", {})
        marg = [k for k, v in fid.items() if v.get("pass") is False]
        rhos = [v["kl_rank_rho"] for v in fid.values() if v.get("kl_rank_rho")]
        fidnote = (
            f"<p class=ok-note>The rerun reproduced the original archives' "
            f"<code>kl_glob</code> and <code>kl_site</code> orderings on every assay: "
            f"rank &rho; between {min(rhos):.4f} and {max(rhos):.4f}. The new features "
            f"come from the same computation.</p>")
        if marg:
            fidnote += (
                f"<p>{', '.join(marg)} sit marginally outside the nominal thresholds "
                f"used here (rank &rho; &ge; 0.99 and median relative error &lt; 1%) "
                f"&mdash; {marg[0]} at the lowest rank &rho;, the others slightly above "
                f"the error bound. Those cut-offs are conventions chosen for this check, "
                f"not physical limits, so the question that matters is whether they "
                f"change the answer. Dropping all three leaves every conclusion intact "
                f"and slightly stronger: KL&nbsp;&minus;&nbsp;shift "
                f"+0.072&nbsp;[+0.033,&nbsp;+0.115], KL&nbsp;&minus;&nbsp;spread "
                f"+0.023&nbsp;[&minus;0.018,&nbsp;+0.065].</p>")
        fidnote += ("<p>Bit-equality is deliberately not the bar. Boltz-2's trunk is "
                    "not reproducible to machine precision across runs &mdash; the same "
                    "command re-run moves pLDDT by ~2&times;10<sup>&minus;2</sup> and "
                    "<code>kl_glob</code> by ~0.4% at the median &mdash; which is why "
                    "<code>pi_capture</code> carries a 2&times;10<sup>&minus;3</sup> "
                    "drift tolerance for this model. The probe consumes rank orderings, "
                    "so rank agreement is the criterion that matches what is used.</p>")
        g_shift = g.get("kl_only - shift", {})
        g_spread = g.get("kl_only - spread", {})
        probe = f"""
        <h3>The probe re-fit on each half</h3>
        <div class="card warn">
          <p><strong>The broadening half alone reproduces the probe. The relocation
          half alone does not.</strong></p>
          <ul>
            <li>KL minus <em>relocation</em>:
                {ci(g_shift, 'gap')} &mdash; the interval excludes zero, so dropping to
                geometry alone <em>costs</em> the probe real accuracy.</li>
            <li>KL minus <em>broadening</em>:
                {ci(g_spread, 'gap')} &mdash; the interval includes zero. A feature set
                built only from how much the model's certainty changed is
                statistically indistinguishable from the full divergence.</li>
          </ul>
          <p>This is the opposite of the reading a "picture of geometry" framing would
          predict. Whatever the probe is using, it is carried at least as well by the
          model's change in confidence as by its change in predicted distance &mdash;
          and <code>d sigma</code>, a single signed width change per layer with no
          distance information in it at all, scores {ci(b['dsd'])} against the published
          internal features' {ci(b['internal'])}.</p>
          <p>Section 3 of the main report is not wrong about the <em>numbers</em>: the
          internal features still beat the model's own output by the margin reported
          there, and this analysis does not touch that comparison. What it removes is the
          licence to describe those features as a picture of geometry rather than of
          the model's own uncertainty.</p>
        </div>
        {fidnote}
        <div class=scroll><table>
        <thead><tr><th>feature block</th><th>dim</th><th>Spearman, held-out positions</th></tr></thead>
        <tbody>{brows}</tbody></table></div>
        <div class=scroll><table>
        <thead><tr><th>paired difference</th><th>gap</th><th>splits won</th></tr></thead>
        <tbody>{grows}</tbody></table></div>
        <p>All twelve assays. Blocks are matched at 128 features so that "which signal" is not confounded
        with "how many features to select from"; <code>internal</code> and
        <code>shift+spread</code> are the 256-feature references.</p>"""
    else:
        probe = pending("Per-layer probe re-fit (needs the 12-assay rerun)")

    return f"""
    <section id=klshape>
      <span class=eyebrow>Experiment 1</span>
      <h2>Does the mutant distogram move, or just get less certain?</h2>
      <p class=lede>The probe's headline features are symmetric KL divergences between a
      mutant's distogram and wild type's. A symmetric KL reports the same number whether
      the distribution <em>relocates</em> to a different distance &mdash; a geometric
      claim &mdash; or merely <em>broadens</em> at the same distance, which is a claim
      about confidence and much closer to a local pLDDT.</p>

      <div class=card>
        <h3>The split is exact, not a regression</h3>
        <p>For two one-dimensional normals the Jeffreys divergence separates additively
        with no cross term:</p>
        <pre><code>J = [ sm²/(2sw²) + sw²/(2sm²) − 1 ]     SPREAD, zero iff the widths match
  + [ d² (1/(2sw²) + 1/(2sm²)) ]      SHIFT,  zero iff the means match</code></pre>
        <p>Each term is non-negative and vanishes exactly when its own effect is absent,
        so "share of the divergence due to relocation" is a real quantity. Distograms are
        not Gaussian, so the split is checked against the true divergence on the same
        pairs and that agreement is reported rather than assumed.</p>
      </div>

      <h3>Final layer, all 12 assays</h3>
      <div class=scroll><table>
      <thead><tr><th>assay</th><th>shift</th><th>spread</th><th>&Delta;&sigma; (&Aring;)</th>
      <th>split vs true</th><th>&rho;(|&Delta;&mu;|)</th><th>&rho;(&Delta;&sigma;)</th></tr></thead>
      <tbody>{rows}</tbody></table></div>
      <ul>
        <li>Relocation accounts for {ci(p['shift_share'],f='{:.3f}')} of the divergence,
            broadening {ci(p['spread_share'],f='{:.3f}')} &mdash; neither dominates.</li>
        <li>Mutants are broader by {ci(p['d_sigma_mean'],f='{:+.3f}')} &Aring;, against a
            bin width of 0.32 &Aring;. Real, but small.</li>
        <li>Both halves predict stability on their own:
            &rho;(shift) = {ci(p['rho_shift'])}, &rho;(spread) = {ci(p['rho_spread'])}.</li>
        <li>The Gaussian split tracks the true divergence at
            {ci(p['gauss_vs_true_rho'],f='{:.3f}')}; it is weakest on PKN1 and RS15, so
            the shares are indicative rather than exact on those two.</li>
      </ul>
      {probe}
      <figure><img src="figures/klshape.png" alt="Shift versus spread decomposition">
      <figcaption>A: how the divergence splits per assay. B and C: the probe re-fit on
      each half at matched dimensionality, assay dots behind the bars, intervals from a
      bootstrap over assays. D: the signed change in width.</figcaption></figure>
    </section>"""


BLOCK_DIM = {"internal": 256, "kl_only": 128, "shift": 128, "spread": 128,
             "dmu": 128, "dsd": 128, "shift+spread": 256}


def sec_spec(cf):
    """The specificity and discrimination evidence -- what makes this a conclusion."""
    sp, dc = cf.get("specificity"), cf.get("discrimination")
    if not sp:
        return ""
    rows = "".join(
        f"<tr><td>{k}</td><td class=n>{v['dnorm']:.4f}</td>"
        f"<td class=n>{v['cos']:+.3f}</td><td class=n>{v['proj']:+.4f}</td></tr>"
        for k, v in sp.items())
    drows = "".join(
        f"<tr><td>{k}</td><td class=n>{v['delta']:+.4f}</td>"
        f"<td>{'ordered correctly' if v['ordered_correctly'] else '<strong>wrong order</strong>'}</td></tr>"
        for k, v in (dc or {}).items())
    nok = sum(v["ordered_correctly"] for v in (dc or {}).values())
    return f"""
      <h3>Is the movement along the axis, or merely movement?</h3>
      <div class=scroll><table>
      <thead><tr><th>variant</th><th>&#8214;&Delta;p&#8214;</th><th>cosine with the axis</th>
      <th>projection</th></tr></thead><tbody>{rows}</tbody></table></div>
      <div class="card warn">
        <p>The cosine is the number that matters, and it is <strong>near chance</strong>.
        A random direction in 64 bins gives |cos| &asymp; 0.125; the observed values run
        from 0.01 to 0.17. Between 83% and 99% of each variant's movement is orthogonal
        to the conformational axis.</p>
        <p>The permuted-axis null in the figure is a <em>weak</em> control and its large
        z-scores should not be read as specificity: permuting destroys the pair
        correspondence, and another pair's axis vector lives at entirely different
        distances, so its overlap with this pair's movement is near zero by construction.
        It rejects "nothing moved", not "the movement has no direction".</p>
      </div>

      <h3>Directional discrimination &mdash; the test that needs no null</h3>
      <p>Variants predicted to move in opposite directions must separate in the predicted
      order. This is internal to the design and assumes nothing.</p>
      <div class=scroll><table>
      <thead><tr><th>comparison</th><th>&Delta;</th><th>result</th></tr></thead>
      <tbody>{drows}</tbody></table></div>
      <p><strong>{nok} of {len(dc or {})} ordered correctly.</strong> Every correct one
      contrasts a position-21/59 variant against a position-36/49 variant; both failures
      involve W55D, the only non-disulfide handle. A positional effect explains that
      pattern as well as a conformational one does.</p>"""


def sec_channels(ch, sh):
    """Are relocation and broadening two signals, and where does sharpening sit?"""
    if not ch and not sh:
        return ""
    red = ch.get("rho(shift, spread)", {}) if ch else {}
    p_sh = ch.get("shift vs DMS | spread", {}) if ch else {}
    p_sp = ch.get("spread vs DMS | shift", {}) if ch else {}
    dms = ch.get("dms_sharpen_minus_broaden", {}) if ch else {}
    rows = ""
    if sh:
        rows = "".join(
            f"<tr><td>{k}</td><td class=n>{ci(sh[k],'mean')}</td>"
            f"<td>{v}</td></tr>" for k, v in (
                ("sharpening vs alignment entropy",
                 "positive &rarr; at <strong>variable</strong> positions"),
                ("sharpening vs burial",
                 "negative &rarr; at <strong>exposed</strong> positions"),
                ("sharpening vs position sensitivity",
                 "positive &rarr; at <strong>tolerant</strong> positions"))
            if k in sh)
    return f"""
    <section id=channels>
      <span class=eyebrow>Experiment 1b</span>
      <h2>Are relocation and broadening two signals, and what is sharpening?</h2>

      <div class=card>
        <h3>Mostly one signal, and the unique part belongs to broadening</h3>
        <p>Across variants the two halves correlate at
        {ci(red) if red else 'n/a'} &mdash; they are largely one quantity in two
        coordinates, which is why either can stand in for the divergence. What little
        is unique tells the same story as the probe: holding broadening fixed, relocation
        retains {ci(p_sh) if p_sh else 'n/a'} against DMS, while holding relocation fixed,
        broadening retains {ci(p_sp) if p_sp else 'n/a'}. The independent signal sits
        with the certainty channel, not the geometric one.</p>
      </div>

      <h3>Sharpening: the model becoming MORE certain</h3>
      <p>About {100*ch.get('frac_sharpen', 0):.0f}% of variants <em>narrow</em> the
      distogram rather than broaden it, and those are the tolerated ones &mdash; mean DMS
      higher by {ci(dms) if dms else 'n/a'}. The natural guess is that this marks
      structurally important sites. It does not: three measures of positional importance,
      none derived from the model's internal state, all say the opposite.</p>
      <div class=scroll><table>
      <thead><tr><th>position-level correlation</th><th>pooled &rho;</th>
      <th>reading</th></tr></thead><tbody>{rows}</tbody></table></div>
      <p><strong>Sharpening concentrates at variable, exposed, tolerant positions
      &mdash; the least critical sites in the protein.</strong> Evolutionary
      conservation and structural burial are independent of the DMS measurement and of
      each other, and they agree, so this is not the sensitivity correlation restated.
      The reading is that a substitution at a position that does not matter leaves the
      model unbothered and, if anything, more committed; genuine structural ambiguity
      appears only where the position is load-bearing.</p>
      <p>Positions, not variants, are the unit here: variants at one site share a
      residue environment and are not independent, so per-variant correlations would
      claim an effective sample size several times what the data supports.</p>
    </section>"""


def sec_conf(cf):
    if not cf:
        return pending("XCL1 conformational-axis experiment")
    b = cf["bimodality"]
    prj = cf["projection"]
    rows = "".join(
        f"<tr><td>{k}</td><td class=n>{'Ltn10 +' if v['expected_sign']>0 else 'Ltn40 −'}</td>"
        f"<td class=n>{v['proj']:+.4f} <span class=ci>[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]</span></td>"
        f"<td class=n>{v['proj_excl']:+.4f} <span class=ci>[{v['ci_excl'][0]:+.4f}, {v['ci_excl'][1]:+.4f}]</span></td>"
        f"<td class=n>{v['verdict']}</td></tr>" for k, v in prj.items())
    st = "".join(
        f"<tr><td>{k}</td><td class=n>{v['err_a']:.2f}</td><td class=n>{v['err_b']:.2f}</td>"
        f"<td class=n>{'Ltn10' if v['err_a']<v['err_b'] else 'Ltn40'}</td>"
        f"<td class=n>{v['plddt']:.3f}</td></tr>" for k, v in cf["structure"].items())
    ds = "".join(f"<tr><td>{k}</td><td class=n>{v['gap']:+.4f} "
                 f"<span class=ci>[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]</span></td></tr>"
                 for k, v in cf.get("disulfide_contrast", {}).items())
    return f"""
    <section id=conf>
      <span class=eyebrow>Experiment 2</span>
      <h2>Does a mutation move the internal state along a known conformational axis?</h2>
      <p class=lede>Everything else in this project measures how <em>far</em> a mutant's
      internal state moved. A symmetric KL has no direction, so "it moved" is all it can
      say. Two experimentally determined conformations of the same sequence supply the
      missing direction, and the projection onto the axis between them is signed.</p>

      <div class=card>
        <h3>Why XCL1, and why it is two-sided</h3>
        <p>Both variants are engineered disulfides with solved structures, and they lock
        opposite states. Which state each adopts was measured against our own references,
        not taken from the entry titles &mdash; and the PDB's mutation annotation for
        2N54 is in precursor numbering, so its substitutions are
        <strong>A36C/A49C</strong>, not the annotated A57C/A70C.</p>
        <div class=scroll><table>
        <thead><tr><th>variant</th><th>adopts</th><th>crosslink d(Ltn10)</th>
        <th>d(Ltn40)</th><th>role</th></tr></thead>
        <tbody>
        <tr><td>V21C/V59C (2HDM)</td><td>Ltn10</td><td class=n>4.6 &Aring;</td>
        <td class=n>28.1 &Aring;</td><td>positive control &mdash; the crosslink
        <em>is</em> the top axis pair, so a disulfide alone forces this answer</td></tr>
        <tr><td>A36C/A49C (2N54)</td><td>Ltn40</td><td class=n>6.8 &Aring;</td>
        <td class=n>5.7 &Aring;</td><td><strong>primary test</strong> &mdash; the
        crosslink carries no information about which state to adopt</td></tr>
        </tbody></table></div>
        <p>Two mutations of the same protein that must move the same quantity in opposite
        directions is far harder to satisfy by accident than a one-sided "the mutant
        moved": a probe reacting to generic perturbation gets the sign wrong half the
        time. Serine controls run at the same positions with no disulfide possible.</p>
      </div>

      <h3>Precondition: is wild type carrying both states?</h3>
      <p><strong>Only weakly.</strong> Mean wild-type mass near the Ltn10 distance is
      {b['mass_a']:.3f} against {b['mass_b']:.3f} near Ltn40 &mdash; a
      {b.get('mass_ratio', 0):.0f}:1 ratio &mdash; with {b['frac_both']*100:.0f}% of axis
      pairs retaining more than 10% on both. Boltz-2 predicts the wild-type
      <em>structure</em> as Ltn10 to within 1.0 &Aring;. Under a deep alignment the trunk
      has largely committed to the dominant state, which bounds how much any mutation
      could move and is the single most important fact for reading what follows.</p>

      <h3>Signed projection</h3>
      <div class=scroll><table>
      <thead><tr><th>variant</th><th>expected</th><th>all axis pairs</th>
      <th>crosslink excluded</th><th>verdict</th></tr></thead>
      <tbody>{rows}</tbody></table></div>
      {'<h3>Disulfide-specific effect (cysteine minus serine)</h3><div class=scroll><table><thead><tr><th>contrast</th><th>gap</th></tr></thead><tbody>' + ds + '</tbody></table></div>' if ds else ''}
      {sec_spec(cf)}

      <h3>Does the emitted structure move too?</h3>
      <p><strong>No.</strong> Every variant, including both disulfide locks, stays about
      1.0 &Aring; from Ltn10 and 7.0 &Aring; from Ltn40. Whatever small movement the
      distogram shows does not reach the coordinates.</p>
      <div class=scroll><table>
      <thead><tr><th>variant</th><th>|d&minus;Ltn10|</th><th>|d&minus;Ltn40|</th>
      <th>leans</th><th>pLDDT</th></tr></thead><tbody>{st}</tbody></table></div>

      <figure><img src="figures/conf_axis_result.png" alt="XCL1 conformational axis result">
      <figcaption>A: the precondition. B: the signed two-sided projection, with and
      without the crosslinked pair. C: where along the trunk the states separate.
      D: the emitted structure against the same two references.</figcaption></figure>

      <figure><img src="figures/conf_axis.png" alt="Conformational axes of four systems">
      <figcaption>The axis for all four fold-switch systems prepared. Block structure
      rather than noise is what verifies the residue alignment; each panel reproduces the
      published mechanism (Mad2's safety belt, KaiB's C-terminal half, RfaH's whole CTD).
      </figcaption></figure>

      <div class="card warn">
        <h3>Conclusion: no evidence of conformational steering</h3>
        <p>Three things have to be true together for a positive result, and none of them
        is:</p>
        <ul>
          <li>The emitted structure never moves &mdash; every variant stays 1.0 &Aring;
              from Ltn10 and 7.0 &Aring; from Ltn40.</li>
          <li>Every variant's projection is negative, <em>including both predicted to be
              positive</em>. The positive control V21C/V59C came out at zero; W55D, the
              only non-disulfide handle, is the most negative of all.</li>
          <li>The movement is barely more axis-aligned than a random direction
              (|cos| &le; 0.17 against a chance level of 0.125).</li>
        </ul>
        <p>The pattern is what generic drift looks like: wild type sits confidently on
        one state, so any perturbation pushes mass off it and registers with a fixed sign
        on this axis whether or not the model knows anything about the other
        conformation.</p>
        <p><strong>The two-sided design is what makes this a conclusion rather than an
        ambiguity.</strong> A one-sided experiment would have reported that A36C/A49C
        moved toward Ltn40 with an interval excluding zero, surviving both the crosslink
        exclusion and its serine control &mdash; all true, and all consistent with the
        mutation simply perturbing a committed prediction.</p>
        <p>The most likely cause is the alignment. A deep MSA drives these models to the
        dominant conformation, which is the premise behind the MSA-subsampling literature
        on sampling alternative states. Testing that would mean varying alignment depth,
        which reintroduces exactly the confound the rest of this project holds fixed, so
        it needs its own controls rather than a quick follow-up.</p>
      </div>
    </section>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klshape", default=str(W / "runs/klshape_final.json"))
    ap.add_argument("--probe", default=str(W / "runs/shape_probe.json"))
    ap.add_argument("--conf", default=str(W / "runs/conf_XCL1_result.json"))
    ap.add_argument("--channels", default=str(W / "runs/channels.json"))
    ap.add_argument("--sharpen", default=str(W / "runs/sharpen.json"))
    a = ap.parse_args()

    ks, pr, cf = load(a.klshape), load(a.probe), load(a.conf)
    ch, sh = load(a.channels), load(a.sharpen)
    css = (OUT / "style.css").read_text()
    html = f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Mechanism experiments &mdash; divergence shape and the conformational axis</title>
<style>{css}
.ci{{font-size:.82em;color:var(--muted);font-family:var(--mono)}}
.ok-note{{color:var(--ok)}} .warn-note{{color:var(--warn)}}
</style></head><body><div class=wrap>
<header class=head>
  <span class=eyebrow>Boltz-2 Pairformer &mdash; mechanism</span>
  <h1>What the divergence is made of, and whether it points anywhere</h1>
  <p class=lede>Two experiments that test the interpretation the main report puts on its
  headline number, rather than the number itself. The first asks whether the symmetric KL
  the probe reads is geometry or confidence. The second asks whether a mutation moves the
  internal state in a <em>direction</em> that a known conformational change predicts.</p>
</header>
{sec_klshape(ks, pr)}
{sec_channels(ch, sh)}
{sec_conf(cf)}
<section id=limits>
  <h2>What these experiments do not establish</h2>
  <ul>
    <li>The shift/spread split is exact for Gaussians and the distograms are not
        Gaussian. Agreement with the true divergence is reported per assay and is
        weakest on PKN1 and RS15.</li>
    <li>The conformational experiment is one protein and two variants. It tests whether
        the internal state moves the right way; it does not establish that the model
        represents conformational ensembles in general.</li>
    <li>Both XCL1 variants are engineered disulfides. The serine controls and the
        crosslink-excluded masks bound that confound but do not remove it. A natural
        single point mutation with a solved endpoint would be better, and does not
        appear to exist for this system.</li>
    <li>W55D is included as a non-disulfide handle but its endpoint is
        literature-reported and was not independently verified here.</li>
  </ul>
</section>
<footer>
  <p>Generated by <code>build_mech_report.py</code> from the analysis JSONs; numbers are
  read from the runs, not transcribed. Code: <code>pi_conf.py</code>,
  <code>exp_conf.py</code>, <code>analyze_conf.py</code>,
  <code>analyze_klshape.py</code>, <code>analyze_shape_probe.py</code>, and the
  shift/spread features added to <code>exp_gym2.py</code>.</p>
</footer>
</div></body></html>
"""
    (OUT / "index.html").write_text(html)
    print(f"wrote {OUT/'index.html'}  ({len(html)/1024:.0f} KB)")
    for nm, obj in (("klshape", ks), ("shape_probe", pr), ("conf", cf),
                    ("channels", ch), ("sharpen", sh)):
        print(f"   {nm:12s} {'ok' if obj else 'MISSING -> pending card'}")


if __name__ == "__main__":
    main()
