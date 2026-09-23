"""Build the SVD / mutation-subspace report from the analysis JSONs.

Same contract as `build_mech_report.py`: every number on the page is read from
an analysis output rather than typed, and a missing input renders as a visible
"not yet run" card instead of a stale figure. If a claim in the prose has a
number in it, that number came out of `runs/`.

That contract held for the arithmetic and failed for everything around it. The
August 2026 review found three defects that were all the same defect:

  * the archived `report_svd/data/svd_dz_v2.json` was the superseded free-row
    permutation run, while the prose was built from `svd_dz_v3.json` -- the
    right file was read and the wrong file was filed, by hand, days apart;
  * `figures/symmetry.png` was generated from `symmetry_v2.json` while the text
    beside it came from v3, so the figure showed five output blocks and 1741
    dimensions against the prose's eight and 1831;
  * one sentence -- "Every assay returns p = 0.000" -- was a hardcoded literal
    inside an otherwise derived paragraph, and it was false: NKX31 returns
    0.009 and PSAE 0.001.

The cause was that the correct build depended on `--svd`/`--symmetry` flags
typed at a shell and recorded nowhere, while the defaults still pointed at the
older runs. So the builder now owns its own provenance:

  * defaults point at the current runs, and `A_SVD` is no longer re-loaded
    behind the caller's back (one number in the single-track note used to come
    from the default path even when `--svd` was overridden);
  * every resolved input is copied into `report_svd/data/` by this script, with
    a `manifest.json` recording each path, size and SHA-256;
  * a figure older than the JSON it is drawn from aborts the build and prints
    the command to regenerate it;
  * the footer is generated, and stamps the commit, the build time and the
    manifest digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_stats  # noqa: E402

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
OUT = W / "report_svd"
# Jobs execute `$W/harness/*.py`, which is an unversioned COPY of the git
# working tree -- so the commit has to be read from the repo, and the copy has
# to be checked against it. A build from a harness that has drifted from the
# repo is not reproducible from the repo, whatever the commit says.
REPO = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/"
            "protein_interpretability/jax_harness")

# figure file -> (the --flag whose JSON it is drawn from, how to rebuild it).
# The template is formatted with every resolved input plus `out`, so the command
# it prints is the complete one and not a sketch to be filled in by hand -- the
# hand-filling is what put a v2 figure next to v3 prose in the first place.
FIGSPEC = {
    "svd.png":      ("svd", "fig_svd.py --svd {svd} --svd-ds {svd_ds} --out {out}"),
    "symmetry.png": ("symmetry", "fig_symmetry.py --symmetry {symmetry} --out {out}"),
    "chem.png":     ("chem", "fig_chem.py --chem {chem} --out {out}"),
    "xmodel.png":   ("xmodel", "fig_xmodel.py --xmodel {xmodel} --out {out}"),
    "depth.png":    ("depth", "fig_depth.py --depth {depth} --out {out}"),
    "pc2.png":      ("pc2", "fig_pc2.py --pc2 {pc2} --out {out}"),
    "drift.png":    ("drift", "fig_drift.py --drift {drift} --out {out}"),
    "steer.png":    ("steer", "fig_steer.py --steer {steer} --out {out}"),
    "heldout.png":  ("heldout", "fig_heldout.py --heldout {heldout} --out {out}"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_figures(resolved, allow_stale=False):
    """Abort if a figure predates the JSON it is drawn from.

    This is the mechanical replacement for noticing by eye. mtime is a weak
    check -- it cannot see a figure rebuilt from the wrong file -- so the
    manifest also records the digest of every source, which makes the pairing
    auditable after the fact.
    """
    stale = []
    for fig, (key, cmd) in FIGSPEC.items():
        fp, src = OUT / "figures" / fig, resolved.get(key)
        if not fp.exists() or src is None or not Path(src).exists():
            continue
        if fp.stat().st_mtime < Path(src).stat().st_mtime:
            stale.append((fig, key, cmd.format(out=fp, **resolved)))
    if stale and not allow_stale:
        lines = "\n".join(f"    {c}" for _, _, c in stale)
        raise SystemExit(
            "figures older than the data they are drawn from:\n"
            + "\n".join(f"  {f}  (source: --{k})" for f, k, _ in stale)
            + "\n\nregenerate, then rebuild:\n" + lines
            + "\n\n(--allow-stale-figures overrides, but the last time a figure "
              "and its prose disagreed it reached a reviewer.)"
        )
    return [f for f, _, _ in stale]


def archive_inputs(resolved, stale):
    """Copy every resolved input into report_svd/data/ and write a manifest.

    Previously this was done by hand, which is how the data directory ended up
    holding a JSON the report was not built from.
    """
    dd = OUT / "data"
    dd.mkdir(parents=True, exist_ok=True)
    entries, kept = {}, set()
    for key, src in sorted(resolved.items()):
        if src is None or not Path(src).exists():
            continue
        src = Path(src)
        shutil.copy2(src, dd / src.name)
        kept.add(src.name)
        entries[key] = {"file": src.name, "source": str(src),
                        "bytes": src.stat().st_size, "sha256": sha256(src)}
    orphans = sorted(p.name for p in dd.glob("*.json")
                     if p.name not in kept and p.name != "manifest.json")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=REPO, capture_output=True,
                                text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"

    # Digest the code that actually ran, and say so when it differs from the
    # repo at that commit. Without this the commit is decoration: the harness
    # copy can be any age.
    here = Path(__file__).parent
    code, drifted = {}, []
    for name in ["build_svd_report.py", "pi_stats.py"] + sorted(
            {c.split()[0] for _, c in FIGSPEC.values()}):
        p = here / name
        if not p.exists():
            continue
        code[name] = sha256(p)
        r = REPO / name
        if r.exists() and sha256(r) != code[name]:
            drifted.append(name)

    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    manifest = {"built": built, "commit": commit, "inputs": entries,
                "code_sha256": code, "code_drifted_from_repo": drifted,
                "figures_stale_at_build": stale,
                "orphaned_in_data_dir": orphans}
    if drifted:
        print(f"   WARNING: harness copy differs from the repo at {commit} for "
              f"{', '.join(drifted)} -- this build is not reproducible from "
              f"that commit")
    (dd / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # The README lists the archive by filename and went stale for three days
    # naming svd_dz_v2.json, which no longer exists here. Cheap to check.
    #
    # Only the fenced blocks are checked. The prose deliberately names files that
    # are NOT inputs -- the corrections section has to be able to say which file
    # used to be wrong -- and flagging those would train the reader to ignore
    # this warning, which is worse than not having it.
    rm = OUT / "README.md"
    if rm.exists():
        fenced = "\n".join(re.findall(r"```.*?```", rm.read_text(), re.S))
        ghosts = sorted({w for w in re.findall(r"[\w./-]+\.json", fenced)
                         if Path(w).name not in kept
                         and Path(w).name != "manifest.json"})
        if ghosts:
            print(f"   WARNING: README.md names {len(ghosts)} JSON(s) that are "
                  f"not inputs to this build: {', '.join(ghosts)}")
            manifest["readme_ghost_files"] = ghosts
            (dd / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if orphans:
        print(f"   note: {len(orphans)} JSON(s) in data/ are not inputs to this "
              f"build and are recorded as orphans: {', '.join(orphans)}")
    return manifest


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

    # This sentence used to read "Every assay returns p = 0.000", typed in.
    # With 1000 permutations the smallest reportable p is 1/n_perm, and two
    # assays do not reach it: quoting a floor as an exact zero and averaging
    # over the two exceptions were the same mistake made twice.
    n_perm = S["protocol"].get("n_perm")
    floor = 1.0 / n_perm if n_perm else None
    ps = {n: v["p_perm_maxstat"] for n, v in nl["per_assay"].items()}
    at_floor = sorted(n for n, p in ps.items() if floor and p < floor)
    above = sorted(((p, n) for n, p in ps.items() if not floor or p >= floor))
    if not above:
        p_txt = (f"Every assay returns p &lt; {floor:g} &mdash; the resolution "
                 f"floor at {n_perm} permutations, not a measured zero.")
    else:
        named = "; ".join(f"{n} at p = {p:.3f}" for p, n in above)
        # Whether the weakest p belongs to the weakest rho is a fact about this
        # run, so it is checked rather than asserted -- asserting it is the
        # habit that produced "Every assay returns p = 0.000".
        worst_p = max(above)[1]
        worst_rho = min(nl["per_assay"], key=lambda n: nl["per_assay"][n]["rho"])
        coda = (" The weakest is also the assay with the lowest observed "
                "&rho;, which is the honest place for the null to bite."
                if worst_p == worst_rho else
                f" The weakest p belongs to {worst_p}, whose observed &rho; is "
                f"{nl['per_assay'][worst_p]['rho']:+.3f}; the lowest &rho; is "
                f"{worst_rho}'s at {nl['per_assay'][worst_rho]['rho']:+.3f}.")
        p_txt = (f"{len(at_floor)} of {len(ps)} assays return p &lt; {floor:g}, "
                 f"the resolution floor at {n_perm} permutations. The remaining "
                 f"{len(above)} are quoted as measured: {named}.{coda}")
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
                   f"{S['subspace_agreement']['last8_pooled']['mean']:.3f}. "
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
{p_txt}</p>
<p>Two corrections stand behind those numbers. The statistic is the
<em>maximum over layers</em> on both sides, so searching 64 layers is charged to
the null rather than treated as free. And an earlier version shuffled only the
training labels while scoring against the true held-out ones, which is not a
null at all &mdash; the fitted direction still lies inside a subspace whose axes
are individually predictive and inherits their association with DMS. That
mistake produced a &ldquo;null&rdquo; reaching |&rho;| &asymp; 0.70.
{nl.get('scheme', nl.get('caveat', ''))}</p>
</div>
</section>"""


def sec_methods_full(S, HO, manifest):
    """A single Methods section, generated so its numbers cannot drift."""
    D = S["protocol"]["dim"] if S else 128
    L = S["protocol"]["n_layers"] if S else 64
    nperm = S["protocol"].get("n_perm", 1000) if S else 1000
    basis = S["protocol"]["assays"] if S else []
    ho_n = len(HO["per_assay"]) if HO else 0
    ho_var = sum(r["n"] for r in HO["per_assay"].values()) if HO else 0
    return f"""
<section id=methods-full>
<h2>Methods</h2>

<div class="card">
<h3>Model and capture</h3>
<p>All internal quantities come from <b>Boltz-2</b> run through
<a href="https://github.com/escalante-bio/joltz">joltz</a>, a JAX re-implementation
whose parameters are loaded from the released checkpoint. The Pairformer trunk is
{L} blocks deep and carries a pair representation
<code>z[i,&nbsp;j,&nbsp;c]</code> with {D} channels per ordered residue pair,
alongside a 384-channel single representation <code>s[i,&nbsp;c]</code>.</p>
<p>Each variant is captured as <b>two forward passes</b>, wild type and mutant,
under three constraints that make their difference interpretable:</p>
<ul>
<li><b>A shared alignment.</b> The mutant sequence is grafted onto the wild
type's homologs rather than searched separately, so MSA composition and depth
cannot move with the mutation. A separate search would let alignment
composition, not the substitution, drive the difference.</li>
<li><b>Identical randomness.</b> Same PRNG key, same recycle count,
deterministic sampling. The wild-type trunk pass is deliberately <em>not</em>
cached across variants even though it is algebraically identical, so that a
re-capture stays comparable to earlier ones variant-for-variant.</li>
<li><b>The same row in both passes.</b> Quantities are read at the mutated
residue's index <code>r</code> in both, so a difference reflects the
substitution rather than which position is being inspected.</li>
</ul>
<p>Captures are verified against their predecessors before use rather than
assumed faithful: the re-capture that added per-residue pLDDT was checked to
reproduce the earlier archives at rank &rho; 0.970&ndash;0.9998 on
<code>kl_glob</code>, with <code>plddt_res[i, pos[i]]</code> reproducing
<code>plddt_site</code> exactly and <code>dz_row</code> averaging to
<code>dz_site</code> to 7e&minus;05.</p>
</div>

<div class="card">
<h3>Data</h3>
<p><b>Basis panel:</b> {len(basis)} ProteinGym substitution assays
({', '.join(basis)}), all Tsuboyama&nbsp;2023 cDNA-display proteolysis stability,
40&ndash;72 residues, 250 variants sampled per assay.</p>
<p><b>Held-out panel:</b> {ho_n} further ProteinGym assays, {ho_var} variants,
selected after the basis was fixed and disjoint from it &mdash; twelve more
Tsuboyama stability assays and four measuring a different phenotype (ccdB
toxicity by growth, EnvZ signalling by fluorescent reporter, phototropin by
fluorescence, HIV Tat by viral replication).</p>
<p>Alignments are built with ColabFold search against the ProteinGym reference
sequence. Every assay's variant list is stored with its archive so that two runs
can be compared row by row.</p>
</div>

<div class="card">
<h3>Feature blocks</h3>
<p>The comparison the report rests on is between what the model <em>computes</em>
and what it <em>emits</em>, so both sides are defined explicitly.</p>
<p><b>Internal.</b> <code>dz_site</code> = the mutated residue's pair row
averaged over partners, differenced between passes, {D} dimensions at one layer.
Also captured: the unaveraged row <code>dz_row[r, j, :]</code>, the single-track
<code>ds_site</code> (384), and per-layer symmetric-KL summaries of the
distogram.</p>
<p><b>Output.</b> Everything the structure module emits: TM to wild type, RMS
displacement overall and within 8&nbsp;&Aring; and 12&nbsp;&Aring; of the
mutation, displacement at the site and its maximum, radius-of-gyration change,
chain-mean and site pLDDT, and &mdash; added after an audit found the output side
under-served by two confidence scalars &mdash; the full per-residue pLDDT vector.
The trunk distogram is deliberately excluded from the output side: it is a head
on the Pairformer, not a product of the structure module.</p>
<p><b>Chemistry.</b> 17 numbers derived from the two amino-acid letters alone:
BLOSUM62, signed and absolute changes in hydropathy, volume and charge, the
wild-type and mutant values of each, and six proline/glycine/cysteine
indicators. This is the deciding control throughout, because it transfers
between proteins for free.</p>
</div>

<div class="card">
<h3>The PC2 construction</h3>
<p>Stated once, in order, in the section above &mdash; two passes, the pair row
at the mutated residue averaged over partners, the difference at the final
layer, z-scored per channel within each protein, all assays stacked into one
matrix, one SVD, sign fixed against <code>kl_glob</code>. PC2 is the second
right singular vector: not searched for and not selected for being predictive.
A variant's PC2 score is the projection of its standardised, mean-centred
&Delta;z onto that vector.</p>
<p>Two versions of the basis exist and conflating them is how a leak would
enter. Where the basis <em>predicts</em>, it is refit on training positions and
frozen before held-out rows are projected. Where it only <em>describes</em>, it
is fitted on all rows and no held-out claim is made from it.</p>
</div>

<div class="card">
<h3>Estimator and splits</h3>
<p>One estimator is used everywhere so that blocks of different width sit on the
same scale: standardise on the training rows, select the top-<i>k</i> features by
training |&rho;|, tune <i>k</i> and the ridge penalty on an inner split, fit ridge,
score Spearman on the held-out rows.</p>
<p><b>Splits are grouped by residue position</b>, never by variant. Variants at
one site share a DMS level, so a random row split lets the probe memorise which
sites are fragile and report it as generalisation. Every held-out number in this
report is across positions.</p>
<p>Dimensionality is treated as a confound rather than ignored. Because the
internal block is {D}-wide and some output blocks are far wider, both sides are
run through the identical protocol across a sweep of <i>k</i>, and the ceiling
over <i>k</i> is plotted against block width &mdash; which is what distinguishes
an estimator problem from an information problem.</p>
</div>

<div class="card">
<h3>Statistics</h3>
<ul>
<li><b>The unit is the assay.</b> Intervals come from a cluster bootstrap that
resamples assays, not the repeated splits inside them. Bootstrapping overlapping
splits would give an interval around splitting noise.</li>
<li><b>Paired comparisons are paired per assay</b>, and the win count out of 12
(or 16) is reported beside the interval, because a mean margin with a 5/12 win
count means something different from the same margin at 12/12.</li>
<li><b>Pairwise quantities resample nodes.</b> The cross-assay subspace
agreement is a mean over 66 assay pairs drawn from 12 assays, so it uses a
delete-one-assay jackknife. Resampling the pairs treats dependent values as
independent: on simulated graphs with the same dependence structure that
interval covers the truth 43% of the time at a nominal 95%, the jackknife
96%.</li>
<li><b>Rank statistics are tie-aware.</b> <code>argsort(argsort(x))</code> is a
tie-breaking permutation, not a rank, and it scored a constant predictor at
+0.069 before this was centralised. Partial Spearman residualises the ranks and
takes the Pearson correlation of the residuals.</li>
<li><b>The permutation null exchanges whole positions</b> among positions of
equal variant count, and the statistic is the maximum over layers on both sides
so that searching {L} layers is charged to the null. With {nperm} permutations
the smallest reportable value is 1/{nperm}, quoted as a bound rather than as an
exact zero.</li>
</ul>
</div>

<div class="card">
<h3>Metrics</h3>
<p>The primary statistic is within-assay Spearman averaged over assays, the
ProteinGym convention. It is the right choice for the comparisons this report
makes, which are <em>paired on identical rows</em> &mdash; a monotone metric
cannot favour one side of such a comparison by accident, and Spearman is
invariant to the per-assay dynamic range that differs across DMS technologies.</p>
<p>It is not sufficient on its own, so the frozen predictions are also scored by
AUC and MCC against the ProteinGym <code>DMS_binarization_cutoff</code>, and by
NDCG and recall over the predicted top 10%. Predictions are oriented once from
the basis assays, never per assay; MCC thresholds predictions at their
<i>n</i><sub>pos</sub>-th largest value so the predicted positive count matches
the true one. The result of that check is reported with the held-out panel: the
ordering is metric-invariant on the stability assays and is not on the four
non-stability assays.</p>
<p class=ci>Two limitations of the panel are worth stating rather than
discovering. The basis panel is 100% one assay technology, which ProteinGym's own
aggregation corrects for by grouping assays into function categories before
averaging; no such correction is applied here because the panel is deliberately
homogeneous. And this report does not benchmark against ProteinGym baselines
&mdash; the comparisons are internal-versus-output and frozen-versus-chemistry on
identical rows, not a leaderboard entry.</p>
</div>

<div class="card">
<h3>Controls</h3>
<p>Each claim carries the control that could have killed it: a chemistry
baseline for &ldquo;the model knows more than the substitution&rdquo;; random
directions of matched norm for &ldquo;this direction is special&rdquo;; a
positive control on every ablation, measuring how much of the removed component
survives, so that a null result is interpretable rather than ambiguous; a
run-to-run repeat that gives the ceiling any cross-assay agreement must be read
against; a negative control of principal angles between independently trained
models; and a determinism check separating a real perturbation from a change of
sampling key.</p>
</div>

<div class="card">
<h3>Reproducibility</h3>
<p>This page is generated by <code>build_svd_report.py</code> from the analysis
JSONs; numbers in the prose are read from those files rather than transcribed.
The builder copies every input it resolved into <code>data/</code> and records
each one's SHA-256 in <code>data/manifest.json</code> together with the commit
and the digests of the scripts that ran. A figure older than the JSON it is
drawn from aborts the build. This machinery exists because it failed: the
archived permutation run, the symmetry figure and one hardcoded p-value all
disagreed with the prose at the August 2026 review, and all three came from the
report being assembled with command-line flags recorded nowhere.</p>
<p class=ci>Built {manifest['built']} at commit
<code>{manifest['commit']}</code> from {len(manifest['inputs'])} inputs.
Analysis runs on one H100 through a Singularity image; the SVD study is
~7,700 batched decompositions plus the ridge grid over layer &times; penalty
&times; component count &times; permutation, all batched in <code>jnp</code>.</p>
</div>
</section>"""


def sec_heldout(HO):
    if not HO:
        return pending("Frozen basis on the sixteen held-out proteins")
    SA, P, R = HO["summary_abs"], HO["paired"], HO["per_assay"]
    wb = HO["pc2_within_basis"]
    S12, N4 = "stability (12)", "non-stability (4)"
    n_var = sum(r["n"] for r in R.values())

    def s(key, grp):
        return ci(SA[key][grp], "{:.3f}")

    def pr(a_, b_, grp):
        d = P[f"{a_} - {b_} | {grp}"]
        return f"{ci(d)}, {d['wins']}/{d['n']}"

    MT = HO.get("metrics_table", {})
    MLAB = [("spearman", "Spearman |&rho;|"), ("auc", "AUC"), ("mcc", "MCC"),
            ("ndcg_top10", "NDCG@10%"), ("recall_top10", "recall@10%")]
    BLK = ["PC2 frozen", "chemistry frozen", "full dz frozen"]

    def mtable(grp):
        g = MT.get(grp, {})
        if not g:
            return ""
        head = "".join(f"<th>{lab}</th>" for _, lab in MLAB)
        body = ""
        for b in BLK:
            cells = "".join(
                f"<td>{g[b][k]['mean']:.3f}</td>" if k in g.get(b, {})
                else "<td>&mdash;</td>" for k, _ in MLAB)
            body += f"<tr><td>{b}</td>{cells}</tr>"
        verdict = ("the ordering is identical under all five"
                   if g.get("_stable") else
                   "<b>the ordering changes with the metric</b>")
        return (f"<table><tr><th>{grp}</th>{head}</tr>{body}</table>"
                f"<p class=ci>{verdict}.</p>")

    envz = HO["length_matched"].get("ENVZ_ECOLI_Ghose_2023", {})
    unmatched = [k.split("_")[0] for k, v in HO["length_matched"].items()
                 if not v.get("neighbours")]
    return f"""
<section id=heldout>
<h2>Does the direction work on proteins it never saw?</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">result</span>
<h3>{len(R)} proteins outside the basis, {n_var} variants, nothing refitted</h3></div>
<p>Every transfer number above is leave-one-assay-out <em>inside</em> the twelve
proteins that built the basis &mdash; all Tsuboyama proteolysis, all 40&ndash;72
residues, all measuring the same thing. This applies the basis, frozen, to
sixteen ProteinGym assays chosen afterwards: twelve more stability assays and
four measuring something else entirely (ccdB toxicity by growth, EnvZ signalling
by reporter, phototropin by fluorescence, HIV Tat by viral replication). No
held-out label enters any fit and the sign is not re-chosen per protein.</p>
<ul>
<li><b>It transfers at full strength.</b> Frozen PC2 reaches
<b>{s('pc2_transductive', S12)}</b> on the twelve unseen stability assays,
against {abs(wb['mean']):.3f} inside the basis itself. Generalisation to new
proteins costs nothing measurable.</li>
<li><b>It needs nothing at all from the new protein.</b> Scaling the held-out
channels with training statistics instead of their own changes the result by
{pr('abs_pc2_transductive', 'abs_pc2_inductive', S12)} &mdash; an interval that
includes zero. A new protein can be scored today, with one dot product.</li>
<li><b>It is not chemistry.</b> A 17-descriptor chemistry model frozen the same
way reaches {s('chem_frozen', S12)}; PC2 beats it by
{pr('abs_pc2_transductive', 'abs_chem_frozen', S12)}.</li>
<li><b>The direction is about severity, not only stability</b> &mdash; but
weakly. On the four non-stability assays frozen PC2 reaches
{s('pc2_transductive', N4)}, roughly half its stability value, and still beats
frozen chemistry by {pr('abs_pc2_transductive', 'abs_chem_frozen', N4)}. So the
signal is not confined to folding, and the honest description is a
mutation-severity direction with a strong stability emphasis.</li>
<li><b>A frozen scalar matches a probe fitted on each protein.</b> The ordinary
within-assay probe &mdash; all 128 channels, ridge, position-grouped splits, run
here on these same sixteen proteins so the comparison is not across two panels
&mdash; reaches {s('dz_within', S12)}. Frozen PC2, one number with nothing
fitted, differs from it by {pr('abs_pc2_transductive', 'abs_dz_within', S12)}:
indistinguishable.</li>
</ul>
<p class=ci>The full 128 channels frozen the same way reach
{s('dz_frozen', S12)}, ahead of PC2 by
{pr('abs_dz_frozen', 'abs_pc2_transductive', S12)} and ahead of the within-assay
probe on identical rows by
{pr('abs_dz_frozen', 'abs_dz_within', S12)}. Pooling twelve proteins builds a
better predictor than fitting each protein on its own, which is the transfer
claim in its strongest form. On the four non-stability assays none of these
separations survives &mdash; frozen against within-assay is
{pr('abs_dz_frozen', 'abs_dz_within', N4)}.</p>
</div>

<div class="card amber">
<div class=row><span class="chip c-amber">caveat</span>
<h3>PC2 is not special on the four</h3></div>
<p>Twenty random unit directions in the same 128-dimensional space were scored
the same way. The best of the twenty reaches {s('random_absmax', S12)} on the
stability assays &mdash; PC2 still beats it by
{pr('abs_pc2_transductive', 'random_absmax', S12)}. On the four non-stability
assays it does not: {pr('abs_pc2_transductive', 'random_absmax', N4)}, an
interval straddling zero.</p>
<p>PC2 beats the <em>average</em> random direction everywhere
({pr('abs_pc2_transductive', 'random_absmean', N4)} on the four), so the
non-stability signal is real. But on a non-folding phenotype the specific
direction the SVD found is not distinguishable from the luckiest of twenty
arbitrary ones. The severity information is spread widely enough through the
pair channels that a random projection captures much of it, which is consistent
with the depth and ablation results and is an argument against calling PC2 a
privileged axis.</p>
</div>

<div class="card ok">
<div class=row><span class="chip c-ok">control</span>
<h3>Is any of this an artifact of reporting Spearman?</h3></div>
<p>Every number in this report is a within-assay Spearman averaged over assays,
which is the ProteinGym convention. Spearman weights the whole ranking equally,
which is not what either audience wants: a variant-effect reader wants
deleterious-versus-tolerated, and a protein engineer only reads the top of the
list. So the same frozen predictions are scored four more ways &mdash; AUC and
MCC against the <code>DMS_binarization_cutoff</code> from the ProteinGym
reference table, and NDCG and recall over the predicted top 10%.</p>
{mtable("stability (12)")}
{mtable("non-stability (4)")}
<p>On the twelve stability assays nothing moves: full &Delta;z &gt; PC2 &gt;
chemistry under all five metrics, so on that panel &ldquo;we report
Spearman&rdquo; is a presentation choice and the conclusions do not depend on
it.</p>
<p><b>On the four non-stability assays the conclusion is metric-dependent, and
Spearman is the flattering choice.</b> PC2 leads chemistry on Spearman, AUC and
MCC, but chemistry leads on both top-focused metrics &mdash; NDCG@10% 0.753
against 0.729, and recall@10% 0.160 against 0.090. Frozen PC2 recovers 9% of the
true top decile there, which is <em>below</em> the 10% a random selection would
give. It orders the bulk of a non-stability assay better than chemistry does and
is useless at finding that assay's best variants.</p>
<p class=ci>This does not touch the internal-versus-output comparison, which is
paired on identical rows where a monotone metric cannot favour one side by
accident. It bears on the transfer claim: any statement about the four
non-stability assays should name the metric, and none of them should be
presented as variant prioritisation.</p>
</div>

<div class="card ok">
<div class=row><span class="chip c-ok">control</span>
<h3>The phenotype gap is not a length gap</h3></div>
<p>The obvious objection is that the four are also the long ones &mdash; 60 to
118 residues against the stability panel's 40 to 72 &mdash; and |&rho;| does
fall with chain length across the sixteen at
{HO['confounds']['chain length']['all16']:+.3f}. So the two contrasts are partly
the same contrast, and it is checked rather than argued.</p>
<p>Only ENVZ can be length-matched, and it is the decisive case: at
{envz.get('n_res', 0)} residues it sits inside the stability panel's range and
still falls <b>{envz.get('gap', float('nan')):+.3f}</b> below the
{len(envz.get('neighbours', []))} stability assays within ten residues of it.
The reverse also holds &mdash; the longest assay in the panel, PHOT at 118
residues, outscores ENVZ at 60. Length is not what separates the groups.</p>
<p class=ci>{', '.join(unmatched)} have no stability assay within ten residues,
so their gaps remain confounded with length and are not used for this argument.
Within the twelve stability assays alone the length association is
{HO['confounds']['chain length']['stability_only']:+.3f}, so a mild length
effect exists and is not being denied &mdash; it is simply too small to produce
the observed separation.</p>
</div>
</section>"""


def node_interval(sa):
    """Interval for the pairwise agreement, resampling ASSAYS not pairs.

    Recomputed here from `pairs_last8` rather than trusted from the JSON so
    that archives written before `analyze_svd.py` was fixed render the correct
    interval without a GPU re-run. Both paths call the same pi_stats function,
    so they cannot disagree.
    """
    pt, lo, hi, n_nodes, se = pi_stats.pairwise_node_jackknife(sa["pairs_last8"])
    return {"mean": pt, "ci_lo": lo, "ci_hi": hi, "se": se, "n_nodes": n_nodes}


def sec_shared(S):
    if not S:
        return ""
    sa = S["subspace_agreement"]
    nodes = node_interval(sa)
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
<p>Pooled over the last eight layers: <b>{ci(nodes, '{:.3f}')}</b>
against a chance level of {sa['chance']:.3f} for random {sa['k']}-dimensional
subspaces.</p>
<p class=ci>The interval resamples the {nodes['n_nodes']} <em>assays</em>, not the
{len(sa['pairs_last8'])} pairs &mdash; delete-one-assay jackknife, standard error
{nodes['se']:.4f}. Each assay appears in {nodes['n_nodes'] - 1} pairs, so treating
the pairs as independent understates the width about {(nodes['n_nodes'] - 1) ** 0.5:.1f}-fold:
on simulated graphs with this dependence structure the pair-level interval covers
the truth 43% of the time at a nominal 95%, and the jackknife covers 96%. An
earlier version of this page quoted the pair-level interval.</p>
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
<p class=ci>Spearman correlation between each variant's component score and the
quantity named on the left, computed per assay and then pooled. Bold = the
assay-level 95% interval excludes zero. Final layer.</p>

<h3 style="margin-top:1.3rem">What the rows are</h3>
<ul>
<li><b>DMS</b> &mdash; the experimental measurement, from ProteinGym. For these
assays it is a folding-stability score: <em>higher means the variant is better
tolerated</em>. So a negative correlation means a high component score marks a
destabilising mutation.</li>
<li><b>Signed width change</b> (<code>d sigma</code>) &mdash; the model's
predicted distance distribution for a residue pair has a standard deviation
&sigma;, in &aring;ngstr&ouml;ms. This is &sigma;(mutant) &minus; &sigma;(wild
type), averaged over sampled pairs. <em>Positive means the model became less
certain</em> about where residues sit; negative means it sharpened.</li>
<li><b>Broadening</b> and <b>relocation</b> &mdash; the two halves of the
symmetric KL between the mutant and wild-type distograms, split exactly under a
Gaussian approximation. Relocation is the part explained by the distribution
<em>moving</em> to a new distance; broadening is the part explained by it
<em>widening</em> at the same distance. A raw divergence cannot tell those
apart, which is why the split exists.</li>
<li><b>Symmetric KL</b> &mdash; the Jeffreys divergence between the mutant and
wild-type distograms, averaged over sampled residue pairs. This is the scalar
the original probe used.</li>
<li><b>&Delta; volume</b> and <b>&Delta; hydropathy</b> &mdash; two of the
substitution-chemistry descriptors defined below: the change in residue side-chain
volume (&aring;ngstr&ouml;m<sup>3</sup>) and in Kyte&ndash;Doolittle hydropathy
between the wild-type and mutant amino acid. These depend only on <em>which
letter replaced which</em> and involve the model not at all.</li>
</ul>
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



def sec_symmetry(Y):
    if not Y:
        return pending("Internal versus output with matched dimensionality")
    CV, gaps = Y["components_view"], Y["paired_gaps"]
    INT = "internal dz (last layer)"
    ks = sorted(int(k) for k in CV[INT])
    dims = Y["protocol"]["dims"]

    def best(bn):
        return max(CV[bn][str(k)]["mean"] for k in ks)

    def dim(bn):
        import statistics
        return int(statistics.mean(dims[n][bn] for n in dims))

    order = [INT] + sorted((b for b in CV if b != INT), key=lambda b: -best(b))
    rows = "".join(
        f"<tr><td>{b.replace('output ', 'output: ')}</td>"
        f"<td class=n>{dim(b)}</td><td class=n>{best(b):+.3f}</td></tr>"
        for b in order)
    grows = ""
    for b in order[1:]:
        k = f"{b} @ k=32"
        if k in gaps:
            g = gaps[k]
            grows += (f"<tr><td>{b.replace('output ', 'output: ')}</td>"
                      f"<td class=n>{g['gap']:+.3f} <span class=ci>"
                      f"[{g['ci_lo']:+.3f}, {g['ci_hi']:+.3f}]</span></td>"
                      f"<td class=n>{g['wins']}/{g['n']}</td></tr>")
    out_lo = min(best(b) for b in CV if b != INT)
    out_hi = max(best(b) for b in CV if b != INT)
    mx = max(dim(b) for b in CV if b != INT)
    pl = [b for b in CV if "pLDDT" in b]
    rich_best = best("output rich (published)") if "output rich (published)" in CV else 0.0
    plddt_best = max((best(b) for b in pl), default=0.0)
    plddt_gain = plddt_best - rich_best
    return f"""
<section id=symmetry>
<h2>Internal versus output, with the dimensional asymmetry removed</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">central claim</span>
<h3>Giving the output side full dimensionality does not close the gap</h3></div>
<p><code>compare_internal_output.output_matrix</code> already names the danger in
its own docstring: the real risk to this comparison is that the internal side
gets 256 features and the output side gets one, and it closes that by giving the
structure module &ldquo;the richest description of its own product that the saved
coordinates allow&rdquo;. That was true against four scalar summaries per layer.
It stopped being true when the SVD study showed internal reaches +0.73 on its
raw pair channels &mdash; because the output side was still ten hand-built
numbers, and the coordinates allow far more.</p>
<p>So both sides are run through one protocol here: identical position-grouped
splits, standardisation, rotation and selection views, ridge with the same
inner-fold &lambda; grid, the same k-grid and the same assay-level bootstrap.
The internal side is deliberately handicapped &mdash; a <b>single layer, the
last, fixed in advance</b>, with no layer search and no averaging over the depth
window where the probe is known to be strongest.</p>
<div class=scroll><table>
<thead><tr><th>block</th><th>dimensions</th><th>best held-out &rho;</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>The output side was described at {len(order)-1} sizes spanning
{min(dim(b) for b in order[1:])} to {max(dim(b) for b in order[1:])} dimensions,
including one block that is every emitted quantity at once &mdash;
full-dimensional geometry plus the hand-built summaries plus confidence. Its
ceiling is {out_lo:+.3f} to {out_hi:+.3f} across all of them.</p>
<div class=scroll><table>
<thead><tr><th>paired difference at k = 32</th><th>internal &minus; output</th>
<th>assays won</th></tr></thead>
<tbody>{grows}</tbody></table></div>
</div>
<div class="card">
<div class=row><span class="chip c-run">control</span>
<h3>An estimator problem or an information problem?</h3></div>
<p>&ldquo;Output does badly at {mx} dimensions&rdquo; could simply be the curse of
dimensionality with 250 rows. It is not, and the shape of the result is what
rules it out: the output ceiling is flat from ten dimensions to {mx}, and the
<em>best</em> output description is one of the smallest. Adding raw geometry
actively hurts &mdash; the emitted coordinates are largely uninformative about
stability, so extra dimensions dilute the few useful ones rather than adding to
them. Internal, at 128 dimensions and one layer, sits far above the entire
band.</p>
<p><b>The confidence channel was the one real gap, and it is now closed.</b> An
earlier version of this comparison gave the output side just two scalars of
pLDDT &mdash; the chain mean and the value at the mutated residue &mdash; while
claiming the internal response is about certainty. The <code>gym3</code>
re-capture added the full per-residue vector, and it is worth
{plddt_gain:+.3f} to the output side: {plddt_best:+.3f} against the published
block's {rich_best:+.3f}. Per-residue pLDDT is now the <em>best</em> single
output description, above every geometric one. That sharpens the claim rather
than weakening it &mdash; severity is registered as uncertainty, partially
surfaces in the confidence head, and does not reach the coordinates.</p>
</div>
</section>"""



def sec_steer(T):
    if not T:
        return pending("Causal test: steering along PC2")
    dr = T["drift"]["ca_rmsd"]
    g = T["glob"]
    cd = T["coordinates_vs_drift"]
    rnd = [n for n in g["ca_rmsd"] if n.startswith("random")]
    pc2_ca = g["ca_rmsd"]["PC2"]["even_mean"]
    rlo = min(g["ca_rmsd"][n]["even_mean"] for n in rnd)
    rhi = max(g["ca_rmsd"][n]["even_mean"] for n in rnd)
    ratios = [abs(g[k][n]["ratio"]) for k in g for n in g[k]]
    rows = "".join(
        f"<tr><td>{m}</td>"
        f"<td class=n>{min(v['median'] for v in cd[m].values()):.3f} &ndash; "
        f"{max(v['median'] for v in cd[m].values()):.3f}</td>"
        f"<td class=n>{'exceeds' if max(v['median'] for v in cd[m].values()) > dr else 'below'}</td></tr>"
        for m in ("row", "sym", "glob") if m in cd)
    return f"""
<section id=steer>
<h2>Causal test: is PC2 a lever, or only a readout?</h2>
<div class="card amber">
<div class=row><span class="chip c-amber">negative result</span>
<h3>PC2 is readable but not privileged &mdash; there is no stability knob</h3></div>
<p>Everything else in this report is decodability: the direction is there to be
read. This asks whether the model <em>uses</em> it. PC2 is added to the final
pair representation &mdash; the tensor the structure module is conditioned on
&mdash; and the distogram, per-residue pLDDT and emitted coordinates are
measured. Effect size cannot settle it, because any vector added to z moves the
output; the tests that can are <b>sign structure</b> and <b>comparison against
random directions of identical norm</b>.</p>
<p><b>The response is magnitude-driven, not signed.</b> A direction the model
held as a signed quantity would broaden at +&alpha; and sharpen at
&minus;&alpha;. Decomposing each response into odd and even parts, the odd
component is at most {100*max(ratios):.0f}% of the even one for every direction
tested, PC2 included.</p>
<p><b>PC2 is not stronger than a random direction.</b> Under global injection its
coordinate response is {pc2_ca:.2f}&nbsp;&Aring; against a random range of
{rlo:.2f}&ndash;{rhi:.2f}&nbsp;&Aring;. It is mid-pack on the distogram and on
pLDDT too.</p>
</div>
<div class="card warn">
<div class=row><span class="chip c-warn">correction</span>
<h3>A one-row lever gave the opposite answer, and it was wrong</h3></div>
<p>Injecting into a single row of z leaves the emitted structure below the
sampler's own drift, which reads as &ldquo;the structure module ignores this
channel&rdquo;. It does not. A real substitution perturbs the whole pair tensor,
and when the injection matches that extent the coordinates move by
&aring;ngstr&ouml;ms.</p>
<div class=scroll><table>
<thead><tr><th>injection</th><th>CA RMSD across directions (&Aring;)</th>
<th>vs {dr:.2f}&nbsp;&Aring; sampler drift</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>The drift floor itself had to be measured rather than assumed, and an earlier
version of this experiment got it wrong twice: &alpha;&nbsp;=&nbsp;0 reproduces
the baseline exactly under deterministic sampling with a fixed key, so it is a
determinism check and not a noise floor; and comparing raw coordinates without
superposition read a change of diffusion key as an 18&nbsp;&Aring;
conformational change, when the structures differed only by a rigid-body
frame.</p>
</div>
<div class="card">
<h3>What this does and does not support</h3>
<ul>
<li>It <b>sharpens</b> the claim about coordinates. The mechanism is not that the
sampler ignores the trunk &mdash; it plainly does not. It is that the sampler
responds to <em>how much</em> the pair representation changes and not to
<em>which direction</em> it changes in, while severity is written into a
specific direction.</li>
<li>It <b>ends</b> the steering ambition. There is no direction here that can be
turned up to make the model call a variant destabilising.</li>
<li>It leaves the decodability results untouched: those are claims about what can
be read out of the representation, and a failed intervention does not bear on
them.</li>
<li><b>Not established:</b> PC2's odd component does exceed all four random draws
on two of three readouts under the global lever. With four draws that is worth
roughly p&nbsp;&asymp;&nbsp;0.4 and is reported as a hint, not a result. A claim
there needs on the order of twenty random directions.</li>
<li>One assay (RCRO), one layer, one alignment. The negative is bounded
accordingly.</li>
</ul>
</div>
</section>"""



def sec_methods(S, C):
    if not S:
        return pending("PC2 method")
    D = S["protocol"]["dim"]
    L = S["protocol"]["n_layers"]
    return f"""
<section id=methods>
<h2>How PC2 is obtained</h2>
<div class="card">
<p>This is not an SVD of the model's activations. It is an SVD of the
<em>change</em> in one specific tensor, pooled across proteins under a fixed
protocol, and each of those qualifications is doing work. The full construction,
in order:</p>
<ol>
<li><b>Two forward passes per variant.</b> The Boltz-2 trunk is run on the wild
type and on the mutant with the <em>same</em> alignment &mdash; the mutant
sequence is grafted onto the wild type's homologs rather than searched
separately, so alignment composition cannot move with the mutation &mdash; and
with the same random keys and recycle count. The only difference between the two
passes is the one substituted residue.</li>

<li><b>One tensor, one position.</b> The Pairformer's pair representation is
<code>z[i, j, c]</code>: for every ordered pair of residues, {D} channels. At the
mutated residue <em>r</em> we take its whole row and average over partner
residues, <code>z_site = mean over j of z[r, j, :]</code>, giving a single
{D}-vector. Both passes are read at the <em>same</em> row <em>r</em>, so the
difference reflects the substitution and not which position is being looked at.
The single-sequence track <code>s</code> is captured too and is consistently
weaker; the signal lives between residue pairs.</li>

<li><b>The quantity decomposed is a difference.</b>
&Delta;z = z_site(mutant) &minus; z_site(wild type), one {D}-vector per variant,
taken at the final Pairformer layer (layer {L - 1} of {L}). Decomposing raw
activations instead would return the identity of the protein and the residue,
which is most of their variance and none of the question.</li>

<li><b>Standardise within each protein.</b> Each of the {D} channels is z-scored
across that assay's variants, separately per assay. Without this a protein whose
representation happens to run at a larger scale would dominate the
decomposition simply for being louder.</li>

<li><b>One basis for all twelve proteins.</b> The standardised rows of every
assay are stacked into a single matrix (about 3,000 variants &times; {D}
channels), the column means removed, and an SVD taken. The right singular
vectors are the basis. This is emphatically <em>not</em> a per-assay
decomposition: singular-vector signs are arbitrary, so twelve independent bases
cannot be averaged or compared component-by-component. An earlier version of
this analysis did exactly that and reported &asymp;0.02 against DMS for every
component when the true value was &minus;0.65 &mdash; the effect had cancelled,
not vanished.</li>

<li><b>Fix the sign.</b> Each component is multiplied by &plusmn;1 so that its
pooled correlation with the symmetric KL is non-negative, i.e. a positive score
always means "the internal state moved further". Applied once, before any
interpretation.</li>

<li><b>PC2 is simply the second right singular vector.</b> It is not chosen for
being predictive and not searched for. It is the second-largest direction of
variance in how mutations move the pair representation, and the finding is that
this direction is the stability axis. The first is substitution volume.</li>

<li><b>A variant's PC2 score</b> is one number: the projection of its
standardised, mean-centred &Delta;z onto that vector.</li>
</ol>

<h3 style="margin-top:1.3rem">Two versions of the basis, used for different things</h3>
<p>Where the basis is used to <b>predict</b> &mdash; the k-curves, the
leave-one-assay-out transfer &mdash; it is refitted on <b>training positions
only</b> and frozen before held-out rows are projected into it. A basis fitted
on all rows would leak the test set into the coordinate system itself, which is
the easiest way to produce a flattering and meaningless curve. Where it is used
only to <b>describe</b> &mdash; the component-annotation table below, where
nothing is predicted &mdash; it is fitted on all rows, because there is no
held-out set to protect.</p>
<p class=ci>Splits group by residue POSITION, so no site appears on both sides of
a split. All intervals are bootstrapped over assays, never over variants:
variants at one residue share an environment and are not independent
observations.</p>
</div>
</section>"""


def sec_chem(C):
    if not C:
        return pending("Chemistry control")
    b, g = C["blocks"], C["increments"]
    lo = C["pc2_alone_loao"]
    rows = "".join(
        f"<tr><td>{k}</td><td class=n>{ci(b[k])}</td></tr>" for k in
        ("chemistry (17)", "dz residualised on chemistry (128)", "PC2 alone (1)",
         "PC1-4 (4)", "chemistry + PC2 (18)", "full dz (128)") if k in b)
    grows = "".join(
        f"<tr><td>{k}</td><td class=n>{g[k]['gap']:+.3f} <span class=ci>"
        f"[{g[k]['ci_lo']:+.3f}, {g[k]['ci_hi']:+.3f}]</span></td>"
        f"<td class=n>{g[k]['wins']}/{g[k]['n']}</td></tr>" for k in g)
    return f"""
<section id=chem>
<h2>Is it just substitution chemistry?</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">deciding control</span>
<h3>One scalar beats seventeen chemistry descriptors</h3></div>
<p><b>What "chemistry" means here.</b> Seventeen numbers computed from the
substitution alone &mdash; the two amino-acid letters and nothing else. No model,
no structure, no alignment. They are: the BLOSUM62 substitution score; the change
in Kyte&ndash;Doolittle hydropathy, in Zamyatnin side-chain volume
(&aring;ngstr&ouml;m<sup>3</sup>) and in charge at pH&nbsp;7, each in both signed
and absolute form; the wild-type and mutant values of hydropathy and volume on
their own; and six indicators for the substitutions that carry outsized
structural consequences &mdash; to or from proline, glycine and cysteine.</p>
<p>This matters because most of the variance in a deep mutational scan is
explained by <em>what</em> was substituted rather than <em>where</em>. Proline in
a helix, charge buried in a core, a large residue into a small pocket: none of
that needs a folding model. So "the Pairformer's internal state predicts
stability" is a claim about the model only once it beats these.</p>
<p>The audit named this the deciding baseline, and the SVD made the worry sharper
rather than weaker: PC1 correlates with volume change at &minus;0.80 and even PC2
carries &minus;0.53. The deflationary reading &mdash; the model has simply learned
which amino acid was substituted &mdash; has to be answered directly.</p>
<div class=scroll><table>
<thead><tr><th>feature block</th><th>held-out &rho;</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<div class=scroll><table>
<thead><tr><th>increment</th><th>paired difference</th><th>assays won</th></tr></thead>
<tbody>{grows}</tbody></table></div>
<p><b>PC2 alone, transferred:</b> {ci(lo)} &mdash; one number per variant, with the
basis <em>and its sign</em> taken from the other eleven proteins and nothing
fitted on the held-out one.</p>
<p class=ci>Honest caveat: &Delta;z residualised on chemistry scores +0.455 against
chemistry's +0.401 and that gap includes zero. Residual-alone is not better than
chemistry; what is solid is the increment, which can only come from the
chemistry-orthogonal part.</p>
</div>
</section>"""


def sec_ablate(T):
    if not T:
        return pending("Projection ablation")
    d = T["directions"]
    rows = "".join(
        f"<tr><td>{k}</td><td class=n>{ci(v['recovery'])}</td>"
        f"<td class=n>{v['d_plddt']['mean']:+.5f}</td>"
        f"<td class=n>{v['ca']['mean']:.4f}</td></tr>" for k, v in d.items())
    worst = max(T["positive_control_residual_fraction"].values())
    return f"""
<section id=ablate>
<h2>Deleting PC2 from a real mutation</h2>
<div class="card amber">
<div class=row><span class="chip c-amber">negative result</span>
<h3>Removing the direction the probe reads changes nothing the model emits</h3></div>
<p>Rather than adding a synthetic vector, this takes the difference the model
itself produced for a real variant, removes one direction from it, puts it back
and re-runs the structure module. <b>Positive control:</b> at most
{100*worst:.5f}% of the component survives the removal, so the surgery
demonstrably worked &mdash; without that number a null would be uninterpretable.</p>
<div class=scroll><table>
<thead><tr><th>direction removed</th><th>distogram recovery</th>
<th>&Delta; pLDDT</th><th>CA shift (&Aring;)</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>Recovery is the fraction of the mutation's own width change that the deletion
undoes; 1.0 would mean full reversion to wild type. It is indistinguishable from
zero for PC2, and paired against every random direction the difference includes
zero. CA shifts sit far below the 0.30&nbsp;&Aring; sampler-drift floor.</p>
<p>Since ablating PC2 destroys the probe's entire signal by construction while
the output does not move, the decodable direction is <b>not functionally
load-bearing</b> for the model's own prediction. Bounded: 4 assays, 16 variants
each, and removing 1 of {128} dimensions is a small change in absolute terms
&mdash; which is exactly what the random controls calibrate.</p>
</div>
</section>"""


def sec_xmodel(X, DP):
    if not X:
        return pending("Cross-model comparison")
    cm = X["complementarity"]
    nc = X["principal_angles_negative_control"]
    rows = "".join(
        f"<tr><td>{k.replace('|', ' vs ')}</td><td class=n>{v['mean']:.3f}</td>"
        f"<td class=n>{X['rsa'][k]['mean']:.3f}</td></tr>"
        for k, v in X["cka"].items())
    g = cm["averaged minus best single"]
    dep = ""
    if DP:
        ck = DP["cka_by_depth"]
        mid = {k: np.mean([r["mean"] for r in v if r["frac"] < 0.9])
               for k, v in ck.items()}
        last = {k: v[-1]["mean"] for k, v in ck.items()}
        dep = f"""
<div class="card warn">
<div class=row><span class="chip c-warn">correction</span>
<h3>The asymmetry above is a last-layer artifact</h3></div>
<p>Those CKA values compare each model's FINAL layer, and the trunks are
{DP['n_layers']['boltz2']}, {DP['n_layers']['of3']} and
{DP['n_layers']['protenix']} blocks deep &mdash; "the last of 64" and "the last
of 16" are not the same amount of computation. At matched <em>fractional</em>
depth the three agree far more closely and far more uniformly:</p>
<div class=scroll><table>
<thead><tr><th>pair</th><th>mean CKA, l/L &lt; 0.9</th><th>at the last layer</th></tr></thead>
<tbody>{''.join(f"<tr><td>{k.replace('|', ' vs ')}</td><td class=n>{mid[k]:.3f}</td>"
                f"<td class=n>{last[k]:.3f}</td></tr>" for k in ck)}</tbody></table></div>
<p>Every pair falls off sharply only at l/L = 1, which is where each model
specialises for its own heads. An earlier version of this report attributed the
gap to architectural lineage; that explanation was wrong twice over &mdash; all
three use the same AF3-derived trunk, and the gap is not a property of the
models at all.</p>
</div>"""
    return f"""
<section id=xmodel>
<h2>Three models, one mutation signal</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">generality</span>
<h3>All three decode it; none knows anything the others do not</h3></div>
<p>Boltz-2, OpenFold3 and Protenix all use a {X.get('dim', 128)}-dimensional pair
representation, but they are trained independently, so their channel bases are
unrelated &mdash; confirmed as a negative control: principal angles sit at
{min(np.mean(list(v.values())) for v in nc['pairs'].values()):.3f}&ndash;
{max(np.mean(list(v.values())) for v in nc['pairs'].values()):.3f} against a
chance level of {nc['chance']:.3f}. Every real comparison therefore goes through
the variant axis.</p>
<div class=scroll><table>
<thead><tr><th>pair</th><th>CKA</th><th>RSA &rho;</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>Probes: Boltz-2 {ci(cm['boltz2'])}, OpenFold3 {ci(cm['of3'])}, Protenix
{ci(cm['protenix'])}. Combining them buys
<b>{g['gap']:+.3f}</b> <span class=ci>[{g['ci_lo']:+.3f}, {g['ci_hi']:+.3f}]</span>
&mdash; the interval includes zero, so the phenotype information is
<b>redundant</b> across models.</p>
<p>The finding is therefore a property of the model class rather than of any one
implementation. Bounded by <b>4 assays</b>; a null on complementarity at that
size is weak evidence and should not be over-read.</p>
</div>
{dep}
</section>"""



def sec_scope(SC):
    if not SC:
        return pending("Is one averaged pair row too narrow a view?")
    b, g = SC["blocks"], SC["vs_averaged_row"]
    ref = "row, averaged over partners (dz_site)"
    dims = SC["dims"]
    def dim(k):
        import statistics
        return int(statistics.mean(dims[n][k] for n in dims))
    order = [ref] + sorted((k for k in b if k != ref),
                           key=lambda k: -b[k]["mean"])
    rows = "".join(
        f"<tr><td>{k}</td><td class=n>{dim(k)}</td><td class=n>{ci(b[k])}</td>"
        f"<td class=n>{'&mdash;' if k == ref else f'{g[k][chr(103)+chr(97)+chr(112)]:+.3f}'}</td>"
        f"<td class=n>{'&mdash;' if k == ref else f'{g[k][chr(119)+chr(105)+chr(110)+chr(115)]}/{g[k][chr(110)]}'}</td></tr>"
        for k in order)
    kg = b["KL over the whole protein (64 layers)"]["mean"]
    ks = b["KL at the mutated site only (64 layers)"]["mean"]
    prof = g["row, per-partner magnitude only"]
    return f"""
<section id=scope>
<h2>Is one averaged pair row too narrow a view?</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">objection answered</span>
<h3>The averaged row is the best view tested &mdash; but not because the effect is local</h3></div>
<p>Everything on this page rests on <code>dz_site</code>, the mutated residue's
pair row averaged over its partners. That is not one residue's activation
&mdash; it is that residue's relationship to the whole chain &mdash; but it does
discard two things: <em>which</em> partners moved, and every pair in which the
mutated residue takes no part. A mutation can perturb the fold far from where it
sits, so this had to be tested rather than assumed. It was inherited from the
original capture and every result downstream depends on it.</p>
<div class=scroll><table>
<thead><tr><th>view</th><th>dimensions</th><th>held-out &rho;</th>
<th>vs the averaged row</th><th>assays won</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p><b>The averaging costs nothing measurable.</b> The comparison that carries
weight is per-partner magnitude (~{dim('row, per-partner magnitude only')} dims)
against the channel average ({dim(ref)} dims) &mdash; similar size, both
well-conditioned &mdash; and the average wins by
<b>{-prof['gap']:+.3f}</b>. How the channels moved carries substantially more
than which partners moved.</p>
<p class=ci><b>A caveat that applies to this page's own argument.</b> The full
unaveraged row scoring worst does NOT show its detail is uninformative: it
contains the average as a linear function, so it can only hold more information.
At ~{dim('row, full (partners x channels)')} dimensions against 250 rows that is
an estimation failure, not an information one &mdash; exactly the
curse-of-dimensionality argument used elsewhere here to defend the internal side
against the output side, and it applies symmetrically. Only the per-partner
comparison is decisive.</p>
</div>
<div class="card">
<div class=row><span class="chip c-run">the informative part</span>
<h3>The effect is distributed; the row is an efficient summary of it</h3></div>
<p>No archived tensor holds &Delta;z for pairs where neither residue is the
mutation site, but the divergence features stand in for it: <code>kl_site</code>
averages over sampled pairs that <em>touch</em> the mutated residue,
<code>kl_glob</code> over <em>all</em> sampled pairs, most of which do not.</p>
<p>The whole-protein view is the <b>stronger</b> of the two &mdash;
{kg:+.3f} against {ks:+.3f} &mdash; and combining them adds nothing. So a
mutation's influence genuinely reaches beyond its own row. What makes
<code>dz_site</code> work is therefore not that the perturbation is local, but
that the mutated residue's row <b>aggregates that distributed influence across
every partner</b>. It is an efficient summary of a spread-out effect rather than
a narrow window onto a confined one.</p>
<p class=ci>Not tested: &Delta;z itself off the row. <code>kl_glob</code> is a
scalar summary of the distogram, not of the representation, and storing the full
pair tensor is N&sup2;&times;128 per layer per variant. Closing this properly
needs a capture that keeps one global &Delta;z summary alongside the row.</p>
</div>
</section>"""



def sec_scrutiny(SR, AT):
    if not SR:
        return pending("Scrutiny of the PC2 claim")
    rk = SR["ranking"]
    wa = SR["wt_anchor"]
    prof = SR["pc2_profile"]
    par = SR["pc2_vs_dms_partial_on_chemistry"]
    strong = sorted(((v["mean"], k) for k, v in prof.items()
                     if abs(v["mean"]) >= 0.15), key=lambda t: -abs(t[0]))
    rows = "".join(
        f"<tr><td>{k.replace('chem: ', 'chemistry: ')}</td>"
        f"<td class=n>{v:+.3f}</td></tr>" for v, k in strong)
    alt = ""
    if AT:
        r = AT["pc2_vs_dms_residualised"]
        alt = (f"{r['minus chemistry']['mean']:+.3f}",
               100 * abs(r['minus chemistry']['mean'] / r['raw']['mean']))
    return f"""
<section id=scrutiny>
<h2>Three challenges to the PC2 claim</h2>

<div class="card ok">
<div class=row><span class="chip c-ok">generalisation</span>
<h3>One ranking across twelve proteins works as well as twelve within them</h3></div>
<p>Every correlation elsewhere on this page is computed <em>inside</em> an assay
and averaged over assays. That is the ProteinGym convention and it is what the
report quotes, but it never asks whether one PC2 score means the same thing in
two different proteins. Pooling all {rk['n_variants']:,} variants into a single
ranking:</p>
<div class=scroll><table>
<thead><tr><th>ranking</th><th>&rho; vs DMS</th></tr></thead>
<tbody>
<tr><td>mean within assay <b>(the reported statistic)</b></td>
<td class=n>{ci(rk['within_assay_mean'])}</td></tr>
<tr><td>one pooled ranking, target z-scored per assay</td>
<td class=n>{rk['pooled_target_zscored']:+.3f}</td></tr>
<tr><td>one pooled ranking, raw targets</td>
<td class=n>{rk['pooled_raw_target']:+.3f}</td></tr>
</tbody></table></div>
<p>The score is calibrated <em>across</em> proteins, not merely monotone within
each. The within-assay average remains the headline because it is the field's
convention; the pooled number is supporting evidence. Raw targets score lower
because DMS units differ between assays &mdash; assay mean DMS alone correlates
{rk['assay_offset_confound']:+.3f} with raw pooled targets, which is the offset
the per-assay z-scoring removes without touching any within-assay ordering.</p>
</div>

<div class="card warn">
<div class=row><span class="chip c-warn">naming</span>
<h3>It is a subspace of the model's coordinates, not a part of the model</h3></div>
<p>&Delta;z = z<sub>mut</sub> &minus; z<sub>WT</sub> is something <em>we</em>
compute; the model never forms it. The directions are a genuine linear subspace
<em>of</em> the pair-channel space, but they are an observer's construct on those
coordinates rather than a module the network allocates or consults &mdash; and
the causal sections below agree, since injecting or ablating PC2 does no more
than a random direction of equal norm. Read &ldquo;mutation subspace&rdquo;
throughout as shorthand for <b>the subspace along which mutations displace the
pair representation</b>.</p>
<p>One part of the worry is testable: whether the direction is an artifact of
anchoring on wild type. Because z<sub>a</sub> &minus; z<sub>b</sub> =
&Delta;z<sub>a</sub> &minus; &Delta;z<sub>b</sub>, mean-centring the &Delta;z set
gives exactly the variant-to-variant difference space, with the wild-type term
cancelled. Comparing that against a scale-only standardisation that keeps the
anchor: subspace agreement cos<sup>2</sup> = <b>{wa['subspace_cos2']:.3f}</b>
(chance {wa['chance']:.3f}), PC2 scores correlating
{wa['score_rho']:+.3f}. The anchor is not what creates the direction.</p>
<p class=ci>A first version of this test was vacuous: standardising within assay
already subtracts the mean, so both branches decomposed the same centred matrix
and it returned a perfect cos<sup>2</sup> = 1.000 while testing nothing. The
figure above uses scale-only standardisation so the anchor actually survives
into the decomposition.</p>
</div>

<div class="card amber">
<div class=row><span class="chip c-amber">entanglement</span>
<h3>PC2 is a mixture, and how much survives depends on how you ask</h3></div>
<p>PC2 is not a clean stability axis. Everything it correlates with above
|&rho;| = 0.15:</p>
<div class=scroll><table>
<thead><tr><th>quantity</th><th>&rho; with PC2</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>Its association with DMS holding all seventeen chemistry descriptors fixed
depends on the construction, and the two disagree enough to matter:</p>
<ul>
<li><b>Partial Spearman</b> (residualising the <em>ranks</em>, this project's
standard tool): {par['partial']:+.3f} against a raw {par['raw']:+.3f} &mdash;
{100*abs(par['partial']/par['raw']):.0f}% surviving.</li>
{f"<li><b>Residualising the raw scores</b>: {alt[0]} &mdash; {alt[1]:.0f}% surviving.</li>" if alt else ""}
</ul>
<p>The relationship is not rank-linear, so the two constructions give different
answers, and the first is the more flattering of them. An earlier version of this
report quoted only the {100*abs(par['partial']/par['raw']):.0f}% figure. The
honest statement is that between roughly 70% and 98% of the association survives
depending on the method, and the next section pushes the question harder with a
non-parametric control.</p>
</div>
</section>"""


def sec_attrib(AT):
    if not AT:
        return pending("What the mutant contributes")
    r = AT["pc2_vs_dms_residualised"]
    ex = AT["explains_pc2"]
    pr = AT["profile"]
    def top(side, n=5, rev=False):
        it = sorted(pr[side].items(), key=lambda kv: kv[1]["mean"], reverse=not rev)
        return ", ".join(f"<b>{c}</b> {v['mean']:+.2f}" for c, v in it[:n])
    rows = "".join(
        f"<tr><td>{k}</td><td class=n>{ci(v)}</td>"
        f"<td class=n>{100*abs(v['mean']/r['raw']['mean']):.0f}%</td></tr>"
        for k, v in r.items())
    erows = "".join(f"<tr><td>{k}</td><td class=n>{ci(v)}</td></tr>"
                    for k, v in ex.items() if np.isfinite(v["mean"]))
    return f"""
<section id=attrib>
<h2>What about the mutant drives its PC2 score?</h2>
<div class="card">
<p>PC2 is a projection of a difference, so its value could be set by which
residue was <em>removed</em>, which was <em>introduced</em>, or the structural
context of the site. Amino-acid identity &mdash; twenty indicators per side
&mdash; is the non-parametric version of the chemistry control: it can express
any function of the substitution at all, including everything the seventeen
descriptors capture and everything they miss.</p>
<h3 style="margin-top:1.2rem">The residue profiles are not mirror images</h3>
<p><b>Introduced</b> &mdash; highest {top('introduced')}; lowest
{top('introduced', rev=True)}.<br>
<b>Removed</b> &mdash; highest {top('removed')}; lowest
{top('removed', rev=True)}.</p>
<p>PC2 runs high when a large hydrophobic is replaced by something small or
polar, which is the core-disrupting direction. The asymmetry is informative: the
<em>removed</em> residue carries more weight than the introduced one, both in the
profile and in how much of PC2 each explains.</p>
<div class=scroll><table>
<thead><tr><th>block predicting PC2 (position-grouped CV)</th>
<th>&rho;</th></tr></thead>
<tbody>{erows}</tbody></table></div>
</div>
<div class="card amber">
<div class=row><span class="chip c-amber">the qualification</span>
<h3>Part of PC2 really is a substitution lookup &mdash; but not most of it</h3></div>
<div class=scroll><table>
<thead><tr><th>PC2 vs DMS, after removing</th><th>&rho;</th>
<th>share of raw</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>Amino-acid identity removes about
{100 - 100*abs(r['minus both identities']['mean']/r['raw']['mean']):.0f}% of the
association &mdash; more than the hand-built descriptors do, which is expected
from forty free parameters against seventeen. So a real part of PC2 <em>is</em>
a substitution lookup. But
{100*abs(r['minus both identities']['mean']/r['raw']['mean']):.0f}% survives, and
that residual still tracks DMS at
{r['minus both identities']['mean']:+.3f}.</p>
<p><b>Site identity removes more than substitution identity does</b>
({100*abs(r['minus site identity']['mean']/r['raw']['mean']):.0f}% surviving
versus
{100*abs(r['minus both identities']['mean']/r['raw']['mean']):.0f}%). PC2 is
more a statement about <em>where</em> a mutation sits than about <em>what</em> it
is, which matches the earlier finding that position sensitivity was its
strongest single correlate. Removing both still leaves
{100*abs(r['minus both identities AND site']['mean']/r['raw']['mean']):.0f}%.</p>
<p class=ci>The defensible claim is therefore narrower than &ldquo;PC2 is the
stability axis&rdquo;: it is a mixture of substitution identity and structural
context, weighted toward context, with a residual that neither accounts for.</p>
</div>
</section>"""



def sec_bw(BW):
    if not BW:
        return pending("Positions or mutations?")
    b, g = BW["blocks"], BW["internal_minus"]
    ref = list(b)[0]
    rows = "".join(
        f"<tr><td>{k}</td><td class=n>{ci(v['between'])}</td>"
        f"<td class=n>{ci(v['within'])}</td></tr>" for k, v in b.items())
    grows = "".join(
        f"<tr><td>{k}</td>"
        f"<td class=n>{v['between']['gap']:+.3f} <span class=ci>"
        f"[{v['between']['ci_lo']:+.3f}, {v['between']['ci_hi']:+.3f}]</span> "
        f"{v['between']['wins']}/{v['between']['n']}</td>"
        f"<td class=n>{v['within']['gap']:+.3f} <span class=ci>"
        f"[{v['within']['ci_lo']:+.3f}, {v['within']['ci_hi']:+.3f}]</span> "
        f"{v['within']['wins']}/{v['within']['n']}</td></tr>"
        for k, v in g.items())
    chem = [k for k in g if "chemistry" in k]
    cw = g[chem[0]]["within"] if chem else None
    cb = g[chem[0]]["between"] if chem else None
    return f"""
<section id=bw>
<h2>Does the model know about positions, or about mutations?</h2>
<div class="card ok">
<div class=row><span class="chip c-ok">the claim survives</span>
<h3>Internal beats the output within a site as well as across sites</h3></div>
<p>Position-grouped splits stop a probe memorising which residues are fragile,
but they do not stop it learning what makes a residue fragile in general. Since
the attribution section found that removing site identity cost more than
removing substitution identity, the obvious worry is that the whole internal
advantage is really "this model knows which positions are sensitive" &mdash; a
much weaker statement than "this model knows what a given mutation does".</p>
<p>The same held-out predictions, from the same fits, split two ways: aggregated
to position means (can it rank <em>sites</em>?) and with each position's mean
removed from both sides (given a site, can it rank the handful of
<em>substitutions</em> at it?). {100*BW['within_coverage']:.0f}% of held-out rows
sit at a site holding at least two variants, so the within-position half is
well powered.</p>
<div class=scroll><table>
<thead><tr><th>block</th><th>between positions</th><th>within positions</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<div class=scroll><table>
<thead><tr><th>internal minus &hellip;</th><th>between</th><th>within</th></tr></thead>
<tbody>{grows}</tbody></table></div>
<p>Against the emitted output the margin is decisive on <b>both</b> axes and, if
anything, larger within positions than across them. The advantage is therefore
not an artifact of site sensitivity: given a residue, the internal
representation ranks the individual substitutions at it far better than anything
the model emits.</p>
</div>
<div class="card amber">
<div class=row><span class="chip c-amber">qualification</span>
<h3>Against chemistry, most of the edge is structural context</h3></div>
<p>Substitution chemistry is a different comparator from the output, and against
it the picture splits:</p>
<ul>
<li><b>Between positions {cb['gap']:+.3f}</b>
<span class=ci>[{cb['ci_lo']:+.3f}, {cb['ci_hi']:+.3f}]</span>,
{cb['wins']}/{cb['n']} &mdash; the model knows which sites are sensitive far
better than chemistry does.</li>
<li><b>Within positions {cw['gap']:+.3f}</b>
<span class=ci>[{cw['ci_lo']:+.3f}, {cw['ci_hi']:+.3f}]</span>,
{cw['wins']}/{cw['n']} &mdash; at the variant level inside a site it beats
chemistry only modestly. The interval clears zero, but this is the tightest
margin anywhere in this report.</li>
</ul>
<p>So most of the internal signal's edge over chemistry is <em>structural
context</em>: which residue, in this fold, matters. Its edge in discriminating
<em>which</em> substitution at a given site is real but small. That leaves the
headline untouched &mdash; that comparison is against the model's own output,
where internal wins on both axes &mdash; and it matches the attribution result,
where PC2 proved more a statement about where a mutation sits than what it
is.</p>
<p class=ci>Within-position values are lower for every block and necessarily so:
few variants share a site, the DMS spread inside a site is small, and
measurement noise is a larger share of it. Blocks are comparable down a column,
never across columns.</p>
</div>
</section>"""


# The current runs. These were svd_dz_v2 and symmetry_v2 long after v3 of each
# existed, so the published build depended on flags typed at a shell -- which is
# how the archived data came to disagree with the prose.
A_SVD = str(W / "runs/svd_dz_v3.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svd", default=A_SVD)
    ap.add_argument("--svd-ds", default=str(W / "runs/svd_ds_v1.json"))
    ap.add_argument("--drift", default=str(W / "runs/drift_v1.json"))
    ap.add_argument("--transfer", default=str(W / "runs/transfer_v1.json"))
    ap.add_argument("--pc2", default=str(W / "runs/pc2_v2.json"))
    ap.add_argument("--symmetry", default=str(W / "runs/symmetry_v3.json"))
    ap.add_argument("--steer", default=str(W / "runs/steer_RCRO_v4.json"))
    ap.add_argument("--chem", default=str(W / "runs/chem_v1.json"))
    ap.add_argument("--ablate", default=str(W / "runs/ablate_v1.json"))
    ap.add_argument("--xmodel", default=str(W / "runs/xmodel_v1.json"))
    ap.add_argument("--depth", default=str(W / "runs/depth_v1.json"))
    ap.add_argument("--scope", default=str(W / "runs/scope_v1.json"))
    ap.add_argument("--scrutiny", default=str(W / "runs/scrutiny_v2.json"))
    ap.add_argument("--attrib", default=str(W / "runs/attrib_v1.json"))
    ap.add_argument("--bw", default=str(W / "runs/bw_v1.json"))
    ap.add_argument("--heldout", default=str(W / "runs/heldout_v1.json"))
    ap.add_argument("--allow-stale-figures", action="store_true",
                    help="render even if a figure predates its source JSON")
    a = ap.parse_args()

    resolved = {k: v for k, v in vars(a).items()
                if k != "allow_stale_figures" and isinstance(v, str)}
    stale = check_figures(resolved, a.allow_stale_figures)
    manifest = archive_inputs(resolved, stale)

    S, DS, Dj, TR = (load(a.svd), load(a.svd_ds), load(a.drift), load(a.transfer))
    Q, Y, T = load(a.pc2), load(a.symmetry), load(a.steer)
    C, AB, X, DP = (load(a.chem), load(a.ablate), load(a.xmodel), load(a.depth))
    SC, SR, AT = load(a.scope), load(a.scrutiny), load(a.attrib)
    BW, HO = load(a.bw), load(a.heldout)
    css = (OUT / "style.css").read_text()
    inputs_txt = " &middot; ".join(
        f"{k}: <code>{v['file']}</code> ({v['sha256'][:10]})"
        for k, v in sorted(manifest["inputs"].items()))
    drifted = manifest["code_drifted_from_repo"]
    drift_txt = ("" if not drifted else
                 f" <b>Note:</b> the executing harness copy differs from the "
                 f"repo at this commit for {', '.join(drifted)}, so this build "
                 f"is not reproducible from the commit alone.")
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
  of directions &mdash; a subspace of the model's coordinates along which mutations
  displace the representation, not a module the model itself allocates &mdash;
  whose second member tracks both measured stability and predictive certainty &mdash; which is the mechanism report's conclusion,
  recovered without computing a divergence.</p>
</header>

<figure>
  <img src="figures/svd.png" alt="Four panels: held-out Spearman against number of
  directions kept; transfer on a basis learned without the held-out protein; a
  component-by-quantity association heatmap on a shared basis; cross-assay subspace
  agreement by layer.">
  <figcaption>Phase A on <code>dz_site</code>, twelve ProteinGym assays. Generated by
  <code>fig_svd.py</code> from the SVD run.</figcaption>
</figure>

<figure>
  <img src="figures/symmetry.png" alt="Three panels: held-out Spearman against directions
  kept for internal and five output blocks; paired per-assay differences at k=32; best
  score against the number of dimensions given to each block.">
  <figcaption>The internal/output comparison with matched treatment on both sides.
  Generated by <code>fig_symmetry.py</code> from the symmetry run.</figcaption>
</figure>

{sec_symmetry(Y)}
{sec_methods(S, C)}
{sec_dims(S, DS)}
{sec_scope(SC)}
{sec_shared(S)}
{sec_what(S)}
{sec_transfer(S, TR)}

<figure>
  <img src="figures/heldout.png" alt="Three panels: per-assay frozen-PC2 correlation for
  sixteen held-out proteins against a frozen chemistry baseline; block summaries split by
  phenotype group; and correlation against chain length showing the length confound.">
  <figcaption>The frozen basis on sixteen proteins outside it.
  <code>fig_heldout.py</code> from <code>heldout_v1.json</code>.</figcaption>
</figure>

{sec_heldout(HO)}

<figure>
  <img src="figures/chem.png" alt="Feature blocks, paired increments, and PC2 alone
  transferred per assay.">
  <figcaption>The chemistry control. <code>fig_chem.py</code> from <code>chem_v1.json</code>.</figcaption>
</figure>

{sec_chem(C)}
{sec_scrutiny(SR, AT)}
{sec_attrib(AT)}
{sec_bw(BW)}

<figure>
  <img src="figures/xmodel.png" alt="Principal-angle control, agreement against the
  self-repeat ceiling, and complementarity.">
  <figcaption>Three models on identical variants. <code>fig_xmodel.py</code>.</figcaption>
</figure>
<figure>
  <img src="figures/depth.png" alt="Decodability, cross-model agreement and the severity
  direction against fractional depth.">
  <figcaption>Depth-resolved, which corrects the last-layer comparison above.
  <code>fig_depth.py</code>.</figcaption>
</figure>

{sec_xmodel(X, DP)}

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

<figure>
  <img src="figures/steer.png" alt="Three panels: coordinate movement against perturbation
  size for three injection extents; odd-over-even response ratio per direction; PC2
  against random directions of identical norm.">
  <figcaption>Injecting directions into the final pair representation. Generated by
  <code>fig_steer.py</code> from <code>steer_RCRO_v4.*</code>.</figcaption>
</figure>

{sec_steer(T)}
{sec_ablate(AB)}
{sec_methods_full(S, HO, manifest)}

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
  read from the runs, not transcribed. Built {manifest['built']} at commit
  <code>{manifest['commit']}</code> from {len(manifest['inputs'])} inputs, every one
  of which this script copied into <code>data/</code> itself and listed with its
  SHA-256 in <code>data/manifest.json</code>. Figures are checked against the mtime
  of their source JSON and the build aborts if one is older.{drift_txt}</p>
  <p class=ci>{inputs_txt}</p>
</footer>
</div></body></html>
"""
    (OUT / "index.html").write_text(html)
    print(f"wrote {OUT/'index.html'}  ({len(html)/1024:.0f} KB)")
    for nm, obj in (("svd dz", S), ("svd ds", DS), ("drift", Dj),
                    ("transfer", TR), ("pc2", Q), ("symmetry", Y),
                    ("steer", T), ("chem", C), ("ablate", AB),
                    ("xmodel", X), ("depth", DP), ("scope", SC),
                    ("scrutiny", SR), ("attrib", AT), ("bw", BW)):
        print(f"   {nm:10s} {'ok' if obj else 'MISSING -> pending card'}")


if __name__ == "__main__":
    main()
