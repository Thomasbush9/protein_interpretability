"""Shared provenance machinery for the report builders.

`build_svd_report.py` grew this contract after the August 2026 review found
three defects that were all one defect: a number typed rather than read, a
figure drawn from a superseded file, and an archive naming a JSON the build had
not used. The rules it settled on are:

  * every number on a page is read from an analysis output at build time;
  * every resolved input is copied next to the page with its SHA-256;
  * a figure older than any JSON it draws from aborts the build;
  * the footer stamps the commit AND a digest of the code that actually ran,
    because jobs execute `$W/harness/*.py`, an unversioned copy of the git
    working tree -- a commit alone is decoration if the copy can be any age.

Splitting the Jacobian study into two pages would have meant a second copy of
all of that, and a second copy is how the first one drifts. So it lives here and
both builders import it.

Nothing in this module knows what a report says; it only knows how to prove
where the numbers came from.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
# The repository ROOT, not jax_harness. Code identity used to be a flat list of
# basenames because every script lived in one directory; now the library modules
# live under src/protein_interpretability/ and the entry points do not, so the
# names in CODE are repo-relative paths and are resolved from here.
REPO = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/"
            "protein_interpretability")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


UNLABELLED = []      # inputs loaded without a protocol block, for the page
UNREADABLE = []      # inputs that could not be parsed at all


def load(p, *, quoted=False):
    """Load an input, refusing one that a page QUOTES but cannot interpret.

    The previous version swallowed every exception and returned None, so a
    missing file, a corrupt file and a fine one were indistinguishable at the
    call site and the page rendered either way.

    `quoted=True` marks an input supplying a number the page STATES. Those
    fail hard, because a sentence built on an archive whose layer convention,
    orientation rule and truncation are unknown is exactly the situation that
    took a session to untangle -- the 0.573 / 0.696 / 0.703 / 0.731 / 0.758
    spread was five numbers that could not be compared without opening five
    files and reading prose.

    Everything else loads and is recorded in UNLABELLED, which the page shows.
    A build that cannot run is a build someone switches the guard off for, so
    only the quoted path is fatal.
    """
    path = Path(p)
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        UNREADABLE.append((path.name, str(e)[:80]))
        if quoted:
            raise ValueError(
                f"{path.name} supplies a quoted number and could not be read: "
                f"{e}") from e
        return None
    ok = isinstance(d, dict) and bool(d.get("protocol", {}).get("script"))
    if not ok:
        UNLABELLED.append(path.name)
        if quoted:
            raise ValueError(
                f"{path.name} supplies a number this page states, and carries "
                f"no protocol block -- so nothing records which layer, which "
                f"split design, or how wide the predictor was before "
                f"truncation. Rerun it through pi_archive.write_result, or "
                f"drop the claim that reads it. Do NOT hand-write a block: an "
                f"inferred protocol is indistinguishable from a recorded one "
                f"afterwards.")
    return d


def provenance_notice():
    """HTML for whatever loaded without a block. Empty when everything did."""
    if not UNLABELLED and not UNREADABLE:
        return ""
    rows = "".join(f"<li><code>{n}</code></li>" for n in sorted(set(UNLABELLED)))
    bad = "".join(f"<li><code>{n}</code> &mdash; {e}</li>" for n, e in UNREADABLE)
    return f"""
<section id=provenance-gaps>
<div class="card warn">
<h3>Inputs on this page that do not say what they are</h3>
<p>These loaded without a protocol block, so the file itself does not record
which layer, which split design, or how wide the predictor was before any
truncation. Nothing here supplies a number the page states &mdash; that case
raises instead of rendering.</p>
<ul>{rows}{bad}</ul>
</div>
</section>"""


def check_figures(out: Path, figspec, resolved, allow_stale=False):
    """Abort if a figure predates ANY input it is drawn from.

    Checking only the nominal source is not enough: these figures read four
    JSONs each, and a stale gate_probe would be as wrong as a stale jac_pooled.
    mtime is a weak check -- it cannot see a figure rebuilt from the wrong file
    -- so the manifest also records every source's digest.
    """
    stale = []
    live = [Path(v) for v in resolved.values() if v and Path(v).exists()]
    if not live:
        return []
    newest = max(p.stat().st_mtime for p in live)
    for fig, cmd in figspec.items():
        fp = out / "figures" / fig
        if not fp.exists():
            continue
        if fp.stat().st_mtime < newest:
            stale.append((fig, cmd.format(out=fp, **resolved)))
    if stale and not allow_stale:
        raise SystemExit(
            "figures older than the data they are drawn from:\n"
            + "\n".join(f"  {f}" for f, _ in stale)
            + "\n\nregenerate, then rebuild:\n"
            + "\n".join(f"    {c}" for _, c in stale)
            + "\n\n(--allow-stale-figures overrides, but the last time a figure "
              "and its prose disagreed it reached a reviewer.)")
    return [f for f, _ in stale]


def archive_inputs(out: Path, resolved, stale, code_files):
    dd = out / "data"
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
                                cwd=REPO, capture_output=True, text=True,
                                timeout=10).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"

    here = REPO
    code, drifted, uncommitted, differs = {}, [], [], []
    for name in code_files:
        p = here / name
        if not p.exists():
            continue
        code[name] = sha256(p)
        # The mirror is a copy of jax_harness only, so the drift check applies
        # to entry points and not to the library modules that now live under
        # src/ -- those cannot drift, because there is no second copy of them.
        if name.startswith("jax_harness/"):
            r = W / "harness" / Path(name).name
            if r.exists() and sha256(r) != code[name]:
                drifted.append(name)
        # Recording a commit is not the same as being AT it. An earlier version
        # of this function compared the repo working tree against the harness
        # copy -- both uncommitted -- and stamped `commit` regardless, so a page
        # could claim to be reproducible from a commit that did not contain the
        # code that built it. A review caught exactly that. Verify against the
        # object store instead.
        if commit != "unknown":
            blob = subprocess.run(
                ["git", "cat-file", "blob", f"{commit}:{name}"],
                cwd=REPO, capture_output=True, timeout=10)
            if blob.returncode != 0:
                uncommitted.append(name)
            elif hashlib.sha256(blob.stdout).hexdigest() != code[name]:
                differs.append(name)

    manifest = {
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": commit, "inputs": entries, "code_sha256": code,
        "code_drifted_from_repo": drifted, "figures_stale_at_build": stale,
        "orphaned_in_data_dir": orphans,
        "code_absent_from_commit": uncommitted,
        "code_differs_from_commit": differs,
        "reproducible_from_commit": not (uncommitted or differs)}
    if drifted:
        print(f"   WARNING: the harness copy that ran differs from the repo at "
              f"{commit} for {', '.join(drifted)} -- this build is not "
              f"reproducible from that commit")
    if uncommitted or differs:
        n = len(uncommitted) + len(differs)
        print(f"   NOT REPRODUCIBLE FROM {commit}: {n} of {len(code)} scripts "
              f"are absent from or differ from that commit.")
        if uncommitted:
            print(f"     absent:  {', '.join(uncommitted)}")
        if differs:
            print(f"     differs: {', '.join(differs)}")
        print(f"     Commit them before the page is cited as reproducible.")
    if orphans:
        print(f"   note: {len(orphans)} orphan JSON(s) in data/: "
              f"{', '.join(orphans)}")
    (dd / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(
        f"<td class=n>{c}</td>" if i else f"<td>{c}</td>"
        for i, c in enumerate(r)) + "</tr>" for r in rows)
    return (f"<div class=scroll><table><thead><tr>{h}</tr></thead>"
            f"<tbody>{b}</tbody></table></div>")


def pending(what):
    return (f'<div class="card amber"><div class=row><span class="chip c-amber">'
            f'pending</span><strong>{what}</strong></div>'
            f'<p>The analysis JSON for this section is missing, so the builder '
            f'emits this card rather than leaving a stale number on the page.'
            f'</p></div>')


def page(out: Path, *, title, eyebrow, h1, lede, nav_items, body, manifest,
         sibling=None):
    """Write index.html and a self-contained standalone.html beside it.

    The standalone copy is generated rather than assembled by hand so a
    published page cannot drift from the one in the report directory.
    """
    src = W / "report_svd" / "style.css"
    if src.exists():
        shutil.copy2(src, out / "style.css")

    nav = "".join(f'<a href="#{i}">{n}</a>' for i, n in nav_items)
    if sibling:
        nav += f'<a href="{sibling[0]}" style="margin-left:auto">{sibling[1]} &rarr;</a>'

    drift = ""
    if manifest["code_drifted_from_repo"]:
        drift = (" &middot; <strong>harness drifted from repo for "
                 f"{', '.join(manifest['code_drifted_from_repo'])}</strong>")

    html = f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel=stylesheet href="style.css"></head>
<body><div class=wrap>
<header class=head>
<div class=eyebrow>{eyebrow}</div>
<h1>{h1}</h1>
<p class=lede>{lede}</p>
<nav>{nav}</nav>
</header>
{body}
<footer>
Built {manifest['built']} from commit <code>{manifest['commit']}</code>{drift}.
Every number on this page was read from the archived analysis JSONs at build
time; see <code>data/manifest.json</code>.
</footer>
</div></body></html>"""
    (out / "index.html").write_text(html)
    print(f"wrote {out / 'index.html'}  ({len(html):,} bytes)")

    css = (out / "style.css").read_text() if (out / "style.css").exists() else ""
    inner = html.split("<body>", 1)[1].rsplit("</body>", 1)[0]
    import base64
    for fp in sorted((out / "figures").glob("*.png")):
        uri = ("data:image/png;base64,"
               + base64.b64encode(fp.read_bytes()).decode())
        inner = inner.replace(f'src="figures/{fp.name}"', f'src="{uri}"')
    single = f"<title>{title}</title>\n<style>\n{css}\n</style>\n{inner}"
    (out / "standalone.html").write_text(single)
    print(f"wrote {out / 'standalone.html'}  ({len(single):,} bytes)")
    return html
