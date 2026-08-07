"""Turn the on-disk report into a single self-contained page.

The report links its figures as relative paths and its CSS as a separate file.
Neither survives publication: the artifact runtime serves one document under a
CSP that blocks every external host, so anything not inlined silently
disappears. This rewrites the page with the stylesheet in a <style> block and
every PNG as a data URI, and adds a section rail, which a fourteen-section
document needs on a phone.

Figures are downscaled on the way in. At full resolution the eight PNGs base64
to roughly 2 MB, which is a slow first paint on a remote connection for detail
nobody can see on a laptop; 1500 px wide keeps every axis label legible at
about a third of the weight.
"""

from __future__ import annotations

import argparse
import base64
import io
import re
from pathlib import Path

from PIL import Image

MAX_W = 1500


def embed(png: Path) -> str:
    im = Image.open(png)
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="WEBP", quality=88, method=5)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


CSS = """
:root{
  --ground:#f6f8fb; --panel:#ffffff; --ink:#101720; --muted:#57647a;
  --rule:#dce3ec; --rail:#eef2f7;
  --accent:#2a78d6; --ok:#1a7a52; --warn:#b23a2f; --amber:#96690a; --run:#6146a8;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --serif:Charter,"Iowan Old Style","Source Serif 4",Georgia,"Times New Roman",serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --ground:#0d131c; --panel:#151d29; --ink:#e7ecf4; --muted:#93a2b6;
  --rule:#24303f; --rail:#111926;
  --accent:#65a3f5; --ok:#4dba8b; --warn:#e57b72; --amber:#d3a13c; --run:#a78ae4;
}}
:root[data-theme=dark]{
  --ground:#0d131c; --panel:#151d29; --ink:#e7ecf4; --muted:#93a2b6;
  --rule:#24303f; --rail:#111926;
  --accent:#65a3f5; --ok:#4dba8b; --warn:#e57b72; --amber:#d3a13c; --run:#a78ae4;
}
:root[data-theme=light]{
  --ground:#f6f8fb; --panel:#ffffff; --ink:#101720; --muted:#57647a;
  --rule:#dce3ec; --rail:#eef2f7;
  --accent:#2a78d6; --ok:#1a7a52; --warn:#b23a2f; --amber:#96690a; --run:#6146a8;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.62;-webkit-font-smoothing:antialiased}
.shell{display:grid;grid-template-columns:minmax(0,1fr);gap:0}
@media(min-width:1080px){.shell{grid-template-columns:15.5rem minmax(0,1fr)}}
nav.rail{display:none}
@media(min-width:1080px){
  nav.rail{display:block;position:sticky;top:0;align-self:start;max-height:100vh;
    overflow-y:auto;padding:2.6rem 1.1rem 2rem 1.6rem;background:var(--rail);
    border-right:1px solid var(--rule)}
  nav.rail p{font-family:var(--mono);font-size:.62rem;letter-spacing:.13em;
    text-transform:uppercase;color:var(--muted);margin:0 0 .9rem}
  nav.rail a{display:block;padding:.3rem 0;font-size:.84rem;line-height:1.35;
    color:var(--muted);text-decoration:none;border-left:2px solid transparent;
    padding-left:.7rem;margin-left:-.7rem}
  nav.rail a:hover,nav.rail a:focus-visible{color:var(--accent);
    border-left-color:var(--accent)}
}
main{padding:2.6rem 1.4rem 5rem;max-width:74rem;margin:0 auto;width:100%}
@media(min-width:760px){main{padding:3.4rem 2.6rem 6rem}}
.masthead{border-bottom:2px solid var(--ink);padding-bottom:1.6rem;margin-bottom:2.4rem}
.eyebrow{font-family:var(--mono);font-size:.66rem;letter-spacing:.15em;
  text-transform:uppercase;color:var(--accent);margin:0 0 .8rem}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(1.9rem,4.4vw,2.9rem);
  line-height:1.12;margin:0 0 .9rem;text-wrap:balance;letter-spacing:-.012em}
.lede{font-size:1.06rem;color:var(--muted);max-width:66ch;margin:0}
h2{font-family:var(--serif);font-weight:600;font-size:clamp(1.3rem,2.6vw,1.72rem);
  line-height:1.2;margin:3.4rem 0 1.1rem;text-wrap:balance;letter-spacing:-.008em;
  padding-top:1.4rem;border-top:1px solid var(--rule)}
h3{font-family:var(--sans);font-weight:650;font-size:1rem;margin:0;line-height:1.35}
p{max-width:70ch}
section>p,.card p{margin:.85rem 0}
a{color:var(--accent)}
b,strong{font-weight:650}
code{font-family:var(--mono);font-size:.88em;background:var(--rail);
  padding:.1em .35em;border-radius:3px;border:1px solid var(--rule)}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:5px;
  padding:1.15rem 1.35rem;margin:1.3rem 0}
.card.ok{border-left:3px solid var(--ok)}
.card.warn{border-left:3px solid var(--warn)}
.card.amber{border-left:3px solid var(--amber)}
.card.run{border-left:3px solid var(--run)}
.row{display:flex;align-items:baseline;gap:.7rem;flex-wrap:wrap;margin-bottom:.5rem}
.chip{font-family:var(--mono);font-size:.6rem;letter-spacing:.12em;
  text-transform:uppercase;padding:.22em .5em;border:1px solid currentColor;
  border-radius:3px;white-space:nowrap}
.c-ok{color:var(--ok)}.c-warn{color:var(--warn)}.c-amber{color:var(--amber)}
.c-run{color:var(--run)}
ul,ol{max-width:70ch;padding-left:1.15rem}
li{margin:.4rem 0}
.scroll{overflow-x:auto;margin:1.15rem 0;border:1px solid var(--rule);
  border-radius:5px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{padding:.5rem .8rem;text-align:left;border-bottom:1px solid var(--rule);
  white-space:nowrap}
th{font-family:var(--mono);font-size:.64rem;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted);font-weight:500;
  background:var(--rail)}
tr:last-child td{border-bottom:none}
td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;
  font-size:.84rem;text-align:right}
.ci{font-family:var(--mono);font-size:.8em;color:var(--muted);
  font-variant-numeric:tabular-nums}
figure{margin:1.9rem 0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--rule);
  border-radius:5px;background:#fcfcfb}
figcaption{font-size:.82rem;color:var(--muted);margin-top:.6rem;max-width:70ch}
footer{margin-top:4rem;padding-top:1.4rem;border-top:1px solid var(--rule);
  font-size:.82rem;color:var(--muted)}
footer p{max-width:74ch}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
@media(prefers-reduced-motion:reduce){*{animation:none!important;
  transition:none!important}}
"""


def main():
    ap = argparse.ArgumentParser()
    W = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files")
    ap.add_argument("--report", default=str(W / "report_svd/index.html"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = Path(a.report)
    html = src.read_text()
    body = html.split("<body>", 1)[1].rsplit("</body>", 1)[0]
    body = body.replace('<div class=wrap>', '').rstrip()
    if body.endswith("</div>"):
        body = body[: -len("</div>")]

    # inline every figure
    n = 0
    def sub(m):
        nonlocal n
        p = src.parent / m.group(1)
        if not p.exists():
            return m.group(0)
        n += 1
        return f'src="{embed(p)}"'
    body = re.sub(r'src="(figures/[^"]+)"', sub, body)

    # section rail from the existing headings
    toc = []
    for sid, h2 in re.findall(r'<section id=([\w-]+)>\s*<h2>(.*?)</h2>', body,
                              re.S):
        toc.append(f'<a href="#{sid}">{re.sub("<[^>]+>", "", h2).strip()}</a>')
    rail = ('<nav class="rail" aria-label="Sections"><p>Contents</p>'
            + "".join(toc) + "</nav>")

    # the on-disk report opens with its own header; keep it as the masthead
    body = body.replace("<header class=head>", '<header class="masthead">', 1)
    body = body.replace("<span class=eyebrow>", '<p class="eyebrow">', 1)
    body = body.replace("</span>\n  <h1>", "</p>\n  <h1>", 1)
    body = body.replace("<p class=lede>", '<p class="lede">', 1)

    out = (f"<title>The mutation subspace — Boltz-2 Pairformer</title>\n"
           f"<style>{CSS}</style>\n"
           f'<div class="shell">{rail}<main>{body}</main></div>\n')
    Path(a.out).write_text(out)
    kb = len(out) / 1024
    print(f"wrote {a.out}  ({kb:.0f} KB, {n} figures inlined, {len(toc)} sections)")


if __name__ == "__main__":
    main()
