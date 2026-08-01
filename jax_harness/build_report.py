"""Concatenate the report into one printable file, and pack the bundle.

The four pages are written and edited separately; this stitches them into
report/all_in_one.html (for reading or printing in one pass) and produces
report.tar.gz. Rerun after editing any page -- all_in_one.html is derived, never
edited by hand.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPORT = Path(__file__).resolve().parent.parent / "report"
PAGES = [("index.html", "Overview"), ("methods.html", "Methods"),
         ("results.html", "Results"), ("corrections.html", "Corrections"),
         ("open.html", "Open questions")]


def body_of(html: str) -> str:
    m = re.search(r"<body[^>]*>(.*)</body>", html, re.S)
    return (m.group(1) if m else html).strip()


def main():
    css = (REPORT / "style.css").read_text()
    parts = []
    for fn, label in PAGES:
        b = body_of((REPORT / fn).read_text())
        # inter-page navigation is meaningless in a single file
        b = re.sub(r"<nav.*?</nav>", "", b, flags=re.S)
        b = re.sub(r'href="(index|methods|results|corrections|open)\.html([^"]*)"',
                   lambda m: f'href="#{m.group(1)}"', b)
        parts.append(f'<section class="page" id="{fn[:-5]}">\n'
                     f'<hr class="pagebreak">\n{b}\n</section>')
    out = (f"<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
           f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
           f"<title>Boltz-2 Pairformer under mutation &mdash; full report</title>\n"
           f"<style>\n{css}\n"
           f"@media print {{ .pagebreak {{ page-break-before: always; }} }}\n"
           f".pagebreak {{ border: 0; }}\n</style></head><body>\n"
           + "\n".join(parts) + "\n</body></html>\n")
    (REPORT / "all_in_one.html").write_text(out)
    print(f"all_in_one.html  {len(out) / 1024:.0f} KB  ({len(PAGES)} pages)")

    figs = sorted((REPORT / "figures").iterdir())
    print(f"figures/         {len(figs)} files")
    tar = REPORT.parent / "report.tar.gz"
    subprocess.run(["tar", "czf", str(tar), "-C", str(REPORT.parent), "report"], check=True)
    print(f"{tar}  {tar.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
