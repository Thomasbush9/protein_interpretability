"""What is in runs/, what reads it, and what carries its own protocol.

Three questions the project cannot currently answer about its own outputs:

  labelled     does this archive say what it is comparable to?
  referenced   does any builder, launcher or script actually read it?
  costly       what is it occupying, against a lab quota near its ceiling?

Referenced-ness is computed, not guessed: every archive basename, its stem, AND
every glob pattern found in the code are matched against each file. The globs
are not optional -- capture archives are referenced only ever by glob, so a
literal-name index calls all 12 GB of them unused.

NOTHING IS MOVED OR DELETED BY DEFAULT. `--attic DIR` moves unreferenced files
and writes an index beside them; the move is reversible and the index records
where each came from. Unreferenced is not the same as worthless -- a replicate
series like probe_seed*.json is unreferenced by construction and may still be
wanted -- so the decision stays with a person.

  sbatch analysis.sbatch pi_archive_audit.py --runs $R/runs --out $R/runs/audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from fnmatch import fnmatch
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pi_archive  # noqa: E402

HERE = Path(__file__).resolve().parent


GLOB_RE = re.compile(r"[A-Za-z0-9_./*?\[\]-]*\*[A-Za-z0-9_./*?\[\]-]*\.(?:npz|json)")
# Literal characters required before the first wildcard. Both failures of this
# function were about this number being implicitly 0 or infinite: with no
# globs, every capture read as unused (12 GB of attic candidates); with bare
# "*.npz" admitted, every file read as referenced and the audit said nothing.
MIN_GLOB_PREFIX = 3


def build_reference_index(code_dirs):
    """Literal names AND glob patterns, because captures are only ever globbed.

    THE FIRST VERSION OF THIS FUNCTION MATCHED LITERAL NAMES ONLY. Every
    capture archive -- gym2_*, gym2s_*, gym3_*, xm_* -- is referenced by a glob
    in an argparse default or a documented invocation and never by its own
    filename, so all 12 GB of them scored as unreferenced and were listed as
    attic candidates. Moving them would have cost roughly four GPU-hours each
    to regenerate, and the tool would have reported success.

    A tool that decides what to delete has to be judged by what it does when it
    is wrong, so globs are collected here and matched with fnmatch.
    """
    blobs, globs, discarded = [], {}, set()
    for d in code_dirs:
        for p in Path(d).rglob("*"):
            if p.suffix in (".py", ".sh", ".sbatch", ".md") and p.is_file():
                try:
                    txt = p.read_text(errors="ignore")
                except Exception:
                    continue
                blobs.append((p.name, txt))
                for g in GLOB_RE.findall(txt):
                    pat = Path(g).name
                    # A pattern with no meaningful literal prefix -- "*.npz",
                    # "_*.npz" -- comes from a generic glob.glob(dir + "/*.npz")
                    # and matches every archive in the directory, which makes
                    # referencing vacuous. Requiring a real prefix is what
                    # separates "analyze_heldout reads gym3_*" from "some
                    # script lists the directory".
                    if len(pat.split("*")[0]) < MIN_GLOB_PREFIX:
                        discarded.add(pat)
                        continue
                    globs.setdefault(pat, set()).add(p.name)
    return blobs, globs, discarded


def referenced_by(name, stem, blobs, globs):
    hits = {fn for fn, txt in blobs if name in txt or stem in txt}
    for pat, srcs in globs.items():
        if fnmatch(name, pat):
            hits |= srcs
    return sorted(hits)


def human(n):
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024 or u == "T":
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--code", nargs="+", default=[str(HERE)])
    ap.add_argument("--attic", default="", help="move unreferenced files here; "
                                                "omitted means report only")
    ap.add_argument("--apply", action="store_true",
                    help="actually move; --attic alone only lists")
    ap.add_argument("--max-bytes", type=int, default=50 * 1024 ** 2,
                    help="refuse to move anything larger; captures are big and "
                         "expensive, and a wrong reference index looks exactly "
                         "like an unused file")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    blobs, globs, discarded = build_reference_index(a.code)
    print(f"searched {len(blobs)} code files; {len(globs)} usable glob patterns "
          f"({', '.join(sorted(globs)[:5])}...)")
    print(f"  discarded {len(discarded)} too-general patterns "
          f"({', '.join(sorted(discarded))}) -- they match every archive and "
          f"would make\n  referencing vacuous\n")

    rows, tot = [], defaultdict(int)
    for p in sorted(Path(a.runs).iterdir()):
        if p.suffix not in (".json", ".npz") or not p.is_file():
            continue
        refs = referenced_by(p.name, p.stem, blobs, globs)
        labelled = None
        if p.suffix == ".json":
            try:
                d = json.loads(p.read_text())
                labelled = isinstance(d, dict) and bool(d.get("protocol"))
            except Exception:
                labelled = False
        else:
            try:
                labelled = pi_archive.npz_meta(p) is not None
            except Exception:
                labelled = False
        sz = p.stat().st_size
        rows.append({"name": p.name, "bytes": sz, "labelled": bool(labelled),
                     "referenced": bool(refs), "referenced_by": refs[:6]})
        tot[(bool(labelled), bool(refs))] += sz
        tot["all"] += sz

    n = len(rows)
    lab = sum(r["labelled"] for r in rows)
    ref = sum(r["referenced"] for r in rows)
    print(f"{n} archives, {human(tot['all'])} total")
    print(f"  labelled with a protocol block : {lab:4d} / {n}")
    print(f"  referenced by any code file    : {ref:4d} / {n}\n")

    print(f"  {'':26s}{'referenced':>14s}{'unreferenced':>15s}")
    for L in (True, False):
        cells = "".join(f"{human(tot[(L, R)]):>15s}" for R in (True, False))
        cnt = [sum(1 for r in rows if r['labelled'] == L and r['referenced'] == R)
               for R in (True, False)]
        print(f"  {'labelled' if L else 'unlabelled':<12s}"
              f"{f'({cnt[0]} / {cnt[1]} files)':>14s}{cells}")

    quoted = [r for r in rows if r["referenced"] and not r["labelled"]]
    print(f"\n  REFERENCED BUT UNLABELLED ({len(quoted)}) -- these are the ones a\n"
          f"  page may already be quoting without saying what they are:\n")
    for r in sorted(quoted, key=lambda r: -r["bytes"])[:20]:
        print(f"    {r['name']:44s} {human(r['bytes']):>8s}  "
              f"<- {', '.join(r['referenced_by'][:3])}")

    unref = [r for r in rows if not r["referenced"]]
    print(f"\n  UNREFERENCED ({len(unref)}, {human(sum(r['bytes'] for r in unref))}) "
          f"-- candidates for the attic, NOT deleted here")
    for r in sorted(unref, key=lambda r: -r["bytes"])[:12]:
        print(f"    {r['name']:44s} {human(r['bytes']):>8s}")

    if a.attic and not a.apply:
        print(f"\n  --attic given without --apply: nothing moved. {len(unref)} "
              f"files ({human(sum(r['bytes'] for r in unref))}) would go to "
              f"{a.attic}.\n  Review the list above, then rerun with --apply.")
    elif a.attic:
        big = [r for r in unref if r["bytes"] > a.max_bytes]
        if big:
            raise SystemExit(
                f"REFUSING to move {len(big)} file(s) over "
                f"{human(a.max_bytes)}: {', '.join(r['name'] for r in big[:4])}"
                f"...\nFiles this size are captures, which cost GPU-hours to "
                f"regenerate. The first version of this audit called every "
                f"capture unreferenced because they are only ever globbed; if "
                f"a capture is in this list the reference index is wrong "
                f"again, not the archive. Raise --max-bytes only after "
                f"checking that.")
        dest = Path(a.attic)
        dest.mkdir(parents=True, exist_ok=True)
        moved = []
        for r in unref:
            src = Path(a.runs) / r["name"]
            shutil.move(str(src), str(dest / r["name"]))
            moved.append({"name": r["name"], "from": str(src),
                          "bytes": r["bytes"]})
        (dest / "ATTIC_INDEX.json").write_text(json.dumps(
            {"moved": moved, "note": "Unreferenced at audit time. Reversible: "
                                     "each entry records its original path.",
             "provenance": pi_archive.run_provenance()}, indent=2))
        print(f"\n  moved {len(moved)} files to {dest} "
              f"({human(sum(m['bytes'] for m in moved))} freed); "
              f"ATTIC_INDEX.json records every origin")

    if a.out:
        Path(a.out).write_text(json.dumps(
            {"summary": {"n": n, "labelled": lab, "referenced": ref,
                         "bytes_total": tot["all"]},
             "archives": rows, "provenance": pi_archive.run_provenance()},
            indent=2, default=float))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
