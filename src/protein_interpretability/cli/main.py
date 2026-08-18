"""`pi` -- reproduce, verify, inspect.

    pi reproduce RESULT.json --out DIR [--submit]
    pi verify OLD.json NEW.json [--name STEM]
    pi inspect FILE.py [FILE.py ...]

Nothing here imports a model backend, numpy or jax, so all three run on a login
node. `inspect` is the one that enforces that property for other files.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

from protein_interpretability.cli.compare import compare, verdict

REPO = Path(__file__).resolve().parents[3]
HARNESS = REPO / "jax_harness"

BACKENDS = {"joltz", "mosaic", "boltz", "torch", "transformers", "jopenfold3",
            "equinox", "esm"}


# ---- reproduce -------------------------------------------------------------

def _argv_of(result: Path) -> list[str]:
    doc = json.loads(result.read_text())
    prov = doc.get("provenance") or {}
    argv = prov.get("argv")
    if not argv:
        raise SystemExit(
            f"{result} carries no provenance.argv, so the command that produced "
            "it is not recorded and cannot be replayed. Results written through "
            "the archive seam always carry one.")
    return list(argv)


def cmd_reproduce(a) -> int:
    result = Path(a.result)
    argv = _argv_of(result)
    script = Path(argv[0]).name
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = [script] + argv[1:]
    if "--out" in argv:
        argv[argv.index("--out") + 1] = str(out_dir / result.name)
    else:
        argv += ["--out", str(out_dir / result.name)]

    submitter = HARNESS / ("checkout.sbatch" if a.checkout else "analysis.sbatch")
    cmd = ["sbatch", "-J", f"repro_{result.stem}", str(submitter)] + argv

    if not a.submit:
        print(" ".join(cmd))
        print(f"\n# {result.name} was produced by {script}", file=sys.stderr)
        print("# add --submit to run it", file=sys.stderr)
        return 0

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr.strip(), file=sys.stderr)
        return 1
    print(r.stdout.strip())
    return 0


# ---- verify ----------------------------------------------------------------

def cmd_verify(a) -> int:
    old_p, new_p = Path(a.old), Path(a.new)
    name = a.name or new_p.stem
    d = compare(json.loads(old_p.read_text()), json.loads(new_p.read_text()))
    ok, why = verdict(name, d)
    print(f"{name}: {'PASS' if ok else 'FAIL'}  "
          f"max_abs {d.max_abs:.3e} over {d.n_numbers} numbers"
          + (f" at {d.max_abs_at}" if d.max_abs else ""))
    print(f"  {why}")
    for issue in d.issues:
        print(f"  {issue}")
    return 0 if ok else 1


# ---- inspect ---------------------------------------------------------------

def _imports(tree: ast.AST) -> set[str]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


def _toplevel_imports(tree: ast.Module) -> set[str]:
    """Imports at module scope only -- the ones that fire on import."""
    out = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module.split(".")[0])
    return out


def cmd_inspect(a) -> int:
    failed = 0
    for path_s in a.files:
        path = Path(path_s)
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:
            print(f"{path.name}: SYNTAX ERROR {exc}")
            failed += 1
            continue

        notes = []
        top = _toplevel_imports(tree) & BACKENDS
        if top:
            notes.append(
                f"imports {sorted(top)} at module scope -- this file cannot be "
                "imported without a model environment, so it cannot be "
                "inspected on a login node")

        src = path.read_text()
        writes_result = "write_result" in src
        # Three legitimate idioms: a module-level PROTOCOL constant, or the
        # block built inline from a protocol() call reached as `pi_protocol.`,
        # `P.` or bare. Matching the text of the first two spellings missed the
        # third and flagged a script whose protocol block is complete enough
        # that protocol() itself would have raised without it -- the same
        # false-positive shape as the version before it, one import alias later.
        # Hence the AST: any call to something *named* protocol counts.
        has_protocol = any(
            isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == "PROTOCOL" for t in n.targets)
            for n in tree.body
        ) or any(
            isinstance(n, ast.Call)
            and (getattr(n.func, "attr", None) == "protocol"
                 or getattr(n.func, "id", None) == "protocol")
            for n in ast.walk(tree)
        )
        if writes_result and not has_protocol:
            notes.append(
                "writes a result but declares no module-level PROTOCOL; the "
                "archive seam requires one, so this fails at the end of a job "
                "rather than at the start")

        if notes:
            failed += 1
            print(f"{path.name}:")
            for n in notes:
                print(f"  - {n}")
        elif a.verbose:
            print(f"{path.name}: ok")

    if not failed:
        print(f"{len(a.files)} file(s) inspected, nothing to report")
    return 1 if failed else 0


# ---- cohort ----------------------------------------------------------------

def cmd_cohort(a) -> int:
    # Imported here, not at module scope: `pi verify` and `pi inspect` must stay
    # runnable from a checkout with nothing configured.
    from protein_interpretability.collection import Cohort, CohortError

    if not a.name:
        for name in Cohort.available():
            c = Cohort.load(name)
            print(f"{name:22s} {len(c):3d} assays  {c.description[:60]}")
        return 0

    c = Cohort.load(a.name)
    print(f"{c.name}: {len(c)} assays")
    print(f"  {c.description}")
    if a.list:
        for assay in c:
            print(f"  {assay.id:34s} len={assay.wt_length or '?':>4} "
                  f"variants={assay.n_single_variants or '?':>5} "
                  f"msa_rows={assay.msa_rows or '?':>6}")
    if a.verify:
        try:
            c.verify(checksums=not a.fast)
        except CohortError as exc:
            print(f"\n{exc}")
            return 1
        how = "exist" if a.fast else "hash-match the manifest"
        print(f"\nverified: every input {how}")
    return 0


# ---- entry point -----------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pi", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reproduce", help="re-run an archived result from its "
                                         "recorded argv")
    r.add_argument("result")
    r.add_argument("--out", required=True)
    r.add_argument("--submit", action="store_true",
                   help="actually sbatch it; without this the command is printed")
    r.add_argument("--checkout", action="store_true",
                   help="run the git checkout rather than the deployed mirror "
                        "-- required for anything that touches the library")
    r.set_defaults(func=cmd_reproduce)

    v = sub.add_parser("verify", help="diff two results with the known bands")
    v.add_argument("old")
    v.add_argument("new")
    v.add_argument("--name", help="producer stem, if the filename is not it")
    v.set_defaults(func=cmd_verify)

    c = sub.add_parser("cohort", help="list cohorts, or verify one against disk")
    c.add_argument("name", nargs="?", help="omit to list every cohort")
    c.add_argument("--list", action="store_true", help="show each assay")
    c.add_argument("--verify", action="store_true",
                   help="check every input still hashes to its manifest value")
    c.add_argument("--fast", action="store_true",
                   help="with --verify, check existence only and skip hashing")
    c.set_defaults(func=cmd_cohort)

    i = sub.add_parser("inspect", help="static checks before submitting a job")
    i.add_argument("files", nargs="+")
    i.add_argument("-v", "--verbose", action="store_true")
    i.set_defaults(func=cmd_inspect)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
