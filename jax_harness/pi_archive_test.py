"""Does the write seam actually refuse what it claims to refuse?

A guard is only worth what it does on the bad path, so every check here asserts
a REFUSAL or a round-trip, not that the happy path runs.

The audit is included because its two bugs were both semantic rather than
mechanical -- it parsed, ran, and printed a reasonable table while calling every
capture archive unused (12 GB of proposed deletions), then in the other
direction while calling every archive referenced. Neither would have been caught
by a smoke test. The glob cases below are the ones that were wrong.

  sbatch analysis.sbatch pi_archive_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_archive  # noqa: E402
import pi_archive_audit as audit  # noqa: E402
import pi_protocol  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(name)


def proto():
    return pi_protocol.protocol(
        script="pi_archive_test.py", design="synthetic",
        layer=pi_protocol.layers("final"),
        features=pi_protocol.features("fake", 8), source="none", n_assays=1)


def main():
    print("\nWRITE SEAM\n")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        try:
            pi_archive.write_result(td / "a.json", {"x": 1})
            check("write_result without a protocol is a TypeError", False)
        except TypeError:
            check("write_result without a protocol is a TypeError", True)

        for bad in ({}, None, "kl_glob"):
            try:
                pi_archive.write_result(td / "b.json", {"x": 1}, protocol=bad)
                check(f"write_result rejects protocol={bad!r}", False)
                break
            except (ValueError, TypeError):
                pass
        else:
            check("write_result rejects an empty or non-dict protocol", True)

        pi_archive.write_result(td / "c.json", {"x": 1.5}, protocol=proto())
        d = json.loads((td / "c.json").read_text())
        check("protocol lands at a fixed path", isinstance(d.get("protocol"), dict))
        check("argv is recorded",
              isinstance(d.get("provenance", {}).get("argv"), list),
              f"argv[0]={d['provenance']['argv'][0].split('/')[-1]}")
        check("payload survives", d.get("x") == 1.5)

        # a caller that already had its own block keeps its extra keys
        pi_archive.write_result(td / "d.json", {"protocol": {"mine": 1}},
                                protocol=proto())
        d = json.loads((td / "d.json").read_text())
        check("an existing block is merged, not dropped",
              d["protocol"].get("mine") == 1 and "script" in d["protocol"])

        # npz round-trip
        pi_archive.write_npz(td / "e.npz", {"V": np.zeros((4, 8))},
                             protocol=proto())
        m = pi_archive.npz_meta(td / "e.npz")
        check("npz carries its block and array shapes",
              m is not None and m["arrays"]["V"]["shape"] == [4, 8])
        np.savez(td / "plain.npz", V=np.zeros((2, 2)))
        check("a pre-convention npz reads as None, not as a fake block",
              pi_archive.npz_meta(td / "plain.npz") is None)

        print("\nREAD GUARD\n")
        (td / "bare.json").write_text(json.dumps({"x": 1}))
        _, has = pi_archive.read_result(td / "bare.json")
        check("an unlabelled archive loads and is flagged", has is False)
        try:
            pi_archive.read_result(td / "bare.json", quoted=True)
            check("an unlabelled QUOTED input raises", False)
        except ValueError:
            check("an unlabelled QUOTED input raises", True)

        print("\nRECONSTRUCTION\n")
        r = pi_archive.reconstruct_provenance(td / "e.npz")
        check("reconstruction is marked inferred", r.get("inferred") is True)
        check("reconstruction measures array shapes",
              r["arrays"]["V"]["shape"] == [4, 8])
        check("reconstruction never invents an invocation",
              not any(k in json.dumps(r) for k in ("argv", "--glob", "--k")))
        pi_archive.stamp_reconstructed(td / "bare.json", dry_run=False)
        d = json.loads((td / "bare.json").read_text())
        check("reconstruction never writes under 'protocol'",
              "protocol" not in d and "provenance_reconstructed" in d)

    print("\nAUDIT REFERENCING  (both historical bugs)\n")
    with tempfile.TemporaryDirectory() as cd:
        cd = Path(cd)
        (cd / "a.py").write_text('ap.add_argument("--glob", default=R + '
                                 '"gym3_*.npz")\n')
        (cd / "b.py").write_text('for f in glob.glob(d + "/*.npz"):\n    pass\n')
        blobs, globs, discarded = audit.build_reference_index([cd])
        check("a capture glob is usable", "gym3_*.npz" in globs)
        check("a bare *.npz is discarded", "*.npz" in discarded)
        # bug 1: literal-name matching called every capture unused
        hit = audit.referenced_by("gym3_ARGR_ECOLI_Tsuboyama_2023_1AOY.npz",
                                  "gym3_ARGR_ECOLI_Tsuboyama_2023_1AOY",
                                  blobs, globs)
        check("a capture referenced only by glob counts as referenced",
              bool(hit), f"<- {hit}")
        # bug 2: admitting *.npz made everything referenced
        miss = audit.referenced_by("totally_unrelated_thing.npz",
                                   "totally_unrelated_thing", blobs, globs)
        check("an unrelated archive does NOT count as referenced", not miss)

    print("\n" + "=" * 58)
    if FAIL:
        print(f"FAILED: {len(FAIL)}\n  " + "\n  ".join(FAIL))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
