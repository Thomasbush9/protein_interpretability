"""Check the recorded model capabilities against a real loaded model.

The registry is a static table, and a static table describing a model is exactly
what stops being true when a wrapper is upgraded. `verify_against_model` exists
so that drift is detectable; this is the job that actually calls it, against
every model the registry claims to know.

It is the only place a backend is imported for this purpose. The registry itself
stays login-node safe, which is what lets a capture be planned before a GPU is
reached.

    sbatch checkout.sbatch probe_capabilities.py --out $W/runs/capabilities.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pi_archive  # noqa: E402
import pi_models  # noqa: E402
from protein_interpretability.collection import capabilities as caps  # noqa: E402

PROTOCOL = dict(
    script="probe_capabilities.py",
    design="load each model and compare it against the recorded capability table",
    layer={"which": "n/a"},
    features={"name": "model metadata", "width": 0},
    source="mosaic wrappers via pi_models.load",
    n_assays=0,
    note="a mismatch means the table has gone stale, not that the model is wrong",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="boltz2,of3,protenix")
    ap.add_argument("--msa", default="full", choices=pi_models.MSA_REGIMES)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    result, failures, unverified = {}, [], []
    for name in [m.strip() for m in a.models.split(",") if m.strip()]:
        declared = caps.capabilities(name)
        entry = {"declared": {
            "n_trunk_blocks": declared.n_trunk_blocks,
            "pair_width": declared.pair_width,
            "plddt_granularity": declared.plddt_granularity,
            "subsamples_msa_by_default": declared.subsamples_msa_by_default,
        }}
        try:
            wrapper = pi_models.load(name, msa=a.msa)
        except Exception as exc:                       # a wrapper that will not build
            entry["load_error"] = repr(exc)[:300]
            result[name] = entry
            failures.append(f"{name}: could not load")
            print(f"{name:10s} LOAD FAILED {exc!r}"[:160], flush=True)
            continue

        try:
            observed = caps.verify_against_model(name, wrapper)
            entry["observed"] = observed
            # "Nothing contradicted me" is not agreement. The first run of this
            # probe reported of3 and protenix as agreeing having read nothing
            # from either, which is the vacuous pass the rest of the suite is
            # written to avoid.
            if observed["checked"]:
                entry["status"] = ("verified" if not observed["unverified"]
                                   else "partially verified")
            else:
                entry["status"] = "NOT VERIFIED -- nothing readable on this wrapper"
                unverified.append(name)
            print(f"{name:10s} {entry['status']}: checked={observed['checked']} "
                  f"unverified={observed['unverified']}", flush=True)
        except caps.CapabilityError as exc:
            entry["status"] = "DRIFT"
            entry["drift"] = str(exc)
            failures.append(f"{name}: {exc}")
            print(f"{name:10s} DRIFT {exc}", flush=True)
        result[name] = entry
        del wrapper

    result["contradictions"] = failures
    result["unverified_models"] = unverified
    result["all_verified"] = not failures and not unverified
    pi_archive.write_result(Path(a.out), result, protocol=PROTOCOL)
    print("\n" + json.dumps({k: v for k, v in result.items()}, indent=1)[:1500])
    # A model whose table could not be checked is not a pass.
    return 1 if (failures or unverified) else 0


if __name__ == "__main__":
    raise SystemExit(main())
