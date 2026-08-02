"""Prove all three models use OUR alignment, not one they fetched themselves.

Two independent checks, because "we passed the path" is not "the model used it":

  1. `block_network()` makes every MSA-server entry point raise, so a fallback
     is a hard failure rather than a quiet substitution.
  2. the MSA depth is read back OUT of the built features and compared across
     models and against the a3m on disk.

Depths may legitimately differ from the file if a model caps rows (Boltz-2
subsamples, OF3 caps at 16384). What must not happen is a model reporting a
depth that has nothing to do with our file -- that is the server's alignment.
"""
from __future__ import annotations
import argparse, sys, traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import pi_models  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--a3m", required=True)
    ap.add_argument("--models", default="boltz2,of3,protenix")
    ap.add_argument("--work", required=True)
    args = ap.parse_args()

    blocked = pi_models.block_network()
    print(f"network blocked at: {blocked}\n", flush=True)
    file_depth = pi_models._a3m_depth(args.a3m)
    print(f"a3m on disk: {args.a3m}\n  sequences = {file_depth}\n", flush=True)

    depths, fails = {}, {}
    for name in [m.strip() for m in args.models.split(",")]:
        print(f"--- {name} ---", flush=True)
        try:
            model = pi_models.load(name)
            feats, depth = pi_models.features_for(
                name, model, args.seq, args.a3m, work=Path(args.work) / name)
            depths[name] = depth
            arr = pi_models.msa_array(name, feats)
            print(f"  msa array shape: {None if arr is None else arr.shape}")
            print(f"  MSA depth seen by the model: {depth}", flush=True)
        except pi_models.MSAServerBlocked as e:
            fails[name] = f"FELL BACK TO SERVER: {e}"
            print(f"  {fails[name]}", flush=True)
        except Exception as e:
            fails[name] = f"{type(e).__name__}: {e}"
            traceback.print_exc()

    print(f"\n{'='*64}\nVERDICT\n{'='*64}")
    print(f"  a3m file depth      {file_depth}")
    for n, d in depths.items():
        rel = "= file" if d == file_depth else f"capped/subsampled from {file_depth}"
        print(f"  {n:9s} depth {d:6d}   ({rel})")
    for n, m in fails.items():
        print(f"  {n:9s} FAILED  {m}")

    # Depth equality is the WRONG criterion: each pipeline dedups, filters and
    # caps differently, so identical row counts are not achievable and not
    # required. What is required is that every model's alignment DERIVES FROM
    # OUR FILE -- guaranteed by (a) nothing reaching the server, and (b) a depth
    # consistent with the file rather than with some other alignment.
    ok = len(fails) == 0 and len(depths) == len(args.models.split(","))
    derived = {n: (1 < d <= file_depth + 1) for n, d in depths.items()}
    print()
    for n, d in depths.items():
        print(f"  {n:9s} depth {d:6d}  derives from our file: {derived[n]}")
    all_derived = all(derived.values()) if derived else False
    print(f"\n  no model reached the MSA server            : {ok}")
    print(f"  every depth consistent with our a3m       : {all_derived}")
    if ok and all_derived:
        print("\n  PASS -- alignments are controlled. Differences in depth are each")
        print("  model's own dedup/cap, not a different alignment.")
    else:
        print("\n  NOT SAFE TO LAUNCH -- see above.")
    return 0 if (ok and all_derived) else 1


if __name__ == "__main__":
    sys.exit(main())
