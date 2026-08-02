"""Do the extractors agree? Run every model through the SAME code path.

The point is not the numbers -- it is that one function produces a schema-valid
result for each model, so the analysis layer never needs a per-model branch.
Reports the bin grid explicitly, because a cross-model KL only means something
if the models bin distances the same way.
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
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=50)
    args = ap.parse_args()

    ok, bad = {}, {}
    for name in [m.strip() for m in args.models.split(",")]:
        print(f"\n{'='*66}\n{name}\n{'='*66}", flush=True)
        try:
            model = pi_models.load(name)
            print(f"  loaded {type(model).__name__}", flush=True)
            e = pi_models.run_one(model, args.seq, args.a3m, name=name,
                                  recycles=args.recycles,
                                  sampling_steps=args.sampling_steps)
            print(f"  logits  {e.logits.shape}   bins={e.n_bins}")
            print(f"  grid    {e.centres[0]:.3f} .. {e.centres[-1]:.3f} A "
                  f"(width {e.centres[1]-e.centres[0]:.4f})")
            print(f"  ed      {e.ed.shape}  mean {e.ed.mean():.2f} A")
            print(f"  entropy mean {e.entropy.mean():.4f} nats")
            print(f"  ca      {e.ca.shape}")
            print(f"  plddt   {e.plddt.shape}  mean {e.plddt.mean():.4f}")
            ok[name] = e
        except Exception as ex:
            bad[name] = f"{type(ex).__name__}: {ex}"
            traceback.print_exc()

    print(f"\n{'='*66}\nSCHEMA AGREEMENT\n{'='*66}")
    for n, e in ok.items():
        print(f"  {n:9s} N={e.ed.shape[0]:4d} bins={e.n_bins:3d} "
              f"grid=[{e.centres[0]:.2f},{e.centres[-1]:.2f}] "
              f"plddt_mean={e.plddt.mean():.3f}")
    if len(ok) > 1:
        Ns = {e.ed.shape[0] for e in ok.values()}
        Bs = {e.n_bins for e in ok.values()}
        grids = {(round(float(e.centres[0]), 3), round(float(e.centres[-1]), 3))
                 for e in ok.values()}
        print(f"\n  same N?    {len(Ns)==1}  {Ns}")
        print(f"  same bins? {len(Bs)==1}  {Bs}")
        print(f"  same grid? {len(grids)==1}  {grids}")
        if len(grids) != 1:
            print("  -> cross-model KL/E[d] must NOT be compared as absolutes.")
    for n, m in bad.items():
        print(f"  FAILED {n}: {m}")
    print("\nMULTI SMOKE COMPLETE")


if __name__ == "__main__":
    main()
