"""How expensive does the trunk get on a real-sized protein?

The three non-Tsuboyama stability assays in ProteinGym are the only test of
whether "internal beats output" survives outside 37-72 residue mini-domains --
and they are 212, 245 and 403 residues. Before committing a sweep to the queue
it is worth knowing whether 403 is a longer run or an impossible one, because
the Pairformer's triangle operations are O(N^3) in the number of tokens and
(403/63)^3 is 262.

This times the Pairformer stack alone, on random z of each size, at the same
64 layers and dtype the real runs use. The stack is not the whole cost -- the
MSA module, diffusion and confidence heads are extra -- but it is the term that
scales worst, so it bounds the shape of the problem.

Reported per size: seconds per full 64-layer pass, the implied exponent against
the 63-residue reference, and the peak pair-tensor footprint. A measured
exponent matters more than the nominal N^3: attention kernels are not
cubic-bound at these sizes, so guessing from the algorithm overstates the cost.

  sbatch analysis.sbatch probe_scale.py --out ../runs/scale_probe.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="63,128,212,245,403")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    sizes = [int(s) for s in a.sizes.split(",")]
    print(f"jax devices: {jax.devices()}\n", flush=True)
    model = pi.load_model(subsample_msa=False)
    pf = model.pairformer_module
    L = pf.stacked_parameters.transition_z.fc1.weight.shape[0]
    dim = pf.stacked_parameters.transition_z.fc1.weight.shape[-1]
    s_dim = model.s_init.weight.shape[0]
    print(f"Pairformer: {L} layers, pair dim {dim}, single dim {s_dim}\n")

    def run_once(N, key):
        s = jnp.zeros((1, N, s_dim), jnp.float32)
        z = jax.random.normal(key, (1, N, N, dim), jnp.float32) * 0.1
        mask = jnp.ones((1, N), jnp.float32)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        out = pi.pairformer_capture(pf, s, z, mask, pair_mask,
                                    key=key, deterministic=True)
        return out[1]

    rows = []
    base = None
    print(f"  {'N':>5s} {'sec/pass':>9s} {'pair MB':>9s} {'vs N=63':>8s} "
          f"{'exponent':>9s}")
    for N in sizes:
        key = jax.random.key(0)
        try:
            f = jax.jit(lambda k, n=N: run_once(n, k))
            r = f(key)
            r.block_until_ready()                      # compile, not timed
            ts = []
            for i in range(a.reps):
                t0 = time.time()
                r = f(jax.random.fold_in(key, i))
                r.block_until_ready()
                ts.append(time.time() - t0)
            t = float(np.median(ts))
        except Exception as e:                          # OOM is a real answer
            print(f"  {N:5d}   FAILED: {type(e).__name__}: {str(e)[:70]}")
            rows.append({"N": N, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        mb = N * N * dim * 4 / 1e6
        if base is None:
            base, base_n = t, N
        ratio = t / base
        expo = (np.log(ratio) / np.log(N / base_n)) if N != base_n else float("nan")
        print(f"  {N:5d} {t:9.3f} {mb:9.1f} {ratio:8.1f} {expo:9.2f}")
        rows.append({"N": N, "sec": t, "pair_mb": mb, "ratio": ratio,
                     "exponent": None if N == base_n else float(expo)})

    ok = [r for r in rows if "sec" in r]
    print("\nExtrapolation to a full exp_gym2 variant is NOT this number: a\n"
          "variant runs the trunk twice (wild type and mutant), three recycles\n"
          "each, plus the MSA module, diffusion and confidence heads. Use the\n"
          "ratio column to scale a measured 63-residue variant cost.\n")
    if len(ok) >= 2:
        r63 = next((r for r in ok if r["N"] == 63), ok[0])
        for r in ok:
            if r is r63:
                continue
            # exp_gym2 on RCRO: 250 variants in 12417 s = 49.7 s/variant.
            est = 49.7 * r["ratio"]
            print(f"  N={r['N']:4d}: a 250-variant sweep would take roughly "
                  f"{est*250/3600:6.1f} h at {est:5.0f} s/variant")

    Path(a.out).write_text(json.dumps({"layers": L, "dim": dim, "rows": rows},
                                      indent=2, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
