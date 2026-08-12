"""Match the decoding protocols exactly, so the headline numbers stop needing prose.

Three "internal, all 128 dimensions" figures circulate in this project and they
differ for reasons that are protocol, not disagreement:

  0.758   leave-one-assay-out, final layer          (analyze_transfer, k=128)
  0.731   within-assay splits, MEAN OF LAST 8       (analyze_svd, plotted curve)
  0.703   within-assay splits, final layer only     (this script)

Only the last two are directly comparable to each other, and only the first and
last share a layer convention. `analyze_svd` pools its plotted curve over the
last eight layers -- deliberately, so that no single layer is chosen by held-out
performance -- but that averaging is itself worth something, and the report was
comparing an 8-layer average against single-layer numbers without saying so.

The per-layer surface is already archived in `svd_dz_v3.npz` as
`curve_pc_var_centered` with axes (assay, seed, k, layer), so nothing needs
re-fitting; this only reduces it at matched settings and writes the result where
a report builder can read it. Every number a page states has to come from a
file, and this one did not exist.

The reduction runs on the accelerator. It is small -- a few tens of thousands of
elements -- and is not the reason this job needs a GPU; there is simply no CPU
partition on this account, so anything submitted lands on a GPU node and should
at least use it rather than idling beside it.

  sbatch analysis.sbatch analyze_layer_match.py --npz $R/svd_dz_v3.npz \
      --out $R/layer_match.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).parent))
import pi_archive  # noqa: E402
import pi_protocol  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--curve", default="curve_pc_var_centered",
                    help="components, variance-ordered; the plotted curve")
    ap.add_argument("--last", type=int, default=8,
                    help="width of the layer window analyze_svd pools over")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    print(f"jax devices: {jax.devices()}\n", flush=True)
    d = np.load(a.npz, allow_pickle=True)
    ks = [int(k) for k in d["ks"]]
    C = jnp.asarray(np.asarray(d[a.curve], np.float64))   # (assay, seed, k, layer)
    n_assay, n_seed, n_k, n_layer = C.shape
    print(f"{a.curve}: {n_assay} assays x {n_seed} seeds x {n_k} ks x "
          f"{n_layer} layers")

    @jax.jit
    def reduce(c):
        # Seeds first, then the layer convention. Averaging over seeds before
        # layers matters: the reverse weights a noisy seed at one layer as if it
        # were a separate layer.
        per_seed = jnp.nanmean(c, axis=1)                  # (assay, k, layer)
        final = per_seed[..., -1]                          # (assay, k)
        window = jnp.nanmean(per_seed[..., -a.last:], axis=-1)
        return final, window, jnp.nanmean(final, 0), jnp.nanmean(window, 0)

    final, window, m_final, m_window = reduce(C)
    final, window = np.asarray(final), np.asarray(window)
    m_final, m_window = np.asarray(m_final), np.asarray(m_window)

    print(f"\nwithin-assay decodability, components (variance-ordered)\n")
    print(f"  {'k':>5s} {'final layer':>13s} {f'mean last {a.last}':>15s} "
          f"{'averaging gain':>16s}")
    for i, k in enumerate(ks):
        print(f"  {k:5d} {m_final[i]:13.3f} {m_window[i]:15.3f} "
              f"{m_window[i]-m_final[i]:+16.3f}")

    i128 = ks.index(128)
    print(f"\n  At all 128 dimensions the plotted curve ({m_window[i128]:.3f}) "
          f"is an {a.last}-layer\n  average; the matched single-layer value is "
          f"{m_final[i128]:.3f}. Any comparison against a\n  final-layer number "
          f"should use the latter.")

    _res = {
        "protocol": pi_protocol.protocol(
            script="analyze_layer_match.py",
            design="within-assay, position-grouped splits (re-reduced from "
                   "analyze_svd's archived surface; nothing refit)",
            layer=pi_protocol.layers("both: final layer and pooled window",
                                     n_layers=int(n_layer), window=a.last),
            features=pi_protocol.features("dz_site components (variance-ordered)",
                                          128),
            source=a.npz, n_assays=int(n_assay), n_seeds=int(n_seed),
            note="Exists to make analyze_svd's curve comparable to final-layer "
                 "numbers; the two conventions differ by ~0.03 at k=128."),
        "curve": a.curve, "ks": ks, "n_assays": int(n_assay),
        "n_seeds": int(n_seed), "layer_window": a.last,
        "final_layer": {str(k): float(m_final[i]) for i, k in enumerate(ks)},
        "last_window": {str(k): float(m_window[i]) for i, k in enumerate(ks)},
        "final_layer_per_assay": {str(k): final[:, i].tolist()
                                  for i, k in enumerate(ks)},
    }
    pi_archive.write_result(a.out, _res, protocol=_res.pop("protocol"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
