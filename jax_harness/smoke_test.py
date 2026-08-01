"""Validate that pi_core's instrumented trunk is numerically identical to joltz.

If this does not pass, nothing downstream is trustworthy: the whole approach
rests on the claim that re-running the scan with `ys` populated changes nothing
about the computation.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402

YAML = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/test_protforge/outputs/sequences/34073/34073.yaml"
)
RECYCLES = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def main():
    print(f"jax devices: {jax.devices()}", flush=True)

    t0 = time.time()
    model = pi.load_model(subsample_msa=False)
    print(f"[{time.time()-t0:6.1f}s] model loaded", flush=True)

    feats, handle = pi.load_features(YAML.read_text(), use_msa_server=False)
    N = int(feats["token_pad_mask"].sum())
    print(
        f"[{time.time()-t0:6.1f}s] features: N={N} tokens, "
        f"MSA depth S={feats['msa'].shape[1]}, msa one-hot {feats['msa'].shape}",
        flush=True,
    )

    key = jax.random.key(0)

    # --- reference: joltz's own trunk, no capture -------------------------
    from mosaic.losses.boltz2 import boltz2_trunk

    t1 = time.time()
    emb_ref, state_ref = boltz2_trunk(
        model, feats, recycling_steps=RECYCLES, deterministic=True, key=key
    )
    d_ref = pi.logit_lens(model, state_ref.z)
    d_ref.block_until_ready()
    print(f"[{time.time()-t0:6.1f}s] reference trunk done ({time.time()-t1:.1f}s)", flush=True)

    # --- instrumented ------------------------------------------------------
    t1 = time.time()
    out = pi.trunk_capture(model, feats, recycling_steps=RECYCLES, key=key, deterministic=True)
    d_cap = out["distogram"]
    d_cap.block_until_ready()
    print(f"[{time.time()-t0:6.1f}s] captured trunk done ({time.time()-t1:.1f}s)", flush=True)

    # --- equivalence -------------------------------------------------------
    dz = float(jnp.abs(state_ref.z - out["trunk_state"].z).max())
    ds = float(jnp.abs(state_ref.s - out["trunk_state"].s).max())
    dd = float(jnp.abs(d_ref - d_cap).max())
    zscale = float(jnp.abs(state_ref.z).max())
    print("\n=== equivalence vs joltz reference ===")
    print(f"  max|dz| = {dz:.3e}   (|z| max {zscale:.3f})")
    print(f"  max|ds| = {ds:.3e}")
    print(f"  max|d_distogram| = {dd:.3e}")

    pf = out["pairformer_layers"]
    msa = out["msa_layers"]
    print("\n=== capture shapes ===")
    print(f"  pairformer s      {pf['s'].shape}")
    print(f"  pairformer z_norm {pf['z_norm'].shape}")
    print(f"  msa z_norm        {msa['z_norm'].shape}")
    print(f"  msa opm_norm      {msa['opm_norm'].shape}")

    # --- the headline number: how much of z does the MSA write? -----------
    m = np.asarray(feats["token_pad_mask"][0]).astype(bool)
    zi = float(jnp.linalg.norm(out["z_after_init"][0], axis=-1)[np.ix_(m, m)].mean())
    zt = float(jnp.linalg.norm(out["z_after_template"][0], axis=-1)[np.ix_(m, m)].mean())
    zm = float(jnp.linalg.norm(out["z_after_msa"][0], axis=-1)[np.ix_(m, m)].mean())
    print("\n=== mean |z| per pair through the trunk ===")
    print(f"  after z_init (query direct path) {zi:.4f}")
    print(f"  after template                   {zt:.4f}")
    print(f"  after MSA module                 {zm:.4f}   (x{zm/max(zi,1e-9):.2f} vs init)")
    opm = np.asarray(msa["opm_norm"])[:, m][:, :, m]
    print(f"  OPM increment per MSA block      {opm.mean(axis=(1,2))}")

    ok = dd < 1e-3 and dz < 1e-2
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}", flush=True)
    handle.cleanup()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
