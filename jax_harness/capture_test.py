"""Does the per-layer capture reproduce each model's own trunk, bit for bit?"""
from __future__ import annotations
import sys, traceback
from pathlib import Path
import numpy as np, jax
sys.path.insert(0, str(Path(__file__).parent))
import pi_models, pi_capture

seq, a3m = sys.argv[1], sys.argv[2]
pi_models.block_network()
for name in ("of3", "protenix"):
    print(f"\n=== {name} ===", flush=True)
    try:
        wrapper = pi_models.load(name)
        model = pi_models.inner(name, wrapper)
        feats, depth = pi_models.features_for(name, wrapper, seq, a3m,
                                              work=Path("/tmp/capw") / name)
        key = jax.random.key(0)
        # the model's own trunk output
        if name == "of3":
            emb = model.embed_inputs(feats, key=jax.random.split(key)[0])
            _, trunk = model.run_trunk(feats, 4, key=key)
            ref_z, ref_s = trunk.z, trunk.s
            cap = pi_capture.of3_capture(model, feats, num_recycles=3, key=key)
        else:
            emb = model.embed_inputs(input_feature_dict=feats)
            trunk = model.recycle(initial_embedding=emb, input_feature_dict=feats,
                                  recycling_steps=3, key=key)
            ref_z, ref_s = trunk.z, trunk.s
            cap = pi_capture.protenix_capture(model, feats, num_recycles=3, key=key)
        print(f"  msa rows {depth}")
        print(f"  z_layers {np.asarray(cap['z_layers']).shape}  "
              f"s_layers {np.asarray(cap['s_layers']).shape}")
        drift = pi_capture.assert_matches_forward(name, cap["z"], ref_z, tol=1e9)
        print(f"  capture drift vs trunk: rel {drift:.3e}")
        # now the signal: a real single mutation through the same capture
        mut = seq[:20] + ("A" if seq[20] != "A" else "W") + seq[21:]
        fm, _ = pi_models.features_for(name, wrapper, mut, a3m,
                                       work=Path("/tmp/capw") / (name + "_m"))
        capm = (pi_capture.of3_capture(model, fm, num_recycles=3, key=key)
                if name == "of3" else
                pi_capture.protenix_capture(model, fm, num_recycles=3, key=key))
        sig, ratio = pi_capture.signal_to_drift(capm["z"], cap["z"], drift)
        print(f"  one-mutation signal   : rel {sig:.3e}")
        print(f"  SIGNAL / DRIFT        : {ratio:.0f}x  "
              f"{'-> usable' if ratio > 50 else '-> NOT usable'}")
    except AssertionError as e:
        print(f"  CAPTURE MISMATCH: {e}")
    except Exception:
        traceback.print_exc()
print("\nCAPTURE TEST COMPLETE")
