"""Does the per-layer capture reproduce each model's own trunk?

This ENFORCES the criteria rather than reporting them: each model's captured
final z is compared against its real trunk output at that model's tolerance
(pi_capture.DRIFT_TOL), and a real single mutation must move z at least
MIN_SIGNAL_TO_DRIFT times further than the residual drift. An earlier version
passed tol=1e9 to both checks, which printed the numbers but could not fail --
so a broken capture would have been recorded as a pass.

Exit status is 0 only if every model passes.
"""
from __future__ import annotations
import sys, traceback
from pathlib import Path
import numpy as np, jax
sys.path.insert(0, str(Path(__file__).parent))
import pi_models, pi_capture

seq, a3m = sys.argv[1], sys.argv[2]
models = sys.argv[3].split(",") if len(sys.argv) > 3 else ["of3", "protenix", "boltz2"]
pi_models.block_network()
failures = []
for name in models:
    print(f"\n=== {name} ===", flush=True)
    try:
        wrapper = pi_models.load(name)
        model = pi_models.inner(name, wrapper)
        feats, depth = pi_models.features_for(name, wrapper, seq, a3m,
                                              work=Path("/tmp/capw") / name)
        key = jax.random.key(0)
        cap = pi_capture.CAPTURE[name](model, feats, num_recycles=3, key=key)
        print(f"  msa rows {depth}")
        # A capture verified on a degenerate input is not verified. Boltz-2
        # silently substitutes a one-row dummy alignment when the a3m query row
        # does not match the input sequence; on that input every code path
        # agrees trivially, and this test once reported drift 0.0 and an
        # infinite signal/drift ratio for a capture that in fact drifts 6e-4.
        if depth < 2:
            raise AssertionError(
                f"alignment collapsed to {depth} row(s) -- the model is not "
                "running the computation under test. Check that the a3m query "
                "row matches the input sequence.")
        print(f"  z_layers {np.asarray(cap['z_layers']).shape}  "
              f"s_layers {np.asarray(cap['s_layers']).shape}")
        drift = pi_capture.verify_capture(name, model, feats, cap,
                                          num_recycles=3, key=key)
        print(f"  capture drift vs trunk: rel {drift:.3e}  "
              f"(tol {pi_capture.DRIFT_TOL[name]:g})  -> PASS")
        # now the signal: a real single mutation through the same capture
        mut = seq[:20] + ("A" if seq[20] != "A" else "W") + seq[21:]
        fm, _ = pi_models.features_for(name, wrapper, mut, a3m,
                                       work=Path("/tmp/capw") / (name + "_m"))
        capm = pi_capture.CAPTURE[name](model, fm, num_recycles=3, key=key)
        sig, ratio = pi_capture.check_signal_to_drift(name, capm["z"], cap["z"], drift)
        print(f"  one-mutation signal   : rel {sig:.3e}")
        print(f"  SIGNAL / DRIFT        : {ratio:.0f}x  "
              f"(need {pi_capture.MIN_SIGNAL_TO_DRIFT:g}x)  -> PASS")
    except AssertionError as e:
        print(f"  FAIL: {e}")
        failures.append(name)
    except Exception:
        traceback.print_exc()
        failures.append(name)
print(f"\nCAPTURE TEST COMPLETE: "
      f"{'all passed' if not failures else 'FAILED for ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
