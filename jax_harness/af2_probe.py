import sys, traceback
from pathlib import Path
sys.path.insert(0, "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/prot_interp_files/harness")
import numpy as np, jax, pi_models
seq = sys.argv[1]
try:
    m = pi_models.load("af2"); print("loaded", type(m).__name__)
    f, d = pi_models.features_for("af2", m, seq, None, work=Path("/tmp/af2w"))
    print("features ok, msa_depth", d)
    o = m.model_output(features=f, recycling_steps=3, sampling_steps=None,
                       key=jax.random.key(0))
    e = pi_models.extraction_from(o, name="af2")
    print(f"logits {e.logits.shape} bins={e.n_bins} grid {e.centres[0]:.3f}..{e.centres[-1]:.3f}")
    print(f"ed mean {e.ed.mean():.2f} A  entropy {e.entropy.mean():.4f}  "
          f"ca {e.ca.shape}  plddt {e.plddt.mean():.4f}")
    print("AF2 OK")
except Exception:
    traceback.print_exc()
