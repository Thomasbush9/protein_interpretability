"""Is the OF3 capture drift a LOGIC error or XLA fusion? Isolate it."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
sys.path.insert(0, str(Path(__file__).parent))
import pi_models

seq, a3m = sys.argv[1], sys.argv[2]
pi_models.block_network()
wrapper = pi_models.load("of3"); m = pi_models.inner("of3", wrapper)
feats, _ = pi_models.features_for("of3", wrapper, seq, a3m, work=Path("/tmp/diagw"))
key = jax.random.key(0)

def rel(a, b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64)
    return float(np.abs(a-b).max()/max(np.abs(b).max(),1e-12))

_, ref = m.run_trunk(feats, 4, key=key)

# A: my loop, using the model's OWN _trunk_iteration for every cycle (no opening)
k_embed, k_loop = jax.random.split(key)
s_in, s_init, z_init = m.input_embedder(batch=feats, key=k_embed)
tm = feats.token_mask; pm = tm[..., None]*tm[..., None, :]
s = jnp.zeros_like(s_init); z = jnp.zeros_like(z_init)
for i in range(4):
    s, z = jax.lax.stop_gradient((s, z)) if i < 3 else (s, z)
    s, z = m._trunk_iteration(feats, s_in, s_init, z_init, tm, pm, s, z,
                              key=jax.random.fold_in(k_loop, i))
print(f"A  python loop, model's own _trunk_iteration : rel err z {rel(z, ref.z):.3e}")

# B: same, but jitted as one function (matches how run_trunk is compiled)
@eqx.filter_jit
def loop(mm, feats, s_in, s_init, z_init, tm, pm, k_loop):
    s = jnp.zeros_like(s_init); z = jnp.zeros_like(z_init)
    for i in range(4):
        s, z = jax.lax.stop_gradient((s, z)) if i < 3 else (s, z)
        s, z = mm._trunk_iteration(feats, s_in, s_init, z_init, tm, pm, s, z,
                                   key=jax.random.fold_in(k_loop, i))
    return s, z
sB, zB = loop(m, feats, s_in, s_init, z_init, tm, pm, k_loop)
print(f"B  same loop, jitted as one function        : rel err z {rel(zB, ref.z):.3e}")

# C: run_trunk twice -- the model's own reproducibility floor
_, ref2 = m.run_trunk(feats, 4, key=key)
print(f"C  model vs itself (determinism floor)      : rel err z {rel(ref2.z, ref.z):.3e}")
print("\nA large, C zero  => fori_loop-vs-python-loop fusion, not logic")
print("A and B both large => a real logic difference to find")
