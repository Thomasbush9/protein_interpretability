"""Decide whether pi_paths.iteration differs from joltz structurally or only in float32 rounding.

Structural bug and fusion noise look the same in a max-abs diff over a 64-layer
residual stack, so compare like for like:

  A. one iteration, identical key and identical input state, mine vs joltz
     -> any difference here is pure numerics (same math, different fusion)
  B. joltz vs joltz with a different key
     -> the scale of "nothing changed at all" (should be 0 if dropout is off)
  C. what the difference costs in the only units that matter: expected
     distance in Angstrom, and contact-map agreement.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pi_core as pi  # noqa: E402
import pi_paths as pp  # noqa: E402
from joltz import TrunkState  # noqa: E402

YAML = Path(
    "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/test_protforge/outputs/sequences/34073/34073.yaml"
)


def rel(a, b):
    return float(jnp.abs(a - b).max() / jnp.abs(b).max())


def main():
    model = pi.load_model(subsample_msa=False)
    feats, handle = pi.load_features(YAML.read_text(), use_msa_server=False)
    emb = model.embed_inputs(feats)
    mask = np.asarray(feats["token_pad_mask"][0]).astype(bool)
    k0, k1 = jax.random.key(0), jax.random.key(12345)

    zero = TrunkState(s=jnp.zeros_like(emb.s_init), z=jnp.zeros_like(emb.z_init))

    # --- B: joltz vs joltz, different keys ------------------------------
    sB0, _ = model.trunk_iteration(zero, emb, feats, key=k0, deterministic=True)
    sB1, _ = model.trunk_iteration(zero, emb, feats, key=k1, deterministic=True)
    print("B. joltz(key0) vs joltz(key1), 1 iteration")
    print(f"     rel max|dz| = {rel(sB1.z, sB0.z):.3e}   <- dropout/subsampling determinism")

    # --- A: mine vs joltz, same key, same state -------------------------
    sA = pp.iteration(model, zero, emb, feats, key=k0, deterministic=True, capture=False)
    print("A. mine vs joltz, 1 iteration, identical key + state")
    print(f"     rel max|dz| = {rel(sA.z, sB0.z):.3e}")
    print(f"     rel max|ds| = {rel(sA.s, sB0.s):.3e}")

    # --- A': with capture on (the scan that populates ys) ---------------
    sAc, _ = pp.iteration(model, zero, emb, feats, key=k0, deterministic=True, capture=True)
    print("A'. capture=True vs capture=False (mine, same key)")
    print(f"     rel max|dz| = {rel(sAc.z, sA.z):.3e}")

    # --- C: what it costs in physical units -----------------------------
    def ed(z):
        return np.asarray(pi.expected_distance(pi.logit_lens(model, z)))[np.ix_(mask, mask)]

    def cp(z):
        return np.asarray(pi.contact_prob(pi.logit_lens(model, z)))[np.ix_(mask, mask)]

    e_mine, e_ref = ed(sA.z), ed(sB0.z)
    c_mine, c_ref = cp(sA.z), cp(sB0.z)
    print("C. downstream cost (1 iteration)")
    print(f"     max |dE[d]|  = {np.abs(e_mine - e_ref).max():.4f} A")
    print(f"     mean|dE[d]|  = {np.abs(e_mine - e_ref).mean():.5f} A")
    print(f"     max |dP(contact)| = {np.abs(c_mine - c_ref).max():.5f}")
    print(
        "     contact agreement @0.5 = "
        f"{((c_mine > 0.5) == (c_ref > 0.5)).mean():.6f}"
    )

    # --- reference for 'how big is a real change?' ----------------------
    # Perturb one buried residue's z_init and see what a genuine change looks like.
    print("\nD. scale reference: same comparison but between recycle 1 and recycle 2")
    s2 = pp.iteration(model, sB0, emb, feats, key=k0, deterministic=True, capture=False)
    e2 = ed(s2.z)
    print(f"     max |dE[d]| recycle1->2 = {np.abs(e2 - e_ref).max():.4f} A")
    print(f"     mean|dE[d]| recycle1->2 = {np.abs(e2 - e_ref).mean():.5f} A")

    handle.cleanup()


if __name__ == "__main__":
    main()
