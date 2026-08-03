"""Per-layer Pairformer capture for OpenFold3 and Protenix.

The Boltz-2 probe reads four divergence quantities at each of 64 Pairformer
layers -- 256 features. The cross-model probe so far reads only the final
distogram (5 features), which is why its numbers are weaker and cannot be
compared with rho = 0.548. This closes that gap.

All three models build their Pairformer the same way: parameters stacked along a
leading axis and run through `jax.lax.scan`. Capture is therefore the same trick
in each -- re-run the scan with `ys` populated so every layer's (s, z) comes back
-- and the only per-model work is reproducing the trunk faithfully up to the
final recycling iteration.

**Every capture is checked against the model's own trunk output, and the run
aborts when it fails.** A capture that silently diverges from the real forward
pass produces features that look reasonable and describe a computation the model
never performed; this project has already lost two conclusions to activation
captures that were not verified (see geom.py's history, and the duplicate
`iteration` note in pi_core.py).

Two criteria, both enforced, neither optional:

  `verify_capture()`        the captured final z must match the model's own
                            trunk z to within a per-model tolerance (DRIFT_TOL)
  `check_signal_to_drift()` a real single mutation must move z at least
                            MIN_SIGNAL_TO_DRIFT times further than that drift

No model is bit-exact over a composed multi-recycle trunk -- see DRIFT_TOL for
the measurements and for the artefact that made Boltz-2 look exact. The ratio
criterion is therefore what makes the features trustworthy in every case.
Passing a permissive tolerance to silence either check defeats the purpose of
the module.

A capture verified on a degenerate input is not verified. Check the alignment
depth the model actually saw before trusting a drift of zero.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# OpenFold3
# ---------------------------------------------------------------------------

def of3_capture(model, batch, *, num_recycles=3, key, deterministic=True):
    """Every Pairformer layer's (s, z) at the FINAL recycling iteration.

    Reproduces `OpenFold3.run_trunk` exactly:
        k_embed, key = split(key)                       (run_trunk)
        iter_key     = fold_in(key, i)   for cycle i    (_run_trunk_loop)
        k_tmpl, k_msa, k_pf = split(iter_key, 3)        (_trunk_iteration)
    Cycles 0..n-2 run through the model's own code; only the last one is opened
    up, so any drift would have to come from the final iteration alone.
    """
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    n_cycles = num_recycles + 1
    k_embed, k_loop = jax.random.split(key)
    s_input, s_init, z_init = model.input_embedder(batch=batch, key=k_embed)

    token_mask = batch.token_mask
    pair_mask = token_mask[..., None] * token_mask[..., None, :]

    # All but the last cycle run through the model's OWN rolled loop.
    #
    # A Python-unrolled loop calling the same `_trunk_iteration` gives 6.4e-04
    # relative drift -- not a logic error (the model is bit-deterministic, and
    # jitting the unrolled version does not help) but XLA numerics differing
    # between a `fori_loop`-rolled body and an unrolled one, accumulated over
    # 4 cycles x 48 layers in float32. Using `_run_trunk_loop` for the first
    # n-1 cycles keeps that path identical to the real forward pass, and it
    # already uses keys fold_in(k_loop, 0 .. n-2) -- exactly what those cycles
    # should see -- so only the final cycle is opened.
    s, z = model._run_trunk_loop(
        batch, s_input, s_init, z_init, n_cycles - 1,
        key=k_loop, deterministic=deterministic)

    # final cycle, opened up to the Pairformer
    iter_key = jax.random.fold_in(k_loop, n_cycles - 1)
    k_tmpl, k_msa, k_pf = jax.random.split(iter_key, 3)
    z = z_init + model.linear_z(model.layer_norm_z(z))
    z = z + model.template_embedder(batch=batch, z=z, pair_mask=pair_mask,
                                    key=k_tmpl, deterministic=deterministic)
    m, msa_mask = model.msa_module_embedder(batch=batch, s_input=s_input)
    z = model.msa_module(m, z, msa_mask=msa_mask.astype(m.dtype),
                         pair_mask=pair_mask.astype(z.dtype),
                         key=k_msa, deterministic=deterministic)
    s = s_init + model.linear_s(model.layer_norm_s(s))

    stack = model.pairformer_stack

    @jax.checkpoint          # the model's own body_fn is checkpointed; match it
    def body_fn(carry, params):
        s, z, k = carry
        block = eqx.combine(params, stack.static)
        s, z = block(s=s, z=z, single_mask=token_mask.astype(z.dtype),
                     pair_mask=pair_mask.astype(s.dtype),
                     key=k, deterministic=deterministic)
        return (s, z, jax.random.fold_in(k, 1)), (s, z)

    (s_f, z_f, _), (s_all, z_all) = jax.lax.scan(
        body_fn, (s, z, k_pf), stack.stacked_params)
    return dict(s=s_f, z=z_f, s_layers=s_all, z_layers=z_all)


# ---------------------------------------------------------------------------
# Protenix
# ---------------------------------------------------------------------------

def protenix_capture(model, feats, *, num_recycles=3, key, deterministic=True):
    """Protenix counterpart, replicating `Protenix.recycle` exactly.

    Three things that a plausible-looking guess gets wrong, and each one alone
    put the capture's drift on the same scale as the mutation signal:
      * `z = self.msa_module(...)` is an ASSIGNMENT, not `z = z + msa_module(...)`;
      * the Pairformer key is `fold_in(key, 1)`, not the cycle key;
      * each cycle consumes one element of `split(key, recycling_steps)`, so
        `recycle()` cannot be reused for the first n-1 cycles -- it would split
        the key into n-1 pieces rather than take the first n-1 of n.
    """
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from protenix.protenij import TrunkEmbedding

    emb = model.embed_inputs(input_feature_dict=feats)
    keys = jax.random.split(key, num_recycles)

    def cycle(state, k, capture=False):
        state = jax.lax.stop_gradient(state)
        s, z = state.s, state.z
        z = emb.z_init + model.linear_no_bias_z_cycle(model.layernorm_z_cycle(z))
        if model.template_embedder.n_blocks > 0:
            z = z + model.template_embedder(feats, z, pair_mask=None, key=k)
        z = model.msa_module(feats, z, emb.s_inputs, pair_mask=None, key=k)
        s = emb.s_init + model.linear_no_bias_s(model.layernorm_s(s))
        if not capture:
            s, z = model.pairformer_stack(s, z, pair_mask=None,
                                          key=jax.random.fold_in(k, 1))
            return TrunkEmbedding(s=s, z=z), None
        stack = model.pairformer_stack

        @jax.checkpoint
        def body_fn(carry, params):
            s, z, kk = carry
            s, z = eqx.combine(stack.static, params)(s=s, z=z, pair_mask=None, key=kk)
            return (s, z, kk), (s, z)

        (s_f, z_f, _), (s_all, z_all) = jax.lax.scan(
            body_fn, (s, z, jax.random.fold_in(k, 1)), stack.stacked_parameters)
        return TrunkEmbedding(s=s_f, z=z_f), (s_all, z_all)

    state = TrunkEmbedding(s=jnp.zeros_like(emb.s_init),
                           z=jnp.zeros_like(emb.z_init))
    for i in range(num_recycles - 1):
        state, _ = cycle(state, keys[i])
    state, (s_all, z_all) = cycle(state, keys[-1], capture=True)
    return dict(s=state.s, z=state.z, s_layers=s_all, z_layers=z_all)


# ---------------------------------------------------------------------------
# Boltz-2
# ---------------------------------------------------------------------------

def boltz2_capture(model, feats, *, num_recycles=3, key, deterministic=True):
    """Boltz-2 counterpart, replicating `mosaic.losses.boltz2.boltz2_trunk`.

    `pi_core.iteration` is already verified bit-identical to joltz's own
    `trunk_iteration` (test_equivalence.py), so the non-final recycles reuse it
    rather than a second copy -- pi_core.py's docstring records what happened
    the last time this file grew a parallel implementation.

    The key schedule is the part a plausible guess gets wrong. joltz THREADS the
    key: cycle 0 sees `key`, and each `trunk_iteration` returns `fold_in(k, 2)`
    for the next one. `pi_core.run_trunk` instead uses `fold_in(key, i)`, which
    is a different schedule. That would be harmless under `deterministic=True`
    -- dropout masks are all-ones regardless of key -- except that mosaic loads
    Boltz-2 with `subsample_msa=True, num_subsampled_msa=1024`, and MSAModule2
    draws its row subset from `jax.random.permutation(key, ...)`. A mismatched
    schedule therefore feeds the capture a DIFFERENT 1024 alignment rows than
    the forward pass saw, which is exactly the "looks plausible, describes a
    computation the model never ran" failure this module exists to prevent.
    """
    import jax
    import jax.numpy as jnp
    import pi_core as pi
    from joltz import TrunkState

    emb = model.embed_inputs(feats)
    state = TrunkState(s=jnp.zeros_like(emb.s_init), z=jnp.zeros_like(emb.z_init))

    k = key
    for _ in range(num_recycles - 1):
        state = pi.iteration(model, state, emb, feats, key=k,
                             deterministic=deterministic, capture=False)
        k = jax.random.fold_in(k, 2)          # joltz.Joltz2.trunk_iteration

    # final recycle, opened up to the Pairformer
    mask = feats["token_pad_mask"]
    pair_mask = mask[:, :, None] * mask[:, None, :]
    s = emb.s_init + model.s_recycle(model.s_norm(state.s))
    z = emb.z_init + model.z_recycle(model.z_norm(state.z))
    z = z + model.template_module(z, feats, pair_mask,
                                  deterministic=deterministic, key=k)
    z = z + model.msa_module(z, emb.s_inputs, feats, deterministic=deterministic,
                             key=jax.random.fold_in(k, 0))
    s_f, z_f, (s_all, z_all) = pi.pairformer_capture(
        model.pairformer_module, s, z, mask, pair_mask,
        key=jax.random.fold_in(k, 1), deterministic=deterministic,
        reduce_fn=lambda s_, z_: (s_, z_),
    )
    return dict(s=s_f, z=z_f, s_layers=s_all, z_layers=z_all)


CAPTURE = {"of3": of3_capture, "protenix": protenix_capture,
           "boltz2": boltz2_capture}


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------

# All three models carry a few times 1e-4 of float32 rounding, and none of them
# is bit-exact over a composed multi-recycle trunk.
#
# OpenFold3 and Protenix wrap their scan bodies in `jax.checkpoint` inside a
# `fori_loop`; reproducing that fusion while also extracting intermediates is not
# possible (of3 4.7e-04, protenix 5.3e-04 -- logs/cap_test_36806987.out).
#
# Boltz-2 was previously believed exact, and this file said so. It is not, and
# the evidence that said otherwise was an artefact: capture_test was run with an
# alignment whose query row did not match the input, boltz silently substituted a
# one-row dummy MSA, and on that degenerate input both paths agreed to 0.0 with a
# signal/drift ratio of infinity. With real alignments the composed 3-recycle
# trunk drifts 2.5e-04 (RCRO) to 6.2e-04 (RS15, PSAE), protein-dependent in the
# way accumulated rounding is. The cause is the one pi_core.py already documents:
# `boltz2_trunk` runs its recycles inside a `jax.lax.scan` while the capture must
# unroll them to open the last one, and XLA fuses the two differently. What IS
# exact is a SINGLE iteration -- test_equivalence.py verifies pi_core.iteration
# against joltz's trunk_iteration at rel err 0.0 -- which is a weaker statement
# than the one this file used to make.
#
# These are enforced ceilings, not observations: a capture that drifts past them
# aborts the run. With no model bit-exact, `check_signal_to_drift` is the
# operative criterion for all three.
DRIFT_TOL = {"boltz2": 2e-3, "of3": 2e-3, "protenix": 2e-3}

# A single substitution must move z far more than the capture's own numerical
# noise, or the per-layer features are measuring rounding. Measured ratios at
# the time of writing: of3 716x, protenix 750x.
MIN_SIGNAL_TO_DRIFT = 50.0


def trunk_reference(name, inner, feats, *, num_recycles=3, key, deterministic=True):
    """The model's OWN final trunk (s, z), computed without any capture path.

    This is the thing a capture must reproduce. Each model's non-capturing
    entry point differs, and note the recycle-count conventions differ too:
    OpenFold3's `run_trunk` takes the number of ITERATIONS (num_recycles + 1),
    while Protenix and Boltz-2 take the number of recycling steps directly.
    """
    if name == "of3":
        _, trunk = inner.run_trunk(feats, num_recycles + 1, key=key)
        return trunk.s, trunk.z
    if name == "protenix":
        emb = inner.embed_inputs(input_feature_dict=feats)
        trunk = inner.recycle(initial_embedding=emb, input_feature_dict=feats,
                              recycling_steps=num_recycles, key=key)
        return trunk.s, trunk.z
    if name == "boltz2":
        from mosaic.losses.boltz2 import boltz2_trunk
        _, trunk = boltz2_trunk(inner, feats, recycling_steps=num_recycles,
                                deterministic=deterministic, key=key)
        return trunk.s, trunk.z
    raise KeyError(f"no trunk reference for {name!r}")


def verify_capture(name, inner, feats, cap, *, num_recycles=3, key,
                   deterministic=True, tol=None):
    """Check a capture against the model's real trunk. Raises on failure.

    Returns the relative drift on z. Callers should follow this with
    `check_signal_to_drift` once a mutant capture exists, because a small
    absolute drift is only meaningful relative to the effect being measured.
    """
    tol = DRIFT_TOL[name] if tol is None else tol
    _, ref_z = trunk_reference(name, inner, feats, num_recycles=num_recycles,
                               key=key, deterministic=deterministic)
    return assert_matches_forward(name, cap["z"], ref_z, tol=tol)


def check_signal_to_drift(name, z_mut, z_wt, drift_rel, minimum=None):
    """Abort if a real mutation moves z by little more than the capture noise."""
    minimum = MIN_SIGNAL_TO_DRIFT if minimum is None else minimum
    sig, ratio = signal_to_drift(z_mut, z_wt, drift_rel)
    if ratio < minimum:
        raise AssertionError(
            f"{name}: mutation signal {sig:.3e} is only {ratio:.1f}x the capture "
            f"drift {drift_rel:.3e} (need {minimum:g}x). The per-layer features "
            "would be dominated by numerical noise rather than the mutation.")
    return sig, ratio


def signal_to_drift(z_mut, z_wt, drift_rel):
    """Is the capture's drift small compared with the effect being measured?

    Bit-identity is the right bar when the capture can reproduce the model's
    exact code path (Boltz-2: rel err 0.0). OpenFold3 and Protenix wrap their
    scan bodies in `jax.checkpoint` inside a `fori_loop`, and reproducing that
    fusion exactly while also extracting intermediates is not possible -- the
    residual is ~7e-4 of float32 rounding, not a different computation
    (established by diagnostic: the model is bit-deterministic, and a loop built
    from the model's OWN `_trunk_iteration` shows the same residual).

    So the usable criterion is a ratio: the mutant-minus-wild-type signal must be
    far larger than the drift. Returns (signal_rel, ratio).
    """
    a = np.asarray(z_mut, dtype=np.float64)
    b = np.asarray(z_wt, dtype=np.float64)
    scale = max(float(np.abs(b).max()), 1e-12)
    signal_rel = float(np.abs(a - b).max() / scale)
    return signal_rel, (signal_rel / drift_rel if drift_rel > 0 else float("inf"))


def assert_matches_forward(name, captured_z, model_z, tol=1e-4):
    """The captured final layer MUST be the model's own trunk output.

    Returns the relative error. Raises if the capture describes a different
    computation from the one the model runs -- the failure mode that makes
    per-layer features fiction while leaving them looking perfectly plausible.
    """
    a = np.asarray(captured_z, dtype=np.float64)
    b = np.asarray(model_z, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"{name}: capture shape {a.shape} != trunk {b.shape}")
    denom = max(float(np.abs(b).max()), 1e-12)
    rel = float(np.abs(a - b).max() / denom)
    if rel > tol:
        raise AssertionError(
            f"{name}: capture does NOT reproduce the trunk (rel err {rel:.3e} > {tol}). "
            "The per-layer features would describe a computation the model never ran.")
    return rel
