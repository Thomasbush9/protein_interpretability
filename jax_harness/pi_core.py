"""Core harness for mechanistic interpretability of the Boltz-2 trunk (JAX/joltz).

Design note
-----------
joltz stores the Pairformer and MSA stacks as a *stacked* pytree of parameters
(`stacked_parameters`) plus a `static` template, and runs them with
`jax.lax.scan`, discarding the per-layer outputs. Rather than patch joltz, we
re-run the same scan here with the `ys` slot populated. That keeps us bit-exact
with the library's own forward pass while giving per-layer activations.

The Boltz-2 trunk routes query-sequence information into the pair
representation `z` along four separable paths:

  P1  s_inputs -> z_init_1/z_init_2 outer sum          (direct query -> pair)
  P2  s_inputs -> msa_module.s_proj -> every MSA row   (query broadcast into MSA)
  P3  msa row 0 (the query itself) -> OuterProductMean (query as one MSA row)
  P4  msa rows 1..S -> OuterProductMean                (the evolutionary prior)

OuterProductMean divides by the number of unmasked rows, so P3's contribution to
`z` is down-weighted by ~1/S relative to the whole MSA. P4 is the "cheat sheet".
Isolating these is what `ablate_*` below is for.

Constants for Boltz-2 (boltz 2.2.x): 64 pairformer blocks, 4 MSA blocks,
s=384, z=128, distogram 64 bins over 2-22 A.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import joltz
from joltz import TrunkState

from boltz.data import const
from boltz.main import (
    BoltzProcessedInput,
    Boltz2InferenceDataModule,
    Manifest,
    check_inputs,
    process_inputs,
)

BOLTZ_CACHE = Path("~/.boltz").expanduser()

# Distogram bin centres: Boltz-2 uses 64 bins spanning 2-22 A.
DISTO_MIN, DISTO_MAX, DISTO_BINS = 2.0, 22.0, 64
BIN_CENTRES = np.linspace(DISTO_MIN, DISTO_MAX, DISTO_BINS + 1)[:-1] + (
    (DISTO_MAX - DISTO_MIN) / DISTO_BINS / 2.0
)


# --------------------------------------------------------------------------
# feature construction (offline: MSA must be a local a3m referenced by the yaml)
# --------------------------------------------------------------------------
def load_features(yaml_str: str, cache: Path = BOLTZ_CACHE, use_msa_server: bool = False):
    """Featurize a Boltz yaml. Returns (features, tmpdir_handle).

    The handle must be kept alive: Boltz's processed inputs live inside it.
    `use_msa_server=False` keeps this runnable on compute nodes with no network,
    which requires every chain in the yaml to carry an explicit `msa:` path.
    """
    handle = TemporaryDirectory()
    out_dir = Path(handle.name)
    yaml_path = out_dir / "input.yaml"
    yaml_path.write_text(yaml_str)

    data = check_inputs(yaml_path)
    manifest = process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=cache / "ccd.pkl",
        mol_dir=cache / "mols",
        use_msa_server=use_msa_server,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        boltz2=True,
    )
    processed_dir = out_dir / "processed"
    if manifest is None:
        manifest = Manifest.load(processed_dir / "manifest.json")

    processed = BoltzProcessedInput(
        manifest=manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=_opt(processed_dir / "constraints"),
        template_dir=_opt(processed_dir / "templates"),
        extra_mols_dir=_opt(processed_dir / "mols"),
    )
    dm = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=0,
        mol_dir=cache / "mols",
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
        override_method=None,
    )
    feats = {k: np.array(v) for k, v in list(dm.predict_dataloader())[0].items() if k != "record"}
    feats["msa"] = jax.nn.one_hot(feats["msa"], const.num_tokens)
    return jax.tree.map(jnp.array, feats), handle


def _opt(p: Path):
    return p if p.exists() else None


def load_model(ckpt: Path = BOLTZ_CACHE / "boltz2_conf.ckpt", subsample_msa: bool = False,
               backend: str = "joltz"):
    """Load Boltz-2 into joltz.

    `subsample_msa=False` is deliberate and differs from mosaic's design-time
    default: for interpretability we need the MSA depth to be an *exact,
    controlled* quantity, not a random 1024-row draw that changes per key.
    Control depth by truncating the a3m instead.

    `backend` selects WHERE the same network comes from, and the two are not
    bit-identical:

        joltz    this function's own checkpoint load. Every archived Boltz-2
                 capture was produced this way, so it stays the default.
        mosaic   `pi_models.load("boltz2")`, unwrapped. One loader for all four
                 models instead of a Boltz-2-only path beside a general one.

    Either way the object returned is the joltz network, so the trunk
    instrumentation below is unaffected -- mosaic's wrapper holds exactly this
    in `.model`, which is why the unification is a swap of the loader and not a
    rewrite of the capture code.

    Measured on ARGR at matched MSA regime, the two agree on expected distance
    to 0.030 A max / 0.0019 A mean (bin width is 0.3125 A). Small, but not zero,
    which is why the default does not move on its own -- see
    `exp_backend_equiv` for the dz_site comparison that decides whether it can.
    """
    if backend not in ("joltz", "mosaic"):
        raise ValueError(f"backend must be 'joltz' or 'mosaic', got {backend!r}")
    if backend == "mosaic":
        import pi_models
        wrapper = pi_models.load(
            "boltz2", msa="subsample" if subsample_msa else "full")
        return wrapper.model
    from dataclasses import asdict

    import torch  # noqa: F401  (checkpoint load only; not in the hot path)
    from boltz import main as boltz_main
    from boltz.model.models.boltz2 import Boltz2

    torch_model = Boltz2.load_from_checkpoint(
        ckpt,
        strict=True,
        map_location="cpu",
        predict_args={"recycling_steps": 0, "sampling_steps": 25, "diffusion_samples": 1},
        diffusion_process_args=asdict(boltz_main.Boltz2DiffusionParams()),
        msa_args=asdict(
            boltz_main.MSAModuleArgs(
                subsample_msa=subsample_msa,
                num_subsampled_msa=1024,
                use_paired_feature=True,
            )
        ),
        pairformer_args=asdict(boltz_main.PairformerArgsV2()),
    ).eval()
    model = joltz.from_torch(torch_model)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    return eqx.combine(jax.device_put(params), static)


# --------------------------------------------------------------------------
# instrumented stacks (bit-identical to joltz, but with ys populated)
# --------------------------------------------------------------------------
def pairformer_capture(pf, s, z, mask, pair_mask, *, key, deterministic=True, reduce_fn=None):
    """Run joltz Pairformer2 capturing per-layer output.

    reduce_fn(s, z) -> pytree is applied inside the scan so we never
    materialise 64 full copies of z (64 x N^2 x 128 floats). Default keeps the
    single rep `s` (cheap) and z's per-pair L2 norm.
    """
    if reduce_fn is None:
        reduce_fn = lambda s_, z_: {  # noqa: E731
            "s": s_[0],
            "z_norm": jnp.linalg.norm(z_[0], axis=-1),
        }

    @jax.checkpoint
    def body(carry, params):
        s_, z_, k_ = carry
        s_, z_, k_ = eqx.combine(pf.static, params)(
            s_, z_, mask, pair_mask, key=k_, deterministic=deterministic
        )
        return (s_, z_, k_), reduce_fn(s_, z_)

    (s, z, key), per_layer = jax.lax.scan(body, (s, z, key), pf.stacked_parameters)
    return s, z, per_layer


def msa_module_capture(msa_mod, z, emb, feats, *, key, deterministic=True):
    """Run joltz MSAModule2 capturing per-layer z and the OPM increment.

    Returns (z_out, per_layer) where per_layer has:
      z_norm     — per-pair L2 norm of z after each of the 4 MSA blocks
      opm_norm   — per-pair L2 norm of the OuterProductMean increment alone
                   (this is the MSA -> pair write; the "cheat sheet" channel)
    """
    msa = feats["msa"]
    m = jnp.concatenate(
        [
            msa,
            feats["has_deletion"][..., None],
            feats["deletion_value"][..., None],
            feats["msa_paired"][..., None],
        ]
        if msa_mod.use_paired_feature
        else [msa, feats["has_deletion"][..., None], feats["deletion_value"][..., None]],
        axis=-1,
    )
    msa_mask = feats["msa_mask"]
    token_mask = feats["token_pad_mask"].astype(jnp.float32)
    token_mask = token_mask[:, :, None] * token_mask[:, None, :]

    if msa_mod.subsample_msa:
        idx = jax.random.permutation(key, jnp.arange(msa.shape[1]))[: msa_mod.num_subsampled_msa]
        m, msa_mask = m[:, idx], msa_mask[:, idx]

    m = msa_mod.msa_proj(m)
    m = m + msa_mod.s_proj(emb)[:, None, ...]

    @jax.checkpoint
    def body(carry, params):
        z_, m_, k_ = carry
        layer = eqx.combine(msa_mod.static, params)
        drop, k2 = joltz.get_dropout_mask(layer.msa_dropout, m_, not deterministic, key=k_)
        m_ = m_ + drop * layer.pair_weighted_averaging(m_, z_, token_mask)
        m_ = m_ + layer.msa_transition(m_)
        opm = layer.outer_product_mean(m_, msa_mask)
        z_ = z_ + opm
        z_ = layer.pairformer_layer(
            z_, token_mask, key=jax.random.fold_in(k2, 0), deterministic=deterministic
        )
        stats = {
            "z_norm": jnp.linalg.norm(z_[0], axis=-1),
            "opm_norm": jnp.linalg.norm(opm[0], axis=-1),
        }
        return (z_, m_, jax.random.fold_in(k2, 0)), stats

    (z, m, _), per_layer = jax.lax.scan(body, (z, m, key), msa_mod.stacked_parameters)
    return z, per_layer


def logit_lens(model, z):
    """Apply the distogram head to an intermediate z. Returns (N, N, 64) logits.

    This is the structural analogue of the LLM 'logit lens': it asks what
    contact map the model would emit if the trunk stopped at this layer.
    """
    return model.distogram_module(z)[0, :, :, 0, :]


def expected_distance(logits):
    """E[d] in Angstrom from distogram logits (..., 64)."""
    return (jax.nn.softmax(logits, axis=-1) * jnp.asarray(BIN_CENTRES)).sum(-1)


def contact_prob(logits, cutoff_bin: int = 19):
    """P(d < ~8 A) from distogram logits."""
    return jax.nn.softmax(logits, axis=-1)[..., :cutoff_bin].sum(-1)


# --------------------------------------------------------------------------
# instrumented trunk
# --------------------------------------------------------------------------
def iteration(model, state, emb, feats, *, key, deterministic=True, capture=False):
    """One recycling iteration. Mirrors joltz.Joltz2.trunk_iteration exactly.

    Verified bit-identical to `model.trunk_iteration` (rel. error 0.0 on s, z,
    distogram and contact map) by harness/test_equivalence.py.

    Reimplemented rather than called so that patched `emb`/`feats` apply on
    every recycle, and so per-layer capture is available on any step. Do not
    add a second variant of this function: an earlier version that ran some
    recycles through joltz's jitted `trunk_iteration` and only the last through
    the capture path produced a ~6e-4 relative drift that looked like float32
    noise but was really two different computations.
    """
    mask = feats["token_pad_mask"]
    pair_mask = mask[:, :, None] * mask[:, None, :]

    s = emb.s_init + model.s_recycle(model.s_norm(state.s))
    z = emb.z_init + model.z_recycle(model.z_norm(state.z))
    z_after_init = z

    z = z + model.template_module(z, feats, pair_mask, deterministic=deterministic, key=key)
    z_after_template = z

    msa_layers = None
    if capture:
        z_msa, msa_layers = msa_module_capture(
            model.msa_module, z, emb.s_inputs, feats,
            key=jax.random.fold_in(key, 0), deterministic=deterministic,
        )
        z = z + z_msa
    else:
        z = z + model.msa_module(
            z, emb.s_inputs, feats, deterministic=deterministic,
            key=jax.random.fold_in(key, 0),
        )
    z_after_msa = z

    pf_layers = None
    if capture:
        s, z, pf_layers = pairformer_capture(
            model.pairformer_module, s, z, mask, pair_mask,
            key=jax.random.fold_in(key, 1), deterministic=deterministic,
        )
    else:
        s, z = model.pairformer_module(
            s, z, mask=mask, pair_mask=pair_mask,
            key=jax.random.fold_in(key, 1), deterministic=deterministic,
        )

    new_state = TrunkState(s=s, z=z)
    if not capture:
        return new_state
    return new_state, {
        "z_after_init": z_after_init,
        "z_after_template": z_after_template,
        "z_after_msa": z_after_msa,
        "pairformer_layers": pf_layers,
        "msa_layers": msa_layers,
    }


def run_trunk(model, emb, feats, *, recycling_steps=3, key, deterministic=True, capture_last=True):
    """Full trunk. Returns dict with trunk_state, distogram, and (if
    capture_last) per-layer captures from the final recycle."""
    state = TrunkState(s=jnp.zeros_like(emb.s_init), z=jnp.zeros_like(emb.z_init))
    n_plain = max(recycling_steps - 1, 0) if capture_last else recycling_steps

    for i in range(n_plain):
        state = iteration(
            model, state, emb, feats,
            key=jax.random.fold_in(key, i), deterministic=deterministic, capture=False,
        )

    out = {}
    if capture_last:
        state, out = iteration(
            model, state, emb, feats,
            key=jax.random.fold_in(key, recycling_steps - 1),
            deterministic=deterministic, capture=True,
        )

    out["trunk_state"] = state
    out["distogram"] = logit_lens(model, state.z)
    out["initial_embedding"] = emb
    return out


def trunk_capture(model, feats, *, recycling_steps=3, key, deterministic=True):
    """Convenience wrapper: embed then run_trunk with capture."""
    emb = model.embed_inputs(feats)
    return run_trunk(
        model, emb, feats, recycling_steps=recycling_steps, key=key,
        deterministic=deterministic, capture_last=True,
    )
