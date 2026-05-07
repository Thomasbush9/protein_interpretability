#!/usr/bin/env python3
"""First-pass chromophore-block gradient attribution for Boltz2.

Loads a mutant YAML and a saved WT distogram, runs one forward pass on the
mutant with grad enabled and `activation_checkpointing=True` on Pairformer +
MSA + diffusion-conditioning, computes a scalar L2-loss on expected pair
distances over WT contacts in B (the chromophore block), backwards from it,
and saves the resulting gradient maps:

  - per-residue input-embedding gradient norm (N,)
  - per-Pairformer-layer pair-rep gradient norm (n_layers, N, N)
  - the captured distogram output for sanity checks

Design + reasoning: `log/2026-05-06-chromophore-attribution-pivot.md`.

Usage::

    python scripts/run_chromophore_attribution.py \\
        path/to/mutant.yaml \\
        --wt_distogram path/to/original_distogram.pt \\
        --b_json path/to/chromo_block.json \\
        --out_dir /path/to/attribution \\
        --cache ~/.boltz \\
        --recycling_steps 1
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

# Bootstrap so the script runs under the cluster boltz env without `python -m`.
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from pytorch_lightning import seed_everything  # noqa: E402
from rdkit import Chem  # noqa: E402

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule  # noqa: E402
from boltz.data.types import Manifest  # noqa: E402
from boltz.main import (  # noqa: E402
    Boltz2DiffusionParams,
    BoltzProcessedInput,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
    check_inputs,
    download_boltz2,
    filter_inputs_structure,
    process_inputs,
)
from boltz.model.models.boltz2 import Boltz2  # noqa: E402


# ---------------------------------------------------------------------------
# Grad-gating helpers (copied from src/protein_interpretability/attribution/runner.py
# so this script is self-contained — the existing module's input-leaf path hit
# the requires_grad=False bug; this script avoids that path entirely).
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _train_mode_for_boltz_grads(model: nn.Module):
    """Make Boltz2's grad gates evaluate correctly during attribution.

    Boltz2.forward gates grads three ways (boltz2.py:411, 440, 550–583).
    Pairformer/MSA also gate `activation_checkpointing` on `self.training`
    (`pairformer.py:189`, msa_module similar). We need:
      - All submodules in training=True (recursive) so the activation-
        checkpointing branches actually fire on Pairformer + MSA + diffusion.
        `model.train()` does this; setting `model.training = True` alone
        only flips the top-level module.
      - structure_prediction_training=True so the outer Boltz2 gate evaluates True.
    Then a forward hook on distogram_module flips `model.training=False`
    *immediately* so the post-distogram branches at L550–583 (which require
    ground-truth coords) never fire. Submodules stay training=True for the
    rest of forward + during backward recompute (irrelevant after distogram
    fires; backward via checkpoint doesn't re-check the gates).

    Boltz uses dropout=0 / msa_dropout=0 in default args, so leaving the
    submodules in train mode for the trunk has no stochastic effect.
    """
    original_training_mode = model.training
    has_struct = hasattr(model, "structure_prediction_training")
    original_struct = getattr(model, "structure_prediction_training", None)

    def _flip_back(_mod, _inp, _out):
        model.training = False  # only top-level — enough for boltz2.py post-distogram gates
        if has_struct:
            model.structure_prediction_training = False

    distogram_module = getattr(model, "distogram_module", None)
    if distogram_module is None:
        raise AttributeError("model has no .distogram_module")
    handle = distogram_module.register_forward_hook(_flip_back)

    model.train()  # recursive: sets training=True on every submodule
    if has_struct:
        model.structure_prediction_training = True
    try:
        yield
    finally:
        handle.remove()
        if original_training_mode:
            model.train()
        else:
            model.eval()
        if has_struct:
            model.structure_prediction_training = original_struct


@contextlib.contextmanager
def _freeze_all_params(model: nn.Module):
    """Temporarily set requires_grad=False on all params (no param grads)."""
    saved = []
    for p in model.parameters():
        saved.append((p, p.requires_grad))
        if p.requires_grad:
            p.requires_grad_(False)
    try:
        yield
    finally:
        for p, was in saved:
            p.requires_grad_(was)


@contextlib.contextmanager
def _temporarily(obj, attr: str, value):
    if not hasattr(obj, attr):
        yield
        return
    old = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old)


# ---------------------------------------------------------------------------
# Loss target
# ---------------------------------------------------------------------------


# Boltz2 distogram bin convention: 64 bins from 2 Å to 22 Å, ~0.3125 Å step.
def _bin_centers(num_bins: int = 64, min_a: float = 2.0, max_a: float = 22.0) -> torch.Tensor:
    step = (max_a - min_a) / num_bins
    centers = torch.tensor(
        [min_a + step * (k + 0.5) for k in range(num_bins)], dtype=torch.float32
    )
    return centers


def expected_distance(logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """E[d] from softmaxed (..., K) bin logits."""
    probs = torch.softmax(logits, dim=-1)
    return (probs * centers.to(logits.device)).sum(dim=-1)


def expected_distance_l2(
    pred_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    block_positions: list[int] | None,
    contact_threshold_a: float = 8.0,
    centers: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """L2 on E[d_pred] vs E[d_WT], averaged over WT contact pairs.

    pred_logits, ref_logits: (B, N, N, K) distogram logits (same shape).
    block_positions: 0-indexed positions defining the region. If None, the
        loss is computed over the *whole* distogram (all valid pairs); if a
        list, restricted to pairs (i,j) ∈ block_positions × block_positions.

    Loss = mean_{(i,j) ∈ region, i≠j, E[d_WT(i,j)] < threshold} (E[d_WT] - E[d_pred])²
    """
    if pred_logits.shape != ref_logits.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred_logits.shape)} vs ref {tuple(ref_logits.shape)}"
        )
    if centers is None:
        centers = _bin_centers(num_bins=pred_logits.shape[-1])

    e_ref = expected_distance(ref_logits, centers)  # (B, N, N)
    e_pred = expected_distance(pred_logits, centers)  # (B, N, N)
    n = e_ref.shape[-1]

    eye = torch.eye(n, dtype=torch.bool, device=e_ref.device)[None]
    contact_mask = (e_ref < contact_threshold_a) & ~eye  # (B, N, N)

    if block_positions is not None:
        block_idx = torch.tensor(block_positions, device=e_ref.device, dtype=torch.long)
        block_mask_1d = torch.zeros(n, dtype=torch.bool, device=e_ref.device)
        block_mask_1d[block_idx] = True
        pair_in_block = block_mask_1d[None, :, None] & block_mask_1d[None, None, :]
        contact_mask = contact_mask & pair_in_block

    diffs2 = (e_ref - e_pred) ** 2
    masked = diffs2 * contact_mask
    denom = contact_mask.sum().clamp(min=1).to(diffs2.dtype)
    loss = masked.sum() / denom

    info = {
        "n_contacts": int(contact_mask.sum()),
        "denom": float(denom),
        "n_block_positions": len(block_positions) if block_positions is not None else n,
        "contact_threshold_a": contact_threshold_a,
        "region": "block" if block_positions is not None else "whole",
    }
    return loss, info


# ---------------------------------------------------------------------------
# Hook plumbing — capture per-Pairformer-layer pair representation z
# ---------------------------------------------------------------------------


class LayerCapture:
    """Captures input-embedder output, distogram, and per-Pairformer-layer z grad.

    - input_embedder is outside activation checkpointing → forward hook +
      retain_grad on the captured tensor; backward populates .grad.
    - distogram_module is the loss target, also outside checkpointing →
      forward hook to capture the logits tensor.
    - Pairformer layers run *inside* `torch.utils.checkpoint.checkpoint`, so
      forward-hook + retain_grad is fragile (the recomputed-during-backward
      tensor is what carries the gradient). Use backward hooks instead:
      `register_full_backward_hook` fires when gradient flows through the
      layer's outputs, capturing `grad_output[1]` (the gradient on the layer's
      z output) directly.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.captured: dict[str, torch.Tensor] = {}
        self.layer_grad_z: list[torch.Tensor | None] = []
        self._handles: list = []
        self._n_layers: int = 0

    def install(self) -> None:
        embedder = getattr(self.model, "input_embedder", None)
        if embedder is None:
            raise AttributeError("model has no .input_embedder")
        self._handles.append(
            embedder.register_forward_hook(self._make_forward_capture_hook("query_emb"))
        )

        # MSA channel: hook msa_proj's output. msa_proj sits BEFORE the
        # checkpointed MSA blocks (`trunkv2.py:638`), so retain_grad on its
        # output works the same way as input_embedder. Output shape is
        # (B, S, N, msa_s) — gradient there gives per-row, per-column
        # MSA-channel attribution.
        msa_module = getattr(self.model, "msa_module", None)
        if msa_module is not None and hasattr(msa_module, "msa_proj"):
            self._handles.append(
                msa_module.msa_proj.register_forward_hook(
                    self._make_forward_capture_hook("msa_emb")
                )
            )

        distogram_module = getattr(self.model, "distogram_module", None)
        if distogram_module is None:
            raise AttributeError("model has no .distogram_module")
        self._handles.append(
            distogram_module.register_forward_hook(
                self._make_forward_capture_hook("distogram")
            )
        )

        pairformer = getattr(self.model, "pairformer_module", None)
        if pairformer is None:
            raise AttributeError("model has no .pairformer_module")
        layers = getattr(pairformer, "layers", None)
        if layers is None:
            raise AttributeError("pairformer_module has no .layers")
        self._n_layers = len(layers)
        self.layer_grad_z = [None] * self._n_layers
        for li, layer in enumerate(layers):
            self._handles.append(
                layer.register_full_backward_hook(self._make_layer_backward_hook(li))
            )

    def _make_forward_capture_hook(self, key: str):
        captured = self.captured

        def hook(_mod, _inp, out):
            tensor = out if isinstance(out, torch.Tensor) else out[0]
            if tensor.requires_grad:
                tensor.retain_grad()
            captured[key] = tensor

        return hook

    def _make_layer_backward_hook(self, layer_idx: int):
        layer_grad_z = self.layer_grad_z

        def hook(_mod, _grad_input, grad_output):
            # Pairformer layer outputs (s, z) — grad_output is (grad_s, grad_z).
            if not isinstance(grad_output, tuple) or len(grad_output) < 2:
                raise RuntimeError(
                    f"pairformer layer {layer_idx}: unexpected grad_output type "
                    f"{type(grad_output)} len={len(grad_output) if hasattr(grad_output, '__len__') else 'n/a'}"
                )
            grad_z = grad_output[1]
            if grad_z is None:
                return
            # Reduce in-hook: norm over hidden dim → (B, N, N), then to CPU.
            # Keeps GPU memory minimal during backward (64 layers × ~30 MB
            # would otherwise live on device until after backward returns).
            layer_grad_z[layer_idx] = (
                grad_z[0].detach().float().norm(dim=-1).cpu()  # (N, N)
            )

        return hook

    @property
    def n_layers(self) -> int:
        return self._n_layers

    def clear(self) -> None:
        self.captured.clear()
        for i in range(len(self.layer_grad_z)):
            self.layer_grad_z[i] = None

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mutant_yaml", type=Path)
    p.add_argument("--wt_distogram", type=Path, required=True,
                   help="Saved WT distogram .pt from capture_distogram.py.")
    p.add_argument("--b_json", type=Path, required=True,
                   help="JSON with `expanded_B` (1-indexed positions) — see "
                        "scripts/analyze_chromophore_block.py.")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--cache", type=str, default="~/.boltz")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--recycling_steps", type=int, default=1,
                   help="Lower = less activation memory. Start at 1 for first run.")
    p.add_argument("--contact_threshold_a", type=float, default=8.0)
    p.add_argument(
        "--whole_structure",
        action="store_true",
        help="Compute the loss over the whole distogram (all WT contact pairs) "
             "instead of restricting to B. Useful as a complement: compare "
             "attribution maps to see whether the chromophore-block signal is "
             "specific or just a slice of a uniform whole-fold signal.",
    )
    p.add_argument("--no_kernels", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    warnings.filterwarnings(
        "ignore", ".*that has Tensor Cores. To properly utilize them.*"
    )
    torch.set_float32_matmul_precision("highest")
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    if args.seed is not None:
        seed_everything(args.seed)
    for k in ("CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"):
        os.environ.setdefault(k, "1")

    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)

    out_dir = Path(args.out_dir).expanduser() / f"attr_{args.mutant_yaml.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load WT reference distogram + B residue list
    wt_state = torch.load(args.wt_distogram, map_location="cpu", weights_only=False)
    wt_logits = wt_state["distogram"].float()  # (1, N, N, K)
    wt_token_mask = wt_state["token_mask"]
    print(
        f"[attr] WT ref: {args.wt_distogram.name} "
        f"shape={tuple(wt_logits.shape)} record={wt_state.get('record_id')}"
    )

    with args.b_json.open() as f:
        b_meta = json.load(f)
    b_1indexed = b_meta["expanded_B"]
    block_positions = [p - 1 for p in b_1indexed]  # 0-indexed
    print(
        f"[attr] B: {len(block_positions)} positions, "
        f"shell_radius={b_meta.get('shell_radius_A')}Å, "
        f"first 5 = {block_positions[:5]}"
    )

    # ---- Stage the mutant YAML for Boltz's pipeline
    staging_dir = out_dir / "_staging"
    if staging_dir.exists():
        for f in staging_dir.iterdir():
            if f.is_file() or f.is_symlink():
                f.unlink()
    staging_dir.mkdir(parents=True, exist_ok=True)
    link = staging_dir / args.mutant_yaml.name
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(args.mutant_yaml.resolve(), link)

    data = check_inputs(staging_dir)
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=False,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        boltz2=True,
    )
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    filtered_manifest = filter_inputs_structure(
        manifest=manifest, outdir=out_dir, override=True
    )
    if not filtered_manifest.records:
        raise SystemExit("No inputs survived structure filtering.")
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(processed_dir / "constraints")
        if (processed_dir / "constraints").exists()
        else None,
        template_dir=(processed_dir / "templates")
        if (processed_dir / "templates").exists()
        else None,
        extra_mols_dir=(processed_dir / "mols")
        if (processed_dir / "mols").exists()
        else None,
    )
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mol_dir,
        num_workers=args.num_workers,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
    )
    data_module.setup(stage="predict")
    dataloader = data_module.predict_dataloader()

    # ---- Load model with activation_checkpointing turned on
    checkpoint = args.checkpoint or cache / "boltz2_conf.ckpt"
    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2(activation_checkpointing=True)
    msa_args = MSAModuleArgs(
        subsample_msa=True,
        use_paired_feature=True,
        activation_checkpointing=True,
    )
    steering_args = BoltzSteeringParams()
    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "max_parallel_samples": None,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    model = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=not args.no_kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
        checkpoint_diffusion_conditioning=True,
    )
    model.eval()

    # Boltz2.__init__ freezes trunk params at construction time when the
    # loaded checkpoint has structure_prediction_training=False (`boltz2.py:351-358`).
    # That happens before our `_train_mode_for_boltz_grads` flips the flag, so the
    # params stay frozen → input_embedder's output ends up with requires_grad=False
    # and the trunk graph is dead. Unfreeze everything explicitly here. We don't
    # *use* the param grads, but they need to exist for activations to carry grad.
    n_unfrozen = 0
    for p in model.parameters():
        if not p.requires_grad:
            p.requires_grad_(True)
            n_unfrozen += 1
    print(f"[attr] unfroze {n_unfrozen} trunk params (set requires_grad=True)")

    device = torch.device(
        "cuda" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    n_pairformer = len(getattr(model.pairformer_module, "layers", []))
    print(
        f"[attr] model loaded — pairformer layers={n_pairformer} "
        f"activation_checkpointing=True"
    )

    # ---- Install hooks
    cap = LayerCapture(model)
    cap.install()

    # ---- Diagnostics: trace grad/training state through forward
    def _diag_pre_model(_mod, _args, _kwargs):
        print(
            f"[diag] BOLTZ2.forward entry: model.training={model.training} "
            f"struct_pred_training={getattr(model, 'structure_prediction_training', 'N/A')} "
            f"grad_enabled={torch.is_grad_enabled()}"
        )

    def _diag_post_embedder(_mod, _inp, out):
        t = out if isinstance(out, torch.Tensor) else out[0]
        print(
            f"[diag] input_embedder output: requires_grad={t.requires_grad} "
            f"grad_enabled={torch.is_grad_enabled()}"
        )

    def _diag_pre_distogram(_mod, _args):
        z = _args[0] if _args else None
        zr = (z.requires_grad if isinstance(z, torch.Tensor) else "n/a")
        print(
            f"[diag] distogram_module entry: input z.requires_grad={zr} "
            f"grad_enabled={torch.is_grad_enabled()} "
            f"model.training={model.training}"
        )

    diag_handles = [
        model.register_forward_pre_hook(_diag_pre_model, with_kwargs=True),
        model.input_embedder.register_forward_hook(_diag_post_embedder),
        model.distogram_module.register_forward_pre_hook(_diag_pre_distogram),
    ]

    # ---- Forward + backward
    batch = next(iter(dataloader))
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
    record_id = batch["record"][0].id
    print(f"[attr] forward: record_id={record_id}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # NOTE: don't freeze params. The legacy install() / retain_grad() approach
    # needs the trunk's autograd graph alive end-to-end. With all params
    # frozen AND integer-indexed residue inputs, no tensor in the trunk has
    # requires_grad=True → checkpoint() warns "None of the inputs have
    # requires_grad=True. Gradients will be None" and the captured distogram
    # has requires_grad=False. The pair-rep-leaf mode could afford to freeze
    # because it injected a leaf at distogram_module's input as a grad source;
    # we don't have one. Param grads accumulate (~2.4 GB) but we ignore them.
    t0 = time.time()
    try:
        with (
            torch.enable_grad(),
            _train_mode_for_boltz_grads(model),
            _temporarily(model, "skip_run_structure", True),
        ):
            cap.clear()
            _ = model(
                batch,
                recycling_steps=args.recycling_steps,
                num_sampling_steps=200,
                diffusion_samples=1,
            )

            # Validate forward captures (backward-side captures checked after backward)
            if "distogram" not in cap.captured:
                raise RuntimeError("distogram_module hook didn't fire")
            if "query_emb" not in cap.captured:
                raise RuntimeError("input_embedder hook didn't fire")
            pred_logits = cap.captured["distogram"]
            if pred_logits.ndim == 5:  # (B, N, N, num_distograms, num_bins)
                pred_logits = pred_logits[..., 0, :]
            if not pred_logits.requires_grad:
                raise RuntimeError(
                    "captured distogram doesn't require grad — grad-gating "
                    "workaround may not have fired. "
                    f"global grad_enabled={torch.is_grad_enabled()}, "
                    f"model.training={model.training}"
                )

            # Slice WT to match pred shape (handles (B, N, N, num_dists, K) ref)
            ref = wt_logits.to(device)
            if ref.ndim == 5:
                ref = ref[..., 0, :]
            loss_block_positions = None if args.whole_structure else block_positions
            loss, info = expected_distance_l2(
                pred_logits=pred_logits,
                ref_logits=ref,
                block_positions=loss_block_positions,
                contact_threshold_a=args.contact_threshold_a,
            )
            print(
                f"[attr] region={info['region']}  loss={float(loss):.4f}  "
                f"n_contacts={info['n_contacts']}  denom={info['denom']:.1f}"
            )
            loss.backward()
    finally:
        forward_backward_seconds = time.time() - t0
        cap.remove()
        for h in diag_handles:
            h.remove()

    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"[attr] peak GPU memory: {peak_gb:.2f} GB")
    else:
        peak_gb = float("nan")

    print(f"[attr] forward+backward time: {forward_backward_seconds:.1f}s")

    # ---- Aggregate gradients to per-(i,j) magnitudes
    # Input embedding: (B, N, D_s) — norm over hidden dim → (N,)
    query_t = cap.captured["query_emb"]
    if query_t.grad is None:
        print("[warn] query_emb.grad is None — input attribution unavailable")
        input_grad_norm = torch.zeros(query_t.shape[1])
    else:
        input_grad_norm = query_t.grad[0].detach().float().norm(dim=-1).cpu()  # (N,)

    # MSA embedding: (B, S, N, msa_s) — L2 over (S, msa_s) → (N,)
    msa_grad_norm = None
    msa_t = cap.captured.get("msa_emb")
    if msa_t is None:
        print("[warn] msa_emb not captured — model has no msa_module.msa_proj?")
    elif msa_t.grad is None:
        print("[warn] msa_emb.grad is None — MSA-channel attribution unavailable")
        msa_grad_norm = torch.zeros(msa_t.shape[2])
    else:
        # (1, S, N, msa_s) → norm over S and msa_s → (N,)
        msa_grad_norm = msa_t.grad[0].detach().float().norm(dim=(0, 2)).cpu()  # (N,)

    # Pairformer layers: backward hooks already produced (N, N) per-layer norms
    # on CPU (hook reduces grad_output[1] in-place to keep GPU minimal).
    N = pred_logits.shape[1]
    pair_grad_norms = []
    n_missing = 0
    for li, grad_norm in enumerate(cap.layer_grad_z):
        if grad_norm is None:
            n_missing += 1
            pair_grad_norms.append(torch.zeros(N, N))
            continue
        pair_grad_norms.append(grad_norm)
    pair_grad_stack = torch.stack(pair_grad_norms, dim=0)  # (n_layers, N, N)
    if n_missing:
        print(
            f"[warn] {n_missing}/{n_pairformer} pairformer-layer backward "
            f"hooks didn't fire (grad_output was None or hook didn't trigger)"
        )

    # ---- Save
    region_tag = "whole" if args.whole_structure else "block"
    out_path = out_dir / f"attribution_{args.mutant_yaml.stem}_{region_tag}.pt"
    save_dict = {
        "record_id": record_id,
        "mutant_yaml": str(args.mutant_yaml),
        "wt_distogram_path": str(args.wt_distogram),
        "b_json_path": str(args.b_json),
        "block_positions_0idx": block_positions,
        "block_positions_1idx": b_1indexed,
        "shell_radius_A": b_meta.get("shell_radius_A"),
        "recycling_steps": args.recycling_steps,
        "contact_threshold_a": args.contact_threshold_a,
        "region": info["region"],
        "loss": float(loss),
        "n_contacts": info["n_contacts"],
        "input_grad_norm": input_grad_norm,  # (N,)
        "pair_grad_norm": pair_grad_stack,  # (n_layers, N, N)
        "n_pairformer_layers": n_pairformer,
        "n_missing_layer_grads": n_missing,
        "peak_gpu_gb": peak_gb,
        "forward_backward_seconds": forward_backward_seconds,
        "token_mask": batch["token_pad_mask"].cpu().bool(),
    }
    if msa_grad_norm is not None:
        save_dict["msa_grad_norm"] = msa_grad_norm  # (N,)
    torch.save(save_dict, out_path)
    print(f"[attr] wrote -> {out_path}")
    print(
        f"[attr] input_grad_norm: max={float(input_grad_norm.max()):.3e} "
        f"mean={float(input_grad_norm.mean()):.3e}"
    )
    if msa_grad_norm is not None:
        print(
            f"[attr] msa_grad_norm:   max={float(msa_grad_norm.max()):.3e} "
            f"mean={float(msa_grad_norm.mean()):.3e}"
        )
    print(
        f"[attr] pair_grad_norm:  max={float(pair_grad_stack.max()):.3e} "
        f"mean={float(pair_grad_stack.mean()):.3e}"
    )

    # Cleanup
    del cap, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
