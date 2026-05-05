"""Per-step attribution runner: forward + backward at each recycling depth.

Pair-rep mode (default): one ``forward(recycling_steps=K)`` per K. All model
parameters are frozen so the trunk runs without an autograd graph; a
forward_pre_hook on ``distogram_module`` replaces its input ``z`` with a leaf
that requires grad. Backward populates ``leaf.grad`` — that's the attribution
on the final pair representation. Memory bounded to one trunk forward +
distogram_module's tiny graph regardless of K.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import torch
import torch.nn as nn

from .capture import GradientCapture
from .io import AttributionResult, collect_provenance
from .targets import AttributionTarget


@contextlib.contextmanager
def _train_mode_for_boltz_grads(model: nn.Module):
    """Make Boltz2's grad gates evaluate True only on the last recycle.

    Boltz2.forward gates grads three ways (boltz2.py:411, 440, 550–583):
      - L411 (outer):   ``set_grad_enabled(self.training and self.structure_prediction_training)``
      - L440 (inner):   ``set_grad_enabled(... and (i == recycling_steps))``
      - L550–583 (post-distogram): ``if self.training: ...`` branches that
        require ``feats["coords"]``.

    With ``model.eval()`` (default), the outer + inner gates evaluate False
    and our backward graph is dead. Neutralising ``set_grad_enabled``
    altogether forces grads on for *every* recycle iteration, which OOMs at
    K≥5 because Boltz's whole design assumes only the last recycle is on the
    autograd graph.

    Right fix: temporarily set ``training=True`` + ``structure_prediction_training=True``
    so Boltz's gates evaluate correctly (outer True, inner True only on last
    recycle). Then a post-hook on ``distogram_module`` flips both back to
    False *immediately after distogram is computed* so the post-distogram
    branches at L550–583 don't fire (they'd crash on missing coords).
    """
    original_training = model.training
    has_struct = hasattr(model, "structure_prediction_training")
    original_struct = getattr(model, "structure_prediction_training", None)

    def _flip_back(_mod, _inp, _out):
        model.training = False
        if has_struct:
            model.structure_prediction_training = False

    distogram_module = getattr(model, "distogram_module", None)
    if distogram_module is None:
        raise AttributeError("model has no .distogram_module")
    handle = distogram_module.register_forward_hook(_flip_back)

    model.training = True
    if has_struct:
        model.structure_prediction_training = True

    try:
        yield
    finally:
        handle.remove()
        model.training = original_training
        if has_struct:
            model.structure_prediction_training = original_struct


@contextlib.contextmanager
def _temporarily(obj, attr: str, value):
    """Temporarily set ``obj.attr = value``, restoring on exit."""
    if not hasattr(obj, attr):
        yield
        return
    old = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old)


@contextlib.contextmanager
def _freeze_all_params(model: nn.Module):
    """Temporarily set ``requires_grad=False`` on all model parameters.

    Standard interpretability practice: we don't want gradients on parameters,
    only on the captured surface (the leaf injected at distogram_module's
    input). Freezing all params keeps the trunk's autograd graph empty
    regardless of whether the global grad state is on — no activations are
    retained, peak memory drops by an order of magnitude.

    Restore on exit so the model is unchanged for subsequent callers.
    """
    saved: list[tuple[torch.Tensor, bool]] = []
    for p in model.parameters():
        saved.append((p, p.requires_grad))
        if p.requires_grad:
            p.requires_grad_(False)
    try:
        yield
    finally:
        for p, was in saved:
            p.requires_grad_(was)


def run_per_step(
    model: nn.Module,
    batch: dict,
    target: AttributionTarget,
    recycling_steps: Sequence[int] = (0, 5, 10),
    *,
    record_id: str | None = None,
    forward_kwargs: dict | None = None,
    extra_provenance: dict | None = None,
) -> list[AttributionResult]:
    """Run forward+backward once per K and return one result per step.

    Args:
        model: Boltz2 instance (eval mode). Caller is responsible for ``.to(device)``.
        batch: Boltz2 batch dict (already on the model's device).
        target: scalar target on distogram logits.
        recycling_steps: K values to evaluate, e.g. ``(0, 5, 10)``.
        record_id: optional id used in saved file names (defaults to batch id).
        forward_kwargs: extra kwargs forwarded to ``model.forward`` (e.g.
            ``num_sampling_steps`` if the diffusion path must run). Sampling
            outputs are unused here.
        extra_provenance: extra fields merged into the provenance dict.

    Returns:
        A list of :class:`AttributionResult`, one per ``recycling_steps`` entry,
        in the same order. Tensors are on CPU.
    """
    if model.training:
        raise RuntimeError("model must be in eval() mode for attribution runs")

    record_id = record_id or _record_id_from_batch(batch)
    forward_kwargs = dict(forward_kwargs or {})
    token_mask = _get_token_mask(batch)

    results: list[AttributionResult] = []
    cap = GradientCapture(model)
    cap.install_pair_rep_attribution()

    # Skip the diffusion sampling path — we only need the distogram (computed
    # before diffusion). Saves forward time and avoids the downstream
    # structure_module assertions that expect ground-truth coords.
    try:
        with (
            torch.enable_grad(),
            _freeze_all_params(model),
            _train_mode_for_boltz_grads(model),
            _temporarily(model, "skip_run_structure", True),
        ):
            for k in recycling_steps:
                cap.clear()

                _ = model(batch, recycling_steps=int(k), **forward_kwargs)

                pair_leaf = cap.pair_input
                if not pair_leaf.requires_grad:
                    raise RuntimeError(
                        "pair_input leaf doesn't require grad — pre-hook on "
                        "distogram_module didn't fire or replacement was "
                        f"discarded. Captured surfaces: {cap.captured_keys()}"
                    )
                logits = cap.distogram
                # Boltz2's distogram is (B, N, N, num_distograms, num_bins).
                # Match boltz2.py:600 — only one distogram is implemented, so
                # collapse the num_distograms dim by selecting index 0. Targets
                # then see the expected (B, N, N, num_bins) shape.
                if logits.ndim == 5:
                    logits = logits[..., 0, :]
                loss = target(logits, token_mask=token_mask)
                loss.backward()

                results.append(
                    _build_result(
                        record_id=record_id,
                        recycling_steps=int(k),
                        loss_value=float(loss.detach().cpu()),
                        target=target,
                        cap=cap,
                        token_mask=token_mask,
                        extra_provenance=extra_provenance,
                    )
                )

                # Free GPU memory between K iterations — backward graph + leaf
                # are now CPU-side in the result; no reason to keep them on device.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        cap.remove()

    return results


def _build_result(
    *,
    record_id: str,
    recycling_steps: int,
    loss_value: float,
    target: AttributionTarget,
    cap: GradientCapture,
    token_mask: torch.Tensor | None,
    extra_provenance: dict | None,
) -> AttributionResult:
    pair_t = cap.pair_input
    pair_grad = _grad_or_zero(pair_t).detach().cpu()
    pair_input = pair_t.detach().cpu()

    n = pair_grad.shape[1] if pair_grad.ndim >= 3 else None
    if token_mask is not None:
        mask_cpu = token_mask.detach().cpu().bool()
    elif n is not None:
        mask_cpu = torch.ones(pair_grad.shape[0], n, dtype=torch.bool)
    else:
        mask_cpu = torch.ones(1, dtype=torch.bool)

    # Zero out attribution at padded (i, j) pair positions so downstream
    # consumers don't pick up garbage outside the real protein.
    if (
        pair_grad.ndim == 4
        and mask_cpu.ndim == 2
        and pair_grad.shape[:2] == mask_cpu.shape
    ):
        valid = mask_cpu  # (B, N)
        pair_mask = valid[:, :, None] & valid[:, None, :]  # (B, N, N)
        invalid = ~pair_mask
        pair_grad[invalid] = 0
        pair_input[invalid] = 0

    return AttributionResult(
        record_id=record_id,
        recycling_steps=recycling_steps,
        target_value=loss_value,
        target_spec=target.spec(),
        token_mask=mask_cpu,
        pair_grad=pair_grad,
        pair_input=pair_input,
        provenance=collect_provenance(extra=extra_provenance),
    )


def _grad_or_zero(t: torch.Tensor) -> torch.Tensor:
    if t.grad is None:
        return torch.zeros_like(t)
    return t.grad


def _get_token_mask(batch: dict) -> torch.Tensor | None:
    for k in ("token_pad_mask", "token_mask"):
        v = batch.get(k)
        if isinstance(v, torch.Tensor):
            return v
    return None


def _record_id_from_batch(batch: dict) -> str:
    rec = batch.get("record")
    if isinstance(rec, list) and rec and hasattr(rec[0], "id"):
        return rec[0].id
    return "record"
