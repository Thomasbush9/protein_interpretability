"""Per-step attribution runner: forward + backward at each recycling depth.

Default capture mode (v1): one ``forward(recycling_steps=K)`` per K, fresh
graph each time, gradients on query + MSA embedding only. Memory bounded to
one forward.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .capture import GradientCapture
from .io import AttributionResult, collect_provenance
from .targets import AttributionTarget


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

    with cap, torch.enable_grad():
        for k in recycling_steps:
            cap.clear()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None

            _ = model(batch, recycling_steps=int(k), **forward_kwargs)

            logits = cap.distogram
            if not logits.requires_grad:
                raise RuntimeError(
                    "distogram tensor has requires_grad=False — Boltz forward "
                    "is running under no_grad/inference_mode. Check that the "
                    "model isn't in inference_mode and that the CLI sets "
                    "torch.set_grad_enabled(True) before model(batch). "
                    f"Captured surfaces: {cap.captured_keys()}"
                )
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
    query_t = cap.query_emb
    msa_t = cap.get(GradientCapture.MSA_KEY)

    query_grad = _grad_or_zero(query_t).detach().cpu()
    query_input = query_t.detach().cpu()
    if msa_t is not None:
        msa_grad = _grad_or_zero(msa_t).detach().cpu()
        msa_input = msa_t.detach().cpu()
    else:
        msa_grad = None
        msa_input = None

    if token_mask is not None:
        mask_cpu = token_mask.detach().cpu().bool()
    else:
        mask_cpu = torch.ones(query_grad.shape[:-1], dtype=torch.bool)

    if mask_cpu is not None and query_grad.shape[:-1] == mask_cpu.shape:
        invalid = ~mask_cpu
        query_grad[invalid] = 0
        query_input[invalid] = 0

    return AttributionResult(
        record_id=record_id,
        recycling_steps=recycling_steps,
        target_value=loss_value,
        target_spec=target.spec(),
        query_grad=query_grad,
        query_input=query_input,
        msa_grad=msa_grad,
        msa_input=msa_input,
        token_mask=mask_cpu,
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
