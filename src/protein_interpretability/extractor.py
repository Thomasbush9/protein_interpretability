# esm_activation_extractor.py
from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn as nn

from esm.models.esm3 import ESM3


class ESM3ActivationExtractor:
    """
    Extract activations from ESM3 forward/generation. Register hooks for
    encoder, transformer (final normalized output), per-layer block outputs,
    per-layer attention outputs, geometric attention outputs, and FFN outputs.
    Attention *weights* are not available (SDPA does not return them).
    """

    SITES = (
        "encoder",           # encoder output (B, L, D)
        "transformer_out",   # final normalized hidden (B, L, D)
        "transformer_embed", # pre-norm final hidden (B, L, D)
        "output_heads_in",   # input to output heads (B, L, D)
    )
    LAYER_SITES = ("block", "attn", "geom_attn", "ffn")

    def __init__(
        self,
        model: ESM3,
        sites: list[str] | None = None,
        layers: list[int] | None = None,
        layer_sites: list[str] | None = None,
        detach: bool = True,
    ):
        """
        Args:
            model: ESM3 model (e.g. from ESM3.from_pretrained()).
            sites: Which top-level sites to capture. None = all SITES.
            layers: Which transformer block indices to capture (e.g. [0, 5, 11]).
                   None = all layers.
            layer_sites: Which per-layer parts: "block", "attn", "geom_attn", "ffn".
                         None = ["block"].
            detach: Whether to detach and clone stored tensors.
        """
        self.model = model
        self.sites = sites or list(self.SITES)
        self.layers = layers
        self.layer_sites = layer_sites or ["block"]
        self.detach = detach
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.activations: list[dict[str, torch.Tensor]] = []

    def _store(self, key: str, value: torch.Tensor) -> None:
        if not self.activations:
            self.activations.append({})
        v = value.detach().clone() if self.detach else value
        self.activations[-1][key] = v

    def _tensor_from_out(self, out: Any) -> torch.Tensor:
        return out[0] if isinstance(out, tuple) else out

    def _install(self) -> None:
        n_blocks = len(self.model.transformer.blocks)
        layer_indices = self.layers if self.layers is not None else list(range(n_blocks))

        # --- encoder: start new step + optional encoder output
        def encoder_hook(module: nn.Module, input: Any, output: torch.Tensor) -> torch.Tensor:
            self.activations.append({})
            if "encoder" in self.sites:
                self._store("encoder", output)
            return output

        self._handles.append(self.model.encoder.register_forward_hook(encoder_hook))

        # --- transformer: normalized out + pre-norm embed
        def transformer_hook(module: nn.Module, input: Any, output: tuple) -> None:
            normed, pre_norm, _ = output
            if "transformer_out" in self.sites:
                self._store("transformer_out", normed)
            if "transformer_embed" in self.sites:
                self._store("transformer_embed", pre_norm)

        self._handles.append(self.model.transformer.register_forward_hook(transformer_hook))

        # --- output_heads: first input is the hidden state
        def output_heads_hook(module: nn.Module, input: tuple, output: Any) -> None:
            if "output_heads_in" in self.sites:
                self._store("output_heads_in", self._tensor_from_out(input[0]))

        self._handles.append(self.model.output_heads.register_forward_hook(output_heads_hook))

        # --- per-layer: blocks[i], attn, geom_attn, ffn
        for i in layer_indices:
            if i < 0 or i >= n_blocks:
                continue
            block = self.model.transformer.blocks[i]

            if "block" in self.layer_sites:
                def block_hook(module: nn.Module, input: Any, output: torch.Tensor, idx: int = i) -> torch.Tensor:
                    self._store(f"layer_{idx}_block", output)
                    return output
                self._handles.append(block.register_forward_hook(block_hook))

            if "attn" in self.layer_sites and getattr(block, "attn", None) is not None:
                def attn_hook(module: nn.Module, input: Any, output: torch.Tensor, idx: int = i) -> torch.Tensor:
                    self._store(f"layer_{idx}_attn", output)
                    return output
                self._handles.append(block.attn.register_forward_hook(attn_hook))

            if "geom_attn" in self.layer_sites and getattr(block, "geom_attn", None) is not None:
                def geom_hook(module: nn.Module, input: Any, output: torch.Tensor, idx: int = i) -> torch.Tensor:
                    self._store(f"layer_{idx}_geom_attn", output)
                    return output
                self._handles.append(block.geom_attn.register_forward_hook(geom_hook))

            if "ffn" in self.layer_sites:
                def ffn_hook(module: nn.Module, input: Any, output: torch.Tensor, idx: int = i) -> torch.Tensor:
                    self._store(f"layer_{idx}_ffn", output)
                    return output
                self._handles.append(block.ffn.register_forward_hook(ffn_hook))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self) -> None:
        """Clear stored activations (hooks remain)."""
        self.activations.clear()

    @contextlib.contextmanager
    def recording(self):
        """Context manager: install hooks, yield, then remove hooks."""
        self._install()
        try:
            yield self
        finally:
            self.remove()

    def __enter__(self) -> ESM3ActivationExtractor:
        self._install()
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()

    # --- Convenience accessors
    def get_step(self, step: int) -> dict[str, torch.Tensor]:
        """Activations for a single forward step (0-indexed)."""
        return self.activations[step] if 0 <= step < len(self.activations) else {}

    def get_site(self, name: str) -> list[torch.Tensor]:
        """List of tensors for one site across all steps."""
        return [s[name] for s in self.activations if name in s]

    def step_keys(self) -> list[str]:
        """All site names that appear in at least one step."""
        out: set[str] = set()
        for s in self.activations:
            out.update(s.keys())
        return sorted(out)

    def get_mean_activations_per_layer(
        self,
        step: int | None = None,
        reduce_batch: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Mean activation per layer (over sequence dim, optionally over batch).

        Only layer keys (layer_*_block, layer_*_attn, etc.) are included.
        If multiple steps exist, they are averaged first.

        Args:
            step: If set, use only this step; else average over all steps.
            reduce_batch: If True and tensor has batch dim (ndim > 2),
                reduce over batch first. Output then (D,) per layer.
                If False, output is (B, D) when batched.

        Returns:
            dict[layer_key] -> tensor of shape (D,) or (B, D).
        """
        layer_keys = [k for k in self.step_keys() if k.startswith("layer_")]
        out: dict[str, torch.Tensor] = {}
        for k in layer_keys:
            if step is not None:
                d = self.get_step(step)
                if k not in d:
                    continue
                t = d[k]
            else:
                tensors = self.get_site(k)
                if not tensors:
                    continue
                t = torch.stack(tensors, dim=0).mean(dim=0)
            if t.dim() > 2 and reduce_batch:
                t = t.mean(dim=0)
            t = t.mean(dim=-2)
            out[k] = t
        return out