"""General-purpose hidden representation extractor for Boltz2.

Captures activations from any layer/module of Boltz2 — sequence states (s),
pairwise states (z), attention weights, triangle module outputs, etc. — using
forward hooks and targeted monkey-patching.

Usage::

    from protein_interpretability.extractor_boltz import Boltz2Extractor

    extractor = Boltz2Extractor(model, sites=["pairformer_s", "pairformer_z"])
    with extractor:
        output = model(batch, recycling_steps=3, ...)
        # extractor.activations has one dict per recycling step
        final = extractor.get_step(-1)
"""

from __future__ import annotations

import contextlib
from functools import wraps
from typing import Any

import torch
import torch.nn as nn

from boltz.model.models.boltz2 import Boltz2


class Boltz2Extractor:
    """Extract hidden representations from Boltz2 modules via hooks.

    Follows the same API as :class:`ESM3ActivationExtractor`.
    """

    SITES = (
        "input_embedder",   # (B, N, token_s) — token embeddings from atoms
        "pairformer_s",     # (B, N, token_s) — final pairformer sequence output
        "pairformer_z",     # (B, N, N, token_z) — final pairformer pairwise output
        "distogram",        # (B, N, N, num_bins) — distance prediction logits
    )

    LAYER_SITES = (
        "layer_s",                # per-layer sequence output (B, N, token_s)
        "layer_z",                # per-layer pairwise output (B, N, N, token_z)
        "attention",              # AttentionPairBias output on s
        "attention_weights",      # post-softmax attention matrix (B, H, N, N)
        "tri_mul_out",            # TriangleMultiplicationOutgoing output
        "tri_mul_in",             # TriangleMultiplicationIncoming output
        "tri_att_start",          # TriangleAttentionStartingNode output (B, N, N, D)
        "tri_att_end",            # TriangleAttentionEndingNode output (B, N, N, D)
        "tri_att_start_weights",  # post-softmax weights (B, I, H, J, J) — forces eager path
        "tri_att_end_weights",    # post-softmax weights (B, I, H, J, J) — forces eager path
        "transition_s",           # sequence transition output
        "transition_z",           # pairwise transition output
    )

    def __init__(
        self,
        model: Boltz2,
        sites: list[str] | None = None,
        layers: list[int] | None = None,
        layer_sites: list[str] | None = None,
        detach: bool = True,
        to_cpu: bool = True,
    ):
        """
        Args:
            model: Boltz2 model instance.
            sites: Top-level sites to capture. ``None`` = all :attr:`SITES`.
            layers: Pairformer layer indices to hook. ``None`` = all layers.
            layer_sites: Per-layer sites to capture.
                ``None`` = ``["layer_s", "layer_z"]``.
            detach: Detach and clone stored tensors (saves memory graph).
            to_cpu: Move stored tensors to CPU (saves GPU memory).
        """
        self.model = model
        self.sites = sites if sites is not None else list(self.SITES)
        self.layers = layers
        self.layer_sites = layer_sites if layer_sites is not None else ["layer_s", "layer_z"]
        self.detach = detach
        self.to_cpu = to_cpu

        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._patched: list[tuple[nn.Module, str, Any]] = []  # (module, attr, original)
        self._installed: bool = False
        self._pairformer_calls: int = 0
        self.activations: list[dict[str, torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(module: nn.Module) -> nn.Module:
        """Unwrap ``torch.compile``'s ``_orig_mod`` wrapper if present."""
        return getattr(module, "_orig_mod", module)

    def _store(self, key: str, value: torch.Tensor) -> None:
        if not self.activations:
            self.activations.append({})
        v = value.detach().clone() if self.detach else value
        if self.to_cpu and v.is_cuda:
            v = v.cpu()
        self.activations[-1][key] = v

    def _tensor_from_out(self, out: Any) -> torch.Tensor:
        return out[0] if isinstance(out, tuple) else out

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Install hooks and monkey-patches on the model.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._installed:
            return

        pairformer = self._unwrap(self.model.pairformer_module)
        n_layers = len(pairformer.layers)
        layer_indices = self.layers if self.layers is not None else list(range(n_layers))

        # --- Validate layer indices up-front
        invalid = [i for i in layer_indices if i < 0 or i >= n_layers]
        if invalid:
            raise ValueError(
                f"Layer indices {invalid} are out of range for "
                f"{n_layers}-layer pairformer (valid: 0..{n_layers - 1})"
            )

        # --- Recycling-step boundary: pre-hook on pairformer appends new dict.
        # On the *first* pairformer call we reuse the dict already seeded by
        # input_embedder (if any); subsequent calls open a new recycling step.
        def pairformer_pre_hook(module: nn.Module, inputs: Any) -> None:
            if self._pairformer_calls > 0:
                self.activations.append({})
            elif not self.activations:
                self.activations.append({})
            self._pairformer_calls += 1

        self._handles.append(
            pairformer.register_forward_pre_hook(pairformer_pre_hook)
        )

        # --- Top-level: input_embedder
        if "input_embedder" in self.sites:
            def input_embedder_hook(module: nn.Module, inp: Any, output: torch.Tensor) -> None:
                self._store("input_embedder", self._tensor_from_out(output))

            self._handles.append(
                self.model.input_embedder.register_forward_hook(input_embedder_hook)
            )

        # --- Top-level: pairformer_s / pairformer_z (post-hook on pairformer)
        if "pairformer_s" in self.sites or "pairformer_z" in self.sites:
            def pairformer_hook(module: nn.Module, inp: Any, output: tuple) -> None:
                s, z = output
                if "pairformer_s" in self.sites:
                    self._store("pairformer_s", s)
                if "pairformer_z" in self.sites:
                    self._store("pairformer_z", z)

            self._handles.append(
                pairformer.register_forward_hook(pairformer_hook)
            )

        # --- Top-level: distogram
        if "distogram" in self.sites:
            def distogram_hook(module: nn.Module, inp: Any, output: torch.Tensor) -> None:
                self._store("distogram", self._tensor_from_out(output))

            self._handles.append(
                self.model.distogram_module.register_forward_hook(distogram_hook)
            )

        # --- Per-layer hooks (indices already validated above)
        for i in layer_indices:
            layer = pairformer.layers[i]
            prefix = f"pairformer_{i}"

            # layer_s / layer_z: post-hook on PairformerLayer
            if "layer_s" in self.layer_sites or "layer_z" in self.layer_sites:
                def layer_hook(mod: nn.Module, inp: Any, output: tuple, p: str = prefix) -> None:
                    s, z = output
                    if "layer_s" in self.layer_sites:
                        self._store(f"{p}_layer_s", s)
                    if "layer_z" in self.layer_sites:
                        self._store(f"{p}_layer_z", z)

                self._handles.append(layer.register_forward_hook(layer_hook))

            # attention (module output, not weights)
            if "attention" in self.layer_sites and hasattr(layer, "attention"):
                def attn_hook(mod: nn.Module, inp: Any, output: torch.Tensor, p: str = prefix) -> None:
                    self._store(f"{p}_attention", self._tensor_from_out(output))

                self._handles.append(layer.attention.register_forward_hook(attn_hook))

            # attention_weights: monkey-patch forward to capture post-softmax weights
            if "attention_weights" in self.layer_sites and hasattr(layer, "attention"):
                self._patch_attention_weights(layer.attention, f"{prefix}_attention_weights")

            # Triangle and transition submodules (plain output hooks)
            _sub_map = {
                "tri_mul_out": "tri_mul_out",
                "tri_mul_in": "tri_mul_in",
                "tri_att_start": "tri_att_start",
                "tri_att_end": "tri_att_end",
                "transition_s": "transition_s",
                "transition_z": "transition_z",
            }
            for site_name, attr_name in _sub_map.items():
                if site_name in self.layer_sites and hasattr(layer, attr_name):
                    submod = getattr(layer, attr_name)

                    def sub_hook(mod: nn.Module, inp: Any, output: torch.Tensor,
                                 p: str = prefix, sn: str = site_name) -> None:
                        self._store(f"{p}_{sn}", self._tensor_from_out(output))

                    self._handles.append(submod.register_forward_hook(sub_hook))

            # Triangle attention weights (monkey-patch; forces eager softmax path)
            _tri_weight_map = {
                "tri_att_start_weights": "tri_att_start",
                "tri_att_end_weights": "tri_att_end",
            }
            for site_name, attr_name in _tri_weight_map.items():
                if site_name in self.layer_sites and hasattr(layer, attr_name):
                    submod = getattr(layer, attr_name)
                    self._patch_tri_attention_weights(
                        submod, f"{prefix}_{site_name}"
                    )

        self._installed = True

    def _patch_attention_weights(self, module: nn.Module, key: str) -> None:
        """Monkey-patch AttentionPairBias.forward to capture post-softmax weights."""
        original_forward = module.forward
        extractor = self

        @wraps(original_forward)
        def patched_forward(s, z, mask, k_in, multiplicity=1):
            B = s.shape[0]

            q = module.proj_q(s).view(B, -1, module.num_heads, module.head_dim)
            k = module.proj_k(k_in).view(B, -1, module.num_heads, module.head_dim)
            v = module.proj_v(k_in).view(B, -1, module.num_heads, module.head_dim)

            bias = module.proj_z(z)
            bias = bias.repeat_interleave(multiplicity, 0)

            g = module.proj_g(s).sigmoid()

            with torch.autocast("cuda", enabled=False):
                attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
                attn = attn / (module.head_dim**0.5) + bias.float()
                attn = attn + (1 - mask[:, None, None].float()) * -module.inf
                attn = attn.softmax(dim=-1)

                # Capture post-softmax attention weights
                extractor._store(key, attn)

                o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
            o = o.reshape(B, -1, module.c_s)
            o = module.proj_o(g * o)
            return o

        module.forward = patched_forward
        self._patched.append((module, "forward", original_forward))

    def _patch_tri_attention_weights(self, tri_module: nn.Module, key: str) -> None:
        """Monkey-patch ``TriangleAttention.forward`` to capture post-softmax weights.

        The fused CUDA kernels (``trifast`` / ``cuequivariance``) never
        materialize the softmax matrix, so this replacement re-implements the
        eager path — same math, same return value — with a ``_store`` call
        slipped in between softmax and the final ``@ V``.

        Captured tensor shape: ``(B, I, H, J, J)``. For ending-node attention
        ``x`` is transposed first, so ``I`` indexes the original ``J`` axis.
        """
        from boltz.model.layers.triangular_attention.primitives import softmax_no_cast
        from boltz.model.layers.triangular_attention.utils import permute_final_dims

        original_forward = tri_module.forward
        extractor = self

        @wraps(original_forward)
        def patched_forward(x, mask=None, chunk_size=None, use_kernels=False):
            if mask is None:
                mask = x.new_ones(x.shape[:-1])

            if not tri_module.starting:
                x = x.transpose(-2, -3)
                mask = mask.transpose(-1, -2)

            x = tri_module.layer_norm(x)

            # [*, I, 1, 1, J]
            mask = mask[..., :, None, None, :]
            mask_bias = tri_module.inf * (mask - 1)

            # [*, H, I, J] -> [*, 1, H, I, J]
            tri_bias = permute_final_dims(tri_module.linear(x), (2, 0, 1))
            tri_bias = tri_bias.unsqueeze(-4)

            # Inline MHA eager path with weight capture
            mha = tri_module.mha
            q, k, v = mha._prep_qkv(x, x, apply_scale=True)
            # q/k/v: [B, I, H, J, C_hidden]

            key_t = permute_final_dims(k, (1, 0))   # [B, I, H, C_hidden, J]
            a = torch.matmul(q, key_t)              # [B, I, H, J, J]
            a = a + mask_bias
            a = a + tri_bias
            a = softmax_no_cast(a, -1)

            extractor._store(key, a)

            o = torch.matmul(a, v)                  # [B, I, H, J, C_hidden]
            o = o.transpose(-2, -3)                 # [B, I, J, H, C_hidden]
            o = mha._wrap_up(o, x)

            if not tri_module.starting:
                o = o.transpose(-2, -3)

            return o

        tri_module.forward = patched_forward
        self._patched.append((tri_module, "forward", original_forward))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def remove(self) -> None:
        """Remove all hooks and restore monkey-patched methods."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

        for module, attr, original in self._patched:
            setattr(module, attr, original)
        self._patched.clear()
        self._installed = False

    def clear(self) -> None:
        """Clear stored activations (hooks remain installed)."""
        self.activations.clear()
        self._pairformer_calls = 0

    @contextlib.contextmanager
    def recording(self):
        """Context manager: install hooks, yield, then remove hooks."""
        self.install()
        try:
            yield self
        finally:
            self.remove()

    def __enter__(self) -> Boltz2Extractor:
        self.install()
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_step(self, step: int) -> dict[str, torch.Tensor]:
        """Activations for a single recycling step (supports negative indexing)."""
        try:
            return self.activations[step]
        except IndexError:
            return {}

    def get_site(self, name: str) -> list[torch.Tensor]:
        """List of tensors for one site across all recycling steps."""
        return [s[name] for s in self.activations if name in s]

    def step_keys(self) -> list[str]:
        """All site names that appear in at least one step."""
        out: set[str] = set()
        for s in self.activations:
            out.update(s.keys())
        return sorted(out)
