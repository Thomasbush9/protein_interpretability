"""General-purpose hidden representation extractor for ESMFold v1.

Captures activations from any layer/module of HuggingFace's
``facebook/esmfold_v1`` (``transformers.EsmForProteinFolding``) — ESM2 backbone
hidden states, folding-trunk sequence/pair states, triangle modules, structure
module, and prediction heads — using forward hooks.

Usage::

    from transformers import EsmForProteinFolding, AutoTokenizer
    from protein_interpretability.extractor_esmfold import ESMFoldExtractor

    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval()
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    batch = tok(["MSKGEE..."], return_tensors="pt", add_special_tokens=False)

    with ESMFoldExtractor(model, layer_sites=["layer_s", "layer_z"]) as ex:
        out = model(batch["input_ids"], num_recycles=3)
        final = ex.get_step(-1)  # dict of activations from last recycling iter

Submodule paths targeted (HF ``transformers.models.esm.modeling_esmfold``):

    model.esm.encoder.layer[i]                    # ESM2 transformer blocks
    model.esm.encoder.layer[i].attention          # ESM2 attention sublayer output
    model.esm.encoder.layer[i].attention.self     # ESM2 attention weights (monkey-patched)
    model.esm.encoder.layer[i].output             # ESM2 FFN sublayer output
    model.trunk.blocks[i]                         # folding trunk blocks (48)
    model.trunk.blocks[i].{seq_attention,tri_mul_out,tri_mul_in,
                           tri_att_start,tri_att_end,mlp_seq,mlp_pair}
    model.trunk.structure_module
    model.{distogram_head, lddt_head, ptm_head}
"""

from __future__ import annotations

import contextlib
from functools import wraps
from typing import Any

import torch
import torch.nn as nn


class ESMFoldExtractor:
    """Extract hidden representations from ESMFold v1 via forward hooks.

    Follows the same API as :class:`Boltz2Extractor` and
    :class:`ESM3ActivationExtractor`.
    """

    SITES = (
        "esm_s_combined",   # (B, L, D_esm) — ESM2 hidden states combined across layers
        "trunk_s",          # (B, L, C_s) — final trunk sequence state
        "trunk_z",          # (B, L, L, C_z) — final trunk pair state
        "structure_out",    # structure module output (dict / tuple)
        "distogram",        # (B, L, L, n_bins)
        "lddt_logits",      # (B, L, n_bins)
        "ptm_logits",       # (B, L, L, n_bins)
        "positions",        # (num_recycles+1, B, L, 14, 3) — returned by model
    )

    LAYER_SITES = (
        # Folding trunk (per-block)
        "layer_s",          # trunk block sequence output (B, L, C_s)
        "layer_z",          # trunk block pair output (B, L, L, C_z)
        "seq_attention",    # sequence self-attention output
        "tri_mul_out",      # triangular multiplicative outgoing
        "tri_mul_in",       # triangular multiplicative incoming
        "tri_att_start",    # triangular attention starting node
        "tri_att_end",      # triangular attention ending node
        "transition_s",     # mlp_seq output
        "transition_z",     # mlp_pair output
        # ESM2 backbone (per-layer) — separate space from trunk
        "esm_layer",            # ESM2 block hidden output (B, L, D_esm)
        "esm_attention_out",    # ESM2 attention sublayer output (B, L, D_esm)
        "esm_attention_weights",  # ESM2 post-softmax attention (B, H, L, L) — monkey-patched
        "esm_ffn_out",          # ESM2 FFN sublayer output (B, L, D_esm)
    )

    TRUNK_SUBMODULE_MAP = {
        "seq_attention":  "seq_attention",
        "tri_mul_out":    "tri_mul_out",
        "tri_mul_in":     "tri_mul_in",
        "tri_att_start":  "tri_att_start",
        "tri_att_end":    "tri_att_end",
        "transition_s":   "mlp_seq",
        "transition_z":   "mlp_pair",
    }

    def __init__(
        self,
        model: nn.Module,
        sites: list[str] | None = None,
        trunk_layers: list[int] | None = None,
        esm_layers: list[int] | None = None,
        layer_sites: list[str] | None = None,
        detach: bool = True,
        to_cpu: bool = True,
    ):
        """
        Args:
            model: ``EsmForProteinFolding`` instance.
            sites: Top-level sites to capture. ``None`` = all :attr:`SITES`.
            trunk_layers: Trunk block indices to hook. ``None`` = all 48.
            esm_layers: ESM2 encoder layer indices to hook (only used if
                ``"esm_layer"`` is in ``layer_sites``). ``None`` = all.
            layer_sites: Per-layer sites. ``None`` = ``["layer_s", "layer_z"]``.
            detach: Detach and clone stored tensors.
            to_cpu: Move stored tensors to CPU (saves GPU memory — trunk z is
                O(L^2 * C_z) per block and adds up fast).
        """
        self.model = model
        self.sites = sites if sites is not None else list(self.SITES)
        self.trunk_layers = trunk_layers
        self.esm_layers = esm_layers
        self.layer_sites = (
            layer_sites if layer_sites is not None else ["layer_s", "layer_z"]
        )
        self.detach = detach
        self.to_cpu = to_cpu

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._patched: list[tuple[nn.Module, str, Any]] = []  # (module, attr, original)
        self._installed = False
        self._trunk_block_calls = 0  # counts calls to trunk.blocks[0] for recycling
        self.activations: list[dict[str, torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store(self, key: str, value: Any) -> None:
        if not self.activations:
            self.activations.append({})
        if isinstance(value, torch.Tensor):
            v = value.detach().clone() if self.detach else value
            if self.to_cpu and v.is_cuda:
                v = v.cpu()
        else:
            v = value  # leave dicts/tuples (e.g. structure module output) as-is
        self.activations[-1][key] = v

    @staticmethod
    def _tensor_from_out(out: Any) -> Any:
        if isinstance(out, tuple):
            return out[0]
        return out

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Install hooks. Safe to call multiple times."""
        if self._installed:
            return

        trunk = self.model.trunk
        n_trunk = len(trunk.blocks)
        trunk_idx = (
            self.trunk_layers
            if self.trunk_layers is not None
            else list(range(n_trunk))
        )
        invalid = [i for i in trunk_idx if i < 0 or i >= n_trunk]
        if invalid:
            raise ValueError(
                f"Trunk layer indices {invalid} out of range (0..{n_trunk - 1})"
            )

        # --- Recycling boundary: a new recycling iteration starts every time
        # trunk.blocks[0] is called. Use its pre-hook to open a new step dict.
        def recycle_boundary_hook(module: nn.Module, inputs: Any) -> None:
            if self._trunk_block_calls > 0 or not self.activations:
                self.activations.append({})
            self._trunk_block_calls += 1

        self._handles.append(
            trunk.blocks[0].register_forward_pre_hook(recycle_boundary_hook)
        )

        # --- Top-level: combined ESM2 hidden state entering the trunk.
        # In HF EsmForProteinFolding, ``esm_s_mlp`` maps ESM2 combined hidden
        # states to the trunk sequence dim; hook its *input* to grab the
        # combined ESM2 representation.
        if "esm_s_combined" in self.sites and hasattr(self.model, "esm_s_mlp"):
            def esm_s_hook(module: nn.Module, inp: tuple, output: Any) -> None:
                self._store("esm_s_combined", self._tensor_from_out(inp[0]))
            self._handles.append(self.model.esm_s_mlp.register_forward_hook(esm_s_hook))

        # --- Top-level: trunk final s / z (post-hook on trunk)
        if "trunk_s" in self.sites or "trunk_z" in self.sites:
            def trunk_hook(module: nn.Module, inp: Any, output: Any) -> None:
                # EsmFoldingTrunk.forward returns a dict-like with s_s, s_z
                # (or a tuple). Handle both.
                if isinstance(output, dict):
                    s, z = output.get("s_s"), output.get("s_z")
                elif isinstance(output, tuple) and len(output) >= 2:
                    s, z = output[0], output[1]
                else:
                    return
                if "trunk_s" in self.sites and s is not None:
                    self._store("trunk_s", s)
                if "trunk_z" in self.sites and z is not None:
                    self._store("trunk_z", z)
            self._handles.append(trunk.register_forward_hook(trunk_hook))

        # --- Top-level: structure module
        if "structure_out" in self.sites and hasattr(trunk, "structure_module"):
            def sm_hook(module: nn.Module, inp: Any, output: Any) -> None:
                # Usually a dict with 'frames', 'positions', etc. Store lightly.
                if isinstance(output, dict):
                    stored = {
                        k: (v.detach().cpu() if isinstance(v, torch.Tensor) and self.to_cpu
                            else v.detach() if isinstance(v, torch.Tensor) else v)
                        for k, v in output.items()
                    }
                else:
                    stored = output
                self._store("structure_out", stored)
            self._handles.append(
                trunk.structure_module.register_forward_hook(sm_hook)
            )

        # --- Top-level: prediction heads
        head_map = {
            "distogram":   "distogram_head",
            "lddt_logits": "lddt_head",
            "ptm_logits":  "ptm_head",
        }
        for site, attr in head_map.items():
            if site in self.sites and hasattr(self.model, attr):
                def _make_head_hook(name: str):
                    def head_hook(module: nn.Module, inp: Any, output: Any) -> None:
                        self._store(name, self._tensor_from_out(output))
                    return head_hook
                self._handles.append(
                    getattr(self.model, attr).register_forward_hook(_make_head_hook(site))
                )

        # --- Per-trunk-block hooks
        for i in trunk_idx:
            block = trunk.blocks[i]
            prefix = f"trunk_{i}"

            if "layer_s" in self.layer_sites or "layer_z" in self.layer_sites:
                def block_hook(mod: nn.Module, inp: Any, output: Any,
                               p: str = prefix) -> None:
                    # EsmFoldTriangularSelfAttentionBlock returns (s, z)
                    if isinstance(output, tuple) and len(output) >= 2:
                        s, z = output[0], output[1]
                    elif isinstance(output, dict):
                        s, z = output.get("s"), output.get("z")
                    else:
                        return
                    if "layer_s" in self.layer_sites and s is not None:
                        self._store(f"{p}_layer_s", s)
                    if "layer_z" in self.layer_sites and z is not None:
                        self._store(f"{p}_layer_z", z)
                self._handles.append(block.register_forward_hook(block_hook))

            for site_name, attr_name in self.TRUNK_SUBMODULE_MAP.items():
                if site_name in self.layer_sites and hasattr(block, attr_name):
                    submod = getattr(block, attr_name)

                    def _make_sub_hook(p: str, sn: str):
                        def sub_hook(mod: nn.Module, inp: Any, output: Any) -> None:
                            self._store(f"{p}_{sn}", self._tensor_from_out(output))
                        return sub_hook

                    self._handles.append(
                        submod.register_forward_hook(_make_sub_hook(prefix, site_name))
                    )

        # --- Per-ESM2-layer hooks (backbone hidden states + sublayers)
        esm_sites_requested = {
            s for s in self.layer_sites
            if s in ("esm_layer", "esm_attention_out", "esm_attention_weights", "esm_ffn_out")
        }
        if esm_sites_requested and hasattr(self.model, "esm"):
            esm_layers = self.model.esm.encoder.layer
            n_esm = len(esm_layers)
            esm_idx = (
                self.esm_layers
                if self.esm_layers is not None
                else list(range(n_esm))
            )
            invalid = [i for i in esm_idx if i < 0 or i >= n_esm]
            if invalid:
                raise ValueError(
                    f"ESM2 layer indices {invalid} out of range (0..{n_esm - 1})"
                )
            for i in esm_idx:
                layer_mod = esm_layers[i]

                # Full layer output
                if "esm_layer" in esm_sites_requested:
                    def _make_esm_hook(idx: int):
                        def esm_hook(mod: nn.Module, inp: Any, output: Any) -> None:
                            self._store(f"esm_{idx}_layer", self._tensor_from_out(output))
                        return esm_hook
                    self._handles.append(
                        layer_mod.register_forward_hook(_make_esm_hook(i))
                    )

                # Attention sublayer output (after output projection + residual)
                if "esm_attention_out" in esm_sites_requested and hasattr(layer_mod, "attention"):
                    def _make_attn_out_hook(idx: int):
                        def attn_out_hook(mod: nn.Module, inp: Any, output: Any) -> None:
                            # EsmAttention returns (attention_output, [attention_probs])
                            self._store(f"esm_{idx}_attention_out", self._tensor_from_out(output))
                        return attn_out_hook
                    self._handles.append(
                        layer_mod.attention.register_forward_hook(_make_attn_out_hook(i))
                    )

                # Post-softmax attention weights (monkey-patched)
                if "esm_attention_weights" in esm_sites_requested and hasattr(layer_mod, "attention"):
                    self._patch_esm_attention_weights(
                        layer_mod.attention.self, f"esm_{i}_attention_weights"
                    )

                # FFN sublayer output (EsmOutput: projection + residual)
                if "esm_ffn_out" in esm_sites_requested and hasattr(layer_mod, "output"):
                    def _make_ffn_hook(idx: int):
                        def ffn_hook(mod: nn.Module, inp: Any, output: Any) -> None:
                            self._store(f"esm_{idx}_ffn_out", self._tensor_from_out(output))
                        return ffn_hook
                    self._handles.append(
                        layer_mod.output.register_forward_hook(_make_ffn_hook(i))
                    )

        self._installed = True

    # ------------------------------------------------------------------
    # Monkey-patching
    # ------------------------------------------------------------------

    def _patch_esm_attention_weights(self, attn_self_module: nn.Module, key: str) -> None:
        """Monkey-patch ``EsmSelfAttention.forward`` to capture post-softmax weights.

        Wraps the original forward so that ``output_attentions`` is forced True
        internally.  The attention probabilities (post-softmax, pre-dropout in
        eval mode) are captured, and the return value is adjusted back to the
        original format so downstream code is unaffected.
        """
        original_forward = attn_self_module.forward
        extractor = self

        @wraps(original_forward)
        def patched_forward(
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
        ):
            outputs = original_forward(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=True,  # force on
            )
            # outputs = (context_layer, attention_probs, [past_kv])
            extractor._store(key, outputs[1])

            if output_attentions:
                return outputs  # caller already expects attention_probs
            # Strip attention_probs so the return shape matches the original
            return (outputs[0],) + outputs[2:]

        attn_self_module.forward = patched_forward
        self._patched.append((attn_self_module, "forward", original_forward))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def remove(self) -> None:
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
        self._trunk_block_calls = 0

    @contextlib.contextmanager
    def recording(self):
        self.install()
        try:
            yield self
        finally:
            self.remove()

    def __enter__(self) -> ESMFoldExtractor:
        self.install()
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_step(self, step: int) -> dict[str, torch.Tensor]:
        """Activations for one recycling iteration (supports negative indexing)."""
        try:
            return self.activations[step]
        except IndexError:
            return {}

    def get_site(self, name: str) -> list[torch.Tensor]:
        return [s[name] for s in self.activations if name in s]

    def step_keys(self) -> list[str]:
        out: set[str] = set()
        for s in self.activations:
            out.update(s.keys())
        return sorted(out)
