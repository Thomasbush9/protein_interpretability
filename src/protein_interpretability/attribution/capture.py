"""Forward-hook plumbing for attribution surfaces on Boltz2.

Two capture modes:

1. ``install_pair_rep_attribution()`` (recommended) — pre-hook on
   ``distogram_module`` replaces the incoming pair representation ``z`` with a
   leaf ``z.detach().requires_grad_(True)``. With model parameters frozen, the
   trunk has no autograd graph, but ``distogram_module(z_leaf)`` does — a
   single Linear's worth. Backward from a scalar target loss populates
   ``z_leaf.grad`` directly. Memory peaks at one trunk forward + distogram's
   tiny graph regardless of K. This is the standard Captum
   LayerIntegratedGradients pattern.

2. ``install()`` (legacy) — post-hooks on ``input_embedder``, ``msa_module``,
   ``distogram_module`` capture full-trunk activations with ``retain_grad()``.
   Requires the trunk's autograd graph to be alive, which means at K=10 the
   activation footprint is ~80GB on a 238-residue protein with eager triangle
   attention. Kept for the mock-model unit tests; not used by the runner.

Both modes capture the distogram output for the loss target.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn as nn


def _unwrap(module: nn.Module) -> nn.Module:
    """Strip ``torch.compile``'s ``_orig_mod`` wrapper if present."""
    return getattr(module, "_orig_mod", module)


def _tensor_from_out(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"cannot extract tensor from output of type {type(out).__name__}")


class GradientCapture:
    """Hooks ``input_embedder``, ``msa_module``, ``distogram_module`` for grads.

    Usage::

        cap = GradientCapture(model)
        with cap:
            cap.clear()
            _ = model(batch, recycling_steps=K, ...)
            loss = target(cap.distogram, batch.get("token_pad_mask"))
            loss.backward()
            query_grad = cap.query_emb.grad   # (B, N, D_s)
            msa_grad   = cap.msa_emb.grad     # (B, S, N, D_m) typ.

    The context manager installs hooks on enter and removes them on exit.
    ``clear()`` resets captured tensors between records.
    """

    QUERY_KEY = "query_emb"
    MSA_KEY = "msa_emb"
    DISTOGRAM_KEY = "distogram"
    PAIR_KEY = "pair_input"

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._installed = False
        self._captured: dict[str, torch.Tensor] = {}

    # -- properties: read-only views into the captured tensors ------------

    @property
    def query_emb(self) -> torch.Tensor:
        return self._require(self.QUERY_KEY)

    @property
    def msa_emb(self) -> torch.Tensor:
        return self._require(self.MSA_KEY)

    @property
    def distogram(self) -> torch.Tensor:
        return self._require(self.DISTOGRAM_KEY)

    @property
    def pair_input(self) -> torch.Tensor:
        return self._require(self.PAIR_KEY)

    def _require(self, key: str) -> torch.Tensor:
        try:
            return self._captured[key]
        except KeyError as e:
            raise RuntimeError(
                f"surface {key!r} not captured — did the forward pass run "
                "inside the GradientCapture context, and does the model expose "
                f"'{key.split('_')[0]}_module' / 'input_embedder' attributes?"
            ) from e

    def get(self, key: str) -> torch.Tensor | None:
        return self._captured.get(key)

    def captured_keys(self) -> list[str]:
        return list(self._captured.keys())

    # -- lifecycle --------------------------------------------------------

    def clear(self) -> None:
        self._captured.clear()

    def install(self) -> None:
        """Legacy mode: capture query_emb / msa_emb / distogram outputs.

        Used by the mock-model unit tests. Not appropriate for real Boltz
        because it requires the full trunk's autograd graph alive (OOM).
        """
        if self._installed:
            return

        embedder = getattr(self.model, "input_embedder", None)
        if embedder is None:
            raise AttributeError("model has no .input_embedder — required for query_emb capture")
        msa_module = getattr(self.model, "msa_module", None)
        distogram_module = getattr(self.model, "distogram_module", None)
        if distogram_module is None:
            raise AttributeError(
                "model has no .distogram_module — required for target loss"
            )

        self._handles.append(
            _unwrap(embedder).register_forward_hook(self._make_capture_hook(self.QUERY_KEY))
        )
        if msa_module is not None:
            self._handles.append(
                _unwrap(msa_module).register_forward_hook(
                    self._make_capture_hook(self.MSA_KEY, overwrite=True)
                )
            )
        self._handles.append(
            _unwrap(distogram_module).register_forward_hook(
                self._make_capture_hook(self.DISTOGRAM_KEY, overwrite=True)
            )
        )
        self._installed = True

    def install_pair_rep_attribution(self) -> None:
        """Pair-rep mode (memory-safe): replace distogram_module's input with a leaf.

        Pre-hook on ``distogram_module`` swaps the incoming pair representation
        ``z`` for ``z.detach().requires_grad_(True)``. The downstream
        distogram_module forward sees the leaf. With params frozen, this is the
        only tensor on the autograd graph — backward from the loss populates
        ``cap.pair_input.grad`` and nothing else.

        Also installs a post-hook that captures the distogram output, so the
        runner can compute its target loss from ``cap.distogram``.
        """
        if self._installed:
            return

        distogram_module = getattr(self.model, "distogram_module", None)
        if distogram_module is None:
            raise AttributeError(
                "model has no .distogram_module — required for pair-rep capture"
            )

        captured = self._captured
        pair_key = self.PAIR_KEY

        def pre_hook(_module: nn.Module, args: tuple) -> tuple | None:
            if not args or not isinstance(args[0], torch.Tensor):
                return None
            z = args[0]
            z_leaf = z.detach().requires_grad_(True)
            z_leaf.retain_grad()
            captured[pair_key] = z_leaf
            return (z_leaf,) + tuple(args[1:])

        # Hook on the wrapper (not _unwrap'd) so the replacement reaches the
        # compiled forward path in compile-mode models. Mirrors the pattern in
        # _train_mode_for_boltz_grads (runner.py) and Boltz2Extractor.
        self._handles.append(distogram_module.register_forward_pre_hook(pre_hook))
        self._handles.append(
            distogram_module.register_forward_hook(
                self._make_capture_hook(self.DISTOGRAM_KEY, overwrite=True)
            )
        )
        self._installed = True

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._installed = False

    def __enter__(self) -> "GradientCapture":
        self.install()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove()

    @contextlib.contextmanager
    def recording(self):
        self.install()
        try:
            yield self
        finally:
            self.remove()

    # -- hook factory -----------------------------------------------------

    def _make_capture_hook(self, key: str, *, overwrite: bool = False):
        captured = self._captured

        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = _tensor_from_out(output)
            # Mirror the Boltz2Extractor pattern: store unconditionally. The
            # runner is responsible for checking requires_grad and producing a
            # clear diagnostic if grad flow is broken.
            if tensor.requires_grad:
                tensor.retain_grad()
            if overwrite or key not in captured:
                captured[key] = tensor

        return hook
