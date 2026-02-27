from __future__ import annotations

import math
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import torch
from protein_interpretability.utils import sort_layer_activations


_LAYER_KEY_RE = re.compile(r"^layer_(\d+)_(block|attn|geom_attn|ffn)$")


def _reduce_batch_if_needed(t: torch.Tensor) -> torch.Tensor:
    """If tensor has batch dim (ndim > 2), return mean over dim=0; else return as-is."""
    if t.dim() > 2:
        return t.mean(dim=0)
    return t


def plot_all_layers_sorted_paged(
    activations: dict[str, torch.Tensor],
    *,
    layer_keys: list[str] | None = None,
    layers: list[int] | None = None,
    layer_sites: list[str] | None = None,
    max_layers_per_plot: int = 30,
    figsize_per_subplot: tuple[float, float] = (6.0, 6.0),
    lognorm: bool = True,
    title: str | None = None,
):
    """
    Creates multiple heatmaps (each up to `max_layers_per_plot` layers) arranged as subplots.

    Selection logic:
      - If layer_keys is provided: use exactly those keys.
      - Else auto-detect keys matching: layer_{i}_{block|attn|geom_attn|ffn}
        and optionally filter by `layers` and/or `layer_sites`.
      - If nothing specified: plot ALL matching layer keys in activations.

    Returns:
      fig, axes (2D array of Axes)
    """

    # ---------- pick keys ----------
    if layer_keys is None:
        layers_set = set(layers) if layers is not None else None
        sites_set = set(layer_sites) if layer_sites is not None else None

        picked: list[tuple[int, str, str]] = []  # (layer_idx, site, key)
        for k in activations.keys():
            m = _LAYER_KEY_RE.match(k)
            if not m:
                continue
            li = int(m.group(1))
            site = m.group(2)
            if layers_set is not None and li not in layers_set:
                continue
            if sites_set is not None and site not in sites_set:
                continue
            picked.append((li, site, k))

        # sort by layer index, then site name
        picked.sort(key=lambda t: (t[0], t[1]))
        layer_keys = [k for _, _, k in picked]

    if not layer_keys:
        raise ValueError(
            "No matching activation keys found. "
            "Expected keys like 'layer_{i}_{block|attn|geom_attn|ffn}'."
        )

    # ---------- compute sorted activations per layer ----------
    acts_sorted = []
    for k in layer_keys:
        if k not in activations:
            raise KeyError(f"Key '{k}' not present in activations.")
        act_sorted, _ = sort_layer_activations(_reduce_batch_if_needed(activations[k]))
        acts_sorted.append(act_sorted)

    # We keep the global matrix for global vmin/vmax
    full_matrix = np.asarray(acts_sorted)  # shape: (n_layers, n_nodes_sorted)
    full_vmin = float(max(full_matrix.min(), 1e-12))
    full_vmax = float(full_matrix.max())

    # ---------- chunk into subplots ----------
    n_layers = len(layer_keys)
    n_plots = math.ceil(n_layers / max_layers_per_plot)

    # grid: near-square
    ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols)

    subplot_w, subplot_h = figsize_per_subplot
    fig_w = subplot_w * ncols
    fig_h = subplot_h * nrows

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

    norm = mcolors.LogNorm(vmin=full_vmin, vmax=full_vmax) if lognorm else None
    cmap = "viridis"  # same for heatmaps and colorbar

    # put a single shared colorbar on the right
    for p in range(n_plots):
        r = p // ncols
        c = p % ncols
        ax = axes[r][c]

        start = p * max_layers_per_plot
        end = min((p + 1) * max_layers_per_plot, n_layers)

        # chunk, reverse order so "higher" layers appear on top within that chunk
        chunk_matrix = full_matrix[start:end][::-1]
        chunk_labels = layer_keys[start:end][::-1]

        sns.heatmap(
            chunk_matrix,
            ax=ax,
            norm=norm,
            cmap=cmap,
            cbar=False,
            xticklabels=False,
            yticklabels=chunk_labels,
        )

        ax.set_xlabel("Node rank (most → least active)")
        ax.set_ylabel("Layer / site")
        ax.set_title(f"Layers {start}–{end-1}")

    # hide unused axes
    for p in range(n_plots, nrows * ncols):
        r = p // ncols
        c = p % ncols
        axes[r][c].axis("off")

    # shared colorbar: same norm and cmap as heatmaps, in reserved space (no overlap)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.tight_layout(rect=[0, 0, 0.88, 1])  # leave right 12% for colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label("mean |act| (sorted)")

    if title is not None:
        fig.suptitle(title, y=1.02, fontsize=14)
    return fig, axes