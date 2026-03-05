"""Interactive Plotly visualization of Boltz pairformer attention on a protein sequence.

Metrics computed:
  - Column sum (incoming attention): how much each residue is attended to
  - Attention rollout (Abnar & Zuidema 2020): propagates attention through
    layers accounting for residual connections — reflects actual information flow
  - Max-head column sum: keeps the strongest head signal per position
  - Column entropy: low entropy = residue receives focused attention from
    few specific positions (structurally specific interactions)
"""

import argparse
import numpy as np
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_sequence(yaml_path: str) -> str:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data["sequences"][0]["protein"]["sequence"]


def load_attention(npz_path: str) -> dict[str, np.ndarray]:
    return dict(np.load(npz_path))


def _get_layers(attention_data: dict) -> list[str]:
    return sorted([k for k in attention_data if k.startswith("pairformer_layer")])


def _head_avg_matrix(att_4d: np.ndarray) -> np.ndarray:
    """(1, H, L, L) -> (L, L) averaged over heads."""
    return att_4d.squeeze(0).mean(axis=0)


# ── Metrics ─────────────────────────────────────────────────────────────────

def column_sum(att_4d: np.ndarray, head: str = "mean") -> np.ndarray:
    """Sum of incoming attention per residue (column sum of the attention matrix)."""
    att = att_4d.squeeze(0)
    if head == "mean":
        att = att.mean(axis=0)
    else:
        att = att[int(head)]
    return att.sum(axis=0)


def max_head_column_sum(att_4d: np.ndarray) -> np.ndarray:
    """For each residue, take the max column sum across heads."""
    att = att_4d.squeeze(0)  # (H, L, L)
    per_head = att.sum(axis=1)  # (H, L) — column sum per head
    return per_head.max(axis=0)  # (L,)


def column_entropy(att_4d: np.ndarray, head: str = "mean") -> np.ndarray:
    """Entropy of incoming attention distribution per residue (column).

    Low entropy = few positions strongly attend to this residue (specific).
    High entropy = many positions attend roughly equally (diffuse).
    """
    att = att_4d.squeeze(0)
    if head == "mean":
        att = att.mean(axis=0)
    else:
        att = att[int(head)]
    # Normalize columns to probability distributions
    col_sums = att.sum(axis=0, keepdims=True)
    col_probs = att / (col_sums + 1e-12)
    ent = -np.sum(col_probs * np.log(col_probs + 1e-12), axis=0)
    return ent


def attention_rollout(attention_data: dict) -> np.ndarray:
    """Attention rollout across all pairformer layers.

    At each layer: A_eff = 0.5 * I + 0.5 * A  (residual connection)
    Rollout: R = A_eff_0 @ A_eff_1 @ ... @ A_eff_N
    Returns per-residue relevance as column sum of R.
    """
    layers = _get_layers(attention_data)
    R = None
    for lyr in layers:
        A = _head_avg_matrix(attention_data[lyr])  # (L, L)
        L = A.shape[0]
        A_eff = 0.5 * np.eye(L) + 0.5 * A
        # Re-normalize rows to sum to 1
        A_eff = A_eff / A_eff.sum(axis=1, keepdims=True)
        R = A_eff if R is None else R @ A_eff
    return R


def rollout_column_sum(attention_data: dict) -> np.ndarray:
    R = attention_rollout(attention_data)
    return R.sum(axis=0)


# ── Color helpers ───────────────────────────────────────────────────────────

def _viridis_rgb(t: float) -> str:
    from matplotlib import cm
    r, g, b, _ = cm.viridis(t)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


# ── Figure 1: Multi-metric comparison bar chart + heatmap ───────────────────

def make_main_figure(sequence: str, attention_data: dict, layer: str, head: str = "mean") -> go.Figure:
    L = len(sequence)
    residues = list(sequence)
    positions = list(range(1, L + 1))

    # Compute all metrics
    metrics = {
        "Column sum (incoming attn)": column_sum(attention_data[layer], head),
        "Attention rollout": rollout_column_sum(attention_data),
        "Max-head column sum": max_head_column_sum(attention_data[layer]),
        "Column entropy (inverted)": None,  # computed below
    }
    # Invert entropy so high = specific/interesting
    ent = column_entropy(attention_data[layer], head)
    metrics["Column entropy (inverted)"] = ent.max() - ent

    n_metrics = len(metrics)
    att_matrix = _head_avg_matrix(attention_data[layer])
    rollout_matrix = attention_rollout(attention_data)

    fig = make_subplots(
        rows=n_metrics + 2, cols=1,
        row_heights=[0.12] * n_metrics + [0.26, 0.26],
        vertical_spacing=0.025,
        subplot_titles=[
            *list(metrics.keys()),
            f"Attention matrix — {layer}",
            "Attention rollout matrix",
        ],
    )

    # ── Bar charts for each metric ──
    for row_idx, (name, values) in enumerate(metrics.items(), start=1):
        hover = [f"<b>{positions[i]} {residues[i]}</b><br>{name}: {values[i]:.4f}"
                 for i in range(L)]
        fig.add_trace(
            go.Bar(
                x=positions, y=values,
                marker_color=values,
                marker_colorscale="Viridis",
                marker_showscale=False,
                hovertext=hover, hoverinfo="text",
                name=name,
            ),
            row=row_idx, col=1,
        )
        fig.update_xaxes(
            tickvals=positions,
            ticktext=residues,
            tickfont=dict(size=8, family="Courier New"),
            tickangle=0,
            row=row_idx, col=1,
        )
        fig.update_yaxes(title_text="", row=row_idx, col=1)

    # ── Heatmap: single layer ──
    heat_row = n_metrics + 1
    heat_hover = [
        [f"Q: {i+1} {residues[i]} → K: {j+1} {residues[j]}<br>Attn: {att_matrix[i,j]:.4f}"
         for j in range(L)] for i in range(L)
    ]
    fig.add_trace(
        go.Heatmap(
            z=att_matrix, x=positions, y=positions,
            colorscale="Viridis", hovertext=heat_hover, hoverinfo="text",
            colorbar=dict(title="Attn", x=1.01, y=0.30, len=0.20),
        ),
        row=heat_row, col=1,
    )

    # ── Heatmap: rollout ──
    roll_row = n_metrics + 2
    roll_hover = [
        [f"Q: {i+1} {residues[i]} → K: {j+1} {residues[j]}<br>Rollout: {rollout_matrix[i,j]:.4f}"
         for j in range(L)] for i in range(L)
    ]
    fig.add_trace(
        go.Heatmap(
            z=rollout_matrix, x=positions, y=positions,
            colorscale="Viridis", hovertext=roll_hover, hoverinfo="text",
            colorbar=dict(title="Rollout", x=1.01, y=0.07, len=0.20),
        ),
        row=roll_row, col=1,
    )

    # Tick labels for heatmaps
    every5 = list(range(0, L, 5))
    t5_vals = [positions[i] for i in every5]
    t5_text = [f"{positions[i]}{residues[i]}" for i in every5]

    for hr in [heat_row, roll_row]:
        fig.update_xaxes(tickvals=t5_vals, ticktext=t5_text, tickangle=90,
                         title_text="Key position", constrain="domain", row=hr, col=1)
        fig.update_yaxes(tickvals=t5_vals, ticktext=t5_text,
                         title_text="Query position", autorange="reversed",
                         scaleanchor=f"x{hr}", scaleratio=1, constrain="domain",
                         row=hr, col=1)

    head_label = f"head {head}" if head != "mean" else "mean of heads"
    width = max(1200, L * 18)
    fig.update_layout(
        title=f"Boltz Pairformer Attention Analysis — {layer} ({head_label})",
        height=350 * n_metrics + 900,
        width=width,
        template="plotly_white",
        showlegend=False,
        bargap=0.08,
    )

    return fig


# ── Figure 2: Sequence colored grid ────────────────────────────────────────

def make_sequence_figure(sequence: str, attention_data: dict, layer: str, head: str = "mean",
                         chars_per_row: int = 50) -> go.Figure:
    L = len(sequence)
    residues = list(sequence)

    # Two metrics side by side: column sum and rollout
    col_sum = column_sum(attention_data[layer], head)
    rollout_rel = rollout_column_sum(attention_data)

    metrics = {
        f"Column sum — {layer}": col_sum,
        "Attention rollout (all layers)": rollout_rel,
    }

    n_rows_per_block = (L + chars_per_row - 1) // chars_per_row
    fig = make_subplots(
        rows=len(metrics), cols=1,
        row_heights=[1.0 / len(metrics)] * len(metrics),
        vertical_spacing=0.08,
        subplot_titles=list(metrics.keys()),
    )

    for metric_idx, (name, relevance) in enumerate(metrics.items()):
        rmin, rmax = relevance.min(), relevance.max()
        norm = (relevance - rmin) / (rmax - rmin + 1e-12)
        row_num = metric_idx + 1

        # Colored rectangles
        for i, aa in enumerate(sequence):
            r, c = divmod(i, chars_per_row)
            color = _viridis_rgb(norm[i])

            fig.add_shape(
                type="rect",
                x0=c - 0.45, x1=c + 0.45,
                y0=r - 0.45, y1=r + 0.45,
                fillcolor=color,
                line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                layer="below",
                xref=f"x{row_num}" if row_num > 1 else "x",
                yref=f"y{row_num}" if row_num > 1 else "y",
            )

            fig.add_trace(go.Scatter(
                x=[c], y=[r],
                mode="text",
                text=[f"<b>{aa}</b>"],
                textfont=dict(size=13, family="Courier New", color="white"),
                hovertext=f"<b>{aa}</b> — pos {i+1}<br>{name}: {relevance[i]:.4f}",
                hoverinfo="text",
                showlegend=False,
            ), row=row_num, col=1)

        # Row labels
        for r in range(n_rows_per_block):
            start = r * chars_per_row + 1
            end = min((r + 1) * chars_per_row, L)
            fig.add_annotation(
                x=-1.5, y=r, text=f"{start}-{end}",
                showarrow=False,
                font=dict(size=10, family="Courier New", color="#666"),
                xanchor="right",
                xref=f"x{row_num}" if row_num > 1 else "x",
                yref=f"y{row_num}" if row_num > 1 else "y",
            )

        # Colorbar
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(
                colorscale="Viridis",
                cmin=rmin, cmax=rmax,
                colorbar=dict(title=name.split("—")[0].strip(), thickness=12,
                              len=0.35, y=1.0 - (metric_idx + 0.5) / len(metrics)),
                showscale=True,
            ),
            showlegend=False, hoverinfo="none",
        ), row=row_num, col=1)

        fig.update_xaxes(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-3, chars_per_row + 0.5],
            row=row_num, col=1,
        )
        fig.update_yaxes(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[n_rows_per_block - 0.5, -0.8],
            scaleanchor=f"x{row_num}" if row_num > 1 else "x",
            scaleratio=1,
            row=row_num, col=1,
        )

    head_label = f"head {head}" if head != "mean" else "mean of heads"
    fig.update_layout(
        title=f"Sequence Attention Map ({head_label})",
        plot_bgcolor="rgb(30,30,30)",
        paper_bgcolor="white",
        width=max(900, chars_per_row * 22),
        height=max(400, n_rows_per_block * 55 * len(metrics) + 150),
        margin=dict(l=60, r=100, t=80, b=30),
    )

    return fig


# ── Figure 3: Conservation vs Attention Rollout ────────────────────────────

def make_conservation_attention_figure(
    sequence: str, attention_data: dict, conservation: np.ndarray,
    head: str = "mean", layer: str | None = None,
) -> go.Figure:
    """Conservation vs attention metric.

    If layer is None, uses attention rollout across all layers.
    If layer is given, uses column sum of that single layer.
    """
    L = len(sequence)
    residues = list(sequence)
    positions = list(range(1, L + 1))

    if layer is None:
        attn_score = rollout_column_sum(attention_data)
        metric_name = "Attention rollout"
    else:
        attn_score = column_sum(attention_data[layer], head)
        metric_name = f"Column sum — {layer}"

    attn_norm = (attn_score - attn_score.min()) / (attn_score.max() - attn_score.min() + 1e-12)
    cons_norm = conservation

    hover = [
        f"<b>Pos {p}: {aa}</b><br>"
        f"Conservation: {conservation[i]:.3f}<br>"
        f"{metric_name}: {attn_score[i]:.4f}"
        for i, (p, aa) in enumerate(zip(positions, residues))
    ]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.12,
        subplot_titles=[
            f"Conservation vs {metric_name} per residue",
            f"Conservation vs {metric_name} scatter",
        ],
    )

    # ── Top: overlaid bar chart ──
    fig.add_trace(
        go.Bar(
            x=positions, y=cons_norm, name="Conservation",
            marker_color="rgba(55, 128, 191, 0.6)",
            hovertext=hover, hoverinfo="text",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=positions, y=attn_norm, name=f"{metric_name} (norm)",
            mode="lines+markers",
            line=dict(color="rgba(219, 64, 82, 0.9)", width=1.5),
            marker=dict(size=3),
            hovertext=hover, hoverinfo="text",
        ),
        row=1, col=1,
    )

    fig.update_xaxes(
        tickvals=positions, ticktext=residues,
        tickfont=dict(size=8, family="Courier New"),
        tickangle=0, row=1, col=1,
    )
    fig.update_yaxes(title_text="Normalized score", row=1, col=1)

    # ── Bottom: scatter plot ──
    fig.add_trace(
        go.Scatter(
            x=conservation, y=attn_score,
            mode="markers+text",
            text=residues,
            textposition="top center",
            textfont=dict(size=7, family="Courier New"),
            marker=dict(
                size=7, color=positions, colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Position", x=1.01, len=0.4, y=0.18),
            ),
            hovertext=hover, hoverinfo="text",
            name="Residues",
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.update_xaxes(title_text="Conservation score", row=2, col=1)
    fig.update_yaxes(title_text=metric_name, row=2, col=1)

    # Correlation
    corr = np.corrcoef(conservation, attn_score)[0, 1]

    width = max(1200, L * 18)
    fig.update_layout(
        title=f"Conservation vs {metric_name} (Pearson r = {corr:.3f})",
        height=900,
        width=width,
        template="plotly_white",
        legend=dict(x=0.85, y=1.0),
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Interactive Boltz attention visualization")
    parser.add_argument("--attention", required=True, help="Path to .npz attention file")
    parser.add_argument("--sequence", required=True, help="Path to YAML sequence file")
    parser.add_argument("--layer", default="pairformer_layer_47")
    parser.add_argument("--head", default="mean", help="Head index or 'mean'")
    parser.add_argument("--msa", default=None, help="Path to .a3m MSA file for conservation")
    parser.add_argument("--output", default=None, help="Output HTML file prefix")
    args = parser.parse_args()

    sequence = load_sequence(args.sequence)
    attention_data = load_attention(args.attention)

    fig_main = make_main_figure(sequence, attention_data, args.layer, args.head)
    fig_seq = make_sequence_figure(sequence, attention_data, args.layer, args.head)

    figs = {"main": fig_main, "sequence": fig_seq}

    if args.msa:
        from protein_interpretability.utils import conservation_score
        conservation = conservation_score(args.msa)
        fig_rollout = make_conservation_attention_figure(
            sequence, attention_data, conservation, args.head, layer=None)
        fig_layer = make_conservation_attention_figure(
            sequence, attention_data, conservation, args.head, layer=args.layer)
        figs["conservation_rollout"] = fig_rollout
        figs["conservation_layer"] = fig_layer

    if args.output:
        prefix = args.output.removesuffix(".html")
        for name, fig in figs.items():
            path = f"{prefix}_{name}.html"
            fig.write_html(path)
            print(f"Saved: {path}")
    else:
        for fig in figs.values():
            fig.show()


if __name__ == "__main__":
    main()
