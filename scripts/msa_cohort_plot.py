#!/usr/bin/env python3
"""Cohort-level Tier 1 MSA summary figure.

Loads WT MSA + multiple labelled mutant MSAs and produces:

  - cohort_summary.png   : 2x2 panel (depth+Neff, conservation r, mean Δentropy,
                           close-homolog retention by identity bucket)
  - entropy_profiles.png : per-position entropy overlay + per-position Δentropy
                           per cohort, with mutated-position markers
  - cohort_metrics.csv   : all scalar metrics in one row per cohort

Usage:
    python scripts/msa_cohort_plot.py \\
        --wt /path/to/wt.a3m \\
        --cohort p20=/path/to/p20.a3m \\
        --cohort p40=/path/to/p40.a3m \\
        --cohort p70=/path/to/p70.a3m \\
        --out reports/msa_tier1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from msa_tier1 import (
    GAP_IDX,
    MSA,
    NUM_TOKENS,
    compute_neff,
    parse_a3m,
    per_position_entropy,
    query_identity,
    shared_id_jaccard,
)


def per_position_aa_distribution(matrix: np.ndarray, include_gap: bool = True) -> np.ndarray:
    """Per-column amino-acid frequency distribution. Shape (L, NUM_TOKENS).

    Includes gap as a 21st token by default — this is what the MSA module
    actually consumes (the outer-product mean is over the full distribution
    including gap-as-token), so it's the more faithful basis for divergence.
    """
    N, L = matrix.shape
    out = np.zeros((L, NUM_TOKENS), dtype=np.float64)
    for col in range(L):
        counts = np.bincount(matrix[:, col], minlength=NUM_TOKENS).astype(np.float64)
        if not include_gap:
            counts[GAP_IDX] = 0
        s = counts.sum()
        if s > 0:
            out[col] = counts / s
    return out


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """JS divergence in nats, per row. p, q shape (L, K). Returns (L,).

    JSD = 1/2 KL(p || m) + 1/2 KL(q || m), m = (p + q) / 2.
    Bounded in [0, ln 2]. Symmetric. Zero iff p == q.
    """
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(a > 0, a * (np.log(a + eps) - np.log(b + eps)), 0.0)
        return term.sum(axis=-1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

ID_BUCKETS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
COHORT_COLORS = {
    "WT":  "#2c7fb8",
    "p20": "#41b6c4",
    "p40": "#fdae61",
    "p70": "#d7191c",
}


def cohort_color(label: str, fallback_idx: int) -> str:
    if label in COHORT_COLORS:
        return COHORT_COLORS[label]
    palette = ["#7570b3", "#1b9e77", "#d95f02", "#e7298a", "#66a61e"]
    return palette[fallback_idx % len(palette)]


def retention_by_bucket(wt: MSA, mut: MSA) -> list[tuple[float, float, int, int]]:
    if wt.length != mut.length:
        return [(lo, hi, 0, 0) for lo, hi in ID_BUCKETS]
    wt_qid = (wt.matrix[1:] == wt.matrix[0][None, :]).sum(axis=1) / wt.length
    mut_ids = set(mut.ids[1:])
    out = []
    for lo, hi in ID_BUCKETS:
        mask = (wt_qid >= lo) & (wt_qid < hi)
        bucket_ids = [wt.ids[1 + i] for i in np.where(mask)[0]]
        kept = sum(1 for x in bucket_ids if x in mut_ids)
        out.append((lo, hi, kept, len(bucket_ids)))
    return out


def mutated_positions(wt: MSA, mut: MSA) -> np.ndarray:
    if wt.length != mut.length:
        return np.array([], dtype=int)
    qa = wt.matrix[0]
    qb = mut.matrix[0]
    both_ng = (qa != GAP_IDX) & (qb != GAP_IDX)
    return np.where(both_ng & (qa != qb))[0]


def compute_all(wt: MSA, mutants: dict[str, MSA], id_threshold: float) -> dict:
    metrics = {}
    metrics["WT"] = {
        "depth": wt.depth,
        "neff": compute_neff(wt.matrix, id_threshold),
        "entropy": per_position_entropy(wt.matrix, include_gap=False),
        "aa_dist": per_position_aa_distribution(wt.matrix, include_gap=True),
    }
    for label, mut in mutants.items():
        e_mut = per_position_entropy(mut.matrix, include_gap=False)
        e_wt = metrics["WT"]["entropy"]
        if e_wt.shape == e_mut.shape:
            r = float(np.corrcoef(e_wt, e_mut)[0, 1])
            d_entropy = e_mut - e_wt
        else:
            r = float("nan")
            d_entropy = np.array([])
        p_mut = per_position_aa_distribution(mut.matrix, include_gap=True)
        p_wt = metrics["WT"]["aa_dist"]
        if p_wt.shape == p_mut.shape:
            jsd = js_divergence(p_wt, p_mut)
        else:
            jsd = np.array([])
        j, inter, na, nb = shared_id_jaccard(wt, mut)
        metrics[label] = {
            "depth": mut.depth,
            "neff": compute_neff(mut.matrix, id_threshold),
            "query_identity": query_identity(wt, mut),
            "shared_jaccard": j,
            "inter": inter,
            "frac_mut_in_wt": (inter / nb) if nb else 0.0,
            "frac_wt_in_mut": (inter / na) if na else 0.0,
            "entropy": e_mut,
            "delta_entropy": d_entropy,
            "entropy_pearson_r": r,
            "mean_delta_entropy": float(d_entropy.mean()) if d_entropy.size else float("nan"),
            "aa_dist": p_mut,
            "jsd_per_position": jsd,
            "mean_jsd": float(jsd.mean()) if jsd.size else float("nan"),
            "retention": retention_by_bucket(wt, mut),
            "mutated_positions": mutated_positions(wt, mut),
        }
    return metrics


def plot_summary(metrics: dict, out_path: Path, id_threshold: float) -> None:
    cohorts = [k for k in metrics if k != "WT"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Tier 1 MSA characterisation — cohort summary "
        f"(GFP query, Neff threshold id≥{id_threshold})",
        fontsize=12,
    )

    # --- Panel A: depth + Neff (paired bars, linear y) ---
    ax = axes[0, 0]
    labels = ["WT"] + cohorts
    depths = [metrics[c]["depth"] for c in labels]
    neffs = [metrics[c]["neff"] for c in labels]
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w / 2, depths, w, label="depth (raw)", color="#888")
    bars2 = ax.bar(x + w / 2, neffs, w, label=f"Neff (id≥{id_threshold})", color="#2c7fb8")
    ax.set_ylabel("# sequences")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("MSA depth and effective depth (Neff)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ymax = max(max(depths), max(neffs))
    ax.set_ylim(0, ymax * 1.10)
    for b, v in list(zip(bars1, depths)) + list(zip(bars2, neffs)):
        ax.annotate(f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                    xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    # --- Panel B: Conservation profile correlation ---
    ax = axes[0, 1]
    rs = [metrics[c]["entropy_pearson_r"] for c in cohorts]
    colors = [cohort_color(c, i) for i, c in enumerate(cohorts)]
    bars = ax.bar(cohorts, rs, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(min(-0.1, min(rs) - 0.1), 1.05)
    ax.set_ylabel("Pearson r")
    ax.set_title("Per-position conservation profile vs WT")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, rs):
        ax.annotate(f"{v:.3f}",
                    xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3 if v >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=9)

    # --- Panel C: Mutant MSA is a WT subset ---
    ax = axes[1, 0]
    frac_in_wt = [metrics[c]["frac_mut_in_wt"] for c in cohorts]
    colors = [cohort_color(c, i) for i, c in enumerate(cohorts)]
    bars = ax.bar(cohorts, frac_in_wt, color=colors,
                  edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("fraction (0–1)")
    ax.set_title("Mutant MSA hits that also appear in the WT MSA\n(near 1.0 ⇒ mutant MSA is a pruned WT MSA)")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, frac_in_wt):
        ax.annotate(f"{v:.3f}",
                    xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)

    # --- Panel D: Retention by identity bucket ---
    ax = axes[1, 1]
    n_buckets = len(ID_BUCKETS)
    bucket_centers = np.arange(n_buckets)
    width = 0.8 / max(len(cohorts), 1)
    for i, c in enumerate(cohorts):
        ret = metrics[c]["retention"]
        frac = [(kept / total) if total else 0.0 for (_, _, kept, total) in ret]
        ax.bar(bucket_centers + (i - (len(cohorts) - 1) / 2) * width,
               frac, width, label=c, color=cohort_color(c, i),
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(bucket_centers)
    ax.set_xticklabels([f"[{lo:.1f},{hi:.1f})" for lo, hi, _, _ in metrics[cohorts[0]]["retention"]],
                       rotation=0, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("WT-hit identity-to-WT-query bucket")
    ax.set_ylabel("retained in mutant MSA")
    ax.set_title("Close-homolog retention by identity bucket")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    # Annotate count above each bar
    for i, c in enumerate(cohorts):
        ret = metrics[c]["retention"]
        for bi, (_, _, kept, total) in enumerate(ret):
            if total == 0:
                continue
            x = bucket_centers[bi] + (i - (len(cohorts) - 1) / 2) * width
            y = kept / total
            ax.annotate(f"{kept}/{total}",
                        xy=(x, y), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=6.5, color="#333")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def plot_js_divergence(metrics: dict, out_path: Path) -> None:
    """Per-position JS divergence between WT and mutant aa distributions.

    Distribution-aware (compares the full per-column aa distribution, gaps
    included), unlike the Pearson-on-entropy summary which only compares
    *amounts* of conservation. Bounded in [0, ln 2 ≈ 0.693] nats.
    """
    cohorts = [k for k in metrics if k != "WT"]
    L = metrics["WT"]["aa_dist"].shape[0]
    pos = np.arange(L)

    fig, axes = plt.subplots(
        2, 1, figsize=(13, 6.5),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Top: per-position JSD per cohort, with mutated-position markers above plot.
    ax = axes[0]
    for i, c in enumerate(cohorts):
        jsd = metrics[c]["jsd_per_position"]
        if jsd.size == 0:
            continue
        ax.plot(pos, jsd,
                label=f"{c} (mean = {metrics[c]['mean_jsd']:.3f})",
                color=cohort_color(c, i), lw=0.9, alpha=0.85)
    ax.axhline(np.log(2), color="k", lw=0.6, ls="--", alpha=0.5)
    ax.text(L - 1, np.log(2), " max = ln 2",
            ha="right", va="bottom", fontsize=8, color="#444")
    ax.set_ylim(0, np.log(2) * 1.10)

    # Mutated-position tick rows above the plot.
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    for i, c in enumerate(cohorts):
        muts = metrics[c]["mutated_positions"]
        if muts.size == 0:
            continue
        y = ymax + 0.04 * yspan + i * 0.05 * yspan
        ax.scatter(muts, np.full_like(muts, y, dtype=float),
                   marker="|", s=50, color=cohort_color(c, i),
                   linewidths=1.0)
        ax.text(L + 1, y, f"{c} mutated sites ({muts.size})",
                va="center", fontsize=8, color=cohort_color(c, i))
    ax.set_ylim(ymin, ymax + 0.25 * yspan)

    ax.set_xlabel("position")
    ax.set_ylabel("JS divergence (nats)")
    ax.set_title(
        "Per-position JS divergence between WT and mutant amino-acid distributions "
        "(includes gap as 21st token)"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom: mean JSD per cohort.
    ax2 = axes[1]
    means = [metrics[c]["mean_jsd"] for c in cohorts]
    bars = ax2.bar(cohorts, means,
                   color=[cohort_color(c, i) for i, c in enumerate(cohorts)],
                   edgecolor="black", linewidth=0.6)
    ax2.axhline(np.log(2), color="k", lw=0.5, ls="--", alpha=0.4)
    ax2.set_ylim(0, np.log(2) * 1.10)
    ax2.set_ylabel("mean JSD (nats)")
    ax2.set_title("Mean per-position JSD per cohort")
    ax2.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, means):
        ax2.annotate(f"{v:.3f}",
                     xy=(b.get_x() + b.get_width() / 2, v),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def plot_entropy_profiles(metrics: dict, out_path: Path) -> None:
    cohorts = [k for k in metrics if k != "WT"]
    e_wt = metrics["WT"]["entropy"]
    L = e_wt.shape[0]
    pos = np.arange(L)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)

    # Top: per-position entropy overlay
    ax = axes[0]
    ax.plot(pos, e_wt, label="WT", color=cohort_color("WT", -1), lw=1.2)
    for i, c in enumerate(cohorts):
        ax.plot(pos, metrics[c]["entropy"],
                label=c, color=cohort_color(c, i), lw=1.0, alpha=0.85)
    ax.set_ylabel("Shannon entropy (nats)")
    ax.set_title("Per-position MSA entropy across cohorts")
    ax.legend(loc="upper right", ncol=4, fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom: per-position Δentropy per cohort, with mutated positions marked
    ax = axes[1]
    ax.axhline(0, color="k", lw=0.6)
    for i, c in enumerate(cohorts):
        d = metrics[c]["delta_entropy"]
        if d.size == 0:
            continue
        ax.plot(pos, d, label=c, color=cohort_color(c, i), lw=0.9, alpha=0.85)
    # Tick-marks at mutated positions, per cohort, stacked rows above plot
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    for i, c in enumerate(cohorts):
        muts = metrics[c]["mutated_positions"]
        if muts.size == 0:
            continue
        y = ymax + 0.05 * yspan + i * 0.05 * yspan
        ax.scatter(muts, np.full_like(muts, y, dtype=float),
                   marker="|", s=50, color=cohort_color(c, i),
                   linewidths=1.0)
        ax.text(L + 1, y, f"{c} mutated sites ({muts.size})",
                va="center", fontsize=8, color=cohort_color(c, i))
    ax.set_ylim(ymin, ymax + 0.25 * yspan)
    ax.set_xlabel("position")
    ax.set_ylabel("Δentropy (mut − WT)")
    ax.set_title("Per-position diversity loss; tick-rows above show mutated sites per cohort")
    ax.legend(loc="lower right", ncol=4, fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def write_csv(metrics: dict, out_path: Path, id_threshold: float) -> None:
    cohorts = [k for k in metrics if k != "WT"]
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "cohort", "depth", "neff",
            "query_identity", "shared_jaccard",
            "inter", "frac_mut_in_wt", "frac_wt_in_mut",
            "entropy_pearson_r", "mean_delta_entropy", "mean_jsd",
            *[f"retention_{lo:.1f}_{hi:.2f}" for lo, hi in ID_BUCKETS],
            f"id_threshold_for_neff",
        ])
        # WT row (most fields N/A)
        w.writerow([
            "WT", metrics["WT"]["depth"], f"{metrics['WT']['neff']:.3f}",
            "1.0", "1.0", "", "1.0", "1.0", "1.0", "0.0", "0.0",
            *["1.0" for _ in ID_BUCKETS],
            id_threshold,
        ])
        for c in cohorts:
            m = metrics[c]
            ret_fracs = [(kept / total) if total else 0.0 for (_, _, kept, total) in m["retention"]]
            w.writerow([
                c, m["depth"], f"{m['neff']:.3f}",
                f"{m['query_identity']:.4f}",
                f"{m['shared_jaccard']:.4f}",
                m["inter"],
                f"{m['frac_mut_in_wt']:.4f}",
                f"{m['frac_wt_in_mut']:.4f}",
                f"{m['entropy_pearson_r']:.4f}",
                f"{m['mean_delta_entropy']:.4f}",
                f"{m['mean_jsd']:.4f}",
                *[f"{f:.4f}" for f in ret_fracs],
                id_threshold,
            ])
    print(f"  saved -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wt", type=Path, required=True)
    p.add_argument("--cohort", action="append", required=True,
                   help="LABEL=PATH, e.g. --cohort p20=/path/to/file.a3m. Repeat.")
    p.add_argument("--out", type=Path, default=Path("reports/msa_tier1"))
    p.add_argument("--id-threshold", type=float, default=0.8)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"loading WT MSA: {args.wt}")
    wt = parse_a3m(args.wt)
    print(f"  -> depth={wt.depth}, L={wt.length}")

    mutants: dict[str, MSA] = {}
    for entry in args.cohort:
        if "=" not in entry:
            raise SystemExit(f"--cohort expects LABEL=PATH, got {entry!r}")
        label, path = entry.split("=", 1)
        mp = Path(path)
        print(f"loading cohort {label!r}: {mp}")
        mutants[label] = parse_a3m(mp)
        print(f"  -> depth={mutants[label].depth}, L={mutants[label].length}")

    print("\ncomputing metrics ...")
    metrics = compute_all(wt, mutants, args.id_threshold)

    print("\nrendering plots and CSV ...")
    plot_summary(metrics, args.out / "cohort_summary.png", args.id_threshold)
    plot_entropy_profiles(metrics, args.out / "entropy_profiles.png")
    plot_js_divergence(metrics, args.out / "js_divergence.png")
    write_csv(metrics, args.out / "cohort_metrics.csv", args.id_threshold)


if __name__ == "__main__":
    main()
