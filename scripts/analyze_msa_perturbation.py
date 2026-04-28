#!/usr/bin/env python
"""Compare perturbed MSAs to a wild-type MSA across perturbation levels.

For each perturbed MSA the script computes:

* depth (number of sequences) and length (number of reference columns)
* Neff at multiple identity thresholds (50/62/80/90/95 %)
* per-hit identity to the query
* per-column AA-distribution Pearson correlation against the WT MSA
* set-overlap of homolog IDs vs the WT MSA: ``frac_pert_in_wt``,
  ``frac_wt_in_pert``, ``jaccard``
* per-position conservation against the mutant query

Files are discovered by globbing ``--input-dir`` for ``*.a3m``. The
perturbation level is inferred from any ``p\\d+`` segment in the path
(``p20``, ``p40``, ``p70``, …); files outside such a segment are skipped
unless ``--include-unlabeled`` is set.

Outputs (under ``--output-dir`` unless overridden):

* per-MSA results CSV — one row per perturbed MSA, default
  ``<output-dir>/msa_metrics.csv``, override with ``--global-csv PATH``.
  First two columns are ``sequence_idx`` (integer parsed from the path /
  filename, e.g. ``seq_00318`` or ``318_protein_.a3m`` → ``318``) and
  ``predicted_path`` (the source MSA path), making the CSV ready to join
  against other per-sequence result tables.
* ``msa_aggregate_metrics.csv`` — mean/std per perturbation level
* ``scalar_metrics_by_level.png`` — depth, Neff80, corr_mean
* ``neff_vs_identity.png`` — Neff curve at each identity threshold
* ``pid_to_query_distribution.png`` — KDE of per-hit identity to query
* ``homolog_set_overlap.png`` — frac_pert_in_wt, frac_wt_in_pert, jaccard
* ``column_correlation_per_position.png``
* ``conservation_per_position.png``

Example::

    uv run python scripts/analyze_msa_perturbation.py \\
        --source /path/to/original/msa/A_protein_.a3m \\
        --input-dir /n/holylfs06/.../augmented \\
        --output-dir /path/to/output_dir \\
        --global-csv /path/to/output_dir/msa_metrics.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from protein_interpretability.msa_analysis import (
    DEFAULT_NEFF_THRESHOLDS,
    MSARecord,
    WTReference,
    analyze_msa,
    discover_msas,
    load_wt_reference,
    perturbation_level_from_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


LEVEL_PALETTE = {
    "p20": "#1f77b4",
    "p40": "#ff7f0e",
    "p70": "#d62728",
}
WT_REF_COLOR = "#444444"
NEFF_KEY = 0.8  # threshold used for the headline scalar Neff column


def _level_sort_key(level: str) -> tuple[int, int]:
    try:
        return (1, int(level.lstrip("p")))
    except ValueError:
        return (2, 0)


def _setup_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


# ---------------------------------------------------------------------------
# Per-file analysis (parallelised)
# ---------------------------------------------------------------------------

# Workers receive the WT reference once via initializer to avoid repeatedly
# pickling its arrays per task.
_WT_REF: WTReference | None = None


def _worker_init(wt_path: str) -> None:
    global _WT_REF
    _WT_REF = load_wt_reference(wt_path)


def _analyze_one(args: tuple[str, str]) -> MSARecord | None:
    path, level = args
    try:
        wt = _WT_REF if _WT_REF is not None else load_wt_reference(path)
        return analyze_msa(path, wt, level=level)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed on %s: %s", path, e)
        return None


def collect_records(
    paths_with_level: list[tuple[Path, str]],
    wt: WTReference,
    *,
    workers: int,
) -> list[MSARecord]:
    work = [(str(p), lv) for p, lv in paths_with_level]
    if workers <= 1:
        global _WT_REF
        _WT_REF = wt
        return [r for r in (_analyze_one(item) for item in work) if r is not None]

    records: list[MSARecord] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(str(wt.path),),
    ) as ex:
        futures = [ex.submit(_analyze_one, item) for item in work]
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                records.append(r)
    return records


# ---------------------------------------------------------------------------
# DataFrames
# ---------------------------------------------------------------------------

def _neff_col(t: float) -> str:
    return f"neff_{int(round(t * 100)):02d}"


BASE_COLS = [
    "sequence_idx", "predicted_path", "level", "seq_id",
    "depth", "length",
    "corr_mean", "corr_median",
    "frac_pert_in_wt", "frac_wt_in_pert", "jaccard",
    "n_shared", "n_pert_unique", "n_wt_unique",
    "length_match",
]


def _record_to_row(r: MSARecord) -> dict:
    return {
        "sequence_idx": r.sequence_idx,
        "predicted_path": str(r.path),
        "level": r.level,
        "seq_id": r.seq_id,
        "depth": r.depth,
        "length": r.length,
        "corr_mean": r.corr_mean,
        "corr_median": r.corr_median,
        "frac_pert_in_wt": r.frac_pert_in_wt,
        "frac_wt_in_pert": r.frac_wt_in_pert,
        "jaccard": r.jaccard,
        "n_shared": r.n_shared,
        "n_pert_unique": r.n_pert_unique,
        "n_wt_unique": r.n_wt_unique,
        "length_match": r.length_match,
    }


def build_per_file_df(
    records: list[MSARecord],
    thresholds: tuple[float, ...],
) -> pd.DataFrame:
    neff_cols = [_neff_col(t) for t in thresholds]
    rows = []
    for r in records:
        row = _record_to_row(r)
        for t, col in zip(thresholds, neff_cols):
            row[col] = r.neff_curve.get(float(t), float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows, columns=BASE_COLS + neff_cols)
    if not df.empty:
        df["level_order"] = df["level"].map(_level_sort_key)
        df = df.sort_values(
            ["level_order", "sequence_idx", "seq_id"], na_position="last",
        ).drop(columns="level_order").reset_index(drop=True)
        # sequence_idx is integer-or-null; pandas keeps it as float when nulls
        # are present. Use the nullable Int64 dtype to preserve the integer view.
        df["sequence_idx"] = df["sequence_idx"].astype("Int64")
    return df


def build_aggregate_df(
    per_file: pd.DataFrame,
    thresholds: tuple[float, ...],
) -> pd.DataFrame:
    if per_file.empty:
        return pd.DataFrame()
    metrics = (
        ["depth", "length", "corr_mean", "frac_pert_in_wt", "frac_wt_in_pert", "jaccard"]
        + [_neff_col(t) for t in thresholds]
    )
    grouped = per_file.groupby("level")[metrics].agg(["mean", "std", "count"])
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped["level_order"] = grouped["level"].map(_level_sort_key)
    grouped = grouped.sort_values("level_order").drop(columns="level_order")
    return grouped


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _stack_by_level(records: list[MSARecord], attr: str) -> dict[str, np.ndarray]:
    by_level: dict[str, list[np.ndarray]] = {}
    for r in records:
        by_level.setdefault(r.level, []).append(getattr(r, attr))
    out: dict[str, np.ndarray] = {}
    for level, arrs in by_level.items():
        lens = [a.shape[0] for a in arrs]
        target = max(set(lens), key=lens.count)
        kept = [a for a in arrs if a.shape[0] == target]
        if len(kept) != len(arrs):
            logger.warning(
                "Level %s: dropping %d/%d MSAs with length != %d (per-position plots)",
                level, len(arrs) - len(kept), len(arrs), target,
            )
        out[level] = (
            np.stack(kept, axis=0)
            if kept
            else np.empty((0, target), dtype=np.float32)
        )
    return out


def _plot_box(
    df: pd.DataFrame,
    metric: str,
    *,
    ax,
    title: str,
    ylabel: str,
    wt_value: float | None = None,
    log: bool = False,
) -> None:
    levels = sorted(df["level"].unique(), key=_level_sort_key)
    palette = {lv: LEVEL_PALETTE.get(lv, "#888888") for lv in levels}
    sns.boxplot(
        data=df, x="level", y=metric, hue="level", order=levels,
        palette=palette, legend=False, ax=ax, showmeans=True, width=0.55,
        meanprops={"marker": "D", "markerfacecolor": "white",
                   "markeredgecolor": "black", "markersize": 6},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )
    sns.stripplot(
        data=df, x="level", y=metric, order=levels,
        ax=ax, color="black", alpha=0.4, size=2.5, jitter=0.18,
    )
    if wt_value is not None and np.isfinite(wt_value):
        ax.axhline(wt_value, color=WT_REF_COLOR, linestyle="--",
                   linewidth=1.2, label=f"WT = {wt_value:.3g}")
        ax.legend(loc="best", frameon=False)
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Perturbation level")
    ax.set_ylabel(ylabel)


def _plot_band(
    stacks: dict[str, np.ndarray],
    *,
    ax,
    title: str,
    ylabel: str,
    wt_curve: np.ndarray | None = None,
) -> None:
    levels = sorted(stacks, key=_level_sort_key)
    if wt_curve is not None and wt_curve.size:
        ax.plot(np.arange(wt_curve.shape[0]), wt_curve, color=WT_REF_COLOR,
                linewidth=1.6, linestyle="--", label="WT")
    for level in levels:
        arr = stacks[level]
        if arr.size == 0:
            continue
        x = np.arange(arr.shape[1])
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        color = LEVEL_PALETTE.get(level, "#888888")
        label = f"{level} (n={arr.shape[0]})"
        ax.plot(x, mean, color=color, label=label, linewidth=1.4)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel("Reference position")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False)


def _plot_neff_vs_identity(
    records: list[MSARecord],
    wt: WTReference,
    thresholds: tuple[float, ...],
    *,
    ax,
) -> None:
    by_level: dict[str, list[np.ndarray]] = {}
    for r in records:
        vec = np.asarray([r.neff_curve.get(float(t), np.nan) for t in thresholds],
                         dtype=np.float32)
        by_level.setdefault(r.level, []).append(vec)

    x = np.asarray(thresholds)
    wt_curve = np.asarray([wt.neff_curve.get(float(t), np.nan) for t in thresholds],
                          dtype=np.float32)
    if np.all(np.isfinite(wt_curve)):
        ax.plot(x, wt_curve, color=WT_REF_COLOR, linestyle="--", linewidth=1.6,
                marker="D", markersize=5, label="WT")

    for level in sorted(by_level, key=_level_sort_key):
        mat = np.stack(by_level[level], axis=0)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        color = LEVEL_PALETTE.get(level, "#888888")
        n = mat.shape[0]
        ax.plot(x, mean, color=color, linewidth=1.6, marker="o", markersize=5,
                label=f"{level} (n={n})")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(round(t * 100))}%" for t in thresholds])
    ax.set_yscale("log")
    ax.set_xlabel("Identity threshold")
    ax.set_ylabel("Neff (log)")
    ax.set_title("Neff vs identity threshold")
    ax.legend(loc="best", frameon=False)


def _plot_pid_distribution(records: list[MSARecord], wt: WTReference, *, ax) -> None:
    rows: list[dict] = []
    for r in records:
        for v in r.pid_to_query.tolist():
            rows.append({"level": r.level, "pid": v})
    for v in wt.pid_to_query.tolist():
        rows.append({"level": "WT", "pid": v})
    if not rows:
        return
    long = pd.DataFrame(rows)
    levels = ["WT", *sorted(long["level"][long["level"] != "WT"].unique(),
                            key=_level_sort_key)]
    palette = {lv: WT_REF_COLOR if lv == "WT" else LEVEL_PALETTE.get(lv, "#888888")
               for lv in levels}
    for level in levels:
        sub = long[long["level"] == level]
        if sub.empty:
            continue
        sns.kdeplot(
            data=sub, x="pid", ax=ax, color=palette[level], linewidth=1.8,
            fill=False, clip=(0, 1), label=f"{level} (n_hits={len(sub)})",
            linestyle="--" if level == "WT" else "-",
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Per-hit identity to query")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of pairwise identity to query")
    ax.legend(loc="best", frameon=False)


def make_plots(
    per_file: pd.DataFrame,
    records: list[MSARecord],
    wt: WTReference,
    output_dir: Path,
    thresholds: tuple[float, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scalar metrics: depth, Neff80 (log), mean column correlation.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    _plot_box(per_file, "depth", ax=axes[0],
              title="MSA depth", ylabel="# sequences (log)",
              wt_value=wt.depth, log=True)
    neff_col = _neff_col(NEFF_KEY)
    _plot_box(per_file, neff_col, ax=axes[1],
              title=f"Neff @ {int(NEFF_KEY * 100)} % identity",
              ylabel="Neff (log)",
              wt_value=wt.neff_curve.get(float(NEFF_KEY)),
              log=True)
    corr_df = per_file[per_file["corr_mean"].notna()]
    _plot_box(corr_df, "corr_mean", ax=axes[2],
              title="Mean per-column AA-dist correlation vs WT",
              ylabel="Pearson r", wt_value=1.0)
    fig.suptitle("Per-MSA scalar metrics by perturbation level",
                 y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "scalar_metrics_by_level.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 2. Neff vs identity threshold.
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_neff_vs_identity(records, wt, thresholds, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "neff_vs_identity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Pairwise identity to query distribution.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    _plot_pid_distribution(records, wt, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "pid_to_query_distribution.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 4. Homolog set overlap vs WT.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    _plot_box(per_file, "frac_pert_in_wt", ax=axes[0],
              title="Perturbed hits also in WT",
              ylabel="|pert AND WT| / |pert|", wt_value=1.0)
    _plot_box(per_file, "frac_wt_in_pert", ax=axes[1],
              title="WT hits recovered in perturbed",
              ylabel="|pert AND WT| / |WT|", wt_value=1.0)
    _plot_box(per_file, "jaccard", ax=axes[2],
              title="Jaccard(pert, WT)",
              ylabel="|intersection| / |union|", wt_value=1.0)
    for ax in axes:
        ax.set_ylim(-0.02, 1.05)
    fig.suptitle("Homolog set overlap with WT MSA",
                 y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "homolog_set_overlap.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # 5. Per-position column correlation (only MSAs with matching length).
    corr_records = [r for r in records if r.length_match]
    if corr_records:
        corr_stacks = _stack_by_level(corr_records, "column_corr")
        fig, ax = plt.subplots(figsize=(14, 4.8))
        _plot_band(
            corr_stacks, ax=ax,
            title="Per-column AA-distribution correlation vs WT (mean ± std across MSAs)",
            ylabel="Pearson r",
        )
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(output_dir / "column_correlation_per_position.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    else:
        logger.warning("No MSAs with matching length — skipping column-correlation plot")

    # 6. Per-position conservation, with WT as reference curve.
    cons_stacks = _stack_by_level(records, "conservation")
    fig, ax = plt.subplots(figsize=(14, 4.8))
    _plot_band(
        cons_stacks, ax=ax,
        title="Per-position conservation vs query (mean ± std across MSAs)",
        ylabel="conservation",
        wt_curve=wt.conservation,
    )
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "conservation_per_position.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", type=Path, required=True,
                   help="Path to the wild-type (reference) .a3m MSA.")
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory tree to search for perturbed .a3m files.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where plots and the aggregate CSV will be written.")
    p.add_argument("--global-csv", type=Path, default=None,
                   help="Explicit path for the global per-MSA results CSV "
                        "(default: <output-dir>/msa_metrics.csv). Parent dirs "
                        "are created if needed.")
    p.add_argument("--pattern", default="**/*.a3m",
                   help="Glob pattern under --input-dir (default: %(default)s).")
    p.add_argument("--levels", nargs="*", default=None,
                   help="Optional whitelist of levels to keep (e.g. p20 p40 p70).")
    p.add_argument("--include-unlabeled", action="store_true",
                   help="Include MSAs whose path has no p<NN> segment "
                        "(label them 'unlabeled').")
    p.add_argument("--neff-thresholds", type=float, nargs="*",
                   default=list(DEFAULT_NEFF_THRESHOLDS),
                   help="Identity thresholds for Neff (default: %(default)s).")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel processes for per-MSA work (default: %(default)s).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source.is_file():
        logger.error("Source MSA not found: %s", args.source)
        sys.exit(1)
    if not args.input_dir.is_dir():
        logger.error("Input directory not found: %s", args.input_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _setup_style()

    thresholds = tuple(float(t) for t in args.neff_thresholds)

    logger.info("Loading WT reference: %s", args.source)
    wt = load_wt_reference(args.source, neff_thresholds=thresholds)
    logger.info("WT depth=%d length=%d n_hits=%d",
                wt.depth, wt.length, len(wt.hit_id_set))
    logger.info("WT Neff curve: %s",
                ", ".join(f"{int(t * 100)}%={v:.1f}"
                          for t, v in wt.neff_curve.items()))

    discovered = discover_msas(args.input_dir, pattern=args.pattern)
    src_resolved = args.source.resolve()
    discovered = [p for p in discovered if p != src_resolved]

    paths_with_level: list[tuple[Path, str]] = []
    for p in discovered:
        level = perturbation_level_from_path(p)
        if level == "original":
            if not args.include_unlabeled:
                continue
            level = "unlabeled"
        if args.levels and level not in args.levels:
            continue
        paths_with_level.append((p, level))

    by_level: dict[str, int] = {}
    for _, lv in paths_with_level:
        by_level[lv] = by_level.get(lv, 0) + 1
    logger.info("Discovered %d perturbed MSA files", len(paths_with_level))
    for lv in sorted(by_level, key=_level_sort_key):
        logger.info("  %s: %d", lv, by_level[lv])
    if not paths_with_level:
        logger.error("No perturbed MSAs to analyze.")
        sys.exit(1)

    records = collect_records(paths_with_level, wt, workers=args.workers)
    if not records:
        logger.error("No MSAs were successfully analyzed.")
        sys.exit(1)
    logger.info("Analyzed %d perturbed MSAs", len(records))

    per_file = build_per_file_df(records, thresholds)
    aggregate = build_aggregate_df(per_file, thresholds)

    per_file_path = args.global_csv or (args.output_dir / "msa_metrics.csv")
    per_file_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.output_dir / "msa_aggregate_metrics.csv"
    per_file.to_csv(per_file_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    logger.info("Wrote %s (%d rows)", per_file_path, len(per_file))
    logger.info("Wrote %s", aggregate_path)

    make_plots(per_file, records, wt, args.output_dir, thresholds)
    logger.info("Plots written to %s", args.output_dir)


if __name__ == "__main__":
    main()
