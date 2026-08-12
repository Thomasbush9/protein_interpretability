"""One standard protocol block, so two numbers can be compared by reading files.

This exists because of a specific failure. `svd_dz_v3.json` records its block,
dimension, seeds, split fraction and permutation count -- but not that its
headline curve is averaged over the LAST EIGHT LAYERS. That fact lived only in a
source comment. Its 0.731 was then compared against a final-layer,
leave-one-assay-out 0.758 as though the two were measured the same way, and the
difference was attributed to the wrong cause twice before anyone measured it.

Thirteen of sixteen archived analyses had no protocol block at all.

The rule this module enforces: an archived number must carry, in the same file,
everything needed to know what it is comparable to. Concretely --

  design        how rows were split. "leave-one-assay-out" and "within-assay,
                position-grouped" are different experiments and their numbers
                are not interchangeable; the training-row count usually differs
                by an order of magnitude and that alone moves Spearman.
  layer         which layer, or which window of layers. A single layer and a
                mean over eight are different measurements.
  features      what went in, and how wide it was BEFORE any truncation, plus
                the truncation actually applied. "128 pair channels" and "16
                of 128 pair channels" are different predictors.
  source        the archives read, so the upstream capture is recoverable.
  n_assays / n_train_rows   the units the interval is over, and how much data
                the fit actually saw.

`require` fails loudly rather than writing a file that will mislead later. A
missing field here costs a rerun; a missing field in an archive costs a
correction after someone has quoted the number.
"""

from __future__ import annotations

from datetime import datetime, timezone

REQUIRED = ("script", "design", "layer", "features", "source", "n_assays")


def protocol(**fields):
    """A complete protocol block. Raises if a required field is missing.

    Extra fields are kept as given -- analyses have their own specifics (lambda,
    number of permutations, orientation rule) and those belong here too.
    """
    missing = [k for k in REQUIRED if k not in fields or fields[k] is None]
    if missing:
        raise ValueError(
            f"protocol() is missing {missing}. Every archived number must say "
            f"what it is comparable to; see pi_protocol.__doc__ for why this "
            f"is enforced rather than encouraged.")
    out = {"recorded": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    out.update(fields)
    return out


def features(name, width, kept=None, note=None):
    """Describe a feature block: what it is, how wide, how much was kept.

    `width` is the number of columns BEFORE selection; `kept` after it. Leaving
    `kept` as None means nothing was dropped. The distinction is the one that
    made a 128-channel probe read as 0.696 while its label said 128.
    """
    d = {"name": name, "width": int(width),
         "kept": int(width if kept is None else kept)}
    d["truncated"] = d["kept"] < d["width"]
    if note:
        d["note"] = note
    return d


def layers(which, n_layers=None, window=None):
    """Describe the layer convention.

    `which` is "final", "all", or an index. `window` is the number of layers
    pooled when a curve averages several -- the field whose absence started
    this module.
    """
    d = {"which": which}
    if n_layers is not None:
        d["n_layers"] = int(n_layers)
    if window is not None:
        d["pooled_over_last"] = int(window)
    return d
