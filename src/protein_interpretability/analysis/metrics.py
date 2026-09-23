"""Metrics beyond Spearman, for checking that a conclusion is not metric-bound.

Everything in this project is reported as a within-assay Spearman averaged over
assays. That is the ProteinGym convention and it is the right primary statistic
for a PAIRED comparison -- two predictors on identical rows, where a monotone
metric cannot favour one by accident. But it is not the only thing a reader
cares about, and it has three known blind spots:

  * it weights the whole ranking equally, while a protein engineer only reads
    the top of it;
  * it is insensitive to where the decision boundary sits, while a variant-effect
    reader wants deleterious-vs-tolerated;
  * on a compressed or bimodal assay a mediocre Spearman can coexist with clean
    separation of the two modes, and a good one with none.

So the same predictions are scored four ways. If the ordering of predictors is
stable across all four, "we report Spearman" is a presentation choice. If it
flips, Spearman was hiding something and the paper has to say which metric it
means.

ORIENTATION. Every metric except Spearman needs the prediction pointing the same
way as the label, and PC2 by the report's kl_glob convention points the other
way. The sign is taken ONCE from the basis assays and applied unchanged, never
re-chosen per assay -- doing that per assay would silently convert rho into
|rho| and inflate every number here.

BINARISATION. `DMS_binarization_cutoff` from the ProteinGym reference table, on
the true scores only. Predictions are thresholded at their own n_pos-th largest
value, so the predicted positive count matches the true one and MCC is not a
function of an arbitrary offset.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _clean(pred, y):
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    m = np.isfinite(pred) & np.isfinite(y)
    return pred[m], y[m]


def auc(pred, y, cutoff):
    """Rank AUC for separating y > cutoff from y <= cutoff."""
    pred, y = _clean(pred, y)
    pos = y > cutoff
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(pred)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mcc(pred, y, cutoff):
    """MCC with the predicted positive count matched to the true one."""
    pred, y = _clean(pred, y)
    pos = y > cutoff
    n1 = int(pos.sum())
    if n1 == 0 or n1 == len(y):
        return np.nan
    thr = np.sort(pred)[::-1][n1 - 1]
    php = pred >= thr
    tp = float((php & pos).sum())
    tn = float((~php & ~pos).sum())
    fp = float((php & ~pos).sum())
    fn = float((~php & pos).sum())
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else np.nan


def ndcg_top(pred, y, frac=0.10):
    """NDCG over the predicted top `frac`, linear gain on min-max scaled y.

    The top-focused metric: it asks whether the variants the model ranks first
    are actually the good ones, and ignores the rest of the list.
    """
    pred, y = _clean(pred, y)
    n = len(y)
    k = max(1, int(round(n * frac)))
    rel = y - y.min()
    rng = rel.max()
    if rng <= 0:
        return np.nan
    rel = rel / rng
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    got = rel[np.argsort(-pred)[:k]]
    best = np.sort(rel)[::-1][:k]
    idcg = float((best * disc).sum())
    return float((got * disc).sum() / idcg) if idcg > 0 else np.nan


def recall_top(pred, y, frac=0.10):
    """Fraction of the true top `frac` that the predicted top `frac` contains."""
    pred, y = _clean(pred, y)
    n = len(y)
    k = max(1, int(round(n * frac)))
    true_top = set(np.argsort(-y)[:k].tolist())
    pred_top = set(np.argsort(-pred)[:k].tolist())
    return float(len(true_top & pred_top) / k)


def all_metrics(pred, y, cutoff, frac=0.10):
    import pi_stats
    return {"spearman": pi_stats.spearman(pred, y),
            "auc": auc(pred, y, cutoff) if cutoff is not None else np.nan,
            "mcc": mcc(pred, y, cutoff) if cutoff is not None else np.nan,
            "ndcg_top10": ndcg_top(pred, y, frac),
            "recall_top10": recall_top(pred, y, frac)}
