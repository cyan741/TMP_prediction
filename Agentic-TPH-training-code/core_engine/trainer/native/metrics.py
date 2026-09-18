"""Metrics as a typed return value instead of a nine-tuple and a print."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class Metrics:
    roc_auc: float
    accuracy: float
    mcc: float
    f1: float
    sensitivity: float
    specificity: float
    precision: float
    recall: float
    aupr: float
    tn: int
    fp: int
    fn: int
    tp: int

    @property
    def selection_score(self) -> float:
        """Return validation AUROC for checkpoint selection, independent of the threshold."""

        return self.roc_auc

    @property
    def _superseded_selection_score(self) -> float:
        """Legacy mean of finite AUROC/AUPRC values; unused for checkpoint selection."""

        values = [self.roc_auc, self.aupr]
        finite = [value for value in values if not math.isnan(value)]
        return float(np.mean(finite)) if finite else float("nan")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selection_score"] = self.selection_score
        return payload


def _safe(numerator: float, denominator: float) -> float:
    """Ratio with sklearn's zero-division convention rather than NaN.

    An empty denominator means the quantity was never observed (no predicted
    positives, no true positives), which scores as zero.  NaN here would
    propagate into the selection score and silently disable checkpointing.
    """

    return float(numerator / denominator) if denominator else 0.0


def evaluate_predictions(
    y_true: Sequence[int], y_prob: Sequence[float], *, threshold: float = 0.5
) -> Metrics:
    """Compute binary classification metrics from labels and positive-class scores.

    Class predictions are derived here using the supplied threshold so the
    reported metrics and threshold refer to the same classification rule.
    """

    true = np.asarray(list(y_true), dtype=int)
    prob = np.asarray(list(y_prob), dtype=float)
    if true.size == 0:
        raise ValueError("no predictions to score")
    pred = (prob > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel().tolist()
    precision = _safe(tp, tp + fp)
    recall = _safe(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    single_class = len(set(true.tolist())) < 2
    curve_precision, curve_recall, _ = precision_recall_curve(true, prob)
    return Metrics(
        roc_auc=float("nan") if single_class else float(roc_auc_score(true, prob)),
        accuracy=_safe(tp + tn, tn + fp + fn + tp),
        mcc=float(matthews_corrcoef(true, pred)),
        f1=f1,
        sensitivity=_safe(tp, tp + fn),
        specificity=_safe(tn, tn + fp),
        precision=precision,
        recall=recall,
        aupr=float(auc(curve_recall, curve_precision)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def best_threshold(y_true, y_prob, *, grid: int = 199) -> tuple[float, float]:
    """The cut that maximises MCC, chosen on validation and then frozen.

    Returns (threshold, mcc).  MCC rather than F1 because F1 ignores true
    negatives entirely, so on a balanced frame it can be maximised by a cut that
    calls almost everything positive.  The sweep is over interior points only;
    a threshold of exactly 0 or 1 predicts one class for every row, which scores
    MCC = 0 and is never the answer this is asked for.
    """

    true = np.asarray(list(y_true), dtype=int)
    prob = np.asarray(list(y_prob), dtype=float)
    if true.size == 0 or len(set(true.tolist())) < 2:
        return 0.5, float("nan")
    best, best_mcc = 0.5, -2.0
    for step in range(1, grid + 1):
        cut = step / (grid + 1)
        score = matthews_corrcoef(true, (prob > cut).astype(int))
        if score > best_mcc:
            best, best_mcc = float(cut), float(score)
    return best, best_mcc


def mean_metrics(rounds: Sequence[Metrics]) -> dict[str, float]:
    """Average metrics over evaluation repeats, ignoring undefined values."""

    if not rounds:
        raise ValueError("no rounds to average")
    names = ("roc_auc", "accuracy", "mcc", "f1", "aupr", "precision", "recall")
    result = {
        f"{name}_avg": float(np.nanmean([getattr(item, name) for item in rounds]))
        for name in names
    }
    result["selection_score_avg"] = float(
        np.nanmean([item.selection_score for item in rounds])
    )
    return result
