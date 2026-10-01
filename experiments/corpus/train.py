"""Crop yield model training and evaluation."""

import math
from dataclasses import dataclass, field

RMSE_TARGET_KG_HA = 380.0
SPATIAL_FOLDS = 5


@dataclass
class FoldResult:
    index: int
    rmse: float
    r2: float
    excluded_rows: list = field(default_factory=list)


def spatial_folds(geometries, count: int = SPATIAL_FOLDS):
    """Split by spatial block so neighbouring plots cannot straddle a split."""
    ordered = sorted(range(len(geometries)), key=lambda i: geometries[i].centroid.x)
    size = math.ceil(len(ordered) / count)
    return [ordered[i * size : (i + 1) * size] for i in range(count)]


def impute_median(values, fallback):
    """Replace missing entries with the median of the observed values."""
    observed = sorted(v for v in values if v is not None)
    if not observed:
        return list(fallback)
    middle = len(observed) // 2
    median = (
        observed[middle]
        if len(observed) % 2
        else (observed[middle - 1] + observed[middle]) / 2
    )
    return [median if v is None else v for v in values]


def evaluate(model, train_index, test_index, target):
    """Score the model on one held out spatial fold."""
    predictions = model.predict(train_index, test_index)
    errors = [p - t for p, t in zip(predictions, target[test_index])]
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    r2 = 1 - sum(errors) ** 2 / sum((t - sum(target[test_index]) / len(errors)) ** 2 for t in target[test_index])
    return FoldResult(index=test_index[0], rmse=rmse, r2=r2)
