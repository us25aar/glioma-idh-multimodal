"""Patient-paired uncertainty analysis. Does not unlock the reserved cohort."""

import numpy as np
from sklearn.metrics import roc_auc_score


def paired_auc_bootstrap(y, first, second, samples=2000, seed=42):
    y, first, second = np.asarray(y), np.asarray(first), np.asarray(second)
    if y.ndim != 1 or first.shape != y.shape or second.shape != y.shape:
        raise ValueError("Patient-aligned arrays with matching shape are required.")
    if set(np.unique(y)) != {0, 1} or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Both classes and finite paired predictions required.")
    if samples < 100:
        raise ValueError("At least 100 bootstrap resamples required.")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in [0, 1]]
    differences = []
    for _ in range(samples):
        idx = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        differences.append(roc_auc_score(y[idx], first[idx]) - roc_auc_score(y[idx], second[idx]))
    return {"auc_difference": float(roc_auc_score(y, first) - roc_auc_score(y, second)),
            "ci95": np.percentile(differences, [2.5, 97.5]).tolist(), "samples": samples,
            "method": "stratified paired patient bootstrap; conditional on fitted predictions"}
