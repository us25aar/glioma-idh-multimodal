"""Lock one patient holdout and assign development-only stratified folds."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import write_json


def create_splits(cohort: pd.DataFrame, seed: int, test_fraction: float, folds: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if cohort.patient_id.duplicated().any():
        raise ValueError("Patient IDs must be unique before partitioning.")
    if set(cohort.idh_label.dropna().unique()) != {0, 1} or cohort.idh_label.isna().any():
        raise ValueError("Both binary classes and no unknown labels are required for splitting.")
    if not 0 < test_fraction < 0.5 or folds < 2:
        raise ValueError("Require 0 < test_fraction < 0.5 and at least two development folds.")
    cohort = cohort.sort_values("patient_id").reset_index(drop=True).copy()
    development, test = train_test_split(
        cohort, test_size=test_fraction, stratify=cohort.idh_label, random_state=seed,
    )
    development = development.sort_values("patient_id").reset_index(drop=True)
    test = test.sort_values("patient_id").reset_index(drop=True)
    actual_folds = min(folds, int(development.idh_label.value_counts().min()))
    if actual_folds < 2:
        raise ValueError("Insufficient minority patients for development cross-validation.")
    development["fold"] = -1
    splitter = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
    for fold, (_, validation_indices) in enumerate(splitter.split(development, development.idh_label.astype(int))):
        development.loc[validation_indices, "fold"] = fold
    assert set(development.patient_id).isdisjoint(test.patient_id)
    return development, test, actual_folds


def persist_splits(development: pd.DataFrame, test: pd.DataFrame, output: Path, source_hash: str, seed: int) -> None:
    lock_path = output / "split_lock.json"
    lock = {
        "source_sha256": source_hash, "seed": seed,
        "development_ids": development.patient_id.tolist(),
        "test_ids": test.patient_id.tolist(),
        "fold_assignments": {str(r.patient_id): int(r.fold) for r in development.itertuples()},
        "test_evaluation_performed": False,
        "instruction": "Keep assignments fixed when MRI QC later excludes patients; do not move patients between partitions.",
    }
    if lock_path.exists() and json.loads(lock_path.read_text(encoding="utf-8")) != lock:
        raise ValueError("Existing split lock differs. Preserve it; use a separately named output for a reviewed protocol revision.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(lock_path, lock)
    development.to_csv(output / "development_patients.csv", index=False)
    test[["patient_id"]].to_csv(output / "locked_test_ids.csv", index=False)
