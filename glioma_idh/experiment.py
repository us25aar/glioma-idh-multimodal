"""Clinical baselines evaluated only on development patients."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .data import clinical_inputs, write_json
from .plots import save_svg_charts

MODEL_NAMES = {"majority", "age_logistic", "clinical_logistic", "random_forest", "xgboost"}


def make_pipeline(name: str, y_train: pd.Series, seed: int) -> Pipeline:
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    transforms = [("age", numeric, ["age"])]
    if name != "age_logistic":
        transforms.append(("sex", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sex"]))
    preprocess = ColumnTransformer(transforms, remainder="drop")
    if name == "majority":
        estimator = DummyClassifier(strategy="most_frequent")
    elif name in {"age_logistic", "clinical_logistic"}:
        estimator = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=5,
            class_weight="balanced", n_jobs=2, random_state=seed,
        )
    else:
        from xgboost import XGBClassifier
        ratio = float((y_train == 0).sum() / (y_train == 1).sum())
        estimator = XGBClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.8,
            colsample_bytree=1.0, reg_lambda=5.0, scale_pos_weight=ratio,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_jobs=2, random_state=seed,
        )
    return Pipeline([("preprocess", preprocess), ("classifier", estimator)])


def binary_metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError('Classification threshold must be between zero and one.')
    predictions = probabilities >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "sensitivity": float(recall_score(y, predictions, zero_division=0)),
        "specificity": float(recall_score(y, predictions, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
    }


def train_baselines(development: pd.DataFrame, models: list[str], seed: int, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The function accepts development data only; it has no test-set argument."""
    if not set(models) <= MODEL_NAMES or not models or len(set(models)) != len(models):
        raise ValueError("Specify a nonempty, unique list of supported models.")
    if development.patient_id.duplicated().any() or development.fold.lt(0).any():
        raise ValueError("Development patients and folds must be valid and unique.")
    rows, summaries, predictions = [], [], []
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in models:
        oof = np.full(len(development), np.nan)
        model_rows = []
        for fold in sorted(development.fold.unique()):
            validation_mask = development.fold.eq(fold)
            train = development.loc[~validation_mask]
            validation = development.loc[validation_mask]
            y_train = train.idh_label.astype(int)
            estimator = make_pipeline(name, y_train, seed)
            with threadpool_limits(limits=2):
                estimator.fit(clinical_inputs(train, name == "age_logistic"), y_train)
                probability = estimator.predict_proba(clinical_inputs(validation, name == "age_logistic"))[:, 1]
            oof[validation_mask.to_numpy()] = probability
            row = {
                "model": name, "fold": int(fold), "n_train": len(train), "n_validation": len(validation),
                **binary_metrics(validation.idh_label.astype(int).to_numpy(), probability),
            }
            rows.append(row)
            model_rows.append(row)
        if not np.isfinite(oof).all():
            raise ValueError("Every development patient must have exactly one finite out-of-fold prediction.")
        fold_table = pd.DataFrame(model_rows)
        summary = {"model": name, "folds": len(model_rows), "development_patients": len(development)}
        for metric in binary_metrics(development.idh_label.astype(int).to_numpy(), oof):
            summary[f"{metric}_mean"] = float(fold_table[metric].mean())
            summary[f"{metric}_std"] = float(fold_table[metric].std(ddof=1))
        summary["oof_roc_auc"] = float(roc_auc_score(development.idh_label.astype(int), oof))
        summaries.append(summary)
        predictions.append(pd.DataFrame({
            "patient_id": development.patient_id, "fold": development.fold,
            "idh_label": development.idh_label.astype(int), "model": name,
            "probability_mutant": oof, "prediction_scope": "development_out_of_fold",
        }))
        refit = make_pipeline(name, development.idh_label.astype(int), seed)
        with threadpool_limits(limits=2):
            refit.fit(clinical_inputs(development, name == "age_logistic"), development.idh_label.astype(int))
        joblib.dump({
            "pipeline": refit, "model_name": name, "features": ["age"] if name == "age_logistic" else ["age", "sex"],
            "fit_scope": "development_only", "positive_class": "IDH-mutant",
            "threshold": 0.5, "threshold_status": "preliminary_default_not_clinically_optimized",
        }, model_dir / f"{name}.joblib")
    fold_results = pd.DataFrame(rows)
    summary_results = pd.DataFrame(summaries).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    oof_results = pd.concat(predictions, ignore_index=True)
    fold_results.to_csv(output / "clinical_cv_folds.csv", index=False)
    summary_results.to_csv(output / "clinical_cv_summary.csv", index=False)
    oof_results.to_csv(output / "clinical_oof_predictions.csv", index=False)
    make_figures(development, oof_results, output / "figures")
    return summary_results, oof_results


def make_figures(development: pd.DataFrame, oof: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    counts = development.idh_label.value_counts()
    curves = []
    for name, frame in oof.groupby("model", sort=True):
        fpr, tpr, _ = roc_curve(frame.idh_label, frame.probability_mutant)
        auc = roc_auc_score(frame.idh_label, frame.probability_mutant)
        curves.append({"name": name, "auc": float(auc), "fpr": fpr.tolist(), "tpr": tpr.tolist()})
    write_json(output / "roc_points.json", curves)
    save_svg_charts(output, curves, int(counts.get(0, 0)), int(counts.get(1, 0)))


def environment_info() -> dict:
    packages = {}
    for name in ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "xgboost"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        gpu_info = gpu.stdout.strip() if gpu.returncode == 0 else "not detected"
    except (OSError, subprocess.TimeoutExpired):
        gpu_info = "not detected"
    return {
        "python": sys.version.split()[0], "platform": platform.system(),
        "packages": packages, "gpu": gpu_info,
        "experiment_device": "CPU; GPU not required for week-one clinical experiments",
    }
