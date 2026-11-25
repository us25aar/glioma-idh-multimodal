"""Small patient-level MRI, clinical and fusion heads with fold-only fitting."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from .data import clinical_inputs, sha256, write_json
from .experiment import binary_metrics


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


class PatientHead(nn.Module):
    def __init__(self, image_dim: int, clinical_dim: int, mode: str, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        if mode not in {"mri", "clinical", "fusion"}:
            raise ValueError("Mode must be mri, clinical or fusion.")
        self.mode = mode
        self.clinical = nn.Sequential(nn.Linear(clinical_dim, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 16), nn.ReLU())
        input_dim = (image_dim if mode != "clinical" else 0) + (16 if mode != "mri" else 0)
        self.classifier = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        if self.mode == "mri":
            fused = image
        elif self.mode == "clinical":
            fused = self.clinical(clinical)
        else:
            fused = torch.cat([image, self.clinical(clinical)], dim=1)
        return self.classifier(fused).squeeze(1)


def clinical_transformer() -> ColumnTransformer:
    age = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())])
    return ColumnTransformer([("age", age, ["age"]), ("sex", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sex"])], remainder="drop")


def fit_head(image_train, clinical_train, y_train, image_val, clinical_val, y_val,
             mode: str, seed: int, max_epochs: int = 50, patience: int = 8, fixed_epochs: int | None = None):
    seed_all(seed)
    model = PatientHead(image_train.shape[1], clinical_train.shape[1], mode)
    train_x = torch.as_tensor(image_train, dtype=torch.float32)
    train_c = torch.as_tensor(clinical_train, dtype=torch.float32)
    train_y = torch.as_tensor(y_train, dtype=torch.float32)
    val_x = torch.as_tensor(image_val, dtype=torch.float32)
    val_c = torch.as_tensor(clinical_val, dtype=torch.float32)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((train_y == 0).sum() / (train_y == 1).sum())))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_auc, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, (fixed_epochs or max_epochs) + 1):
        model.train()
        order = torch.randperm(len(train_y))
        losses = []
        for idx in order.split(32):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(train_x[idx], train_c[idx]), train_y[idx])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            probability = torch.sigmoid(model(val_x, val_c)).numpy()
        auc = float(roc_auc_score(y_val, probability)) if len(np.unique(y_val)) == 2 else 0.5
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation_auc": auc})
        if auc > best_auc + 1e-6:
            best_auc, best_epoch, best_state = auc, epoch, copy.deepcopy(model.state_dict())
        if fixed_epochs is None and epoch - best_epoch >= patience:
            break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        prediction = torch.sigmoid(model(val_x, val_c)).numpy()
    return model, prediction, history, fixed_epochs or best_epoch


def load_common_development(features_path: Path, lock_path: Path, development_csv: Path):
    lock = json.loads(lock_path.read_text())
    development = pd.read_csv(development_csv)
    if set(development.patient_id) != set(lock["development_ids"]):
        raise ValueError("Development CSV does not match the locked patient set.")
    if any(int(row.fold) != int(lock["fold_assignments"][row.patient_id]) for row in development.itertuples()):
        raise ValueError("Development fold assignments differ from the lock.")
    with np.load(features_path, allow_pickle=False) as bundle:
        ids, features = bundle["patient_ids"].astype(str), bundle["features"]
    if len(set(ids)) != len(ids) or features.shape != (len(ids), 2048) or not np.isfinite(features).all():
        raise ValueError("Invalid or duplicate MRI feature records.")
    # Restrict to development IDs even if a cache also contains reserved test inputs.
    positions = {patient: index for index, patient in enumerate(ids)}
    common = development.loc[development.patient_id.isin(positions)].copy().reset_index(drop=True)
    if len(common) < 10 or common.idh_label.nunique() != 2:
        raise ValueError("At least ten common development patients and both classes are required; use software tests for smaller pilots.")
    image = features[[positions[patient] for patient in common.patient_id]]
    if set(common.patient_id) & set(lock["test_ids"]):
        raise ValueError("Test patient entered development training.")
    return common, image


def run_development(features_path: Path, feature_manifest: Path, development_csv: Path, lock_path: Path,
                    output: Path, max_epochs: int = 50, seed: int = 42) -> dict:
    if max_epochs < 1:
        raise ValueError("Maximum epochs must be positive.")
    started = time.perf_counter()
    provenance = json.loads(feature_manifest.read_text())
    if provenance.get("data_scopes") != ["real_utsw"]:
        raise ValueError("Research development experiments require real UTSW feature provenance; synthetic tests cannot be reported as patient results.")
    if provenance.get("features_sha256") != sha256(features_path):
        raise ValueError("Feature content does not match provenance.")
    common, image = load_common_development(features_path, lock_path, development_csv)
    output.mkdir(parents=True, exist_ok=True)
    summaries, predictions, all_history = [], [], []
    for mode in ["clinical", "mri", "fusion"]:
        oof = np.full(len(common), np.nan)
        epochs = []
        fold_results = []
        for fold in sorted(common.fold.unique()):
            val = common.fold.to_numpy() == fold
            train = ~val
            if min(np.bincount(common.idh_label.to_numpy()[train], minlength=2)) < 2 or len(np.unique(common.idh_label.to_numpy()[val])) < 2:
                raise ValueError("Each retained fold needs both classes; preserve patient splits and acquire more development images.")
            scaler, transform = StandardScaler(), clinical_transformer()
            x_train, x_val = scaler.fit_transform(image[train]), scaler.transform(image[val])
            c_train = transform.fit_transform(clinical_inputs(common.loc[train]))
            c_val = transform.transform(clinical_inputs(common.loc[val]))
            _, probability, history, epoch = fit_head(x_train, c_train, common.idh_label.to_numpy()[train],
                x_val, c_val, common.idh_label.to_numpy()[val], mode, seed, max_epochs)
            oof[val] = probability
            epochs.append(epoch)
            all_history += [{"mode": mode, "fold": int(fold), **h} for h in history]
            fold_results.append({"mode": mode, "fold": int(fold), "patients": int(val.sum()), "selected_epoch": epoch,
                                 **binary_metrics(common.idh_label.to_numpy()[val], probability)})
        if not np.isfinite(oof).all():
            raise ValueError("Incomplete out-of-fold predictions.")
        summaries.append({"mode": mode, "patients": len(common), "mean_fold_auc": float(np.mean([r["roc_auc"] for r in fold_results])),
                          "oof_auc": float(roc_auc_score(common.idh_label, oof)), "evaluation": "development_model_selection_only"})
        predictions.append(pd.DataFrame({"patient_id": common.patient_id, "idh_label": common.idh_label,
                                         "fold": common.fold, "mode": mode, "probability_mutant": oof}))
        pd.DataFrame(fold_results).to_csv(output / f"{mode}_folds.csv", index=False)
        # Final refit uses a development-derived epoch budget, never test early stopping.
        scaler, transform = StandardScaler(), clinical_transformer()
        x = scaler.fit_transform(image)
        clinical = transform.fit_transform(clinical_inputs(common))
        budget = max(1, int(np.median(epochs)))
        model, _, _, _ = fit_head(x, clinical, common.idh_label.to_numpy(), x, clinical, common.idh_label.to_numpy(), mode, seed,
                                  fixed_epochs=budget)
        torch.save({"state_dict": model.state_dict(), "mode": mode, "image_dim": 2048, "clinical_dim": clinical.shape[1],
                    "epochs": budget, "feature_provenance": provenance, "fit_scope": "development_only"}, output / f"{mode}.pt")
        joblib.dump({"image_scaler": scaler, "clinical_transformer": transform}, output / f"{mode}_transforms.joblib")
    pd.DataFrame(summaries).to_csv(output / "development_summary.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "development_oof.csv", index=False)
    pd.DataFrame(all_history).to_csv(output / "learning_curves.csv", index=False)
    common[["patient_id", "idh_label", "fold"]].to_csv(output / "common_development_patients.csv", index=False)
    info = {"patients": len(common), "models": summaries, "test_evaluated": False, "seed": seed,
            "max_epochs": max_epochs, "feature_sha256": sha256(features_path),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "split_lock_sha256": sha256(lock_path), "development_csv_sha256": sha256(development_csv),
            "source_sha256": {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
            "torch_version": str(torch.__version__), "device": "cpu", "patient_batch_size": 32,
            "optimizer": "AdamW lr=0.001 weight_decay=0.001", "patience": 8,
            "limitation": "Fold validation selects stopping epochs; these are development estimates, not independent confirmation."}
    write_json(output / "run_manifest.json", info)
    return info


def predict_cached(model_dir: Path, mode: str, image: np.ndarray, patients: pd.DataFrame) -> np.ndarray:
    saved = torch.load(model_dir / f"{mode}.pt", map_location="cpu", weights_only=True)
    transforms = joblib.load(model_dir / f"{mode}_transforms.joblib")
    model = PatientHead(saved["image_dim"], saved["clinical_dim"], saved["mode"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    x = transforms["image_scaler"].transform(image)
    c = transforms["clinical_transformer"].transform(clinical_inputs(patients))
    with torch.inference_mode():
        return torch.sigmoid(model(torch.tensor(x, dtype=torch.float32), torch.tensor(c, dtype=torch.float32))).numpy()
