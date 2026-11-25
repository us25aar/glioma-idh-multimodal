"""Patient-level outer validation with separate stopping and calibration data."""
from __future__ import annotations

import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import clinical_inputs, sha256, write_json
from .evaluation import paired_auc_bootstrap
from .experiment import binary_metrics, make_pipeline
from .multimodal import clinical_transformer, fit_head, load_common_development, predict_cached

MODELS = ('clinical', 'mri', 'fusion', 'age_logistic', 'clinical_logistic')
PROTOCOL = 'outer-fold-inner-stopping-heldout-calibration-v1'


def split_training_roles(labels, seed: int, calibration_fraction: float = .2,
                         stopping_fraction: float = .2) -> dict[str, np.ndarray]:
    """Return positions within an outer training set, without outer labels."""
    labels = np.asarray(labels)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Both classes are required in the outer training set.')
    if not 0 < calibration_fraction < .5 or not 0 < stopping_fraction < .5:
        raise ValueError('Stopping and calibration fractions must be between zero and 0.5.')
    positions = np.arange(len(labels))
    try:
        refit, calibration = train_test_split(positions, test_size=calibration_fraction,
                                              stratify=labels, random_state=seed)
        fit, stopping = train_test_split(refit, test_size=stopping_fraction,
                                         stratify=labels[refit], random_state=seed + 1)
    except ValueError as error:
        raise ValueError('More patients per class are needed for stopping and calibration splits.') from error
    roles = {'fit': np.sort(fit), 'stopping': np.sort(stopping),
             'calibration': np.sort(calibration), 'refit': np.sort(refit)}
    if any(set(np.unique(labels[index])) != {0, 1} for index in roles.values()):
        raise ValueError('Every training role needs both classes. Expand the available cohort.')
    return roles


def probability_logits(probabilities) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError('Finite one-dimensional probabilities are required.')
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Probabilities must be between zero and one.')
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def select_threshold(labels, probabilities) -> float:
    """Maximise calibration-set Youden J, resolving ties towards 0.5."""
    _, _, thresholds = roc_curve(labels, probabilities, drop_intermediate=False)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    candidates = np.append(thresholds[valid], .5)
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    scores = np.array([((probabilities[labels == 1] >= threshold).mean()
                        + (probabilities[labels == 0] < threshold).mean() - 1) for threshold in candidates])
    best = candidates[np.isclose(scores, scores.max())]
    return float(best[np.argmin(abs(best - .5))])


def calibrated_metrics(labels, raw, calibrated, threshold: float) -> dict:
    metrics = {f'raw_{key}': value for key, value in binary_metrics(labels, raw).items()}
    metrics.update({f'calibrated_{key}': value for key, value in binary_metrics(labels, calibrated, threshold).items()})
    tn, fp, fn, tp = confusion_matrix(labels, np.asarray(calibrated) >= threshold, labels=[0, 1]).ravel()
    metrics.update({'raw_brier': float(brier_score_loss(labels, raw)),
                    'calibrated_brier': float(brier_score_loss(labels, calibrated)),
                    'calibrated_log_loss': float(log_loss(labels, calibrated, labels=[0, 1])),
                    'true_negative': int(tn), 'false_positive': int(fp),
                    'false_negative': int(fn), 'true_positive': int(tp)})
    return metrics


def fit_model(mode: str, patients: pd.DataFrame, image: np.ndarray, roles: dict,
              destination: Path, provenance: dict, seed: int, max_epochs: int) -> dict:
    """Fit a base model, then use its untouched calibration holdout."""
    destination.mkdir(parents=True, exist_ok=True)
    labels = patients.idh_label.to_numpy(dtype=int)
    refit, calibration = roles['refit'], roles['calibration']
    selected_epoch, history = None, []
    if mode in {'clinical', 'mri', 'fusion'}:
        fit, stopping = roles['fit'], roles['stopping']
        scaler, transform = StandardScaler(), clinical_transformer()
        train_x = scaler.fit_transform(image[fit])
        train_c = transform.fit_transform(clinical_inputs(patients.iloc[fit]))
        _, _, history, selected_epoch = fit_head(
            train_x, train_c, labels[fit], scaler.transform(image[stopping]),
            transform.transform(clinical_inputs(patients.iloc[stopping])), labels[stopping],
            mode, seed, max_epochs=max_epochs)
        scaler, transform = StandardScaler(), clinical_transformer()
        refit_x = scaler.fit_transform(image[refit])
        refit_c = transform.fit_transform(clinical_inputs(patients.iloc[refit]))
        model, _, _, _ = fit_head(refit_x, refit_c, labels[refit], refit_x, refit_c, labels[refit],
                                  mode, seed, fixed_epochs=selected_epoch)
        torch.save({'state_dict': model.state_dict(), 'mode': mode, 'image_dim': image.shape[1],
                    'clinical_dim': refit_c.shape[1], 'epochs': selected_epoch,
                    'feature_provenance': provenance, 'fit_scope': 'development_only'}, destination / f'{mode}.pt')
        joblib.dump({'image_scaler': scaler, 'clinical_transformer': transform},
                    destination / f'{mode}_transforms.joblib')
        raw_calibration = predict_cached(destination, mode, image[calibration], patients.iloc[calibration])
    else:
        estimator = make_pipeline(mode, patients.iloc[refit].idh_label, seed)
        estimator.fit(clinical_inputs(patients.iloc[refit], mode == 'age_logistic'), labels[refit])
        joblib.dump(estimator, destination / f'{mode}.joblib')
        raw_calibration = estimator.predict_proba(clinical_inputs(patients.iloc[calibration], mode == 'age_logistic'))[:, 1]
    calibrator = LogisticRegression(C=1.0, solver='lbfgs', random_state=seed)
    calibrator.fit(probability_logits(raw_calibration), labels[calibration])
    calibrated = calibrator.predict_proba(probability_logits(raw_calibration))[:, 1]
    threshold = select_threshold(labels[calibration], calibrated)
    base_files = [f'{mode}.pt', f'{mode}_transforms.joblib'] if mode in {'clinical', 'mri', 'fusion'} else [f'{mode}.joblib']
    joblib.dump({'calibrator': calibrator, 'threshold': threshold,
                 'base_sha256': {name: sha256(destination / name) for name in base_files}},
                destination / f'{mode}_calibration.joblib')
    membership = {role: patients.iloc[index].patient_id.tolist() for role, index in roles.items()}
    info = {'mode': mode, 'selected_epoch': selected_epoch, 'threshold': threshold,
            'threshold_method': 'maximum calibration Youden J, ties nearest 0.5',
            'calibration_method': 'regularised sigmoid on heldout raw logits',
            'training_roles': membership, 'protocol': PROTOCOL, 'history': history}
    write_json(destination / f'{mode}_fit.json', info)
    return info


def predict_model(directory: Path, mode: str, image: np.ndarray, patients: pd.DataFrame) -> tuple:
    saved = joblib.load(directory / f'{mode}_calibration.joblib')
    if any(sha256(directory / name) != expected for name, expected in saved['base_sha256'].items()):
        raise ValueError('Base model or preprocessing differs from the fitted calibration artifact.')
    if mode in {'clinical', 'mri', 'fusion'}:
        raw = predict_cached(directory, mode, image, patients)
    else:
        estimator = joblib.load(directory / f'{mode}.joblib')
        raw = estimator.predict_proba(clinical_inputs(patients, mode == 'age_logistic'))[:, 1]
    calibrated = saved['calibrator'].predict_proba(probability_logits(raw))[:, 1]
    return raw, calibrated, saved['threshold']


def run_validation(features_path: Path, feature_manifest: Path, development_csv: Path, lock_path: Path,
                   output: Path, max_epochs: int = 50, seed: int = 42,
                   calibration_fraction: float = .2, stopping_fraction: float = .2) -> dict:
    if max_epochs < 1:
        raise ValueError('Maximum epochs must be positive.')
    started = time.perf_counter()
    provenance = json.loads(feature_manifest.read_text())
    if provenance.get('data_scopes') != ['real_utsw'] or provenance.get('features_sha256') != sha256(features_path):
        raise ValueError('Verified real UTSW feature provenance is required.')
    common, image = load_common_development(features_path, lock_path, development_csv)
    output.mkdir(parents=True, exist_ok=True)
    config = {'protocol': PROTOCOL, 'seed': seed, 'max_epochs': max_epochs,
              'calibration_fraction_of_outer_training': calibration_fraction,
              'stopping_fraction_of_noncalibration_training': stopping_fraction,
              'models': list(MODELS), 'feature_sha256': sha256(features_path),
              'split_lock_sha256': sha256(lock_path), 'patients': common.patient_id.tolist()}
    write_json(output / 'protocol.json', config)
    write_json(output / 'validation_status.json', {'status': 'running', 'protocol': PROTOCOL})
    predictions, fold_rows, history_rows, memberships = [], [], [], []
    for fold in sorted(common.fold.unique()):
        validation_mask = common.fold.to_numpy() == fold
        train = common.loc[~validation_mask].reset_index(drop=True)
        validation = common.loc[validation_mask].reset_index(drop=True)
        if validation.idh_label.nunique() != 2:
            raise ValueError('Both classes are required in every outer evaluation fold.')
        roles = split_training_roles(train.idh_label, seed + int(fold), calibration_fraction, stopping_fraction)
        directory = output / f'fold_{fold}'
        membership = {role: train.iloc[index].patient_id.tolist() for role, index in roles.items()}
        membership['evaluation'] = validation.patient_id.tolist()
        memberships.append({'fold': int(fold), **membership})
        for mode in MODELS:
            info = fit_model(mode, train, image[~validation_mask], roles, directory, provenance, seed, max_epochs)
            raw, calibrated, threshold = predict_model(directory, mode, image[validation_mask], validation)
            fold_rows.append({'mode': mode, 'fold': int(fold), 'patients': len(validation),
                              'selected_epoch': info['selected_epoch'], 'threshold': threshold,
                              **calibrated_metrics(validation.idh_label, raw, calibrated, threshold)})
            predictions.append(pd.DataFrame({'patient_id': validation.patient_id, 'idh_label': validation.idh_label,
                                              'fold': fold, 'mode': mode, 'raw_probability': raw,
                                              'calibrated_probability': calibrated, 'threshold': threshold,
                                              'prediction': (calibrated >= threshold).astype(int)}))
            history_rows.extend([{'mode': mode, 'fold': int(fold), **row} for row in info['history']])
        print('Completed independent outer fold', int(fold), 'patients', len(validation), flush=True)
    oof = pd.concat(predictions, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    oof.to_csv(output / 'validation_oof.csv', index=False)
    folds.to_csv(output / 'validation_folds.csv', index=False)
    pd.DataFrame(history_rows).to_csv(output / 'learning_curves.csv', index=False)
    write_json(output / 'fold_membership.json', memberships)
    common.to_csv(output / 'common_development_patients.csv', index=False)
    summaries = []
    for mode, group in oof.groupby('mode', sort=False):
        metrics = calibrated_metrics(group.idh_label, group.raw_probability, group.calibrated_probability, .5)
        tn, fp, fn, tp = confusion_matrix(group.idh_label, group.prediction, labels=[0, 1]).ravel()
        summaries.append({'mode': mode, 'patients': len(group),
                          'mean_fold_raw_auc': float(folds.loc[folds['mode'].eq(mode), 'raw_roc_auc'].mean()),
                          **metrics, 'selected_threshold_accuracy': float((group.prediction == group.idh_label).mean()),
                          'selected_threshold_precision': float(tp / (tp + fp)) if tp + fp else 0.0,
                          'selected_threshold_sensitivity': float(tp / (tp + fn)),
                          'selected_threshold_specificity': float(tn / (tn + fp)),
                          'selected_threshold_balanced_accuracy': float((tp / (tp + fn) + tn / (tn + fp)) / 2),
                          'selected_threshold_f1': float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
                          'selected_threshold_true_negative': int(tn), 'selected_threshold_false_positive': int(fp),
                          'selected_threshold_false_negative': int(fn), 'selected_threshold_true_positive': int(tp)})
    pd.DataFrame(summaries).to_csv(output / 'validation_summary.csv', index=False)
    paired_frame = oof.pivot(index=['patient_id', 'idh_label'], columns='mode', values='raw_probability').reset_index()
    paired = paired_auc_bootstrap(paired_frame.idh_label, paired_frame.fusion, paired_frame.mri, seed=seed)
    write_json(output / 'paired_fusion_minus_mri.json', paired)
    final_roles = split_training_roles(common.idh_label, seed + 1000, calibration_fraction, stopping_fraction)
    for mode in MODELS:
        fit_model(mode, common, image, final_roles, output / 'final_models', provenance, seed, max_epochs)
        raw, calibrated, _ = predict_model(output / 'final_models', mode, image, common)
        if not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
            raise ValueError('Nonfinite saved-model inference.')
    manifest = {'protocol': config, 'patients': len(common), 'idh_mutant': int(common.idh_label.sum()),
                'reserved_test_evaluated': False, 'elapsed_seconds': round(time.perf_counter() - started, 3),
                'summaries': summaries, 'paired_fusion_minus_mri': paired,
                'source_sha256': {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
                'limitation': 'Internal outer validation conditional on this cohort and fixed protocol. Calibration and threshold choices use separate development holdouts. External transportability remains untested.'}
    write_json(output / 'validation_manifest.json', manifest)
    write_json(output / 'validation_status.json', {'status': 'complete', 'protocol': PROTOCOL,
                                                   'manifest_sha256': sha256(output / 'validation_manifest.json')})
    return manifest
