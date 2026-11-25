"""Audit completed development outputs without opening reserved patient data."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score

from glioma_idh.data import sha256, write_json
from glioma_idh.multimodal import load_common_development
from glioma_idh.validation import MODELS, predict_model


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_roles(roles, training_ids, evaluation_ids):
    groups = {name: set(roles[name]) for name in ('fit', 'stopping', 'calibration', 'refit')}
    for name, group in groups.items():
        require(len(roles[name]) == len(group), f'Duplicate {name} membership.')
        require(group.isdisjoint(evaluation_ids), f'Evaluation patient entered {name}.')
    fit, stop, calibration = [groups[name] for name in ('fit', 'stopping', 'calibration')]
    require(fit.isdisjoint(stop) and fit.isdisjoint(calibration) and stop.isdisjoint(calibration),
            'Training roles overlap.')
    require(fit | stop == groups['refit'], 'Refit membership differs from fit plus stopping.')
    require(fit | stop | calibration == training_ids, 'Training membership is incomplete.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/development_validation.json'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    output = Path(config['output_dir'])
    status = json.loads((output / 'pipeline_status.json').read_text())
    require(status['stage'] == 'completed', 'The development pipeline has not completed.')
    require(status['acquisition_complete'] and not status['reserved_test_evaluated'],
            'Unexpected acquisition or evaluation scope.')
    require(status['protocol_sha256'] == sha256(args.config), 'Configuration changed after execution.')
    inputs = output / 'inputs'
    lock_path = inputs / Path(config['split_lock']).name
    lock = json.loads(lock_path.read_text())
    require(sha256(lock_path) == sha256(Path(config['split_lock'])), 'Original partition lock changed.')
    development_ids, reserved_ids = set(lock['development_ids']), set(lock['test_ids'])
    require(development_ids.isdisjoint(reserved_ids), 'Locked partitions overlap.')
    selection = pd.read_csv(config['selection_csv'])
    require(set(selection.patient_id) == development_ids and not selection.patient_id.duplicated().any(),
            'Acquisition selection differs from the original development set.')
    features = inputs / 'features.npz'
    provenance = json.loads((inputs / 'feature_manifest.json').read_text())
    require(provenance['features_sha256'] == sha256(features) and provenance['data_scopes'] == ['real_utsw'],
            'Feature provenance failed verification.')
    common, image = load_common_development(features, lock_path,
                                            inputs / Path(config['development_csv']).name)
    patient_ids = set(common.patient_id)
    qc = pd.read_csv(Path(config['processed_dir']) / 'image_qc.csv')
    require(set(qc.patient_id) == development_ids and not qc.patient_id.duplicated().any(),
            'Technical QC inventory differs from the selected cohort.')
    require(set(qc.loc[qc.status.eq('passed'), 'patient_id']) == patient_ids,
            'Feature cohort differs from technical QC eligibility.')
    require(len(common) == status['patients_in_experiment'] and patient_ids.isdisjoint(reserved_ids),
            'Unexpected experimental cohort.')
    records = []
    for seed in config['seeds']:
        directory = output / f'seed_{seed}'
        completed = json.loads((directory / 'validation_status.json').read_text())
        require(completed['status'] == 'complete', f'Seed {seed} is incomplete.')
        require(completed['manifest_sha256'] == sha256(directory / 'validation_manifest.json'),
                'Validation manifest checksum changed.')
        predictions = pd.read_csv(directory / 'validation_oof.csv')
        summary = pd.read_csv(directory / 'validation_summary.csv').set_index('mode')
        require(set(predictions['mode']) == set(MODELS), 'Unexpected model comparison.')
        require(not predictions.duplicated(['patient_id', 'mode']).any(), 'Duplicate OOF prediction.')
        joined = predictions.merge(common[['patient_id', 'idh_label', 'fold']], on='patient_id',
                                   how='left', validate='many_to_one', suffixes=('', '_locked'))
        require(joined.idh_label.eq(joined.idh_label_locked).all()
                and joined.fold.eq(joined.fold_locked).all(), 'OOF labels or folds differ from the lock.')
        for mode, frame in predictions.groupby('mode'):
            require(set(frame.patient_id) == patient_ids, 'OOF cohort differs between models.')
            probabilities = frame[['raw_probability', 'calibrated_probability', 'threshold']].to_numpy()
            require(np.isfinite(probabilities).all() and ((0 <= probabilities) & (probabilities <= 1)).all(),
                    'Invalid probabilities or thresholds.')
            require(np.array_equal(frame.prediction, (frame.calibrated_probability >= frame.threshold).astype(int)),
                    'Saved classifications differ from saved thresholds.')
            require(np.isclose(roc_auc_score(frame.idh_label, frame.raw_probability), summary.loc[mode, 'raw_roc_auc'])
                    and np.isclose(brier_score_loss(frame.idh_label, frame.calibrated_probability),
                                   summary.loc[mode, 'calibrated_brier']), 'Summary metrics differ from OOF predictions.')
        memberships = json.loads((directory / 'fold_membership.json').read_text())
        require({row['fold'] for row in memberships} == set(common.fold.unique()), 'Missing outer-fold membership.')
        for membership in memberships:
            evaluation_mask = common.fold.eq(membership['fold']).to_numpy()
            evaluation_frame = common.loc[evaluation_mask]
            evaluation = set(evaluation_frame.patient_id)
            require(set(membership['evaluation']) == evaluation, 'Outer-fold membership changed.')
            check_roles(membership, patient_ids - evaluation, evaluation)
            for mode in MODELS:
                fitted = json.loads((directory / f"fold_{membership['fold']}" / f'{mode}_fit.json').read_text())
                require(all(fitted['training_roles'][role] == membership[role]
                            for role in ('fit', 'stopping', 'calibration', 'refit')), 'Model training roles differ.')
                raw, calibrated, threshold = predict_model(directory / f"fold_{membership['fold']}", mode,
                                                             image[evaluation_mask], evaluation_frame)
                saved = predictions.loc[predictions['mode'].eq(mode)].set_index('patient_id').loc[evaluation_frame.patient_id]
                require(np.allclose(raw, saved.raw_probability, rtol=0, atol=1e-6)
                        and np.allclose(calibrated, saved.calibrated_probability, rtol=0, atol=1e-6)
                        and np.allclose(threshold, saved.threshold, rtol=0, atol=1e-12),
                        'Saved fold-model inference differs from the OOF predictions.')
        for mode in MODELS:
            fitted = json.loads((directory / 'final_models' / f'{mode}_fit.json').read_text())
            check_roles(fitted['training_roles'], patient_ids, reserved_ids)
            raw, calibrated, _ = predict_model(directory / 'final_models', mode, image[:1], common.iloc[:1])
            require(np.isfinite(raw).all() and np.isfinite(calibrated).all(), 'Saved-model inference failed.')
        records.append({'seed': seed, 'outer_folds': len(memberships), 'models': len(MODELS),
                        'oof_rows': len(predictions), 'patients': len(common)})
    evidence = {'passed': True, 'patients': len(common), 'idh_mutant': int(common.idh_label.sum()),
                'reserved_patients': len(reserved_ids), 'reserved_test_evaluated': False,
                'feature_sha256': sha256(features), 'split_lock_sha256': sha256(lock_path),
                'seeds': records, 'checks': ['cohort and feature provenance', 'unchanged patient folds',
                'disjoint stopping/calibration/evaluation membership', 'one OOF prediction per patient and model',
                'probability and threshold consistency', 'OOF metric recomputation',
                'saved-fold inference reproduces OOF predictions',
                'final-model calibration checksums and inference']}
    write_json(output / 'verification.json', evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
