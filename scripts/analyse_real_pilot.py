"""Summarise the real MRI development pilot and verify saved-model inference."""
import json
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from glioma_idh.data import write_json, sha256
from glioma_idh.evaluation import paired_auc_bootstrap
from glioma_idh.experiment import train_baselines, binary_metrics
from glioma_idh.multimodal import load_common_development, predict_cached, run_development


def main():
    output = Path('outputs/ipr/real_pilot')
    features = Path('data/features/real_pilot/features.npz')
    lock_path = Path('outputs/week1/split_lock.json')
    common, image = load_common_development(features, lock_path, Path('outputs/week1/development_patients.csv'))
    clinical_dir = output / 'clinical_controls'
    clinical_summary, _ = train_baselines(common, ['majority', 'age_logistic', 'clinical_logistic'], 42, clinical_dir)
    oof = pd.read_csv(output / 'development_oof.csv')
    summaries = []
    reload_rows = []
    for mode, frame in oof.groupby('mode'):
        folds = pd.read_csv(output / f'{mode}_folds.csv')
        summaries.append({'mode': mode, 'patients': len(frame),
                          **{f'{metric}_mean': float(folds[metric].mean()) for metric in binary_metrics(frame.idh_label, frame.probability_mutant)},
                          'roc_auc_std': float(folds.roc_auc.std(ddof=1)),
                          'brier_oof': float(brier_score_loss(frame.idh_label, frame.probability_mutant))})
        first = predict_cached(output, mode, image, common)
        second = predict_cached(output, mode, image, common)
        np.testing.assert_array_equal(first, second)
        if not np.isfinite(first).all():
            raise ValueError('Nonfinite saved-model predictions.')
        reload_rows.append({'mode': mode, 'patients': len(common), 'repeat_reload_equal': True,
                            'probability_range': [float(first.min()), float(first.max())],
                            'scope': 'development_refit_inference_consistency_only'})
    patient = common.iloc[0]
    manifest = pd.read_csv('data/image_manifest_real_pilot.csv').set_index('patient_id')
    inputs = manifest.loc[patient.patient_id]
    command = [sys.executable, 'scripts/predict_case.py', '--models', str(output),
               '--age', str(patient.age), '--sex', patient.sex]
    for sequence in ['t1', 't1ce', 't2', 'flair', 'mask']:
        command.extend([f'--{sequence}', inputs[f'{sequence}_path']])
    raw = subprocess.run(command, text=True, capture_output=True, check=True)
    raw_prediction = json.loads(raw.stdout)
    cached_probability = float(predict_cached(output, 'fusion', image[:1], common.iloc[:1])[0])
    difference = abs(raw_prediction['probability_idh_mutant_uncalibrated'] - cached_probability)
    if difference > 1e-6:
        raise ValueError('Raw-case CLI and cached feature inference disagree.')
    pd.DataFrame(summaries).to_csv(output / 'pilot_metrics.csv', index=False)
    pivot = oof.pivot(index=['patient_id', 'idh_label'], columns='mode', values='probability_mutant').reset_index()
    paired = paired_auc_bootstrap(pivot.idh_label, pivot.fusion, pivot.mri, samples=2000)
    seed_runs = [{'seed': 42, **row} for row in pd.read_csv(output / 'development_summary.csv').to_dict('records')]
    for seed in [43, 44]:
        repeated = run_development(features, features.with_name('feature_manifest.json'),
                                   Path('outputs/week1/development_patients.csv'), lock_path,
                                   output / f'seed_{seed}', seed=seed)
        seed_runs.extend([{'seed': seed, **row} for row in repeated['models']])
    pd.DataFrame(seed_runs).to_csv(output / 'seed_sensitivity.csv', index=False)
    feature_info = json.loads(features.with_name('feature_manifest.json').read_text())
    prep_info = [json.loads((Path('data/processed/real_pilot') / f'{patient}.json').read_text()) for patient in common.patient_id]
    lock = json.loads(lock_path.read_text())
    if set(common.patient_id) & set(lock['test_ids']):
        raise ValueError('Reserved patient found in pilot analysis.')
    info = {
        'scope': 'real_utsw_development_feasibility_pilot',
        'patients': len(common), 'idh_mutant': int(common.idh_label.sum()),
        'idh_wildtype': int(len(common) - common.idh_label.sum()),
        'selection': 'First two sorted patient IDs per IDH class per original development fold',
        'fold_counts': common.groupby(['fold', 'idh_label']).size().unstack().to_dict(),
        'mask_policy': 'Supplied automatic tumorseg_FeTS.nii.gz for every patient',
        'input_images': 4 * len(common), 'input_masks': len(common),
        'downloaded_directory_bytes': sum(path.stat().st_size for patient in common.patient_id
                                           for path in Path(manifest.loc[patient, 't1_path']).parent.rglob('*')
                                           if path.is_file()),
        'processed_cache_bytes': sum((Path('data/processed/real_pilot') / f'{patient}.npz').stat().st_size
                                     for patient in common.patient_id),
        'preprocessing_passed': len(prep_info),
        'grid_shapes': sorted({str(case['shape']) for case in prep_info}),
        'preprocessing_seconds_sum': round(sum(case['elapsed_seconds'] for case in prep_info), 3),
        'feature_extraction': feature_info,
        'training': json.loads((output / 'run_manifest.json').read_text()),
        'metrics': summaries, 'clinical_controls': clinical_summary.to_dict('records'),
        'paired_fusion_minus_mri': paired, 'saved_model_reload': reload_rows,
        'seed_sensitivity': seed_runs,
        'raw_case_cli_check': {'patient_id': patient.patient_id, 'absolute_probability_difference': difference,
                               'scope': 'development_refit_consistency_only', 'passed': True},
        'test_patients_used': 0,
        'files': {'image_manifest_sha256': sha256(Path('data/image_manifest_real_pilot.csv')),
                  'oof_sha256': sha256(output / 'development_oof.csv')},
        'limitation': 'Small balanced file-order pilot with epoch selection on validation folds. Estimates assess feasibility and remain sensitive to selection and overfitting. The full-cohort comparison and independent test evaluation are pending.'
    }
    write_json(output / 'real_pilot_evidence.json', info)
    print(json.dumps({'patients': len(common), 'metrics': summaries, 'paired': paired, 'reload': reload_rows}, indent=2))


if __name__ == '__main__':
    main()
