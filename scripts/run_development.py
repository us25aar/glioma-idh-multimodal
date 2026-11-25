"""Prepare the full development cohort and run independent outer validation."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from glioma_idh.data import write_json, sha256
from glioma_idh.features import extract_features
from glioma_idh.imaging import prepare_manifest
from glioma_idh.quality import audit_masks
from glioma_idh.utsw_imaging import build_utsw_manifest
from glioma_idh.validation import run_validation
from glioma_idh.validation_plots import make_validation_figures


def write_results_report(summary: pd.DataFrame, status: dict, results: list[dict], output: Path) -> None:
    """Write measured results and their interpretation limits from saved evidence."""
    calibration_percent = 100 * results[0]['protocol']['calibration_fraction_of_outer_training']
    lines = ['# Development validation results', '', '## 1. Executed scope', '',
             f"Acquired file sets: {status['image_file_sets_available']} of {status['image_file_sets_selected']} selected development patients.",
             f"Common image-eligible cohort: {results[0]['patients']} patients, including {results[0]['idh_mutant']} IDH-mutant patients.",
             f"Seeds: {', '.join(map(str, status['seeds']))}. Each uses the original five outer folds and five model comparisons.",
             'The reserved patients have not been evaluated.', '',
             f'Within each outer training set, {calibration_percent:g}% of patients are held out for calibration. The remaining patients are split for fitting and stopping; the selected neural epoch is then used to refit on fitting plus stopping patients. Calibrators and Youden-J thresholds use the separate calibration holdout. The outer evaluation patients enter none of these fitting steps.', '',
             '## 2. Discrimination and calibrated probabilities', '',
             '| Model | Mean pooled raw AUC | SD across seeds | Mean fold raw AUC | Mean calibrated Brier | Mean threshold balanced accuracy |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, frame in summary.groupby('mode'):
        lines.append(f"| {name} | {frame.raw_roc_auc.mean():.3f} | {frame.raw_roc_auc.std(ddof=1):.3f} | {frame.mean_fold_raw_auc.mean():.3f} | {frame.calibrated_brier.mean():.3f} | {frame.selected_threshold_balanced_accuracy.mean():.3f} |")
    lines.extend(['', 'Pooled AUC combines predictions from separately fitted outer-fold models; it can differ from mean fold AUC. The seed SD measures changes in initialization and inner partitions on the same cohort and outer folds. It is not a confidence interval.', '',
                  '## 3. Paired fusion-minus-MRI comparison', '',
                  '| Seed | Raw pooled AUC difference | Conditional paired 95% bootstrap interval |',
                  '| --- | ---: | --- |'])
    for result in results:
        paired = result['paired_fusion_minus_mri']
        lower, upper = paired['ci95']
        lines.append(f"| {result['protocol']['seed']} | {paired['auc_difference']:+.4f} | [{lower:+.4f}, {upper:+.4f}] |")
    lines.extend(['', 'Each interval uses 2,000 stratified paired patient resamples, conditional on the fitted out-of-fold predictions. It does not repeat training or measure external transportability. These internal development results do not establish a fusion benefit when the intervals include zero.', '',
                  '## 4. Technical QC and review limitations', ''])
    for case in status['technical_qc_exclusions']:
        lines.append(f"- Excluded {case['patient_id']} from every model comparison: {case['reason']}")
    quality = status['mask_quality']
    lines.extend(['', f"Mask heuristics flagged {quality['patients_flagged']} of {quality['patients_audited']} retained patients. Flags are review prompts, not clinical failure rates. Cases remain included in this initial fixed-protocol comparison.",
                  'Full-span slice sampling often selects low-area endpoint slices. Small disconnected mask components can enlarge a crop. Sampled engineering review does not replace complete clinical segmentation review. Area-aware sampling and mask-component policy require separately versioned development ablations.', '',
                  '## 5. Reproducibility evidence', '',
                  '- `pipeline_status.json`: acquisition, cohort and exclusion status.',
                  '- `inputs/`: exact features, provenance, development table, configuration and original split lock.',
                  '- `seed_*/`: fold membership, model-selection histories, saved models/calibrators, out-of-fold predictions, metrics, paired analysis and diagnostic figures.',
                  '- `mask_quality_audit.csv`: per-patient review flags; `selected_view_review.json` records the additional sampled review when present.',
                  '- Run `scripts/verify_development.py` for checksum, split, metric and saved-model reproduction checks.',
                  '- Run `scripts/analyse_development.py` for descriptive demographic subgroup and error records.', '',
                  'Reserved evaluation, broader visual review and independent external validation remain pending.'])
    (output / 'executed_results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/development_validation.json'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--available-only', action='store_true')
    mode.add_argument('--watch-acquisition', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    session = json.loads(Path('outputs/ipr/download_session.json').read_text())
    root = Path(session.get('destination', 'dataset'))
    output = Path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    manifest = Path(config['image_manifest'])
    while True:
        inventory = build_utsw_manifest(root, Path(config['selection_csv']), manifest, Path(config['split_lock']))
        complete = inventory['complete_file_sets'] == inventory['selected_patients']
        write_json(output / 'pipeline_status.json', {'stage': 'preparing_available_images', **inventory})
        print(json.dumps(inventory), flush=True)
        if inventory['complete_file_sets'] == 0:
            raise ValueError('No complete image file sets are available.')
        if not complete and not (args.available_only or args.watch_acquisition):
            raise ValueError('MRI acquisition is incomplete. Use --watch-acquisition or --available-only.')
        qc = prepare_manifest(manifest, Path(config['processed_dir']))
        print('Technical QC', qc.status.value_counts().to_dict(), 'reused', int(qc.cache_reused.sum()), flush=True)
        if complete or args.available_only:
            break
        time.sleep(10)
    feature_dir = Path(config['features_dir'])
    feature_info = extract_features(Path(config['processed_dir']) / 'image_qc.csv', feature_dir,
                                    Path(config['weights_dir']))
    if args.available_only and not complete:
        output = output / f"available_{feature_info['patients']}"
    inputs = output / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    for source in [feature_dir / 'features.npz', feature_dir / 'feature_manifest.json',
                   Path(config['development_csv']), Path(config['split_lock']), args.config]:
        shutil.copy2(source, inputs / source.name)
    records = []
    results = []
    quality = audit_masks(manifest, Path(config['processed_dir']), output)
    print(json.dumps(quality), flush=True)
    for seed in config['seeds']:
        seed_dir = output / f'seed_{seed}'
        result = run_validation(inputs / 'features.npz', inputs / 'feature_manifest.json',
                                inputs / Path(config['development_csv']).name,
                                inputs / Path(config['split_lock']).name, seed_dir,
                                max_epochs=config['max_epochs'], seed=seed,
                                calibration_fraction=config['calibration_fraction'],
                                stopping_fraction=config['stopping_fraction'])
        make_validation_figures(seed_dir)
        results.append(result)
        records.extend([{'seed': seed, **row} for row in result['summaries']])
        print('Completed seed', seed, 'patients', result['patients'], flush=True)
    summary = pd.DataFrame(records)
    summary.to_csv(output / 'seed_comparison.csv', index=False)
    status = {'stage': 'completed', 'image_file_sets_selected': inventory['selected_patients'],
              'image_file_sets_available': inventory['complete_file_sets'],
              'patients_in_experiment': results[0]['patients'],
              'technical_qc_exclusions': qc.loc[qc.status.ne('passed'), ['patient_id', 'reason']].to_dict('records'),
              'acquisition_complete': complete, 'reserved_test_evaluated': False,
              'mask_quality': quality,
              'seeds': config['seeds'], 'protocol_sha256': sha256(args.config),
              'output_dir': str(output)}
    write_json(output / 'pipeline_status.json', status)
    write_results_report(summary, status, results, output)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
