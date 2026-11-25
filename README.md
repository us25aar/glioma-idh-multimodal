# Glioma IDH Multimodal MS Project

Latest executed milestone: all 497 development MRI file sets have been acquired. One empty supplied tumour mask was excluded consistently, leaving 496 patients for frozen ResNet-18 extraction and five-model validation across five outer folds and three seeds. All 46 automated checks pass, all 7,440 out-of-fold predictions were reproduced from saved fold models, and raw-MRI inference matches cached inference exactly. Evidence is under `outputs/development/validation`; the 125 reserved patients remain unevaluated. Earlier IPR documents and pilot evidence remain local historical records.

Predict IDH-mutant versus IDH-wildtype status from four MRI sequences and preoperative age and sex. The implementation includes real UTSW acquisition, reproducible cohort construction, fixed patient partitions, tumour-focused MRI processing, frozen feature extraction, clinical/MRI/fusion models, independent outer-fold scoring, calibration and saved-model inference. The current experiments run on the local CPU; cloud access remains unconfirmed.

## Current development results

The common cohort contains 141 IDH-mutant and 355 wildtype patients. The table reports mean raw pooled out-of-fold ROC-AUC across seeds 42, 43 and 44, using the same original outer folds and separate inner stopping/calibration data.

| Model | Mean pooled raw ROC-AUC | SD across seeds | Mean fold raw ROC-AUC |
| --- | ---: | ---: | ---: |
| Age-only logistic regression | 0.868 | 0.002 | 0.871 |
| Age/sex logistic regression | 0.865 | 0.003 | 0.867 |
| Clinical MLP | 0.835 | 0.014 | 0.866 |
| MRI-only neural head | 0.806 | 0.017 | 0.814 |
| MRI/clinical fusion | 0.816 | 0.013 | 0.823 |

Fusion-minus-MRI pooled AUC differences are +0.0118, +0.0124 and +0.0032. Their conditional paired 95% bootstrap intervals all include zero. The current results do not establish a fusion benefit, and the simple age-only comparator is stronger. Pooled AUC combines predictions from separate fold models and can differ from mean fold AUC. Seed SD is a sensitivity measure on one cohort, not a confidence interval or external validation.

Technical QC passed for 496 patients. Heuristic mask review flagged 426 retained cases: 425 have a selected slice with less than 1% of the peak mask area, 18 have crop enlargement from small disconnected components, and 17 have both flags. Full-span sampling commonly selects low-area endpoint slices; these flags are not clinical failure rates. The current run retains its fixed preprocessing. Area-aware sampling, mask-component sensitivity analysis and broader visual review remain development tasks.

`outputs/development/validation/executed_results.md` contains the measured comparison and limitations. `verification.json` records the output audit, `raw_case_inference_check.json` records exact raw/cached inference agreement and rejection of incompatible preprocessing, and `descriptive_subgroups.csv`/`development_errors.csv` support further investigation. Subgroups include only 13 mutant patients aged 60 or older, so subgroup comparisons are descriptive. The historical 20-patient pilot below uses a different stopping protocol and is not a direct performance benchmark for this experiment.

## Repository setup

The repository contains the Python implementation, experiment configurations, tests and dependency lists. MRI data, patient manifests, trained weights, generated results and assessment documents are local files. Commands below recreate the experiment outputs after the required data have been acquired.

For a fresh checkout on Windows, install the clinical and imaging dependencies before running the complete test suite.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-imaging.txt
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-xgboost.txt
.\.venv\Scripts\python.exe scripts/run_checks.py
```

The earlier progress reports and meeting notes referenced below are local project records. The clinical and real MRI sections include the commands needed to reproduce their experimental results.

## Show progress at the meeting

1. Open `reports/Week 1 Progress Report.pdf` for the progress summary and results.
2. Open `outputs/week1/executed_results.md` for the machine-generated clinical experiment summary.
3. Show `outputs/week1/figures/clinical_development_roc.svg` and `development_cohort.svg` in a browser or the report's ROC figure.
4. Show `configs/utsw_week1.json`, `glioma_idh/`, and `tests/test_week1.py` for the runnable implementation.
5. Use `reports/Professor Questions.md` to record meeting decisions.

Week-one scores are preliminary development cross-validation estimates from real clinical metadata. They are not final test scores, MRI results, or evidence that multimodal fusion improves prediction. Synthetic records appear only in software tests.

## Run on Windows

From the project root, create an isolated Python 3.11 environment and install the fixed CPU dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The existing global Python installation has incompatible binary packages on this machine; use `.venv` for this project. The setup does not modify those global packages. The lock file records every installed dependency, while requirements.txt lists the direct CPU dependencies. GPU packages are deliberately deferred until the professor confirms compute access.

Run the actual week-one workflow:

```powershell
.\.venv\Scripts\python.exe -m glioma_idh download
.\.venv\Scripts\python.exe scripts/run_checks.py
.\.venv\Scripts\python.exe -m glioma_idh week1 --config configs/utsw_week1.json
```

The download command fetches only the small public clinical TSV. It preserves an existing file and records its SHA256 and provenance. It does not download MRI volumes. Alternatively, use `scripts/run_week1.ps1`; pass `-InstallDependencies` on a fresh setup.

Inspect the active environment:

```powershell
.\.venv\Scripts\python.exe -m glioma_idh doctor
```

The historical week-one configuration runs four clinical controls. XGBoost is now installed and included in configs/utsw_ipr.json, which runs all five baselines. Hyperparameters are fixed for the initial comparison.

## Input and output contracts

- Source: `data/raw/utsw/UTSW_Glioma_Metadata-2-1.tsv`, preserved unchanged.
- Mapping: source `mutated` becomes 1; `wild type` becomes 0; unknown labels are excluded. Unrecognized encodings fail rather than being guessed.
- Features: only `age` and encoded `sex`. The feature matrix cannot include identifiers, IDH results, diagnosis, grade, pathology, treatment, scanner information, or segmentation-review status.
- Eligibility: a known label, preoperative status, and all four sequences declared available in the metadata. MRI existence, alignment and quality remain unverified until images are acquired.
- Partitions: one seed-42 stratified 20 percent holdout; five stratified folds within development. Existing split locks cannot be silently replaced.
- Predictions: one out-of-fold probability per development patient per model. No test predictions are produced by this runner.
- Threshold: 0.5 for initial threshold-dependent metrics. A final operating threshold and probability calibration remain future development decisions.

Important outputs under `outputs/week1/`:

| File | Purpose |
| --- | --- |
| `metadata_audit.json` | Observed cohort counts and data-quality limits |
| `patient_manifest.csv` | Auditable eligibility and exclusions |
| `split_lock.json` | Exact patient and fold assignments with source hash |
| `development_patients.csv` | Development cohort and fold assignments |
| `locked_test_ids.csv` | Reserved test IDs, without a prediction file |
| `clinical_cv_folds.csv` | Per-fold clinical scores |
| `clinical_cv_summary.csv` | Fold means and standard deviations |
| `clinical_oof_predictions.csv` | Development-only out-of-fold probabilities |
| `models/` | Clinical pipelines refitted on development patients only |
| `run_manifest.json` | Configuration, source/code hashes, software versions, execution scope |
| `executed_results.md` | Human-readable executed results |

After MRI quality control, retain original patient assignments and use the common verified cohort for every primary comparison. Metadata-only clinical results are provisional until that common cohort is finalized. Refitting clinical baselines on a revised image-eligible cohort must be documented.

## Eight week milestones

| Week | Outcome |
| --- | --- |
| 1 | Metadata acquisition and audit, fixed patient partitions, clinical baselines, tests, protocol and meeting report |
| 2 | Confirm cloud access; acquire MRI; inventory masks; verify alignment; complete a small development preprocessing pilot |
| 3 | Extract frozen MRI features for the eligible cohort; develop the MRI-only head |
| 4 | Implement clinical MLP and intermediate fusion; compare against matching MRI-only features |
| 5 | Complete selected ablations; perform limited fine tuning only if justified and feasible; freeze model choices |
| 6 | Refit models; select thresholds on development data; run the locked internal evaluation and paired uncertainty analysis |
| 7 | Complete error analysis, interpretation and calibration; attempt independent external validation if data are ready |
| 8 | Finish methods, results, limitations, reproducibility package and defense demonstration |

Write the thesis methods and literature review from week one. External testing is conditional; preserve the three core model comparisons if resources or time are constrained.

## Interim implementation added November 14 2026

The local prototype now includes registered NIfTI grid checks, deterministic tumour-focused MRI preprocessing, QC figures, official frozen ResNet-18 feature extraction and caching, MRI/clinical/fusion neural heads, a development-only training driver, saved-model inference and patient-paired bootstrap analysis. See `scripts/imaging_workflow.py` and `scripts/predict_case.py` for executable commands. Real clinical experiments are under `outputs/ipr/clinical`; synthetic imaging integration evidence is under `outputs/ipr/synthetic_integration` and must never be presented as patient performance.

Install CPU PyTorch and imaging dependencies in the existing project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-imaging.txt
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-xgboost.txt
.\.venv\Scripts\python.exe scripts/run_checks.py
.\.venv\Scripts\python.exe -m glioma_idh week1 --config configs/utsw_ipr.json
.\.venv\Scripts\python.exe scripts/ipr_analysis.py
```

The imaging manifest must contain one row per patient and columns `patient_id,t1_path,t1ce_path,t2_path,flair_path,mask_path`. Paths may be absolute or relative to the CSV. Select one consistent registered/skull-stripped pipeline and supplied mask type across patients; ambiguous files must be resolved manually. The workflow rejects grid differences and does not pretend that resampling performs registration.

```powershell
.\.venv\Scripts\python.exe scripts/imaging_workflow.py prepare data/image_manifest.csv data/processed/utsw
.\.venv\Scripts\python.exe scripts/imaging_workflow.py features data/processed/utsw/image_qc.csv data/features/utsw
.\.venv\Scripts\python.exe scripts/imaging_workflow.py train data/features/utsw outputs/imaging_development
```

Initial extraction downloads the official 44.7 MB ImageNet checkpoint into outputs/weights. Image models always reuse the original patient/fold lock. Training requires real-data provenance, at least ten development cases and both classes in retained training and validation folds. All three neural heads use the same common patients. Early-stopping results are development model-selection estimates. Full-cohort MRI acquisition and independent evaluation remain pending; no MRI research score is claimed from the synthetic integration run.

```powershell
.\.venv\Scripts\python.exe scripts/run_imaging_smoke.py
```

This separately marked software test creates synthetic NIfTIs, executes preprocessing and pretrained extraction, and trains each head for at most three epochs. It does not measure medical prediction performance. Additional tests reject empty/fractional masks, nonfinite MRI, mismatched affines, duplicate cases and synthetic provenance in research training.

## Real MRI pilot completed November 14 2026

The pilot contains 20 original development patients, selected as the first two sorted IDs per IDH class per original fold. It has 10 mutants and 10 wildtypes. Each four-patient validation fold therefore contains both classes. This is a file-order feasibility sample, with a different class prevalence from the complete cohort.

Aspera downloaded patient directories into `dataset/`, which was selected through its folder chooser. The common image variant uses `brain_t1.nii.gz`, `brain_t1ce.nii.gz`, `brain_t2.nii.gz`, `brain_flair.nii.gz` and the automatic `tumorseg_FeTS.nii.gz` mask for every patient. The cropped `tumorseg_manual_correction.nii.gz` has a different grid in the inspected case. The explicit manifest builder keeps mask selection consistent.

With the official Aspera Connect client running, resume the same public acquisition using the saved application session. A first-time destination choice may require `--select-folder` and user handling of host warnings.

```powershell
.\.venv\Scripts\python.exe scripts/download_via_connect.py --patients-per-class-fold 2
.\.venv\Scripts\python.exe scripts/run_real_pilot.py
```

The second command inventories all 20 selected file sets, preprocesses the real volumes, extracts features, trains three heads at seed 42, reruns age-based controls on the same patients, repeats seeds 43 and 44, and checks raw-image inference against cached-feature inference. It saves per-case input hashes and model provenance. The separate sampled-overlay review is recorded in `outputs/ipr/real_pilot/visual_qc_review.json`.

| Model at seed 42 | Mean fold AUC | Pooled OOF AUC | Mean accuracy | Mean F1 |
| --- | --- | --- | --- | --- |
| Clinical MLP | 0.700 | 0.530 | 0.500 | 0.000 |
| MRI only | 0.700 | 0.690 | 0.600 | 0.567 |
| MRI and clinical fusion | 0.700 | 0.670 | 0.750 | 0.793 |

Same-patient age-and-sex logistic mean AUC is 0.800, and age-only logistic is 0.775. Fusion minus MRI pooled AUC is -0.020, with conditional paired bootstrap interval [-0.190, 0.140]. Stopping epochs were selected on the validation folds, so these are development estimates. The balanced sample, very small folds and seed variation limit scientific conclusions.

All 20 finite/grid checks passed and sampled overlays were inspected. BT0018 has lower T2 detail, BT0027 has disconnected mask regions, and BT0062 has remote small mask components that enlarge the crop. They remain in the feasibility pilot with review flags. Broader QC and the final mask policy should be resolved before expanding the primary experiment.

The report includes the measured runtime and results. Main outputs include `pilot_metrics.csv`, `development_oof.csv`, `seed_sensitivity.csv`, `real_pilot_evidence.json`, model checkpoints and learning curves under `outputs/ipr/real_pilot`. Full-cohort acquisition, protocol freeze, calibration and reserved evaluation remain pending.

## Expanded development workflow

The expanded workflow uses every patient in the original 497-patient development partition. Its acquisition script requests the four MRI sequences and the automatic FeTS mask, reuses completed inputs, and records resumable transfer state. It saves a separate selection file and preserves the historical pilot records.

With the existing Aspera Connect client running, select the destination for a new client session and acquire the development inputs.

```powershell
.\.venv\Scripts\python.exe scripts/download_development.py --select-folder
.\.venv\Scripts\python.exe scripts/run_development.py
```

Resume acquisition with `scripts/download_development.py`. When acquisition is already running in another terminal, `scripts/run_development.py --watch-acquisition` preprocesses complete cases while the download continues. `--available-only` explicitly runs an interim experiment on the available subset and stores it under a separate output directory.

The settings are in `configs/development_validation.json`. Preprocessing and feature caches verify their input and output checksums before reuse. The optimized normalization applies the original whole-volume percentile bounds to the selected views. Its output matches the original real-patient tensor exactly in the regression check.

Each of the five original outer folds is held out from model fitting, stopping, calibration and threshold selection. Within its training patients, 20 percent form a calibration holdout. The remaining patients are split into fitting and stopping subsets, using 20 percent for stopping. After selecting a neural stopping epoch, the model is refitted on fitting plus stopping patients. A regularised sigmoid is fitted to the untouched calibration predictions, and the calibration-set Youden J determines the operating threshold. The outer fold then provides raw discrimination, calibrated probability metrics and threshold-dependent scores.

The comparison includes clinical MLP, MRI-only, MRI/clinical fusion, age-only logistic regression and age/sex logistic regression. All models use the same eligible patients and calibration holdouts. Seeds 42, 43 and 44 are recorded independently. The original pilot runner retains its earlier validation procedure for reproducing the historical IPR results.

Outputs under `outputs/development/validation` include the exact feature snapshot, outer-fold membership, model histories, raw and calibrated probabilities, ROC and calibration curves, confusion matrices, conditional paired bootstrap analysis and three-seed summaries. Columns prefixed `calibrated_` report probability metrics and classification at 0.5. Columns prefixed `selected_threshold_` report classification using each fold's calibration-selected threshold. Final candidate models retain an independent development calibration holdout. Their calibration artifacts record base-model and preprocessing checksums, and the raw-case inference command reports calibrated outputs when those artifacts are available.

Mask auditing records disconnected components, crop enlargement and selected slices with very small mask area. These are review flags, with cases retained for the initial comparison. The audit and sampled overlays support subsequent visual review and a documented mask-policy sensitivity analysis.

After validation, run the output audit and descriptive error analysis:

```powershell
.\.venv\Scripts\python.exe scripts/verify_development.py
.\.venv\Scripts\python.exe scripts/analyse_development.py
```

The audit verifies feature provenance, unchanged patient folds, disjoint training roles, complete out-of-fold predictions, metric recomputation, reproduction of predictions from all saved fold models, and final-model calibration checksums. It writes `verification.json`. The analysis saves descriptive sex and age subgroup counts, performance measures and error records. These analyses support error investigation and require cautious interpretation.

Use `scripts/predict_case.py --help` for raw-MRI inference arguments and supply `--models outputs/development/validation/seed_42/final_models` to use a calibrated development candidate. Match the four MRI sequences, supplied mask, age and sex for the same case. The command rejects a preprocessing version, sequence order or encoder checkpoint that differs from training provenance.

The method follows the [scikit-learn guidance on nested validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) and [disjoint calibration data](https://scikit-learn.org/stable/modules/calibration.html). The implementation uses the project's pinned scikit-learn version. The reserved 125-patient cohort is retained for the final evaluation stage.

## Sources and attribution

- [UTSW-Glioma collection](https://www.cancerimagingarchive.net/collection/utsw-glioma/), dataset DOI [10.7937/DFAE-1B86](https://doi.org/10.7937/DFAE-1B86), CC BY 4.0. Cite the full dataset attribution in the collection's Data Usage Policy.
- Reddy et al. (2026), [The University of Texas Southwestern Glioma Dataset](https://doi.org/10.1038/s41597-026-07274-4).
- [scikit-learn guidance on preprocessing and leakage](https://scikit-learn.org/stable/common_pitfalls.html).

The original planning document remains at `Model Selection.docx`. The current detailed implementation plan is `Implementation Plan.md`.
