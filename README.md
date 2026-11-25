# Glioma IDH Multimodal MS Project

Current interim submission: `reports/Glioma IDH Interim Progress Report.docx` (14 November 2026). The public MRI transfer now works. A 20-patient real development pilot has completed preprocessing, frozen ResNet-18 extraction, matched clinical/MRI/fusion training across five folds and three seeds, and raw-image inference verification. All 32 automated checks pass. Detailed evidence is under `outputs/ipr/real_pilot`. The 125 reserved patients remain unevaluated.

Predict IDH-mutant versus IDH-wildtype status from four MRI sequences and preoperative clinical inputs. The eight-week project begins with real UTSW clinical metadata, reproducible cohort construction, fixed patient partitions, and development-only clinical baselines. MRI processing and multimodal training follow after data acquisition and compute decisions.

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

## Sources and attribution

- [UTSW-Glioma collection](https://www.cancerimagingarchive.net/collection/utsw-glioma/), dataset DOI [10.7937/DFAE-1B86](https://doi.org/10.7937/DFAE-1B86), CC BY 4.0. Cite the full dataset attribution in the collection's Data Usage Policy.
- Reddy et al. (2026), [The University of Texas Southwestern Glioma Dataset](https://doi.org/10.1038/s41597-026-07274-4).
- [scikit-learn guidance on preprocessing and leakage](https://scikit-learn.org/stable/common_pitfalls.html).

The original planning document remains at `Model Selection.docx`. The current detailed implementation plan is `Implementation Plan.md`.
