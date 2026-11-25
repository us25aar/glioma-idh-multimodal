# Implementation Plan for Multimodal Glioma IDH Classification

Status at 14 November 2026: the local pipeline has passed 32 automated checks and completed a 20-patient real MRI pilot. All 80 MRI volumes and 20 FeTS masks passed finite/grid checks. Frozen extraction produced 20 by 2,048 features, the three modality heads ran on five original folds at seeds 42, 43 and 44, and raw-image inference matched cached prediction exactly. Full clinical baselines including XGBoost remain available on 497 development patients. Seed-42 mean pilot AUC is 0.700 for MRI and fusion, with pooled AUCs 0.690 and 0.670. The reserved 125-patient cohort remains unevaluated. Broader QC, cohort expansion and final evaluation are pending. The current report is reports/Glioma IDH Interim Progress Report.docx. The design rationale below is retained, with current commands in README.md.

Build a patient-level research pipeline that predicts IDH mutation status from preoperative T1, T1 contrast enhanced, T2, and FLAIR MRI together with age and sex. Use UTSW-Glioma as the primary dataset. For the available Intel Core i7 10th generation laptop and NVIDIA GeForce MX230 with 4 GB of dedicated memory, start by extracting and caching frozen ResNet-18 features, then train small MRI-only and multimodal classification heads. The main experiment tests whether clinical information improves prediction over the same MRI-only representation. Encoder fine tuning is a subsequent extension after measuring local memory and runtime.

The plan follows the project described in [Model Selection.docx](Model%20Selection.docx) and the subsequent dataset and hardware details. The [UTSW-Glioma dataset paper](https://www.nature.com/articles/s41597-026-07274-4) reports 625 patients, with known IDH status in 622. Establish the final eligible count by matching labels, clinical records, and usable MRI. The confirmed project duration is eight weeks, with the current work in week one. Cloud resources will be discussed with the professor on October 29, 2026. Model dimensions and training settings below are starting choices to validate during development.

## Dataset access and first actions

The [TCIA collection](https://www.cancerimagingarchive.net/collection/utsw-glioma/) lists Version 1 dated March 18, 2026, approximately 22.9 GB of NIfTI images and segmentations, and a separate approximately 96.3 KB clinical TSV file. The collection is licensed CC BY 4.0 and requires dataset attribution. Its associated [paper](https://www.nature.com/articles/s41597-026-07274-4) describes four MRI sequences, clinical predictors, FeTS segmentations, and 362 manually refined masks. Do not assume all masks have the same review status.

1. Obtain the small clinical metadata file first. It is tab separated; load it with an explicit tab delimiter, for example pandas.read_csv(path, sep="\t"). Review the actual column names and label encodings before creating a model table.
2. Create candidate patient IDs using known IDH labels. Exclude the three cases without IDH labels from supervised classification; do not relabel them as wildtype.
3. Check local system RAM and free disk space before downloading images. Provisionally allow 80 to 120 GB free for the archive, extraction, working files, and selected caches; this is a workspace allowance, not a verified expanded dataset size. Measure actual extraction and cache sizes during the pilot.
4. Follow TCIA's current download instructions; the imaging listing indicates IBM Aspera Connect. If case selection is supported, obtain a small development pilot first. If distribution is a single archive, download it when storage is ready and initially process only the pilot cases.
5. Freeze the candidate patient partitions from metadata before selecting a 20 to 30 patient pilot from the development partition, with both classes represented. Use this pilot to verify file matching, mask labels, crop quality, feature extraction, and runtime.
6. Expand to all eligible patients after the pilot. Use the small subset for debugging, not as the final research cohort or a way to select favorable results.

The public clinical metadata and a 20-patient development MRI pilot have now been acquired and locally audited. The remainder of the image cohort awaits acquisition. See README.md and outputs/week1/ for runnable code, the cohort audit, patient partitions, and executed development-only clinical experiments.

## 1 Research question and scope

**Primary question:** Does adding preoperative age and sex to a lightweight MRI encoder improve patient-level IDH mutation classification?

**Primary comparison:** A clinical MLP fused with frozen MRI features versus an MRI-only head on the same frozen features, using identical patients, preprocessing, partitions, and comparable head-training budgets. If fine tuning is later performed, repeat the paired MRI-only and multimodal comparison within that training regime.

**Primary outcome:** Difference in held-out patient-level ROC-AUC, with a paired confidence interval. Treat IDH-mutant as the positive class throughout.

The core scope includes acquisition, cohort construction, clinical baselines, MRI feature extraction, MRI-only and fusion heads, patient-level evaluation, interpretation, and reproducible inference. The CNN remains frozen in the initial implementation; only the classification and clinical branches learn from this cohort. This is transfer learning with feature fusion, and should be described explicitly rather than called end-to-end MRI training. A frozen natural-image encoder may provide insufficient task-specific features; its suitability is an empirical question. Fine tuning is the first modeling extension if necessary and feasible. Late fusion and independent external testing follow as resources permit. A transformer, a full 3D classifier, a newly trained segmentation model, and a production application are beyond the core scope.

MRI and clinical fusion already appear in the literature, including [GlioMT](https://www.nature.com/articles/s41746-025-01530-4). Frame the contribution around a computationally manageable implementation, controlled comparison of the added value of clinical data, and rigorous validation. Confirm the thesis contribution with the supervisor during protocol preparation. Do not promise an improvement or a particular accuracy; a well-supported finding of no improvement remains a valid result to discuss.

## 2 Cohort construction and data audit

Create a patient manifest before training. Keep molecular labels separate from the clinical input feature table.

| Manifest field | Purpose |
| --- | --- |
| patient_id | Stable identifier for the person |
| exam_id | Identifier for the selected preoperative examination |
| source_collection and site_id | Provenance and hospital or acquisition site, when known |
| t1_path, t1ce_path, t2_path, flair_path | Matched imaging files for the same examination |
| tumor_mask_path | Tumor localization, if available |
| age and sex | Initial allowed clinical predictors |
| idh_label | 1 for mutant, 0 for wildtype, missing for unknown |
| qc_status and exclusion_reason | Auditable eligibility decision |

Tasks:

1. Record UTSW-Glioma as the source, its downloaded release version, dataset DOI 10.7937/DFAE-1B86, license, access conditions, and required university ethics determination. Keep the original metadata as TSV; a derived CSV manifest can be generated separately.
2. Reconcile CSV identifiers with imaging directories. Detect duplicate records, repeated exams, and shared cases across collections. Prefer one predefined pretreatment examination per person; document any prior biopsy and exclude or separately define previously treated cases.
3. Verify that all four MRI sequences belong to the same patient and examination. Inspect file readability, image dimensions, orientation, spacing, and spatial alignment.
4. Verify the ground-truth IDH definition and assay provenance where available. Do not map unknown, indeterminate, untested, or blank results to wildtype. Resolve contradictory labels explicitly.
5. Report class counts, missing age and sex, missing sequences, mask availability, site distribution, and reasons for exclusion. Define age-range eligibility with the supervisor.
6. Define a primary cohort with known labels and all four usable sequences. Handle missing clinical inputs with a declared training-fitted imputation policy. Record which cases require imputation.
7. Use this same primary cohort for the clinical, MRI-only, and multimodal comparisons. Clinical-only results from additional patients can be secondary, clearly identified analyses.
8. Explicitly allow only approved preoperative predictors into the clinical model. Exclude molecular results, final integrated diagnosis, tumor grade, postoperative pathology, treatments, resection extent, survival, identifiers, and directory names from model features. Do not use a diagnosis containing IDH information as a substitute for a verified label.

**Deliverables:** patient manifest, cohort flow diagram, data dictionary, class distribution table, and MRI quality-control examples.

**Decision gate:** Inventory which automated and manually refined masks are actually distributed for each patient and verify the numeric region labels. Prefer a consistent supplied automated-mask policy across the main cohort if coverage permits. Use the manually refined subset for a clearly labeled sensitivity analysis, rather than silently mixing mask sources or limiting the entire project to 362 patients. If consistent masks are not available, define and document another uniform localization policy. Do not make manual slice selection an undocumented prerequisite.

## 3 Validation design before model development

Reserve approximately 20 percent of eligible patients as a locked internal test set. If all 622 remain eligible, a conventional 20 percent split gives approximately 497 development patients and 125 test patients. Recalculate after the audit.

Use stratified five-fold cross-validation within the development set for hyperparameter selection, epoch selection, and preprocessing choices. If minority counts cannot support five folds, reduce the fold count. All scans and slices from one patient must remain together; use grouped splitting if repeated examinations are retained. Apply exactly the same patient partitions to every core model.

UTSW-Glioma is a single-institution collection. Inspect scanner, acquisition, and available demographic distributions. A scanner-vendor or temporal analysis can be secondary if enough cases exist; it is not external-hospital validation. A separate independent collection would be needed to evaluate generalization to another institution.

Within each development fold:

- Fit clinical imputation, scaling, category handling, feature selection, and any population-level image normalization using only that fold's training patients.
- Compute class weights using training patients only.
- Apply random image augmentation only to training data.
- Perform deterministic validation and inference with one patient-level output per examination or person, according to the protocol.

Use development out-of-fold predictions to choose the operating threshold and, if implemented, late-fusion weights and probability calibration. Prespecify a rule such as maximizing balanced accuracy. Use a small declared hyperparameter search; internal cross-validation scores used for tuning are development results. The untouched test set provides the primary performance estimate. Unbiased cross-validation claims after extensive tuning would require a nested design.

Before opening the test results, freeze the architecture, slice selection, preprocessing, training duration, seed policy, threshold, and evaluation code. Derive the final training epoch budget from development folds, then refit using all development patients. Do not use the test set for early stopping. This follows the separation of fitting and testing described in [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

**Deliverables:** saved patient split files, fold assignments, reproducible split seed, and a written evaluation protocol.

## 4 MRI and clinical preprocessing

### MRI pipeline

1. Keep original files unchanged. Use NIfTI inputs; convert DICOM only when necessary and retain conversion provenance.
2. Inspect the dataset's supplied preprocessing. Reuse verified registration, skull stripping, and bias correction rather than automatically repeating them.
3. Standardize orientation. Ensure all sequences share one anatomical reference, voxel grid, and spacing. Resampling to a common grid does not correct anatomical misregistration; register sequences when needed and inspect overlays.
4. Use continuous interpolation for MRI and nearest-neighbor interpolation for segmentation masks. A common spacing near 1 mm can be a starting choice if appropriate for the source images.
5. Normalize each sequence per patient within a verified brain mask or suitable brain foreground. Use z-score normalization and a fixed, documented clipping policy. This individual-image normalization does not require fitting on test patients as a population. Any population-derived statistics must come from training patients only.
6. Where tumor masks exist, derive a bounding box around the whole tumor with a predefined margin, initially 10 to 15 mm. Retain surrounding tissue. Use the same crop across all sequences.
7. Initially select up to four axial slices distributed across the tumor extent using a deterministic rule. Include small lesions consistently. If fewer than four distinct usable slices exist, use only those valid slices. Never treat duplicate or padded slices as extra independent evidence.
8. Resize with preserved aspect ratio and pad to 224 by 224 pixels. In the frozen-feature implementation, process each sequence separately: apply a fixed per-image MRI intensity policy, map its clipped intensity range to 0 to 1, replicate the grayscale slice to three channels, and apply the pretrained encoder's documented input normalization. Fix this mapping before final evaluation. Keeping sequence features separate preserves sequence identity without retraining the first convolution.
9. Run the fixed encoder in evaluation mode with gradients disabled, one small image batch at a time. Average slice features within each sequence, then concatenate sequence vectors in the fixed order T1, T1ce, T2, FLAIR. Store a 2048-dimensional vector per patient with its encoder and preprocessing version.
10. The initial cache uses deterministic views. Random image augmentation requires new encoder forwards or separately cached predefined training views; it cannot be applied to an already pooled feature vector. Reuse the same view policy in MRI-only and fusion comparisons.

Tumor-mask localization is an input dependency, not the prediction target. If the classifier uses expert masks, the inference demonstration must accept a mask or include the declared localization method. Describe the resulting assistance requirement. If using a pretrained segmenter, record its training cohorts and check possible overlap with evaluation cohorts.

### Clinical pipeline

- Preserve age as a continuous variable. Use training-fold median imputation and training-fold standardization where required by the estimator.
- Encode sex according to the source data dictionary, with explicit handling of missing and previously unseen categories. Do not silently coerce unknown values.
- Add missingness indicators if prespecified. Fit all preprocessing inside the development folds and save the fitted transformer for inference.
- Use age-only logistic regression as a diagnostic baseline, then compare age plus sex. With only two initial variables, complex tabular models may not improve on logistic regression.

**Deliverables:** reusable preprocessing functions, before-and-after MRI overlays, cached tensors, and fitted clinical transformers.

## 5 Models and experiments

| Experiment | Inputs and model | Role |
| --- | --- | --- |
| E0 | Majority-class predictor | Basic accuracy and balanced-accuracy reference |
| E1 | Age only with logistic regression | Measures how much prediction age explains |
| E2 | Age and sex with logistic regression | Main simple clinical baseline |
| E3 | Age and sex with random forest | Nonlinear clinical comparison |
| E4 | Age and sex with XGBoost | Boosted clinical comparison |
| E5 | Frozen per-sequence ResNet-18 features with a small MRI-only head | Main imaging baseline |
| E6 | The same frozen MRI features plus a clinical MLP and fusion head | Initial proposed system |
| E7 | Weighted combination of E5 and the selected clinical model | First optional fusion comparison |
| E8 | Matched MRI-only and multimodal models with encoder fine tuning | Optional extension after a feasibility pilot |

Choose the clinical model by development results rather than assuming XGBoost is strongest. Give E5 and E6 exactly the same cached MRI features, slice aggregation, and comparable tuning budget. Add regularized logistic regression on the frozen MRI features as a useful secondary check on whether the neural head adds value.

### Concrete intermediate fusion architecture

```text
Each MRI sequence, processed separately
    -> up to four grayscale slices, each replicated to three channels
    -> the same frozen pretrained ResNet-18 without its classifier
    -> 512 features per slice
    -> mean over valid slices for that sequence
    -> 512 features per sequence

Concatenate T1, T1ce, T2, FLAIR vectors
    -> 2048 cached MRI features per patient
    -> training-fitted feature standardization

Clinical vector [B, d]
    -> Linear(d, 16), ReLU, Dropout
    -> Linear(16, 16), ReLU
    -> 16 clinical features per patient

Concatenate 2048 + 16 = 2064 features
    -> Linear(2064, 64), ReLU, Dropout
    -> Linear(64, 1)
    -> one logit per patient
    -> sigmoid for an IDH-mutant probability estimate
```

Here, B is the patient batch size for the small head and d is the dimension after clinical encoding. Compute the training loss once per patient. Do not count slices as independent patients. Fit the feature scaler within each training fold. The fixed encoder can cache features for every patient without learning from their labels, but no scaler, PCA, feature selector, or task-trained encoder may be fitted using held-out patients.

For E5, use a comparable 2048 to 64 to 1 MRI-only head. For E6, start with concatenation; add attention only after the basic study is complete. Strong regularization and small heads remain necessary because the number of patients is limited.

Standard [Torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) supports ImageNet pretrained weights. Keep its original three-channel input for the frozen per-sequence implementation. Record that this is natural-image pretraining, not a glioma-trained representation.

For an optional fine-tuned four-channel implementation, stack the aligned sequences as channels, replace the first convolution, and train the new input layer and selected encoder blocks using development patients only. Pool valid slice features to one 512-dimensional MRI vector and concatenate it with 16 clinical features before the classifier. An initial input-layer policy can average and repeat pretrained filters, but the layer then needs task training to learn sequence distinctions. Use matched four-channel MRI-only and multimodal models for this separate comparison. A frozen four-channel convolution with identical repeated filters would collapse sequence distinctions at the first layer; the initial per-sequence approach avoids that issue.

If E7 is implemented, combine probabilities as p_fused = alpha * p_MRI + (1 - alpha) * p_clinical. Select alpha using development out-of-fold predictions only. Because class weighting can affect probability calibration, assess calibration before interpreting these estimates as risks.

## 6 Training strategy and checks

Recommended initial configuration:

| Setting | Starting choice |
| --- | --- |
| Image size | 224 by 224 |
| Selected slices | Initially up to four valid axial slices per sequence |
| Feature-extraction image batch | Start at one to four images; benchmark memory |
| Patient batch size for cached-feature heads | Start at 16 to 32 on CPU or GPU |
| Optimizer | AdamW |
| Encoder learning rate | None in the initial frozen-feature implementation |
| Head learning rate | Start near 0.001; compare a small declared range |
| Loss | BCEWithLogitsLoss with training-derived positive-class weight |
| Dropout | Start at 0.2 to 0.4 |
| Training budget | Start at a maximum of 50 epochs during development |
| Early stopping | Development-fold patient ROC-AUC, initially patience of eight epochs |
| Precision | Start with float32; benchmark before enabling mixed precision |

For positive class IDH-mutant, an initial positive weight is N_wildtype / N_mutant in the training fold. Avoid automatically combining weighted loss with heavy class oversampling. Sample patient bags so patients with more slices do not dominate the loss.

In the initial implementation, freeze all encoder parameters, call evaluation mode to retain fixed batch-normalization statistics, and disable gradients during feature extraction. Train only the heads and clinical MLP using cached patient vectors. Feature standardization, optional dimensionality reduction, and class weights must still be fitted within each training fold.

For optional encoder fine tuning, benchmark a patient batch of one and only one to three selected slice locations first, unfreeze only a limited portion of the model, and use a lower encoder learning rate near 0.00001. Freeze batch-normalization running statistics for very small batches. Use gradient accumulation if helpful, but recognize that it does not reduce the memory needed for a single patient forward pass. Run full float32 first on the MX230; mixed precision is an option to measure rather than an assumed speedup. This extension may require substantially more runtime and can be moved to an available university GPU if necessary.

Before full experiments:

- Confirm channel order, spatial alignment, label mapping, and patient split disjointness.
- Confirm no identifiers or disallowed clinical columns enter the feature tensors.
- Confirm one patient produces one probability and padded slices do not affect pooling.
- Overfit a very small training subset as a debugging check; this is not a performance result.
- Run one complete fold through preprocessing, checkpoint save, reload, and prediction export.
- Confirm validation has deterministic slice selection and no random augmentation.

Use one fixed seed for initial development. If resources permit, predeclare three final training seeds for both E5 and E6 and average their probabilities for the primary comparison. Record individual-seed variation separately. Decide whether to use an ensemble before test evaluation; do not choose the best seed using test results.

**Deliverables:** versioned configurations, learning curves, checkpoints, timing and memory logs, and development prediction files.

## 7 Evaluation and interpretation

Evaluate every core model on the same held-out patients.

- Primary metric: patient ROC-AUC, with a 95 percent confidence interval.
- Primary comparison: fusion minus MRI-only ROC-AUC with a paired patient bootstrap interval, initially 2,000 resamples. Resample the same patients for both models and ensure both classes are represented.
- Imbalance-aware metric: precision-recall performance, reporting average precision with the positive-class prevalence as its reference. State which precision-recall summary is used.
- Threshold metrics: IDH-mutant sensitivity, IDH-wildtype specificity, precision, F1, balanced accuracy, accuracy, and the confusion matrix using the development-selected threshold.
- Probability assessment: calibration plot and Brier score. Fit any calibrator on development data only and freeze it before testing.
- Stability: seed variation and development-fold variation, clearly distinguished from confidence intervals for held-out performance.

Report sample counts and class counts with every cohort result. An interval for the AUC difference that includes zero does not establish improved discrimination. A small test cohort may give wide intervals; report that uncertainty rather than repeatedly changing the test split.

Use Grad-CAM on representative correct predictions and errors to inspect where the CNN responds. For a cached-feature model, regenerate the selected patient's image forwards with the required activation gradients enabled and propagate through the saved feature scaling, aggregation, and classification head. A detached feature cache alone cannot produce an image heatmap. For clinical baselines, show logistic coefficients or suitable feature-attribution summaries. These are descriptive model explanations, not proof of biological mechanisms. Examine errors by class, age range, sex, scanner, and image quality where subgroup sizes support interpretation.

Useful controlled ablations, in priority order:

1. MRI-only versus MRI plus age versus MRI plus age and sex.
2. Learned clinical MLP versus direct concatenation of encoded clinical variables.
3. Intermediate versus late fusion.
4. Four sequences versus prespecified reduced sequence sets.

Select ablations during development and label additional comparisons as secondary or exploratory. Do not select favorable test cases as representative explanations without also showing errors.

**Deliverables:** model comparison table, ROC and precision-recall curves, calibration plot, confusion matrices, paired difference interval, and an error-analysis report.

## 8 External validation

Consider UCSF-PDGM only after establishing the source of the primary cohort and excluding overlap. The [current TCIA collection listing](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) reports 495 unique patients and 501 examinations and provides metadata identifying overlap with BraTS cases. Deduplicate people and check cross-collection identifiers, not just directory names. A different download source does not make shared patients independent.

For an eligible independent external cohort:

1. Apply compatible inclusion criteria, label definitions, sequence requirements, and MRI preprocessing.
2. Reuse the frozen clinical transformer, model weights, calibration, and classification threshold.
3. Report how many patients were excluded and why, including missing sequences or predictors.
4. Evaluate once without target-cohort retraining, tuning, or threshold adjustment.
5. Report external results separately. If adaptation is later studied, use a separate adaptation partition and retain an untouched external test partition.

If independent external data are unavailable, complete the internal study and describe its generalization limits. Do not describe internal validation as external testing.

## 9 Software and reproducibility

Use Python, PyTorch and Torchvision for models; MONAI, NiBabel and SimpleITK for medical imaging as needed; pandas and NumPy for tables; scikit-learn and XGBoost for tabular baselines and evaluation; and Matplotlib or Seaborn for figures. Select a mutually compatible stable environment and pin the versions that pass the pilot run.

The older MX230 needs a compatible PyTorch binary. The [PyTorch packaging notice](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128) documents removal of Pascal support from CUDA 12.8 builds starting with PyTorch 2.8. Do not install an arbitrary newest GPU build. One candidate to test is Python 3.11, PyTorch 2.7.1 with CUDA 11.8, and Torchvision 0.22.1, using the matching command on the [official previous-versions page](https://pytorch.org/get-started/previous-versions/). Verify CUDA availability and an actual small ResNet forward before pinning the environment; a detected GPU alone does not prove the binary can execute its kernels. If this fails, CPU feature extraction remains a slower fallback. No package installation is required just to read this plan.

Suggested project organization:

```text
glioma_idh/
  README.md
  requirements.txt
  configs/
  data/
    raw/                    # unchanged source files
    manifests/
    processed/
    features/               # fixed per-patient encoder outputs
  splits/                   # fixed patient IDs and fold assignments
  src/
    audit_data.py
    prepare_manifest.py
    preprocess_mri.py
    extract_features.py
    clinical_preprocessing.py
    datasets.py
    models/
      clinical.py
      mri.py
      fusion.py
    train_clinical.py
    train_mri.py
    train_fusion.py
    evaluate.py
    explain.py
    predict.py
  tests/                    # targeted checks for labels, splits, pooling
  outputs/
    checkpoints/
    predictions/
    metrics/
    figures/
  reports/
```

Record dataset versions, manifest and split hashes, preprocessing parameters, software versions, pretrained-weight source, random seeds, configuration, and source revision with each run. Save patient-level predictions so comparisons can be reproduced. Keep raw images, personal data, and credentials out of the source repository.

An inference command should load four matched MRI volumes, age and sex, and the declared localization input; apply saved preprocessing; regenerate the frozen features for the new patient; apply the training-fitted scaler and heads; and export the IDH-mutant probability estimate, fixed-threshold classification, and any preprocessing failure message. The feature cache accelerates research training but does not replace the encoder for a new patient's MRI. A simple local demonstration can follow once this path works. Building a separate website is unnecessary for the core thesis.

## 10 Eight week schedule

| Week | Work | Completion evidence |
| --- | --- | --- |
| 1 | Acquire and audit clinical metadata, implement patient splits and clinical baselines, test leakage controls, and prepare supervisor meeting | Runnable CPU pipeline, audit, development scores, tests, report and questions |
| 2 | Confirm cloud access, acquire MRI, inventory masks, verify alignment and complete a small development preprocessing pilot | Common image-eligible cohort, preserved patient assignments, QC examples and measured runtime |
| 3 | Extract frozen MRI features and develop the MRI-only classifier | Versioned feature matrix and MRI development results |
| 4 | Implement the clinical MLP and intermediate fusion | Matched MRI-only and multimodal development comparisons |
| 5 | Complete selected ablations and optional limited fine tuning if feasible; freeze model choices | Final development configuration, feature policy, seed policy and training budget |
| 6 | Refit models, select thresholds using development predictions, and run the locked internal evaluation | Patient-level metrics, paired uncertainty analysis, calibration and prediction exports |
| 7 | Complete interpretation and error analysis; test an independent external cohort only if already prepared | Explanation figures and external report or explicit internal-validation limitation |
| 8 | Complete thesis results, reproducibility package, inference demonstration and defense materials | Final written project, code, figures and demonstration |

Write the literature review and methods throughout the project. Resolve external-validation requirements and dataset access during the first supervisor meeting; begin external access and overlap checks in week two if required. If cloud or imaging access is delayed, retain the three core comparisons and reduce optional work. Do not assume more GPU memory will recover time lost to dataset acquisition and quality checks.

For the eight-week deadline, preserve the clinical baselines, MRI-only model, fusion model, fixed patient partitions, and held-out evaluation. Reduce ablations, seed repetitions, and interface work before reducing validation quality. A full 3D model, transformer and new segmentation-model training are out of scope unless the supervisor explicitly replaces other deliverables.

## 11 Resources and principal risks

The available machine reports an NVIDIA GeForce MX230 with 4096 MiB dedicated GPU memory and approximately 23.8 GiB system RAM, consistent with a nominal 24 GB installation. Together with the stated Intel Core i7 10th generation CPU, this supports a practical starting path through streaming preprocessing, frozen feature extraction, and small-head training. Full runtime and fine-tuning feasibility still require the pilot. Avoid loading all patient volumes into RAM or caching the whole cohort on the GPU. An upgrade or paid cloud service is not a prerequisite for beginning the study.

With four slices from each of four sequences, 622 patients require about 9,952 slice-level encoder forwards for one deterministic pass. Pooling each sequence to 512 values gives 2048 float32 features per patient; the numeric feature matrix is about 4.9 MiB, excluding identifiers and file-format overhead. Consequently, head training and cross-validation can run on CPU after extraction without repeating all image forwards. More views, alternative crops, or fine-tuned encoders require additional extraction work.

Benchmark preprocessing and encoding separately for 20 to 30 development patients. Estimate full extraction time as measured mean time per patient multiplied by the eligible patient count, then allow for file I/O and interrupted-run recovery. Save each completed patient's features immediately. Measure head-training time separately. A university GPU can be useful for a later fine-tuning extension, but its availability is not assumed in the core schedule.

The primary imaging download is approximately 22.9 GB according to [TCIA](https://www.cancerimagingarchive.net/collection/utsw-glioma/). Check actual disk availability and expanded storage before acquisition; keep a provisional 80 to 120 GB workspace allowance until sizes are measured. External UCSF imaging can be deferred rather than adding its much larger download to the first milestone.

| Risk | Planned response |
| --- | --- |
| Fewer complete cases than the reported 622 | Publish the cohort flow, revise counts and folds, and keep a common cohort for comparisons |
| Automated versus manually refined masks | Inventory coverage, choose a consistent main policy, and assess refined masks as a labeled sensitivity analysis |
| Overfitting | Small heads, staged fine tuning, augmentation, regularization, and development-based stopping |
| Class imbalance | Training-derived weights plus PR analysis, sensitivity, specificity, and balanced accuracy |
| Age or site dominates prediction | Age-only and modality ablations, subgroup analysis, and site-aware evaluation when feasible |
| Duplicate patients across collections | Use provenance, crosswalks, and patient-level deduplication before splitting |
| Inadequate GPU memory | Use frozen-feature inference batches of one, stream patients, and train heads on CPU; benchmark any fine-tuning extension separately |
| Unsupported GPU binary | Test an officially documented compatible PyTorch build with a real forward pass; retain CPU extraction as a fallback |
| No improvement from fusion | Report the paired comparison and uncertainty; investigate through predefined ablations |
| External cohort is unavailable or overlaps | Complete the locked internal study and state the remaining validation limitation |

Do not prescribe a universal training-time estimate before the pilot. Measure preprocessing time, epoch time, and memory, then estimate the number of fold and final-seed runs the available compute can support.

## 12 Completion criteria

The implementation is complete when it includes:

1. An auditable patient cohort, label dictionary, and reproducible partition files.
2. Training-fitted clinical preprocessing and verified multi-sequence MRI preprocessing.
3. Reproducible clinical baselines, an MRI-only classifier, and the proposed fusion classifier, with the frozen or fine-tuned encoder policy explicitly recorded.
4. One probability per patient, saved checkpoints, and saved patient-level predictions.
5. A locked internal comparison with confidence intervals and a paired fusion-versus-MRI analysis.
6. Clearly labeled ablations, interpretation examples, and error analysis.
7. An inference path that uses the same inputs and preprocessing as evaluation.
8. A thesis methods and results package with source references, resource use, limitations, and external evaluation when independently feasible.

The first implementation milestone is clinical metadata acquisition, a verified patient manifest, and fixed partitions, followed by the age-only and age-plus-sex baselines. Complete a small image-processing and feature-extraction pilot before processing all MRI. Train small heads using the frozen cache first; consider encoder fine tuning only after the core workflow is working and local feasibility is measured.

## References

- [Project source document](Model%20Selection.docx).
- [TCIA UTSW-Glioma collection](https://www.cancerimagingarchive.net/collection/utsw-glioma/). Primary dataset access, release, file formats, download size, and attribution requirements. Dataset DOI: 10.7937/DFAE-1B86.
- [Reddy et al. The University of Texas Southwestern Glioma Dataset](https://www.nature.com/articles/s41597-026-07274-4). Primary description of the cohort, available predictors and labels, processing, and segmentation review.
- [Byeon et al. Interpretable multimodal transformer for prediction of molecular subtypes and grades in adult-type diffuse gliomas](https://www.nature.com/articles/s41746-025-01530-4). Supports the relevance of MRI and clinical fusion and the need to frame a distinct, appropriately scoped contribution.
- [Truong et al. Two-Stage Training Framework Using Multicontrast MRI Radiomics for IDH Mutation Status Prediction in Glioma](https://pubs.rsna.org/doi/10.1148/ryai.230218). Provides a related radiomics and multicohort evaluation reference.
- [Li et al. Deep learning radiomics nomograms predict IDH genotypes in brain glioma](https://pubmed.ncbi.nlm.nih.gov/39708927/). Provides a related hybrid imaging and clinical study; its results are contextual literature, not performance targets for this project.
- [TCIA UCSF-PDGM collection](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/). Source for collection access, current unique-patient counts, repeated examinations, and BraTS overlap information.
- [Torchvision ResNet-18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). Source for the standard model and available pretrained weights.
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html). Source for training-only fitting of learned preprocessing and separation of model selection from testing.
- [PyTorch GPU architecture packaging notice](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128) and [official previous versions](https://pytorch.org/get-started/previous-versions/). Sources for choosing and verifying a compatible GPU binary rather than assuming a current default build supports the MX230.
