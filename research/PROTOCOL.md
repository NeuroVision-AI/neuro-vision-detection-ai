# AI NeuroOnco prespecified research protocol

Version: 1.0
Date frozen: 2026-07-15
Status: protocol complete; development data locked; internal empirical execution complete with one recorded resource deviation

## Prespecified working title

Leakage-aware, calibrated, externally tested explainable four-class brain MRI classification: a public-data proof-of-concept.

Because no audited candidate qualified for the external endpoint and validation-fitted temperature scaling did not improve either model, the current evidence-constrained manuscript title is “Leakage-aware four-class brain MRI classification: internal model comparison, calibration analysis, quantitative Grad-CAM, and external-candidate reuse audits.” The title change narrows claims; it does not change the prespecified endpoint.

## Claim boundary

This is a retrospective research study of single 2D images. The system is not a medical device and will not be described as clinical-grade, volumetric, diagnostic, prognostic, or validated in an Indian population without appropriate local data. Glioma, meningioma, pituitary tumour, and no-tumour are source-dataset labels—not integrated WHO CNS5 diagnoses.

## Primary objective

Estimate the macro-F1 of the locked EfficientNet-B0 model on an untouched external dataset, with a 95% confidence interval obtained by resampling patients or, where patient identifiers are unavailable, duplicate/provenance groups.

## Secondary objectives

1. Compare EfficientNet-B0 with the existing custom CNN under the same splits and preprocessing.
2. Measure per-class sensitivity, specificity, precision and F1; balanced accuracy; MCC; macro one-vs-rest ROC-AUC and PR-AUC.
3. Measure calibration using validation-fitted temperature scaling, ECE, multiclass Brier score and negative log-likelihood.
4. Evaluate selective prediction using risk–coverage curves and a prespecified 0.70 abstention threshold. This threshold is exploratory until validated.
5. Evaluate Grad-CAM repeatability, model-randomization sensitivity, localization when masks exist, and expert-rated utility.

## Data sources

### Development data

The intended initial source is the public four-class brain MRI collection currently referenced by the downloader. Before training, the exact dataset version, license, original component datasets, patient/image counts, acquisition information, and source split must be recorded in `data/manifests/dataset_manifest.csv`.

### External data

An external dataset must pass a locked cross-corpus audit before evaluation. BRISC and BDNeuro-MRI v7 were assessed because they provide current four-class images or masks, but compatibility of labels, sequence, license, provenance and cross-corpus image reuse must be confirmed independently. Any exact or material near-duplicate reuse disqualifies a candidate from independent external validation. An overlapping dataset may be used only for a transparently labelled ancillary XAI-mask analysis on already locked test images. The external source may not be used for augmentation, model selection, threshold selection or calibration.

### Inclusion

- Images assigned to one of the four prespecified source labels.
- Decodable JPEG, PNG, TIFF or DICOM single images.
- Records that pass provenance and duplicate audit.

### Exclusion

- Unreadable or truncated images.
- Conflicting class labels within the same patient/duplicate component.
- Images with uncertain provenance that cannot be assigned to a locked split.
- Unsupported volumes or multi-frame DICOM studies unless a separate volumetric protocol is approved.

## Leakage control and split formation

1. Preserve any official source test partition.
2. Build raw-byte SHA-256 and perceptual difference hashes.
3. Extract patient identifiers using a documented dataset-specific regex when available.
4. Construct transitive components across patient ID, exact hash and perceptual hash.
5. Lock every component touching the official test partition to test.
6. Allocate remaining components to training and validation while approximately preserving class balance.
7. Fail before training if a component or exact hash crosses splits or an image cannot be decoded.
8. If a component contains conflicting source labels, retain the failed pre-exclusion audit and exclude the entire component only through the explicit logged exclusion policy; never relabel it from image appearance.

If the source does not expose patient identifiers, duplicate/provenance grouping is a fallback rather than proof of patient-level separation. Residual same-patient overlap across differently appearing images must be disclosed as a limitation.

The manifest, audit JSON and final split files are immutable study artifacts.

## Model development

### Models

- Primary: EfficientNet-B0 initialized from ImageNet weights.
- Comparator: existing five-block custom CNN.

Both models use 224 × 224 RGB tensors derived from the source 2D image. ImageNet normalization is used for the transfer-learning model. The same split and evaluation code apply to both models.

### Augmentation

Training-only transformations may include resize, rotation up to ±15°, horizontal flip, small translation/scale, mild brightness/contrast changes and random erasing. Vertical flips are prohibited. Augmentation parameters are frozen in `src/config.py` and `configs/experiment.yaml`.

### Class imbalance

Use class-weighted cross-entropy. Weighted sampling is disabled to avoid simultaneously correcting the same imbalance in both sampler and loss.

### Selection and calibration

Early stopping and model selection use validation data only. Temperature scaling is fitted to validation logits after the best checkpoint is selected. Test and external data are never used for temperature fitting.

## Statistical analysis

The primary estimate is external macro-F1. Accuracy and macro-F1 receive percentile 95% bootstrap intervals. The bootstrap unit is patient/duplicate group where the manifest permits; image-level intervals must be explicitly labelled and treated as weaker evidence.

No significance claim will be based solely on overlapping or non-overlapping confidence intervals. A model comparison may use paired group bootstrap differences and, for paired top-label errors, McNemar's test if assumptions are satisfied. Missing subgroup metadata will be reported rather than imputed.

## Explainability analysis

Grad-CAM is an experimental model explanation, not a lesion detector. Required checks are:

- Repeat generation under identical model/input settings.
- Parameter-randomization comparison; explanations should materially change.
- IoU and pointing-game analysis when reference masks exist.
- A blinded expert rubric addressing localization usefulness, misleading emphasis and failure modes.
- Representative successes and failures chosen by prespecified strata, not visual appeal.

## RAG scope

The RAG subsystem is ancillary to this imaging paper. It may only be claimed as evaluated after row-level corpus provenance and the held-out benchmark report retrieval recall@k, precision@k, hit@k and reciprocal rank. Answer faithfulness and citation precision require a separate blinded annotation protocol.

## Reporting and bias assessment

The manuscript will use [CLAIM 2024](https://pubs.rsna.org/doi/10.1148/ryai.240300) and [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378). Risk of bias and applicability will be reviewed using [PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505). Clinical terminology will follow [WHO CNS5](https://publications.iarc.who.int/Book-And-Report-Series/Who-Classification-Of-Tumours/Central-Nervous-System-Tumours-2021).

## Deviations

Any change after data inspection must be recorded with date, rationale, affected outcome and whether it was made before or after observing test performance. Confirmatory and post-hoc analyses must remain visibly separated. The active log is `research/DEVIATIONS.md`.
