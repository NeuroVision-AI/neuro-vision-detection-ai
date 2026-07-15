# Leakage-aware four-class brain MRI classification: internal model comparison, calibration analysis, quantitative Grad-CAM, and external-candidate reuse audits

> Draft status: completed results below are populated only from generated study artifacts. The EfficientNet-B0 run was stopped after nine completed epochs for CPU-resource reasons and is explicitly analysed as a non-protocol-conformant internal comparator. External validation and blinded expert review remain unevaluated.

## Abstract

### Background

Public brain MRI classification studies frequently report high internal performance, but duplicate images, image-level splitting, source confounding, absent calibration and limited external testing can produce optimistic estimates. We developed a reproducible four-class 2D MRI classification proof-of-concept with explicit leakage controls, calibration and explainability evaluation.

### Methods

We constructed patient/duplicate components using available patient identifiers, raw-byte SHA-256 hashes and perceptual hashes. Official test data were preserved, and all connected records were assigned to a single split. EfficientNet-B0 transfer learning was compared with a custom CNN. Temperature scaling was fitted on validation data only. The prespecified primary endpoint was external macro-F1 with patient- or duplicate-group bootstrap 95% confidence intervals. Secondary analyses included per-class sensitivity, specificity and F1, MCC, macro ROC-AUC and PR-AUC, ECE, multiclass Brier score, risk–coverage, and quantitative Grad-CAM checks.

### Results

The analysis cohort contained 7,193 images; patient counts could not be established from the distributed metadata. On 2,107 locked-test images, the resource-constrained EfficientNet-B0 checkpoint achieved macro-F1 0.908 (duplicate/provenance-group bootstrap 95% CI 0.891–0.923) and accuracy 0.907 (0.889–0.925), versus 0.833 (0.812–0.854) and 0.839 (0.817–0.861) for the protocol-conformant custom CNN. Paired differences were 0.075 macro-F1 (0.057–0.094) and 0.068 accuracy (0.050–0.086). Validation-only temperature scaling worsened internal-test calibration for both models. On 1,045 tumour images with mapped masks, custom-CNN Grad-CAM repeatability correlation was 1.00, but mean localization IoU was 0.029 (image-bootstrap 95% CI 0.026–0.032) and pointing-game success was 6.4% (5.0–8.0%). BRISC and BDNeuro-MRI v7 failed cross-corpus independence audits, so the external primary endpoint was not evaluated.

### Conclusion

Leakage controls and independent candidate audits materially reduced the scope of supportable claims. EfficientNet-B0 outperformed the custom CNN internally, but its optimization was resource-constrained, calibration did not transfer cleanly from validation to test, and custom-CNN Grad-CAM did not reliably localize reference lesions. Without a qualifying external cohort, clinical or cross-source generalization cannot be claimed. This retrospective public-data 2D system is a research proof-of-concept and is not intended for clinical diagnosis or WHO CNS integrated tumour classification.

## Introduction

MRI-based machine learning may support neuro-oncology research, but apparent model accuracy is highly sensitive to cohort provenance, split unit and external distribution shift. The common four-class public benchmark combines labels that are convenient for image classification but do not represent the molecular-histologic integration required by [WHO CNS5](https://publications.iarc.who.int/Book-And-Report-Series/Who-Classification-Of-Tumours/Central-Nervous-System-Tumours-2021). A credible study must therefore separate dataset-label prediction from clinical diagnosis.

The existing literature tracker identified extensive classification, segmentation, radiomics, prognosis and implementation research. However, the directly relevant evidence remains dominated by small proof-of-concept or single-centre studies, while calibration, patient-level splitting and quantitative explanation validation are less consistently represented. [CLAIM 2024](https://pubs.rsna.org/doi/10.1148/ryai.240300), [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378), and [PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505) emphasize intended use, data provenance, separation of development and testing, reproducibility, risk of bias and applicability.

We therefore designed a leakage-aware comparison of a transfer-learning model and a lightweight custom CNN. We prespecified external macro-F1 as the primary endpoint and evaluated calibration, uncertainty and Grad-CAM behavior rather than treating confidence scores or visually plausible heatmaps as self-validating.

## Methods

### Study design and intended use

This was a retrospective public-data model-development and external-candidate audit study. Independent external validation was prespecified, but candidate cohorts were required to pass cross-corpus reuse and provenance review before performance evaluation. The intended use was methodological research on four source-dataset labels. The system was not evaluated as a medical device, clinical workflow intervention or diagnostic report generator.

### Data sources and participants

Development and external-candidate sources, licenses, provenance limitations, image counts and exclusions are summarized in Table 1. Eligibility and exclusions followed the frozen protocol. No local or Indian validation claim was made because no qualifying local cohort was evaluated.

**Table 1. Cohorts and prespecified analytical role**

| Source | Distributed records | Retained / evaluated | Patient identifiers | Independence audit | Analytical role |
|---|---:|---:|---|---|---|
| Brain Tumor MRI Dataset, version 2 | 7,200 | 7,193 analysis images; 2,107 locked test | Not available | Development source; one seven-record cross-label perceptual component excluded whole | Development and internal validation |
| BRISC 2026 classification set | 6,000 | 0 external performance records; 1,045 unique locked-test mask mappings used for XAI | Not used for the mapped analysis | 4,781/6,000 exact matches to retained development images | Rejected as independent external validation; ancillary masks only |
| BDNeuro-MRI v7 | 5,941 | 0 performance records | Not available in distributed files | 3,314/5,941 exact matches and 4,490/5,941 dHash-near matches to development images | Rejected as independent external validation |

Counts describe distributed image files, not participants. Source licenses and retrieval details are recorded in the machine-readable provenance artifacts and data card.

### Data integrity and partitioning

Before image transformation, we recorded source paths, labels, original split, SHA-256, perceptual hash, decoded dimensions and patient identifiers where available. Transitive identity components were formed across patient, exact-duplicate and perceptual-duplicate links. Components touching an official test split remained in test. All remaining components were allocated to training or validation without crossing component boundaries. The audit failed closed on cross-split overlap or unreadable images. Whole components with conflicting source labels were excluded and logged without visual relabelling. Where patient identifiers were absent, duplicate grouping reduced observable reuse but could not establish patient-level separation.

### Models and training

[EfficientNet-B0](https://proceedings.mlr.press/v97/tan19a.html) initialized from ImageNet was the primary architecture; the existing custom five-block CNN was the comparator. Both received 224 × 224 RGB tensors. Training-only augmentation and optimization followed `configs/experiment.yaml`. Class-weighted cross-entropy was used; weighted sampling was disabled to avoid double correction. Early stopping and checkpoint selection used validation data only. The custom CNN completed its prespecified early-stopping run. EfficientNet training was stopped for CPU-resource reasons after nine completed epochs, after epoch 9 produced the best validation accuracy (0.9744); the frozen epoch-9 checkpoint was then calibrated on validation data and evaluated once on the locked test. This deviation was decided before inspecting EfficientNet test performance.

### Calibration and uncertainty

A single positive temperature was fitted by minimizing validation negative log-likelihood after model selection, following the post-hoc calibration approach evaluated by [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html). The locked test and external sets were not used for calibration. We reported ECE, multiclass Brier score, negative log-likelihood and risk–coverage. The 0.70 abstention threshold was exploratory unless separately validated.

### Explainability

[Grad-CAM](https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html) maps were assessed for repeatability, sensitivity to parameter randomization, localization IoU and pointing-game performance where masks existed. Blinded expert utility review was prespecified but not completed. The protocol was informed by saliency-map [sanity checks](https://proceedings.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) and medical-imaging [trustworthiness criteria](https://pubs.rsna.org/doi/10.1148/ryai.2021200267). Example maps were sampled by prespecified performance strata.

### Statistical analysis

External macro-F1 was primary. Accuracy and macro-F1 confidence intervals used percentile bootstrap resampling by patient/duplicate component. Secondary metrics and subgroup results were descriptive with uncertainty intervals. Analyses added after viewing test results were labelled post hoc.

## Results

### Cohort and integrity audit

The retrieved source contained 7,200 images. One cross-label perceptual-duplicate component containing seven records was excluded without relabelling, leaving 7,193 images: 4,187 training, 899 validation and 2,107 locked-test images. The manifest contained 6,160 duplicate/provenance groups, including 153 exact-duplicate and 574 perceptual-duplicate groups; no group or exact hash crossed final partitions, and all included images decoded successfully. Patient identifiers were unavailable, so patient-level separation could not be verified.

All retained source files decoded as 8-bit grayscale before RGB replication. Raw widths ranged from 150 to 1,375 pixels (median 512) and heights from 167 to 1,446 (median 512). Class-distribution, dimension, deterministic sampled-intensity and representative locked-test figures are archived in the generated data-quality report; sequence, scanner and acquisition metadata were unavailable.

Preserving the source's official test folder produced an imbalanced locked test: 799 no-tumour images versus 403 glioma, 499 meningioma and 406 pituitary images. Macro-averaged and class-specific metrics were therefore emphasized alongside overall accuracy; class-weighted loss was fitted from training counts only.

BRISC was assessed as an external candidate. Final-manifest exact-hash comparison found that 4,781 of 6,000 classification records (79.7%) reused images from the development analysis cohort. BRISC was therefore rejected for independent external validation. Its masks were retained only as an ancillary localization reference: 1,069 mappings resolved to 1,045 unique already-locked internal-test images.

BDNeuro-MRI v7 was also assessed as a post-protocol external candidate. Independent hashing found 3,314 of 5,941 images (55.8%) byte-identical to the development source and 4,490 (75.6%) within a 64-bit difference-hash Hamming distance of five. The candidate was therefore rejected before any model performance was calculated on it. No qualifying independent external cohort was available in the workspace, so the prespecified external primary endpoint remains unevaluated.

### Discrimination and external generalization

The resource-constrained EfficientNet checkpoint's internal locked-test macro-F1 was 0.908 (95% CI 0.891–0.923), accuracy was 0.907 (0.889–0.925), balanced accuracy was 0.893, MCC was 0.876, macro one-vs-rest ROC-AUC was 0.980 and macro PR-AUC was 0.963. Class F1 was 0.882 for glioma, 0.852 for meningioma, 0.909 for no tumour and 0.988 for pituitary. The custom CNN's macro-F1 was 0.833 (0.812–0.854), accuracy was 0.839 (0.817–0.861), balanced accuracy was 0.816, MCC was 0.787, macro ROC-AUC was 0.954 and macro PR-AUC was 0.912. Its class F1 was 0.832, 0.677, 0.860 and 0.962, respectively. Paired group bootstrap estimated EfficientNet-minus-custom differences of 0.075 macro-F1 (0.057–0.094) and 0.068 accuracy (0.050–0.086). An exploratory image-level exact McNemar test found 164 EfficientNet-only correct and 20 custom-CNN-only correct images (two-sided p=2.58×10−29), but its independence assumption is weakened because some duplicate/provenance components contain multiple images. All intervals resampled 1,460 observable components, not verified patients. No external performance was calculated because both audited candidates failed independence checks.

### Calibration and selective prediction

The validation-fitted temperature was 0.608 for EfficientNet and 0.550 for the custom CNN. On the locked internal test, scaling worsened EfficientNet ECE from 0.053 to 0.056, multiclass Brier score from 0.142 to 0.150, and negative log-likelihood from 0.319 to 0.347. For the custom CNN it worsened ECE from 0.029 to 0.099, Brier score from 0.254 to 0.271, and negative log-likelihood from 0.502 to 0.636. Thus, validation-fitted scaling did not improve calibration on this test distribution and the resulting probabilities should not be described as calibrated. The risk–coverage analyses are descriptive research outputs only.

### Explainability

Across 1,045 unique locked-test tumour images with BRISC mask mappings, repeated deterministic Grad-CAM generation had mean rank correlation 1.00. Full-model randomization sensitivity was 0.632 (95% CI 0.616–0.649), but mean localization IoU was only 0.029 (0.026–0.032) and pointing-game success was 0.064 (0.050–0.080). Localization was particularly low for pituitary images (mean IoU 0.009; pointing game 0.012). These findings do not support treating Grad-CAM as a lesion localizer. Blinded expert utility review remains incomplete.

## Discussion

### Principal findings

The main completed finding is not a high headline accuracy but a set of evidence boundaries. The resource-constrained EfficientNet checkpoint outperformed the protocol-conformant custom CNN on the internal source test, especially for meningioma, but cannot be presented as a fully prespecified primary run. Validation-fitted temperature scaling degraded all three reported calibration scores for both models, illustrating that calibration itself can shift. Custom-CNN Grad-CAM was perfectly repeatable under identical computation yet localized reference masks poorly; deterministic stability therefore did not imply clinical faithfulness. Most importantly, two plausible contemporary external candidates reused substantial portions of the development source, preventing an external-generalization claim.

### Comparison with prior work

Recent same-source four-class studies commonly report internal accuracies near 99%, but their validation units and evidence scopes differ from ours. [Iftikhar et al.](https://doi.org/10.1186/s40708-025-00257-y) used 5,618/702/702 images from the 7,023-image collection and reported 99.21% accuracy internally plus 94.72% on a second image collection; their explanation analysis relied on qualitative Grad-CAM, SHAP and LIME interpretation rather than parameter-randomization or mask-localization tests. [Gorenshtein et al.](https://doi.org/10.1007/s10278-025-01686-1) reported 98.5–99.2% for pretrained ResNets on the public 5,712/1,311 split and described fivefold patient-wise cross-validation, although the distributed image collection does not document patient identifiers. [Aiya et al.](https://doi.org/10.1038/s41598-025-04591-3) restructured the same public source into 5,688 training, 632 validation and 703 test images, reported 99% accuracy, and used qualitative review of Grad-CAM by one consultant radiologist; that paper also identifies external validation as necessary. These results are therefore contextual comparators rather than a common leaderboard.

The first-reviewer full-text audit also found examples of test-set use during model development and misaligned binary studies in the tracker shortlist. [Khan et al.](https://doi.org/10.1007/s12672-025-02671-4) used a random 80/20 image split of 4,517 images, stated that no separate validation set was used, and described early stopping based on testing loss. Two shortlisted papers evaluated binary tumor/no-tumor or low-/high-grade tasks rather than the four-class endpoint and were removed from the core comparison. Together with [review-level reports](https://doi.org/10.3390/diagnostics12081850) of inconsistent validation methods, metrics and training data, these findings support emphasizing provenance, split independence, calibration and external-corpus auditing over headline accuracy alone. Detailed extraction and corrections are in `research/FULL_TEXT_EXTRACTION.md`.

### Strengths

Prespecified claim scope; provenance manifest; transitive duplicate grouping; preservation of official test data; external primary endpoint; validation-only calibration; grouped confidence intervals; quantitative Grad-CAM checks; reproducible artifacts.

### Limitations

Expected limitations include retrospective public data, incomplete acquisition/subgroup metadata, 2D sampling, dataset-label taxonomy, an imbalanced official source-test partition, possible residual near-duplicate detection error, and limited clinical applicability. EfficientNet optimization was stopped after nine epochs for CPU-resource reasons and is not protocol-conformant. When source patient identifiers are unavailable, differently appearing images from the same person may cross partitions despite duplicate controls. Only the custom CNN underwent quantitative Grad-CAM testing. RAG performance is outside the primary imaging endpoint.

### Future directions

The next technical step is not to attach a general-purpose language model to isolated 2D images, but to evaluate patient-level volumetric and multimodal representations under endpoint-specific protocols. Brain-MRI foundation encoders such as [BrainIAC](https://www.nature.com/articles/s41593-026-02202-6) and [BrainFound](https://arxiv.org/abs/2510.23415) support transfer to volumetric segmentation, molecular prediction and other neuroimaging tasks. The separate multimodal [Brainfound](https://doi.org/10.1016/j.patter.2026.101538) framework connects brain CT/MRI sequences with language for diagnosis, segmentation, report generation and dialogue. [Med-Gemini-3D](https://arxiv.org/abs/2405.03162) provides evidence for 3D CT report generation, but not a validated brain-tumour MRI detector. A future experiment would require co-registered T1, T1ce, T2 and FLAIR volumes, patient-level identifiers, tumour masks or molecular outcomes, independent-site testing, adequate accelerator compute and blinded clinician evaluation. These models are therefore future-direction context rather than direct comparators to the present single-image classifier.

## Data and code availability

Code, configuration, split manifests, audit outputs and aggregate metrics should be released subject to source-dataset licenses. Raw images must be obtained from their original repositories.

## Ethics

The study uses de-identified public data. Confirm each source's terms and institutional requirements before submission. No prospective patient-care claim is made.

## Primary source and reporting references

- Development collection: [Brain Tumor MRI Dataset, version 2](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
- BRISC candidate: [BRISC Scientific Data article](https://www.nature.com/articles/s41597-026-06753-y).
- BDNeuro-MRI candidate: [Mendeley Data version 7 record](https://data.mendeley.com/datasets/zwr4ntf94j/7).
- Model, calibration, explanation, taxonomy and reporting references are linked at their first mention above.

The 74-record tracker remains a focused evidence map rather than a completed systematic review. Nine of ten shortlisted records have first-reviewer full-text extraction and the remaining preprint has publisher/abstract verification; independent second-reviewer extraction and adjudication are still required before submission.
