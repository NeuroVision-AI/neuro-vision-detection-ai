# Evidence-backed Track 1 / Track 2 handoff

Status date: 2026-07-15
Source question set: `questions_for_track1_track2_v2.pdf`
Rule: unavailable evidence is stated as unavailable; it is not inferred.

## Track 1: data collection and research

### Q1. Which datasets are downloaded and ready?

| Dataset | Local status | Usable role |
|---|---|---|
| Kaggle Brain Tumor MRI Dataset, version 2 | Downloaded; 7,200 distributed images, 7,193 retained | Development, validation and locked internal source testing |
| BraTS 2020/2021 | Not present | None |
| TCIA TCGA-GBM / TCGA-LGG | Not present | None |
| UPENN-GBM | Not present | None |
| UCSF-PDGM | Not present | None |
| BRISC | Downloaded/audited | Rejected as independent external validation; masks used only for internal-test XAI |
| BDNeuro-MRI v7 | Downloaded/audited | Rejected as independent external validation |

The question set's 7,023-image count reflects an earlier form of the public collection; the retrieved version contains 7,200 files. Exact source, retrieval and license evidence is in `data/manifests/source_provenance.json` and `research/DATA_CARD.md`.

### Q2. Are BraTS/TCIA access agreements complete?

No completed access agreement or corresponding local data is evidenced in the workspace. Volumetric BraTS/TCIA work is therefore outside the completed study.

### Q3. What is the folder structure?

The materialized analysis data use `data/processed/{train|val|test}/{class_name}/image`. A row-level CSV manifest records source path, source split, class, hashes, duplicate/provenance group, dimensions, mode, final split and materialized path. Patient folders, patient IDs and MRI-sequence folders are unavailable.

### Q4. Exact image counts per class

| Class | Train | Validation | Locked test | Total |
|---|---:|---:|---:|---:|
| Glioma | 1,149 | 247 | 403 | 1,799 |
| Meningioma | 1,066 | 229 | 499 | 1,794 |
| No tumour | 824 | 177 | 799 | 1,800 |
| Pituitary | 1,148 | 246 | 406 | 1,800 |
| **Total** | **4,187** | **899** | **2,107** | **7,193** |

### Q5. Which architecture should be prioritized?

The completed study prioritizes option (a): ImageNet-initialized EfficientNet-B0 versus a custom five-block CNN under one frozen 2D classification protocol. A segmentation pipeline is not supportable from the current four-class image folders because compatible volumetric images and ground-truth segmentation masks are not available for development.

### Q6. Is Indian institutional MRI data available?

No. There is no local/Indian cohort, patient count, ethics approval or acquisition record in the workspace. The paper is therefore framed as a retrospective public-data 2D proof-of-concept, not Indian validation.

### Q7. Which manuscript sections does this work support?

Directly: Methods, cohort/integrity Results, classification/calibration Results, quantitative XAI Results, Discussion, Limitations, reproducibility and data/code availability. Radiogenomics, prognosis, treatment response, radiotherapy, surgery and pathology outcomes are background-only and should not be presented as model endpoints.

### Q8. WHO classification alignment

WHO CNS5 (2021) is the contextual taxonomy. The four outputs remain source-dataset image labels and are not WHO-integrated molecular-histologic diagnoses.

### Q9. Is the literature synthesis written?

Yes. `research/LITERATURE_SYNTHESIS.md` synthesizes the 74-record tracker, and `research/FULL_TEXT_EXTRACTION.md` contains first-reviewer extraction for nine of ten shortlisted papers plus publisher/abstract verification for the remaining preprint. Independent second-reviewer adjudication remains required.

### Q10. Which citations are essential?

The manuscript's methodological core is: CLAIM 2024, TRIPOD+AI, PROBAST+AI, WHO CNS5, EfficientNet, Guo et al. on temperature scaling, Grad-CAM, Adebayo et al. sanity checks, Arun et al. medical-imaging saliency trustworthiness, and the BRISC data article. Same-source classifier papers are contextual comparators, not a common leaderboard. URLs and decisions are recorded in the companion workbook and literature synthesis.

## Track 2: data analysis and validation

### Q11. Is the ready-to-use dataset complete?

Yes for the public-data study: 7,193 materialized images. Seven records in one cross-label perceptual component were excluded whole; no retained exact hash or duplicate/provenance group crosses final splits, and all retained images decode.

### Q12. What preprocessing was applied?

- Source images are copied unchanged into leakage-controlled split/class folders.
- At load time, images are converted from grayscale to RGB, resized to 224 x 224, converted to tensors and normalized with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- No skull stripping, N4 bias correction, CLAHE, denoising or histogram equalization is applied.
- Validation and test receive no augmentation.

### Q13. Image characteristics

- All 7,193 retained source files decode as 8-bit grayscale (`L`) before RGB replication.
- Raw widths range from 150 to 1,375 pixels (median 512); heights range from 167 to 1,446 (median 512).
- No reliable sequence labels are distributed; the intended input is described only as a single 2D contrast-enhanced T1-like image.
- A deterministic 64-image-per-class intensity sample spans 0-255; class-level descriptive summaries are in `outputs/data_quality/data_quality_report.json`.

### Q14. Class imbalance handling

Final class totals are nearly balanced overall, but the preserved official test partition is not. Training uses inverse-frequency class-weighted cross-entropy. Weighted sampling is disabled to avoid double correction.

### Q15. Are labels verified and clean?

Folder labels are not independently pathology-verified, so label accuracy cannot be estimated. One cross-label perceptual component was excluded without visual relabelling; no retained duplicate/provenance component has conflicting labels. No manual label corrections were made. BraTS masks are not part of the development data.

### Q16. Training augmentation

Training only: rotation +/-15 degrees, horizontal flip probability 0.5, affine translation up to 5%, scale 0.95-1.05, brightness and contrast jitter 0.2, and random erasing probability 0.1. Vertical flip and elastic deformation are not used. The applicability of horizontal flipping to every acquisition orientation remains a limitation because orientation metadata are absent.

### Q17. Train/validation/test split

The final proportions are 58.2% / 12.5% / 29.3%, reflecting preservation of the large official test set rather than forcing a 70/15/15 image split. Transitive exact/perceptual duplicate components stay within one partition. Patient-level splitting cannot be verified because patient identifiers are absent. The split manifest is `data/manifests/dataset_manifest.csv`.

### Q18. Data quality report

Completed at `outputs/data_quality/DATA_QUALITY_REPORT.md`, with class-distribution, raw-dimension, sampled-intensity and representative locked-test figures plus a machine-readable JSON report. It also records missing patient/acquisition metadata and the excluded anomaly rather than hiding them.

## Joint questions

### Q19. Compute resources

The captured host is Apple arm64/macOS with PyTorch 2.8.0. CUDA and MPS were unavailable to the study environment, so the prespecified runs execute on CPU. No Colab, Kaggle GPU or cloud budget is evidenced.

### Q20. Handoff format

Both forms are available: `data/processed/{split}/{class}/image` for training and `data/manifests/dataset_manifest.csv` for auditable provenance, grouping and splits.

### Q21. Timeline

No team deadline is recorded. The computational pipeline, both internal evaluations, and paired comparison are complete. EfficientNet optimization was stopped after nine completed epochs for CPU-resource reasons and is explicitly non-protocol-conformant. Human/external-data tasks cannot be assigned a completion date without owners and access.

### Q22. Demo format

The repository contains a Next.js/FastAPI web dashboard and a Streamlit demo. Both are research-only and inference fails closed if a trained checkpoint is absent.

### Q23. Agreed evaluation metrics

Implemented: accuracy; per-class precision, sensitivity/recall, specificity and F1; macro-F1; balanced accuracy; MCC; macro ROC-AUC and PR-AUC; confusion matrix; grouped bootstrap confidence intervals; ECE, Brier score, NLL and risk-coverage. Quantitative Grad-CAM adds repeatability, randomization sensitivity, IoU and pointing game. Segmentation Dice is not applicable to the classification endpoint, and inference-time benchmarking has not been promoted as a study endpoint.

## Residual handoff blockers

1. Obtain a genuinely independent external cohort with compatible labels and auditable provenance.
2. Obtain patient IDs and acquisition/subgroup metadata if patient-level and fairness claims are desired.
3. Complete blinded expert XAI review.
4. Complete independent second-reviewer literature extraction/adjudication.
5. Obtain domain-expert review of the ancillary RAG benchmark if it remains in scope.
