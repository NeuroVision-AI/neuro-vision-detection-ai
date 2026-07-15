# Track 3 completion plan: AI integration and development

Status date: 2026-07-15
Authority: team research objective and Track 3 charter in the shared Google document
Evidence base: source literature tracker, companion audit workbook, generated model/data/XAI artifacts, and repository knowledge graph

## Recommended paper design

The team document calls this an original experimental AI study, while the research objective also requires a critical analysis of current AI applications. The coherent design is therefore:

1. **Structured critical review:** explain the major machine-learning, deep-learning, detection, segmentation, transformer, hybrid, foundation-model and vision-language approaches; compare public MRI resources; and assess validation, clinical applicability, ethical risks and future directions.
2. **Original experimental component:** report the leakage-controlled 2D four-class experiment already implemented on the Kaggle Brain Tumor MRI Dataset, comparing a custom CNN with EfficientNet-B0 under one locked evaluation protocol.

The paper must not be called a systematic review unless database-specific search strings, dates, hit counts, deduplication, screening decisions and a reproducible flow diagram are completed. It must not imply that all reviewed architectures were implemented.

## What the paper will cover

| Required objective element | Planned coverage | Track 3 evidence or action |
|---|---|---|
| AI applications in MRI brain-tumour detection and diagnosis | Classification, detection, segmentation, grading/subtyping, molecular prediction, prognosis, XAI and report generation | Separate endpoints so incomparable metrics are not placed on one leaderboard |
| Machine-learning techniques | Conventional feature/radiomics pipelines and shallow baselines | Review from tracker; do not present as newly implemented unless a reproducible baseline is added |
| Deep-learning techniques | CNN, ResNet, DenseNet, VGG, EfficientNet, ViT, U-Net/nnU-Net, YOLO and hybrid models | Review all families; implement only models compatible with the available endpoint/data |
| Contemporary AI | 3D brain-MRI foundation models and multimodal brain-imaging VLMs | Add BrainIAC, BrainFound, Brainfound and Med-Gemini as future-direction/context sources with capability-specific wording |
| Datasets | BraTS, TCIA collections, Figshare and Kaggle; provenance, patients, modality, labels, masks, licensing and overlap | Kaggle is implemented; the others are reviewed unless acquired under a separate volumetric protocol |
| Model performance | Endpoint-specific metrics, confidence intervals and validation design | Internal classification evaluation is complete; external performance is not estimable from rejected candidates |
| Clinical applicability | Multi-site validation, workflow role, calibration, uncertainty, expert review and failure modes | Current evidence supports research-only internal source discrimination, not diagnosis or deployment |
| Challenges and ethics | Leakage, source confounding, privacy/consent, bias, missing subgroups, label validity, explainability and generative hallucination | Most controls are documented; patient metadata and independent expert review remain unavailable |
| Future directions | Volumetric multimodal learning, foundation models, prospective evaluation, federated learning and molecular/pathology integration | Present as future work, not achieved capability |

## Architecture decisions

| Architecture | Paper role | Implementation decision |
|---|---|---|
| Custom CNN | Baseline and reproducibility comparator | Implemented and protocol-conformant |
| EfficientNet-B0 | Transfer-learning comparator | Evaluated, but optimization stopped after nine epochs for CPU-resource reasons; rerun to prespecified stopping on suitable compute for a conformant result |
| ResNet, DenseNet and VGG | Established transfer-learning context | Review; optional additional baseline only if compute and a frozen analysis plan are approved |
| Vision Transformer / Swin | Transformer context and data-efficiency discussion | Review; do not train opportunistically after test inspection |
| U-Net / nnU-Net | Tumour segmentation | Review only for the current paper because the development folders lack compatible volumetric images and masks |
| YOLO models | Lesion localization/detection | Review only unless bounding-box annotations and a separate detection endpoint are acquired |
| Hybrid models | CNN-attention/transformer/fusion context | Review and critically compare validation design rather than headline accuracy alone |
| 3D foundation models | Volumetric representation learning and label-efficient transfer | Future direction; requires patient-level MRI volumes and substantially greater compute |
| Multimodal VLMs | Image-text alignment, report generation and dialogue | Future direction; requires paired volumes/reports and clinician factuality evaluation |

## Dataset decisions

- **Kaggle Brain Tumor MRI Dataset:** development and locked internal source test for the implemented 2D four-class experiment. It does not provide verified patients, acquisition metadata or integrated WHO diagnoses.
- **Figshare brain MRI:** review as an important public source and potential component of redistributed collections. It must be cross-audited before being called external because image reuse is common.
- **BraTS:** review as the principal multimodal volumetric segmentation benchmark. A BraTS experiment would be a separate T1/T1ce/T2/FLAIR segmentation protocol with patient-level volumes and Dice/HD95 endpoints.
- **TCIA:** review as a repository rather than a single homogeneous dataset. Any TCIA experiment must name the exact collection, version, access terms, patient count, sequences, labels and clinical endpoint.
- **Independent external cohort:** still required for an external-generalization claim. BRISC and BDNeuro-MRI v7 failed the current cross-corpus independence rule.

## Endpoint-specific evaluation

| Task | Required metrics |
|---|---|
| Four-class classification | Macro-F1, accuracy, balanced accuracy, per-class precision/recall/specificity/F1, MCC, macro ROC-AUC and PR-AUC, grouped 95% CIs, confusion matrix |
| Calibration and selective prediction | ECE, multiclass Brier score, NLL, calibration curve and risk-coverage; validation-only fitting |
| Segmentation | Dice, IoU/Jaccard, HD95, sensitivity and lesion-wise performance |
| Object detection | mAP@0.5 and mAP@0.5:0.95, sensitivity, precision and false positives per image/scan |
| Explainability | Repeatability, parameter randomization, mask localization and blinded expert utility |
| Report-generating VLM | Clinician factuality/completeness review, clinically significant error rate and task-appropriate text metrics; BLEU/ROUGE alone are insufficient |

## Completion gates

### P0 — required before the paper can claim completion

1. Choose and state the final design as an experimental paper with a structured critical review; avoid “systematic review” language unless the missing search audit is supplied.
2. Rerun EfficientNet-B0 to the prespecified stopping rule on appropriate compute, or retain the current result as an explicitly resource-constrained exploratory comparator.
3. Obtain a genuinely independent, label-compatible external cohort and evaluate both frozen models, or change the title/objective so external validation is not implied.
4. Complete the architecture/dataset evidence tables with full-text-supported cohort, split, endpoint, metric and validation-unit fields.
5. Complete independent second-reviewer extraction/adjudication and resolve the remaining preprint.
6. Complete blinded neuroradiology/neuro-oncology review before making any XAI utility statement.
7. Perform team internal review, reconcile Track 1/2/3 outputs, and freeze the final manuscript tables and figures.

### P1 — required for the broad research objective

1. Preserve database-specific search strings, search dates, hit counts, export files and deduplication decisions for PubMed, IEEE Xplore, Scopus, ScienceDirect, SpringerLink, Google Scholar and arXiv.
2. Add explicit critical-review coverage of ResNet, DenseNet, VGG, EfficientNet, ViT, U-Net, YOLO, hybrid methods and modern brain-imaging foundation models.
3. Add a dataset comparison table for BraTS, exact TCIA collections, Figshare and Kaggle, including volume/slice unit, sequences, patients, labels/masks, license and known reuse risk.
4. Expand error analysis with model disagreement examples and failure strata. Patient/site/fairness analysis must remain non-estimable unless metadata are obtained.
5. Finalize clinical-applicability and ethics sections covering intended use, privacy/consent, bias, calibration, explainability, hallucination risk and human oversight.

### P2 — optional extensions, not prerequisites for this paper

1. Add a frozen ResNet or ViT baseline only if it is specified before further locked-test inspection.
2. Run a separate BraTS volumetric segmentation study using U-Net/nnU-Net or a 3D foundation encoder.
3. Evaluate a multimodal VLM only after obtaining paired imaging/report data and a blinded clinician rubric.
4. Retain the literature RAG subsystem as an engineering appendix or separate study unless domain-expert review is completed.

## Current definition of done

Track 3 is complete for **internal computational implementation** when the data manifest, preprocessing, custom CNN, resource-qualified EfficientNet result, internal metrics, paired comparison, calibration analysis, XAI metrics, figures, model/data cards and reproducibility tests are delivered. The **research paper is complete** only when the P0 human/external-evidence gates above are resolved or the corresponding claims are removed.

## Key contemporary sources for future-direction coverage

- Med-Gemini: https://arxiv.org/abs/2405.03162
- Brainfound multimodal brain-imaging model: https://doi.org/10.1016/j.patter.2026.101538
- BrainFound 3D brain-MRI model: https://arxiv.org/abs/2510.23415
- BrainIAC 3D brain-MRI foundation model: https://www.nature.com/articles/s41593-026-02202-6
- Team Track 3 document: https://docs.google.com/document/d/1rS4p6vMjmXOci4Ny_Guxqjpdo18BBXiJWXcLVdX5kT4/edit?pli=1&tab=t.0
