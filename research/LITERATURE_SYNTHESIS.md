# Literature synthesis and paper alignment

Status date: 2026-07-15
Source: 74 records in `AI_NeuroOnco_Literature_Tracker.xlsx`, normalized in the companion audit workbook.

## Evidence boundary

The source tracker originally marked all 74 records as abstract-level while 29 were simultaneously marked “Read Deeply,” an internally inconsistent status. A first reviewer has now completed full-text extraction for nine of the ten shortlisted records and publisher/abstract verification for the remaining preprint; corrections are recorded in `FULL_TEXT_EXTRACTION.md` and the companion workbook. Independent second-reviewer extraction and adjudication remain incomplete, so this is an evidence map rather than a systematic review.

## Paper-facing synthesis

The tracker spans classification, segmentation, radiomics, prognosis, datasets and implementation. The initial audit placed 13 records in the core classifier tier and 10 in the verification queue; full-text correction removed two binary studies, leaving 11 records in the core tier and eight in the corrected core shortlist. The recurring opportunity is methodological rather than architectural: evaluate a deliberately simple model comparison under explicit provenance, transitive duplicate control, locked testing, validation-only calibration, uncertainty intervals, quantitative explanation checks and a genuine external-corpus audit.

That framing matches the evidence generated in this repository:

- The analysis manifest contains 7,193 images after whole-component exclusion and has no observable exact-hash or duplicate/provenance-group crossing between splits.
- The custom-CNN comparator has a completed locked internal-test evaluation with grouped confidence intervals and an explicit calibration failure analysis.
- Quantitative Grad-CAM repeatability, randomization sensitivity and mask-localization analyses are complete; blinded expert utility review is not.
- BRISC and BDNeuro-MRI v7 materially reuse the development collection and cannot support an independent external-performance claim.
- The RAG subsystem is ancillary and has only a preliminary, internally authored retrieval benchmark.

The defensible paper contribution is therefore a leakage-aware internal model comparison plus quantitative XAI and external-candidate reuse audits—not a claim of clinical diagnosis or external generalization.

## Architecture and future-direction coverage

The team's broad objective requires the review to distinguish classification, object detection, segmentation, molecular prediction, prognosis and report generation rather than ranking unlike endpoints by headline accuracy. CNN, ResNet, DenseNet, VGG, EfficientNet, Vision Transformer, U-Net/nnU-Net, YOLO and hybrid families should all be reviewed, but only the custom CNN and EfficientNet-B0 are direct experimental models in the current 2D four-class study. U-Net and YOLO require segmentation masks or bounding boxes, and volumetric models require patient-level MRI volumes; those inputs are absent from the current development folders.

Recent foundation-model work belongs in the future-direction synthesis. [Med-Gemini-3D](https://arxiv.org/abs/2405.03162) demonstrated multimodal-model report generation from 3D CT rather than high-accuracy brain-tumour MRI detection. [Brainfound](https://doi.org/10.1016/j.patter.2026.101538) is a multimodal brain CT/MRI and language foundation model spanning diagnosis, segmentation, enhancement, report generation and dialogue. A distinct model named [BrainFound](https://arxiv.org/abs/2510.23415) learns representations from sequential 3D brain-MRI slices for downstream detection and segmentation, while [BrainIAC](https://www.nature.com/articles/s41593-026-02202-6) evaluates a 3D brain-MRI encoder across glioma segmentation, IDH prediction, survival and other tasks. These studies indicate a volumetric and multimodal research direction, not a single universally best tumour detector. Direct evaluation would require co-registered patient-level volumes, sequence metadata, labels appropriate to the endpoint, external-site testing and substantially greater compute.

## Ten-paper verification status

| Tracker ID | Citation lead | Status / decision |
|---:|---|---|
| 10 | [Kaggle Brain Tumor MRI Dataset-based Explainable CNN (2025)](https://doi.org/10.1186/s40708-025-00257-y) | Full text extracted; direct comparator with unresolved patient/duplicate independence and qualitative XAI validation |
| 11 | [Transfer Learning for Accurate Brain Tumor Classification (2025)](https://link.springer.com/article/10.1007/s12672-025-02671-4) | Full text extracted; test-loss early stopping conflicts with final-test isolation |
| 12 | [Lightweight Transfer Learning Models for Multi-Class Brain Tumor Classification](https://doi.org/10.1007/s10278-025-01686-1) | Full text extracted; “patient-wise” CV basis is unclear because source patient identifiers are not documented |
| 13 | [Fine-Tuned Transfer Learning Model (2025)](https://www.mdpi.com/2075-1729/15/3/327) | Full text extracted; binary task, removed from four-class core shortlist |
| 14 | [Hybrid Attention and Clinical Explainability study](https://doi.org/10.1038/s41598-025-04591-3) | Full text extracted; year corrected to 2025; same-source internal study with qualitative single-expert XAI review |
| 15 | [Light Weight CNN (2025)](https://arxiv.org/abs/2504.21188) | Publisher/abstract verified; detailed full-PDF extraction and second review remain |
| 16 | [Deep CNN classification study (2021)](https://doi.org/10.18280/ts.380428) | Full text extracted; year corrected and binary tasks removed from four-class core shortlist |
| 31 | [Systematic review, 2020–2024](https://link.springer.com/article/10.1007/s10278-024-01283-8) | Publisher full text extracted for field breadth and methods context |
| 32 | [CNN techniques review, 2015–2022](https://www.mdpi.com/2075-4418/12/8/1850) | Full text extracted; DOI resolved to 10.3390/diagnostics12081850 |
| 34 | [Deep learning for brain tumour MRI review (2025)](https://doi.org/10.1038/s41698-024-00789-2) | Full text extracted; publication year corrected to 2025; supports taxonomy/site-shift limitations |

## Mandatory methodological anchors

- Reporting: [CLAIM 2024](https://pubs.rsna.org/doi/10.1148/ryai.240300) and [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378).
- Bias/applicability: [PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505).
- Taxonomy: [WHO CNS5](https://publications.iarc.who.int/Book-And-Report-Series/Who-Classification-Of-Tumours/Central-Nervous-System-Tumours-2021).
- Explanation validity: [Adebayo et al.](https://proceedings.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) and [Arun et al.](https://pubs.rsna.org/doi/10.1148/ryai.2021200267).
- Retrieval evaluation, if retained: [MIRAGE/MedRAG](https://aclanthology.org/2024.findings-acl.372/); answer faithfulness and citation precision still require blinded annotation.

## Work that cannot be completed from the current files alone

1. A genuinely independent external cohort with auditable provenance and compatible labels.
2. Participant-level separation or participant-bootstrap intervals without patient identifiers.
3. Age, sex, scanner, vendor or site subgroup analyses without source metadata.
4. Blinded neuroradiology/neuro-oncology review of Grad-CAM utility.
5. Second-reviewer full-text extraction and adjudication of the ten-paper queue.

These are explicit evidence requirements, not blank values to infer.
