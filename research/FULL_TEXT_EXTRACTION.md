# Core literature full-text extraction

Extraction date: 2026-07-15
Scope: the ten-paper shortlist from the companion literature audit.
Reviewer status: first-reviewer extraction only; independent second-reviewer adjudication remains required before submission.

## Extraction table

| Tracker ID | Verification | Cohort / split | Performance evidence | Methodological relevance and limitation | Paper action |
|---:|---|---|---|---|---|
| 10 | Full text verified | 7,023-image Nickparvar collection split 5,618/702/702 (80/10/10); a 3,264-image “NeuroMRI” collection used as an unseen test | Reports 99.21% accuracy/F1 on dataset 1 and 94.72% on dataset 2 | No patient or duplicate audit found; XAI validation is qualitative Grad-CAM/SHAP/LIME inspection rather than mask localization or randomization testing | Cite as a direct comparator, but qualify the external-independence and XAI-validity uncertainty |
| 11 | Full text verified | 4,517 Kaggle images; random 80/20 image split; training-set cross-validation; no separate validation set | GoogleNet 99.2% accuracy; MobileNetV2 98.7%; AlexNet 96.5% | The paper says the test set was held for final evaluation but also reports early stopping based on testing loss. No patient IDs, duplicate audit, calibration or external validation were found | Use as a contrast for validation leakage and claim-scope discipline, not as a directly comparable external result |
| 12 | Full text verified | 7,023 Nickparvar images; 5,712 labelled training and 1,311 “validation”; claims fivefold patient-wise CV | Pretrained ResNet models 98.5–99.2%; custom CNN 87.03% | The public dataset description supplies images, not documented patient identifiers, so the basis of “patient-wise” folds is unclear. No duplicate audit, calibration or external evaluation was found | Strong architecture comparator; explicitly state that split-unit comparability is unresolved |
| 13 | Full text verified | 3,762 images from a different binary Kaggle tumor/no-tumor collection; random 80/10/10 split | Xception 96.11% accuracy | This is not the four-class Nickparvar task and should not have been in the core four-class shortlist | Reclassify as supporting transfer-learning literature only |
| 14 | Full text verified | 7,023-image Nickparvar collection; official 5,712/1,311 structure was reworked into 5,688 train, 632 validation and 703 test | Reports 99% test accuracy and AUC 1.00 | Published in 2025, not 2026. No patient or duplicate audit and no calibration were found. Grad-CAM review is qualitative opinion from one consultant radiologist; the authors themselves state external validation is still needed | Cite as a high-performance same-source comparator with non-equivalent leakage/XAI controls |
| 15 | Abstract and publisher record verified; full PDF extraction incomplete | Public four-class collection; fivefold CV described in abstract | Reports 98.78% accuracy | Six-page arXiv preprint; patient/duplicate separation, calibration and external testing remain unverified | Retain in queue; do not use detailed comparison claims yet |
| 16 | Full text verified | Two binary tasks: 253 Kaggle tumor/no-tumor images and 332 BraTS low-/high-grade images; random 70/30 splits | Reports 99.0% and 98.86% accuracy with augmentation | Published in 2021, not 2023; not the four-class task. No calibration or external-site evaluation | Remove from four-class core shortlist; retain only as historical transfer-learning context |
| 31 | Publisher full-text record verified | Systematic review of studies published 2020–2024; extraction form reported | Qualitative synthesis of architectures/datasets | No calibration signal was found in the accessible full text; broad scope mixes detection, classification and segmentation | Cite for field breadth only; do not use it to establish patient-level validation quality without deeper table-level checking |
| 32 | Full text verified | Systematic review of CNN classification studies from 2015–2022 | Reports inconsistent validation methods, metrics and training data across studies | DOI resolved as 10.3390/diagnostics12081850; directly supports cautious cross-study comparison and the need for clinical-usability evidence | Cite as the main review supporting non-comparability of headline accuracy |
| 34 | Full text verified | Narrative review of deep learning for brain-tumour MRI segmentation/classification | No single benchmark estimate | Published 3 January 2025, not 2024. It emphasizes WHO-integrated taxonomy, 3D/2D trade-offs, dataset heterogeneity and site generalization | Cite for clinical taxonomy, volumetric limitations and site-shift context |

## Corrected shortlist interpretation

The full-text audit narrows the genuinely comparable evidence. Tracker IDs 13 and 16 should be removed from the core four-class shortlist because they evaluate binary tasks. IDs 14 and 34 require publication-year correction to 2025, ID 16 to 2021, and ID 32 now has a resolved DOI. ID 15 remains only partially verified.

Across the verified four-class primary studies, very high internal image-level results are common, but duplicate audits, calibration and genuinely independent external evaluation were not identified. Some validation descriptions are internally difficult to reconcile with the distributed source metadata. This makes the current study’s chief contribution its provenance and claim discipline rather than a new headline architecture.

## Paper-ready synthesis

Direct numerical comparison should be limited to studies using the same four labels and should carry a split-unit qualifier. The current internal results can be discussed alongside IDs 10, 12 and 14, but not presented as a conventional leaderboard because:

1. the present analysis excludes a cross-label perceptual component and groups observable reuse transitively;
2. patient identifiers are unavailable and patient-wise independence cannot be verified;
3. BRISC and BDNeuro-MRI v7 materially overlap the development source;
4. validation-only temperature scaling and quantitative Grad-CAM checks expose failures hidden by accuracy alone; and
5. no external macro-F1 can be estimated until an independent cohort is available.

## Primary records checked

- ID 10: [Iftikhar et al., Brain Informatics (2025)](https://doi.org/10.1186/s40708-025-00257-y)
- ID 11: [Khan et al., Discover Oncology (2025)](https://link.springer.com/article/10.1007/s12672-025-02671-4)
- ID 12: [Gorenshtein et al., Journal of Imaging Informatics in Medicine](https://link.springer.com/article/10.1007/s10278-025-01686-1)
- ID 13: [Rastogi et al., Life (2025)](https://www.mdpi.com/2075-1729/15/3/327)
- ID 14: [Aiya et al., Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-04591-3)
- ID 15: [Alemayehu, arXiv (2025)](https://arxiv.org/abs/2504.21188)
- ID 16: [Kuraparthi et al., Traitement du Signal (2021)](https://www.iieta.org/journals/ts/paper/10.18280/ts.380428)
- ID 31: [Bouhafra and El Bahi, systematic review](https://link.springer.com/article/10.1007/s10278-024-01283-8)
- ID 32: [Xie et al., Diagnostics (2022)](https://www.mdpi.com/2075-4418/12/8/1850)
- ID 34: [Dorfner et al., npj Precision Oncology (2025)](https://www.nature.com/articles/s41698-024-00789-2)
