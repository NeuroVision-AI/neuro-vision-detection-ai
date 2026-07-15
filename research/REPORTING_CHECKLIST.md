# Reporting and bias checklist map

| Requirement | Status | Project evidence / action |
|---|---|---|
| Intended use and users | Complete in protocol | Research-only 2D public-data proof-of-concept |
| Clinical role and decision pathway | Not applicable to current study | No clinical workflow claim |
| Data source, version and license | Complete for development and audited candidates | Machine-readable source-provenance artifacts and data card; confirm institutional license interpretation before submission |
| Participant and image counts | Partial | Image counts are complete; patient counts are not estimable from distributed metadata |
| Acquisition/scanner/sequence details | Not estimable | Source metadata are absent; missingness is disclosed rather than imputed |
| Split unit and leakage prevention | Implemented | Patient/exact/perceptual transitive grouping |
| Official test preservation | Implemented | `src/data_integrity.py` and downloader |
| Preprocessing and augmentation | Implemented/documented | `src/config.py`, experiment YAML |
| Model architecture and initialization | Implemented/documented | EfficientNet-B0 and custom CNN |
| Hyperparameter selection | Complete | Frozen in `configs/experiment.yaml`; deviations are logged |
| Primary endpoint | Complete in protocol | External macro-F1 with grouped bootstrap CI |
| Comparator | Complete in protocol | Custom CNN under identical splits |
| Calibration | Implemented | Validation-only temperature scaling, ECE/Brier/NLL |
| External validation | Blocked by candidate independence | BRISC and BDNeuro-MRI v7 failed reuse audits; obtain a genuinely independent cohort before claiming generalization |
| Subgroup analysis | Not estimable | Age/sex/site/vendor metadata are unavailable; no subgroup claim is made |
| Explainability validation | Partial | Quantitative repeatability, randomization, IoU and pointing-game complete; blinded expert utility review remains |
| Failure analysis | Partial | Per-class errors, calibration failure and XAI performance strata are reported; expert qualitative review remains |
| Code/data availability | Substantially complete | Release code, manifests and aggregate artifacts subject to source licenses; raw images remain at original repositories |
| CLAIM 2024 mapping | Partial | Core intended-use, provenance, split, evaluation and reproducibility fields are represented; complete journal-form checklist before submission |
| TRIPOD+AI mapping | Partial | Development/evaluation reporting is represented; external validation and participant-level fields remain unavailable |
| PROBAST+AI assessment | Partial | Risks are explicitly recorded; independent reviewer assessment is still required before submission |
| WHO CNS5 terminology | Complete in claim boundary | Dataset labels are not integrated diagnoses |
