# Research model card

Status date: 2026-07-15

## Model purpose

The project compares a prespecified ImageNet-initialized EfficientNet-B0 with a five-block custom CNN for four source-dataset labels from one 2D brain MRI image. The software is a methodological research artifact and is not a medical device, diagnostic system, triage tool, treatment adviser, or WHO CNS classifier.

## Inputs and outputs

- Input: one image resized to 224 × 224, converted to RGB, and normalized with ImageNet mean and standard deviation.
- Output: logits and temperature-scaled probabilities over `glioma`, `meningioma`, `no_tumor`, and `pituitary`.
- The 0.70 low-confidence flag is exploratory and not clinically validated.
- Inference fails closed if the expected provenance-bearing checkpoint is absent.

## Development protocol

- Locked manifest: `data/manifests/dataset_manifest.csv`.
- Primary architecture: EfficientNet-B0 with early layers frozen and final blocks/head trainable.
- Comparator: custom CNN.
- Loss: class-weighted cross-entropy with label smoothing; no weighted sampler.
- Optimizer: AdamW; cosine learning-rate schedule; early stopping on validation loss.
- Calibration: one positive scalar temperature fitted on validation logits only.
- Reproducibility: seed, manifest hash, experiment-config hash, class order, image size, hyperparameters, and run ID are stored in checkpoints and run summaries.

## Evaluation

Internal locked-test artifacts report macro-F1 and accuracy with duplicate/provenance-group bootstrap 95% intervals; per-class metrics; MCC; ROC- and PR-AUC; NLL, ECE, and Brier score before/after temperature scaling; and risk–coverage. Quantitative Grad-CAM evaluation includes repeatability, full-model randomization sensitivity, IoU, and pointing-game performance where BRISC masks map to locked internal-test images.

Numerical results must be copied only from generated artifacts under `outputs/metrics/`, `outputs/models/`, and `outputs/xai_evaluation/`. The live status and unresolved prerequisites are in `outputs/research_readiness.json`.

### Completed internal evidence

On 2,107 locked internal-test images representing 1,460 duplicate/provenance groups, the resource-constrained EfficientNet checkpoint achieved macro-F1 0.908 (95% CI 0.891–0.923), accuracy 0.907 (0.889–0.925), and MCC 0.876. The protocol-conformant custom CNN achieved macro-F1 0.833 (0.812–0.854), accuracy 0.839 (0.817–0.861), and MCC 0.787. Paired EfficientNet-minus-custom differences were 0.075 macro-F1 (0.057–0.094) and 0.068 accuracy (0.050–0.086). EfficientNet training was stopped after nine complete epochs for CPU-resource reasons, before its test performance was inspected, so it must not be described as a fully protocol-conformant primary run.

Validation-only temperature scaling worsened test ECE from 0.053 to 0.056 for EfficientNet and from 0.029 to 0.099 for the custom CNN, so calibrated-probability claims are not supported. Custom-CNN Grad-CAM mean localization IoU was 0.029 and pointing-game success was 6.4% on 1,045 mapped tumour images, so lesion-localization claims are not supported; EfficientNet XAI was not evaluated.

## External validation boundary

No qualifying independent external cohort is currently available. BRISC and BDNeuro-MRI v7 were rejected after exact cross-corpus reuse audits. Therefore, internal performance cannot be interpreted as clinical generalization, and the prespecified external macro-F1 endpoint remains unevaluated.

## Major limitations and risks

- Unknown patient identities and acquisition/subgroup metadata.
- Residual same-patient and source-confounding risk despite duplicate controls.
- Coarse folder labels and 2D sampling.
- Public-source preprocessing artifacts may be predictive.
- Calibration and uncertainty estimates remain source-specific.
- Grad-CAM is not a lesion detector; expert utility review is still required.
- Performance, fairness, and failure behavior are unknown in real clinical workflows and local populations.
