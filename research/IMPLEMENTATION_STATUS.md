# Research implementation status

Status date: 2026-07-15

## Completed in the repository

- The literature tracker has a companion audit/index workbook; the source workbook remains unchanged.
- The study scope, hypotheses, primary endpoint, split policy, models, calibration, statistical analysis, XAI checks, RAG evaluation, and claim boundaries are prespecified.
- Data preparation now preserves an official test partition and prevents transitive patient/exact-image/perceptual-duplicate groups from crossing splits.
- Training checkpoints carry provenance, class order, image size, seed, timestamp, and intended use. Temperature scaling is fitted on validation data only.
- Internal and external evaluators produce discrimination, calibration, class-level, bootstrap-confidence-interval, and selective-prediction artifacts.
- Inference fails closed when a trained checkpoint is absent. Reports and UI outputs are explicitly research-only and non-diagnostic.
- Literature rows retain record-level provenance through retrieval, and a retrieval benchmark/evaluator is available.
- Automated tests cover split leakage, materialization safety, evaluation metrics, checkpoint safety, RAG metrics, and XAI metrics.
- The current 7,200-image development source was acquired and independently audited. One seven-image cross-label perceptual component was excluded whole, leaving a leakage-controlled 7,193-image analysis manifest.
- The protocol-conformant custom-CNN run completed after 39 epochs with validation-only calibration and locked internal-test evaluation.
- A resource-constrained EfficientNet-B0 checkpoint completed validation-only calibration and one-time locked internal-test evaluation after the epoch-9 best-validation-accuracy checkpoint was frozen; the deviation is explicit in the run summary and deviations log.
- A paired duplicate/provenance-group bootstrap comparison of both models is complete; the image-level McNemar result is labelled exploratory.
- Quantitative Grad-CAM analysis completed on 1,045 unique locked internal-test tumour images with mapped BRISC masks.
- The paper-only, row-level literature index was rebuilt and the preliminary 10-question retrieval benchmark was executed.
- BRISC and BDNeuro-MRI v7 were independently cross-audited and rejected as external validation because of material exact reuse from the development source.
- A data card, model card, environment snapshot, deviations log, paper-facing results builder, and machine-readable readiness audit are present.
- The question-setting PDF now has an evidence-backed Q1-Q23 handoff, and the requested data-quality report includes class, dimension, sampled-intensity and representative-image outputs.

## Current empirical findings

- Custom CNN internal macro-F1: 0.833 (duplicate/provenance-group bootstrap 95% CI 0.812–0.854).
- Custom CNN internal accuracy: 0.839 (0.817–0.861).
- Resource-constrained EfficientNet internal macro-F1: 0.908 (0.891–0.923); accuracy: 0.907 (0.889–0.925).
- EfficientNet-minus-custom paired differences: macro-F1 0.075 (0.057–0.094); accuracy 0.068 (0.050–0.086).
- Validation-fitted temperature scaling worsened locked-test ECE from 0.053 to 0.056 for EfficientNet and from 0.029 to 0.099 for the custom CNN; probabilities are not claimed to be calibrated.
- Grad-CAM mean IoU: 0.029 (image-bootstrap 95% CI 0.026–0.032); pointing-game success: 6.4% (5.0–8.0%). These results do not support a lesion-localization claim.
- Corrected-companion, non-expert-reviewed RAG precision@5: 0.26; recall@5: 0.865; hit@5: 1.00; MRR: 0.867.

## Remaining evidence boundaries

- EfficientNet-B0 optimization was stopped after nine completed epochs for CPU-resource reasons; its internal result is a non-protocol-conformant comparator, not completion of the prespecified external primary endpoint.
- No candidate has passed the independent external-corpus audit, so external macro-F1 remains unevaluated.
- Patient identifiers and acquisition/subgroup metadata are absent from the development source.
- Blinded XAI expert utility review has not been performed.
- The RAG benchmark has not been independently reviewed by a domain expert.

These are evidence boundaries, not values to infer or fill. `scripts/check_research_readiness.py --strict` must remain non-zero until the core paper prerequisites are complete.
