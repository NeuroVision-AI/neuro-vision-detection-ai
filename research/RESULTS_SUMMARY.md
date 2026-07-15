# Generated research results summary

> Values below are copied from generated artifacts. The external primary endpoint remains unevaluated.
> EfficientNet was evaluated from the frozen epoch-9 best-validation-accuracy checkpoint after a CPU-resource stop and is not a protocol-conformant completed primary run.

## Internal locked-test results

| Model | Images / groups | Accuracy (95% CI) | Macro-F1 (95% CI) | MCC | ECE before → after | Temperature |
|---|---:|---:|---:|---:|---:|---:|
| efficientnet | 2107 / 1460 | 0.907 (0.889–0.925) | 0.908 (0.891–0.923) | 0.876 | 0.053 → 0.056 | 0.608 |
| custom_cnn | 2107 / 1460 | 0.839 (0.817–0.861) | 0.833 (0.812–0.854) | 0.787 | 0.029 → 0.099 | 0.550 |

Duplicate/provenance components—not verified patients—were the bootstrap unit because patient identifiers were unavailable.

## Paired internal comparison

EfficientNet minus custom-CNN accuracy: 0.068 (group-bootstrap 95% CI 0.050 to 0.086); macro-F1 difference: 0.075 (95% CI 0.057 to 0.094). Exploratory image-level exact McNemar p=2.584e-29.

## External validation

Not evaluated. BRISC and BDNeuro-MRI v7 were rejected before performance evaluation because material exact cross-corpus reuse was detected.

## Explainability and RAG

- Quantitative Grad-CAM cases: 1045.
- Preliminary RAG recall@5: 0.865; hit@5: 1.000; MRR: 0.867.
- XAI expert review and RAG benchmark expert review remain incomplete.
