# Reproducible execution plan

This runbook produces the study artifacts without fabricating results. Commands assume the project virtual environment is active.

## 1. Confirm readiness

```bash
python scripts/check_research_readiness.py
```

The current expected result remains `paper_ready: false` until both prespecified models, XAI, an independently reviewed RAG benchmark, and a qualifying external cohort are complete. The development manifest and source audit are already present.

## 2. Prepare development data

For the current Kaggle layout:

```bash
python scripts/download_data.py
```

For a manually supplied source:

```bash
python scripts/prepare_research_data.py /path/to/source \
  --source-name public_dataset_version \
  --patient-pattern '(?P<patient>DATASET_SPECIFIC_REGEX)' \
  --exclude-conflicting-groups
```

Use conflict exclusion only after reviewing `dataset_audit_pre_exclusion.json`; it removes the entire contradictory component and records every removed path. Do not use `--overwrite` until the prior manifest and processed directory have been archived. Verify that `data/manifests/dataset_audit.json` reports `leakage_free`, `label_consistent`, and `all_images_decodable` as true.

## 3. Audit, prepare and lock external data

First run an independent cross-corpus audit. A candidate that overlaps the development source is not external evidence:

```bash
python scripts/audit_external_candidate.py \
  --candidate-root /path/to/external/source \
  --development-manifest data/manifests/dataset_manifest.csv \
  --source-name external_dataset_version \
  --output-dir data/external_manifests/candidate_overlap_audit
```

Proceed only if the audit, provenance, licensing/consent review, and label/sequence review pass.

```bash
python scripts/prepare_research_data.py /path/to/external/source \
  --source-name external_dataset_version \
  --output data/external_processed \
  --manifest-dir data/external_manifests \
  --patient-pattern '(?P<patient>DATASET_SPECIFIC_REGEX)'
```

Record the external data version and hashes before training. Do not inspect external performance while tuning.

## 4. Train both prespecified models

```bash
python -m src.train --model efficientnet
python -m src.train --model custom_cnn
```

Training saves provenance-bearing checkpoints. The EfficientNet run also fits validation-only temperature scaling and produces internal test metrics.

## 5. Locked external evaluation

```bash
python scripts/evaluate_external.py \
  --test-dir data/external_processed/test \
  --manifest data/external_manifests/dataset_manifest.csv \
  --checkpoint outputs/models/efficientnet/best_accuracy.pth \
  --model efficientnet \
  --temperature VALUE_FROM_CALIBRATION_JSON
```

Repeat for the custom CNN. Do not fit temperature on the external set.

## 6. Grad-CAM evaluation

Generate identically shaped trained, repeat, randomized-model, and mask arrays from the locked mapping:

```bash
python scripts/generate_xai_arrays.py \
  --mapping data/external_manifests/brisc_overlap_audit/xai_mask_mapping.csv \
  --checkpoint outputs/models/custom_cnn/best_accuracy.pth \
  --model custom_cnn \
  --candidate-root data/external_raw/brisc/brisc2025 \
  --output-root outputs/xai_arrays/custom_cnn
```

The generated structure is:

```text
outputs/xai_arrays/
  original/
  repeat/
  randomized/
  masks/          # optional
```

Then run:

```bash
python scripts/evaluate_xai.py outputs/xai_arrays/custom_cnn
```

## 7. RAG retrieval evaluation

After the row-level tracker is re-indexed:

```bash
python scripts/evaluate_rag.py --k 5
```

The supplied benchmark is a starting set and must be reviewed by a domain expert before being treated as a final test set.

## 8. Verification

```bash
python -m unittest discover -s tests -v
cd web && npm run build
python scripts/check_research_readiness.py --strict
```

The strict readiness check succeeds only when the core empirical artifacts exist.
