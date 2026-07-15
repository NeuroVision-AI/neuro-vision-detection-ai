# Development data card

Status date: 2026-07-15

## Intended research use

This corpus supports a retrospective four-class, single-2D-image classification proof-of-concept. The labels `glioma`, `meningioma`, `pituitary`, and `no_tumor` are source-dataset categories, not WHO CNS5 integrated diagnoses. The data must not be used to claim clinical diagnosis, treatment selection, prognosis, volumetric analysis, or population representativeness.

## Source and version

- Public source: Kaggle `masoudnickparvar/brain-tumor-mri-dataset`, reported as Version 2 at retrieval.
- Retrieval date: 2026-07-15.
- Distributed size: 7,200 images.
- Reported component sources: Figshare, SARTAJ, and Br35H.
- Source provenance and archive/manifest hashes: `data/manifests/source_provenance.json`.
- Exact analysis manifest: `data/manifests/dataset_manifest.csv`.

## Integrity processing

Every image was decoded and assigned a raw-byte SHA-256 and 64-bit perceptual difference hash before transformation. Patient identifiers were not available. Transitive components joined exact and perceptual duplicates, preserved the official test partition, and prevented groups from crossing final splits.

The pre-exclusion audit detected one cross-label perceptual component containing seven images. The entire component was excluded without visual relabelling. The failed audit and excluded paths remain available in `data/manifests/`.

## Final cohort

| Split | Images |
|---|---:|
| Train | 4,187 |
| Validation | 899 |
| Locked internal test | 2,107 |
| Total | 7,193 |

| Class | Train | Validation | Locked test | Total |
|---|---:|---:|---:|---:|
| Glioma | 1,149 | 247 | 403 | 1,799 |
| Meningioma | 1,066 | 229 | 499 | 1,794 |
| No tumour | 824 | 177 | 799 | 1,800 |
| Pituitary | 1,148 | 246 | 406 | 1,800 |

The final manifest has 6,160 duplicate/provenance groups, including 153 exact-duplicate and 574 perceptual-duplicate groups. No final group or exact hash crosses splits, all retained images decode, and no final group has conflicting labels. The preserved official test partition is disproportionately no-tumour, so macro/class-specific metrics are necessary alongside accuracy.

Class-distribution, raw-dimension, deterministic sampled-intensity and representative-image figures are generated under `outputs/data_quality/`, with machine-readable summaries in `data_quality_report.json`.

## Known limitations

- No patient, age, sex, scanner, sequence, institution, ethnicity, or acquisition metadata are distributed in the working source.
- Duplicate grouping cannot exclude cross-split reuse of different slices or appearances from the same patient.
- Folder labels may contain historical source errors and do not establish pathology or molecular ground truth.
- The mixed public source may encode source-specific borders, resolution, preprocessing, or acquisition cues.
- The official test partition is an internal source test, not independent clinical external validation.

## External-candidate audits

BRISC and BDNeuro-MRI v7 were rejected as independent external cohorts after material exact cross-corpus reuse was detected. Their audit artifacts are under `data/external_manifests/`. BRISC masks are used only for ancillary localization analysis on already-locked internal-test images.

## Governance

Confirm source licenses, consent/ethics statements, and institutional requirements before redistribution or submission. Raw images should be obtained from their original repositories; release only derived manifests and aggregate results where licensing permits.
