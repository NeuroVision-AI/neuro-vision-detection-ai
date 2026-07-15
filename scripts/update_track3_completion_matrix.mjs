import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = process.cwd();
const workbookPath = path.join(
  root,
  "outputs/literature_audit_2026-07-15/AI_NeuroOnco_Literature_Audit_and_Index.xlsx",
);
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));

for (const name of ["Track 3 Completion", "Scope Coverage"]) {
  const existing = workbook.worksheets.items.find((item) => item.name === name);
  if (existing) existing.delete();
}

const colors = {
  navy: "#17324D",
  blue: "#2F75B5",
  paleBlue: "#D9EAF7",
  lightBlue: "#DDEBF7",
  green: "#E2F0D9",
  amber: "#FFF2CC",
  red: "#FCE4D6",
  grey: "#E7E6E6",
  white: "#FFFFFF",
  border: "#D9E1F2",
  text: "#17324D",
};

const baseFont = { name: "Carlito", size: 10, color: colors.text };
const titleFormat = {
  fill: colors.navy,
  font: { name: "Carlito", size: 18, bold: true, color: colors.white },
  verticalAlignment: "center",
};
const headerFormat = {
  fill: colors.blue,
  font: { name: "Carlito", size: 10, bold: true, color: colors.white },
  wrapText: true,
  verticalAlignment: "center",
  borders: { preset: "all", style: "thin", color: colors.border },
};
const bodyFormat = {
  font: baseFont,
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "all", style: "thin", color: colors.border },
};

const tasks = [
  ["P0", "Study framing", "Approve one coherent design: structured critical review plus original experimental component", "Partial", "Team document calls the project original experimental research; the objective also requires broad critical analysis", "Team formally approves the two-layer design and uses it consistently in title, abstract and methods", "Design statement frozen before final editing", "Whole team / Track 4", "Title, abstract, methods", "https://docs.google.com/document/d/1rS4p6vMjmXOci4Ny_Guxqjpdo18BBXiJWXcLVdX5kT4/edit?pli=1&tab=t.0"],
  ["P0", "Claim boundary", "Define intended use and distinguish dataset labels from clinical/WHO diagnosis", "Complete", "Protocol, README, model card and manuscript use research-only 2D dataset-label wording", "Retain the boundary during team editing", "No diagnostic, clinical-grade, volumetric or local-population claim", "Track 3 / Track 4", "Throughout", "research/PROTOCOL.md"],
  ["P1", "Search reproducibility", "Preserve database search strings, dates, hit counts, exports and deduplication", "Partial", "A 74-record tracker exists, but database-specific search logs are not evidenced", "Track 1 supplies an auditable log for PubMed, IEEE Xplore, Scopus, ScienceDirect, SpringerLink, Google Scholar and arXiv", "Search can be rerun and PRISMA-style counts reconcile", "Track 1", "Literature methods", "AI_NeuroOnco_Literature_Tracker.xlsx"],
  ["P0", "Literature verification", "Complete full-text extraction and independent adjudication of the core evidence", "Partial", "9/10 shortlisted records have first-reviewer extraction; one preprint is abstract/publisher verified", "Second reviewer extracts all ten and resolves disagreements; remaining PDF is fully checked", "Two-reviewer extraction table frozen", "Track 1 / Track 2", "Methods, discussion", "research/FULL_TEXT_EXTRACTION.md"],
  ["P1", "Architecture review", "Critically compare CNN, ResNet, DenseNet, VGG, EfficientNet, ViT, U-Net, YOLO and hybrid methods", "Partial", "Tracker title/metadata coverage is strong for CNN/U-Net/hybrid but sparse for named ResNet, DenseNet, VGG, EfficientNet, ViT and YOLO families", "Add full-text-supported architecture rows with endpoint, cohort, split unit, metrics, validation and limitations", "Every requested family is covered without implying it was implemented", "Track 1 / Track 2 / Track 3", "Introduction, related work", "research/TRACK3_COMPLETION_PLAN.md"],
  ["P1", "Dataset review", "Compare BraTS, exact TCIA collections, Figshare and Kaggle", "Partial", "Kaggle is fully audited; BraTS/TCIA are literature context; Figshare reuse risk is not yet tabulated", "Create one dataset table with unit, sequences, patients, labels/masks, license, endpoint and overlap risk", "No repository or redistribution is mislabeled as independent", "Track 1 / Track 2", "Data sources", "research/TRACK3_COMPLETION_PLAN.md"],
  ["P1", "Future models", "Cover 3D foundation models and multimodal vision-language models accurately", "Partial", "BrainIAC, BrainFound, Brainfound and Med-Gemini sources have been added to the synthesis/future-work plan", "Full-text verify capability, modality and validation claims; keep them future context rather than direct comparators", "Capability-specific wording passes second review", "Track 1 / Track 2", "Discussion, future directions", "https://doi.org/10.1016/j.patter.2026.101538"],
  ["P0", "Data acquisition", "Acquire and version the implemented development dataset", "Complete", "Kaggle version 2: 7,200 distributed images; source provenance and hash recorded", "Retain immutable provenance and license evidence", "Dataset/version/retrieval/license are reproducible", "Track 3", "Methods", "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"],
  ["P0", "Data integrity", "Audit decode quality, duplicates, label conflicts and source split reuse", "Complete", "7,193 retained; one seven-record cross-label component excluded; no final cross-split exact hash/group", "Retain failed pre-exclusion audit and final manifest", "Integrity audit passes without hidden relabeling", "Track 3", "Methods, cohort results", "data/manifests/dataset_audit.json"],
  ["P0", "Label validity", "Verify the limits of folder labels", "Complete", "No manual relabeling; folder labels are not pathology-verified and remain source-dataset labels", "Keep label-validity limitation and do not infer WHO diagnoses", "No claim exceeds available annotation evidence", "Track 3 / clinical reviewer", "Limitations", "research/DATA_CARD.md"],
  ["P0", "Preprocessing", "Freeze preprocessing and training-only augmentation", "Complete", "RGB replication, 224x224 resize, ImageNet normalization and training-only augmentation are documented", "Retain orientation/sequence limitations", "Transforms match frozen configuration", "Track 3", "Methods", "configs/experiment.yaml"],
  ["P0", "Partitioning", "Finalize leakage-controlled train/validation/locked-test split", "Complete", "4,187 / 899 / 2,107 images; duplicate/provenance components remain within one split", "Do not replace with a random image split", "Manifest and materialized folders reconcile", "Track 3", "Methods", "data/manifests/dataset_manifest.csv"],
  ["P0", "External validation", "Obtain a genuinely independent, compatible external cohort", "Blocked", "BRISC and BDNeuro-MRI v7 failed cross-corpus independence audits", "Acquire a provenance-auditable independent cohort; freeze both models before evaluation", "External macro-F1 and CI can be estimated without source reuse", "Track 1 / institutions", "Methods, results", "outputs/research_readiness.json"],
  ["P0", "Baseline model", "Implement and evaluate a reproducible custom CNN", "Complete", "Protocol-conformant custom CNN completed with locked-test metrics and provenance-bearing checkpoints", "Retain code, hashes, seed and run summary", "Run is reproducible from manifest/config", "Track 3", "Methods, results", "outputs/models/custom_cnn/run_summary.json"],
  ["P0", "Transfer model", "Complete the EfficientNet-B0 comparison under the prespecified stopping rule", "Partial", "Epoch-9 best-validation checkpoint was evaluated after a CPU-resource stop; result is explicitly non-protocol-conformant", "Rerun to early stopping/max epochs on suitable compute, or retain only as resource-constrained exploratory evidence", "Run summary reports protocol_conformant=true for confirmatory use", "Track 3 / compute", "Methods, results", "outputs/models/efficientnet/run_summary.json"],
  ["P2", "Additional baseline", "Consider one frozen ResNet or ViT baseline", "Not started", "Not required for the present two-model contribution", "Add only with a pre-test frozen plan and available compute", "No opportunistic leaderboard expansion after test inspection", "Track 3", "Optional supplement", "research/TRACK3_COMPLETION_PLAN.md"],
  ["P0", "Optimization protocol", "Document optimizer, loss, class handling, scheduler, stopping and seed", "Complete", "AdamW, class-weighted loss, cosine schedule, validation selection and seeds are frozen", "Retain EfficientNet deviation separately", "No test-guided hyperparameter selection", "Track 3", "Methods", "configs/experiment.yaml"],
  ["P0", "Classification evaluation", "Report endpoint-appropriate metrics and uncertainty", "Complete", "Accuracy, macro-F1, balanced accuracy, per-class metrics, MCC, ROC/PR-AUC and grouped CIs generated", "Copy values only from generated artifacts", "Tables reconcile with metrics JSON", "Track 3 / Track 2", "Results", "research/RESULTS_SUMMARY.md"],
  ["P0", "Model comparison", "Perform a paired comparison on the same locked cases", "Complete", "Grouped paired-bootstrap differences are complete; image-level McNemar is labelled exploratory", "Retain duplicate/provenance-group caveat", "Paired estimates and CIs are reported", "Track 3 / Track 2", "Results", "outputs/model_comparison/model_comparison.json"],
  ["P1", "Calibration", "Evaluate calibration and selective prediction", "Complete", "Validation-only temperatures, ECE, Brier, NLL and risk-coverage are generated; scaling worsened both models", "Do not call probabilities calibrated", "Before/after results and failure are reported", "Track 3", "Results, limitations", "research/RESULTS_SUMMARY.md"],
  ["P1", "Error analysis", "Analyze class failures, model disagreements and representative errors", "Partial", "Per-class results and confusion matrices exist; EfficientNet improved meningioma performance", "Add blinded/stratified disagreement examples; avoid patient/site analysis without metadata", "Failure table is prespecified and reproducible", "Track 3 / Track 2", "Results, discussion", "outputs/metrics"],
  ["P1", "Quantitative XAI", "Test Grad-CAM stability, randomization sensitivity and mask localization", "Complete", "1,045 mapped internal-test tumour cases evaluated; localization was poor", "Keep analysis limited to the custom CNN and internal reused-mask cohort", "No lesion-localization claim", "Track 3", "XAI results", "outputs/xai_evaluation/xai_metrics.json"],
  ["P0", "Expert XAI review", "Complete blinded clinical utility and misleading-attention review", "Blocked", "No completed expert-review metadata", "Neuroradiology/neuro-oncology reviewers score prespecified cases with adjudication", "Expert review file marked complete", "Clinical experts", "XAI results, limitations", "research/xai_expert_review_metadata.json"],
  ["P1", "Clinical applicability", "Critically assess intended use, workflow, external validity and deployment readiness", "Partial", "Claim boundaries and external-audit failures are documented", "Add clinician/workflow perspective and clearly separate technical proof-of-concept from diagnosis", "Clinical section contains no unsupported deployment claim", "Track 2 / Track 4 / experts", "Discussion", "research/MODEL_CARD.md"],
  ["P1", "Ethics and governance", "Cover consent/license, privacy, bias, transparency, hallucination and human oversight", "Partial", "Public-data and license limitations are documented; no patient-level data are stored", "Complete source-term/institutional review and add VLM hallucination/human-oversight discussion", "Ethics section is source-specific and reviewed", "Track 1 / Track 4", "Ethics, discussion", "research/DATA_CARD.md"],
  ["P1", "Fairness/subgroups", "Evaluate age, sex, site, scanner and vendor performance where possible", "Blocked", "The development source lacks these fields", "Obtain metadata or preserve the non-estimable boundary", "No fairness claim is made without data", "Track 1 / data owner", "Results, limitations", "outputs/research_readiness.json"],
  ["P1", "Results visualization", "Produce auditable tables and figures", "Complete", "Confusion matrices, ROC, calibration, risk-coverage, data-quality and XAI figures are generated", "Select figures before final layout; label internal/exploratory status", "Every figure traces to a generated artifact", "Track 3 / Track 4", "Results, supplement", "outputs"],
  ["P1", "Reproducibility", "Deliver code, tests, environment, data/model cards and run provenance", "Complete", "17 tests pass; web build passes; cards, protocol, hashes and manifests exist", "Upgrade runtime from Python 3.9 before long-term release", "Independent reviewer can audit the workflow", "Track 3", "Methods, availability", "research/IMPLEMENTATION_STATUS.md"],
  ["P0", "First manuscript", "Populate a claim-bounded first draft with completed evidence", "Complete", "Evidence-constrained manuscript draft exists", "Track 4 integrates the broad review tables and team authorship details", "All reported numbers reconcile", "Track 3 / Track 4", "Full manuscript", "research/MANUSCRIPT_DRAFT.md"],
  ["P0", "Internal review", "Conduct cross-track scientific, statistical and clinical review", "Not started", "No completed team sign-off is evidenced", "Track 1 checks sources, Track 2 checks synthesis/statistics, Track 3 checks artifacts, Track 4 checks reporting; clinical reviewer checks claims", "Signed issue/adjudication log has no unresolved P0 items", "Whole team", "Before final manuscript", "research/TRACK3_COMPLETION_PLAN.md"],
  ["P0", "Final manuscript", "Resolve blockers, freeze tables/figures/references and format for a target journal", "Not started", "Draft exists; paper_ready remains false", "Choose journal, apply checklist, incorporate reviews and freeze a versioned submission package", "Readiness audit has no unresolved manuscript P0 claim", "Track 4 / whole team", "Submission package", "outputs/research_readiness.json"],
  ["P0", "Submission", "Complete authorship, ethics, data/code statements and submit", "Not started", "No target journal or submission record", "Confirm authorship contributions/conflicts/funding and submit only after final approvals", "Submission receipt and archived package exist", "Lead / Track 4", "Administrative", "https://www.icmje.org/recommendations/"],
];

const completion = workbook.worksheets.add("Track 3 Completion");
completion.showGridLines = false;
completion.mergeCells("A1:J1");
completion.getRange("A1").values = [["Track 3 Completion Matrix — AI Integration and Development"]];
completion.getRange("A1:J1").format = titleFormat;
completion.getRange("A1:J1").format.rowHeight = 30;
completion.mergeCells("A3:J3");
completion.getRange("A3").values = [["Reconciles the team charter, broad research objective, source literature tracker and generated experimental evidence. Complete = artifact delivered; blocked = new people/data are required."]];
completion.getRange("A3:J3").format = {
  fill: colors.paleBlue,
  font: { ...baseFont, italic: true },
  wrapText: true,
  verticalAlignment: "center",
};
completion.getRange("A3:J3").format.rowHeight = 34;

const firstTaskRow = 11;
const lastTaskRow = firstTaskRow + tasks.length - 1;
completion.getRange("A5:J6").format = bodyFormat;
completion.getRange("A5").values = [["Total"]];
completion.getRange("B5").formulas = [[`=COUNTA(D${firstTaskRow}:D${lastTaskRow})`]];
completion.getRange("C5").values = [["Complete"]];
completion.getRange("D5").formulas = [[`=COUNTIF(D${firstTaskRow}:D${lastTaskRow},"Complete")`]];
completion.getRange("E5").values = [["Partial"]];
completion.getRange("F5").formulas = [[`=COUNTIF(D${firstTaskRow}:D${lastTaskRow},"Partial")`]];
completion.getRange("G5").values = [["Blocked"]];
completion.getRange("H5").formulas = [[`=COUNTIF(D${firstTaskRow}:D${lastTaskRow},"Blocked")`]];
completion.getRange("I5").values = [["Not started"]];
completion.getRange("J5").formulas = [[`=COUNTIF(D${firstTaskRow}:D${lastTaskRow},"Not started")`]];
completion.getRange("A6").values = [["Progress"]];
completion.getRange("B6").formulas = [["=IF(B5=0,0,(D5+0.5*F5)/B5)"]];
completion.getRange("B6").format.numberFormat = "0%";
completion.getRange("A5:J6").format.font = { name: "Carlito", size: 10, bold: true, color: colors.text };
completion.getRange("A5:J5").format.fill = colors.lightBlue;
completion.getRange("D5").format.fill = colors.green;
completion.getRange("F5").format.fill = colors.amber;
completion.getRange("H5").format.fill = colors.red;
completion.getRange("J5").format.fill = colors.grey;

completion.mergeCells("A8:J8");
completion.getRange("A8").values = [["Decision rule: review all requested architecture families, but implement only models compatible with the frozen endpoint and available annotations. The present experiment is 2D four-class classification, not segmentation, object detection or volumetric report generation."]];
completion.getRange("A8:J8").format = {
  fill: colors.amber,
  font: { ...baseFont, bold: true },
  wrapText: true,
  verticalAlignment: "center",
};
completion.getRange("A8:J8").format.rowHeight = 42;

const taskHeaders = ["Priority", "Workstream", "Required deliverable", "Status", "Evidence now", "Remaining action", "Completion criterion", "Dependency / owner", "Paper section", "Source / artifact"];
completion.getRange("A10:J10").values = [taskHeaders];
completion.getRange("A10:J10").format = headerFormat;
completion.getRange(`A${firstTaskRow}:J${lastTaskRow}`).values = tasks;
completion.getRange(`A${firstTaskRow}:J${lastTaskRow}`).format = bodyFormat;
completion.getRange(`D${firstTaskRow}:D${lastTaskRow}`).dataValidation = {
  rule: { type: "list", values: ["Complete", "Partial", "Blocked", "Not started"] },
};
completion.getRange(`A${firstTaskRow}:A${lastTaskRow}`).dataValidation = {
  rule: { type: "list", values: ["P0", "P1", "P2"] },
};
completion.getRange(`D${firstTaskRow}:D${lastTaskRow}`).conditionalFormats.add("containsText", { text: "Complete", format: { fill: colors.green } });
completion.getRange(`D${firstTaskRow}:D${lastTaskRow}`).conditionalFormats.add("containsText", { text: "Partial", format: { fill: colors.amber } });
completion.getRange(`D${firstTaskRow}:D${lastTaskRow}`).conditionalFormats.add("containsText", { text: "Blocked", format: { fill: colors.red } });
completion.getRange(`D${firstTaskRow}:D${lastTaskRow}`).conditionalFormats.add("containsText", { text: "Not started", format: { fill: colors.grey } });
completion.getRange(`A${firstTaskRow}:A${lastTaskRow}`).conditionalFormats.add("containsText", { text: "P0", format: { fill: colors.red, font: { bold: true, color: "#C00000" } } });
completion.getRange(`A${firstTaskRow}:A${lastTaskRow}`).conditionalFormats.add("containsText", { text: "P1", format: { fill: colors.amber, font: { bold: true } } });
completion.getRange(`A${firstTaskRow}:A${lastTaskRow}`).conditionalFormats.add("containsText", { text: "P2", format: { fill: colors.lightBlue } });
completion.freezePanes.freezeRows(10);
completion.freezePanes.freezeColumns(2);

const completionWidths = [58, 128, 260, 105, 285, 300, 250, 135, 120, 255];
completionWidths.forEach((width, index) => {
  completion.getRangeByIndexes(0, index, lastTaskRow, 1).format.columnWidthPx = width;
});
completion.getRange(`A${firstTaskRow}:J${lastTaskRow}`).format.rowHeight = 64;
completion.getRange(`C${firstTaskRow}:G${lastTaskRow}`).format.rowHeight = 72;

const scope = workbook.worksheets.add("Scope Coverage");
scope.showGridLines = false;
scope.mergeCells("A1:H1");
scope.getRange("A1").values = [["Research Objective Coverage — Architectures, Datasets, Databases and Metrics"]];
scope.getRange("A1:H1").format = titleFormat;
scope.getRange("A1:H1").format.rowHeight = 30;
scope.mergeCells("A3:H3");
scope.getRange("A3").values = [["Coverage is endpoint-specific: classification, segmentation, detection and report generation require different annotations and metrics. Review coverage does not imply implementation."]];
scope.getRange("A3:H3").format = { fill: colors.paleBlue, font: { ...baseFont, italic: true }, wrapText: true };
scope.getRange("A3:H3").format.rowHeight = 32;

const architectureRows = [
  ["Conventional ML / radiomics", "Review", "Feature-based classification, grading and prognosis", "Do not implement unless a frozen shallow baseline is approved", "Present in tracker context", "Extract features/cohort/validation details from full text", "AUC, macro-F1, calibration, external validation", "research/LITERATURE_SYNTHESIS.md"],
  ["Custom CNN", "Review + implement", "2D four-class classification", "Implemented comparator", "Complete", "Retain reproducibility evidence", "Macro-F1, accuracy, MCC, ROC/PR-AUC, CIs", "outputs/metrics/custom_cnn/research_metrics.json"],
  ["ResNet", "Review", "2D/3D classification or encoder", "Optional frozen baseline", "Mentioned in extracted comparator papers; no title-level tracker row", "Add explicit architecture comparison row", "Endpoint-specific discrimination and validation unit", "https://doi.org/10.1007/s10278-025-01686-1"],
  ["DenseNet", "Review", "Transfer-learning classification/feature encoder", "Review only", "No explicit title-level companion-index match", "Add one verified direct study or review extraction", "Macro-F1/AUC plus external and calibration evidence", "research/TRACK3_COMPLETION_PLAN.md"],
  ["VGG", "Review", "Transfer-learning classification", "Review only", "No explicit title-level companion-index match", "Add verified direct comparator evidence", "Macro-F1/AUC and split provenance", "research/TRACK3_COMPLETION_PLAN.md"],
  ["EfficientNet-B0", "Review + implement", "2D four-class classification", "Evaluated primary architecture", "Resource-constrained result", "Rerun to prespecified stopping for confirmatory status", "Macro-F1 0.908; accuracy 0.907 internally", "outputs/models/efficientnet/run_summary.json"],
  ["Vision Transformer / Swin", "Review", "Classification and volumetric encoding", "Review; optional preplanned baseline", "No explicit title-level companion-index match", "Add full-text evidence and data-efficiency discussion", "Endpoint metrics, compute, external validation", "research/TRACK3_COMPLETION_PLAN.md"],
  ["U-Net / nnU-Net", "Review", "Tumour segmentation", "Not compatible with current folder labels", "3 explicit title-level tracker records", "Compare 2D/3D inputs, masks and validation", "Dice, IoU, HD95, lesion-wise sensitivity", "https://doi.org/10.1038/s41592-020-01008-z"],
  ["YOLO-based models", "Review", "Bounding-box/object detection", "Not compatible without boxes", "No explicit title-level companion-index match", "Add a verified MRI detection source; keep detection separate from classification", "mAP@0.5, mAP@0.5:0.95, sensitivity, FP/image", "research/TRACK3_COMPLETION_PLAN.md"],
  ["Hybrid / ensemble", "Review", "CNN-attention, CNN-transformer or multimodal fusion", "Review only", "3 title-level tracker matches", "Compare added complexity with validation quality", "Endpoint metrics, ablation, external validation", "https://doi.org/10.1038/s41598-025-04591-3"],
  ["3D MRI foundation models", "Future direction", "Volumetric representation, segmentation, molecular/prognostic transfer", "Not implementable from current 2D folders", "New anchors added", "Full-text verify BrainIAC and BrainFound", "Dice/AUC/survival metrics with patient/external splits", "https://www.nature.com/articles/s41593-026-02202-6"],
  ["Multimodal VLMs", "Future direction", "Image-text alignment, reports and dialogue", "Requires paired scan/report data and expert rubric", "Brainfound and Med-Gemini added", "Distinguish CT report generation from MRI tumour detection", "Clinician factuality, significant error rate; text metrics secondary", "https://doi.org/10.1016/j.patter.2026.101538"],
];

scope.getRange("A5:H5").values = [["Architecture family", "Paper role", "Endpoint / data", "Experiment decision", "Current coverage", "Must complete", "Metrics / evidence", "Primary source / artifact"]];
scope.getRange("A5:H5").format = headerFormat;
scope.getRange(`A6:H${5 + architectureRows.length}`).values = architectureRows;
scope.getRange(`A6:H${5 + architectureRows.length}`).format = bodyFormat;

const datasetHeaderRow = 20;
const datasetRows = [
  ["Kaggle Brain Tumor MRI v2", "7,200 distributed 2D images; four folder labels", "Classification", "Implemented development/internal source test", "Complete", "No patients, sequences, site/scanner or WHO-integrated labels", "Do not call the locked test external", "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"],
  ["Figshare brain MRI", "Public 2D slices; commonly redistributed", "Classification", "Review/source-provenance context", "Partial", "Potential overlap with compiled Kaggle collections", "Cross-audit before any external claim", "https://figshare.com/articles/dataset/brain_tumor_dataset/1512427"],
  ["BraTS", "Patient-level multimodal MRI volumes and tumour masks", "Segmentation; selected subtyping/prognosis tasks", "Review; separate future volumetric protocol", "Not acquired", "Label/task mismatch with current four-class folders", "Name challenge year, sequences, patients and license", "https://doi.org/10.1109/TMI.2014.2377694"],
  ["TCIA collections", "Repository of heterogeneous named collections", "Varies by collection", "Review; name exact collection if used", "Not acquired", "TCIA is not one dataset", "Record collection/version/access/patients/sequences/endpoints", "https://www.cancerimagingarchive.net/"],
  ["BRISC", "6,000 four-class records plus masks", "Classification/XAI", "Rejected as external; masks used on locked internal cases", "Audited", "4,781/6,000 classification records exactly overlap development", "Keep as internal-test ancillary XAI only", "https://www.nature.com/articles/s41597-026-06753-y"],
  ["BDNeuro-MRI v7", "5,941 four-class candidate images", "Classification", "Rejected as external", "Audited", "Material exact/near overlap and incomplete distributed provenance text", "Do not calculate external performance", "https://data.mendeley.com/datasets/zwr4ntf94j/7"],
];
scope.getRange(`A${datasetHeaderRow}:H${datasetHeaderRow}`).values = [["Dataset / source", "Unit and content", "Natural endpoint", "Role in this paper", "Current status", "Key limitation", "Completion rule", "Primary source"]];
scope.getRange(`A${datasetHeaderRow}:H${datasetHeaderRow}`).format = headerFormat;
scope.getRange(`A${datasetHeaderRow + 1}:H${datasetHeaderRow + datasetRows.length}`).values = datasetRows;
scope.getRange(`A${datasetHeaderRow + 1}:H${datasetHeaderRow + datasetRows.length}`).format = bodyFormat;

const dbHeaderRow = 29;
const databaseRows = [
  ["PubMed", "Biomedical indexing and clinical literature", "Search log not evidenced", "Exact query, date, filters, hit count and export", "Track 1", "Needed before systematic/scoping-review language", "https://pubmed.ncbi.nlm.nih.gov/", "Partial"],
  ["IEEE Xplore", "Engineering and imaging methods", "Search log not evidenced", "Exact query, date, filters, hit count and export", "Track 1", "Deduplicate against other databases", "https://ieeexplore.ieee.org/", "Partial"],
  ["Scopus", "Cross-disciplinary citation database", "Search log not evidenced", "Exact query, date, filters, hit count and export", "Track 1", "Record institutional access and export format", "https://www.scopus.com/", "Partial"],
  ["ScienceDirect", "Elsevier journals", "Publisher links appear in tracker", "Preserve query/date/hits; avoid treating publisher search as an independent index", "Track 1", "Use as full-text source where licensed", "https://www.sciencedirect.com/", "Partial"],
  ["SpringerLink", "Springer/Nature platform", "Publisher links appear in tracker", "Preserve query/date/hits; distinguish platform from database indexing", "Track 1", "Use for full text and publisher metadata", "https://link.springer.com/", "Partial"],
  ["Google Scholar", "Broad discovery and citation chaining", "Search log not evidenced", "Record query/date and screening cutoff; results are unstable", "Track 1", "Use for supplementary discovery, not sole reproducible source", "https://scholar.google.com/", "Partial"],
  ["arXiv", "Recent preprints", "Preprints present in tracker", "Record query/date/version and later peer-reviewed publication status", "Track 1", "Do not equate preprint with peer-reviewed evidence", "https://arxiv.org/", "Partial"],
];
scope.getRange(`A${dbHeaderRow}:H${dbHeaderRow}`).values = [["Database", "Purpose", "Evidence now", "Required search record", "Owner", "Use rule", "URL", "Status"]];
scope.getRange(`A${dbHeaderRow}:H${dbHeaderRow}`).format = headerFormat;
scope.getRange(`A${dbHeaderRow + 1}:H${dbHeaderRow + databaseRows.length}`).values = databaseRows;
scope.getRange(`A${dbHeaderRow + 1}:H${dbHeaderRow + databaseRows.length}`).format = bodyFormat;
scope.getRange(`H${dbHeaderRow + 1}:H${dbHeaderRow + databaseRows.length}`).conditionalFormats.add("containsText", { text: "Partial", format: { fill: colors.amber } });

const metricHeaderRow = 39;
const metricRows = [
  ["Four-class classification", "Macro-F1 primary; accuracy, balanced accuracy, per-class precision/recall/specificity/F1, MCC, macro ROC-AUC/PR-AUC", "Grouped 95% bootstrap CIs", "Complete internally", "External endpoint remains missing", "research/RESULTS_SUMMARY.md", "Do not compare directly with segmentation/detection metrics", "Track 3 / Track 2"],
  ["Calibration", "ECE, Brier score, NLL and calibration curve", "Fit temperature on validation only", "Complete", "Scaling worsened both models", "research/RESULTS_SUMMARY.md", "Report calibration failure honestly", "Track 3"],
  ["Selective prediction", "Risk-coverage and exploratory confidence threshold", "No threshold tuning on test", "Complete descriptively", "Not clinically validated", "outputs/metrics", "Do not present abstention as a clinical safety system", "Track 3"],
  ["Segmentation", "Dice, IoU/Jaccard, HD95, sensitivity and lesion-wise outcomes", "Patient-level volume split", "Review only", "No compatible development masks/volumes", "research/TRACK3_COMPLETION_PLAN.md", "Use only in segmentation literature/future protocol", "Track 2 / Track 3"],
  ["Object detection", "mAP@0.5, mAP@0.5:0.95, sensitivity, precision and FP/image or FP/scan", "Box-level annotations and independent cases", "Review only", "No boxes", "research/TRACK3_COMPLETION_PLAN.md", "Do not call four-class image classification object detection", "Track 2 / Track 3"],
  ["Report-generating VLM", "Clinician factuality/completeness, significant-error rate and task-appropriate text metrics", "Blinded expert review on independent cases", "Future direction", "No paired reports/volumes", "https://arxiv.org/abs/2405.03162", "BLEU/ROUGE alone are insufficient", "Clinical experts / Track 3"],
];
scope.getRange(`A${metricHeaderRow}:H${metricHeaderRow}`).values = [["Endpoint", "Required metrics", "Validation unit", "Current status", "Boundary", "Evidence / source", "Interpretation rule", "Owner"]];
scope.getRange(`A${metricHeaderRow}:H${metricHeaderRow}`).format = headerFormat;
scope.getRange(`A${metricHeaderRow + 1}:H${metricHeaderRow + metricRows.length}`).values = metricRows;
scope.getRange(`A${metricHeaderRow + 1}:H${metricHeaderRow + metricRows.length}`).format = bodyFormat;

scope.freezePanes.freezeRows(5);
scope.freezePanes.freezeColumns(1);
const scopeLastRow = metricHeaderRow + metricRows.length;
const scopeWidths = [155, 185, 210, 210, 175, 245, 240, 240];
scopeWidths.forEach((width, index) => {
  scope.getRangeByIndexes(0, index, scopeLastRow, 1).format.columnWidthPx = width;
});
scope.getRange(`A6:H${scopeLastRow}`).format.rowHeight = 60;

const missing = workbook.worksheets.getItem("Missing Anchors");
missing.getRange("A19:F22").values = [
  ["BrainIAC 2026", "3D brain-MRI foundation model", "Future direction", "General-purpose 3D brain-MRI encoder evaluated for glioma segmentation, IDH, survival and other tasks", "https://www.nature.com/articles/s41593-026-02202-6", "Full-text verify and cite as volumetric future context"],
  ["BrainFound 2025/26", "3D brain-MRI self-supervised model", "Future direction", "Sequential-slice volumetric representation for downstream detection and segmentation; distinct from Brainfound", "https://arxiv.org/abs/2510.23415", "Verify final publication/version and extract task-specific results"],
  ["Brainfound 2026", "Multimodal brain imaging foundation model", "Future direction", "Brain CT/MRI plus language across diagnosis, segmentation, enhancement, reports and dialogue", "https://doi.org/10.1016/j.patter.2026.101538", "Full-text extract modality, cohorts, task units and expert evaluation"],
  ["Med-Gemini 2024", "Medical multimodal model", "Future direction", "Med-Gemini-3D demonstrates 3D CT report generation; it is not direct evidence of brain-tumour MRI detection", "https://arxiv.org/abs/2405.03162", "Use capability-specific wording and report the expert-quality limitation"],
];
missing.getRange("A19:F22").format = bodyFormat;
missing.getRange("A19:F22").format.rowHeight = 58;
missing.getRange("A3").values = [["High-value additions that support reporting, bias appraisal, taxonomy, XAI, RAG evaluation, external-candidate auditing, and contemporary volumetric/multimodal future directions."]];

const summary = workbook.worksheets.getItem("Audit Summary");
summary.getRange("A18").values = [["Workbook map: Research Status = generated evidence; Track 3 Completion = actionable definition of done; Scope Coverage = architectures/datasets/databases/metrics; Paper Index = normalized source records; Gap Matrix = risks; Missing Anchors = additions; Paper Plan and Review Protocol = study workflow."]];

const completionInspect = await workbook.inspect({
  kind: "table",
  range: `Track 3 Completion!A1:J${lastTaskRow}`,
  include: "values,formulas",
  tableMaxRows: 40,
  tableMaxCols: 10,
  tableMaxCellChars: 140,
  maxChars: 18000,
});
await fs.writeFile(
  path.join(root, "outputs/literature_audit_2026-07-15/track3_completion_inspect.ndjson"),
  completionInspect.ndjson,
);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "Track 3 update formula-error scan",
  maxChars: 5000,
});
console.log(errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(workbookPath);
console.log(workbookPath);
