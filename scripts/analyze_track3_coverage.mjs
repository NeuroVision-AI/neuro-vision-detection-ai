import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = "outputs/literature_audit_2026-07-15/AI_NeuroOnco_Literature_Audit_and_Index.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const sheet = workbook.worksheets.getItem("Paper Index");
const rows = sheet.getRange("A6:V79").values;

const patterns = {
  CNN: /\bcnn\b|convolutional neural network/i,
  ResNet: /resnet/i,
  DenseNet: /densenet/i,
  VGG: /\bvgg(?:16|19)?\b/i,
  EfficientNet: /efficientnet/i,
  ViT: /vision transformer|\bvit\b|swin transformer/i,
  UNet: /u[- ]?net/i,
  YOLO: /\byolo\b/i,
  Hybrid: /hybrid|ensemble|fusion/i,
  FoundationModel: /foundation model|brainfound|brainiac|med-gemini/i,
};

const output = {};
for (const [label, pattern] of Object.entries(patterns)) {
  const matches = rows
    .filter((row) => row[0] !== null && row[0] !== undefined)
    .filter((row) => pattern.test(row.map((value) => value ?? "").join(" ")))
    .map((row) => ({ id: row[0], title: row[1], year: row[2], tier: row[16] }));
  output[label] = { count: matches.length, matches };
}

console.log(JSON.stringify(output, null, 2));
