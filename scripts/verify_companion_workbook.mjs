import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = process.cwd();
const workbookPath = path.join(
  root,
  "outputs/literature_audit_2026-07-15/AI_NeuroOnco_Literature_Audit_and_Index.xlsx",
);
const qaDir = path.join(root, "outputs/literature_audit_2026-07-15/qa");
await fs.mkdir(qaDir, { recursive: true });

const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const overview = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 4000,
});
console.log(overview.ndjson);

for (const [sheetName, range] of [
  ["Audit Summary", "A1:H18"],
  ["Research Status", "A1:F20"],
  ["Paper Index", "A1:V20"],
  ["Gap Matrix", "A1:I17"],
  ["Missing Anchors", "A1:H20"],
  ["Paper Plan", "A1:H20"],
  ["Review Protocol", "A1:H30"],
  ["Track 3 Completion", "A1:J45"],
  ["Scope Coverage", "A1:H45"],
]) {
  const inspected = await workbook.inspect({
    kind: "table",
    range: `${sheetName}!${range}`,
    include: "values,formulas",
    tableMaxRows: 20,
    tableMaxCols: 22,
    tableMaxCellChars: 120,
    maxChars: 10000,
  });
  await fs.writeFile(
    path.join(qaDir, `${sheetName.replaceAll(" ", "_")}_inspect.ndjson`),
    inspected.ndjson,
  );
  const preview = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    path.join(qaDir, `${sheetName.replaceAll(" ", "_")}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
  maxChars: 6000,
});
console.log(errors.ndjson);
await fs.writeFile(path.join(qaDir, "formula_error_scan.ndjson"), errors.ndjson);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(workbookPath);
console.log(workbookPath);
