#!/usr/bin/env python3
"""
Generate a comprehensive, publication-grade PDF report summarizing the
AI NeuroOnco datasets, dataset sizes, trained model accuracy, and benchmark evaluations.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Set Matplotlib config dir to avoid writable directory warnings
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_ai_neuroonco"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PDF_OUTPUT_PATH = REPORTS_DIR / "AI_NeuroOnco_Results_Report.pdf"
ROOT_PDF_COPY = ROOT_DIR / "AI_NeuroOnco_Results_Report.pdf"
CHART_DIR = ROOT_DIR / "tmp" / "report_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE HIGH-RES CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def generate_charts():
    # Chart 1: Model Accuracy & F1 Comparison
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=300)
    models = ["EfficientNet-B0\n(Transfer Learning)", "Custom CNN\n(~2M Params)", "DenseNet121\n(Fine-Tuned Ref)", "ResNet50\n(Fine-Tuned Ref)"]
    acc = [90.75, 83.91, 98.31, 97.08]
    f1 = [90.78, 83.27, 98.31, 97.08]

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, acc, width, label="Test Accuracy (%)", color="#2563EB", edgecolor="#1D4ED8", linewidth=1)
    rects2 = ax.bar(x + width/2, f1, width, label="Macro F1-Score (%)", color="#10B981", edgecolor="#059669", linewidth=1)

    ax.set_ylabel("Percentage (%)", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title("Model Performance Comparison (Accuracy & Macro F1)", fontsize=11, fontweight="bold", color="#0F172A", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5, color="#334155")
    ax.set_ylim(70, 103)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#94A3B8")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#1E3A8A")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#065F46")

    plt.tight_layout()
    chart1_path = CHART_DIR / "chart_model_comparison.png"
    plt.savefig(chart1_path, bbox_inches="tight")
    plt.close()

    # Chart 2: Class & Split Distribution
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=300)
    classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    train_counts = [1149, 1066, 824, 1148]
    val_counts = [247, 229, 177, 246]
    test_counts = [403, 499, 799, 406]

    x = np.arange(len(classes))
    width = 0.55

    p1 = ax.bar(x, train_counts, width, label="Train Split (58.2%)", color="#3B82F6", edgecolor="#2563EB")
    p2 = ax.bar(x, val_counts, width, bottom=train_counts, label="Validation Split (12.5%)", color="#F59E0B", edgecolor="#D97706")
    bottom_val = [t + v for t, v in zip(train_counts, val_counts)]
    p3 = ax.bar(x, test_counts, width, bottom=bottom_val, label="Locked Test Split (29.3%)", color="#8B5CF6", edgecolor="#7C3AED")

    ax.set_ylabel("Number of Images", fontsize=10, fontweight="bold", color="#1E293B")
    ax.set_title("Dataset Composition Across Diagnostic Classes & Splits (Total = 7,193)", fontsize=11, fontweight="bold", color="#0F172A", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9, fontweight="bold", color="#334155")
    ax.set_ylim(0, 2200)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#94A3B8")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, (t, v, te) in enumerate(zip(train_counts, val_counts, test_counts)):
        tot = t + v + te
        ax.annotate(f"Total: {tot:,}", xy=(i, tot), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#0F172A")

    plt.tight_layout()
    chart2_path = CHART_DIR / "chart_dataset_distribution.png"
    plt.savefig(chart2_path, bbox_inches="tight")
    plt.close()

    return chart1_path, chart2_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. CUSTOM CANVAS WITH NUMBERED PAGES & RUNNING HEADERS
# ─────────────────────────────────────────────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "AI NeuroOnco — Research Study & Performance Evaluation Report")
            self.drawRightString(612 - 54, 755, "CONFIDENTIAL & RESEARCH USE ONLY")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 747, 612 - 54, 747)

        # Running Footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        self.drawString(54, 32, "Track 3: AI Integration, Computer Vision & Multimodal Medical RAG")
        self.drawRightString(612 - 54, 32, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD PDF DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────
def build_pdf():
    chart1_path, chart2_path = generate_charts()

    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom typography
    c_navy = colors.HexColor("#0F2942")
    c_slate = colors.HexColor("#1E293B")
    c_muted = colors.HexColor("#64748B")
    c_blue = colors.HexColor("#2563EB")
    c_emerald = colors.HexColor("#059669")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=c_navy,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        textColor=c_muted,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=c_navy,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#334155")
    )

    body_bold = ParagraphStyle(
        "BodyBold_Custom",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    callout_style = ParagraphStyle(
        "Callout_Custom",
        parent=body_style,
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#1E293B")
    )

    tbl_header = ParagraphStyle(
        "TblHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    tbl_cell = ParagraphStyle(
        "TblCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )

    tbl_cell_bold = ParagraphStyle(
        "TblCellBold",
        parent=tbl_cell,
        fontName="Helvetica-Bold"
    )

    tbl_cell_center = ParagraphStyle(
        "TblCellCenter",
        parent=tbl_cell,
        alignment=1
    )

    tbl_cell_right = ParagraphStyle(
        "TblCellRight",
        parent=tbl_cell,
        alignment=2
    )

    story = []

    # ── HEADER BLOCK ──
    story.append(Paragraph("AI NeuroOnco: Comprehensive Results Report", title_style))
    story.append(Paragraph("Brain Tumor MRI Classification, Clinical Benchmarks & Multimodal RAG Performance", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_blue, spaceBefore=0, spaceAfter=12))

    # Metadata Summary Table
    now_str = datetime.now().strftime("%B %d, %Y")
    meta_data = [
        [
            Paragraph("<b>Study Phase:</b> Track 3 (AI Integration)", body_style),
            Paragraph(f"<b>Report Date:</b> {now_str}", body_style)
        ],
        [
            Paragraph("<b>Diagnostic Task:</b> 4-Class Brain Tumor MRI", body_style),
            Paragraph("<b>Modality:</b> Axial T1/T2 & Volumetric DICOM", body_style)
        ],
        [
            Paragraph("<b>Primary Architecture:</b> EfficientNet-B0 (TL)", body_style),
            Paragraph("<b>Evaluation Cohort:</b> 2,107 Locked Test Images", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # ── SECTION 1: DATASETS & PROVENANCE ──
    story.append(Paragraph("1. Datasets, Origins & Data Governance", h1_style))
    p_datasets = (
        "The AI NeuroOnco research platform utilizes curated and audited neuro-imaging cohorts for "
        "model training, internal locked-set testing, and clinical retrieval grounding:"
    )
    story.append(Paragraph(p_datasets, body_style))
    story.append(Spacer(1, 6))

    ds_summary_data = [
        [
            Paragraph("Dataset Reference", tbl_header),
            Paragraph("Type / Role", tbl_header),
            Paragraph("Sample Size", tbl_header),
            Paragraph("Description & Components", tbl_header)
        ],
        [
            Paragraph("<b>Brain Tumor MRI Dataset</b><br/>(Kaggle / masoudnickparvar V2)", tbl_cell),
            Paragraph("Primary Development<br/>(Train / Val / Test)", tbl_cell_center),
            Paragraph("<b>7,193</b> images<br/>(audited from 7,200)", tbl_cell_center),
            Paragraph("Multiclass pool combining <b>Figshare</b> (Cheng et al.), <b>SARTAJ</b>, and <b>Br35H</b> datasets. CC BY 4.0 license.", tbl_cell)
        ],
        [
            Paragraph("<b>BRISC 2025</b><br/>(Fateh et al., arXiv:2506.14318)", tbl_cell),
            Paragraph("Auxiliary / Localization<br/>Pixel Mask Audit", tbl_cell_center),
            Paragraph("<b>6,000</b> scans<br/>(T1-weighted)", tbl_cell_center),
            Paragraph("Physician-validated pixel-level tumor masks across 3 anatomical planes (axial, coronal, sagittal). Used for Grad-CAM overlap analysis.", tbl_cell)
        ],
        [
            Paragraph("<b>BDNeuro-MRI v7</b><br/>(Epic & CSCR Hospital)", tbl_cell),
            Paragraph("External Candidate<br/>Deduplicated Benchmark", tbl_cell_center),
            Paragraph("<b>5,941</b> scans<br/>(deduplicated)", tbl_cell_center),
            Paragraph("Hospital MRI collection with exact & perceptual duplicate removal (Hamming dist <= 5). Used for baseline fine-tuning comparison.", tbl_cell)
        ],
        [
            Paragraph("<b>Literature Tracker</b><br/>(AI_NeuroOnco RAG Corpus)", tbl_cell),
            Paragraph("Knowledge Base<br/>Vector Store (Chroma)", tbl_cell_center),
            Paragraph("<b>74</b> papers<br/>372 text chunks", tbl_cell_center),
            Paragraph("Curated full-text neuro-oncology peer-reviewed literature indexed with sentence-transformers for grounded RAG query responses.", tbl_cell)
        ]
    ]

    ds_table = Table(ds_summary_data, colWidths=[130, 95, 85, 194])
    ds_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_navy),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(ds_table)
    story.append(Spacer(1, 8))

    # Audit Note
    p_audit = (
        "<b>Data Integrity & Leakage Prevention:</b> During strict data auditing, 1 cross-label perceptual duplicate "
        "component consisting of 7 conflicting images was detected and completely excluded without visual relabelling. "
        "The finalized 7,193-image development manifest was formally audited with <b>zero cross-split leakage</b> "
        "and 100% label consistency."
    )
    audit_callout = Table([[Paragraph(p_audit, callout_style)]], colWidths=[504])
    audit_callout.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#BFDBFE")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(audit_callout)
    story.append(Spacer(1, 14))

    # ── SECTION 2: DATASET SIZE & SPLIT COMPOSITION ──
    story.append(Paragraph("2. Dataset Size & Split Breakdown", h1_style))
    p_splits = (
        "The primary dataset (7,193 images) is partitioned into Training (58.2%), Validation (12.5%), and a strictly "
        "held-out, locked Internal Test partition (29.3%). All images are preprocessed and resized to 224x224x3:"
    )
    story.append(Paragraph(p_splits, body_style))
    story.append(Spacer(1, 6))

    split_table_data = [
        [
            Paragraph("Diagnostic Class", tbl_header),
            Paragraph("Training Set", tbl_header),
            Paragraph("Validation Set", tbl_header),
            Paragraph("Locked Test Set", tbl_header),
            Paragraph("Total Cohort", tbl_header),
            Paragraph("Class %", tbl_header)
        ],
        [
            Paragraph("<b>Glioma</b>", tbl_cell),
            Paragraph("1,149", tbl_cell_center),
            Paragraph("247", tbl_cell_center),
            Paragraph("403", tbl_cell_center),
            Paragraph("<b>1,799</b>", tbl_cell_center),
            Paragraph("25.0%", tbl_cell_center)
        ],
        [
            Paragraph("<b>Meningioma</b>", tbl_cell),
            Paragraph("1,066", tbl_cell_center),
            Paragraph("229", tbl_cell_center),
            Paragraph("499", tbl_cell_center),
            Paragraph("<b>1,794</b>", tbl_cell_center),
            Paragraph("24.9%", tbl_cell_center)
        ],
        [
            Paragraph("<b>No Tumor (Healthy)</b>", tbl_cell),
            Paragraph("824", tbl_cell_center),
            Paragraph("177", tbl_cell_center),
            Paragraph("799", tbl_cell_center),
            Paragraph("<b>1,800</b>", tbl_cell_center),
            Paragraph("25.0%", tbl_cell_center)
        ],
        [
            Paragraph("<b>Pituitary Tumor</b>", tbl_cell),
            Paragraph("1,148", tbl_cell_center),
            Paragraph("246", tbl_cell_center),
            Paragraph("406", tbl_cell_center),
            Paragraph("<b>1,800</b>", tbl_cell_center),
            Paragraph("25.0%", tbl_cell_center)
        ],
        [
            Paragraph("<b>TOTALS</b>", tbl_cell_bold),
            Paragraph("<b>4,187 (58.2%)</b>", tbl_cell_center),
            Paragraph("<b>899 (12.5%)</b>", tbl_cell_center),
            Paragraph("<b>2,107 (29.3%)</b>", tbl_cell_center),
            Paragraph("<b>7,193</b>", tbl_cell_center),
            Paragraph("<b>100.0%</b>", tbl_cell_center)
        ]
    ]

    split_table = Table(split_table_data, colWidths=[124, 76, 76, 86, 76, 66])
    split_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_slate),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor("#F8FAFC")]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#E2E8F0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(split_table)
    story.append(Spacer(1, 10))

    # Embed Chart 2: Dataset Distribution
    story.append(Image(str(chart2_path), width=6.8 * inch, height=3.1 * inch))
    story.append(Spacer(1, 14))

    # ── SECTION 3: MODEL ACCURACY & PERFORMANCE BENCHMARKS ──
    story.append(Paragraph("3. Trained Model Performance & Statistical Accuracy", h1_style))
    p_perf = (
        "Models were evaluated against the official locked internal test cohort consisting of <b>2,107 unseen scans</b> "
        "across 1,460 independent bootstrap duplicate components. Metrics report point estimates and 95% confidence "
        "intervals (CI) calculated via group-stratified bootstrapping:"
    )
    story.append(Paragraph(p_perf, body_style))
    story.append(Spacer(1, 6))

    perf_table_data = [
        [
            Paragraph("Performance Metric", tbl_header),
            Paragraph("EfficientNet-B0<br/>(Transfer Learning)", tbl_header),
            Paragraph("Custom CNN<br/>(~2M Parameters)", tbl_header),
            Paragraph("Statistical Significance<br/>(Paired Delta)", tbl_header)
        ],
        [
            Paragraph("<b>Overall Test Accuracy</b>", tbl_cell),
            Paragraph("<b>90.75%</b> (88.90% – 92.47%)", tbl_cell_center),
            Paragraph("<b>83.91%</b> (81.66% – 86.11%)", tbl_cell_center),
            Paragraph("+6.84% (p = 2.58e-29, McNemar)", tbl_cell_center)
        ],
        [
            Paragraph("<b>Balanced Accuracy</b>", tbl_cell),
            Paragraph("<b>89.31%</b>", tbl_cell_center),
            Paragraph("<b>81.63%</b>", tbl_cell_center),
            Paragraph("+7.68% improvement", tbl_cell_center)
        ],
        [
            Paragraph("<b>Macro F1-Score</b>", tbl_cell),
            Paragraph("<b>90.78%</b> (89.09% – 92.27%)", tbl_cell_center),
            Paragraph("<b>83.27%</b> (81.24% – 85.37%)", tbl_cell_center),
            Paragraph("+7.51% (95% CI: 5.7% – 9.4%)", tbl_cell_center)
        ],
        [
            Paragraph("<b>Macro ROC-AUC (OvR)</b>", tbl_cell),
            Paragraph("<b>97.98%</b> (0.9798)", tbl_cell_center),
            Paragraph("<b>95.38%</b> (0.9538)", tbl_cell_center),
            Paragraph("+2.60% discriminative gain", tbl_cell_center)
        ],
        [
            Paragraph("<b>Macro PR-AUC (OvR)</b>", tbl_cell),
            Paragraph("<b>96.30%</b> (0.9630)", tbl_cell_center),
            Paragraph("<b>91.17%</b> (0.9117)", tbl_cell_center),
            Paragraph("+5.13% precision-recall gain", tbl_cell_center)
        ],
        [
            Paragraph("<b>Matthews Correlation (MCC)</b>", tbl_cell),
            Paragraph("<b>0.8759</b>", tbl_cell_center),
            Paragraph("<b>0.7866</b>", tbl_cell_center),
            Paragraph("+0.0893 correlation shift", tbl_cell_center)
        ],
        [
            Paragraph("<b>Expected Calibration Error (ECE)</b>", tbl_cell),
            Paragraph("0.0525 → 0.0559 (calibrated)", tbl_cell_center),
            Paragraph("0.0288 → 0.0995 (calibrated)", tbl_cell_center),
            Paragraph("Temp T = 0.608 (EfficientNet)", tbl_cell_center)
        ]
    ]

    perf_table = Table(perf_table_data, colWidths=[150, 125, 115, 114])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 10))

    # Embed Chart 1: Model Comparison Bar Chart
    story.append(Image(str(chart1_path), width=6.8 * inch, height=3.3 * inch))
    story.append(Spacer(1, 14))

    # ── SECTION 4: CLINICAL BENCHMARKS & REFERENCE ARCHITECTURES ──
    story.append(Paragraph("4. Benchmark Comparisons on Reference Cohorts", h1_style))
    p_bench = (
        "To contextualize performance across convolutional and transformer architectures, standard transfer learning "
        "paradigms were benchmarked on the deduplicated BDNeuro-MRI cohort (5,941 scans):"
    )
    story.append(Paragraph(p_bench, body_style))
    story.append(Spacer(1, 6))

    bench_table_data = [
        [
            Paragraph("Architecture", tbl_header),
            Paragraph("Training Paradigm", tbl_header),
            Paragraph("Test Accuracy (%)", tbl_header),
            Paragraph("Test Macro-F1", tbl_header),
            Paragraph("Parameter Complexity", tbl_header)
        ],
        [
            Paragraph("<b>DenseNet121</b>", tbl_cell),
            Paragraph("Fine-Tuned Pretrained", tbl_cell),
            Paragraph("<b>98.31%</b>", tbl_cell_center),
            Paragraph("<b>0.9831</b>", tbl_cell_center),
            Paragraph("~7.98M parameters", tbl_cell)
        ],
        [
            Paragraph("<b>ResNet50</b>", tbl_cell),
            Paragraph("Fine-Tuned Pretrained", tbl_cell),
            Paragraph("<b>97.08%</b>", tbl_cell_center),
            Paragraph("<b>0.9708</b>", tbl_cell_center),
            Paragraph("~25.6M parameters", tbl_cell)
        ],
        [
            Paragraph("<b>EfficientNet-B0</b>", tbl_cell),
            Paragraph("Transfer Learning (Primary)", tbl_cell),
            Paragraph("<b>90.75%</b>", tbl_cell_center),
            Paragraph("<b>0.9078</b>", tbl_cell_center),
            Paragraph("~5.3M (Optimal efficiency)", tbl_cell)
        ],
        [
            Paragraph("<b>DenseNet121</b>", tbl_cell),
            Paragraph("Frozen Pretrained Backbone", tbl_cell),
            Paragraph("<b>87.06%</b>", tbl_cell_center),
            Paragraph("0.8703", tbl_cell_center),
            Paragraph("Linear classification head", tbl_cell)
        ],
        [
            Paragraph("<b>Custom CNN</b>", tbl_cell),
            Paragraph("Trained from scratch", tbl_cell),
            Paragraph("<b>84.70% / 83.91%</b>", tbl_cell_center),
            Paragraph("0.8457 / 0.8327", tbl_cell_center),
            Paragraph("~2.0M parameters (Lightweight)", tbl_cell)
        ],
        [
            Paragraph("<b>Vision Transformer (ViT)</b>", tbl_cell),
            Paragraph("Trained from scratch", tbl_cell),
            Paragraph("<b>82.56%</b>", tbl_cell_center),
            Paragraph("0.8240", tbl_cell_center),
            Paragraph("Patch projection self-attention", tbl_cell)
        ],
        [
            Paragraph("<b>Hybrid CNN-ViT</b>", tbl_cell),
            Paragraph("Trained from scratch", tbl_cell),
            Paragraph("<b>81.89%</b>", tbl_cell_center),
            Paragraph("0.8201", tbl_cell_center),
            Paragraph("Conv stem + Transformer blocks", tbl_cell)
        ]
    ]

    bench_table = Table(bench_table_data, colWidths=[120, 120, 90, 80, 94])
    bench_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_navy),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(bench_table)
    story.append(Spacer(1, 14))

    # ── SECTION 5: CLINICAL SAFETY & TRACK 3 ENHANCEMENTS ──
    story.append(Paragraph("5. Clinical AI Decision Support & Safety Frameworks", h1_style))
    p_safety = (
        "Beyond core classification accuracy, the AI NeuroOnco framework implements four critical safety "
        "and diagnostic governance mechanisms to prevent automated diagnostic errors in clinical settings:"
    )
    story.append(Paragraph(p_safety, body_style))
    story.append(Spacer(1, 6))

    safety_items = [
        [
            Paragraph("<b>1. Normalized Shannon Entropy Index:</b>", body_bold),
            Paragraph(
                "Calculates prediction distribution conflict: <i>H(P) = -Σ p_i log2(p_i) / 2.0</i>. "
                "Provides quantitative tri-state decision classification: <b>Decisive</b> (&lt; 0.300), "
                "<b>Ambivalent</b> (0.300 – 0.600), or <b>Conflicted</b> (&gt; 0.600) requiring mandatory human radiologist review.",
                body_style
            )
        ],
        [
            Paragraph("<b>2. Explainable AI (Grad-CAM):</b>", body_bold),
            Paragraph(
                "Activates feature hooks on final convolutional layers to generate pixel-level attention heatmaps, "
                "enabling visual audit of tumor morphology alignment (1,045 cases quantitatively evaluated).",
                body_style
            )
        ],
        [
            Paragraph("<b>3. PACS / HL7 FHIR Integration:</b>", body_bold),
            Paragraph(
                "Generates standardized FHIR <code>DiagnosticReport</code> schemas containing LOINC/SNOMED diagnostic codes, "
                "model timestamps, prediction probability distributions, and entropy metrics for electronic health records.",
                body_style
            )
        ],
        [
            Paragraph("<b>4. Multimodal RAG with MedGemma:</b>", body_bold),
            Paragraph(
                "Integrates Google's medical foundation models (MedGemma / Gemini 2.0 Flash) with ChromaDB vector search across "
                "74 indexed research papers, providing grounded literature citations and source snippet verification.",
                body_style
            )
        ]
    ]

    safety_table = Table(safety_items, colWidths=[160, 344])
    safety_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(safety_table)
    story.append(Spacer(1, 14))

    # ── SECTION 6: ETHICAL GOVERNANCE & LIMITATIONS ──
    story.append(Paragraph("6. Research Governance & Clinical Disclaimers", h1_style))
    p_disclaimer = (
        "<b>Non-Diagnostic Research Notice:</b> This software and report are intended solely for retrospective academic research "
        "and algorithm evaluation. The diagnostic labels ('glioma', 'meningioma', 'pituitary', 'no_tumor') reflect source dataset "
        "annotations and do not substitute for formal histopathological examination or WHO CNS5 molecular classification. "
        "No individual treatment or surgical decisions should be based on automated outputs without independent board-certified "
        "radiologist verification."
    )
    disclaimer_box = Table([[Paragraph(p_disclaimer, callout_style)]], colWidths=[504])
    disclaimer_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#FECACA")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(disclaimer_box)

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)

    # Copy to root directory for convenient access
    if PDF_OUTPUT_PATH.exists():
        import shutil
        shutil.copyfile(PDF_OUTPUT_PATH, ROOT_PDF_COPY)
        print(f"✓ PDF successfully generated at: {PDF_OUTPUT_PATH}")
        print(f"✓ Copied to root workspace: {ROOT_PDF_COPY}")
        file_size_kb = os.path.getsize(PDF_OUTPUT_PATH) / 1024
        print(f"  File size: {file_size_kb:.1f} KB")


if __name__ == "__main__":
    build_pdf()
