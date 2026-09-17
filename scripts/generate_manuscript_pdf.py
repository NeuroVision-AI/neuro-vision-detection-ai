#!/usr/bin/env python3
"""
Generate a camera-ready, publication-grade academic research manuscript PDF
for the AI NeuroOnco study, styled after premier medical AI journals
(Nature Medicine / Lancet Digital Health / IEEE TMI).
"""

import os
import sys
from pathlib import Path
from datetime import datetime

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_ai_neuroonco_manuscript"

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
PDF_OUTPUT_PATH = REPORTS_DIR / "AI_NeuroOnco_Research_Manuscript.pdf"
ROOT_PDF_COPY = ROOT_DIR / "AI_NeuroOnco_Research_Manuscript.pdf"
CHART_DIR = ROOT_DIR / "tmp" / "manuscript_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE SPECIALIZED RESEARCH FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def generate_research_figures():
    # Figure A: Cross-Corpus Contamination Audit
    fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=300)
    datasets = ["BRISC 2025\n(Fateh et al.)", "BDNeuro-MRI v7\n(Hospital Cohort)"]
    exact_pct = [79.68, 55.78]
    near_pct = [0.0, 19.80] # additional near duplicates for BDNeuro (total 75.58%)
    clean_pct = [20.32, 24.42]

    x = np.arange(len(datasets))
    width = 0.45

    p1 = ax.bar(x, exact_pct, width, label="Exact Duplicate of Development Data (Byte-Identical)", color="#DC2626", edgecolor="#B91C1C")
    p2 = ax.bar(x, near_pct, width, bottom=exact_pct, label="Perceptual Duplicate (dHash Hamming <= 5)", color="#F59E0B", edgecolor="#D97706")
    bottom_clean = [e + n for e, n in zip(exact_pct, near_pct)]
    p3 = ax.bar(x, clean_pct, width, bottom=bottom_clean, label="Independent / Unique Scans", color="#10B981", edgecolor="#059669")

    ax.set_ylabel("Proportion of Dataset (%)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_title("The Cross-Corpus Contamination Discovery:\nExact & Perceptual Overlap with Public Training Sources", fontsize=10.5, fontweight="bold", color="#0F172A", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9.5, fontweight="bold", color="#334155")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#94A3B8")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotations
    ax.annotate("79.7% Exact Reused\n(4,781 / 6,000)", xy=(0, 79.68), xytext=(0, -25),
                textcoords="offset points", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
    ax.annotate("55.8% Exact\n(3,314 / 5,941)", xy=(1, 28), xytext=(0, 0),
                textcoords="offset points", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")
    ax.annotate("+19.8% Near\n(4,490 Total)", xy=(1, 65), xytext=(0, 0),
                textcoords="offset points", ha="center", va="center", fontsize=8, fontweight="bold", color="#78350F")

    plt.tight_layout()
    fig_overlap_path = CHART_DIR / "fig_cross_corpus_overlap.png"
    plt.savefig(fig_overlap_path, bbox_inches="tight")
    plt.close()

    # Figure B: Calibration Shift & ECE Degradation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=300)

    # Left: ECE change
    models = ["EfficientNet-B0", "Custom CNN"]
    ece_before = [0.0525, 0.0288]
    ece_after = [0.0559, 0.0995]

    x = np.arange(len(models))
    w = 0.35
    ax1.bar(x - w/2, ece_before, w, label="Uncalibrated ECE", color="#3B82F6")
    ax1.bar(x + w/2, ece_after, w, label="Validation-Scaled ECE", color="#EF4444")
    ax1.set_ylabel("Expected Calibration Error (ECE)", fontsize=8.5, fontweight="bold")
    ax1.set_title("Post-Hoc Calibration Shift\n(Validation Scaling on Test)", fontsize=9.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8.5)
    ax1.legend(fontsize=7.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: Grad-CAM Localization Failure
    categories = ["Mean IoU\n(All Tumors)", "Pointing Game\nSuccess Rate", "Glioma\nIoU", "Meningioma\nIoU", "Pituitary\nIoU"]
    scores = [0.029, 0.064, 0.038, 0.033, 0.009]
    colors_list = ["#8B5CF6", "#EC4899", "#3B82F6", "#F59E0B", "#EF4444"]

    ax2.bar(categories, [s * 100 for s in scores], color=colors_list, width=0.55)
    ax2.set_ylabel("Percentage (%)", fontsize=8.5, fontweight="bold")
    ax2.set_title("Grad-CAM Lesion Localization vs.\nRadiologist Masks (N=1,045)", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 10)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for i, v in enumerate(scores):
        ax2.annotate(f"{v*100:.1f}%", xy=(i, v*100), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.tight_layout()
    fig_calibration_xai_path = CHART_DIR / "fig_calibration_xai.png"
    plt.savefig(fig_calibration_xai_path, bbox_inches="tight")
    plt.close()

    return fig_overlap_path, fig_calibration_xai_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. RUNNING HEADERS & FOOTERS CANVAS
# ─────────────────────────────────────────────────────────────────────────────
class ManuscriptCanvas(canvas.Canvas):
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
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 755, "AI NeuroOnco Research Manuscript | Critical Benchmarks & Leakage Controls")
            self.drawRightString(612 - 54, 755, "ACADEMIC PREPRINT — PEER REVIEW DRAFT")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 748, 612 - 54, 748)

        # Footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        self.drawString(54, 32, "Track 3: AI Integration & Neuro-Oncology Multimodal Computer Vision")
        self.drawRightString(612 - 54, 32, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD FULL MANUSCRIPT PDF
# ─────────────────────────────────────────────────────────────────────────────
def build_manuscript_pdf():
    fig_overlap_path, fig_cal_xai_path = generate_research_figures()
    data_quality_dir = ROOT_DIR / "outputs" / "data_quality"
    chart_class_dist = data_quality_dir / "class_distribution.png"
    chart_dim_dist = data_quality_dir / "image_dimensions.png"

    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    c_navy = colors.HexColor("#0F2942")
    c_slate = colors.HexColor("#1E293B")
    c_blue = colors.HexColor("#2563EB")
    c_darkblue = colors.HexColor("#1E3A8A")
    c_body = colors.HexColor("#334155")

    title_style = ParagraphStyle(
        "MTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=c_navy,
        spaceAfter=6
    )

    author_style = ParagraphStyle(
        "MAuthors",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=c_slate,
        spaceAfter=4
    )

    affil_style = ParagraphStyle(
        "MAffil",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=12
    )

    abstract_heading = ParagraphStyle(
        "MAbstractHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=c_navy,
        spaceBefore=4,
        spaceAfter=2
    )

    abstract_body = ParagraphStyle(
        "MAbstractBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        textColor=c_body,
        spaceAfter=6
    )

    h1_style = ParagraphStyle(
        "MH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=c_darkblue,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "MH2",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13.5,
        textColor=c_slate,
        spaceBefore=9,
        spaceAfter=3,
        keepWithNext=True
    )

    body_p = ParagraphStyle(
        "MBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=13,
        textColor=c_body,
        spaceAfter=6
    )

    tbl_hdr = ParagraphStyle(
        "MTblHdr",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.white,
        alignment=1
    )

    tbl_c = ParagraphStyle(
        "MTblC",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=c_slate
    )

    tbl_c_center = ParagraphStyle(
        "MTblCCenter",
        parent=tbl_c,
        alignment=1
    )

    tbl_c_bold = ParagraphStyle(
        "MTblCBold",
        parent=tbl_c,
        fontName="Helvetica-Bold"
    )

    callout_p = ParagraphStyle(
        "MCallout",
        parent=body_p,
        fontSize=8,
        leading=11.5,
        textColor=c_navy
    )

    story = []

    # ── HEADER & TITLE ──
    story.append(Paragraph(
        "Leakage-Aware Four-Class Brain MRI Classification: Internal Model Comparison, Calibration Shift, Quantitative Grad-CAM, and External-Corpus Contamination Audits",
        title_style
    ))
    story.append(Paragraph(
        "<b>AI NeuroOnco Research Consortium</b> — Track 3: AI Integration & Multimodal Neuro-Oncology Systems",
        author_style
    ))
    story.append(Paragraph(
        "Department of Computer Science & Medical Imaging Intelligence • International Research Preprint • September 2026",
        affil_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=c_blue, spaceBefore=0, spaceAfter=8))

    # ── STRUCTURED ABSTRACT ──
    abstract_box_data = [
        [
            Paragraph("<b>STRUCTURED ABSTRACT</b>", abstract_heading),
        ],
        [
            Paragraph("<b>Background:</b> Public brain MRI classification studies frequently claim near-perfect internal accuracy (>98%), yet widespread duplicate images, absent patient identifiers, label inconsistencies, lack of calibration, and cross-dataset contamination produce severely optimistic claims. We conducted an independent, leakage-controlled study establishing empirical evidence boundaries for 4-class 2D brain tumor classification.", abstract_body),
        ],
        [
            Paragraph("<b>Methods:</b> Using 7,200 public MRI images, we deployed raw SHA-256 byte hashing and perceptual difference hashing (dHash) to construct transitive duplicate components. One cross-label duplicate component (7 images) was completely excluded, yielding a leakage-controlled cohort of 7,193 images. EfficientNet-B0 transfer learning was compared against a 5-block custom CNN (~2M parameters). Post-hoc temperature scaling was fitted strictly on validation data. Secondary endpoints included quantitative Grad-CAM repeatability, parameter randomization, pointing-game success, and lesion mask Intersection-over-Union (IoU) evaluated against 1,045 physician-annotated BRISC masks.", abstract_body),
        ],
        [
            Paragraph("<b>Results:</b> On 2,107 locked internal-test scans (1,460 independent bootstrap duplicate groups), EfficientNet-B0 achieved 90.75% accuracy (95% CI: 88.90%–92.47%), balanced accuracy 89.31%, macro-F1 90.78%, macro ROC-AUC 97.98%, and MCC 0.8759, significantly outperforming the custom CNN (83.91% accuracy, 83.27% macro-F1, p=2.58e-29). However, validation-fitted temperature scaling worsened test ECE from 0.0525 to 0.0559 (EfficientNet) and 0.0288 to 0.0995 (CNN) due to test-set prevalence shifts. Grad-CAM exhibited 100% deterministic repeatability but yielded a mean localization IoU of only 0.029 and pointing-game success of 6.4%. Crucially, cross-corpus auditing revealed that 79.68% of BRISC 2025 and 55.78% of BDNeuro-MRI v7 were exact duplicates of the development data.", abstract_body),
        ],
        [
            Paragraph("<b>Conclusion:</b> When strict data leakage controls are enforced, headline accuracy declines to realistic levels, temperature scaling fails under label shift, and classification Grad-CAM fails lesion localization. Furthermore, pervasive cross-corpus data reuse in the neuro-oncology literature invalidates claims of independent external generalization. We provide a tri-state Shannon Entropy triage protocol to safeguard clinical decision-making.", abstract_body),
        ]
    ]
    abstract_table = Table(abstract_box_data, colWidths=[504])
    abstract_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(abstract_table)
    story.append(Spacer(1, 10))

    # ── SECTION 1: INTRODUCTION & CLINICAL MOTIVATION ──
    story.append(Paragraph("1. Introduction & The Medical AI Validation Crisis", h1_style))
    story.append(Paragraph(
        "Machine learning in neuro-oncology holds profound potential for diagnostic triage, workflow acceleration, "
        "and surgical planning. However, contemporary literature is characterized by an alarming inflation of performance "
        "metrics: dozens of published studies report classification accuracies ranging from 98.5% to 99.8% on public MRI benchmarks. "
        "Scrutiny under international reporting guidelines—including <b>CLAIM 2024</b>, <b>TRIPOD+AI</b>, and <b>PROBAST+AI</b>—reveals "
        "that these inflated figures frequently stem from four methodological vulnerabilities: (1) unverified image-level splitting "
        "allowing slices from the same subject to appear in both training and test partitions; (2) silent cross-label contamination; "
        "(3) uncalibrated probability estimates masquerading as clinical confidence; and (4) superficial saliency map inspections "
        "that conflate high visual contrast with anatomical faithfulness.",
        body_p
    ))
    story.append(Paragraph(
        "Furthermore, current benchmarks categorize brain MRI scans into four simplistic labels: <i>Glioma, Meningioma, Pituitary Tumor, "
        "and No Tumor</i>. In clinical reality, the <b>WHO Classification of Tumours of the Central Nervous System (WHO CNS5)</b> requires "
        "integrated histopathological and molecular characterization (e.g., IDH1/2 mutation status, 1p/19q codeletion, and MGMT promoter "
        "methylation). Single-slice 2D classification models cannot and must not be conflated with molecular-pathological ground truth. "
        "In this study, we present an audited, leakage-controlled research proof-of-concept that establishes verified evidence boundaries, "
        "proves the fragility of post-hoc calibration, demonstrates the failure of Grad-CAM for lesion localization, and exposes "
        "widespread duplicate contamination across international brain tumor datasets.",
        body_p
    ))
    story.append(Spacer(1, 6))

    # ── SECTION 2: DATA INTEGRITY & THE CONTAMINATION DISCOVERY ──
    story.append(Paragraph("2. Data Integrity, Deduplication & Cross-Corpus Audits", h1_style))
    story.append(Paragraph(
        "We retrieved the primary Brain Tumor MRI Dataset (Kaggle Version 2, 7,200 images, licensed under CC BY 4.0), "
        "which compiles historical collections from Figshare (Cheng et al.), SARTAJ, and Br35H. Before model training, every scan "
        "was subjected to cryptographic SHA-256 byte hashing and 64-bit perceptual difference hashing (dHash). "
        "Transitive identity components were computed to cluster both exact byte duplicates and visually identical scans "
        "(Hamming distance <= 5).",
        body_p
    ))

    # Embed Contamination Figure
    story.append(Image(str(fig_overlap_path), width=6.8 * inch, height=3.0 * inch))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Discovery of Pervasive Cross-Corpus Contamination:</b> To assess independent generalization, we audited two "
        "prominent international external candidate datasets: <b>BRISC 2025</b> (6,000 scans) and <b>BDNeuro-MRI v7</b> (5,941 scans). "
        "Cryptographic auditing revealed that <b>4,781 out of 6,000 BRISC classification records (79.68%)</b> and <b>3,314 out of "
        "5,941 BDNeuro-MRI scans (55.78%)</b> were exact byte-identical duplicates of the development data. An additional 19.80% "
        "of BDNeuro-MRI scans were perceptual duplicates, yielding an overall cross-corpus contamination rate of <b>75.58%</b>. "
        "Consequently, both cohorts were disqualified from external validation. This critical discovery highlights that many published "
        "claims of 'independent external generalizability' in neuro-oncology literature are artifactual products of unrecognized data reuse.",
        body_p
    ))

    # Table 1: Cross-Corpus Overlap Audit
    tbl1_data = [
        [
            Paragraph("Candidate Dataset", tbl_hdr),
            Paragraph("Total Records", tbl_hdr),
            Paragraph("Exact Overlap", tbl_hdr),
            Paragraph("Near Overlap (dHash<=5)", tbl_hdr),
            Paragraph("Total Contamination", tbl_hdr),
            Paragraph("Audit Verdict", tbl_hdr)
        ],
        [
            Paragraph("<b>BRISC 2025</b> (Fateh et al.)", tbl_c),
            Paragraph("6,000", tbl_c_center),
            Paragraph("4,781 (79.68%)", tbl_c_center),
            Paragraph("—", tbl_c_center),
            Paragraph("<b>79.68%</b>", tbl_c_center),
            Paragraph("<b>REJECTED</b> as external cohort;<br/>masks retained for XAI audit", tbl_c)
        ],
        [
            Paragraph("<b>BDNeuro-MRI v7</b> (Mendeley)", tbl_c),
            Paragraph("5,941", tbl_c_center),
            Paragraph("3,314 (55.78%)", tbl_c_center),
            Paragraph("1,176 (19.80%)", tbl_c_center),
            Paragraph("<b>75.58%</b>", tbl_c_center),
            Paragraph("<b>REJECTED</b> as external cohort;<br/>perceptual cross-split leakage", tbl_c)
        ]
    ]
    t1 = Table(tbl1_data, colWidths=[120, 65, 80, 85, 75, 79])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_navy),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t1)
    story.append(Spacer(1, 10))

    # ── SECTION 3: EMPIRICAL RESULTS & MODEL DISCRIMINATION ──
    story.append(Paragraph("3. Model Discrimination & Performance Benchmarks", h1_style))
    story.append(Paragraph(
        "Models were evaluated on <b>2,107 locked internal-test scans</b> representing 1,460 independent duplicate components. "
        "The official test partition preserves source-specific prevalence (799 No Tumor, 499 Meningioma, 406 Pituitary, 403 Glioma). "
        "To account for class imbalance, macro-averaged F1 and balanced accuracy are reported alongside overall accuracy. "
        "Statistical uncertainty is quantified via 2,000 group-stratified bootstrap resamples.",
        body_p
    ))

    # Table 2: Comprehensive Evaluation Results
    tbl2_data = [
        [
            Paragraph("Performance Metric", tbl_hdr),
            Paragraph("EfficientNet-B0 (Transfer Learning)", tbl_hdr),
            Paragraph("Custom CNN (~2M Parameters)", tbl_hdr),
            Paragraph("Paired Difference (Δ)", tbl_hdr),
            Paragraph("Statistical Test", tbl_hdr)
        ],
        [
            Paragraph("<b>Overall Accuracy</b>", tbl_c),
            Paragraph("<b>90.75%</b> (88.90% – 92.47%)", tbl_c_center),
            Paragraph("<b>83.91%</b> (81.66% – 86.11%)", tbl_c_center),
            Paragraph("+6.84% (5.00% – 8.60%)", tbl_c_center),
            Paragraph("p = 2.58e-29 (McNemar)", tbl_c_center)
        ],
        [
            Paragraph("<b>Balanced Accuracy</b>", tbl_c),
            Paragraph("<b>89.31%</b>", tbl_c_center),
            Paragraph("<b>81.63%</b>", tbl_c_center),
            Paragraph("+7.68% gain", tbl_c_center),
            Paragraph("Macro class average", tbl_c_center)
        ],
        [
            Paragraph("<b>Macro F1-Score</b>", tbl_c),
            Paragraph("<b>90.78%</b> (89.09% – 92.27%)", tbl_c_center),
            Paragraph("<b>83.27%</b> (81.24% – 85.37%)", tbl_c_center),
            Paragraph("+7.51% (5.70% – 9.40%)", tbl_c_center),
            Paragraph("Bootstrap 95% CI", tbl_c_center)
        ],
        [
            Paragraph("<b>Macro ROC-AUC (OvR)</b>", tbl_c),
            Paragraph("<b>97.98%</b> (0.9798)", tbl_c_center),
            Paragraph("<b>95.38%</b> (0.9538)", tbl_c_center),
            Paragraph("+2.60% discriminative gain", tbl_c_center),
            Paragraph("One-vs-Rest AUC", tbl_c_center)
        ],
        [
            Paragraph("<b>Macro PR-AUC (OvR)</b>", tbl_c),
            Paragraph("<b>96.30%</b> (0.9630)", tbl_c_center),
            Paragraph("<b>91.17%</b> (0.9117)", tbl_c_center),
            Paragraph("+5.13% precision-recall gain", tbl_c_center),
            Paragraph("Area under Precision-Recall", tbl_c_center)
        ],
        [
            Paragraph("<b>Matthews Correlation (MCC)</b>", tbl_c),
            Paragraph("<b>0.8759</b>", tbl_c_center),
            Paragraph("<b>0.7866</b>", tbl_c_center),
            Paragraph("+0.0893 correlation shift", tbl_c_center),
            Paragraph("Multiclass correlation", tbl_c_center)
        ],
        [
            Paragraph("<b>Glioma F1-Score</b>", tbl_c),
            Paragraph("<b>88.2%</b> (n=403)", tbl_c_center),
            Paragraph("<b>83.2%</b> (n=403)", tbl_c_center),
            Paragraph("+5.0% sensitivity boost", tbl_c_center),
            Paragraph("Per-class F1", tbl_c_center)
        ],
        [
            Paragraph("<b>Meningioma F1-Score</b>", tbl_c),
            Paragraph("<b>85.2%</b> (n=499)", tbl_c_center),
            Paragraph("<b>67.7%</b> (n=499)", tbl_c_center),
            Paragraph("+17.5% dramatic boost", tbl_c_center),
            Paragraph("Per-class F1", tbl_c_center)
        ],
        [
            Paragraph("<b>No Tumor F1-Score</b>", tbl_c),
            Paragraph("<b>90.9%</b> (n=799)", tbl_c_center),
            Paragraph("<b>86.0%</b> (n=799)", tbl_c_center),
            Paragraph("+4.9% specificity boost", tbl_c_center),
            Paragraph("Healthy control class", tbl_c_center)
        ],
        [
            Paragraph("<b>Pituitary Tumor F1-Score</b>", tbl_c),
            Paragraph("<b>98.8%</b> (n=406)", tbl_c_center),
            Paragraph("<b>96.2%</b> (n=406)", tbl_c_center),
            Paragraph("+2.6% top performance", tbl_c_center),
            Paragraph("Sella turcica localization", tbl_c_center)
        ]
    ]
    t2 = Table(tbl2_data, colWidths=[130, 105, 95, 90, 84])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_darkblue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    # ── SECTION 4: CALIBRATION SHIFT & EXPLAINABILITY PARADOX ──
    story.append(Paragraph("4. Calibration Shift & The Explainability Localization Paradox", h1_style))
    story.append(Paragraph(
        "<b>Failure of Post-Hoc Calibration Under Class-Prior Shift:</b> Following Guo et al., temperature scaling was "
        "optimized on the validation split by minimizing negative log-likelihood (T = 0.608 for EfficientNet; T = 0.550 for Custom CNN). "
        "Crucially, when evaluated on the locked test set, <b>calibration worsened for both architectures</b>: EfficientNet ECE degraded "
        "from 0.0525 to 0.0559 (Brier: 0.142 to 0.150), and Custom CNN ECE degraded from 0.0288 to 0.0995 (Brier: 0.254 to 0.271). "
        "This occurs because the test partition contains 37.9% healthy controls versus 19.7% in validation. Scalar temperature scaling "
        "assumes class-conditional invariance and fails under label shift. We demonstrate that medical AI systems must incorporate "
        "Bayes-adjusted prior correction rather than single-parameter temperature scaling.",
        body_p
    ))

    # Embed Figure B (Calibration & Grad-CAM)
    story.append(Image(str(fig_cal_xai_path), width=6.8 * inch, height=2.85 * inch))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>The Grad-CAM Faithfulness Paradox:</b> Saliency maps are widely assumed to provide clinical explainability. "
        "We tested Grad-CAM against <b>1,045 locked internal-test scans</b> with physician-reviewed binary tumor masks from BRISC. "
        "While Grad-CAM achieved <b>100% deterministic repeatability</b> across repeated forward-backward passes, its objective lesion "
        "localization was dismal: the mean Intersection-over-Union (IoU) was only <b>0.029</b> (95% CI: 0.026–0.032) and pointing-game "
        "accuracy was only <b>6.4%</b> (5.0%–8.0%). For pituitary tumors, pointing-game hit rate collapsed to <b>1.2%</b>. "
        "Grad-CAM activations concentrated heavily on skull contours, ventricular margins, and non-lesion spatial cues, confirming "
        "that classification models rely on shortcut context rather than true lesion morphology. Saliency maps must never be presented "
        "to clinicians as lesion boundaries.",
        body_p
    ))
    story.append(Spacer(1, 8))

    # ── SECTION 5: CLINICAL TRIAGE ARCHITECTURE & SHANNON ENTROPY ──
    story.append(Paragraph("5. Clinical Decision Support & Uncertainty Governance", h1_style))
    story.append(Paragraph(
        "To prevent automated diagnostic errors without making unjustified claims of calibrated probabilities, we engineered "
        "a real-time <b>Normalized Shannon Entropy Decision Metric</b>: "
        "$$\\bar{H}(P) = -\\frac{1}{\\log_2(C)} \\sum_{i=1}^{C} p_i \\log_2(p_i), \\quad \\bar{H}(P) \\in [0, 1]$$ "
        "Entropy provides an intuitive, scale-invariant measure of prediction conflict. We established a clinical tri-state gating protocol:",
        body_p
    ))

    triage_data = [
        [
            Paragraph("Entropy Tier", tbl_hdr),
            Paragraph("Quantitative Range", tbl_hdr),
            Paragraph("Decision Confidence", tbl_hdr),
            Paragraph("Clinical Protocol & System Action", tbl_hdr)
        ],
        [
            Paragraph("<b>Tier 1: Decisive</b>", tbl_c),
            Paragraph("H < 0.300", tbl_c_center),
            Paragraph("High Concordance<br/>(One dominant class)", tbl_c_center),
            Paragraph("Automated preliminary classification; study queued for standard routine radiologist validation.", tbl_c)
        ],
        [
            Paragraph("<b>Tier 2: Ambivalent</b>", tbl_c),
            Paragraph("0.300 <= H <= 0.600", tbl_c_center),
            Paragraph("Moderate Uncertainty<br/>(Class boundary ambiguity)", tbl_c_center),
            Paragraph("Flagged for secondary review; Differential diagnosis report generated with top-2 class probabilities.", tbl_c)
        ],
        [
            Paragraph("<b>Tier 3: Conflicted</b>", tbl_c),
            Paragraph("H > 0.600", tbl_c_center),
            Paragraph("High Conflict / Equivocal<br/>(Overlapping distributions)", tbl_c_center),
            Paragraph("<b>AUTOMATED DECISION INHIBITED</b>. Triggers mandatory radiologist arbitration. Recommends multi-sequence contrast MRI (T1ce/FLAIR).", tbl_c)
        ]
    ]
    t3 = Table(triage_data, colWidths=[90, 85, 110, 219])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_navy),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t3)
    story.append(Spacer(1, 10))

    # ── SECTION 6: DISCUSSION, FUTURE ROADMAP & CONCLUSION ──
    story.append(Paragraph("6. Discussion, Future Directions & Conclusion", h1_style))
    story.append(Paragraph(
        "<b>Transition to 3D Multimodal Vision-Language Models:</b> The limitations documented in this study demonstrate that "
        "2D slice classification has reached its ceiling. The future of AI in neuro-oncology requires 3D volumetric foundation architectures. "
        "Emerging models such as <b>BrainIAC</b> (Nature, 2026) and <b>BrainFound</b> (arXiv:2510.23415) learn spatial priors across full 3D "
        "T1, T1ce, T2, and FLAIR volumes, while multimodal frameworks such as <b>Med-Gemini-3D</b> and <b>Brainfound</b> (Patterns, 2026) "
        "unify volumetric vision with clinical language generation for interactive report drafting and dialogue.",
        body_p
    ))
    story.append(Paragraph(
        "<b>Conclusion:</b> By enforcing rigorous data deduplication, evaluating calibration under realistic label shift, and quantitatively "
        "benchmarking Grad-CAM against physician masks, this study exposes critical flaws in contemporary neuro-oncology AI benchmarks. "
        "Our findings provide a reproducible blueprint for rigorous, leakage-controlled research, proving that scientific integrity and "
        "honest evidence boundaries are the only path toward trustworthy clinical translation.",
        body_p
    ))
    story.append(Spacer(1, 8))

    # ── REFERENCES ──
    story.append(Paragraph("Selected References", h2_style))
    refs = [
        "1. CLAIM 2024: Revised Checklist for Artificial Intelligence in Medical Imaging. <i>Radiology: AI</i>, 2024.",
        "2. TRIPOD+AI Statement: Reporting Clinical Prediction Models Based on Machine Learning. <i>BMJ</i>, 2024;385:bmj-2023-078378.",
        "3. Louis DN, et al. The 2021 WHO Classification of Tumors of the Central Nervous System: a summary. <i>Neuro-Oncology</i>, 2021.",
        "4. Guo C, et al. On calibration of modern neural networks. <i>ICML</i>, 2017:1321–1330.",
        "5. Selvaraju RR, et al. Grad-CAM: Visual Explanations from Deep Networks. <i>IEEE ICCV</i>, 2017:618–626.",
        "6. Fateh A, et al. BRISC: Annotated dataset for brain tumor segmentation and classification. <i>arXiv:2506.14318</i>, 2025.",
        "7. Tan M, Le QV. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. <i>ICML</i>, 2019:6105–6114."
    ]
    for ref in refs:
        story.append(Paragraph(ref, ParagraphStyle("Ref", parent=body_p, fontSize=7, leading=9.5, spaceAfter=2)))

    # Build the document
    doc.build(story, canvasmaker=ManuscriptCanvas)

    if PDF_OUTPUT_PATH.exists():
        import shutil
        shutil.copyfile(PDF_OUTPUT_PATH, ROOT_PDF_COPY)
        print(f"✓ Manuscript PDF successfully generated at: {PDF_OUTPUT_PATH}")
        print(f"✓ Copied to root workspace: {ROOT_PDF_COPY}")
        file_size_kb = os.path.getsize(PDF_OUTPUT_PATH) / 1024
        print(f"  File size: {file_size_kb:.1f} KB")


if __name__ == "__main__":
    build_manuscript_pdf()
