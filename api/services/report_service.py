"""Generate a clearly labelled, non-diagnostic research summary PDF."""

import io
import base64
from datetime import datetime
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class ReportService:
    """Handles in-memory generation of research model summaries."""

    def generate_pdf_report(
        self,
        predicted_class: str,
        confidence: float,
        predictions: dict,
        model_used: str,
        heatmap_base64: str = None,
        patient_name: str = "Anonymous Patient",
        patient_id: str = "N/A",
        comments: str = "",
        calibrated: bool = False,
        confidence_threshold: float = 0.7,
    ) -> io.BytesIO:
        """
        Creates a PDF report in-memory and returns a BytesIO buffer.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=54,
            leftMargin=54,
            topMargin=54,
            bottomMargin=54
        )

        styles = getSampleStyleSheet()

        # Define restrained research-report styles
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#1E293B'), # Slate 800
            spaceAfter=15
        )

        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#64748B'), # Slate 500
            spaceAfter=25
        )

        section_heading = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#0F172A'), # Slate 900
            spaceBefore=15,
            spaceAfter=10
        )

        body_style = ParagraphStyle(
            'BodyText',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#334155') # Slate 700
        )

        bold_body_style = ParagraphStyle(
            'BoldBodyText',
            parent=body_style,
            fontName='Helvetica-Bold'
        )

        table_header_style = ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            textColor=colors.white
        )

        story = []

        # ── Header Section ──
        story.append(Paragraph("AI NeuroOnco — Research Model Output", title_style))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (UTC) · Public-data 2D proof-of-concept", subtitle_style))
        story.append(Spacer(1, 10))

        # ── Patient & Study Information Table ──
        info_data = [
            [
                Paragraph("<b>Patient Name:</b>", body_style),
                Paragraph(patient_name, body_style),
                Paragraph("<b>Date of Analysis:</b>", body_style),
                Paragraph(datetime.now().strftime('%Y-%m-%d'), body_style)
            ],
            [
                Paragraph("<b>Patient / Study ID:</b>", body_style),
                Paragraph(patient_id, body_style),
                Paragraph("<b>Model Architecture:</b>", body_style),
                Paragraph(model_used.upper(), body_style)
            ]
        ]
        info_table = Table(info_data, colWidths=[1.5*inch, 2.0*inch, 1.5*inch, 2.0*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 20))

        # ── Primary model-output banner ──
        formatted_class = predicted_class.replace('_', ' ').title()
        is_uncertain = confidence < confidence_threshold
        banner_bg = colors.HexColor('#FEF3C7') if is_uncertain else colors.HexColor('#ECFDF5') # Amber vs Emerald
        banner_border = colors.HexColor('#F59E0B') if is_uncertain else colors.HexColor('#10B981')
        banner_text = colors.HexColor('#78350F') if is_uncertain else colors.HexColor('#065F46')

        confidence_label = "Calibrated confidence" if calibrated else "Uncalibrated model score"
        banner_msg = f"<b>Dataset-label output:</b> {formatted_class} ({confidence * 100:.1f}% {confidence_label})"
        if is_uncertain:
            banner_msg += f"<br/><i>Score is below the prespecified {confidence_threshold:.0%} research abstention threshold.</i>"

        banner_data = [[Paragraph(banner_msg, ParagraphStyle('Banner', parent=body_style, textColor=banner_text, fontSize=12, leading=16))]]
        banner_table = Table(banner_data, colWidths=[7*inch])
        banner_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), banner_bg),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOX', (0,0), (-1,-1), 1.5, banner_border),
            ('TOPPADDING', (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ]))
        story.append(banner_table)
        story.append(Spacer(1, 20))

        # ── Side-by-Side: Class Probabilities & Heatmap ──
        story.append(Paragraph("Research Output & Visual Explanation", section_heading))

        # Probabilities table data
        prob_data = [[Paragraph("Class", table_header_style), Paragraph("Probability", table_header_style)]]
        for cls, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            cls_name = cls.replace('_', ' ').title()
            prob_data.append([
                Paragraph(cls_name, body_style),
                Paragraph(f"{prob * 100:.1f}%", bold_body_style if cls == predicted_class else body_style)
            ])

        prob_table = Table(prob_data, colWidths=[1.8*inch, 1.2*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ]))

        # Grad-CAM visual image flowable
        img_flowable = None
        if heatmap_base64:
            try:
                img_data = base64.b64decode(heatmap_base64)
                img_buf = io.BytesIO(img_data)

                # Resize keeping aspect ratio
                pil_img = PILImage.open(img_buf)
                aspect = pil_img.height / pil_img.width
                img_width = 3.2 * inch
                img_height = img_width * aspect

                img_flowable = Image(img_buf, width=img_width, height=img_height)
            except Exception as e:
                print(f"[ReportService] Heatmap PDF extraction failed: {e}")

        # Assemble details layout table
        layout_data = [
            [
                prob_table,
                img_flowable if img_flowable else Paragraph("Grad-CAM visualization not available for this analysis run.", body_style)
            ]
        ]
        layout_table = Table(layout_data, colWidths=[3.2*inch, 3.8*inch])
        layout_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (1,0), (1,0), 20), # Spacer between cols
        ]))
        story.append(layout_table)
        story.append(Spacer(1, 20))

        # ── Notes & Comments Section ──
        if comments:
            story.append(Paragraph("Research Notes", section_heading))
            comments_data = [[Paragraph(comments.replace('\n', '<br/>'), body_style)]]
            comments_table = Table(comments_data, colWidths=[7*inch])
            comments_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ]))
            story.append(comments_table)
            story.append(Spacer(1, 25))

        # ── Disclaimer / Footer ──
        disclaimer = (
            "<b>Research-only disclaimer:</b> This output is from a retrospective public-data 2D proof-of-concept. "
            "It is not a radiology report, medical device, diagnosis, prognosis, triage result, or treatment recommendation. "
            "The four outputs are dataset labels rather than WHO CNS integrated diagnoses. Grad-CAM is an experimental "
            "visual explanation and is not a validated radiological marker."
        )
        story.append(Paragraph(disclaimer, ParagraphStyle('DisclaimerStyle', parent=body_style, fontSize=8, textColor=colors.HexColor('#64748B'), leading=11)))

        doc.build(story)
        buffer.seek(0)
        return buffer
