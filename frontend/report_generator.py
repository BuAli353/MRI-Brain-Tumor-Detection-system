import os
import random
import datetime
import tempfile
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def generate_pdf_report(original_pil: PILImage.Image, heatmap_pil: PILImage.Image, label: str, confidence: float, scores: dict, latency: float, model_choice: str) -> str:
    """
    Generates a professional clinical PDF report for the MRI brain tumor analysis.
    Saves the report inside a 'reports' directory in the frontend workspace and returns the path.
    """
    # 1. Create reports directory in the frontend folder
    frontend_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(frontend_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate a unique Report ID
    report_id = f"MRI-{random.randint(1000, 9999)}-{random.randint(10, 99)}"
    pdf_filename = os.path.join(reports_dir, f"MRI_Report_{report_id}.pdf")
    
    # 2. Save PIL images to temporary files for ReportLab to import
    temp_orig = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_orig_path = temp_orig.name
    temp_orig.close()
    
    temp_heat = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_heat_path = temp_heat.name
    temp_heat.close()
    
    # Save PIL images to those paths
    original_pil.convert("RGB").save(temp_orig_path, format="JPEG")
    if heatmap_pil is not None:
        heatmap_pil.convert("RGB").save(temp_heat_path, format="JPEG")
    else:
        # Fallback to original image if no heatmap exists
        original_pil.convert("RGB").save(temp_heat_path, format="JPEG")
        
    # 3. Initialize PDF document
    # Margins: 0.5 inches (36 points) for maximum content space
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    # 4. Define Document Styles
    styles = getSampleStyleSheet()
    
    # Custom styles to avoid name collisions and ensure strict leading/sizing
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1a365d'), # Navy clinical blue
        spaceAfter=0
    )
    
    meta_title_style = ParagraphStyle(
        'MetaTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2d3748'),
        alignment=2 # Right aligned
    )
    
    meta_val_style = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#4a5568'),
        alignment=2 # Right aligned
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2b6cb0'), # Medical blue accent
        spaceBefore=14,
        spaceAfter=6
    )
    
    banner_style = ParagraphStyle(
        'BannerText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        alignment=1 # Centered
    )
    
    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#2d3748')
    )
    
    cell_normal = ParagraphStyle(
        'CellNormal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#4a5568')
    )
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#718096'),
        alignment=1
    )
    
    story = []
    
    # ── HEADER SECTION ────────────────────────────────────────────────────────
    # Logo / Title on Left, Report Meta on Right
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    header_data = [
        [
            Paragraph("🧠 CLINICAL MRI ANALYSIS REPORT", title_style),
            Paragraph(f"<b>Report ID:</b> {report_id}<br/><b>Date:</b> {now_str}", meta_val_style)
        ]
    ]
    
    # Available width is 8.5" - 1.0" = 7.5" = 540 points
    header_table = Table(header_data, colWidths=[340, 200])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(header_table)
    
    # Colored horizontal separator line
    line_table = Table([[""]], colWidths=[540])
    line_table.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 2, colors.HexColor('#2b6cb0')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(line_table)
    story.append(Spacer(1, 10))
    
    # ── METADATA ROW ──────────────────────────────────────────────────────────
    metadata_data = [
        [
            Paragraph("<b>Patient Name:</b>", cell_bold), Paragraph("N/A (Clinical Session)", cell_normal),
            Paragraph("<b>Analysis Model:</b>", cell_bold), Paragraph(model_choice, cell_normal),
        ],
        [
            Paragraph("<b>Tracking ID:</b>", cell_bold), Paragraph(report_id, cell_normal),
            Paragraph("<b>Inference Latency:</b>", cell_bold), Paragraph(f"{latency} ms", cell_normal),
        ]
    ]
    metadata_table = Table(metadata_data, colWidths=[90, 180, 100, 170])
    metadata_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f7fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(metadata_table)
    story.append(Spacer(1, 12))
    
    # ── DIAGNOSTIC HIGHLIGHT BANNER ───────────────────────────────────────────
    is_tumor = label != "No Tumor"
    if is_tumor:
        banner_bg = colors.HexColor('#fed7d7')    # Soft Red
        banner_border = colors.HexColor('#feb2b2') # Light Red Border
        banner_text_color = colors.HexColor('#9b1c1c') # Dark Red Text
        banner_text = f"🚨 CLINICAL DETECTION: YES ({label.upper()} DETECTED) — CONFIDENCE: {confidence * 100:.2f}%"
    else:
        banner_bg = colors.HexColor('#c6f6d5')    # Soft Green
        banner_border = colors.HexColor('#9ae6b4') # Light Green Border
        banner_text_color = colors.HexColor('#22543d') # Dark Green Text
        banner_text = f"✅ CLINICAL DETECTION: NO TUMOR DETECTED — CONFIDENCE: {confidence * 100:.2f}%"
        
    banner_p = Paragraph(f"<font color='{banner_text_color.hexval()}'>{banner_text}</font>", banner_style)
    banner_table = Table([[banner_p]], colWidths=[540])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), banner_bg),
        ('BOX', (0,0), (-1,-1), 1, banner_border),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 14))
    
    # ── SCAN COMPARISON SECTION ───────────────────────────────────────────────
    story.append(Paragraph("🔬 Imaging & Visualization", section_heading))
    
    # Resize images to fit perfectly (240x240 points each)
    img_width = 240
    img_height = 240
    
    img1 = Image(temp_orig_path, width=img_width, height=img_height)
    img2 = Image(temp_heat_path, width=img_width, height=img_height)
    
    comparison_data = [
        [img1, img2],
        [
            Paragraph("<b>Original MRI Scan</b>", ParagraphStyle('Cap1', parent=cell_bold, alignment=1)),
            Paragraph(f"<b>Model Focus Overlay ({'YOLO Seg' if model_choice == 'YOLOv8 Segmentation' else 'Grad-CAM'})</b>", ParagraphStyle('Cap2', parent=cell_bold, alignment=1))
        ]
    ]
    
    # Columns of 260 each gives 520 points (leaving 20 points padding total)
    comparison_table = Table(comparison_data, colWidths=[265, 265])
    comparison_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BOTTOMPADDING', (0,1), (-1,1), 4),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#ffffff')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor('#f7fafc')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(comparison_table)
    story.append(Spacer(1, 14))
    
    # ── CLASSIFICATION SCORES SECTION ─────────────────────────────────────────
    story.append(Paragraph("📊 Categorical Probability Breakdown", section_heading))
    
    # Create the table rows
    table_rows = [
        [
            Paragraph("<b>Diagnostic Class</b>", ParagraphStyle('Th1', parent=cell_bold, textColor=colors.white)),
            Paragraph("<b>Probability Score (%)</b>", ParagraphStyle('Th2', parent=cell_bold, textColor=colors.white, alignment=2))
        ]
    ]
    
    # Fill in the sorted scores
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    for cls, val in sorted_scores:
        is_prediction = (cls == label)
        class_name_formatted = f"<b>{cls} (Predicted)</b>" if is_prediction else cls
        percent_str = f"<b>{val * 100:.2f}%</b>" if is_prediction else f"{val * 100:.2f}%"
        
        row_bg = colors.HexColor('#edf2f7') if is_prediction else colors.white
        
        table_rows.append([
            Paragraph(class_name_formatted, cell_bold if is_prediction else cell_normal),
            Paragraph(percent_str, ParagraphStyle('ValP', parent=cell_bold if is_prediction else cell_normal, alignment=2))
        ])
        
    scores_table = Table(table_rows, colWidths=[300, 240])
    
    # Table Styling
    scores_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2b6cb0')), # Blue Header
        ('ALIGN', (0,0), (-1,0), 'LEFT'),
        ('ALIGN', (1,0), (1,-1), 'RIGHT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
    ])
    
    # Dynamic background colors for rows
    for i in range(1, len(table_rows)):
        cls_name = sorted_scores[i-1][0]
        if cls_name == label:
            # Predicted row gets highlighted background
            scores_style.add('BACKGROUND', (0, i), (-1, i), colors.HexColor('#ebf8ff'))
        elif i % 2 == 0:
            # Alternating rows get subtle light grey
            scores_style.add('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f7fafc'))
            
    scores_table.setStyle(scores_style)
    story.append(scores_table)
    story.append(Spacer(1, 20))
    
    # ── CLINICAL DISCLAIMER FOOTER ────────────────────────────────────────────
    disclaimer_text = (
        "<b>Clinical Disclaimer:</b> This report is generated automatically by a deep learning artificial intelligence model "
        "for computer-aided decision support and educational purposes only. It does NOT constitute a formal medical diagnosis or "
        "radiology opinion. The findings contained in this document must be thoroughly reviewed and verified by a certified clinical "
        "professional or radiologist before initiating any treatment planning or diagnostic decisions."
    )
    disclaimer_p = Paragraph(disclaimer_text, disclaimer_style)
    disclaimer_box = Table([[disclaimer_p]], colWidths=[540])
    disclaimer_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f7fafc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(disclaimer_box)
    
    # 5. Build Document
    doc.build(story)
    
    # 6. Clean up temporary files
    try:
        os.unlink(temp_orig_path)
        os.unlink(temp_heat_path)
    except Exception as e:
        print(f"⚠️ Error cleaning temporary image files: {e}")
        
    return pdf_filename
