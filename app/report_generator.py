
from datetime import datetime
from io import BytesIO

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2878C8")
LIGHT_BLUE = colors.HexColor("#EAF3FC")
DARK_GREY = colors.HexColor("#344054")
LIGHT_GREY = colors.HexColor("#F4F7FA")


def generate_pdf_report(
    image,
    predicted_class,
    confidence,
    probabilities,
):
    """Generate and return a PDF report as bytes."""

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=40,
        bottomMargin=40,
        title="Brain MRI Analysis Report",
        author="NeuroScan AI",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=26,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=DARK_GREY,
        alignment=TA_CENTER,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=NAVY,
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=DARK_GREY,
    )

    story = []

    story.append(Paragraph("NEUROSCAN AI", title_style))
    story.append(
        Paragraph("Brain MRI Analysis Report", subtitle_style)
    )
    story.append(Spacer(1, 12))

    generated_at = datetime.now().strftime(
        "%d %B %Y, %I:%M %p"
    )

    metadata = [
        ["Report generated", generated_at],
        ["Analysis type", "Brain MRI image classification"],
    ]

    metadata_table = Table(
        metadata,
        colWidths=[1.7 * inch, 4.5 * inch],
    )

    metadata_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
            ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.white),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )

    story.append(metadata_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Uploaded MRI Image", heading_style))

    if isinstance(image, PILImage.Image):
        image_buffer = BytesIO()
        image.convert("RGB").save(image_buffer, format="PNG")
        image_buffer.seek(0)
    else:
        image_buffer = BytesIO(image)
        image_buffer.seek(0)

    report_image = Image(
        image_buffer,
        width=3.2 * inch,
        height=3.2 * inch,
        kind="proportional",
    )
    report_image.hAlign = "CENTER"

    story.append(report_image)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Prediction Result", heading_style))

    formatted_class = predicted_class.replace("_", " ").title()
    formatted_confidence = f"{confidence * 100:.2f}%"

    result_table = Table(
        [
            ["Predicted Class", formatted_class],
            ["Model Confidence", formatted_confidence],
        ],
        colWidths=[2.2 * inch, 4.0 * inch],
    )

    result_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
            ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
            ("TEXTCOLOR", (1, 0), (1, -1), DARK_GREY),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ])
    )

    story.append(result_table)

    story.append(
        Paragraph("Class-wise Probability Scores", heading_style)
    )

    probability_rows = [["Class", "Probability"]]

    for class_name, probability in probabilities.items():
        probability_rows.append([
            class_name.replace("_", " ").title(),
            f"{probability * 100:.2f}%",
        ])

    probability_table = Table(
        probability_rows,
        colWidths=[3.8 * inch, 2.4 * inch],
        repeatRows=1,
    )

    probability_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("TEXTCOLOR", (0, 1), (-1, -1), DARK_GREY),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GREY),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.append(probability_table)
    story.append(Spacer(1, 18))

    story.append(
        Paragraph(
            "End of Report",
            ParagraphStyle(
                "ReportFooter",
                parent=body_style,
                alignment=TA_CENTER,
                textColor=BLUE,
                fontName="Helvetica-Bold",
            ),
        )
    )

    document.build(story)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
