
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(algo, acc):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []
    content.append(Paragraph("Machine Learning Report", styles['Title']))
    content.append(Spacer(1,10))
    content.append(Paragraph(f"Algorithm: {algo}", styles['Normal']))
    content.append(Paragraph(f"Accuracy: {acc}", styles['Normal']))
    doc.build(content)
