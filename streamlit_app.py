# Add these imports at the top
import pandas as pd
from fpdf import FPDF
import io

# Function to generate Excel file
def generate_excel(cv_names, scores):
    df = pd.DataFrame({
        "CV Name": cv_names,
        "Similarity Score": scores
    })
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Top Matches')
    output.seek(0)
    return output

# Function to generate PDF file
def generate_pdf(cv_names, scores):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Top Matching CVs", ln=True, align='C')
    pdf.ln(10)
    for name, score in zip(cv_names, scores):
        pdf.cell(200, 10, txt=f"{name} - Similarity Score: {score:.4f}", ln=True)
    pdf_output = io.BytesIO(pdf.output(dest='S').encode('latin1'))
    return pdf_output
