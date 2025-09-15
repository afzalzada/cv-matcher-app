import streamlit as st
import fitz  # PyMuPDF
import docx
import pandas as pd
from fpdf import FPDF
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

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

# Streamlit UI
st.title("📄 CV Matcher App")
st.write("Upload a Job Description (JD) and multiple CVs (PDF or DOCX) to find the top matches.")

# Upload JD
jd_file = st.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])

# Upload CVs
cv_files = st.file_uploader("Upload CVs (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if jd_file and cv_files:
    # Extract JD text
    if jd_file.name.lower().endswith(".pdf"):
        jd_text = extract_text_from_pdf(jd_file)
    elif jd_file.name.lower().endswith(".docx"):
        jd_text = extract_text_from_docx(jd_file)
    else:
        st.error("Unsupported JD file format.")
        st.stop()

    # Extract CV texts
    cv_texts = []
    cv_names = []
    for cv_file in cv_files:
        if cv_file.name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(cv_file)
        elif cv_file.name.lower().endswith(".docx"):
            text = extract_text_from_docx(cv_file)
        else:
            st.warning(f"Unsupported file format for {cv_file.name}. Skipping.")
            continue
        cv_texts.append(text)
        cv_names.append(cv_file.name)

    if not cv_texts:
        st.error("No valid CVs uploaded.")
        st.stop()

    # Combine JD and CVs for TF-IDF
    documents = [jd_text] + cv_texts
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute cosine similarity between JD and each CV
    jd_vector = tfidf_matrix[0]
    cv_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(jd_vector, cv_vectors).flatten()

    # Rank CVs by similarity
    ranked_indices = similarities.argsort()[::-1]
    top_indices = ranked_indices[:20]
    top_cv_names = [cv_names[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]

    # Display results
    st.subheader("🏆 Top Matching CVs")
    for name, score in zip(top_cv_names, top_scores):
        st.write(f"{name} - Similarity Score: {score:.4f}")

    # Generate Excel and PDF files
    excel_file = generate_excel(top_cv_names, top_scores)
    pdf_file = generate_pdf(top_cv_names, top_scores)

    # Download buttons
    st.download_button(
        label="📥 Download Results as Excel",
        data=excel_file,
        file_name="top_matching_cvs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.download_button(
        label="📥 Download Results as PDF",
        data=pdf_file,
        file_name="top_matching_cvs.pdf",
        mime="application/pdf"
    )
