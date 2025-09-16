import streamlit as st
import fitz  # PyMuPDF
import docx
import pandas as pd
from fpdf import FPDF
import io
import requests
import re

# Load API key from Streamlit secrets
API_KEY = st.secrets["OPENROUTER_API_KEY"]
API_URL = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": "https://your-app-url.com",  # Optional
    "X-Title": "CV Matcher App"                  # Optional
}

# Extract text from PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Extract email and phone
def extract_contacts(text):
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern = r"(\+?\d{1,3}[\s-]?)?(\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}"
    emails = re.findall(email_pattern, text)
    phones = ["".join(p).strip() for p in re.findall(phone_pattern, text) if any(p)]
    return emails, phones

# AI scoring via Qwen3-4B
def get_match_score_qwen(cv_text, jd_text):
    prompt = f"""
Compare the following CV to the job description and return:
1. A match score from 0 to 100
2. A short explanation of why this CV matches or doesn't match

Job Description:
{jd_text}

CV:
{cv_text}

Respond in this format:
Score: <number>
Explanation: <text>
"""
    payload = {
        "model": "qwen/qwen3-4b:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    content = response.json()["choices"][0]["message"]["content"]
    score_line = content.split('\n')[0]
    explanation_line = "\n".join(content.split('\n')[1:])
    score = int(score_line.replace("Score:", "").strip())
    explanation = explanation_line.replace("Explanation:", "").strip()
    return score, explanation

# Generate Excel
def generate_excel(names, scores, explanations, emails, phones):
    df = pd.DataFrame({
        "CV Name": names,
        "Match Score": scores,
        "Explanation": explanations,
        "Email": emails,
        "Phone": phones
    })
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Top Matches')
    output.seek(0)
    return output

# Generate PDF
def generate_pdf(names, scores, explanations, emails, phones):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Top Matching CVs", ln=True, align='C')
    pdf.ln(10)
    for name, score, explanation, email, phone in zip(names, scores, explanations, emails, phones):
        pdf.multi_cell(0, 10, txt=f"{name} — Score: {score}\n{explanation}\n📧 Email: {email}\n📞 Phone: {phone}\n", align='L')
        pdf.ln(5)
    return io.BytesIO(pdf.output(dest='S').encode('latin1'))

# Streamlit UI
st.title("📄 AI CV Matcher (Qwen3-4B)")
st.write("Upload a Job Description and multiple CVs to find the best matches using AI.")

jd_file = st.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])
cv_files = st.file_uploader("Upload CVs (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if jd_file and cv_files:
    jd_text = extract_text_from_pdf(jd_file) if jd_file.name.endswith(".pdf") else extract_text_from_docx(jd_file)

    cv_texts, cv_names = [], []
    for cv_file in cv_files:
        text = extract_text_from_pdf(cv_file) if cv_file.name.endswith(".pdf") else extract_text_from_docx(cv_file)
        cv_texts.append(text)
        cv_names.append(cv_file.name)

    scores, explanations, emails, phones = [], [], [], []
    with st.spinner("Analyzing CVs with AI..."):
        for text in cv_texts:
            score, explanation = get_match_score_qwen(text, jd_text)
            email_list, phone_list = extract_contacts(text)
            scores.append(score)
            explanations.append(explanation)
            emails.append(", ".join(email_list) if email_list else "Not found")
            phones.append(", ".join(phone_list) if phone_list else "Not found")

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_names = [cv_names[i] for i in ranked]
    top_scores = [scores[i] for i in ranked]
    top_explanations = [explanations[i] for i in ranked]
    top_emails = [emails[i] for i in ranked]
    top_phones = [phones[i] for i in ranked]

    st.subheader("🏆 Top Matching CVs")
    for name, score, explanation, email, phone in zip(top_names, top_scores, top_explanations, top_emails, top_phones):
        st.markdown(f"**{name}** — Match Score: `{score}`")
        st.write(explanation)
        st.markdown(f"📧 **Email:** {email}")
        st.markdown(f"📞 **Phone:** {phone}")
        st.markdown("---")

    excel_file = generate_excel(top_names, top_scores, top_explanations, top_emails, top_phones)
    pdf_file = generate_pdf(top_names, top_scores, top_explanations, top_emails, top_phones)

    st.download_button("📥 Download Results as Excel", data=excel_file, file_name="top_matching_cvs.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("📥 Download Results as PDF", data=pdf_file, file_name="top_matching_cvs.pdf", mime="application/pdf")
