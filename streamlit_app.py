import streamlit as st
import fitz  # PyMuPDF
import docx
import pandas as pd
from fpdf import FPDF
import io
import requests
import re
import zipfile
import textract
import striprtf

# ✅ Load Gemini API key securely
API_KEY = st.secrets["gemini"]["key"]
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

headers = {
    "Content-Type": "application/json"
}

def test_api_key():
    test_payload = {
        "contents": [{
            "parts": [{"text": "Say hello"}]
        }]
    }
    try:
        response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, json=test_payload)
        if response.status_code == 200:
            st.success("✅ Gemini API key is working.")
        else:
            st.error(f"❌ API key failed. Status code: {response.status_code}")
            st.write(response.text)
    except Exception as e:
        st.error(f"⚠️ Error connecting to Gemini API: {e}")

def extract_text(file):
    try:
        filename = file.name.lower()
        if filename.endswith(".pdf"):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return "".join([page.get_text() for page in doc])
        elif filename.endswith(".docx"):
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif filename.endswith(".doc"):
            return textract.process(file).decode("utf-8", errors="ignore")
        elif filename.endswith(".rtf"):
            rtf_text = file.read().decode("utf-8", errors="ignore")
            return striprtf.rtf_to_text(rtf_text)
        else:
            st.warning(f"⚠️ Skipping unsupported file type: {file.name}")
            return ""
    except Exception as e:
        st.error(f"❌ Failed to read file: {file.name}")
        st.error(f"Error: {e}")
        return ""

def extract_contacts(text):
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+"
    phone_pattern = r"(\\+?\\d{1,3}[\\s-]?)?(\\(?\\d{2,4}\\)?[\\s-]?)?\\d{3,4}[\\s-]?\\d{4}"
    emails = re.findall(email_pattern, text)
    phones = ["".join(p).strip() for p in re.findall(phone_pattern, text) if any(p)]
    return emails, phones

def get_match_score_gemini(cv_text, jd_text):
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
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    try:
        response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, json=payload)
        response_json = response.json()
        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            content = response_json["candidates"][0]["content"]["parts"][0]["text"]
            score_line = content.split('\n')[0]
            explanation_line = "\n".join(content.split('\n')[1:])
            score = int(score_line.replace("Score:", "").strip())
            explanation = explanation_line.replace("Explanation:", "").strip()
            return score, explanation
        else:
            st.error("❌ Failed to parse Gemini response: 'candidates' missing or malformed.")
            st.write(response_json)
            return 0, "No explanation available due to malformed Gemini response."
    except Exception as e:
        st.error(f"❌ Failed to parse Gemini response: {e}")
        return 0, "No explanation available due to Gemini API error."

def hybrid_match_score(cv_text, jd_text):
    ai_score, ai_explanation = get_match_score_gemini(cv_text, jd_text)
    jd_keywords = set(re.findall(r'\\b\\w+\\b', jd_text.lower()))
    cv_words = set(re.findall(r'\\b\\w+\\b', cv_text.lower()))
    matched_keywords = jd_keywords.intersection(cv_words)
    keyword_score = int((len(matched_keywords) / len(jd_keywords)) * 100) if jd_keywords else 0
    keyword_explanation = f"{len(matched_keywords)} of {len(jd_keywords)} keywords matched."

    experience_score = 20 if re.search(r'\\b\\d+\\s+(years|year)\\s+experience\\b', cv_text.lower()) else 0
    education_score = 20 if re.search(r'\\b(bachelor|master|phd|mba)\\b', cv_text.lower()) else 0
    skills_score = 30 if re.search(r'\\b(skills|proficient|expert|tools|technologies)\\b', cv_text.lower()) else 0
    cert_score = 10 if re.search(r'\\b(certified|certification|certificate)\\b', cv_text.lower()) else 0
    weighted_score = experience_score + education_score + skills_score + cert_score
    weighted_explanation = f"Experience: {experience_score}, Education: {education_score}, Skills: {skills_score}, Certifications: {cert_score}"

    final_score = int((ai_score * 0.5) + (keyword_score * 0.3) + (weighted_score * 0.2))
    explanation = (
        f"🤖 AI Score: {ai_score} — {ai_explanation}\n"
        f"🔍 Keyword Match Score: {keyword_score} — {keyword_explanation}\n"
        f"📊 Weighted Criteria Score: {weighted_score} — {weighted_explanation}\n"
        f"📈 Final Hybrid Score: {final_score}"
    )
    return final_score, explanation

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

def generate_zip_of_top_cvs(files, ranked_indices, top_n=30):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for i in ranked_indices[:top_n]:
            file = files[i]
            zip_file.writestr(file.name, file.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

# 🌐 Streamlit UI
st.title("📄 AI CV Matcher (Gemini)")
st.write("Upload a Job Description and multiple CVs to find the best matches using AI.")

if st.button("🔍 Test API Key"):
    test_api_key()

jd_file = st.file_uploader("Upload Job Description (PDF, DOCX, DOC, RTF)", type=["pdf", "docx", "doc", "rtf"])
cv_files = st.file_uploader("Upload CVs (PDF, DOCX, DOC, RTF)", type=["pdf", "docx", "doc", "rtf"], accept_multiple_files=True)

if jd_file and cv_files:
    jd_text = extract_text(jd_file)
    cv_texts, cv_names, valid_files = [], [], []
    for cv_file in cv_files:
        text = extract_text(cv_file)
        if text:
            cv_texts.append(text)
            cv_names.append(cv_file.name)
            valid_files.append(cv_file)

    scores, explanations, emails, phones = [], [], [], []
    with st.spinner("Analyzing CVs with AI..."):
        for text in cv_texts:
            score, explanation = hybrid_match_score(text, jd_text)
            email_list, phone_list = extract_contacts(text)
            scores.append(score)
            explanations.append(explanation)
            emails.append(", ".join(email_list) if email_list else "Not found")
            phones.append(", ".join(phone_list) if phone_list else "Not found")

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_n = min(30, len(scores))
    top_indices = ranked[:top_n]
    top_names = [cv_names[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]
    top_explanations = [explanations[i] for i in top_indices]
    top_emails = [emails[i] for i in top_indices]
    top_phones = [phones[i] for i in top_indices]

    st.subheader(f"🏆 Top {top_n} Matching CVs")
    for name, score, explanation, email, phone in zip(top_names, top_scores, top_explanations, top_emails, top_phones):
        st.markdown(f"**{name}** — Match Score: `{score}`")
        st.write(explanation)
        st.markdown(f"📧 **Email:** {email}")
        st.markdown(f"📞 **Phone:** {phone}")
        st.markdown("---")

    excel_file = generate_excel(top_names, top_scores, top_explanations, top_emails, top_phones)
    pdf_file = generate_pdf(top_names, top_scores, top_explanations, top_emails, top_phones)
    zip_file = generate_zip_of_top_cvs(valid_files, ranked, top_n=30)

    st.download_button("📥 Download Results as Excel", data=excel_file, file_name="top_matching_cvs.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("📥 Download Results as PDF", data=pdf_file, file_name="top_matching_cvs.pdf", mime="application/pdf")
    st.download_button("📥 Download Top 30 CVs as ZIP", data=zip_file, file_name="top_30_cvs.zip", mime="application/zip")
