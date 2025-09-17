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
import plotly.express as px
from collections import Counter

# Sidebar settings
st.sidebar.title("⚙️ Settings")
API_KEY = st.sidebar.text_input("🔑 OpenRouter API Key", type="password")
top_n = st.sidebar.slider("Top N CVs to Display", min_value=1, max_value=50, value=30)
scoring_mode = st.sidebar.radio("Scoring Mode", ["Hybrid", "AI Only"])
min_score = st.sidebar.slider("Minimum Score Filter", min_value=0, max_value=100, value=0)
keyword_filter = st.sidebar.text_input("Keyword Filter (optional)")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": "https://your-app-url.com",
    "X-Title": "CV Matcher App"
}

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
            return textract.process(file.name).decode("utf-8", errors="ignore")
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
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern = r"(\+?\d{1,3}[\s-]?)?(\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}"
    emails = re.findall(email_pattern, text)
    phones = ["".join(p).strip() for p in re.findall(phone_pattern, text) if any(p)]
    return emails, phones

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

def hybrid_match_score(cv_text, jd_text):
    ai_score, ai_explanation = get_match_score_qwen(cv_text, jd_text)
    jd_keywords = set(re.findall(r'\b\w+\b', jd_text.lower()))
    cv_words = set(re.findall(r'\b\w+\b', cv_text.lower()))
    matched_keywords = jd_keywords.intersection(cv_words)
    keyword_score = int((len(matched_keywords) / len(jd_keywords)) * 100) if jd_keywords else 0
    keyword_explanation = f"{len(matched_keywords)} of {len(jd_keywords)} keywords matched."

    experience_score = 20 if re.search(r'\b\d+\s+(years|year)\s+experience\b', cv_text.lower()) else 0
    education_score = 20 if re.search(r'\b(bachelor|master|phd|mba)\b', cv_text.lower()) else 0
    skills_score = 30 if re.search(r'\b(skills|proficient|expert|tools|technologies)\b', cv_text.lower()) else 0
    cert_score = 10 if re.search(r'\b(certified|certification|certificate)\b', cv_text.lower()) else 0
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

def generate_summary(cv_text):
    prompt = f"Summarize this CV in 3-5 sentences:\n{cv_text}"
    payload = {
        "model": "qwen/qwen3-4b:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def generate_heatmap(text):
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    common_words = word_counts.most_common(20)
    df = pd.DataFrame(common_words, columns=["Word", "Frequency"])
    fig = px.bar(df, x="Word", y="Frequency", title="Top 20 Keywords in Job Description")
    st.plotly_chart(fig)

st.title("📄 Enhanced AI CV Matcher")

jd_file = st.file_uploader("Upload Job Description", type=["pdf", "docx", "doc", "rtf"])
cv_files = st.file_uploader("Upload CVs", type=["pdf", "docx", "doc", "rtf"], accept_multiple_files=True)

if jd_file and cv_files and API_KEY:
    jd_text = extract_text(jd_file)
    generate_heatmap(jd_text)

    cv_texts, cv_names = [], []
    for cv_file in cv_files:
        text = extract_text(cv_file)
        if text:
            cv_texts.append(text)
            cv_names.append(cv_file.name)

    scores, explanations, emails, phones, summaries = [], [], [], [], []
    progress = st.progress(0)
    for i, text in enumerate(cv_texts):
        if scoring_mode == "Hybrid":
            score, explanation = hybrid_match_score(text, jd_text)
        else:
            score, explanation = get_match_score_qwen(text, jd_text)
        email_list, phone_list = extract_contacts(text)
        summary = generate_summary(text)
        scores.append(score)
        explanations.append(explanation)
        emails.append(", ".join(email_list) if email_list else "Not found")
        phones.append(", ".join(phone_list) if phone_list else "Not found")
        summaries.append(summary)
        progress.progress((i + 1) / len(cv_texts))

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = [i for i in ranked if scores[i] >= min_score and (keyword_filter.lower() in cv_texts[i].lower() if keyword_filter else True)][:top_n]

    st.subheader(f"🏆 Top {len(top_indices)} Matching CVs")
    for i in top_indices:
        with st.expander(f"{cv_names[i]} — Score: {scores[i]}"):
            st.markdown(f"📋 **Summary:**\n{summaries[i]}")
            st.markdown(f"📧 **Email:** {emails[i]}")
            st.markdown(f"📞 **Phone:** {phones[i]}")
            st.markdown("🧠 **Explanation:**")
            st.text(explanations[i])
