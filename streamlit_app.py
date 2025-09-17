import streamlit as st
from huggingface_hub import InferenceClient
import pdfplumber
from docx import Document
import pypandoc
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import io
import re

# Streamlit app configuration
st.set_page_config(page_title="CV-JD Matcher", layout="wide")
st.title("CV-JD Matcher")
st.write("Upload a Job Description and CVs/Resumes (PDF, DOCX, DOC, RTF) to find the top 20 candidates using AI (Phi-3 via Hugging Face).")

# Initialize Hugging Face Inference Client
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
if not HF_TOKEN:
    st.error("Hugging Face API token not found in secrets.toml. Please add it to continue.")
    st.stop()
client = InferenceClient(token=HF_TOKEN)

# Function to extract text from supported file types
def extract_text(file):
    try:
        file_ext = file.name.lower().split(".")[-1]
        if file_ext == "pdf":
            with pdfplumber.open(file) as pdf:
                return " ".join(page.extract_text() or "" for page in pdf.pages)
        elif file_ext in ["docx", "doc"]:
            doc = Document(file)
            return " ".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
        elif file_ext == "rtf":
            with open("temp.rtf", "wb") as f:
                f.write(file.read())
            text = pypandoc.convert_file("temp.rtf", "plain")
            os.remove("temp.rtf")
            return text
        else:
            return None  # Unsupported file type
    except Exception as e:
        st.warning(f"Skipping file '{file.name}': Corrupted or unreadable ({str(e)})")
        return None

# Function to check if file is a CV/resume (not a cover letter)
def is_resume(file_name, text):
    # Simple heuristic: Check file name and content for CV/resume indicators
    resume_keywords = ["curriculum vitae", "resume", "cv", "work experience", "education"]
    cover_letter_keywords = ["cover letter", "dear ", "application for", "to whom"]
    name_lower = file_name.lower()
    text_lower = text.lower()
    
    # File name check
    if any(keyword in name_lower for keyword in cover_letter_keywords):
        return False
    if any(keyword in name_lower for keyword in resume_keywords):
        return True
    
    # Content check
    resume_score = sum(1 for keyword in resume_keywords if keyword in text_lower)
    cover_score = sum(1 for keyword in cover_letter_keywords if keyword in text_lower)
    return resume_score > cover_score or resume_score > 0

# Function to extract candidate details using Phi-3
def extract_candidate_details(cv_text, cv_name):
    try:
        prompt = f"""
        CV: {cv_text}
        Extract the following details from the CV:
        1. Name of applicant
        2. Current or last position held
        3. Current or last organization
        4. Total work experience (in years, estimate if not explicit)
        5. Relevant work experience (in years, for roles similar to or above the JD role, especially in telecom)
        6. Education (list all degrees, e.g., 'BSc Computer Science, MSc Data Science')
        7. Phone number
        8. Email address
        Output ONLY in this format:
        Name: [text] | Position: [text] | Organization: [text] | TotalExp: [number] | RelevantExp: [number] | Education: [text] | Phone: [text] | Email: [text]
        If a field is missing, use 'Not found'.
        """
        response = client.text_generation(
            prompt,
            model="microsoft/Phi-3-mini-4k-instruct",
            max_new_tokens=200,
            temperature=0.1
        )
        # Parse response
        details = {
            "Name": "Not found", "Position": "Not found", "Organization": "Not found",
            "TotalExp": 0, "RelevantExp": 0, "Education": "Not found",
            "Phone": "Not found", "Email": "Not found"
        }
        match = re.search(r"Name: (.+?) \| Position: (.+?) \| Organization: (.+?) \| TotalExp: (\d+) \| RelevantExp: (\d+) \| Education: (.+?) \| Phone: (.+?) \| Email: (.+)", response)
        if match:
            details["Name"] = match.group(1)
            details["Position"] = match.group(2)
            details["Organization"] = match.group(3)
            details["TotalExp"] = int(match.group(4))
            details["RelevantExp"] = int(match.group(5))
            details["Education"] = match.group(6)
            details["Phone"] = match.group(7)
            details["Email"] = match.group(8)
        return {"cv_name": cv_name, **details}
    except Exception as e:
        st.warning(f"Error extracting details for '{cv_name}': {str(e)}")
        return {"cv_name": cv_name, "Name": "Not found", "Position": "Not found", "Organization": "Not found",
                "TotalExp": 0, "RelevantExp": 0, "Education": "Not found", "Phone": "Not found", "Email": "Not found"}

# Function to score a single CV against JD using Phi-3
def score_cv(jd_text, cv_text, cv_name):
    try:
        prompt = f"""
        JD: {jd_text}
        CV: {cv_text}
        Analyze the CV against the JD and score the fit (0-100) based on the following prioritized criteria:
        1. Relevant Experience (50% weight): Prioritize experience in the same or higher role as the JD, with longer tenure in such roles being better. Experience in the telecom industry is preferred.
        2. Education (30% weight): Degrees and fields aligning with JD requirements (e.g., relevant majors, higher degrees preferred).
        3. Skills and Abilities (20% weight): Technical and soft skills matching JD requirements.
        Output ONLY in this format:
        Score: X/100 | Reasons: [1-2 sentences explaining the score, focusing on experience, education, and skills]
        """
        response = client.text_generation(
            prompt,
            model="microsoft/Phi-3-mini-4k-instruct",
            max_new_tokens=100,
            temperature=0.1
        )
        # Parse response
        score_line = [line for line in response.split("\n") if "Score:" in line]
        if score_line:
            match = re.search(r"Score: (\d+)/100 \| Reasons: (.+)", score_line[0])
            if match:
                score = int(match.group(1))
                reasons = match.group(2)
                return {"cv_name": cv_name, "score": score, "reasons": reasons}
        return {"cv_name": cv_name, "score": 0, "reasons": "Failed to parse AI response"}
    except Exception as e:
        st.warning(f"Error processing '{cv_name}': {str(e)}")
        return {"cv_name": cv_name, "score": 0, "reasons": f"Processing failed: {str(e)}"}

# File uploaders
jd_file = st.file_uploader("Upload Job Description (PDF, DOCX, DOC, RTF)", type=["pdf", "docx", "doc", "rtf"])
cv_files = st.file_uploader("Upload CVs/Resumes (PDF, DOCX, DOC, RTF)", type=["pdf", "docx", "doc", "rtf"], accept_multiple_files=True)

# Process button
if st.button("Match CVs"):
    if not jd_file or not cv_files:
        st.error("Please upload both a Job Description and at least one CV/Resume.")
    else:
        with st.spinner("Processing files..."):
            # Extract JD text
            jd_text = extract_text(jd_file)
            if not jd_text:
                st.error(f"Could not extract text from JD file '{jd_file.name}'. Please check the file.")
                st.stop()

            # Extract CV texts and filter valid resumes
            cv_data = []
            for cv_file in cv_files:
                cv_text = extract_text(cv_file)
                if cv_text and is_resume(cv_file.name, cv_text):
                    cv_data.append({"name": cv_file.name, "text": cv_text})
                else:
                    st.warning(f"Skipping '{cv_file.name}': Not a CV/Resume or corrupted.")

            if not cv_data:
                st.error("No valid CVs/Resumes could be processed. Please check file formats or content.")
                st.stop()

            # Process CVs in parallel for scoring and details
            results = []
            candidate_details = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit scoring tasks
                score_futures = {executor.submit(score_cv, jd_text, cv["text"], cv["name"]): cv for cv in cv_data}
                # Submit details extraction tasks
                details_futures = {executor.submit(extract_candidate_details, cv["text"], cv["name"]): cv for cv in cv_data}
                
                # Collect scoring results
                for future in as_completed(score_futures):
                    results.append(future.result())
                
                # Collect details results
                for future in as_completed(details_futures):
                    candidate_details.append(future.result())

            # Rank and get top 20
            top_20 = sorted(results, key=lambda x: x["score"], reverse=True)[:20]
            
            # Match details to top 20
            top_20_details = []
            for result in top_20:
                detail = next((d for d in candidate_details if d["cv_name"] == result["cv_name"]), None)
                if detail:
                    top_20_details.append({
                        "CV Name": result["cv_name"],
                        "Name": detail["Name"],
                        "Current/Last Position": detail["Position"],
                        "Current/Last Organization": detail["Organization"],
                        "Total Work Experience (Years)": detail["TotalExp"],
                        "Relevant Work Experience (Years)": detail["RelevantExp"],
                        "Education": detail["Education"],
                        "Phone": detail["Phone"],
                        "Email": detail["Email"],
                        "Notes": result["reasons"],
                        "Score": result["score"]
                    })

            # Display results
            st.subheader("Top 20 Matching CVs")
            if top_20_details:
                df = pd.DataFrame(top_20_details)
                st.table(df.drop(columns=["Score"]))  # Hide score in display for clarity
                
                # Generate Excel file for download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df.drop(columns=["Score"]).to_excel(writer, index=False, sheet_name="Top Candidates")
                st.download_button(
                    label="Download Top 20 Candidates (Excel)",
                    data=output.getvalue(),
                    file_name="top_20_candidates.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No valid results returned. Please check files or API connectivity.")
