import json
import os
import pdfplumber
# from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from jd_resume_match.llm_client import invoke_with_fallback
RESUME_EXTRACTION_PROMPT = """You are a precise resume-parsing system.
Extract structured information from the resume text below.

Return ONLY valid JSON (no markdown, no preamble) matching this exact schema:
{{
  "candidate_name": "string",
  "skills": ["skill1", "skill2", ...],
  "experience_years": number,
  "projects": ["short project title/desc", ...],
  "education": "string (highest degree + institute)"
}}

Rules:
- Normalize skill names same way a recruiter ATS would (e.g. "ReactJS" -> "React").
- CRITICAL for experience_years: convert durations to FRACTIONAL years, do not
  round up to whole numbers. "3 months" = 0.25, "6-month internship" = 0.5,
  "1 year 6 months" = 1.5. If multiple internships/jobs are listed, sum their
  durations. A single unpaid or paid internship of a few months is NOT the
  same as 1 full year of experience — never round a partial-year duration
  up to the nearest whole number.
- If experience is not explicit at all, estimate conservatively from project
  count and depth rather than assuming a full year.
- CRITICAL: Extract every skill, tool, technique, or technology mentioned ANYWHERE
  in the text — in the Skills section, in project bullet points, in parenthetical
  details, and in the summary — not just an explicit "Skills:" line. Do not skip a
  skill just because it only appears once, inside a sentence, or in parentheses.

  Example 1: if the text says "AWS (IAM, GuardDuty, CloudTrail)", extract all of:
  "AWS", "AWS IAM", "GuardDuty", "CloudTrail" as separate skills.

  Example 2: if the text says "Built a RAG pipeline with FAISS+BM25 hybrid
  retrieval", extract "RAG", "FAISS", "BM25" even though "RAG" and "BM25"
  never appear in a bullet-pointed Skills list.

  Example 3: if the text says "Trained Logistic Regression, Decision Tree,
  Random Forest, and XGBoost models" inside a project description, extract
  all four algorithm names as skills, not just "Machine Learning".

- Err on the side of extracting more candidate skills rather than fewer — a
  human recruiter will verify the list afterward.

Resume Text:
{resume_text}
"""


def extract_text_from_pdf(pdf_path: str) -> str:
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
    return "\n".join(text_chunks)


# def get_llm():
#     return ChatGroq(
#         model="openai/gpt-oss-20b",
#         temperature=0,
#         api_key=os.environ.get("GROQ_API_KEY"),
#         reasoning_format="hidden",
#         max_tokens=2048,
#         reasoning_effort="low",
#     )


import re

def parse_resume(resume_text: str) -> dict:
    prompt = RESUME_EXTRACTION_PROMPT.format(resume_text=resume_text)
    response = invoke_with_fallback(prompt,tag="resume")

    content = response.content.strip()

    # Robust JSON extraction — works regardless of markdown fences or
    # any stray reasoning/preamble text around the JSON.
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"Resume parser returned invalid JSON: {content[:200]}")
    content = match.group(0)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Resume parser returned invalid JSON: {content[:200]}") from e

    parsed.setdefault("candidate_name", "Unknown Candidate")
    parsed.setdefault("skills", [])
    parsed.setdefault("experience_years", 0)
    parsed.setdefault("projects", [])
    parsed.setdefault("education", "")
    parsed["raw_text"] = resume_text
    return parsed