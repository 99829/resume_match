import json
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from jd_resume_match.llm_client import invoke_with_fallback
load_dotenv()
import re
JD_EXTRACTION_PROMPT = """You are a precise information extraction system.
Extract structured information from the job description below.

Return ONLY valid JSON (no markdown, no preamble) matching this exact schema:
{{
  "role_title": "string",
  "required_skills": ["skill1", "skill2", ...],
  "nice_to_have_skills": ["skill1", ...],
  "min_experience_years": number,
  "responsibilities": ["responsibility1", ...]
}}

Rules:
- Normalize skill names (e.g. "React.js" -> "React", "python3" -> "Python").
- If experience isn't mentioned, estimate from seniority level (e.g. "fresher" = 0, "senior" = 5).
- Keep lists concise (max 15 items each).
- If the JD presents alternative/interchangeable skills using "or" (e.g. "Java or Python",
  "Power BI or Tableau"), keep them as ONE list item joined by " or " exactly
  (e.g. "Java or Python") rather than splitting into separate required skills.
  Only split into separate items when skills are genuinely both required (joined by "and"
  or listed with commas as a plain series).

Job Description:
{jd_text}
"""





def parse_jd(jd_text: str) -> dict:
    prompt = JD_EXTRACTION_PROMPT.format(jd_text=jd_text)

    for attempt in range(2):
        response = invoke_with_fallback(prompt,tag="jd")
        print("RAW CONTENT:", repr(response.content))
        print("RESPONSE METADATA:", response.response_metadata)
        content = response.content.strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)
        if content.startswith("```"):
            content = content.strip("`")
            content = content.replace("json\n", "", 1) if content.startswith("json\n") else content
        try:
            parsed = json.loads(content)
            break
        except json.JSONDecodeError:
            if attempt == 1:
                parsed = {}
            continue

    parsed.setdefault("role_title", "Unknown Role")
    parsed.setdefault("required_skills", [])
    parsed.setdefault("nice_to_have_skills", [])
    parsed.setdefault("min_experience_years", 0)
    parsed.setdefault("responsibilities", [])
    return parsed
def _skill_overlap(jd_skills: list[str], resume_skills: list[str]) -> dict:
    if not jd_skills:
        return {"matched": [], "missing": [], "match_pct": 100.0}

    norm_resume = {normalize_skill(s) for s in resume_skills}
    matched, missing = [], []
    semantic_candidates = []  # list of (original_jd_skill, alternative_to_check)

    for jd_skill in jd_skills:
        alternatives = [a.strip() for a in jd_skill.split(" or ")] if " or " in jd_skill.lower() else [jd_skill]
        norm_alts = [normalize_skill(a) for a in alternatives]

        if any(na in norm_resume for na in norm_alts):
            matched.append(jd_skill)
            continue

        missing.append(jd_skill)  # tentative
        for alt, norm_alt in zip(alternatives, norm_alts):
            if norm_alt not in NAMED_TECH_SKILLS and " " not in norm_alt:
                semantic_candidates.append((jd_skill, alt))

    if semantic_candidates and resume_skills:
        alt_texts = [alt for _, alt in semantic_candidates]
        sim_matrix = semantic_similarity_matrix(alt_texts, resume_skills)
        assert sim_matrix.shape == (len(alt_texts), len(resume_skills)), sim_matrix.shape
        for row, (jd_skill, alt) in zip(sim_matrix, semantic_candidates):
            if row.max() >= SEMANTIC_MATCH_THRESHOLD and jd_skill in missing:
                missing.remove(jd_skill)
                matched.append(jd_skill)

    match_pct = (len(matched) / len(jd_skills)) * 100 if jd_skills else 100.0
    return {"matched": matched, "missing": missing, "match_pct": round(match_pct, 1)}