"""
JD-Resume Match Engine — Streamlit App
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from src.jd_resume_match.llm_client import LAST_USED_MODEL
from dotenv import load_dotenv
load_dotenv()

from src.parsers.jd_parser import parse_jd
from src.parsers.resume_parser import parse_resume, extract_text_from_pdf
from src.matching.scorer import compute_match_score
from eval.metrics import tier_from_score

st.set_page_config(page_title="JD-Resume Match Engine", page_icon="🎯", layout="wide")

st.title("🎯 JD-Resume Match Engine")
st.caption("Paste a job description and upload a resume to see how well they match.")

TIER_COLORS = {"strong_match": "🟢", "moderate_match": "🟡", "weak_match": "🔴"}
TIER_LABELS = {"strong_match": "Strong Match", "moderate_match": "Moderate Match", "weak_match": "Weak Match"}


def skill_chips(skills: list[str], kind: str) -> str:
    """Renders a list of skills as color-coded HTML badges.
    kind: 'matched' (green) or 'missing' (red)."""
    if not skills:
        return "_None_"
    color = "#1f7a3f" if kind == "matched" else "#a13030"
    bg = "#173d24" if kind == "matched" else "#3d1a1a"
    chips = "".join(
        f'<span style="background:{bg};color:{color};padding:3px 10px;'
        f'border-radius:12px;margin:2px;display:inline-block;font-size:0.85rem;">{s}</span>'
        for s in skills
    )
    return chips


def process_resume(jd_text: str, resume_text: str) -> dict:
    """Parses + scores a single resume against a JD. Raises on failure —
    caller decides whether to stop (single mode) or skip-and-continue (bulk mode)."""
    parsed_jd = parse_jd(jd_text)
    parsed_resume = parse_resume(resume_text)
    scores = compute_match_score(parsed_jd, parsed_resume)
    tier = tier_from_score(scores["match_score"], scores["required_skill_match_pct"])
    return {
        "parsed_jd": parsed_jd,
        "parsed_resume": parsed_resume,
        "scores": scores,
        "tier": tier,
    }


def render_result(result: dict, heading: str = None):
    scores, tier, parsed_jd, parsed_resume = (
        result["scores"], result["tier"], result["parsed_jd"], result["parsed_resume"]
    )
    if heading:
        st.markdown(f"### {heading}")
    st.caption(f"🤖 JD parsed with: `{LAST_USED_MODEL.get('jd', 'unknown')}` | "
               f"Resume parsed with: `{LAST_USED_MODEL.get('resume', 'unknown')}`")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Match Score", f"{scores['match_score']} / 100")
    with c2:
        st.metric("Tier", f"{TIER_COLORS.get(tier, '')} {TIER_LABELS.get(tier, tier)}")
    with c3:
        st.metric("Experience Fit", f"{scores['experience_fit']}%")

    st.markdown("**✅ Matched Skills**")
    st.markdown(skill_chips(scores["matched_skills"], "matched"), unsafe_allow_html=True)
    st.markdown("**❌ Missing Skills**")
    st.markdown(skill_chips(scores["missing_skills"], "missing"), unsafe_allow_html=True)

    with st.expander("🔎 See parsed JD and resume details"):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Parsed JD**")
            st.json(parsed_jd)
        with d2:
            st.markdown("**Parsed Resume**")
            display_resume = {k: v for k, v in parsed_resume.items() if k != "raw_text"}
            st.json(display_resume)


# ---------- JD input (shared across both modes) ----------
st.subheader("📋 Job Description")
jd_text = st.text_area(
    "Paste the JD text here",
    height=250,
    placeholder="We are hiring a Python Backend Developer with 2+ years experience...",
)

st.divider()

mode = st.radio("Mode", ["Single Resume", "Bulk Mode"], horizontal=True)

# ================= SINGLE RESUME MODE =================
if mode == "Single Resume":
    st.subheader("📄 Resume")
    resume_input_mode = st.radio("Resume input method", ["Upload PDF", "Paste text"], horizontal=True)

    resume_text = ""
    if resume_input_mode == "Upload PDF":
        uploaded_file = st.file_uploader("Upload resume PDF", type=["pdf"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                resume_text = extract_text_from_pdf(tmp_path)
                with st.expander("Extracted resume text (preview)"):
                    st.text(resume_text[:1500] + ("..." if len(resume_text) > 1500 else ""))
            finally:
                os.unlink(tmp_path)
    else:
        resume_text = st.text_area("Paste resume text here", height=300)

    st.divider()
    if st.button("🔍 Match Resume to JD", type="primary", use_container_width=True):
        if not jd_text.strip():
            st.error("Please paste a job description.")
        elif not resume_text.strip():
            st.error("Please provide a resume (upload a PDF or paste text).")
        else:
            with st.spinner("Parsing JD and resume, then scoring the match..."):
                try:
                    result = process_resume(jd_text, resume_text)
                except Exception as e:
                    st.error(f"Something went wrong while parsing or scoring: {e}")
                    st.stop()
            st.success("Done!")
            st.divider()
            render_result(result)

# ================= BULK MODE =================
else:
    st.subheader("📄 Resumes (multiple)")
    uploaded_files = st.file_uploader(
        "Upload resume PDFs", type=["pdf"], accept_multiple_files=True
    )

    st.divider()
    if st.button("🔍 Match All Resumes to JD", type="primary", use_container_width=True):
        if not jd_text.strip():
            st.error("Please paste a job description.")
        elif not uploaded_files:
            st.error("Please upload at least one resume PDF.")
        else:
            rows = []
            detailed_results = {}
            errors = []
            progress = st.progress(0, text="Starting...")

            for i, uploaded_file in enumerate(uploaded_files):
                progress.progress(
                    (i) / len(uploaded_files),
                    text=f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...",
                )
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    resume_text = extract_text_from_pdf(tmp_path)
                    result = process_resume(jd_text, resume_text)
                    detailed_results[uploaded_file.name] = result
                    rows.append({
                        "Filename": uploaded_file.name,
                        "Candidate": result["parsed_resume"].get("candidate_name", "Unknown"),
                        "Score": result["scores"]["match_score"],
                        "Tier": TIER_LABELS.get(result["tier"], result["tier"]),
                        "Experience Fit": result["scores"]["experience_fit"],
                        "Missing Skills Count": len(result["scores"]["missing_skills"]),
                    })
                except Exception as e:
                    errors.append(f"{uploaded_file.name}: {e}")
                finally:
                    os.unlink(tmp_path)

            progress.progress(1.0, text="Done!")

            if errors:
                st.warning("Some resumes failed to process:\n\n" + "\n".join(f"- {e}" for e in errors))

            if rows:
                df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
                st.subheader("📊 Ranked Results")
                st.dataframe(df, use_container_width=True)

                st.divider()
                st.subheader("🔎 Per-candidate detail")
                # sort filenames by score for expander order too
                ordered_filenames = df["Filename"].tolist()
                for fname in ordered_filenames:
                    with st.expander(f"{fname} — Score: {detailed_results[fname]['scores']['match_score']}"):
                        render_result(detailed_results[fname])
            else:
                st.error("No resumes could be processed successfully.")

st.divider()
st.caption("Built with LangChain + Groq for parsing, and hybrid exact/semantic skill matching.")