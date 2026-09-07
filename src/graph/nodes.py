import os
from langgraph.types import Send
from langchain_groq import ChatGroq

from src.graph.state import MainState, ResumeMatchState
from src.parsers.jd_parser import parse_jd
from src.parsers.resume_parser import parse_resume
from src.matching.scorer import compute_match_score


def parse_jd_node(state: MainState) -> dict:
    parsed = parse_jd(state["jd_text"])
    return {"parsed_jd": parsed}


def fan_out_to_resumes(state: MainState) -> list[Send]:
    """Conditional edge: dispatches one Send per resume so they process in parallel."""
    return [
        Send(
            "parse_and_match_resume",
            {
                "resume_id": r["resume_id"],
                "resume_text": r["resume_text"],
                "parsed_jd": state["parsed_jd"],
                "parsed_resume": None,
                "match_result": None,
            },
        )
        for r in state["resumes"]
    ]


def parse_and_match_resume_node(state: ResumeMatchState) -> dict:
    """Runs inside the fan-out branch: parse one resume, score it, generate gap analysis."""
    parsed_resume = parse_resume(state["resume_text"])
    scores = compute_match_score(state["parsed_jd"], parsed_resume)
    gap_text, suggestions = _generate_gap_analysis(state["parsed_jd"], parsed_resume, scores)

    match_result = {
        "candidate_name": parsed_resume["candidate_name"],
        "resume_id": state["resume_id"],
        **scores,
        "gap_analysis": gap_text,
        "suggestions": suggestions,
    }
    return {"match_results": [match_result]}


def _generate_gap_analysis(parsed_jd: dict, parsed_resume: dict, scores: dict) -> tuple[str, list[str]]:
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3, api_key=os.environ.get("GROQ_API_KEY"))
    prompt = f"""Role: {parsed_jd.get('role_title')}
Missing required skills: {', '.join(scores['missing_skills']) or 'None'}
Candidate skills: {', '.join(parsed_resume.get('skills', []))}
Candidate experience: {parsed_resume.get('experience_years')} years (JD wants {parsed_jd.get('min_experience_years')}+)

Write:
1. A 2-sentence gap analysis (plain text, no markdown).
2. Exactly 3 actionable suggestions to improve this resume's fit, each on its own line prefixed with "- ".
"""
    response = llm.invoke(prompt).content.strip()
    lines = response.split("\n")
    gap_lines = [l for l in lines if not l.strip().startswith("-")]
    suggestion_lines = [l.strip("- ").strip() for l in lines if l.strip().startswith("-")]
    gap_text = " ".join(gap_lines).strip()
    return gap_text, suggestion_lines[:3]


def rank_node(state: MainState) -> dict:
    ranked = sorted(state["match_results"], key=lambda r: r["match_score"], reverse=True)
    return {"ranked_results": ranked}