from typing import TypedDict, List, Dict, Optional, Annotated
import operator


class ParsedJD(TypedDict):
    role_title: str
    required_skills: List[str]
    nice_to_have_skills: List[str]
    min_experience_years: float
    responsibilities: List[str]


class ParsedResume(TypedDict):
    candidate_name: str
    skills: List[str]
    experience_years: float
    projects: List[str]
    education: str
    raw_text: str


class MatchResult(TypedDict):
    candidate_name: str
    resume_id: str
    match_score: float
    skill_match_pct: float
    experience_fit: float
    matched_skills: List[str]
    missing_skills: List[str]
    gap_analysis: str
    suggestions: List[str]


class ResumeMatchState(TypedDict):
    """Ek single resume ke fan-out branch ka state"""
    resume_id: str
    resume_text: str
    parsed_jd: ParsedJD
    parsed_resume: Optional[ParsedResume]
    match_result: Optional[MatchResult]


class MainState(TypedDict):
    jd_text: str
    parsed_jd: Optional[ParsedJD]
    resumes: List[Dict[str, str]]
    match_results: Annotated[List[MatchResult], operator.add]
    ranked_results: Optional[List[MatchResult]]