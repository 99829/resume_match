from src.matching.embeddings import semantic_similarity_matrix
from src.matching.normalization import normalize_skill, NAMED_TECH_SKILLS

SEMANTIC_MATCH_THRESHOLD = 0.45  # cosine sim above this counts as "matched"

DEFAULT_WEIGHTS = {
    "required_skills": 0.55,
    "nice_to_have_skills": 0.15,
    "experience": 0.30,
}


def _skill_overlap(jd_skills: list[str], resume_skills: list[str]) -> dict:
    if not jd_skills:
        return {"matched": [], "missing": [], "match_pct": 100.0}

    norm_resume = {normalize_skill(s) for s in resume_skills}
    matched, missing = [], []
    semantic_candidates = []

    for jd_skill in jd_skills:
        alternatives = [a.strip() for a in jd_skill.split(" or ")] if " or " in jd_skill.lower() else [jd_skill]
        norm_alts = [normalize_skill(a) for a in alternatives]

        if any(na in norm_resume for na in norm_alts):
            matched.append(jd_skill)
            continue

        missing.append(jd_skill)
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

def _experience_fit(min_required: float, candidate_years: float) -> float:
    if min_required <= 0:
        return 100.0
    if candidate_years >= min_required:
        return 100.0
    ratio = candidate_years / min_required
    return round(max(ratio, 0) * 100, 1)


def _validate_weights(weights: dict) -> dict:
    """Ensures weights sum to 1.0 (100%); normalizes if they don't, so a
    recruiter passing e.g. {required_skills: 70, nice_to_have: 10, experience: 20}
    (out of 100, not fractions) still works correctly."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive number")
    return {k: v / total for k, v in weights.items()}


def compute_match_score(parsed_jd: dict, parsed_resume: dict, weights: dict = None) -> dict:
    w = _validate_weights(weights) if weights else DEFAULT_WEIGHTS

    jd_nice_to_have = parsed_jd.get("nice_to_have_skills", [])
    required = _skill_overlap(parsed_jd.get("required_skills", []), parsed_resume.get("skills", []))
    nice_to_have = _skill_overlap(jd_nice_to_have, parsed_resume.get("skills", []))
    exp_fit = _experience_fit(
        parsed_jd.get("min_experience_years", 0),
        parsed_resume.get("experience_years", 0),
    )

    active_w = dict(w)
    if not jd_nice_to_have:
        active_w["required_skills"] += active_w["nice_to_have_skills"]
        active_w["nice_to_have_skills"] = 0.0

    # A flat percentage understates how much missing even one required
    # skill matters — a candidate missing 1 of 5 required skills isn't
    # "80% as good," they're missing something the JD explicitly called
    # non-negotiable. Applying an exponent > 1 penalizes partial matches
    # progressively harder while leaving a perfect 100% match unchanged
    # (100^1.5 / 100 = 100, so already-correct pairs don't move).
    REQUIRED_SKILL_PENALTY_EXPONENT = 1.5
    required_pct_scored = round(
        (required["match_pct"] / 100) ** REQUIRED_SKILL_PENALTY_EXPONENT * 100, 1
    )

    overall_skill_pct = round(required["match_pct"] * 0.8 + nice_to_have["match_pct"] * 0.2, 1)
    final_score = round(
        required_pct_scored * active_w["required_skills"]
        + nice_to_have["match_pct"] * active_w["nice_to_have_skills"]
        + exp_fit * active_w["experience"],
        1,
    )
    

    return {
        "match_score": final_score,
        "skill_match_pct": overall_skill_pct,
        "required_skill_match_pct": required["match_pct"],  # raw %, still used for tier gating
        "experience_fit": exp_fit,
        "matched_skills": required["matched"] + nice_to_have["matched"],
        "missing_skills": required["missing"],
        "weights_used": active_w,
    }