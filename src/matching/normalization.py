import re

# Words that add no matching signal once a skill is normalized —
# "CI/CD pipelines" and "CI/CD" should compare equal.
_FILLER_WORDS = {
    "basics", "basic", "systems", "system", "pipelines", "pipeline",
    "design", "integration", "experience", "development", "or",
}

# Canonical -> known phrasing variants. Extend this as new false-negatives
# turn up in eval runs; it's cheaper and more precise than lowering the
# semantic threshold.
_ALIASES = {
    "ci/cd": {"cicd", "continuous integration", "continuous deployment",
              "jenkins ci/cd", "jenkins cicd"},
    "machine learning": {"ml", "machine learning deployment",
                          "ml deployment", "production ml deployment"},
    "rest api": {"rest apis", "restful api", "restful apis"},
    "typescript": {"ts"},
    "siem": {"splunk", "qradar", "arcsight"},
    "vector database": {"vector databases", "faiss", "pinecone", "chroma", "weaviate", "milvus"},
    "fine-tuning llms": {"fine-tuning llm", "lora", "qlora", "peft", "lora fine-tuning", "llm fine-tuning"},
    "etl": {"etl pipelines", "etl pipeline"},
}

_VARIANT_TO_CANONICAL = {}
for _canonical, _variants in _ALIASES.items():
    _VARIANT_TO_CANONICAL[_canonical] = _canonical
    for _v in _variants:
        _VARIANT_TO_CANONICAL[_v] = _canonical


def normalize_skill(raw: str) -> str:
    """Lowercase, drop filler words, collapse known aliases to one canonical form."""
    s = raw.lower().strip()
    s = re.sub(r"[^\w\s/+#.-]", "", s)  # keep C++, C#, CI/CD punctuation
    tokens = [t for t in s.split() if t not in _FILLER_WORDS]
    s = " ".join(tokens).strip()
    return _VARIANT_TO_CANONICAL.get(s, s)


# Specific named tools/algorithms/frameworks: exact-or-alias match ONLY.
# Embeddings routinely rate "Random Forest" close to "scikit-learn" or
# "XGBoost" close to "gradient boosting" because they're topically
# adjacent — but for hiring, knowing the *specific* tool is the point,
# so semantic substitution here is a false positive by definition.
NAMED_TECH_SKILLS = {
    "xgboost", "random forest", "redux", "terraform", "kubernetes",
    "docker", "typescript", "postgresql", "django", "flask", "fastapi",
    "langchain", "langgraph", "faiss", "ansible", "jenkins",
    "testng", "selenium", "prometheus", "grafana", "kotlin",
    "jetpack compose", "mvvm",
}
# jd_skills = set(s.lower() for s in jd_data["required_skills"])
# resume_skills = set(s.lower() for s in resume_data["skills"])

# matched = jd_skills & resume_skills
# missing = jd_skills - resume_skills
# extra   = resume_skills - jd_skills
def compute_skill_overlap(jd_skills: list[str], resume_skills: list[str]) -> dict:
    """
    Compares JD-required skills against resume skills using the same
    normalization/alias logic as normalize_skill(), so 'CI/CD pipelines'
    and 'CI/CD' correctly count as matched, not missing.
    """
    jd_normalized = {normalize_skill(s) for s in jd_skills}
    resume_normalized = {normalize_skill(s) for s in resume_skills}

    matched = jd_normalized & resume_normalized
    missing = jd_normalized - resume_normalized
    extra = resume_normalized - jd_normalized

    return {
        "matched": sorted(matched),
        "missing": sorted(missing),
        "extra": sorted(extra),
    }