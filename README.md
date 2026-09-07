<<<<<<< HEAD
# JD-Resume Match Engine

An AI-powered system that scores how well a candidate's resume matches a
job description, using LLM-based structured extraction + hybrid
exact/semantic skill matching.

## How it works
1. **Parse** — JD and resume text are converted to structured JSON
   (role, required/nice-to-have skills, experience) via an LLM (Groq,
   `gpt-oss-120b`).
2. **Match** — required and nice-to-have skills are compared using
   exact match → alias/synonym match → semantic (embedding) similarity
   as a fallback, with named technologies restricted to exact/alias
   match only (to avoid false positives like "Random Forest" ≈
   "scikit-learn").
3. **Score** — a weighted blend of required-skill match, nice-to-have
   match, and experience fit produces a 0-100 score and a
   weak/moderate/strong tier.

## Eval
A 15-pair hand-labeled golden dataset (`eval/golden_dataset.json`) is
used to validate scoring and missing-skill detection.

Run: `python -m eval.run_eval`

**Current results:**
- Score-band accuracy: 80%
- Tier agreement: 80%
- Missing-skill F1: 0.909

## Known limitations
- LLM-based extraction is not fully deterministic even at temperature=0;
  parse results are cached (`eval/.parse_cache/`) for eval reproducibility.
- Score calibration is tuned against a small hand-labeled dataset and
  reflects some inherent subjectivity in human tier judgments.

## Tech
Python, LangChain, Groq (`gpt-oss-120b`), sentence embeddings for
semantic skill matching.
=======
# resume_match
>>>>>>> 773dd2e3a7eb69e87f0b6f31f9b87acc9843def9
