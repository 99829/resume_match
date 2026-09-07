"""
Quick-start runner: tests the full pipeline with real Groq LLM calls.

Prerequisite: set GROQ_API_KEY environment variable before running.

Run:
    python run_demo.py
"""

import os
import sys

if not os.environ.get("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not set.")
    sys.exit(1)

from src.graph.builder import run_bulk_match

jd_text = """
We are hiring an AI/ML Engineer (Fresher level).
Required: Python, LangChain or LangGraph, RAG systems, FastAPI.
Nice to have: Docker, AWS, vector databases.
0-1 years experience acceptable.
"""

resumes = [
    {
        "resume_id": "candidate_1",
        "resume_text": """
        Harsh Sarvaiya. BSc IT 2026 graduate.
        Built a RAG pipeline using hybrid FAISS+BM25 retrieval, RRF fusion, AST-aware chunking.
        Built a LangGraph-based coding agent orchestrator with tool-calling and MCP integration.
        Skills: Python, LangChain, LangGraph, FAISS, Groq API, Docker, FastAPI.
        """,
    },
    {
        "resume_id": "candidate_2",
        "resume_text": """
        Priya Shah. BSc IT graduate, fresher.
        Completed a college project on Titanic dataset classification using scikit-learn.
        Skills: Python, pandas, scikit-learn, basic SQL.
        """,
    },
    {
        "resume_id": "candidate_3",
        "resume_text": """
        Aman Verma. 1 year experience as backend developer.
        Built an inventory management system.
        Skills: Java, Spring Boot, MySQL, REST APIs.
        """,
    },
]

if __name__ == "__main__":
    print("Running bulk match pipeline (real Groq LLM calls)...\n")
    ranked = run_bulk_match(jd_text, resumes)

    print("=== RANKED CANDIDATES ===\n")
    for i, r in enumerate(ranked, 1):
        print(f"#{i} {r['candidate_name']}  -  Score: {r['match_score']}/100")
        print(f"   Matched skills : {r['matched_skills']}")
        print(f"   Missing skills : {r['missing_skills']}")
        print(f"   Gap analysis   : {r['gap_analysis']}")
        print(f"   Suggestions    : {r['suggestions']}")
        print()