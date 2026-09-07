from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import tempfile
import os

from src.graph.builder import run_bulk_match
from src.parsers.resume_parser import extract_text_from_pdf

app = FastAPI(title="JD-Resume Match Engine")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/match/bulk")
async def match_bulk(jd_text: str = Form(...), resumes: List[UploadFile] = File(...)):
    resume_payload = []
    for i, file in enumerate(resumes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            text = extract_text_from_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)
        resume_payload.append({"resume_id": f"r{i}_{file.filename}", "resume_text": text})

    ranked = run_bulk_match(jd_text, resume_payload)
    return {"ranked_candidates": ranked}