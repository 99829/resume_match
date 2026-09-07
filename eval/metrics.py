"""
Eval metrics for the JD-Resume Match Engine.

Metrics computed:
1. Score-band accuracy: does predicted match_score fall in the human-labeled
   expected_score_range?
2. Missing-skill precision/recall: does the scorer correctly identify the
   human-labeled missing required skills?
3. Tier agreement: does predicted tier match the label?
"""
from src.matching.normalization import normalize_skill

from typing import List, Dict


def score_in_range(predicted_score: float, expected_range: List[float]) -> bool:
    lo, hi = expected_range
    return lo <= predicted_score <= hi


def tier_from_score(score: float, required_skill_match_pct: float = 100.0) -> str:
    if score >= 80:
        tier = "strong_match"
    elif score >= 30:
        tier = "moderate_match"
    else:
        tier = "weak_match"

    if required_skill_match_pct < 100 and tier == "strong_match":
        tier = "moderate_match"          # even one missing required skill blocks "strong"
    if required_skill_match_pct < 50:
        tier = "weak_match"
    return tier

def missing_skills_precision_recall(predicted_missing: List[str], expected_missing: List[str]) -> Dict:
    pred_set = {normalize_skill(s) for s in predicted_missing}
    exp_set = {normalize_skill(s) for s in expected_missing}

    if not exp_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0}
    if not pred_set:
        return {"precision": 1.0 if not exp_set else 0.0, "recall": 0.0 if exp_set else 1.0}
    if not exp_set:
        return {"precision": 0.0, "recall": 1.0}

    true_positives = len(pred_set & exp_set)
    precision = true_positives / len(pred_set)
    recall = true_positives / len(exp_set)
    return {"precision": round(precision, 3), "recall": round(recall, 3)}

def aggregate_eval_results(per_pair_results: List[Dict]) -> Dict:
    n = len(per_pair_results)
    score_band_accuracy = sum(r["score_in_range"] for r in per_pair_results) / n
    tier_agreement = sum(r["tier_match"] for r in per_pair_results) / n
    avg_precision = sum(r["precision"] for r in per_pair_results) / n
    avg_recall = sum(r["recall"] for r in per_pair_results) / n
    f1 = (
        2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0.0
    )

    return {
        "num_pairs": n,
        "score_band_accuracy": round(score_band_accuracy, 3),
        "tier_agreement": round(tier_agreement, 3),
        "missing_skills_precision": round(avg_precision, 3),
        "missing_skills_recall": round(avg_recall, 3),
        "missing_skills_f1": round(f1, 3),
    }