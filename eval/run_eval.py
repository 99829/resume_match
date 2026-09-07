"""
Runs the golden eval set through the parse -> match pipeline and reports metrics.

Usage:
    python -m eval.run_eval
"""

import json
import sys
import os
from eval.cache_utils import cached_parse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.jd_parser import parse_jd
from src.parsers.resume_parser import parse_resume
from src.matching.scorer import compute_match_score
from eval.metrics import score_in_range, tier_from_score, missing_skills_precision_recall, aggregate_eval_results


def _status(ok: bool) -> str:
    return "✅" if ok else "❌"


def run_eval(dataset_path: str = "eval/golden_dataset.json"):
    with open(dataset_path) as f:
        dataset = json.load(f)

    per_pair_results = []

    print(f"\nRunning eval on {len(dataset['pairs'])} pairs...\n")
    print(f"{'ID':<10}{'Score':<10}{'Range':<12}{'Tier':<16}{'Expected Tier':<16}{'P':<6}{'R':<6}{'':<3}")
    print("-" * 78)

    for pair in dataset["pairs"]:
        parsed_jd = parse_jd(pair["jd_text"])
        parsed_resume = parse_resume(pair["resume_text"])
        scores = compute_match_score(parsed_jd, parsed_resume)

        predicted_score = scores["match_score"]
        predicted_tier = tier_from_score(predicted_score, scores["required_skill_match_pct"])
        pr = missing_skills_precision_recall(scores["missing_skills"], pair["expected_missing_skills"])
        in_range = score_in_range(predicted_score, pair["expected_score_range"])
        tier_match = predicted_tier == pair["expected_tier"]
        parsed_jd = cached_parse(pair["jd_text"], parse_jd, "jd")
        parsed_resume = cached_parse(pair["resume_text"], parse_resume, "resume")
        result = {
            "id": pair["id"],
            "predicted_score": predicted_score,
            "expected_range": pair["expected_score_range"],
            "score_in_range": in_range,
            "predicted_tier": predicted_tier,
            "expected_tier": pair["expected_tier"],
            "tier_match": tier_match,
            "precision": pr["precision"],
            "recall": pr["recall"],
            "predicted_missing": scores["missing_skills"],
            "expected_missing": pair["expected_missing_skills"],
        }
        per_pair_results.append(result)

        range_str = f"[{pair['expected_score_range'][0]}-{pair['expected_score_range'][1]}]"
        overall_ok = in_range and tier_match
        print(f"{pair['id']:<10}{predicted_score:<10}{range_str:<12}{predicted_tier:<16}"
              f"{pair['expected_tier']:<16}{pr['precision']:<6}{pr['recall']:<6}{_status(overall_ok)}")

        if scores["missing_skills"] != pair["expected_missing_skills"]:
            print(f"    predicted missing: {scores['missing_skills']}")
            print(f"    expected missing:  {pair['expected_missing_skills']}")

    summary = aggregate_eval_results(per_pair_results)

    print("\n" + "=" * 78)
    print("EVAL SUMMARY".center(78))
    print("=" * 78)
    print(f"  Pairs evaluated        : {summary['num_pairs']}")
    print(f"  Score-band accuracy    : {summary['score_band_accuracy']:.1%}")
    print(f"  Tier agreement         : {summary['tier_agreement']:.1%}")
    print(f"  Missing-skill precision: {summary['missing_skills_precision']:.3f}")
    print(f"  Missing-skill recall   : {summary['missing_skills_recall']:.3f}")
    print(f"  Missing-skill F1       : {summary['missing_skills_f1']:.3f}")
    print("=" * 78 + "\n")

    with open("eval/eval_results.json", "w") as f:
        json.dump({"per_pair": per_pair_results, "summary": summary}, f, indent=2)

    return summary


if __name__ == "__main__":
    run_eval()