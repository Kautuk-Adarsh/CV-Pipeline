import json
from datetime import datetime
from config import (
    SCORED_DIR,
    RANKED_LIST_PATH,
    REPORT_PATH,
    DECISION_RECEIPTS_DIR,
    CANDIDATE_ID_MAPPING_PATH,
    TIERS,
)


def run() -> dict:
    """
    Step 3: Aggregate all scorecards into final outputs.

    No LLM calls in this step — pure aggregation and formatting.

    Reads:  data/output/scored/candidate_XXX_scorecard.json
            data/processed/candidate_id_mapping.json
    Writes: data/output/ranked_list.json
            data/output/report.md
            data/output/decision_receipts/candidate_XXX_receipt.json

    Returns the ranked list dict.
    """
    DECISION_RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all scorecards
    scorecard_files = sorted(SCORED_DIR.glob("C-*_scorecard.json"))
    if not scorecard_files:
        raise FileNotFoundError(f"No scorecards found in {SCORED_DIR}")

    print(f"  Loading {len(scorecard_files)} scorecard(s)...")
    scorecards = []
    for f in scorecard_files:
        scorecard = json.loads(f.read_text(encoding="utf-8"))
        scorecards.append(scorecard)

    # Load candidate ID → original filename mapping
    # Names are re-attached here — they were never seen by the scoring model
    id_mapping = {}
    if CANDIDATE_ID_MAPPING_PATH.exists():
        id_mapping = json.loads(CANDIDATE_ID_MAPPING_PATH.read_text(encoding="utf-8"))

    # Build ranked list
    ranked_list = _build_ranked_list(scorecards, id_mapping)

    # Save outputs
    _save_ranked_list(ranked_list)
    _save_report(ranked_list)
    _save_decision_receipts(scorecards, id_mapping)

    # Print summary to terminal
    _print_terminal_summary(ranked_list)

    return ranked_list


def _build_ranked_list(scorecards: list[dict], id_mapping: dict) -> dict:
    """
    Group candidates by tier and sort by score within each tier.
    """
    # Sort all candidates by final score descending
    sorted_candidates = sorted(
        scorecards,
        key=lambda x: x.get("final_score", 0),
        reverse=True,
    )

    # Group by tier
    tier_groups = {tier: [] for tier in TIERS.keys()}
    tier_groups["Disqualified"] = []
    tier_groups["Error"] = []

    for scorecard in sorted_candidates:
        tier = scorecard.get("tier", "Error")
        candidate_id = scorecard.get("candidate_id")
        original_filename = id_mapping.get(candidate_id, "Unknown")

        entry = {
            "rank": None,  # Assigned below within tier
            "candidate_id": candidate_id,
            "original_file": original_filename,
            "tier": tier,
            "final_score": scorecard.get("final_score", 0),
            "pass2_score": scorecard.get("pass2", {}).get("total_pass2_score", 0),
            "pass3_score": scorecard.get("pass3", {}).get("total_pass3_score", 0),
            "gate_passed": scorecard.get("pass1", {}).get("overall_pass1", False),
            "summary": scorecard.get("summary", ""),
            "score_breakdown": {
                "institution": scorecard.get("pass2", {}).get("criterion_2_1", {}).get("score", 0),
                "internship": scorecard.get("pass2", {}).get("criterion_2_2", {}).get("score", 0),
                "technical": scorecard.get("pass2", {}).get("criterion_2_3", {}).get("total_score", 0),
                "academic": scorecard.get("pass2", {}).get("criterion_2_4", {}).get("score", 0),
                "bonus": scorecard.get("pass3", {}).get("total_pass3_score", 0),
            },
        }

        if tier in tier_groups:
            tier_groups[tier].append(entry)
        else:
            tier_groups["Error"].append(entry)

    # Assign ranks within each tier
    overall_rank = 1
    for tier_name in ["Strong", "Competitive", "Weak", "Below Threshold", "Disqualified", "Error"]:
        for i, entry in enumerate(tier_groups.get(tier_name, []), start=1):
            entry["rank"] = overall_rank
            overall_rank += 1

    return {
        "generated_at": datetime.now().isoformat(),
        "total_candidates": len(scorecards),
        "tier_summary": {tier: len(group) for tier, group in tier_groups.items() if group},
        "tiers": tier_groups,
    }


def _save_ranked_list(ranked_list: dict) -> None:
    RANKED_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    RANKED_LIST_PATH.write_text(json.dumps(ranked_list, indent=2), encoding="utf-8")
    print(f"  Ranked list saved: {RANKED_LIST_PATH}")


def _save_report(ranked_list: dict) -> None:
    """Generate human-readable Markdown report."""
    lines = []
    lines.append("# CV Screening Report — Meridian Law Partners LLP")
    lines.append(f"**Generated:** {ranked_list['generated_at']}")
    lines.append(f"**Total candidates evaluated:** {ranked_list['total_candidates']}")
    lines.append("")

    # Tier summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Tier | Count | Recommendation |")
    lines.append("|------|-------|---------------|")
    tier_recommendations = {
        "Strong": "Recommend for interview",
        "Competitive": "Consider for interview",
        "Weak": "Hold",
        "Below Threshold": "Do not proceed",
        "Disqualified": "Failed gate check",
        "Error": "Technical error — review manually",
    }
    for tier, count in ranked_list["tier_summary"].items():
        rec = tier_recommendations.get(tier, "")
        lines.append(f"| {tier} | {count} | {rec} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Candidates by tier
    tier_colors = {
        "Strong": "🟢",
        "Competitive": "🔵",
        "Weak": "🟡",
        "Below Threshold": "🔴",
        "Disqualified": "⛔",
        "Error": "⚠️",
    }

    for tier_name, candidates in ranked_list["tiers"].items():
        if not candidates:
            continue

        emoji = tier_colors.get(tier_name, "")
        lines.append(f"## {emoji} {tier_name} Candidates")
        lines.append("")

        for candidate in candidates:
            cid = candidate["candidate_id"]
            score = candidate["final_score"]
            breakdown = candidate["score_breakdown"]
            summary = candidate.get("summary", "")

            lines.append(f"### Rank {candidate['rank']} — {cid} | Score: {score}/115")
            lines.append("")
            lines.append("**Score breakdown:**")
            lines.append(f"- Institution prestige (20 pts): {breakdown.get('institution', 0)}")
            lines.append(f"- Internship quality (25 pts): {breakdown.get('internship', 0)}")
            lines.append(f"- Technical knowledge (45 pts): {breakdown.get('technical', 0)}")
            lines.append(f"- Academic performance (10 pts): {breakdown.get('academic', 0)}")
            lines.append(f"- Bonus points (max 15): +{breakdown.get('bonus', 0)}")
            lines.append("")
            if summary:
                lines.append(f"**Summary:** {summary}")
                lines.append("")
            lines.append("---")
            lines.append("")

    lines.append("*This report was generated automatically by the CV Screening Pipeline.*")
    lines.append("*All scoring is evidence-based — see individual scorecards for full details.*")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Markdown report saved: {REPORT_PATH}")


def _save_decision_receipts(scorecards: list[dict], id_mapping: dict) -> None:
    """
    Save lightweight decision receipt per candidate.
    Contains minimum information needed to defend any scoring decision.
    Not the full scorecard — just the key evidence trail.
    """
    for scorecard in scorecards:
        candidate_id = scorecard.get("candidate_id")
        receipt = {
            "candidate_id": candidate_id,
            "original_file": id_mapping.get(candidate_id, "Unknown"),
            "generated_at": datetime.now().isoformat(),
            "gate_result": scorecard.get("pass1", {}).get("overall_pass1", False),
            "final_score": scorecard.get("final_score", 0),
            "tier": scorecard.get("tier", "Unknown"),
            "key_evidence": {
                "degree": scorecard.get("pass1", {}).get("gate_1_1_degree", {}).get("evidence", ""),
                "institution": scorecard.get("pass2", {}).get("criterion_2_1", {}).get("evidence", ""),
                "best_internship_evidence": scorecard.get("pass2", {}).get("criterion_2_2", {}).get("overall_reasoning", ""),
                "technical_evidence": scorecard.get("pass2", {}).get("criterion_2_3", {}).get("subscore_a", {}).get("evidence", ""),
                "academic_evidence": scorecard.get("pass2", {}).get("criterion_2_4", {}).get("evidence", ""),
            },
        }

        receipt_path = DECISION_RECEIPTS_DIR / f"{candidate_id}_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print(f"  Decision receipts saved: {DECISION_RECEIPTS_DIR}")


def _print_terminal_summary(ranked_list: dict) -> None:
    print("\n" + "=" * 50)
    print("  RESULTS SUMMARY")
    print("=" * 50)
    print(f"  Total candidates:   {ranked_list['total_candidates']}")

    for tier, count in ranked_list["tier_summary"].items():
        print(f"  {tier:<20} {count}")

    print("=" * 50)
    print(f"\n  Report:     {REPORT_PATH}")
    print(f"  Ranked list: {RANKED_LIST_PATH}")
    print(f"  Scorecards:  {SCORED_DIR}")
    print(f"  Receipts:    {DECISION_RECEIPTS_DIR}")


if __name__ == "__main__":
    run()
