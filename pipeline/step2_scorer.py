import json
import time
from google import genai
from google.genai import types
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    EXTRACTED_DIR,
    JD_RUBRIC_PATH,
    SCORED_DIR,
    SCORE_ALL_CRITERIA_PROMPT_PATH,
)
from rag.retriever import build_scoring_rag_context

client = genai.Client(api_key=GEMINI_API_KEY)


def run(candidate_ids: list[str] = None) -> list[str]:
    """
    Step 2: Score all candidates against the rubric.

    ISOLATION GUARANTEE:
    Each candidate is scored in a completely separate Gemini API call.
    No candidate's data is ever in the same context as another candidate's data.
    This prevents model drift and in-context comparison bias.

    Reads:  data/processed/extracted/candidate_XXX.json
            data/processed/jd_rubric.json
            vector_store/ (ChromaDB, via RAG retriever)
    Writes: data/output/scored/candidate_XXX_scorecard.json

    Args:
        candidate_ids: Optional list of IDs to score. If None, scores all extracted profiles.

    Returns list of successfully scored candidate IDs.
    """
    SCORED_DIR.mkdir(parents=True, exist_ok=True)

    # Load rubric (generated once from JD)
    if not JD_RUBRIC_PATH.exists():
        raise FileNotFoundError(
            f"JD rubric not found at {JD_RUBRIC_PATH}. Run Step 0A first."
        )

    jd_rubric = json.loads(JD_RUBRIC_PATH.read_text(encoding="utf-8"))
    prompt_template = SCORE_ALL_CRITERIA_PROMPT_PATH.read_text(encoding="utf-8")

    # Collect all extracted profiles
    if candidate_ids:
        profiles_to_score = [EXTRACTED_DIR / f"{cid}.json" for cid in candidate_ids]
    else:
        profiles_to_score = sorted(EXTRACTED_DIR.glob("C-*.json"))

    if not profiles_to_score:
        raise FileNotFoundError(f"No extracted profiles found in {EXTRACTED_DIR}")

    print(f"  Scoring {len(profiles_to_score)} candidate(s)")
    print(f"  Each candidate scored in complete isolation — 1 API call per candidate")

    scored_ids = []

    for i, profile_path in enumerate(profiles_to_score, start=1):
        candidate_id = profile_path.stem
        print(f"\n  [{i}/{len(profiles_to_score)}] Scoring {candidate_id}...")

        try:
            scorecard = _score_candidate(
                candidate_id=candidate_id,
                profile_path=profile_path,
                jd_rubric=jd_rubric,
                prompt_template=prompt_template,
            )

            # Save scorecard
            out_path = SCORED_DIR / f"{candidate_id}_scorecard.json"
            out_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
            scored_ids.append(candidate_id)

            # Print quick summary
            final_score = scorecard.get("final_score", 0)
            tier = scorecard.get("tier", "Unknown")
            passed_gate = scorecard.get("pass1", {}).get("overall_pass1", False)
            print(
                f"    Score: {final_score}/115 | Tier: {tier} | Gate: {'PASS' if passed_gate else 'FAIL'}"
            )

        except Exception as e:
            print(f"    Error scoring {candidate_id}: {e}")
            # Write error scorecard so the output step can handle it
            error_scorecard = _error_scorecard(candidate_id, str(e))
            out_path = SCORED_DIR / f"{candidate_id}_scorecard.json"
            out_path.write_text(json.dumps(error_scorecard, indent=2), encoding="utf-8")

        # Brief pause between API calls to avoid rate limiting
        if i < len(profiles_to_score):
            time.sleep(1)

    print(
        f"\n  Scoring complete: {len(scored_ids)}/{len(profiles_to_score)} candidates scored"
    )
    return scored_ids


def _score_candidate(
    candidate_id: str,
    profile_path,
    jd_rubric: dict,
    prompt_template: str,
) -> dict:
    """
    Score a single candidate in complete isolation.

    1. Load the candidate's extracted profile
    2. Build RAG context (relevant CV sections + knowledge base entries)
    3. Build the scoring prompt
    4. Call Gemini (1 API call — only this candidate's data is in context)
    5. Parse and validate the response
    6. Return the scorecard
    """
    # Load profile
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    # Check for extraction failure
    if profile.get("_extraction_failed"):
        print(
            f"    Warning: Extraction failed for {candidate_id} — scoring with null profile"
        )

    # Build RAG context — retrieved from ChromaDB for this candidate only
    print(f"    Building RAG context...")
    rag_context = build_scoring_rag_context(candidate_id, profile)

    # Build prompt — inject all three variables
    prompt = prompt_template
    prompt = prompt.replace("{candidate_profile}", json.dumps(profile, indent=2))
    prompt = prompt.replace("{jd_rubric}", json.dumps(jd_rubric, indent=2))
    prompt = prompt.replace("{rag_context}", rag_context)

    # Call Gemini — ONLY this candidate's data is in this call
    print(f"    Calling Gemini for scoring...")
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    raw = response.text.strip()

    try:
        scorecard = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned invalid JSON for {candidate_id}: {e}")

    # Validate and fix common issues
    scorecard = _validate_scorecard(scorecard, candidate_id)

    return scorecard


def _validate_scorecard(scorecard: dict, candidate_id: str) -> dict:
    """
    Validate scorecard structure and fix common issues.
    Ensures scores are within bounds and final_score is correct.
    """
    # Ensure candidate_id is set correctly
    scorecard["candidate_id"] = candidate_id

    # If Pass 1 failed, zero out Pass 2 and 3
    if not scorecard.get("pass1", {}).get("overall_pass1", True):
        scorecard["final_score"] = 0
        scorecard["tier"] = "Disqualified"
        return scorecard

    # Cap Pass 2 total at 100
    pass2 = scorecard.get("pass2", {})
    total_pass2 = pass2.get("total_pass2_score", 0)
    if total_pass2 > 100:
        print(f"    Warning: Pass 2 score {total_pass2} > 100, capping at 100")
        pass2["total_pass2_score"] = 100

    # Cap Pass 3 total at 15
    pass3 = scorecard.get("pass3", {})
    total_pass3 = pass3.get("total_pass3_score", 0)
    if total_pass3 > 15:
        print(f"    Warning: Pass 3 score {total_pass3} > 15, capping at 15")
        pass3["total_pass3_score"] = 15

    # Recalculate final score to ensure consistency
    final = pass2.get("total_pass2_score", 0) + pass3.get("total_pass3_score", 0)
    scorecard["final_score"] = min(final, 115)

    # Validate tier
    score = scorecard["final_score"]
    if score >= 90:
        scorecard["tier"] = "Strong"
    elif score >= 70:
        scorecard["tier"] = "Competitive"
    elif score >= 50:
        scorecard["tier"] = "Weak"
    else:
        scorecard["tier"] = "Below Threshold"

    return scorecard


def _error_scorecard(candidate_id: str, error_message: str) -> dict:
    """
    Returns a scorecard indicating a scoring error.
    The candidate appears in the output with 0 score and a clear error note.
    """
    return {
        "candidate_id": candidate_id,
        "pass1": {
            "overall_pass1": False,
            "gate_1_1_degree": {
                "result": "error",
                "evidence": "Scoring failed",
                "reasoning": error_message,
            },
            "gate_1_2_backlogs": {
                "result": "error",
                "evidence": "Scoring failed",
                "reasoning": error_message,
            },
        },
        "pass2": {"total_pass2_score": 0},
        "pass3": {"total_pass3_score": 0},
        "final_score": 0,
        "tier": "Error",
        "summary": f"Scoring failed for this candidate due to a technical error: {error_message}",
        "_scoring_error": True,
    }


if __name__ == "__main__":
    run()
