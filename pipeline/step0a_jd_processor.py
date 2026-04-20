import json
from google import genai
from google.genai import types
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    JD_PATH,
    JD_RUBRIC_PATH,
    JD_PARSE_PROMPT_PATH,
    DEFAULT_WEIGHTS,
)

client = genai.Client(api_key=GEMINI_API_KEY)


def run() -> dict:
    """
    Step 0A: Parse the Job Description and generate a scoring rubric.

    Reads:  data/input/job_description.txt
    Writes: data/processed/jd_rubric.json
    Calls:  Gemini API (1 time — fixed cost regardless of CV count)

    Returns the rubric dict for use by downstream steps.
    """
    print("  Reading job description...")
    jd_text = JD_PATH.read_text(encoding="utf-8")

    prompt_template = JD_PARSE_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{jd_text}", jd_text)

    print("  Sending JD to Gemini for rubric generation...")
    response = client.models.generate_content(
    model=GEMINI_MODEL,
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
    ),
)
    raw = response.text.strip()

    # Parse and validate
    try:
        rubric = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  Warning: Gemini returned invalid JSON — {e}")
        print("  Falling back to default weights...")
        rubric = _build_fallback_rubric()

    # Validate weights sum to 100
    rubric = _validate_and_fix_weights(rubric)

    # Save to disk
    JD_RUBRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    JD_RUBRIC_PATH.write_text(json.dumps(rubric, indent=2), encoding="utf-8")

    print(f"  Rubric saved to {JD_RUBRIC_PATH}")
    _print_rubric_summary(rubric)

    return rubric


def _validate_and_fix_weights(rubric: dict) -> dict:
    """
    Ensure criterion weights sum to exactly 100.
    If not, normalise them proportionally.
    """
    weights = rubric.get("criterion_weights", {})
    total = sum(weights.values())

    if total != 100:
        print(f"  Warning: Weights sum to {total}, normalising to 100...")
        factor = 100 / total
        normalised = {k: round(v * factor) for k, v in weights.items()}
        # Fix any rounding error on the largest weight
        diff = 100 - sum(normalised.values())
        largest_key = max(normalised, key=normalised.get)
        normalised[largest_key] += diff
        rubric["criterion_weights"] = normalised

    return rubric


def _build_fallback_rubric() -> dict:
    """
    Returns a safe default rubric if Gemini's response is unparseable.
    Uses config.py DEFAULT_WEIGHTS.
    """
    return {
        "criterion_weights": DEFAULT_WEIGHTS,
        "semantic_domain_map": {
            "core_domain": "Indian Capital Markets and Securities Law",
            "high_relevance_adjacent": ["Mergers and Acquisitions", "Banking and Finance", "Private Equity"],
            "low_relevance_adjacent": ["General Corporate", "Commercial Contracts"],
            "not_relevant": ["Criminal Law", "Family Law", "Intellectual Property", "Litigation"],
        },
        "jd_keywords": [
            "SEBI", "ICDR", "LODR", "DRHP", "IPO", "QIP", "Rights Issue",
            "FEMA", "NSE", "BSE", "Due Diligence", "Offer Document",
            "Merchant Banker", "Preferential Allotment", "FPI",
        ],
        "_fallback": True,
    }


def _print_rubric_summary(rubric: dict) -> None:
    weights = rubric.get("criterion_weights", {})
    keywords = rubric.get("jd_keywords", [])
    domain = rubric.get("semantic_domain_map", {}).get("core_domain", "Unknown")
    print(f"  Core domain: {domain}")
    print(f"  Weights: {weights}")
    print(f"  JD keywords extracted: {len(keywords)}")


if __name__ == "__main__":
    run()
