import json
import time
from google import genai
from google.genai import types
from pathlib import Path
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MARKDOWN_DIR,
    EXTRACTED_DIR,
    JD_RUBRIC_PATH,
    EXTRACT_ANONYMISE_PROMPT_PATH,
    EXTRACTION_BATCH_SIZE,
    EXTRACTION_MAX_RETRIES,
)
from rag.indexer import index_candidate_cv, index_knowledge_bases

client = genai.Client(api_key=GEMINI_API_KEY)


def run() -> list[str]:
    """
    Step 1: Extract structured JSON profiles from all candidate Markdown CVs.
    Processes CVs in batches of EXTRACTION_BATCH_SIZE for efficiency.
    After extraction, indexes all profiles and knowledge bases into ChromaDB.

    Reads:  data/processed/markdown/candidate_XXX.md
    Writes: data/processed/extracted/candidate_XXX.json
            vector_store/ (ChromaDB populated)
    Calls:  Gemini API (~N/BATCH_SIZE times)

    Returns list of successfully extracted candidate IDs.
    """
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all markdown files
    md_files = sorted(MARKDOWN_DIR.glob("C-*.md"))
    if not md_files:
        raise FileNotFoundError(f"No markdown CV files found in {MARKDOWN_DIR}")

    print(f"  Found {len(md_files)} markdown CVs to extract")
    print(f"  Batch size: {EXTRACTION_BATCH_SIZE} CVs per API call")

    prompt_template = EXTRACT_ANONYMISE_PROMPT_PATH.read_text(encoding="utf-8")

    # Split into batches
    batches = _make_batches(md_files, EXTRACTION_BATCH_SIZE)
    print(f"  Total batches: {len(batches)}")

    all_extracted = {}

    for batch_num, batch in enumerate(batches, start=1):
        print(f"\n  Batch {batch_num}/{len(batches)} — {[f.stem for f in batch]}")
        extracted = _process_batch(batch, prompt_template, batch_num)
        all_extracted.update(extracted)

    # Save each candidate's profile to disk
    success_ids = []
    for candidate_id, profile in all_extracted.items():
        out_path = EXTRACTED_DIR / f"{candidate_id}.json"
        out_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        success_ids.append(candidate_id)
        print(f"  Saved extracted profile: {out_path.name}")

    # Index into ChromaDB
    print(f"\n  Indexing {len(success_ids)} candidate profiles into ChromaDB...")
    for candidate_id in success_ids:
        profile_path = EXTRACTED_DIR / f"{candidate_id}.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        index_candidate_cv(candidate_id, profile)
        print(f"    Indexed {candidate_id}")

    print("\n  Indexing knowledge bases into ChromaDB...")
    index_knowledge_bases()

    print(
        f"\n  Extraction complete: {len(success_ids)}/{len(md_files)} candidates processed"
    )
    return success_ids


def _make_batches(files: list[Path], batch_size: int) -> list[list[Path]]:
    """Split a list of files into batches of batch_size."""
    return [files[i : i + batch_size] for i in range(0, len(files), batch_size)]


def _process_batch(
    batch: list[Path],
    prompt_template: str,
    batch_num: int,
) -> dict:
    """
    Process one batch of CVs — send to Gemini, parse response, return dict of profiles.
    Retries up to EXTRACTION_MAX_RETRIES times on failure.
    Falls back to single-CV processing if batch fails repeatedly.
    """
    cv_batch_text = _build_batch_text(batch)
    prompt = prompt_template.replace("{cv_batch}", cv_batch_text)

    for attempt in range(1, EXTRACTION_MAX_RETRIES + 2):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip()
            extracted = json.loads(raw)

            # Validate — every candidate in the batch should be in the response
            expected_ids = [f.stem for f in batch]
            missing = [cid for cid in expected_ids if cid not in extracted]
            if missing:
                print(f"    Warning: Missing candidates in response: {missing}")
                # Add null profiles for missing ones
                for cid in missing:
                    extracted[cid] = _null_profile(cid)

            print(
                f"    Batch {batch_num} extracted successfully ({len(extracted)} profiles)"
            )
            return extracted

        except json.JSONDecodeError as e:
            print(f"    Attempt {attempt}: JSON parse error — {e}")
            if attempt <= EXTRACTION_MAX_RETRIES:
                time.sleep(2)
            else:
                print(f"    Batch {batch_num} failed — falling back to single-CV mode")
                return _process_individually(batch, prompt_template)

        except Exception as e:
            print(f"    Attempt {attempt}: API error — {e}")
            if attempt <= EXTRACTION_MAX_RETRIES:
                time.sleep(3)
            else:
                print(f"    Batch {batch_num} failed — falling back to single-CV mode")
                return _process_individually(batch, prompt_template)

    return {}


def _process_individually(batch: list[Path], prompt_template: str) -> dict:
    """
    Fallback: process each CV in the batch one at a time.
    Used when batch processing fails after all retries.
    """
    results = {}
    for cv_path in batch:
        candidate_id = cv_path.stem
        print(f"    Individual processing: {candidate_id}")
        single_batch_text = _build_batch_text([cv_path])
        prompt = prompt_template.replace("{cv_batch}", single_batch_text)

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip()
            extracted = json.loads(raw)
            # Response might be wrapped in candidate_id key or be the profile directly
            if candidate_id in extracted:
                results[candidate_id] = extracted[candidate_id]
            else:
                results[candidate_id] = extracted
        except Exception as e:
            print(f"    Individual processing failed for {candidate_id}: {e}")
            results[candidate_id] = _null_profile(candidate_id)

        time.sleep(1)  # Brief pause between individual calls

    return results


def _build_batch_text(batch: list[Path]) -> str:
    """
    Build the batch input text for the extraction prompt.
    Each CV is clearly delimited with its candidate ID.
    """
    parts = []
    for cv_path in batch:
        candidate_id = cv_path.stem
        cv_text = cv_path.read_text(encoding="utf-8", errors="replace")
        parts.append(
            f"=== CANDIDATE ID: {candidate_id} ===\n"
            f"{cv_text}\n"
            f"=== END OF {candidate_id} ==="
        )
    return "\n\n".join(parts)


def _null_profile(candidate_id: str) -> dict:
    """
    Returns a null profile for a candidate when extraction fails.
    All fields are null — this candidate will score 0 on all criteria.
    """
    return {
        "candidate_id": candidate_id,
        "degree_type": None,
        "degree_status": None,
        "year_of_study": None,
        "institution_name": None,
        "cgpa": None,
        "cgpa_scale": None,
        "cgpa_trend": None,
        "current_backlogs": None,
        "academic_awards": None,
        "class_rank": None,
        "internships": [],
        "publications": None,
        "moot_courts": None,
        "languages": None,
        "leadership_roles": None,
        "international_exposure": None,
        "other_certifications": None,
        "_extraction_failed": True,
    }


if __name__ == "__main__":
    run()
