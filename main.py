import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
import sys
import time
import webbrowser
import threading
from pathlib import Path

from pipeline import step0a_jd_processor
from pipeline import step0b_cv_parser
from pipeline import step1_extractor
from pipeline import step2_scorer
from pipeline import step3_output
from config import DASHBOARD_HOST, DASHBOARD_PORT


def print_step(step_num: int, total: int, description: str) -> None:
    print(f"\n[{step_num}/{total}] {description}...")


def print_done(elapsed: float) -> None:
    print(f"  done  ({elapsed:.1f}s)")


def check_inputs() -> None:
    """Verify all required input files exist before starting."""
    from config import JD_PATH, CVS_DIR

    errors = []
    if not JD_PATH.exists():
        errors.append(f"Job description not found: {JD_PATH}")

    if not CVS_DIR.exists() or not any(CVS_DIR.iterdir()):
        errors.append(f"No CV files found in: {CVS_DIR}")

    if errors:
        print("\nStartup check failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


def launch_dashboard() -> None:
    """Start Flask dashboard and open browser after a short delay."""
    from dashboard.app import app

    def open_browser():
        time.sleep(1.5)
        url = f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"
        print(f"\n  Opening dashboard: {url}")
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"\n  Dashboard running at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print("  Press Ctrl+C to stop.\n")

    app.run(
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        debug=False,
        use_reloader=False,
    )


def main() -> None:
    print("\n" + "=" * 55)
    print("  CV SCREENING PIPELINE")
    print("  Meridian Law Partners LLP — Capital Markets")
    print("=" * 55)

    total_steps = 6
    start_total = time.time()

    # Pre-flight check
    check_inputs()

    # ── Step 0A: Parse JD and generate rubric ────────────────
    print_step(1, total_steps, "Parsing job description and generating rubric")
    t = time.time()
    rubric = step0a_jd_processor.run()
    print_done(time.time() - t)

    # ── Step 0B: Parse and anonymise CVs ─────────────────────
    print_step(2, total_steps, "Parsing and anonymising CVs")
    t = time.time()
    id_mapping = step0b_cv_parser.run()
    cv_count = len(id_mapping)
    print_done(time.time() - t)

    # ── Step 1: Extract structured profiles + index ChromaDB ─
    print_step(3, total_steps, f"Extracting profiles from {cv_count} CV(s) and building vector store")
    t = time.time()
    extracted_ids = step1_extractor.run()
    print_done(time.time() - t)

    if not extracted_ids:
        print("\nExtraction produced no results. Check logs above.")
        sys.exit(1)

    # ── Step 2: Score all candidates (isolated per candidate) ─
    print_step(4, total_steps, f"Scoring {len(extracted_ids)} candidate(s)")
    t = time.time()
    scored_ids = step2_scorer.run(candidate_ids=extracted_ids)
    print_done(time.time() - t)

    # ── Step 3: Generate ranked list + report + receipts ─────
    print_step(5, total_steps, "Generating ranked output and report")
    t = time.time()
    ranked_list = step3_output.run()
    print_done(time.time() - t)

    # ── Summary ───────────────────────────────────────────────
    total_elapsed = time.time() - start_total
    tier_summary = ranked_list.get("tier_summary", {})

    print(f"\n[6/{total_steps}] Pipeline complete ({total_elapsed:.1f}s total)")
    print("\n" + "=" * 55)
    print("  FINAL RESULTS")
    print("=" * 55)
    print(f"  Total candidates:        {cv_count}")
    print(f"  Successfully scored:     {len(scored_ids)}")

    for tier, count in tier_summary.items():
        if count > 0:
            print(f"  {tier:<25} {count}")

    print("=" * 55)

    # ── Launch dashboard ──────────────────────────────────────
    print_step(6, total_steps, "Launching dashboard")
    launch_dashboard()


if __name__ == "__main__":
    main()