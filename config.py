import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Base paths ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
PROMPTS_DIR = BASE_DIR / "prompts"

# Input
JD_PATH = DATA_DIR / "input" / "job_description.txt"
CVS_DIR = DATA_DIR / "input" / "cvs"

# Processed
MARKDOWN_DIR = DATA_DIR / "processed" / "markdown"
EXTRACTED_DIR = DATA_DIR / "processed" / "extracted"
JD_RUBRIC_PATH = DATA_DIR / "processed" / "jd_rubric.json"
CANDIDATE_ID_MAPPING_PATH = DATA_DIR / "processed" / "candidate_id_mapping.json"

# Output
SCORED_DIR = DATA_DIR / "output" / "scored"
RANKED_LIST_PATH = DATA_DIR / "output" / "ranked_list.json"
REPORT_PATH = DATA_DIR / "output" / "report.md"
DECISION_RECEIPTS_DIR = DATA_DIR / "output" / "decision_receipts"

# ─── API configuration ────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

# Safety check — fail loudly if key is missing
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. "
        "Copy .env.example to .env and add your API key."
    )

# ─── Extraction configuration ─────────────────────────────────────────────────
EXTRACTION_BATCH_SIZE = 5
EXTRACTION_MAX_RETRIES = 2
SCORING_BATCH_SIZE = 1

# ─── Rubric weights (must sum to 100) ─────────────────────────────────────────
# These are used as fallback only if JD parsing fails to produce weights

DEFAULT_WEIGHTS = {
    "institution_prestige": 20,
    "internship_quality": 25,
    "technical_knowledge": 45,
    "academic_performance": 10,
}

# Technical knowledge sub-score split (must sum to 100)
TECHNICAL_SUBSCORE_SPLIT = {
    "jd_alignment": 80,      
    "semantic_depth": 20,    
}

PASS3_CAPS = {
    "moot_court": 5,
    "publications": 5,
    "international_exposure": 4,
    "leadership": 3,
    "languages": 3,
}
PASS3_TOTAL_CAP = 15

# ─── Tier thresholds (out of 115 total) ───────────────────────────────────────

TIERS = {
    "Strong":            (90, 115),   
    "Competitive":       (70, 89),    
    "Weak":              (50, 69),    
    "Below Threshold":   (0,  49),    
}

# ─── RAG / ChromaDB configuration ────────────────────────────────────────────

CHROMA_COLLECTION_CVS = "candidate_cv_sections"
CHROMA_COLLECTION_INSTITUTIONS = "knowledge_institutions"
CHROMA_COLLECTION_ORGANISATIONS = "knowledge_organisations"
CHROMA_COLLECTION_CAPM_TERMS = "knowledge_capm_terms"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of chunks to retrieve per RAG query
RAG_TOP_K = 3

# ─── Knowledge base file paths ───────────────────────────────────────────────

INSTITUTIONS_KB_PATH = KNOWLEDGE_BASE_DIR / "institutions.json"
ORGANISATIONS_KB_PATH = KNOWLEDGE_BASE_DIR / "organisations.json"
CAPM_TERMS_KB_PATH = KNOWLEDGE_BASE_DIR / "capm_terms.json"

# ─── Prompt file paths ───────────────────────────────────────────────────────

JD_PARSE_PROMPT_PATH = PROMPTS_DIR / "jd_parse_prompt.txt"
EXTRACT_ANONYMISE_PROMPT_PATH = PROMPTS_DIR / "extract_and_anonymise.txt"
SCORE_ALL_CRITERIA_PROMPT_PATH = PROMPTS_DIR / "score_all_criteria.txt"

# ─── Dashboard configuration ─────────────────────────────────────────────────

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 5000
DASHBOARD_DEBUG = False

# ─── Supported CV file extensions ────────────────────────────────────────────

SUPPORTED_CV_EXTENSIONS = [".pdf", ".docx", ".txt"]
