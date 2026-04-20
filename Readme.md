# CV Screening Pipeline

Automated CV screening pipeline that converts a job description and a folder of CVs into a ranked shortlist with evidence-backed scorecards.

## Overview

This project parses CVs, anonymizes candidate data, derives a rubric from the job description, scores each candidate independently, and generates ranked outputs with an interactive Flask dashboard.

## Key Features

- Rubric is derived from the JD automatically — no hardcoded weights
- All CVs are anonymised before any scoring begins
- Every score includes the exact CV sentence that produced it
- Each candidate is scored in complete isolation — no cross-candidate comparison
- Institution and organisation tiers verified against local knowledge bases via RAG
- Results served through a local Flask dashboard with clickable scorecards
- Outputs in JSON, Markdown, and interactive HTML

---

## Tech Stack

| Layer           | Technology                                  |
|-----------------|---------------------------------------------|
| Language        | Python 3.11+                                |
| LLM             | Gemini 2.5 Flash via google-genai           |
| PDF parsing     | Docling (IBM)                               |
| Vector database | ChromaDB                                    |
| Embeddings      | all-MiniLM-L6-v2 via sentence-transformers  |
| Web framework   | Flask                                       |

## Prerequisites

- Python 3.11+
- Google AI Studio API key: https://aistudio.google.com
- Gemini model enabled on your project: https://ai.dev/rate-limit
- 4 GB free disk space (Docling model download on first run)

## Installation

```bash
git clone https://github.com/yourusername/cv-screening-pipeline.git
cd cv-screening-pipeline

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up environment file
cp .env.example .env
# Open .env and add your GEMINI_API_KEY

# Verify setup
python -c "import config; print('Setup successful')"
```

## Configuration

The only required environment variable:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Common settings in `config.py`:

```python
GEMINI_MODEL = "gemini-2.5-flash"
EXTRACTION_BATCH_SIZE = 5
DASHBOARD_PORT = 5000
```

## Usage

1. Add inputs:
- Put job description text in `data/input/job_description.txt`
- Put candidate PDFs in `data/input/cvs/`
- Replace `data/input/job_description.txt` with your JD (plain text)
- Place candidate PDFs in `data/input/cvs/`

### 2. Run

```bash
python main.py
```

3. View outputs:
- Dashboard: `http://127.0.0.1:5000`
- Ranked report: `data/output/report.md`
- Scorecards: `data/output/scored/`
- Decision receipts: `data/output/decision_receipts/`

Run dashboard only:

```bash
python dashboard/app.py
```

## Common Issues

**Model not found (404)** — Check [ai.dev/rate-limit](https://ai.dev/rate-limit) and update `GEMINI_MODEL` in `config.py` to a model with non-zero RPM.

**Quota exhausted (429)** — Free tier daily limit reached. Wait 24 hours.

**Frozen after Docling shows 100%** — Not frozen. Loading 560MB of ML models into RAM silently. Wait up to 15 minutes. Subsequent runs take under 30 seconds.

**ChromaDB telemetry noise** — Add to the very top of `main.py` before all imports:
```python
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
```

## Project Structure

```
cv_screening_pipeline/
|-- main.py                    Entry point
|-- config.py                  
|-- requirements.txt
|-- .env.example
|
|-- data/
|   |-- input/
|   |   |-- job_description.txt
|   |   |-- cvs/             
|   |-- processed/             
|   |-- output/                
|
|-- knowledge_base/
|-- pipeline/
|-- rag/
|-- prompts/
|-- dashboard/
```

---

For the full pipeline walkthrough, scoring rubric, RAG architecture, API call strategy, troubleshooting, and roadmap — see the **[documentation site](https://yourusername.github.io/cv-screening-pipeline)**.
