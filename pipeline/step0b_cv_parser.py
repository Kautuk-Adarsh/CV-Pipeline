import json
import re
import shutil
from pathlib import Path
from config import (
    CVS_DIR,
    MARKDOWN_DIR,
    CANDIDATE_ID_MAPPING_PATH,
    SUPPORTED_CV_EXTENSIONS,
)


def run() -> dict:
    """
    Step 0B: Parse all CV files using Docling and produce anonymised Markdown.

    Reads:  data/input/cvs/* (PDF, DOCX, or TXT files)
    Writes: data/processed/markdown/candidate_XXX.md (one per CV)
            data/processed/candidate_id_mapping.json (ID to original filename)

    Returns the candidate ID mapping dict.

    Note: Anonymisation of PII happens in Step 1 (LLM-based).
    This step only converts format and assigns IDs.
    The mapping file is stored separately and is never seen by the scoring model.
    """
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

    cv_files = _collect_cv_files()
    if not cv_files:
        raise FileNotFoundError(f"No CV files found in {CVS_DIR}. Supported: {SUPPORTED_CV_EXTENSIONS}")

    print(f"  Found {len(cv_files)} CV file(s) to process")

    # Assign anonymised IDs — original names never reach the scoring model
    id_mapping = {}
    for i, cv_path in enumerate(sorted(cv_files), start=1):
        candidate_id = f"C-{i:03d}"
        id_mapping[candidate_id] = cv_path.name

    # Save mapping file
    CANDIDATE_ID_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_ID_MAPPING_PATH.write_text(json.dumps(id_mapping, indent=2), encoding="utf-8")

    # Parse each CV
    success_count = 0
    for candidate_id, filename in id_mapping.items():
        cv_path = CVS_DIR / filename
        print(f"  Parsing {filename} → {candidate_id}...")

        try:
            markdown_text = _parse_cv(cv_path)
            output_path = MARKDOWN_DIR / f"{candidate_id}.md"
            output_path.write_text(markdown_text, encoding="utf-8")
            success_count += 1
            print(f"    Saved {output_path.name} ({len(markdown_text)} chars)")
        except Exception as e:
            print(f"    Error parsing {filename}: {e}")
            # Write an empty file so downstream steps can detect and handle it
            output_path = MARKDOWN_DIR / f"{candidate_id}.md"
            output_path.write_text(f"# PARSE ERROR\n\nFailed to parse {filename}: {e}", encoding="utf-8")

    print(f"  Parsed {success_count}/{len(id_mapping)} CVs successfully")
    return id_mapping


def _collect_cv_files() -> list[Path]:
    """
    Find all supported CV files in the input directory.
    """
    if not CVS_DIR.exists():
        raise FileNotFoundError(f"CVs directory not found: {CVS_DIR}")

    files = []
    for ext in SUPPORTED_CV_EXTENSIONS:
        files.extend(CVS_DIR.glob(f"*{ext}"))
        files.extend(CVS_DIR.glob(f"*{ext.upper()}"))

    # Remove duplicates and hidden files
    return [f for f in set(files) if not f.name.startswith(".")]


def _parse_cv(cv_path: Path) -> str:
    """
    Parse a CV file to Markdown.

    Tries Docling first (handles multi-column PDF layouts).
    Falls back to plain text extraction for .txt files or if Docling fails.
    """
    ext = cv_path.suffix.lower()

    if ext in [".pdf", ".docx"]:
        return _parse_with_docling(cv_path)
    elif ext == ".txt":
        return _parse_txt(cv_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _parse_with_docling(cv_path: Path) -> str:
    """
    Use Docling for layout-aware PDF/DOCX parsing.
    Docling understands multi-column layouts and section headers,
    producing clean Markdown that LLMs read more reliably than raw text.
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(cv_path))
        markdown = result.document.export_to_markdown()

        if not markdown or len(markdown.strip()) < 100:
            raise ValueError("Docling produced empty or very short output — trying fallback")

        return markdown

    except ImportError:
        print("    Docling not installed — falling back to text extraction")
        return _parse_txt_fallback(cv_path)
    except Exception as e:
        print(f"    Docling failed ({e}) — trying Marker fallback")
        return _parse_with_marker_or_fallback(cv_path)


def _parse_with_marker_or_fallback(cv_path: Path) -> str:
    """
    Fallback 1: Try Marker.
    Fallback 2: Basic text extraction using pypdf2 (if available).
    Fallback 3: Read as plain text.
    """
    try:
        # Try marker
        import subprocess
        result = subprocess.run(
            ["marker_single", str(cv_path), "--output_dir", str(cv_path.parent)],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            md_path = cv_path.parent / (cv_path.stem + ".md")
            if md_path.exists():
                return md_path.read_text(encoding="utf-8")
    except Exception:
        pass

    # Last resort — try PyPDF2
    try:
        import PyPDF2
        with open(cv_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = [page.extract_text() for page in reader.pages]
            text = "\n\n".join(p for p in pages if p)
            return f"# CV (plain text extraction — layout may be imperfect)\n\n{text}"
    except Exception:
        pass

    return f"# PARSE ERROR\n\nCould not parse {cv_path.name} with any available method."


def _parse_txt(cv_path: Path) -> str:
    """
    For .txt CV files — read directly and add basic Markdown structure.
    """
    raw = cv_path.read_text(encoding="utf-8", errors="replace")
    return f"# CV (plain text)\n\n{raw}"


def _parse_txt_fallback(cv_path: Path) -> str:
    """
    Last resort text extraction for PDF when Docling is unavailable.
    """
    return _parse_txt(cv_path)


if __name__ == "__main__":
    mapping = run()
    print(mapping)
