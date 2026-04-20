import json
import chromadb
from pathlib import Path
from rag.embedder import embed, embed_batch
from config import (
    VECTOR_STORE_DIR,
    CHROMA_COLLECTION_CVS,
    CHROMA_COLLECTION_INSTITUTIONS,
    CHROMA_COLLECTION_ORGANISATIONS,
    CHROMA_COLLECTION_CAPM_TERMS,
    INSTITUTIONS_KB_PATH,
    ORGANISATIONS_KB_PATH,
    CAPM_TERMS_KB_PATH,
)

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Returns a persistent ChromaDB client.
    Data is saved to disk at VECTOR_STORE_DIR automatically.
    """
    return chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))


def index_candidate_cv(candidate_id: str, extracted_profile: dict) -> None:
    """
    Index a candidate's CV sections into ChromaDB.
    Each meaningful section is stored as a separate chunk
    so the retriever can fetch only what is needed per criterion.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_CVS)

    chunks = []
    metadatas = []
    ids = []

    # Build text chunks from the extracted profile
    # Each chunk represents one scoreable section of the CV

    # Education chunk
    edu_parts = []
    if extracted_profile.get("institution_name"):
        edu_parts.append(f"Institution: {extracted_profile['institution_name']}")
    if extracted_profile.get("degree_type"):
        edu_parts.append(f"Degree: {extracted_profile['degree_type']}")
    if extracted_profile.get("cgpa") is not None:
        edu_parts.append(f"CGPA: {extracted_profile['cgpa']} / {extracted_profile.get('cgpa_scale', 10)}")
    if extracted_profile.get("class_rank"):
        edu_parts.append(f"Rank: {extracted_profile['class_rank']}")
    if extracted_profile.get("academic_awards"):
        edu_parts.append(f"Awards: {', '.join(extracted_profile['academic_awards'])}")
    if edu_parts:
        chunks.append("\n".join(edu_parts))
        metadatas.append({"candidate_id": candidate_id, "section": "education"})
        ids.append(f"{candidate_id}_education")

    # Internship chunks — one per internship
    internships = extracted_profile.get("internships") or []
    for i, internship in enumerate(internships):
        parts = [
            f"Organisation: {internship.get('organisation', 'Unknown')}",
            f"Team: {internship.get('team_or_practice', 'Not specified')}",
            f"Duration: {internship.get('duration_months', 'Not specified')} months",
            f"Work: {internship.get('role_description', '')}",
            f"Domain terms: {', '.join(internship.get('domain_terms_found', []))}",
        ]
        chunks.append("\n".join(parts))
        metadatas.append({"candidate_id": candidate_id, "section": f"internship_{i}"})
        ids.append(f"{candidate_id}_internship_{i}")

    # Publications chunk
    publications = extracted_profile.get("publications") or []
    if publications:
        chunks.append("Publications:\n" + "\n".join(publications))
        metadatas.append({"candidate_id": candidate_id, "section": "publications"})
        ids.append(f"{candidate_id}_publications")

    # Moot courts chunk
    moots = extracted_profile.get("moot_courts") or []
    if moots:
        moot_texts = []
        for m in moots:
            moot_texts.append(
                f"{m.get('name', '')} — {m.get('result', '')} — Topic: {m.get('topic', 'Not specified')}"
            )
        chunks.append("Moot Courts:\n" + "\n".join(moot_texts))
        metadatas.append({"candidate_id": candidate_id, "section": "moot_courts"})
        ids.append(f"{candidate_id}_moot_courts")

    # Languages + international chunk
    extras = []
    languages = extracted_profile.get("languages") or []
    if languages:
        extras.append(f"Languages: {', '.join(languages)}")
    international = extracted_profile.get("international_exposure") or []
    if international:
        extras.append(f"International: {', '.join(international)}")
    leadership = extracted_profile.get("leadership_roles") or []
    if leadership:
        extras.append(f"Leadership: {', '.join(leadership)}")
    if extras:
        chunks.append("\n".join(extras))
        metadatas.append({"candidate_id": candidate_id, "section": "extras"})
        ids.append(f"{candidate_id}_extras")

    if not chunks:
        print(f"  Warning: No indexable content found for {candidate_id}")
        return

    # Generate embeddings and upsert into ChromaDB
    embeddings = embed_batch(chunks)
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def index_knowledge_bases() -> None:
    """
    Index all three knowledge base JSON files into ChromaDB.
    Called once per pipeline run — safe to re-run (upsert is idempotent).
    """
    client = get_chroma_client()

    _index_institutions(client)
    _index_organisations(client)
    _index_capm_terms(client)

    print("  Knowledge bases indexed successfully")


def _index_institutions(client: chromadb.PersistentClient) -> None:
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_INSTITUTIONS)
    institutions = json.loads(INSTITUTIONS_KB_PATH.read_text())

    docs, embeddings, metadatas, ids = [], [], [], []
    for inst in institutions:
        text = (
            f"Institution: {inst['name']}\n"
            f"Abbreviation: {inst.get('abbreviation', '')}\n"
            f"City: {inst.get('city', '')}, {inst.get('state', '')}\n"
            f"Tier: {inst['tier']}\n"
            f"Is NLU: {inst['is_nlu']}\n"
            f"Notes: {inst.get('notes', '')}"
        )
        docs.append(text)
        embeddings.append(embed(text))
        metadatas.append({"tier": inst["tier"], "is_nlu": str(inst["is_nlu"])})
        ids.append(f"inst_{inst['abbreviation'].replace(' ', '_')}")

    collection.upsert(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)


def _index_organisations(client: chromadb.PersistentClient) -> None:
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_ORGANISATIONS)
    orgs = json.loads(ORGANISATIONS_KB_PATH.read_text())

    docs, embeddings, metadatas, ids = [], [], [], []
    for i, org in enumerate(orgs):
        text = (
            f"Organisation: {org['name']}\n"
            f"Type: {org['type']}\n"
            f"Tier: {org['tier']}\n"
            f"Capital Markets Relevance: {org['capm_relevance']}\n"
            f"Notes: {org.get('notes', '')}"
        )
        docs.append(text)
        embeddings.append(embed(text))
        metadatas.append({"tier": org["tier"], "capm_relevance": org["capm_relevance"]})
        ids.append(f"org_{i}_{org['name'][:20].replace(' ', '_')}")

    collection.upsert(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)


def _index_capm_terms(client: chromadb.PersistentClient) -> None:
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_CAPM_TERMS)
    terms = json.loads(CAPM_TERMS_KB_PATH.read_text())

    docs, embeddings, metadatas, ids = [], [], [], []
    for term in terms:
        text = (
            f"Term: {term['term']}\n"
            f"Full name: {term.get('full_name', '')}\n"
            f"Domain: {term.get('domain', '')}\n"
            f"Correct usage example: {term.get('correct_usage_example', '')}\n"
            f"Shallow usage example: {term.get('shallow_usage_example', '')}\n"
            f"Notes: {term.get('notes', '')}"
        )
        docs.append(text)
        embeddings.append(embed(text))
        metadatas.append({"domain": term.get("domain", ""), "category": term.get("category", "")})
        ids.append(f"term_{term['term'].replace(' ', '_')[:40]}")

    collection.upsert(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)

