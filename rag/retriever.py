import chromadb
from rag.embedder import embed
from rag.indexer import get_chroma_client
from config import (
    CHROMA_COLLECTION_CVS,
    CHROMA_COLLECTION_INSTITUTIONS,
    CHROMA_COLLECTION_ORGANISATIONS,
    CHROMA_COLLECTION_CAPM_TERMS,
    RAG_TOP_K,
)


def get_cv_sections(candidate_id: str, sections: list[str] = None) -> str:
    """
    Retrieve specific sections of a candidate's CV from ChromaDB.

    Args:
        candidate_id: The anonymised candidate ID e.g. "C-001"
        sections: Optional list of section names to filter by
                  e.g. ["internship_0", "internship_1", "education"]
                  If None, retrieves all sections for this candidate.

    Returns:
        Formatted string of all retrieved sections for use in the scoring prompt.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_CVS)

    where_filter = {"candidate_id": candidate_id}

    try:
        results = collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        return f"No CV sections found for {candidate_id}: {e}"

    if not results["documents"]:
        return f"No CV sections found for {candidate_id}"

    # Filter by section if specified
    output_parts = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        if sections is None or meta.get("section") in sections:
            output_parts.append(f"[{meta.get('section', 'unknown')}]\n{doc}")

    return "\n\n".join(output_parts) if output_parts else f"No matching sections for {candidate_id}"


def get_institution_info(institution_name: str) -> str:
    """
    Look up an institution in the knowledge base by name.
    Returns the closest matching entry with tier and NLU status.

    Args:
        institution_name: Name as extracted from the CV

    Returns:
        Formatted string with institution details for scoring context.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_INSTITUTIONS)

    query_embedding = embed(institution_name)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(RAG_TOP_K, 3),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        return f"Institution lookup failed: {e}"

    if not results["documents"] or not results["documents"][0]:
        return f"Institution '{institution_name}' not found in knowledge base — treat as Tier 3."

    # Return the top match
    top_doc = results["documents"][0][0]
    top_meta = results["metadatas"][0][0]
    distance = results["distances"][0][0]

    # If distance is very high, the match is poor — flag it
    confidence = "HIGH" if distance < 0.5 else "MEDIUM" if distance < 1.0 else "LOW"

    return (
        f"Institution match (confidence: {confidence}):\n"
        f"{top_doc}\n"
        f"Tier: {top_meta.get('tier')} | Is NLU: {top_meta.get('is_nlu')}"
    )


def get_organisation_info(organisation_name: str) -> str:
    """
    Look up an organisation in the knowledge base.
    Used for scoring internship quality (criterion 2.2).

    Args:
        organisation_name: Name as extracted from the CV

    Returns:
        Formatted string with organisation tier and CapM relevance.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_ORGANISATIONS)

    query_embedding = embed(organisation_name)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(RAG_TOP_K, 3),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        return f"Organisation lookup failed: {e}"

    if not results["documents"] or not results["documents"][0]:
        return f"Organisation '{organisation_name}' not found — treat as Tier 3, low CapM relevance."

    top_doc = results["documents"][0][0]
    top_meta = results["metadatas"][0][0]
    distance = results["distances"][0][0]
    confidence = "HIGH" if distance < 0.5 else "MEDIUM" if distance < 1.0 else "LOW"

    return (
        f"Organisation match (confidence: {confidence}):\n"
        f"{top_doc}\n"
        f"Tier: {top_meta.get('tier')} | CapM Relevance: {top_meta.get('capm_relevance')}"
    )


def get_capm_term_context(terms: list[str]) -> str:
    """
    Retrieve definitions and usage context for a list of Capital Markets terms.
    Used for scoring technical knowledge (criterion 2.3).

    Args:
        terms: List of domain terms found in the candidate's CV

    Returns:
        Formatted string with term definitions and correct vs shallow usage examples.
    """
    if not terms:
        return "No domain terms provided."

    client = get_chroma_client()
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_CAPM_TERMS)

    all_results = []
    seen_terms = set()

    for term in terms[:10]:  # Cap at 10 terms to avoid massive context
        query_embedding = embed(term)
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=["documents", "distances"],
            )
            if results["documents"] and results["documents"][0]:
                doc = results["documents"][0][0]
                distance = results["distances"][0][0]
                if distance < 1.2 and doc not in seen_terms:
                    all_results.append(doc)
                    seen_terms.add(doc)
        except Exception:
            continue

    if not all_results:
        return "No matching Capital Markets terms found in knowledge base."

    return "Capital Markets Term Reference:\n\n" + "\n\n---\n\n".join(all_results)


def build_scoring_rag_context(candidate_id: str, extracted_profile: dict) -> str:
    """
    Build the complete RAG context string for a single candidate's scoring call.
    This is what gets injected into the {rag_context} placeholder in the scoring prompt.

    Retrieves:
    - All CV sections for this candidate
    - Institution info
    - Info for each internship organisation
    - CapM term context for all domain terms found across internships

    Args:
        candidate_id: The anonymised ID
        extracted_profile: The extracted JSON profile for this candidate

    Returns:
        Complete formatted RAG context string
    """
    sections = []

    # 1. All CV sections
    cv_text = get_cv_sections(candidate_id)
    sections.append(f"=== CANDIDATE CV SECTIONS ===\n{cv_text}")

    # 2. Institution info
    institution = extracted_profile.get("institution_name")
    if institution:
        inst_info = get_institution_info(institution)
        sections.append(f"=== INSTITUTION KNOWLEDGE BASE ===\n{inst_info}")

    # 3. Organisation info for each internship
    internships = extracted_profile.get("internships") or []
    if internships:
        org_infos = []
        for internship in internships:
            org_name = internship.get("organisation")
            if org_name:
                org_info = get_organisation_info(org_name)
                org_infos.append(f"Org: {org_name}\n{org_info}")
        if org_infos:
            sections.append(f"=== ORGANISATION KNOWLEDGE BASE ===\n" + "\n\n".join(org_infos))

    # 4. CapM terms context — collect all domain terms across all internships
    all_terms = []
    for internship in internships:
        all_terms.extend(internship.get("domain_terms_found") or [])
    all_terms = list(set(all_terms))  # deduplicate

    if all_terms:
        term_context = get_capm_term_context(all_terms)
        sections.append(f"=== CAPITAL MARKETS TERMINOLOGY REFERENCE ===\n{term_context}")

    return "\n\n" + "\n\n".join(sections)
