from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

_model = None


def get_model() -> SentenceTransformer:
    """
    Load the embedding model once and cache it.
    All other modules import from here — the model is never loaded twice.
    """
    global _model
    if _model is None:
        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(text: str) -> list[float]:
    """
    Embed a single string. Returns a list of floats.
    """
    model = get_model()
    return model.encode(text, convert_to_numpy=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings. More efficient than calling embed() in a loop.
    """
    model = get_model()
    return model.encode(texts, convert_to_numpy=True).tolist()
