from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings row-wise for cosine similarity search.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Convert a list of texts into normalized embeddings.

    Returns:
        np.ndarray of shape (n_texts, embedding_dim)
    """
    if not texts:
        return np.array([], dtype=np.float32)

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=False
    ).astype("float32")

    return normalize_embeddings(embeddings).astype("float32")


def get_query_embedding(text: str) -> np.ndarray:
    """
    Convert a single query into a normalized embedding.

    Returns:
        np.ndarray of shape (1, embedding_dim)
    """
    if not text or not text.strip():
        return np.array([], dtype=np.float32)

    embedding = embedding_model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=False
    ).astype("float32")

    return normalize_embeddings(embedding).astype("float32")