import os
import json
import faiss
import numpy as np

VECTOR_DIR = "data/vectorstore"

# Ensure vectorstore path is a directory
if os.path.exists(VECTOR_DIR):
    if not os.path.isdir(VECTOR_DIR):
        raise RuntimeError(
            f"'{VECTOR_DIR}' exists but is not a directory. "
            f"Please delete or rename that file and create a folder instead."
        )
else:
    os.makedirs(VECTOR_DIR, exist_ok=True)


def save_faiss_index(embeddings, chunks, file_name="index"):
    """
    Save normalized embeddings into a FAISS index and store corresponding chunks.

    Uses inner product search. With normalized embeddings, this approximates
    cosine similarity, so higher score = more similar.
    """
    embeddings = np.array(embeddings, dtype="float32")

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError("Number of chunks must match number of embeddings.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path = os.path.join(VECTOR_DIR, f"{file_name}.faiss")
    metadata_path = os.path.join(VECTOR_DIR, f"{file_name}_chunks.json")

    faiss.write_index(index, index_path)

    metadata = [
        {
            "chunk_id": i,
            "text": chunk
        }
        for i, chunk in enumerate(chunks)
    ]

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "index_path": index_path,
        "metadata_path": metadata_path,
        "total_chunks": len(chunks),
        "dimension": dimension,
        "metric": "cosine_similarity_via_inner_product"
    }


def load_faiss_index(file_name="index"):
    """
    Load a saved FAISS index and its chunk metadata.
    """
    index_path = os.path.join(VECTOR_DIR, f"{file_name}.faiss")
    metadata_path = os.path.join(VECTOR_DIR, f"{file_name}_chunks.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Chunk metadata not found: {metadata_path}")

    index = faiss.read_index(index_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


def search_faiss(query_embedding, file_name="index", top_k=3):
    """
    Search the FAISS index for the most relevant chunks.

    Returns:
        list of dicts with:
        - chunk_id
        - text
        - score
    """
    index, chunks = load_faiss_index(file_name)

    query_embedding = np.array(query_embedding, dtype="float32")

    if query_embedding.ndim != 2 or query_embedding.shape[0] == 0:
        raise ValueError("Query embedding must be a non-empty 2D array.")

    top_k = max(1, min(top_k, len(chunks)))

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if 0 <= idx < len(chunks):
            chunk_item = chunks[idx]

            # Support both old metadata format (plain strings)
            # and new metadata format (dicts with chunk_id/text)
            if isinstance(chunk_item, dict):
                text = chunk_item.get("text", "")
                chunk_id = chunk_item.get("chunk_id", idx)
            else:
                text = str(chunk_item)
                chunk_id = idx

            results.append({
                "chunk_id": int(chunk_id),
                "text": text,
                "score": float(scores[0][rank])
            })

    return results