# app/services/retriever_service.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import faiss
from sentence_transformers import SentenceTransformer


VECTORSTORE_DIR = Path("data/vectorstore")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model = SentenceTransformer(EMBEDDING_MODEL)


def normalize_file_name(file_name: str) -> str:
    """
    Keep the provided file name as-is after trimming whitespace.
    It must match the stored FAISS/chunks file base name.
    """
    return file_name.strip()


def get_index_and_chunks(file_name: str):
    """
    Load the FAISS index and matching chunk metadata JSON for one document.
    """
    base_name = normalize_file_name(file_name)

    faiss_path = VECTORSTORE_DIR / f"{base_name}.faiss"
    chunks_path = VECTORSTORE_DIR / f"{base_name}_chunks.json"

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS file not found: {faiss_path}")

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks JSON not found: {chunks_path}")

    index = faiss.read_index(str(faiss_path))

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks


def _extract_chunk_text(chunk: Any) -> str:
    """
    Convert a chunk into plain text.
    Supports:
    - plain strings
    - dict chunks like {"chunk_id": 1, "text": "..."}
    """
    if isinstance(chunk, str):
        return chunk

    if isinstance(chunk, dict):
        if "text" in chunk:
            return str(chunk["text"])
        if "chunk" in chunk:
            return str(chunk["chunk"])
        if "content" in chunk:
            return str(chunk["content"])
        return str(chunk)

    return str(chunk)


def retrieve_chunks(query: str, file_name: str, top_k: int = 3) -> List[str]:
    """
    Retrieve top-k similar chunks from a saved FAISS index for one file.
    Returns only plain text chunks.
    """
    index, chunks = get_index_and_chunks(file_name)

    query_embedding = _model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results: List[str] = []

    for idx in indices[0]:
        if idx == -1:
            continue
        if 0 <= idx < len(chunks):
            results.append(_extract_chunk_text(chunks[idx]))

    return results


def build_context(query: str, file_name: str, top_k: int = 3) -> str:
    """
    Join retrieved chunks into one context string.
    """
    chunks = retrieve_chunks(query=query, file_name=file_name, top_k=top_k)
    return "\n\n".join(chunks)