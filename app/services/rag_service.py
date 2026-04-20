import os
import re
from typing import Dict, List, Optional, Any

from pypdf import PdfReader

from app.utils.text_chunker import chunk_text
from app.services.embedding_service import get_embeddings, get_query_embedding
from app.services.vector_store_service import save_faiss_index, search_faiss


document_store: Dict[str, Dict[str, Any]] = {}


def normalize_file_name(file_name: str) -> str:
    if not file_name:
        return ""

    return os.path.splitext(file_name.strip())[0]


def normalize_document_type(document_type: str) -> str:
    if not document_type:
        return "other"

    normalized = document_type.strip().lower()

    allowed_types = {"cv", "job_description", "cover_letter", "other"}
    if normalized not in allowed_types:
        return "other"

    return normalized


def clean_text(text: str) -> str:
    return " ".join(text.split())


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text.append(page_text)

    raw_text = "\n".join(full_text)
    return clean_text(raw_text)


def process_pdf(
    pdf_path: str,
    file_name: Optional[str] = None,
    document_type: str = "other"
) -> dict:
    if file_name is None:
        file_name = os.path.basename(pdf_path)

    normalized_name = normalize_file_name(file_name)
    normalized_type = normalize_document_type(document_type)

    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        raise ValueError("No readable text found in the PDF.")

    chunks = chunk_text(text, chunk_size=800, overlap=100)

    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")

    embeddings = get_embeddings(chunks)

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Embeddings could not be generated.")

    vector_info = save_faiss_index(
        embeddings=embeddings,
        chunks=chunks,
        file_name=normalized_name
    )

    document_store[normalized_name] = {
        "file_name": normalized_name,
        "original_file_name": file_name,
        "document_type": normalized_type,
        "pdf_path": pdf_path,
        "text": text,
        "chunks": chunks,
        "total_chunks": len(chunks),
    }

    return {
        "message": f"PDF processed successfully for file: {normalized_name}",
        "file_name": normalized_name,
        "original_file_name": file_name,
        "document_type": normalized_type,
        "total_chunks": len(chunks),
        "vector_store": vector_info,
    }


def _query_looks_like_literal_keyword(query: str) -> bool:
    """
    Decide whether the query behaves like a keyword-style search.
    Examples:
    - 'python'
    - 'introduction'
    - 'asdhjklqwerty'
    Not examples:
    - 'what does the document say about python?'
    - 'summarize this document'
    """
    query = query.strip()
    if not query:
        return False

    if len(query.split()) <= 3:
        return True

    return False


def _chunk_contains_query_term(chunks: List[str], query: str) -> bool:
    """
    Check whether the literal query text appears in any chunk.
    Useful to avoid returning semantic matches for nonsense keyword queries.
    """
    q = query.strip().lower()
    if not q:
        return False

    return any(q in chunk.lower() for chunk in chunks)


def retrieve_relevant_chunks(query: str, file_name: str, top_k: int = 3) -> List[Dict]:
    """
    Retrieve relevant chunks from the FAISS vector store.

    Improvements:
    - validates document existence
    - prevents nonsense keyword queries from returning random semantic matches
    - filters out weak semantic matches using a score threshold
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    normalized_name = normalize_file_name(file_name)
    doc = document_store.get(normalized_name)

    if not doc:
        print("Available documents:", list(document_store.keys()))
        raise ValueError(f"Document '{normalized_name}' not found.")

    chunks = doc.get("chunks", [])

    # For short keyword-like queries, require literal presence somewhere in the document.
    # This fixes cases like 'asdhjklqwerty' returning unrelated CV chunks.
    if _query_looks_like_literal_keyword(query):
        if not _chunk_contains_query_term(chunks, query):
            return []

    query_embedding = get_query_embedding(query)
    if query_embedding is None or len(query_embedding) == 0:
        raise ValueError("Query embedding could not be generated.")

    results = search_faiss(
        query_embedding=query_embedding,
        file_name=normalized_name,
        top_k=top_k
    )

    if not results:
        return []

    filtered_results = []

    for item in results:
        score = item.get("score", 0)

        # Keep only reasonably relevant results.
        # Adjust this threshold later if needed based on your FAISS score behavior.
        if score >= 0.30:
            filtered_results.append(item)

    return filtered_results


def list_documents() -> List[str]:
    return list(document_store.keys())


def list_documents_with_types() -> List[Dict[str, str]]:
    return [
        {
            "file_name": file_name,
            "original_file_name": doc.get("original_file_name", file_name),
            "document_type": doc.get("document_type", "other")
        }
        for file_name, doc in document_store.items()
    ]


def list_documents_by_type(document_type: str) -> List[str]:
    normalized_type = normalize_document_type(document_type)

    return [
        file_name
        for file_name, doc in document_store.items()
        if doc.get("document_type") == normalized_type
    ]


def get_latest_document_by_type(document_type: str) -> Optional[str]:
    docs = list_documents_by_type(document_type)
    return docs[-1] if docs else None


def get_document(file_name: str) -> Optional[Dict]:
    normalized_name = normalize_file_name(file_name)
    return document_store.get(normalized_name)


def summarize_document(file_name: str, max_chars: int = 4000) -> Optional[str]:
    doc = get_document(file_name)
    if not doc:
        return None

    text = doc.get("text", "").strip()
    if not text:
        return None

    return text[:max_chars]


def split_into_sections(text: str) -> List[Dict[str, str]]:
    if not text.strip():
        return []

    pattern = r"(?im)(chapter\s+\d+\b.*?|^\d+[\.\)]?\s+[A-Z][^\n]*$)"
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))

    if not matches:
        return [{"title": "Full Document", "content": text}]

    sections = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        title = match.group().strip()
        content = text[start:end].strip()

        sections.append({
            "title": title,
            "content": content
        })

    return sections


def summarize_section(
    file_name: str,
    section_query: str,
    max_chars: int = 3000
) -> Optional[Dict[str, str]]:
    doc = get_document(file_name)
    if not doc:
        return None

    text = doc.get("text", "")
    if not text.strip():
        return None

    sections = split_into_sections(text)
    query = section_query.lower().strip()

    for section in sections:
        if query in section["title"].lower():
            return {
                "title": section["title"],
                "content": section["content"][:max_chars]
            }

    for section in sections:
        if query in section["content"].lower():
            return {
                "title": section["title"],
                "content": section["content"][:max_chars]
            }

    return None


def search_document(file_name: str, keyword: str, max_results: int = 5) -> Optional[List[Dict]]:
    doc = get_document(file_name)
    if not doc:
        return None

    keyword = keyword.lower().strip()
    if not keyword:
        raise ValueError("Keyword cannot be empty.")

    results = []
    chunks = doc.get("chunks", [])

    for idx, chunk in enumerate(chunks):
        if keyword in chunk.lower():
            results.append({
                "file_name": doc["file_name"],
                "chunk_id": idx,
                "text": chunk
            })

        if len(results) >= max_results:
            break

    return results


def search_all_documents(keyword: str, max_results: int = 5) -> List[Dict]:
    keyword = keyword.lower().strip()
    if not keyword:
        raise ValueError("Keyword cannot be empty.")

    all_results = []

    for file_name, doc in document_store.items():
        chunks = doc.get("chunks", [])

        for idx, chunk in enumerate(chunks):
            if keyword in chunk.lower():
                all_results.append({
                    "file_name": file_name,
                    "chunk_id": idx,
                    "text": chunk
                })

            if len(all_results) >= max_results:
                return all_results

    return all_results


def semantic_search_document(query: str, file_name: str, top_k: int = 5) -> List[Dict]:
    return retrieve_relevant_chunks(query=query, file_name=file_name, top_k=top_k)


def semantic_search_by_type(query: str, document_type: str, top_k: int = 5) -> List[Dict]:
    normalized_type = normalize_document_type(document_type)
    matching_docs = list_documents_by_type(normalized_type)

    all_results = []

    for file_name in matching_docs:
        try:
            results = retrieve_relevant_chunks(query=query, file_name=file_name, top_k=top_k)
            for item in results:
                item_copy = item.copy()
                item_copy["file_name"] = file_name
                item_copy["document_type"] = normalized_type
                all_results.append(item_copy)
        except Exception:
            continue

    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_results[:top_k]


def retrieve_chunks_for_documents(
    query: str,
    file_names: List[str],
    top_k_per_doc: int = 3
) -> List[Dict]:
    all_results = []

    for file_name in file_names:
        try:
            results = retrieve_relevant_chunks(
                query=query,
                file_name=file_name,
                top_k=top_k_per_doc
            )
            for item in results:
                item_copy = item.copy()
                item_copy["file_name"] = file_name
                all_results.append(item_copy)
        except Exception:
            continue

    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_results


def build_multi_doc_context(results: List[Dict], max_chars: int = 5000) -> str:
    if not results:
        return ""

    parts = []
    total_len = 0

    for item in results:
        text = item.get("text", "").strip()
        file_name = item.get("file_name", "unknown")

        block = f"[Document: {file_name}]\n{text}\n"
        if total_len + len(block) > max_chars:
            break

        parts.append(block)
        total_len += len(block)

    return "\n\n".join(parts)