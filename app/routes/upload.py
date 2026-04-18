from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import os
import shutil

from app.services.rag_service import process_pdf
from app.services.document_store import save_chunks

router = APIRouter()

UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_unique_file_path(folder: str, filename: str) -> tuple[str, str]:
    """
    Create a unique file path if a file with the same name already exists.

    Returns:
        tuple[str, str]:
            - full file path
            - file_name without extension
    """
    base_name, ext = os.path.splitext(filename)
    candidate_filename = filename
    candidate_path = os.path.join(folder, candidate_filename)
    counter = 1

    while os.path.exists(candidate_path):
        candidate_filename = f"{base_name}_{counter}{ext}"
        candidate_path = os.path.join(folder, candidate_filename)
        counter += 1

    file_name_without_ext = os.path.splitext(candidate_filename)[0]
    return candidate_path, file_name_without_ext


@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    document_type: str = Form("other")
):
    """
    Upload a PDF, save it locally, process it for RAG,
    generate embeddings, and store the chunks in FAISS.

    Supported document types:
    - cv
    - job_description
    - cover_letter
    - other
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path, file_name = get_unique_file_path(UPLOAD_FOLDER, file.filename)
    saved_filename = os.path.basename(file_path)

    try:
        # Save uploaded PDF locally
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF for RAG + summarization/search support
        result = process_pdf(
            pdf_path=file_path,
            file_name=file_name,
            document_type=document_type
        )

        # Optional: keep chunk storage if used elsewhere in your project
        save_chunks(saved_filename, result.get("chunks", []))

        return {
            "message": "PDF uploaded and indexed successfully.",
            "filename": saved_filename,
            "file_name": result["file_name"],
            "document_type": result["document_type"],
            "path": file_path,
            "total_chunks": result["total_chunks"],
            "vector_store": result["vector_store"],
            "chunks": result.get("chunks", []),
        }

    except ValueError as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )