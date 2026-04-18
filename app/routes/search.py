from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.rag_service import retrieve_relevant_chunks

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    file_name: str
    top_k: int = 3


@router.post("/search")
def search_document(request: SearchRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        if not request.file_name.strip():
            raise HTTPException(status_code=400, detail="file_name cannot be empty.")

        results = retrieve_relevant_chunks(
            query=request.query,
            file_name=request.file_name,
            top_k=request.top_k
        )

        if not results:
            return {
                "query": request.query,
                "file_name": request.file_name,
                "top_k": request.top_k,
                "results": [],
                "message": "No relevant results found."
            }

        return {
            "query": request.query,
            "file_name": request.file_name,
            "top_k": request.top_k,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))