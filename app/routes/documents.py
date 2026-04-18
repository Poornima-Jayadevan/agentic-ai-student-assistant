from fastapi import APIRouter
from app.services.rag_service import list_documents_with_types

router = APIRouter()

@router.get("/documents")
def get_documents():
    return {"documents": list_documents_with_types()}