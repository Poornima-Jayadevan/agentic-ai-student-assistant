from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest
from app.services.chat_execution_service import execute_chat_flow

router = APIRouter()


@router.post("/chat")
def chat(request: ChatRequest):
    """
    Basic chatbot route using shared execution flow.
    """
    try:
        return execute_chat_flow(request, engine="basic")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat-langchain")
def chat_langchain(request: ChatRequest):
    """
    LangChain chatbot route using shared execution flow.
    """
    try:
        return execute_chat_flow(request, engine="langchain")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))