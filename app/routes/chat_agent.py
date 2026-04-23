# app/routes/chat_agent.py

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest
from app.services.chat_execution_service import execute_chat_flow

router = APIRouter()


@router.post("/chat-agent")
def chat_agent(request: ChatRequest):
    """
    Agent chatbot route using shared execution flow.
    """
    try:
        return execute_chat_flow(request, engine="basic")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))