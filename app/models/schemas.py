from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    user_id: str
    message: str
    file_name: Optional[str] = None


class ChatResponse(BaseModel):
    response: str