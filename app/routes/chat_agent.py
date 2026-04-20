# app/routes/chat_agent.py

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest
from app.services.llm_service import get_llm_response
from app.services.memory_service import get_memory, add_message
from app.services.tool_service import (
    calculator_tool,
    retriever_tool,
    looks_like_calculation,
    extract_math_expression,
)

router = APIRouter()


@router.post("/chat-agent")
def chat_agent(request: ChatRequest):
    try:
        user_id = request.user_id
        user_message = request.message.strip()
        file_name = request.file_name

        # Store user message
        add_message(user_id, "user", user_message)

        # Keep this if you still want memory for later,
        # but don't pass it into get_llm_response for now.
        history = get_memory(user_id)

        # -----------------------------
        # Tool 1: Calculator
        # -----------------------------
        if looks_like_calculation(user_message):
            expression = extract_math_expression(user_message)
            tool_result = calculator_tool(expression)

            add_message(user_id, "assistant", tool_result)

            return {
                "response": tool_result,
                "source": "calculator_tool"
            }

        # -----------------------------
        # Tool 2: Retriever
        # -----------------------------
        if file_name:
            tool_result = retriever_tool(
                query=user_message,
                file_name=file_name,
                top_k=3,
            )

            if tool_result.startswith("Error:"):
                return {
                    "response": tool_result,
                    "source": "retriever_tool"
                }

            if tool_result != "No relevant content found.":
                prompt = f"""
You are a helpful student assistant chatbot.

Answer the user's question using only the document context below.
If the answer is not clearly available in the context, say that honestly.

Document Context:
{tool_result}

User Question:
{user_message}
""".strip()

                response = get_llm_response(prompt)

                add_message(user_id, "assistant", response)

                return {
                    "response": response,
                    "source": "retriever_tool"
                }

        # -----------------------------
        # Default: Normal chat
        # -----------------------------
        response = get_llm_response(user_message)

        add_message(user_id, "assistant", response)

        return {
            "response": response,
            "source": "llm"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))