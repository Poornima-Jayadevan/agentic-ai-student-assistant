# app/routes/chat_agent.py

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest
from app.services.agent_service import route_message
from app.services.llm_service import get_llm_response
from app.services.memory_service import get_memory, add_message
from app.services.tool_service import (
    calculator_tool,
    retriever_tool,
    looks_like_calculation,
    extract_math_expression,
)
from app.services.rag_answer_service import generate_rag_answer

router = APIRouter()


@router.post("/chat-agent")
def chat_agent(request: ChatRequest):
    try:
        user_id = request.user_id
        user_message = request.message.strip()
        file_name = request.file_name

        # Store user message
        add_message(user_id, "user", user_message)

        # Memory for fallback LLM prompt
        history = get_memory(user_id)

        # ---------------------------------
        # Step 1: Agent router decides first
        # ---------------------------------
        agent_result = route_message(user_id, user_message)
        route = agent_result.get("route")
        agent_response = agent_result.get("response")

        # ---------------------------------
        # Step 2: Direct-return agent routes
        # ---------------------------------
        if route in [
            "memory",
            "planner",
            "comparison",
            "interview_prep",
            "document_list",
            "document_search",
        ]:
            add_message(user_id, "assistant", agent_response)

            return {
                "response": agent_response,
                "mode": route,
                "source": "agent_router"
            }

        # ---------------------------------
        # Step 3: Calculator route
        # ---------------------------------
        if route == "calculator":
            add_message(user_id, "assistant", agent_response)

            return {
                "response": agent_response,
                "mode": "calculator",
                "source": "agent_router"
            }

        # ---------------------------------
        # Step 4: RAG route from agent router
        # ---------------------------------
        if route == "rag":
            if file_name:
                tool_result = retriever_tool(
                    query=user_message,
                    file_name=file_name,
                    top_k=3,
                )

                if tool_result.startswith("Error:"):
                    add_message(user_id, "assistant", tool_result)

                    return {
                        "response": tool_result,
                        "mode": "rag",
                        "source": "retriever_tool"
                    }

                if tool_result != "No relevant content found.":
                    response = generate_rag_answer(
                        user_question=user_message,
                        context=tool_result
                    )

                    add_message(user_id, "assistant", response)

                    return {
                        "response": response,
                        "mode": "rag",
                        "source": "retriever_tool"
                    }

                fallback_response = "I could not find relevant content in the uploaded document."

                add_message(user_id, "assistant", fallback_response)

                return {
                    "response": fallback_response,
                    "mode": "rag",
                    "source": "retriever_tool"
                }

            fallback_response = "Please upload or select a document so I can answer from it."

            add_message(user_id, "assistant", fallback_response)

            return {
                "response": fallback_response,
                "mode": "rag",
                "source": "agent_router"
            }

        # ---------------------------------
        # Step 5: Backward-compatible fallback calculator
        # ---------------------------------
        if looks_like_calculation(user_message):
            expression = extract_math_expression(user_message)
            tool_result = calculator_tool(expression)

            add_message(user_id, "assistant", tool_result)

            return {
                "response": tool_result,
                "mode": "calculator",
                "source": "calculator_tool"
            }

        # ---------------------------------
        # Step 6: Backward-compatible fallback retriever
        # ---------------------------------
        if file_name:
            tool_result = retriever_tool(
                query=user_message,
                file_name=file_name,
                top_k=3,
            )

            if tool_result.startswith("Error:"):
                add_message(user_id, "assistant", tool_result)

                return {
                    "response": tool_result,
                    "mode": "rag",
                    "source": "retriever_tool"
                }

            if tool_result != "No relevant content found.":
                response = generate_rag_answer(
                    user_question=user_message,
                    context=tool_result
                )

                add_message(user_id, "assistant", response)

                return {
                    "response": response,
                    "mode": "rag",
                    "source": "retriever_tool"
                }

        # ---------------------------------
        # Step 7: Default normal chat
        # ---------------------------------
        history_text = "\n".join(
            [f"{item['role']}: {item['content']}" for item in history[-10:]]
        )

        prompt = f"""
You are a helpful student assistant chatbot.

Conversation History:
{history_text}

User: {user_message}
Assistant:
""".strip()

        response = get_llm_response(prompt)

        add_message(user_id, "assistant", response)

        return {
            "response": response,
            "mode": "llm",
            "source": "llm"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))