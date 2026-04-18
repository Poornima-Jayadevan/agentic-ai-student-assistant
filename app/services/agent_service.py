# app/services/agent_service.py

import re
from app.services.memory_service import (
    save_user_goal,
    update_user_profile,
    get_user_profile
)
from app.services.planner_service import generate_study_plan
from app.services.rag_service import (
    retrieve_relevant_chunks,
    search_all_documents,
    list_documents
)


def is_math_question(message: str) -> bool:
    math_symbols = ["+", "-", "*", "/", "%"]
    return any(sym in message for sym in math_symbols) or bool(
        re.search(r"\b(calculate|solve|what is)\b", message.lower())
    )


def extract_study_goal(message: str) -> dict | None:
    """
    Detect study goal updates.
    Examples:
    - I want to prepare for XAI interviews
    - My exam is in 10 days
    """
    msg = message.lower()

    if "i want to prepare for" in msg:
        goal = message.split("I want to prepare for", 1)[-1].strip()
        return {"study_goal": goal}

    match = re.search(r"exam is in (\d+) days", msg)
    if match:
        return {"exam_days": match.group(1)}

    match = re.search(r"interview is in (\d+) days", msg)
    if match:
        return {"exam_days": match.group(1)}

    return None


def is_study_plan_request(message: str) -> bool:
    keywords = [
        "study plan",
        "make me a plan",
        "create a plan",
        "revision plan",
        "preparation plan",
        "schedule for exam",
    ]
    msg = message.lower()
    return any(keyword in msg for keyword in keywords)


def is_pdf_question(message: str) -> bool:
    keywords = [
        "in the pdf",
        "from the pdf",
        "in the document",
        "from the document",
        "uploaded file",
        "uploaded pdf",
        "lecture notes",
        "summarize this document",
        "search this document",
    ]
    msg = message.lower()
    return any(keyword in msg for keyword in keywords)


def calculate_expression(message: str) -> str:
    """
    Very basic calculator logic.
    Only for simple expressions for now.
    """
    try:
        cleaned = message.lower().replace("what is", "").replace("calculate", "").strip()
        result = eval(cleaned, {"__builtins__": {}})
        return f"The answer is: **{result}**"
    except Exception:
        return "I could not calculate that. Please provide a simple expression like `25 * 4`."


def route_message(user_id: str, message: str) -> dict:
    """
    Decide what action the chatbot should take.
    Returns a dict like:
    {
        "route": "memory" / "calculator" / "rag" / "planner" / "llm",
        "response": "..."
    }
    """
    goal_update = extract_study_goal(message)
    if goal_update:
        update_user_profile(user_id, goal_update)
        return {
            "route": "memory",
            "response": f"I saved this to memory: {goal_update}"
        }

    if is_study_plan_request(message):
        profile = get_user_profile(user_id)
        plan = generate_study_plan(profile)
        return {
            "route": "planner",
            "response": plan
        }

    if is_math_question(message):
        result = calculate_expression(message)
        return {
            "route": "calculator",
            "response": result
        }

    if is_pdf_question(message):
        chunks = retrieve_relevant_chunks(message, top_k=3)
        if chunks:
            context = "\n\n".join(chunks)
            return {
                "route": "rag",
                "response": context
            }
        return {
            "route": "rag",
            "response": "I could not find relevant content in the uploaded documents."
        }

    return {
        "route": "llm",
        "response": None
    }