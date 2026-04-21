# app/services/agent_service.py

import re
from typing import Optional

from app.services.memory_service import (
    update_user_profile,
    get_user_profile
)
from app.services.planner_service import generate_study_plan
from app.services.rag_service import (
    retrieve_relevant_chunks,
    search_all_documents,
    list_documents
)
from app.services.comparison_service import compare_cv_with_job_description
from app.services.interview_prep_service import generate_interview_prep


# -------------------------------
# ROUTE DETECTION HELPERS
# -------------------------------

def is_math_question(message: str) -> bool:
    """
    Detect simple calculator-style questions.
    Avoids catching normal text that contains "/" or "-".
    """
    msg = message.lower().strip()

    # direct math expression like: 25*4, 8 + 9, 100 / 5
    if re.fullmatch(r"\s*\d+(\.\d+)?\s*[\+\-\*\/%]\s*\d+(\.\d+)?\s*", msg):
        return True

    # text-based calculator requests
    calculator_patterns = [
        r"\bcalculate\b",
        r"\bsolve\b",
        r"\bwhat is\s+\d+(\.\d+)?\s*[\+\-\*\/%]\s*\d+(\.\d+)?\b",
    ]

    return any(re.search(pattern, msg) for pattern in calculator_patterns)


def extract_study_goal(message: str) -> Optional[dict]:
    """
    Detect study goal or timeline updates.

    Examples:
    - I want to prepare for XAI interviews
    - My exam is in 10 days
    - My interview is in 5 days
    """
    msg = message.lower().strip()

    # Study goal
    match = re.search(r"i want to prepare for (.+)", msg)
    if match:
        return {"study_goal": match.group(1).strip()}

    match = re.search(r"i am preparing for (.+)", msg)
    if match:
        return {"study_goal": match.group(1).strip()}

    # Exam timeline
    match = re.search(r"exam is in (\d+) days", msg)
    if match:
        return {"exam_days": int(match.group(1))}

    # Interview timeline
    match = re.search(r"interview is in (\d+) days", msg)
    if match:
        return {"interview_days": int(match.group(1))}

    return None


def is_study_plan_request(message: str) -> bool:
    keywords = [
        "study plan",
        "make me a plan",
        "create a plan",
        "revision plan",
        "preparation plan",
        "schedule for exam",
        "make me a schedule",
        "create a study schedule",
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
        "summarise this document",
        "search this document",
        "search in document",
        "search in pdf",
        "find in document",
        "find in pdf",
        "this document",
        "this pdf",
        "uploaded document",
    ]
    msg = message.lower()
    return any(keyword in msg for keyword in keywords)


def is_document_list_request(message: str) -> bool:
    keywords = [
        "list documents",
        "show documents",
        "show uploaded documents",
        "what documents do i have",
        "which documents are uploaded",
        "list uploaded files",
        "show my files",
    ]
    msg = message.lower()
    return any(keyword in msg for keyword in keywords)


def is_document_search_request(message: str) -> bool:
    keywords = [
        "search all documents",
        "search documents",
        "find in all documents",
        "search across documents",
    ]
    msg = message.lower()
    return any(keyword in msg for keyword in keywords)


def is_comparison_request(message: str) -> bool:
    msg = message.lower()
    return (
        "compare" in msg and
        (
            "cv" in msg or
            "resume" in msg or
            "job description" in msg or
            "jd" in msg or
            "document" in msg
        )
    )


def is_interview_prep_request(message: str) -> bool:
    keywords = [
        "interview prep",
        "interview questions",
        "quiz me",
        "ask me questions",
        "mock interview",
        "prepare me for interview",
    ]
    msg = message.lower()
    return any(keyword in msg for keyword in keywords)


# -------------------------------
# ACTION HELPERS
# -------------------------------

def calculate_expression(message: str) -> str:
    """
    Safe basic calculator for simple arithmetic expressions only.
    """
    try:
        cleaned = message.lower().strip()
        cleaned = cleaned.replace("what is", "")
        cleaned = cleaned.replace("calculate", "")
        cleaned = cleaned.replace("solve", "")
        cleaned = cleaned.strip()

        # keep only numbers, operators, spaces, and decimal points
        if not re.fullmatch(r"[\d\.\+\-\*\/%\(\)\s]+", cleaned):
            return "I could not calculate that. Please provide a simple expression like `25 * 4`."

        result = eval(cleaned, {"__builtins__": {}})
        return f"The answer is: **{result}**"

    except Exception:
        return "I could not calculate that. Please provide a simple expression like `25 * 4`."


def handle_rag_request(message: str) -> str:
    """
    Retrieve relevant chunks from uploaded documents.
    """
    try:
        chunks = retrieve_relevant_chunks(message, top_k=3)

        if not chunks:
            return "I could not find relevant content in the uploaded documents."

        context = "\n\n".join(chunks)
        return context

    except Exception:
        return "There was an issue retrieving content from the uploaded documents."


def handle_document_search(message: str) -> str:
    """
    Search across all uploaded documents.
    """
    try:
        results = search_all_documents(message)

        if not results:
            return "I could not find matching content across your uploaded documents."

        if isinstance(results, list):
            formatted = "\n\n".join(str(item) for item in results[:5])
            return f"Here are the most relevant results I found:\n\n{formatted}"

        return str(results)

    except Exception:
        return "There was an issue searching across the uploaded documents."


def handle_list_documents() -> str:
    """
    List uploaded documents.
    """
    try:
        docs = list_documents()

        if not docs:
            return "No documents are currently uploaded."

        if isinstance(docs, list):
            formatted = "\n".join(f"- {doc}" for doc in docs)
            return f"Uploaded documents:\n{formatted}"

        return f"Uploaded documents:\n{docs}"

    except Exception:
        return "There was an issue retrieving the list of uploaded documents."


# -------------------------------
# MAIN ROUTER
# -------------------------------

def route_message(user_id: str, message: str) -> dict:
    """
    Decide what action the chatbot should take.

    Returns:
    {
        "route": "memory" / "planner" / "calculator" / "rag" /
                 "document_search" / "document_list" / "comparison" /
                 "interview_prep" / "llm",
        "response": "..."
    }
    """

    # 1. Memory updates
    goal_update = extract_study_goal(message)
    if goal_update:
        update_user_profile(user_id, goal_update)
        return {
            "route": "memory",
            "response": f"I saved this to memory: {goal_update}"
        }

    # 2. Study planner
    if is_study_plan_request(message):
        profile = get_user_profile(user_id)
        plan = generate_study_plan(profile)
        return {
            "route": "planner",
            "response": plan
        }

    # 3. Calculator
    if is_math_question(message):
        result = calculate_expression(message)
        return {
            "route": "calculator",
            "response": result
        }

    # 4. List documents
    if is_document_list_request(message):
        return {
            "route": "document_list",
            "response": handle_list_documents()
        }

    # 5. Search across all documents
    if is_document_search_request(message):
        return {
            "route": "document_search",
            "response": handle_document_search(message)
        }

    # 6. Comparison request
    if is_comparison_request(message):
        comparison_result = compare_cv_with_job_description()
        return {
            "route": "comparison",
            "response": comparison_result
        }

    # 7. Interview prep request
    if is_interview_prep_request(message):
        interview_prep_result = generate_interview_prep()
        return {
                "route": "interview_prep",
                "response": interview_prep_result
            }
        

    # 8. PDF / RAG request
    if is_pdf_question(message):
        return {
            "route": "rag",
            "response": None
        }

    # 9. Default fallback
    return {
        "route": "llm",
        "response": None
    }