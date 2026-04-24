# app/services/agent_service.py

import re
from typing import Optional

from app.services.memory_service import (
    update_user_profile,
    get_user_profile
)
from app.services.planner_service import generate_study_plan
from app.services.rag_service import (
    search_all_documents,
    list_documents
)
from app.services.comparison_service import compare_cv_with_job_description
from app.services.interview_prep_service import generate_interview_prep


# -------------------------------
# STANDARD RESULT BUILDER
# -------------------------------

def build_route_result(
    route: str,
    answer: str = "",
    source: str = "agent",
    metadata: Optional[dict] = None,
    success: bool = True,
) -> dict:
    """
    Build a consistent router response object.

    Standard format:
    {
        "success": True,
        "route": "<route_name>",
        "answer": "<final user-facing text or empty string>",
        "source": "<agent|tool|rag|llm|memory>",
        "metadata": {}
    }
    """
    return {
        "success": success,
        "route": route,
        "answer": answer,
        "source": source,
        "metadata": metadata or {},
    }


# -------------------------------
# ROUTE DETECTION HELPERS
# -------------------------------

def is_math_question(message: str) -> bool:
    """
    Detect simple calculator-style questions.
    Avoids catching normal text that contains '/' or '-'.
    """
    msg = message.lower().strip()

    if re.fullmatch(r"\s*\d+(\.\d+)?\s*[\+\-\*\/%]\s*\d+(\.\d+)?\s*", msg):
        return True

    calculator_patterns = [
        r"\bcalculate\b",
        r"\bsolve\b",
        r"\bwhat is\s+\d+(\.\d+)?\s*[\+\-\*\/%]\s*\d+(\.\d+)?\b",
    ]

    return any(re.search(pattern, msg) for pattern in calculator_patterns)


def extract_study_goal(message: str) -> Optional[dict]:
    """
    Detect study goal or timeline updates.
    """
    msg = message.lower().strip()

    match = re.search(r"i want to prepare for (.+)", msg)
    if match:
        return {"study_goal": match.group(1).strip()}

    match = re.search(r"i am preparing for (.+)", msg)
    if match:
        return {"study_goal": match.group(1).strip()}

    match = re.search(r"exam is in (\d+) days", msg)
    if match:
        return {"exam_days": int(match.group(1))}

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


def detect_job_command(message: str) -> Optional[str]:
    """
    Detect job assistant tasks.
    """
    msg = message.strip().lower()

    if "summarize this job description" in msg or "summarise this job description" in msg:
        return "summarize_job_description"

    if "tailor my cv" in msg or "compare my cv with this job" in msg:
        return "compare_cv_with_job"

    if "what skills am i missing" in msg or "missing skills" in msg:
        return "identify_missing_skills"

    if "generate a cover letter" in msg or "write a cover letter" in msg:
        return "generate_cover_letter"

    if "interview questions" in msg or "prepare interview questions" in msg:
        return "generate_interview_questions"

    return None


def detect_document_command(message: str) -> Optional[dict]:
    """
    Detect whether the user is asking for:
    - full document summary
    - section/chapter summary
    - keyword search
    """
    msg = message.strip().lower()

    full_doc_phrases = {
        "summarize this document",
        "summarise this document",
        "summarize document",
        "summarise document",
        "summarize my cv",
        "summarise my cv",
        "summarize this cv",
        "summarise this cv",
        "summarize cv",
        "summarise cv",
        "summarize my resume",
        "summarise my resume",
        "summarize this resume",
        "summarise this resume",
        "summarize resume",
        "summarise resume",
        "summarize uploaded document",
        "summarise uploaded document",
        "summarize this pdf",
        "summarise this pdf",
        "summarize pdf",
        "summarise pdf",
    }

    # 1. Exact full-document summary requests
    if msg in full_doc_phrases:
        return {"type": "summarize_document"}

    # 2. Section / chapter summary requests
    if msg.startswith("summarize the section "):
        return {
            "type": "summarize_section",
            "section": message[len("summarize the section "):].strip(),
        }

    if msg.startswith("summarise the section "):
        return {
            "type": "summarize_section",
            "section": message[len("summarise the section "):].strip(),
        }

    if msg.startswith("summarize the chapter "):
        return {
            "type": "summarize_section",
            "section": message[len("summarize the chapter "):].strip(),
        }

    if msg.startswith("summarise the chapter "):
        return {
            "type": "summarize_section",
            "section": message[len("summarise the chapter "):].strip(),
        }

    if msg.startswith("summarize chapter "):
        return {
            "type": "summarize_section",
            "section": message[len("summarize chapter "):].strip(),
        }

    if msg.startswith("summarise chapter "):
        return {
            "type": "summarize_section",
            "section": message[len("summarise chapter "):].strip(),
        }

    if msg.startswith("summarize section "):
        return {
            "type": "summarize_section",
            "section": message[len("summarize section "):].strip(),
        }

    if msg.startswith("summarise section "):
        return {
            "type": "summarize_section",
            "section": message[len("summarise section "):].strip(),
        }

    if msg.startswith("summarize ") and " section" in msg:
        cleaned = (
            message[len("summarize "):]
            .replace(" section of the document", "")
            .replace(" section in the document", "")
            .replace(" section of document", "")
            .replace(" section in document", "")
            .replace(" section", "")
            .strip()
        )
        return {
            "type": "summarize_section",
            "section": cleaned,
        }

    if msg.startswith("summarise ") and " section" in msg:
        cleaned = (
            message[len("summarise "):]
            .replace(" section of the document", "")
            .replace(" section in the document", "")
            .replace(" section of document", "")
            .replace(" section in document", "")
            .replace(" section", "")
            .strip()
        )
        return {
            "type": "summarize_section",
            "section": cleaned,
        }

    # 3. Flexible full-document summary fallback
    if msg.startswith("summarize ") or msg.startswith("summarise "):
        if any(term in msg for term in [" cv", "resume", "document", "pdf"]):
            return {"type": "summarize_document"}

    # 4. Generic summarize fallback -> treat as section request
    if msg.startswith("summarize "):
        section_name = message[len("summarize "):].strip()
        if section_name.lower() not in [
            "this document",
            "document",
            "my cv",
            "this cv",
            "cv",
            "my resume",
            "this resume",
            "resume",
            "uploaded document",
            "this pdf",
            "pdf",
        ]:
            return {
                "type": "summarize_section",
                "section": section_name,
            }

    if msg.startswith("summarise "):
        section_name = message[len("summarise "):].strip()
        if section_name.lower() not in [
            "this document",
            "document",
            "my cv",
            "this cv",
            "cv",
            "my resume",
            "this resume",
            "resume",
            "uploaded document",
            "this pdf",
            "pdf",
        ]:
            return {
                "type": "summarize_section",
                "section": section_name,
            }

    # 5. Keyword search
    if msg.startswith("find mentions of "):
        return {
            "type": "keyword_search",
            "keyword": message[len("find mentions of "):].strip(),
        }

    if msg.startswith("search for "):
        return {
            "type": "keyword_search",
            "keyword": message[len("search for "):].strip(),
        }

    if msg.startswith("search "):
        return {
            "type": "keyword_search",
            "keyword": message[len("search "):].strip(),
        }

    return None


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

        if not re.fullmatch(r"[\d\.\+\-\*\/%\(\)\s]+", cleaned):
            return "I could not calculate that. Please provide a simple expression like `25 * 4`."

        result = eval(cleaned, {"__builtins__": {}})
        return f"The answer is: **{result}**"

    except Exception:
        return "I could not calculate that. Please provide a simple expression like `25 * 4`."


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
    Single source of routing order.

    Order:
    1. memory
    2. planner
    3. calculator
    4. document list
    5. document search
    6. comparison
    7. interview prep
    8. job assistant
    9. document commands
    10. broad rag
    11. llm fallback
    """

    # 1. Memory updates
    goal_update = extract_study_goal(message)
    if goal_update:
        update_user_profile(user_id, goal_update)
        return build_route_result(
            route="memory",
            answer=f"I saved this to memory: {goal_update}",
            source="memory",
            metadata={
                "updated_fields": goal_update
            }
        )

    # 2. Study planner
    if is_study_plan_request(message):
        profile = get_user_profile(user_id)
        plan = generate_study_plan(profile)
        return build_route_result(
            route="planner",
            answer=plan,
            source="planner",
            metadata={
                "profile_used": profile
            }
        )

    # 3. Calculator
    if is_math_question(message):
        result = calculate_expression(message)
        return build_route_result(
            route="calculator",
            answer=result,
            source="tool",
            metadata={
                "original_message": message
            }
        )

    # 4. List documents
    if is_document_list_request(message):
        return build_route_result(
            route="document_list",
            answer=handle_list_documents(),
            source="agent",
            metadata={}
        )

    # 5. Search across all documents
    if is_document_search_request(message):
        return build_route_result(
            route="document_search",
            answer=handle_document_search(message),
            source="agent",
            metadata={
                "query": message
            }
        )

    # 6. Comparison request
    if is_comparison_request(message):
        comparison_result = compare_cv_with_job_description()
        return build_route_result(
            route="comparison",
            answer=comparison_result,
            source="agent",
            metadata={}
        )

    # 7. Interview prep request
    if is_interview_prep_request(message):
        interview_prep_result = generate_interview_prep()
        return build_route_result(
            route="interview_prep",
            answer=interview_prep_result,
            source="agent",
            metadata={}
        )

    # 8. Job assistant
    job_command = detect_job_command(message)
    if job_command:
        return build_route_result(
            route="job_assistant",
            answer="",
            source="agent",
            metadata={
                "task": job_command
            }
        )

    # 9. Document commands
    document_command = detect_document_command(message)
    if document_command:
        return build_route_result(
            route=document_command["type"],
            answer="",
            source="agent",
            metadata=document_command
        )

    # 10. PDF / RAG request
    if is_pdf_question(message):
        return build_route_result(
            route="rag",
            answer="",
            source="rag",
            metadata={
                "query": message
            }
        )

    # 11. Default fallback
    return build_route_result(
        route="llm",
        answer="",
        source="llm",
        metadata={
            "query": message
        }
    )