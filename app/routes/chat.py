from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest
from app.services.llm_service import (
    get_llm_response,
    summarize_with_llm,
    summarize_section_with_llm,
    explain_search_results
)
from app.services.memory_service import (
    get_memory,
    add_message,
    update_user_profile,
    get_user_profile
)
from app.services.rag_service import (
    retrieve_relevant_chunks,
    summarize_document,
    summarize_section,
    search_document,
    search_all_documents,
    list_documents
)
from app.services.tool_router import detect_and_run_tool
from app.services.job_assistant_service import (
    summarize_job_description,
    compare_cv_with_job,
    identify_missing_skills,
    generate_cover_letter,
    generate_interview_questions
)

router = APIRouter()


def detect_document_command(message: str) -> dict | None:
    """
    Detect whether the user is asking for:
    - full document summary
    - section/chapter summary
    - keyword search
    - normal QA
    """
    msg = message.strip().lower()

    # Full document summary
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

    if msg in full_doc_phrases:
        return {"type": "summarize_document"}

    # More flexible fallback for full-document summary
    if msg.startswith("summarize ") or msg.startswith("summarise "):
        if any(term in msg for term in [" cv", "resume", "document", "pdf"]):
            return {"type": "summarize_document"}

    # Section / chapter summary
    if msg.startswith("summarize chapter "):
        return {
            "type": "summarize_section",
            "section": message[len("summarize chapter "):].strip()
        }

    if msg.startswith("summarise chapter "):
        return {
            "type": "summarize_section",
            "section": message[len("summarise chapter "):].strip()
        }

    if msg.startswith("summarize section "):
        return {
            "type": "summarize_section",
            "section": message[len("summarize section "):].strip()
        }

    if msg.startswith("summarise section "):
        return {
            "type": "summarize_section",
            "section": message[len("summarise section "):].strip()
        }

    # Generic summarize fallback:
    # only treat it as section summary if it is not obviously asking
    # for a full document summary.
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
            "pdf"
        ]:
            return {
                "type": "summarize_section",
                "section": section_name
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
            "pdf"
        ]:
            return {
                "type": "summarize_section",
                "section": section_name
            }

    if msg.startswith("find mentions of "):
        return {
            "type": "keyword_search",
            "keyword": message[len("find mentions of "):].strip()
        }

    if msg.startswith("search for "):
        return {
            "type": "keyword_search",
            "keyword": message[len("search for "):].strip()
        }

    if msg.startswith("search "):
        return {
            "type": "keyword_search",
            "keyword": message[len("search "):].strip()
        }

    return None


def detect_memory_update(message: str) -> dict | None:
    """
    Detect simple long-term memory updates like:
    - I want to prepare for XAI interviews
    - My exam is in 10 days
    - My interview is in 7 days
    """
    msg = message.strip().lower()

    if "i want to prepare for " in msg:
        marker = "i want to prepare for "
        start_index = msg.find(marker)
        goal = message[start_index + len(marker):].strip()

        if goal:
            return {
                "study_goal": goal
            }

    if "my exam is in " in msg and " days" in msg:
        try:
            days_part = msg.split("my exam is in ", 1)[1].split(" days", 1)[0].strip()
            days = int(days_part)
            return {
                "exam_days": str(days)
            }
        except ValueError:
            pass

    if "my interview is in " in msg and " days" in msg:
        try:
            days_part = msg.split("my interview is in ", 1)[1].split(" days", 1)[0].strip()
            days = int(days_part)
            return {
                "exam_days": str(days)
            }
        except ValueError:
            pass

    return None


def detect_job_command(message: str) -> str | None:
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


def format_tool_response(tool_result: dict) -> str:
    """
    Convert tool output into a readable chatbot reply.
    """
    tool_name = tool_result.get("tool")

    if "error" in tool_result:
        return f"Tool error: {tool_result['error']}"

    if tool_name == "calculator":
        expression = tool_result.get("input", "")
        result = tool_result.get("result", "")
        return f"**Calculation Result**\n\n`{expression}` = **{result}**"

    if tool_name == "study_plan":
        subject = tool_result.get("subject", "Study")
        days = tool_result.get("days", 7)
        plan = tool_result.get("plan", [])

        lines = [f"**{days}-Day Study Plan for {subject}**\n"]

        for item in plan:
            day = item.get("day", "")
            task = item.get("task", "")
            topic = item.get("topic", "")
            lines.append(f"**Day {day}:** {task} ({topic})")

        return "\n".join(lines)

    return str(tool_result)


@router.post("/chat")
def chat(request: ChatRequest):
    try:
        user_id = request.user_id
        user_message = request.message.strip()
        file_name = request.file_name.strip() if request.file_name else ""

        # Store user message in short-term memory
        add_message(user_id, "user", user_message)

        # -----------------------------------
        # 0. LONG-TERM MEMORY UPDATE DETECTION
        # -----------------------------------
        memory_update = detect_memory_update(user_message)
        if memory_update:
            update_user_profile(user_id, memory_update)

            saved_items = ", ".join(
                [f"{key}: {value}" for key, value in memory_update.items()]
            )
            reply = f"I saved this to memory: {saved_items}"

            add_message(user_id, "assistant", reply)

            return {
                "mode": "memory_update",
                "memory": get_user_profile(user_id),
                "reply": reply
            }

        # -----------------------------
        # 1. TOOL DETECTION
        # -----------------------------
        tool_result = detect_and_run_tool(user_message)

        if tool_result:
            reply = format_tool_response(tool_result)
            add_message(user_id, "assistant", reply)

            return {
                "mode": "tool",
                "tool_output": tool_result,
                "reply": reply
            }

        # -----------------------------
        # 2. JOB ASSISTANT DETECTION
        # -----------------------------
        job_command = detect_job_command(user_message)

        if job_command == "summarize_job_description":
            reply = summarize_job_description()
            add_message(user_id, "assistant", reply)

            return {
                "mode": "job_assistant",
                "task": job_command,
                "reply": reply
            }

        if job_command == "compare_cv_with_job":
            reply = compare_cv_with_job()
            add_message(user_id, "assistant", reply)

            return {
                "mode": "job_assistant",
                "task": job_command,
                "reply": reply
            }

        if job_command == "identify_missing_skills":
            reply = identify_missing_skills()
            add_message(user_id, "assistant", reply)

            return {
                "mode": "job_assistant",
                "task": job_command,
                "reply": reply
            }

        if job_command == "generate_cover_letter":
            reply = generate_cover_letter()
            add_message(user_id, "assistant", reply)

            return {
                "mode": "job_assistant",
                "task": job_command,
                "reply": reply
            }

        if job_command == "generate_interview_questions":
            reply = generate_interview_questions()
            add_message(user_id, "assistant", reply)

            return {
                "mode": "job_assistant",
                "task": job_command,
                "reply": reply
            }

        # -----------------------------
        # 3. DOCUMENT COMMAND DETECTION
        # -----------------------------
        command = detect_document_command(user_message)

        # If file_name is not provided, try latest available processed doc
        if not file_name:
            docs = list_documents()
            if docs:
                file_name = docs[-1]

        # -----------------------------
        # 4. FULL DOCUMENT SUMMARIZATION
        # -----------------------------
        if command and command["type"] == "summarize_document":
            if not file_name:
                raise ValueError("No document available. Please upload a PDF first.")

            doc_text = summarize_document(file_name)

            if not doc_text:
                raise ValueError(f"Could not summarize document '{file_name}'.")

            summary = summarize_with_llm(doc_text)

            add_message(user_id, "assistant", summary)

            return {
                "mode": "summarize_document",
                "file_name": file_name,
                "reply": summary
            }

        # -----------------------------
        # 5. SECTION / CHAPTER SUMMARY
        # -----------------------------
        if command and command["type"] == "summarize_section":
            if not file_name:
                raise ValueError("No document available. Please upload a PDF first.")

            result = summarize_section(file_name, command["section"])

            if not result:
                doc_text = summarize_document(file_name)

                if not doc_text:
                    raise ValueError(f"Could not summarize document '{file_name}'.")

                summary = summarize_with_llm(doc_text)
                final_reply = (
                    f"I couldn't find a matching section for '{command['section']}', "
                    f"so here is a summary of the full document instead.\n\n{summary}"
                )

                add_message(user_id, "assistant", final_reply)

                return {
                    "mode": "summarize_document",
                    "file_name": file_name,
                    "reply": final_reply
                }

            summary = summarize_section_with_llm(
                section_title=result["title"],
                text=result["content"]
            )

            final_reply = f"**Summary of {result['title']}:**\n\n{summary}"

            add_message(user_id, "assistant", final_reply)

            return {
                "mode": "summarize_section",
                "file_name": file_name,
                "section": result["title"],
                "reply": final_reply
            }

        # -----------------------------
        # 6. KEYWORD / DOCUMENT SEARCH
        # -----------------------------
        if command and command["type"] == "keyword_search":
            keyword = command["keyword"]

            if file_name:
                matches = search_document(file_name, keyword, max_results=5)

                if matches is None:
                    raise ValueError(f"Document '{file_name}' was not found.")

                reply = explain_search_results(keyword, matches)

                add_message(user_id, "assistant", reply)

                return {
                    "mode": "keyword_search",
                    "file_name": file_name,
                    "keyword": keyword,
                    "reply": reply,
                    "matches": matches
                }

            matches = search_all_documents(keyword, max_results=5)
            reply = explain_search_results(keyword, matches)

            add_message(user_id, "assistant", reply)

            return {
                "mode": "keyword_search_all_documents",
                "keyword": keyword,
                "reply": reply,
                "matches": matches
            }

        # -----------------------------
        # 7. NORMAL RAG / LLM QA
        # -----------------------------
        messages = get_memory(user_id)

        retrieved_chunks = []
        context = ""

        if file_name:
            retrieved_chunks = retrieve_relevant_chunks(
                query=user_message,
                file_name=file_name,
                top_k=3
            )

            context = "\n\n".join(
                [chunk["text"] for chunk in retrieved_chunks if "text" in chunk]
            )

        reply = get_llm_response(
            messages,
            context=context,
            user_profile=get_user_profile(user_id)
        )

        add_message(user_id, "assistant", reply)

        return {
            "mode": "rag_qa" if file_name else "chat",
            "file_name": file_name if file_name else None,
            "reply": reply,
            "retrieved_chunks": retrieved_chunks,
            "memory": get_user_profile(user_id)
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))