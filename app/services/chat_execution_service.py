# app/services/chat_execution_service.py

from app.services.llm_service import (
    get_llm_response,
    summarize_with_llm,
    summarize_section_with_llm,
    explain_search_results,
)
from app.services.langchain_service import get_langchain_response
from app.services.memory_service import (
    get_memory,
    add_message,
    get_user_profile,
)
from app.services.rag_service import (
    retrieve_relevant_chunks,
    summarize_document,
    summarize_section,
    search_document,
    search_all_documents,
    list_documents,
)
from app.services.tool_router import detect_and_run_tool
from app.services.job_assistant_service import (
    summarize_job_description,
    compare_cv_with_job,
    identify_missing_skills,
    generate_cover_letter,
    generate_interview_questions,
)
from app.services.agent_service import route_message


def build_chat_response(
    mode: str,
    reply: str,
    source: str,
    metadata: dict | None = None,
    data: dict | None = None,
    success: bool = True,
) -> dict:
    """
    Standard API response shape for chat routes.
    """
    return {
        "success": success,
        "mode": mode,
        "reply": reply,
        "source": source,
        "metadata": metadata or {},
        "data": data or {},
    }


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


def resolve_file_name(file_name: str) -> str:
    """
    Return the provided file_name if available.
    Otherwise, fall back to the most recently available document.
    """
    if file_name:
        return file_name

    docs = list_documents()
    if docs:
        return docs[-1]

    return ""


def require_file_name(file_name: str) -> str:
    """
    Ensure a valid file_name is available for document-based operations.
    """
    resolved_file_name = resolve_file_name(file_name)

    if not resolved_file_name:
        raise ValueError("No document available. Please upload a PDF first.")

    return resolved_file_name


def execute_chat_flow(request, engine: str = "basic"):
    """
    Shared execution flow for:
    - /chat
    - /chat-langchain

    Routing is decided by route_message().
    Execution is handled here.
    """
    user_id = request.user_id
    user_message = request.message.strip()
    file_name = request.file_name.strip() if request.file_name else ""

    add_message(user_id, "user", user_message)

    agent_result = route_message(user_id, user_message)
    route = agent_result.get("route")
    agent_answer = agent_result.get("answer", "")
    agent_source = agent_result.get("source", "agent")
    agent_metadata = agent_result.get("metadata", {})

    resolved_file_name = resolve_file_name(file_name)

    # 1. Direct-return routes
    if route in [
        "memory",
        "planner",
        "calculator",
        "document_list",
        "document_search",
        "comparison",
        "interview_prep",
    ]:
        add_message(user_id, "assistant", agent_answer)

        return build_chat_response(
            mode=route,
            reply=agent_answer,
            source=agent_source,
            metadata=agent_metadata,
        )

    # 2. Job assistant routes
    if route == "job_assistant":
        task = agent_metadata.get("task")
        required_file_name = require_file_name(resolved_file_name)

        if task == "summarize_job_description":
            reply = summarize_job_description(file_name=required_file_name)
        elif task == "compare_cv_with_job":
            reply = compare_cv_with_job(file_name=required_file_name)
        elif task == "identify_missing_skills":
            reply = identify_missing_skills(file_name=required_file_name)
        elif task == "generate_cover_letter":
            reply = generate_cover_letter(file_name=required_file_name)
        elif task == "generate_interview_questions":
            reply = generate_interview_questions(file_name=required_file_name)
        else:
            raise ValueError("Unknown job assistant task.")

        add_message(user_id, "assistant", reply)

        return build_chat_response(
            mode="job_assistant",
            reply=reply,
            source=agent_source,
            metadata=agent_metadata,
            data={
                "task": task,
                "file_name": required_file_name,
            },
        )

    # 3. Full document summarization
    if route == "summarize_document":
        required_file_name = require_file_name(resolved_file_name)

        doc_text = summarize_document(required_file_name)

        if not doc_text:
            raise ValueError(f"Could not summarize document '{required_file_name}'.")

        summary = summarize_with_llm(doc_text)
        add_message(user_id, "assistant", summary)

        return build_chat_response(
            mode="summarize_document",
            reply=summary,
            source=agent_source,
            metadata=agent_metadata,
            data={"file_name": required_file_name},
        )

    # 4. Section summarization
    if route == "summarize_section":
        required_file_name = require_file_name(resolved_file_name)
        section_name = agent_metadata.get("section", "")
        result = summarize_section(required_file_name, section_name)

        if not result:
            doc_text = summarize_document(required_file_name)

            if not doc_text:
                raise ValueError(f"Could not summarize document '{required_file_name}'.")

            summary = summarize_with_llm(doc_text)
            final_reply = (
                f"I couldn't find a matching section for '{section_name}', "
                f"so here is a summary of the full document instead.\n\n{summary}"
            )

            add_message(user_id, "assistant", final_reply)

            return build_chat_response(
                mode="summarize_document",
                reply=final_reply,
                source=agent_source,
                metadata=agent_metadata,
                data={"file_name": required_file_name},
            )

        summary = summarize_section_with_llm(
            section_title=result["title"],
            text=result["content"],
        )

        final_reply = f"**Summary of {result['title']}:**\n\n{summary}"
        add_message(user_id, "assistant", final_reply)

        return build_chat_response(
            mode="summarize_section",
            reply=final_reply,
            source=agent_source,
            metadata=agent_metadata,
            data={
                "file_name": required_file_name,
                "section": result["title"],
            },
        )

    # 5. Keyword search
    if route == "keyword_search":
        keyword = agent_metadata.get("keyword", "")

        if resolved_file_name:
            matches = search_document(resolved_file_name, keyword, max_results=5)

            if matches is None:
                raise ValueError(f"Document '{resolved_file_name}' was not found.")

            reply = explain_search_results(keyword, matches)
            add_message(user_id, "assistant", reply)

            return build_chat_response(
                mode="keyword_search",
                reply=reply,
                source=agent_source,
                metadata=agent_metadata,
                data={
                    "file_name": resolved_file_name,
                    "keyword": keyword,
                    "matches": matches,
                },
            )

        matches = search_all_documents(keyword, max_results=5)
        reply = explain_search_results(keyword, matches)
        add_message(user_id, "assistant", reply)

        return build_chat_response(
            mode="keyword_search_all_documents",
            reply=reply,
            source=agent_source,
            metadata=agent_metadata,
            data={
                "keyword": keyword,
                "matches": matches,
            },
        )

    # 6. Tool fallback
    tool_result = detect_and_run_tool(user_message)

    if tool_result:
        reply = format_tool_response(tool_result)
        add_message(user_id, "assistant", reply)

        return build_chat_response(
            mode="tool",
            reply=reply,
            source="tool",
            data={"tool_output": tool_result},
        )

    # 7. Normal RAG / LLM QA
    messages = get_memory(user_id)

    retrieved_chunks = []
    context = ""

    if resolved_file_name:
        retrieved_chunks = retrieve_relevant_chunks(
            query=user_message,
            file_name=resolved_file_name,
            top_k=3,
        )

        context = "\n\n".join(
            [chunk["text"] for chunk in retrieved_chunks if "text" in chunk]
        )

    user_profile = get_user_profile(user_id)

    if engine == "langchain":
        enhanced_user_message = user_message

        if context:
            enhanced_user_message = (
                f"Use the following document context to answer the user.\n\n"
                f"Context:\n{context}\n\n"
                f"User question:\n{user_message}"
            )

        if user_profile:
            enhanced_user_message += f"\n\nUser profile:\n{user_profile}"

        reply = get_langchain_response(
            user_message=enhanced_user_message,
            memory=messages,
        )

        mode = "langchain_rag_qa" if resolved_file_name else "langchain_chat"
    else:
        reply = get_llm_response(
            messages,
            context=context,
            user_profile=user_profile,
        )

        mode = "rag_qa" if resolved_file_name else "chat"

    add_message(user_id, "assistant", reply)

    return build_chat_response(
        mode=mode,
        reply=reply,
        source=agent_source if route == "rag" else "llm",
        metadata=agent_metadata,
        data={
            "file_name": resolved_file_name if resolved_file_name else None,
            "retrieved_chunks": retrieved_chunks,
            "memory": user_profile,
        },
    )