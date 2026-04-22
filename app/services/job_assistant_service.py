from app.services.rag_service import (
    get_latest_document_by_type,
    summarize_document,
    retrieve_chunks_for_documents,
    build_multi_doc_context
)
from app.services.llm_service import call_ollama


def summarize_job_description() -> str:
    """
    Summarize the latest uploaded job description.
    """
    file_name = get_latest_document_by_type("job_description")
    if not file_name:
        return "No job description document was found."

    doc_text = summarize_document(file_name, max_chars=4000)
    if not doc_text:
        return "Could not read the job description."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful career assistant. "
                "Summarize the job description clearly. "
                "Use this structure:\n"
                "Job Title\n"
                "Responsibilities\n"
                "Skills\n"
                "Overall Fit\n"
                "Keep the summary concise, practical, and well-structured."
            )
        },
        {
            "role": "user",
            "content": f"Summarize this job description:\n\n{doc_text}"
        }
    ]

    return call_ollama(messages)


def compare_cv_with_job() -> str:
    """
    Compare the latest CV with the latest job description using
    multi-document semantic retrieval.
    """
    cv_file = get_latest_document_by_type("cv")
    jd_file = get_latest_document_by_type("job_description")

    if not cv_file:
        return "No CV document was found."
    if not jd_file:
        return "No job description document was found."

    query = (
        "Compare candidate CV with the job description. "
        "Identify strengths, missing skills, and practical improvements."
    )

    results = retrieve_chunks_for_documents(
        query=query,
        file_names=[cv_file, jd_file],
        top_k_per_doc=4
    )

    context = build_multi_doc_context(results)

    if not context.strip():
        return "I could not retrieve enough relevant information from the CV and job description."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a career assistant. "
                "Compare the candidate with the role using only the provided context. "
                "Do not assume the candidate has experience unless it appears in the CV context. "
                "Do not copy years of experience from the job description into the candidate profile. "
                "Use this exact structure:\n"
                "✅ Your Strengths\n"
                "❌ Missing Skills\n"
                "🔧 How to Fix\n"
                "Keep the response practical, concise, and recruiter-friendly."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                "Compare the candidate with the job and identify gaps."
            )
        }
    ]

    return call_ollama(messages)


def identify_missing_skills() -> str:
    """
    Identify the most important skill gaps between the latest CV
    and the latest job description using multi-document semantic retrieval.
    """
    cv_file = get_latest_document_by_type("cv")
    jd_file = get_latest_document_by_type("job_description")

    if not cv_file:
        return "No CV document was found."
    if not jd_file:
        return "No job description document was found."

    query = (
        "What skills are missing when comparing the candidate CV "
        "with the job description?"
    )

    results = retrieve_chunks_for_documents(
        query=query,
        file_names=[cv_file, jd_file],
        top_k_per_doc=4
    )

    context = build_multi_doc_context(results)

    if not context.strip():
        return "I could not retrieve enough relevant information from the CV and job description."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a career assistant. "
                "Use only the provided CV and job description context. "
                "Do not invent missing skills. "
                "Use this exact structure:\n"
                "✅ Your Strengths\n"
                "❌ Missing Skills\n"
                "🔧 How to Fix\n"
                "Mention only the most important 3 to 5 gaps."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                "What skills is the candidate missing for this role?"
            )
        }
    ]

    return call_ollama(messages)


def generate_cover_letter(file_name: str | None = None):
    """
    Generate a tailored and honest cover letter using the latest CV
    and the latest job description with multi-document RAG.
    """
    cv_file = get_latest_document_by_type("cv")
    jd_file = get_latest_document_by_type("job_description")

    if not cv_file:
        return "No CV document was found."
    if not jd_file:
        return "No job description document was found."

    query = (
        "Write a tailored cover letter using the candidate CV and the job description."
    )

    results = retrieve_chunks_for_documents(
        query=query,
        file_names=[cv_file, jd_file],
        top_k_per_doc=4
    )

    context = build_multi_doc_context(results, max_chars=6000)

    if not context.strip():
        return "I could not retrieve enough relevant information from the CV and job description."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional job application assistant. "
                "Write a tailored cover letter using only the provided context. "
                "Do not invent experience. "
                "Do not claim the candidate has requirements they do not have. "
                "Do not copy years of experience from the job description. "
                "If there is a gap, frame it honestly and positively. "
                "Keep the tone professional, natural, and specific."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                "Write a professional cover letter."
            )
        }
    ]

    return call_ollama(messages)


def generate_interview_questions() -> str:
    """
    Generate likely interview questions using the job description and,
    if available, the candidate CV.
    """
    cv_file = get_latest_document_by_type("cv")
    jd_file = get_latest_document_by_type("job_description")

    if not jd_file:
        return "No job description document was found."

    files = [jd_file]
    if cv_file:
        files.append(cv_file)

    query = (
        "Generate interview questions based on the role and candidate background."
    )

    results = retrieve_chunks_for_documents(
        query=query,
        file_names=files,
        top_k_per_doc=4
    )

    context = build_multi_doc_context(results)

    if not context.strip():
        return "I could not retrieve enough relevant information for interview preparation."

    messages = [
        {
            "role": "system",
            "content": (
                "You are an interview assistant. "
                "Generate realistic interview questions based on the provided context. "
                "Use this structure:\n"
                "Technical Questions\n"
                "Behavioral Questions\n"
                "Motivation Questions\n"
                "If the candidate seems to have skill gaps, include likely questions around those areas."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                "Generate interview questions for this role."
            )
        }
    ]

    return call_ollama(messages)