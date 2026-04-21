# app/services/comparison_service.py

from app.services.rag_service import (
    get_latest_document_by_type,
    retrieve_relevant_chunks,
    build_multi_doc_context,
    get_document,
)
from app.services.llm_service import get_llm_response


def compare_cv_with_job_description() -> str:
    try:
        cv_file = get_latest_document_by_type("cv")
        jd_file = get_latest_document_by_type("job_description")

        if not cv_file:
            return "I could not find an uploaded CV."

        if not jd_file:
            return "I could not find an uploaded job description."

        # -----------------------
        # CV Context
        # -----------------------
        cv_results = retrieve_relevant_chunks(
            query="skills experience projects education achievements technical skills",
            file_name=cv_file,
            top_k=5
        )

        if cv_results:
            cv_context = build_multi_doc_context(cv_results, max_chars=3000)
        else:
            cv_doc = get_document(cv_file)
            cv_context = cv_doc["text"][:3000]

        # -----------------------
        # JD Context
        # -----------------------
        jd_results = retrieve_relevant_chunks(
            query="requirements qualifications responsibilities skills experience",
            file_name=jd_file,
            top_k=5
        )

        if jd_results:
            jd_context = build_multi_doc_context(jd_results, max_chars=3000)
        else:
            jd_doc = get_document(jd_file)
            jd_context = jd_doc["text"][:3000]

        prompt = f"""
You are a helpful career assistant.

Compare this CV against the job description.

CV:
{cv_context}

Job Description:
{jd_context}

Return:

1. Overall Match
2. Matching Skills
3. Missing Skills
4. CV Improvement Suggestions
5. Interview Focus Areas

Be specific.
Do not invent information.
""".strip()

        return get_llm_response(prompt)

    except Exception as e:
        return f"Error while comparing documents: {str(e)}"