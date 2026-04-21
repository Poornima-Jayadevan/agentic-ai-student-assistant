# app/services/interview_prep_service.py

from app.services.rag_service import (
    get_latest_document_by_type,
    list_documents,
    retrieve_relevant_chunks,
    build_multi_doc_context,
)
from app.services.llm_service import get_llm_response


def generate_interview_prep() -> str:
    """
    Generate interview preparation content using the latest uploaded
    CV and job description if available.

    Fallback:
    - if only one of them exists, use that
    - if neither exists, return a helpful message
    """
    try:
        cv_file = get_latest_document_by_type("cv")
        jd_file = get_latest_document_by_type("job_description")

        cv_context = ""
        jd_context = ""

        if cv_file:
            cv_results = retrieve_relevant_chunks(
                query="skills experience projects education achievements technical skills work experience",
                file_name=cv_file,
                top_k=5
            )
            cv_context = build_multi_doc_context(cv_results, max_chars=2500) if cv_results else ""

        if jd_file:
            jd_results = retrieve_relevant_chunks(
                query="requirements responsibilities qualifications required skills preferred skills experience",
                file_name=jd_file,
                top_k=5
            )
            jd_context = build_multi_doc_context(jd_results, max_chars=2500) if jd_results else ""

        if not cv_context and not jd_context:
            docs = list_documents()
            if not docs:
                return "I could not find any uploaded documents for interview preparation."

            return (
                "I found uploaded documents, but not enough structured CV or job description content "
                "to generate interview prep. Please upload a CV and a job description with the correct document types."
            )

        prompt = f"""
You are a helpful interview preparation assistant.

Use the provided content to create interview preparation material.

Candidate CV Content:
{cv_context if cv_context else "Not available."}

Job Description Content:
{jd_context if jd_context else "Not available."}

Create the response in this format:

1. Interview Summary
- Brief 2-3 sentence summary of what the interview is likely to focus on.

2. Likely Interview Questions
- 8 to 10 interview questions
- Mix technical, role-specific, and behavioral questions where relevant

3. What the Candidate Should Highlight
- Bullet points based on the CV that the candidate should mention

4. Skills / Topics to Revise Before the Interview
- Bullet points

5. Strong Answer Tips
- Short practical advice for answering well

Important:
- Only use the provided content
- Do not invent experience, skills, or projects
- If CV or JD content is missing, say so clearly
- Keep the response practical and interview-focused
""".strip()

        response = get_llm_response(prompt)
        return response

    except Exception as e:
        return f"Error while generating interview preparation: {str(e)}"