# app/services/rag_answer_service.py

from app.services.llm_service import get_llm_response


def generate_rag_answer(user_question: str, context: str) -> str:
    """
    Generate a polished final answer using retrieved document context.
    """
    if not context or not context.strip():
        return "I could not find enough relevant information in the document to answer that."

    prompt = f"""
You are a helpful student assistant chatbot.

Use only the document context below to answer the user's question.

Instructions:
- Give a clear, natural, well-structured answer.
- Do not copy large chunks of the context unless necessary.
- If the answer is only partially supported by the context, say that clearly.
- If the answer is not available in the context, say that honestly.
- Keep the response concise but useful.

Document Context:
{context}

User Question:
{user_question}

Answer:
""".strip()

    response = get_llm_response(prompt)

    if not response or not response.strip():
        return "I found relevant document content, but I could not generate a clear answer."

    return response.strip()