import os
import requests
from dotenv import load_dotenv
from typing import Union

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


def call_ollama(messages: list) -> str:
    """
    Send messages to Ollama chat endpoint and return the assistant response.

    Args:
        messages (list): Ollama/OpenAI-style chat messages

    Returns:
        str: Assistant response text
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        assistant_message = data.get("message", {})
        content = assistant_message.get("content", "").strip()

        if not content:
            return "I received an empty response from the model."

        return content

    except requests.exceptions.ConnectionError:
        return "Could not connect to Ollama. Make sure Ollama is running."

    except requests.exceptions.Timeout:
        return "The request to Ollama timed out."

    except requests.exceptions.RequestException as e:
        return f"An error occurred while calling Ollama: {str(e)}"

    except Exception as e:
        return f"Unexpected error in llm_service: {str(e)}"


def format_user_profile(user_profile: dict | None) -> str:
    """
    Convert stored long-term memory into a readable prompt block.

    Args:
        user_profile (dict | None): Stored user memory

    Returns:
        str: Formatted profile text
    """
    if not user_profile:
        return ""

    lines = []

    if user_profile.get("study_goal"):
        lines.append(f"- Study goal: {user_profile['study_goal']}")

    if user_profile.get("exam_days"):
        lines.append(f"- Exam/interview in: {user_profile['exam_days']} days")

    if user_profile.get("subject"):
        lines.append(f"- Subject: {user_profile['subject']}")

    if user_profile.get("preferred_style"):
        lines.append(f"- Preferred response style: {user_profile['preferred_style']}")

    if not lines:
        return ""

    return "User profile:\n" + "\n".join(lines)


def build_system_message(context: str = "", user_profile: dict | None = None) -> dict:
    """
    Build the system message depending on whether document context
    and/or user long-term memory are available.

    Args:
        context (str): Optional retrieved document context
        user_profile (dict | None): Optional long-term memory

    Returns:
        dict: System message
    """
    profile_text = format_user_profile(user_profile)

    if context.strip():
        content = (
            "You are a student assistant chatbot. "
            "Answer the user using the provided document context whenever it is relevant. "
            "Do not invent information that is not supported by the context. "
            "If the answer is not found in the context, clearly say that you could not find it in the uploaded document. "
            "Use the user's stored study information if it helps make the answer more useful and personalized. "
            "Answer clearly in short paragraphs. "
            "Use simple formatting when helpful, such as numbered points or short bullet points. "
            "For section titles, use markdown bold like **Key idea:** "
            "Do not introduce yourself. "
            "Do not say your name. "
            "Keep answers concise unless the user asks for detail."
        )
    else:
        content = (
            "You are a student assistant chatbot. "
            "Use the user's stored study information if it helps make the answer more useful and personalized. "
            "Answer clearly in short paragraphs. "
            "Use simple formatting when helpful, such as numbered points or short bullet points. "
            "For section titles, use markdown bold like **Key idea:** "
            "Do not introduce yourself. "
            "Do not say your name. "
            "Keep answers concise unless the user asks for detail."
        )

    if profile_text:
        content += f"\n\n{profile_text}"

    return {
        "role": "system",
        "content": content
    }


def normalize_messages(messages_or_prompt: Union[list, str]) -> list:
    """
    Normalize input into Ollama/OpenAI-style message list.

    Supports:
    - list of messages
    - plain string prompt
    """
    if isinstance(messages_or_prompt, list):
        return messages_or_prompt

    if isinstance(messages_or_prompt, str):
        prompt = messages_or_prompt.strip()
        if not prompt:
            return []
        return [{"role": "user", "content": prompt}]

    raise ValueError("messages must be either a list of messages or a string prompt.")


def get_llm_response(
    messages: Union[list, str],
    context: str = "",
    user_profile: dict | None = None
) -> str:
    """
    Send conversation history or a plain prompt to Ollama and optionally include
    retrieved document context and user profile for more grounded/personalized answers.

    Args:
        messages (list | str): Chat history in Ollama/OpenAI-style format,
                               or a plain prompt string
        context (str): Retrieved document text to ground the response
        user_profile (dict | None): Stored user memory

    Returns:
        str: Assistant response text
    """
    normalized_messages = normalize_messages(messages)

    if not normalized_messages:
        return "No valid input was provided to the language model."

    system_message = build_system_message(context=context, user_profile=user_profile)

    full_messages = [system_message]

    if context.strip():
        context_message = {
            "role": "system",
            "content": f"Document context:\n{context}"
        }
        full_messages.append(context_message)

    full_messages.extend(normalized_messages)

    return call_ollama(full_messages)


def summarize_with_llm(text: str) -> str:
    """
    Summarize a full document or long text.

    Args:
        text (str): Document text

    Returns:
        str: Summary
    """
    if not text.strip():
        return "No text was provided for summarization."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful academic assistant. "
                "Summarize the given document clearly and accurately. "
                "Write in short paragraphs. "
                "Use markdown headings or bold labels when useful. "
                "Focus on the main topic, key points, important findings, and conclusion. "
                "Do not invent details that are not present in the text."
            )
        },
        {
            "role": "user",
            "content": (
                "Please summarize the following document.\n\n"
                f"{text}"
            )
        }
    ]

    return call_ollama(messages)


def summarize_section_with_llm(section_title: str, text: str) -> str:
    """
    Summarize a specific section or chapter.

    Args:
        section_title (str): Title of the section
        text (str): Section text

    Returns:
        str: Section summary
    """
    if not text.strip():
        return f"No text was provided for summarizing {section_title}."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful academic assistant. "
                "Summarize the given section of a document clearly and accurately. "
                "Write in short paragraphs. "
                "Use markdown bold labels when useful. "
                "Focus only on the content of this section. "
                "Do not invent details that are not present in the text. "
                "Do NOT assume the candidate meets all requirements. "
                "Do NOT copy years of experience from the job description. "
                "Base the response ONLY on the CV."
            )
        },
        {
            "role": "user",
            "content": (
                f"Please summarize this section titled '{section_title}'.\n\n"
                f"{text}"
            )
        }
    ]

    return call_ollama(messages)


def answer_with_context(query: str, context: str, user_profile: dict | None = None) -> str:
    """
    Answer a specific user question using retrieved document context.

    Args:
        query (str): User question
        context (str): Retrieved context from the document
        user_profile (dict | None): Optional stored user memory

    Returns:
        str: Grounded answer
    """
    if not query.strip():
        return "No question was provided."

    if not context.strip():
        return "I could not find relevant information in the uploaded document."

    messages = [
        {
            "role": "user",
            "content": query
        }
    ]

    return get_llm_response(messages, context=context, user_profile=user_profile)


def explain_search_results(keyword: str, matches: list) -> str:
    """
    Format keyword search results more naturally.

    Args:
        keyword (str): Searched keyword
        matches (list): Matching chunks

    Returns:
        str: Formatted response
    """
    if not matches:
        return f"No mentions of '{keyword}' were found in the uploaded documents."

    formatted_results = []
    for match in matches:
        file_name = match.get("file_name", "unknown_file")
        chunk_id = match.get("chunk_id", "unknown_chunk")
        text = match.get("text", "").strip()

        formatted_results.append(
            f"**Document:** {file_name}\n"
            f"**Chunk:** {chunk_id}\n"
            f"{text}"
        )

    return "\n\n".join(formatted_results)