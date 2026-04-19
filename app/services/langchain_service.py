from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3"


def build_chat_model():
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
    )


def get_langchain_response(user_message: str, memory: list | None = None) -> str:
    llm = build_chat_model()

    messages = [
        SystemMessage(
            content="You are a helpful student assistant chatbot."
        )
    ]

    if memory:
        for msg in memory:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_message))

    response = llm.invoke(messages)
    return response.content