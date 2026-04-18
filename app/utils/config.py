import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    # ===============================
    # LLM / Ollama Configuration
    # ===============================
    OLLAMA_BASE_URL: str = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    MODEL_NAME: str = os.getenv(
        "MODEL_NAME", "llama3"
    )

    # ===============================
    # Chat Memory Configuration
    # ===============================
    MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "10"))

    # ===============================
    # App Configuration
    # ===============================
    APP_NAME: str = "Student Assistant Chatbot"
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

# Create a single settings object to import everywhere
settings = Settings()