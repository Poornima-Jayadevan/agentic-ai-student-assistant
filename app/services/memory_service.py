# app/services/memory_service.py

from typing import Dict, List

# -------------------------------
# Chat memory (short-term memory)
# -------------------------------
chat_memory: Dict[str, List[dict]] = {}

# -------------------------------
# User memory (long-term memory)
# -------------------------------
user_memory: Dict[str, dict] = {}


# -------------------------------
# CHAT MEMORY FUNCTIONS
# -------------------------------

def get_memory(user_id: str, limit: int = 10) -> List[dict]:
    """
    Return last N messages for a user.
    """
    return chat_memory.get(user_id, [])[-limit:]


def add_message(user_id: str, role: str, content: str):
    """
    Add a message to the user's chat history.
    """
    if user_id not in chat_memory:
        chat_memory[user_id] = []

    chat_memory[user_id].append({
        "role": role,
        "content": content
    })


def clear_memory(user_id: str):
    """
    Clear a user's chat history safely.
    """
    if user_id in chat_memory:
        chat_memory[user_id] = []


# -------------------------------
# USER MEMORY FUNCTIONS (NEW)
# -------------------------------

def get_user_profile(user_id: str) -> dict:
    """
    Get stored long-term memory for a user.
    Example:
    {
        "study_goal": "XAI interviews",
        "exam_days": "10"
    }
    """
    return user_memory.get(user_id, {})


def save_user_goal(user_id: str, key: str, value: str):
    """
    Save a single user goal or preference.

    Example:
    key = "study_goal"
    value = "prepare for XAI interviews"
    """
    if user_id not in user_memory:
        user_memory[user_id] = {}

    user_memory[user_id][key] = value


def update_user_profile(user_id: str, updates: dict):
    """
    Update multiple user memory fields at once.

    Example:
    updates = {
        "study_goal": "XAI interviews",
        "exam_days": "10"
    }
    """
    if user_id not in user_memory:
        user_memory[user_id] = {}

    user_memory[user_id].update(updates)


def clear_user_profile(user_id: str):
    """
    Clear long-term memory for a user.
    """
    if user_id in user_memory:
        user_memory[user_id] = {}


# -------------------------------
# OPTIONAL: DEBUG FUNCTION
# -------------------------------

def get_full_memory(user_id: str) -> dict:
    """
    Returns both chat + user memory (useful for debugging).
    """
    return {
        "chat_memory": chat_memory.get(user_id, []),
        "user_memory": user_memory.get(user_id, {})
    }