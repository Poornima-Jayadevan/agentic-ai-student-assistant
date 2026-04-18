import os
import json

STORE_DIR = "data/chunks"
os.makedirs(STORE_DIR, exist_ok=True)


def save_chunks(filename: str, chunks: list[str]):
    json_path = os.path.join(STORE_DIR, f"{filename}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def get_chunks(filename: str):
    json_path = os.path.join(STORE_DIR, f"{filename}.json")
    if not os.path.exists(json_path):
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)