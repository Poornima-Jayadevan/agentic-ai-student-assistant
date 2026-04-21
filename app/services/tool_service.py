# app/services/tool_service.py

from __future__ import annotations

import ast
import operator
from typing import Optional

from app.services.retriever_service import build_context
import re

# -----------------------------
# Calculator Tool
# -----------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """
    Safely evaluate a simple math expression using Python AST.
    Allowed:
    +, -, *, /, %, **, //
    """
    if isinstance(node, ast.Constant):  # numbers
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.Num):  # fallback for older Python versions
        return node.n

    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_type = type(node.op)

        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Unsupported operator.")

        return _ALLOWED_OPERATORS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_type = type(node.op)

        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Unsupported unary operator.")

        return _ALLOWED_OPERATORS[op_type](operand)

    raise ValueError("Invalid expression.")


def calculator_tool(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression safely.

    Example:
        input:  "25 * 3"
        output: "75"
    """
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed.body)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as e:
        return f"Error: invalid calculation ({e})."


# -----------------------------
# Retriever Tool
# -----------------------------

def retriever_tool(query: str, file_name: str, top_k: int = 3) -> str:
    """
    Retrieve relevant document context for a given query and file.

    Example:
        input:  query="What skills are required?", file_name="job description_1"
        output: relevant chunk text joined together
    """
    try:
        context = build_context(query=query, file_name=file_name, top_k=top_k)

        if not context.strip():
            return "No relevant content found."

        return context

    except Exception as e:
        return f"Error: could not retrieve document context ({e})."


# -----------------------------
# Optional helper detectors
# -----------------------------

def looks_like_calculation(message: str) -> bool:
    msg = message.lower().strip()

    # direct math expression
    if re.fullmatch(r"\d+(\.\d+)?\s*[\+\-\*\/%]\s*\d+(\.\d+)?", msg):
        return True

    # text requests with numbers
    if re.search(r"\b(calculate|solve)\b", msg):
        return True

    if re.search(r"what is\s+\d+(\.\d+)?\s*[\+\-\*\/%]\s*\d+(\.\d+)?", msg):
        return True

    return False


def looks_like_document_query(message: str) -> bool:
    """
    Simple detection for document/PDF retrieval intent.
    """
    msg = message.strip().lower()

    keywords = [
        "in this document",
        "in this pdf",
        "from this pdf",
        "from this document",
        "according to the document",
        "according to the pdf",
        "find in",
        "search in",
        "what does this document say",
        "what does this pdf say",
    ]

    return any(keyword in msg for keyword in keywords)

def extract_math_expression(message: str) -> str:
    """
    Extract a clean math expression from a user message.
    Example:
        'What is 25 * 3?' -> '25 * 3'
    """
    msg = message.lower().strip()

    phrases_to_remove = [
        "what is",
        "calculate",
        "compute",
        "solve",
    ]

    for phrase in phrases_to_remove:
        msg = msg.replace(phrase, "")

    msg = msg.replace("?", "").strip()
    return msg