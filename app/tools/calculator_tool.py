import ast
import operator as op
import re

# Supported operators
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
}


def _evaluate(node):
    if isinstance(node, ast.Constant):  # numbers
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")

    elif isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError("Operator not allowed.")

        return ALLOWED_OPERATORS[operator_type](left, right)

    elif isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand)
        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError("Operator not allowed.")

        return ALLOWED_OPERATORS[operator_type](operand)

    raise ValueError("Invalid expression.")


def calculate_expression(expression: str):
    """
    Safely evaluate a simple math expression.
    Supported: +, -, *, /, %, **, parentheses
    """

    try:
        # ----------------------------
        # 1. Clean user input
        # ----------------------------
        cleaned = expression.lower()

        # Remove common words
        cleaned = cleaned.replace("what is", "")
        cleaned = cleaned.replace("calculate", "")
        cleaned = cleaned.replace("please", "")
        cleaned = cleaned.replace("?", "")
        cleaned = cleaned.strip()

        # ----------------------------
        # 2. Validate allowed characters
        # ----------------------------
        if not re.fullmatch(r"[0-9+\-*/(). %]+", cleaned):
            return {
                "tool": "calculator",
                "input": expression,
                "error": "Only simple math expressions are supported."
            }

        # ----------------------------
        # 3. Parse safely using AST
        # ----------------------------
        parsed = ast.parse(cleaned, mode="eval")
        result = _evaluate(parsed.body)

        return {
            "tool": "calculator",
            "input": cleaned,
            "result": result
        }

    except ZeroDivisionError:
        return {
            "tool": "calculator",
            "input": expression,
            "error": "Division by zero is not allowed."
        }

    except Exception as e:
        return {
            "tool": "calculator",
            "input": expression,
            "error": f"Invalid calculation: {str(e)}"
        }