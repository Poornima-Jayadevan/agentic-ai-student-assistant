import re
from app.tools.calculator_tool import calculate_expression
from app.tools.study_plan_tool import generate_study_plan


def is_calculation_query(message: str) -> bool:
    msg = message.lower().strip()

    calculator_keywords = [
        "calculate",
        "what is",
        "solve",
        "compute",
        "evaluate"
    ]

    math_pattern = r"^\s*\d+(\.\d+)?\s*[\+\-\*\/\%]\s*\d+(\.\d+)?\s*$"

    if re.match(math_pattern, msg):
        return True

    if any(keyword in msg for keyword in calculator_keywords):
        if re.search(r"[\d]+\s*[\+\-\*\/\%]", msg):
            return True

    return False


def extract_expression(message: str) -> str:
    msg = message.lower().strip()

    msg = msg.replace("calculate", "")
    msg = msg.replace("what is", "")
    msg = msg.replace("solve", "")
    msg = msg.replace("compute", "")
    msg = msg.replace("evaluate", "")
    msg = msg.replace("?", "").strip()

    return msg


def is_study_plan_query(message: str) -> bool:
    msg = message.lower()

    study_plan_keywords = [
        "study plan",
        "make me a study plan",
        "create a study plan",
        "7-day study plan",
        "plan for",
        "revision plan"
    ]

    return any(keyword in msg for keyword in study_plan_keywords)


def extract_study_plan_details(message: str):
    msg = message.strip()

    # Default values
    days = 7
    subject = "General Study"

    # Extract days like "7-day", "10 day", "for 5 days"
    day_match = re.search(r"(\d+)\s*[- ]?\s*day", msg.lower())
    if day_match:
        days = int(day_match.group(1))

    # Try extracting subject after "for"
    subject_match = re.search(r"for\s+(.+)", msg, re.IGNORECASE)
    if subject_match:
        subject = subject_match.group(1).strip()

    return subject, days


def build_tool_result(
    tool: str,
    success: bool = True,
    result=None,
    error: str = "",
    metadata: dict | None = None,
    **extra
) -> dict:
    return {
        "success": success,
        "tool": tool,
        "result": result,
        "error": error,
        "metadata": metadata or {},
        **extra
    }


def detect_and_run_tool(message: str):
    try:
        if is_calculation_query(message):
            expression = extract_expression(message)
            result = calculate_expression(expression)

            return build_tool_result(
                tool="calculator",
                result=result,
                input=expression,
            )

        if is_study_plan_query(message):
            subject, days = extract_study_plan_details(message)
            plan = generate_study_plan(subject, days)

            return build_tool_result(
                tool="study_plan",
                result=plan,
                subject=subject,
                days=days,
                plan=plan,
            )

        return None

    except Exception as e:
        return build_tool_result(
            tool="unknown",
            success=False,
            error=str(e),
        )