def generate_study_plan(subject: str, days: int = 7):
    """
    Generate a simple multi-day study plan for a subject.
    """
    subject = subject.strip()

    if days <= 0:
        return {
            "tool": "study_plan",
            "error": "Days must be greater than 0."
        }

    generic_tasks = [
        "Review core concepts",
        "Read class notes / lecture slides",
        "Practice important problems",
        "Revise weak areas",
        "Solve past questions or exercises",
        "Summarize key ideas",
        "Final revision and self-test"
    ]

    plan = []

    for i in range(days):
        task = generic_tasks[i % len(generic_tasks)]
        plan.append({
            "day": i + 1,
            "topic": f"{subject} - Day {i + 1}",
            "task": task
        })

    return {
        "tool": "study_plan",
        "subject": subject,
        "days": days,
        "plan": plan
    }