# app/services/planner_service.py

def generate_study_plan(user_profile: dict) -> str:
    """
    Generate a basic study plan from stored user memory.
    """
    study_goal = user_profile.get("study_goal", "your exam/interview preparation")
    exam_days = user_profile.get("exam_days", "unknown")

    if exam_days != "unknown":
        return f"""
Here is a basic study plan for **{study_goal}** over the next **{exam_days} days**:

### Study Plan
- **Day 1–3:** Review fundamentals and core concepts
- **Day 4–6:** Practice questions and topic-wise revision
- **Day 7–8:** Work on weak areas
- **Day 9–10:** Do mock interview/mock test and revise notes

### Daily Structure
- 1 hour concept revision
- 1 hour practice
- 30 minutes recap
- 30 minutes notes/formulas review
"""
    else:
        return f"""
Here is a basic study plan for **{study_goal}**:

### Weekly Study Plan
- **Monday–Tuesday:** Learn concepts
- **Wednesday–Thursday:** Practice and apply
- **Friday:** Revise weak areas
- **Saturday:** Mock questions / exercises
- **Sunday:** Quick recap and rest

### Daily Structure
- 1 hour learning
- 1 hour practice
- 30 minutes revision
"""