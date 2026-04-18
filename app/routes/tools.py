from fastapi import APIRouter
from pydantic import BaseModel
from app.tools.calculator_tool import calculate_expression
from app.tools.study_plan_tool import generate_study_plan

router = APIRouter(prefix="/tools", tags=["Tools"])


class CalculatorRequest(BaseModel):
    expression: str


class StudyPlanRequest(BaseModel):
    subject: str
    days: int = 7


@router.post("/calculator")
def calculator(request: CalculatorRequest):
    return calculate_expression(request.expression)


@router.post("/study-plan")
def study_plan(request: StudyPlanRequest):
    return generate_study_plan(request.subject, request.days)