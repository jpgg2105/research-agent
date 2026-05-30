from __future__ import annotations

from pydantic import BaseModel, Field

from .. import prompts
from ..llm.base import LLMClient
from ..llm.structured import structured_complete
from ..models import Plan, SubQuestion


class _PlannedSubQuestion(BaseModel):
    question: str
    rationale: str = Field(default="")


class _PlannerOutput(BaseModel):
    sub_questions: list[_PlannedSubQuestion]


class Planner:
    """Decomposes the main question into focused sub-questions."""

    role = "planner"

    def __init__(self, llm: LLMClient, max_sub_questions: int = 5) -> None:
        self.llm = llm
        self.max_sub_questions = max_sub_questions

    async def plan(self, question: str, follow_ups: list[str] | None = None) -> Plan:
        user = prompts.planner_user(question, self.max_sub_questions, follow_ups or [])
        out = await structured_complete(
            self.llm, system=prompts.PLANNER_SYSTEM, user=user, response_model=_PlannerOutput
        )
        sub_questions = [
            SubQuestion(id=i, question=sq.question, rationale=sq.rationale)
            for i, sq in enumerate(out.sub_questions[: self.max_sub_questions], start=1)
        ]
        return Plan(main_question=question, sub_questions=sub_questions)
