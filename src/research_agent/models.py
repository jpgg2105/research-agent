from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A retrieved piece of evidence. Produced by a RetrievalTool, never by the LLM."""

    title: str | None = None
    locator: str | None = Field(default=None, description="URL, document id, page, etc.")
    snippet: str | None = None


class SubQuestion(BaseModel):
    id: int
    question: str
    rationale: str = ""


class Plan(BaseModel):
    main_question: str
    sub_questions: list[SubQuestion]


class Finding(BaseModel):
    sub_question_id: int
    sub_question: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class Critique(BaseModel):
    is_sufficient: bool
    missing: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


class Report(BaseModel):
    question: str
    summary: str
    body: str
    citations: list[Source] = Field(default_factory=list)


class StepTrace(BaseModel):
    agent: str
    action: str
    detail: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchState(BaseModel):
    """The full, inspectable state of one research run."""

    question: str
    plan: Plan | None = None
    findings: list[Finding] = Field(default_factory=list)
    critiques: list[Critique] = Field(default_factory=list)
    report: Report | None = None
    iterations: int = 0
    trace: list[StepTrace] = Field(default_factory=list)

    def log(self, agent: str, action: str, detail: str = "") -> None:
        self.trace.append(StepTrace(agent=agent, action=action, detail=detail))
