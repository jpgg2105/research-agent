from __future__ import annotations

from pydantic import BaseModel, Field

from .. import prompts
from ..llm.base import LLMClient
from ..llm.structured import structured_complete
from ..models import Finding, SubQuestion
from ..tools.base import RetrievalTool


class _ResearcherOutput(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)


class Researcher:
    """Answers a single sub-question using only retrieved evidence."""

    role = "researcher"

    def __init__(self, llm: LLMClient, tool: RetrievalTool, top_k: int = 5) -> None:
        self.llm = llm
        self.tool = tool
        self.top_k = top_k

    async def research(self, sub_question: SubQuestion) -> Finding:
        sources = await self.tool.search(sub_question.question, top_k=self.top_k)
        user = prompts.researcher_user(sub_question.question, sources)
        out = await structured_complete(
            self.llm, system=prompts.RESEARCHER_SYSTEM, user=user, response_model=_ResearcherOutput
        )
        # Citations come from the retrieval tool, not the model, so the agent
        # cannot fabricate sources.
        return Finding(
            sub_question_id=sub_question.id,
            sub_question=sub_question.question,
            answer=out.answer,
            sources=sources,
            confidence=out.confidence,
        )
