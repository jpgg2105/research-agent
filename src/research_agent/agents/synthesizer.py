from __future__ import annotations

from pydantic import BaseModel

from .. import prompts
from ..llm.base import LLMClient
from ..llm.structured import structured_complete
from ..models import Finding, Report
from ..utils import dedupe_sources


class _SynthOutput(BaseModel):
    summary: str
    body: str


class Synthesizer:
    """Integrates findings into a cited report."""

    role = "synthesizer"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def synthesize(self, question: str, findings: list[Finding]) -> Report:
        user = prompts.synthesizer_user(question, findings)
        out = await structured_complete(
            self.llm, system=prompts.SYNTHESIZER_SYSTEM, user=user, response_model=_SynthOutput
        )
        citations = dedupe_sources([s for f in findings for s in f.sources])
        return Report(question=question, summary=out.summary, body=out.body, citations=citations)
