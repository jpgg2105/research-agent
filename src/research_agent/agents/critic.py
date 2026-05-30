from __future__ import annotations

from .. import prompts
from ..llm.base import LLMClient
from ..llm.structured import structured_complete
from ..models import Critique, Finding


class Critic:
    """Judges whether the findings sufficiently answer the main question."""

    role = "critic"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def assess(self, question: str, findings: list[Finding]) -> Critique:
        user = prompts.critic_user(question, findings)
        return await structured_complete(
            self.llm, system=prompts.CRITIC_SYSTEM, user=user, response_model=Critique
        )
