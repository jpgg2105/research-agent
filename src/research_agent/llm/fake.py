from __future__ import annotations

import json

from .usage import Usage


class FakeLLMClient:
    """Deterministic, offline LLM stand-in for demos, smoke tests, and CI.

    It branches on the agent role embedded in the system prompt and returns
    schema-shaped JSON. It does no real reasoning and is not for production --
    its purpose is to let the full pipeline run end to end with no API key.
    """

    model = "fake"

    def __init__(self) -> None:
        self.usage = Usage()

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        self.usage.add(len(system.split()) + len(user.split()), 0)
        if "Planner" in system:
            return json.dumps(
                {
                    "sub_questions": [
                        {"question": "What are the key components involved?", "rationale": "scope"},
                        {"question": "What trade-offs or limitations exist?", "rationale": "weaknesses"},
                    ]
                }
            )
        if "Researcher" in system:
            return json.dumps(
                {
                    "answer": "Based on the retrieved sources, the principal factors are summarized here.",
                    "confidence": 0.78,
                }
            )
        if "Critic" in system:
            return json.dumps({"is_sufficient": True, "missing": [], "follow_up_questions": []})
        if "Synthesizer" in system:
            return json.dumps(
                {
                    "summary": "A concise synthesized summary of the findings.",
                    "body": "A longer, integrated discussion combining the findings into one answer.",
                }
            )
        return "{}"
