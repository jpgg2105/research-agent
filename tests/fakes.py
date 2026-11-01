import json


class FakeLLM:
    """Role-aware fake LLM for orchestrator tests.

    The ``sufficient`` flag controls whether the Critic asks for another round,
    so we can test both the happy path and the iteration-cap path.
    """

    model = "fake"

    def __init__(self, sufficient: bool = True) -> None:
        self._sufficient = sufficient

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        if "Planner" in system:
            return json.dumps(
                {
                    "sub_questions": [
                        {"question": "What is X?", "rationale": "core"},
                        {"question": "What is Y?", "rationale": "core"},
                    ]
                }
            )
        if "Researcher" in system:
            return json.dumps({"answer": "An answer grounded in the sources.", "confidence": 0.82})
        if "Critic" in system:
            return json.dumps(
                {
                    "is_sufficient": self._sufficient,
                    "missing": [] if self._sufficient else ["an unresolved gap"],
                    "follow_up_questions": [] if self._sufficient else ["What about Z?"],
                }
            )
        if "Synthesizer" in system:
            return json.dumps({"summary": "Short summary.", "body": "Longer body."})
        return "{}"
