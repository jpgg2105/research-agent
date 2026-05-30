import asyncio
import json

from pydantic import BaseModel

from research_agent.llm.structured import _strip_fences, structured_complete


class _Out(BaseModel):
    name: str
    n: int


class _RetryLLM:
    """Returns garbage once, then valid JSON -- exercises the repair retry."""

    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        self.calls += 1
        if self.calls == 1:
            return "this is not json"
        return json.dumps({"name": "ok", "n": 3})


def test_strip_fences_removes_json_block():
    assert _strip_fences('```json\n{"a":1}\n```') == '{"a":1}'
    assert _strip_fences('{"a":1}') == '{"a":1}'


def test_structured_complete_retries_then_parses():
    llm = _RetryLLM()
    out = asyncio.run(structured_complete(llm, system="s", user="u", response_model=_Out))
    assert out.name == "ok"
    assert out.n == 3
    assert llm.calls == 2  # one failure + one success
