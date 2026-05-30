from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal provider-agnostic chat interface.

    Implementations: OpenAIClient, AnthropicClient, FakeLLMClient.
    """

    @property
    def model(self) -> str: ...

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str: ...
