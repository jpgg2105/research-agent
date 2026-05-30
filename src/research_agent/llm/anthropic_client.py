from __future__ import annotations


class AnthropicClient:
    """Anthropic Messages API backed LLMClient. SDK imported lazily."""

    def __init__(self, api_key: str, model: str, timeout: float = 60.0) -> None:
        from anthropic import AsyncAnthropic  # lazy import

        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts)
