from __future__ import annotations

from .ratelimit import limiter_for
from .usage import Usage


class AnthropicClient:
    """Anthropic Messages API backed LLMClient. SDK imported lazily.

    Token usage is recorded on ``self.usage``.
    """

    def __init__(self, api_key: str, model: str, timeout: float = 60.0, max_retries: int = 6, rpm: float = 0) -> None:
        from anthropic import AsyncAnthropic  # lazy import

        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._model = model
        self._limiter = limiter_for(f"anthropic|{model}", rpm)
        self.usage = Usage()

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        if self._limiter:
            await self._limiter.wait()
        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        u = getattr(resp, "usage", None)
        self.usage.add(getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts)
