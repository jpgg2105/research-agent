from __future__ import annotations

from .ratelimit import limiter_for
from .usage import Usage


class OpenAIClient:
    """OpenAI chat-completions backed LLMClient. SDK imported lazily.

    ``base_url`` makes it work with any OpenAI-compatible endpoint (Gemini,
    a local vLLM server, ...). Token usage is recorded on ``self.usage``.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        base_url: str | None = None,
        max_retries: int = 6,
        rpm: float = 0,
        reasoning_effort: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI  # lazy: package imports without the extra installed

        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout, base_url=base_url, max_retries=max_retries)
        self._model = model
        self._limiter = limiter_for(f"{base_url or 'openai'}|{model}", rpm)
        self._extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        self.usage = Usage()

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        if self._limiter:
            await self._limiter.wait()
        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self._extra,
        )
        u = getattr(resp, "usage", None)
        self.usage.add(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
        return resp.choices[0].message.content or ""
