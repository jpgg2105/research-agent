from __future__ import annotations


class OpenAIClient:
    """OpenAI chat-completions backed LLMClient. SDK imported lazily."""

    def __init__(self, api_key: str, model: str, timeout: float = 60.0) -> None:
        from openai import AsyncOpenAI  # lazy: package imports without the extra installed

        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
