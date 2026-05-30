from __future__ import annotations

from ..config import Settings
from .base import LLMClient


def build_llm_client(settings: Settings) -> LLMClient:
    provider = settings.llm_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from .openai_client import OpenAIClient

        return OpenAIClient(settings.openai_api_key, settings.openai_model, settings.request_timeout_s)

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        from .anthropic_client import AnthropicClient

        return AnthropicClient(
            settings.anthropic_api_key, settings.anthropic_model, settings.request_timeout_s
        )

    if provider == "fake":
        from .fake import FakeLLMClient

        return FakeLLMClient()

    raise ValueError(
        f"Unknown LLM_PROVIDER: {settings.llm_provider!r} (expected 'openai', 'anthropic', or 'fake')"
    )
