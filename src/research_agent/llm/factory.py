from __future__ import annotations

from ..config import Settings
from .base import LLMClient


def build_llm_client(settings: Settings) -> LLMClient:
    provider = settings.llm_provider.lower()

    if provider == "openai":
        key = settings.openai_api_key
        if settings.openai_base_url and "generativelanguage.googleapis.com" in settings.openai_base_url:
            key = settings.gemini_api_key or key   # Gemini's OpenAI-compatible endpoint
        if not key:
            raise ValueError("OPENAI_API_KEY (or GEMINI_API_KEY for Gemini) is required when LLM_PROVIDER=openai")
        from .openai_client import OpenAIClient

        return OpenAIClient(
            key,
            settings.openai_model,
            settings.request_timeout_s,
            base_url=settings.openai_base_url,
            max_retries=settings.llm_max_retries,
            rpm=settings.llm_rpm,
            reasoning_effort=settings.openai_reasoning_effort,
        )

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        from .anthropic_client import AnthropicClient

        return AnthropicClient(
            settings.anthropic_api_key,
            settings.anthropic_model,
            settings.request_timeout_s,
            max_retries=settings.llm_max_retries,
            rpm=settings.llm_rpm,
        )

    if provider == "fake":
        from .fake import FakeLLMClient

        return FakeLLMClient()

    raise ValueError(
        f"Unknown LLM_PROVIDER: {settings.llm_provider!r} (expected 'openai', 'anthropic', or 'fake')"
    )
