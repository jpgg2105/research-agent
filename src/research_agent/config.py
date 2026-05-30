from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """12-factor configuration, read from environment or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # --- LLM provider ---
    llm_provider: str = Field(default="openai")  # openai | anthropic | fake
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-sonnet-4-20250514"  # set to a model you have access to

    # --- Retrieval tool ---
    retrieval_mode: str = Field(default="stub")  # rag | web | stub
    rag_base_url: str = "http://localhost:8000"   # the ai-document-analyst service
    rag_query_path: str = "/api/v1/query"
    top_k: int = 5

    # --- Orchestration ---
    max_iterations: int = 2       # hard cap on research<->critique rounds
    max_sub_questions: int = 5
    request_timeout_s: float = 60.0


def get_settings() -> Settings:
    return Settings()
