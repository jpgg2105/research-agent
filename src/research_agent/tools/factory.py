from __future__ import annotations

from ..config import Settings
from .base import RetrievalTool


def build_retrieval_tool(settings: Settings) -> RetrievalTool:
    mode = settings.retrieval_mode.lower()

    if mode == "rag":
        from .retrieval import RAGRetrievalTool

        return RAGRetrievalTool(settings.rag_base_url, settings.rag_query_path, settings.request_timeout_s)

    if mode == "web":
        from .web_search import WebSearchTool

        return WebSearchTool()

    if mode == "local":
        if not settings.local_corpus_path:
            raise ValueError("LOCAL_CORPUS_PATH is required when RETRIEVAL_MODE=local")
        from .local_bm25 import LocalBM25Tool

        return LocalBM25Tool(settings.local_corpus_path)

    if mode == "stub":
        from .stub import StubRetrievalTool

        return StubRetrievalTool()

    raise ValueError(
        f"Unknown RETRIEVAL_MODE: {settings.retrieval_mode!r} (expected 'rag', 'web', 'local', or 'stub')"
    )
