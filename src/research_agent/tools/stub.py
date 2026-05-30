from __future__ import annotations

from ..models import Source


class StubRetrievalTool:
    """Deterministic, offline retrieval tool.

    Lets the whole pipeline (and the test suite) run with no external services
    and no API keys. It returns plausible-but-synthetic sources derived from the
    query. Swap it for RAGRetrievalTool or WebSearchTool for real evidence.
    """

    name = "stub_retrieval"

    def __init__(self, corpus: dict[str, list[Source]] | None = None) -> None:
        self._corpus = corpus or {}

    async def search(self, query: str, *, top_k: int = 5) -> list[Source]:
        for key, sources in self._corpus.items():
            if key.lower() in query.lower():
                return sources[:top_k]
        return [
            Source(
                title=f"Reference for: {query[:60]}",
                locator="stub://doc/1",
                snippet=(
                    f"Synthesized context relevant to '{query}'. Replace StubRetrievalTool "
                    "with RAGRetrievalTool or WebSearchTool for real data."
                ),
            )
        ][:top_k]
