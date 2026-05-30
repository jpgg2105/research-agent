from __future__ import annotations

from ..models import Source


class WebSearchTool:
    """Pluggable web-search retrieval.

    Intentionally a stub: wire in your provider of choice (Tavily, Brave,
    SerpAPI, ...) and return a list[Source]. The contract is the same as every
    other RetrievalTool, so the orchestrator is unaffected by the choice.
    """

    name = "web_search"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    async def search(self, query: str, *, top_k: int = 5) -> list[Source]:
        raise NotImplementedError(
            "WebSearchTool is a stub. Implement it against your preferred search API, "
            "returning a list[Source]. See tools/stub.py for a working reference."
        )
