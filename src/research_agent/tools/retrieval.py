from __future__ import annotations

from ..models import Source


class RAGRetrievalTool:
    """Queries the ai-document-analyst RAG service over HTTP.

    This is the composition seam between the two projects: the multi-agent
    researcher treats your existing RAG API as one of its tools, so the systems
    stack instead of duplicating retrieval. The response parsing is defensive
    about field names; adjust to match your service's exact schema.
    """

    name = "rag_retrieval"

    def __init__(self, base_url: str, query_path: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._query_path = query_path
        self._timeout = timeout

    async def search(self, query: str, *, top_k: int = 5) -> list[Source]:
        import httpx  # lazy import

        url = f"{self._base_url}{self._query_path}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json={"question": query, "top_k": top_k})
            resp.raise_for_status()
            data = resp.json()

        sources: list[Source] = []
        for item in data.get("sources", []):
            sources.append(
                Source(
                    title=item.get("title") or item.get("document"),
                    locator=str(item.get("id") or item.get("locator") or "") or None,
                    snippet=item.get("snippet") or item.get("text"),
                )
            )
        # If the service returns a synthesized answer but no source list, keep it.
        if not sources and data.get("answer"):
            sources.append(Source(title="RAG answer", snippet=data["answer"]))
        return sources
