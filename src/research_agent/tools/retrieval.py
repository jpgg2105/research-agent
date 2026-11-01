from __future__ import annotations

from ..models import Source


class RAGRetrievalTool:
    """Queries the ai-document-analyst RAG service over HTTP.

    The multi-agent researcher uses the RAG service as one of its tools, so the two
    projects stack instead of duplicating retrieval. Request and response follow the
    service's ``QueryRequest`` / ``QueryResponse`` schemas (``src/api/schemas.py`` in
    ai-document-analyst); ``tests/test_rag_tool.py`` pins that contract.
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
            # QueryRequest: {"query": str, "document_id": str | None, "stream": bool}
            resp = await client.post(url, json={"query": query, "stream": False})
            resp.raise_for_status()
            data = resp.json()

        # QueryResponse.sources: [{"chunk_id", "text", "filename", "page_number", "section", "score", ...}]
        sources: list[Source] = []
        for item in data.get("sources", [])[:top_k]:
            page = item.get("page_number")
            sources.append(
                Source(
                    title=item.get("filename") or None,
                    locator=f"{item.get('chunk_id', '')}" + (f" (p. {page})" if page else "") or None,
                    snippet=item.get("text"),
                )
            )
        # If the service returns a synthesized answer but no source list, keep it.
        if not sources and data.get("answer"):
            sources.append(Source(title="RAG answer", snippet=data["answer"]))
        return sources
