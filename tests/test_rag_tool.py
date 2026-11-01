"""Contract test: RAGRetrievalTool speaks ai-document-analyst's /query schema."""

import asyncio
import sys
import types

from research_agent.tools.retrieval import RAGRetrievalTool

# A real QueryResponse from ai-document-analyst (src/api/schemas.py), trimmed.
RESPONSE = {
    "answer": "Return within 30 days.",
    "sources": [
        {"chunk_id": "c1", "text": "Refunds within 30 days.", "filename": "refunds.md", "page_number": 1,
         "section": "Refunds", "score": 0.91, "retrieval_method": "reranked"},
        {"chunk_id": "c2", "text": "Shipping takes 3 days.", "filename": "shipping.md", "page_number": 0,
         "section": "", "score": 0.40, "retrieval_method": "reranked"},
    ],
    "confidence": 0.91, "query": "refund window?", "latency_ms": 12.0, "refused": False,
}


def _fake_httpx(sent):
    class Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return RESPONSE

    class Client:
        def __init__(self, timeout=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json):
            sent.append((url, json))
            return Resp()

    return types.SimpleNamespace(AsyncClient=Client)


def test_request_and_response_follow_the_service_schema(monkeypatch):
    sent = []
    monkeypatch.setitem(sys.modules, "httpx", _fake_httpx(sent))
    tool = RAGRetrievalTool("http://rag:8000/", "/api/v1/query")
    sources = asyncio.run(tool.search("refund window?", top_k=1))
    url, body = sent[0]
    assert url == "http://rag:8000/api/v1/query"
    assert body["query"] == "refund window?" and "question" not in body   # QueryRequest.query is required
    assert len(sources) == 1
    assert sources[0].title == "refunds.md" and sources[0].snippet == "Refunds within 30 days."
    assert sources[0].locator.startswith("c1")
