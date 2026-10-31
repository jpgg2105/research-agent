import asyncio
import json

from research_agent.llm.fake import FakeLLMClient
from research_agent.tools.local_bm25 import LocalBM25Tool


def _corpus(tmp_path):
    docs = [
        {"id": "d1", "title": "Eiffel Tower", "text": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France."},
        {"id": "d2", "title": "Gustave Eiffel", "text": "Gustave Eiffel was a French civil engineer whose company built the tower."},
        {"id": "d3", "title": "Colosseum", "text": "The Colosseum is an oval amphitheatre in the centre of Rome, Italy."},
    ]
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    return p


def test_bm25_ranks_the_matching_document_first(tmp_path):
    tool = LocalBM25Tool(_corpus(tmp_path))
    hits = asyncio.run(tool.search("Which engineer built the tower in Paris?", top_k=2))
    assert [h.locator for h in hits][0] in {"d1", "d2"}
    assert all(h.locator != "d3" for h in hits)


def test_bm25_drops_documents_with_no_overlap(tmp_path):
    tool = LocalBM25Tool(_corpus(tmp_path))
    assert asyncio.run(tool.search("quantum chromodynamics", top_k=3)) == []


def test_clients_record_usage():
    llm = FakeLLMClient()
    asyncio.run(llm.complete(system="You are the Planner", user="hello there"))
    assert llm.usage.calls == 1 and llm.usage.input_tokens > 0
