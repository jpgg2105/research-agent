from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from .config import get_settings
from .factory import build_orchestrator
from .orchestrator import ResearchOrchestrator


class ResearchRequest(BaseModel):
    question: str


@lru_cache(maxsize=1)
def _orchestrator() -> ResearchOrchestrator:
    # Built once and reused so we don't recreate provider clients per request.
    return build_orchestrator(get_settings())


app = FastAPI(title="Research Agent", version="0.1.0")


@app.get("/api/v1/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/research")
async def research(req: ResearchRequest) -> dict:
    state = await _orchestrator().run(req.question)
    return {
        "question": state.question,
        "report": state.report.model_dump() if state.report else None,
        "iterations": state.iterations,
        "num_findings": len(state.findings),
        "trace": [t.model_dump(mode="json") for t in state.trace],
    }
