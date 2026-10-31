# Research Agent

> A multi-agent research system that **plans, retrieves, critiques, and synthesizes** a cited report from a single question. It is designed to compose with the [ai-document-analyst](https://github.com/jpgg2105/ai-document-analyst) RAG service — the agent uses that RAG API as one of its retrieval tools.

## Why this exists

A single RAG call answers one question against one retrieval. Real research questions need more: decomposition into parts, multiple retrievals, a check for what's missing, and synthesis into a coherent answer. This project adds that orchestration layer — a small team of role-specialised agents that loop until the question is actually covered, then write the report.

Two deliberate constraints make it a useful portfolio piece rather than a framework demo:

- **Framework-light.** No LangChain / CrewAI dependency. The agent loop — planning, tool use, reflection, termination — is visible and auditable in a few hundred lines instead of hidden inside a framework.
- **Provider-agnostic.** OpenAI *or* Anthropic behind one `LLMClient` interface, switchable with an environment variable — matching how most teams actually ship.

## How it composes with the RAG project

```
question
   │
   ▼
┌───────────────────────── Research Agent ─────────────────────────┐
│                                                                   │
│   Planner ──▶ Researcher (×N, concurrent) ──▶ Critic ──┐          │
│                      │                                 │ gaps?    │
│                      │  each Researcher calls a         │ (loop)   │
│                      │  RetrievalTool                   ▼          │
│                      │                            Synthesizer      │
│                      ▼                                  │          │
│              RetrievalTool (pluggable)                  ▼          │
│              ├─ rag  ── HTTP ─▶ ai-document-analyst /api/v1/query  │
│              ├─ web  ── your search provider                      │
│              └─ stub ── offline, deterministic (default)          │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
   │
   ▼
cited report  +  full execution trace
```

Set `RETRIEVAL_MODE=rag` and `RAG_BASE_URL` to point the researcher at your running RAG service, or `RETRIEVAL_MODE=local` with `LOCAL_CORPUS_PATH` to search a JSONL file in-process with BM25 (used by the benchmark). Out of the box it runs in `stub` mode, so the whole pipeline works with no external services.

## The agents

| Agent | Responsibility | Validated output |
| --- | --- | --- |
| **Planner** | Decompose the question into focused, non-overlapping sub-questions | `Plan` |
| **Researcher** | Answer one sub-question using *only* retrieved sources; report calibrated confidence | `Finding` |
| **Critic** | Decide whether the findings are sufficient; name gaps and propose follow-ups | `Critique` |
| **Synthesizer** | Weave findings into a summary + body with de-duplicated citations | `Report` |

The **orchestrator** runs the Planner once, researches all sub-questions concurrently, then loops Researcher → Critic while the Critic still finds gaps — bounded by `MAX_ITERATIONS` — before calling the Synthesizer.

## Tech stack

| Component | Choice | Why |
| --- | --- | --- |
| Language | Python 3.11+ (async) | Concurrent retrieval via `asyncio.gather` |
| API | FastAPI | Async-native, automatic OpenAPI docs |
| Validation | Pydantic v2 | Every agent returns a typed, validated model — no raw-text parsing downstream |
| LLM | OpenAI **or** Anthropic | One `LLMClient` interface, swap with an env var |
| Retrieval | Pluggable tool | `rag` (your service) / `web` / `stub` |
| Config | pydantic-settings | 12-factor environment configuration |
| Tests | pytest | Run **fully offline** with fakes — no API keys, no network |

## Quick start

### 1. Offline smoke test (no API key, no services)

```bash
pip install -e ".[dev]"
LLM_PROVIDER=fake python -m research_agent.cli "How does hybrid retrieval improve RAG precision?"
```

This runs the entire Planner → Researcher → Critic → Synthesizer flow using a deterministic fake LLM and the offline stub retrieval tool. Useful for demos, screen recordings, and CI.

### 2. Real run

```bash
pip install -e ".[openai]"        # or ".[anthropic]"
export LLM_PROVIDER=openai        # or anthropic
export OPENAI_API_KEY=sk-...
export RETRIEVAL_MODE=stub        # or rag / web
python -m research_agent.cli "What are the trade-offs of cross-encoder reranking?"
```

Or serve the HTTP API:

```bash
uvicorn research_agent.api:app --reload --port 8001
# POST http://localhost:8001/api/v1/research   {"question": "..."}
# docs http://localhost:8001/docs
```

### 3. Wire it to your RAG service

```bash
export RETRIEVAL_MODE=rag
export RAG_BASE_URL=http://localhost:8000   # your ai-document-analyst instance
```

The researcher will now answer each sub-question against documents you've ingested into the RAG system, with real citations flowing through to the final report.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/research` | Run the pipeline on `{"question": "..."}`; returns the report plus the full execution trace |

## Design decisions

- **Citations come from the retrieval tool, not the model.** The Researcher's LLM output is only an answer + a confidence; the sources attached to every `Finding` are exactly what the tool returned. This makes fabricated citations structurally impossible.
- **Structured outputs via schema-instructed prompting + validation + repair.** Each agent gets the target JSON Schema and its output is validated against a Pydantic model; on a parse failure the error is fed back for one repair attempt. This keeps a single code path across providers. In production you might swap in each provider's native structured-output mode.
- **The reflection loop is bounded.** A strict Critic could otherwise loop forever; `MAX_ITERATIONS` guarantees termination, and the loop also stops early the moment the Critic is satisfied.
- **Sub-questions are researched concurrently** with `asyncio.gather`, so wall-clock time scales with the slowest retrieval, not the sum.
- **Retrieval is the only seam that varies.** Swapping `stub` → `rag` → `web` changes nothing about the agents or the orchestrator.

## Lessons learned

- A planning step plus a critic loop catches the *confidently incomplete* answers that a single RAG call quietly misses — the Critic's follow-up questions are where most of the quality comes from.
- Taking citations from the tool rather than the model eliminated fabricated sources entirely, with no prompt-engineering required.
- Schema-instructed JSON with a one-shot repair retry proved more portable across providers than native structured-output APIs, at the cost of a little prompt overhead.
- Capping iterations is not optional: without the cap, a strict critic and a weak retriever can loop indefinitely.

## Testing

```bash
pip install -e ".[dev]"
pytest -q
```

The suite runs fully offline using a role-aware fake LLM and the stub retrieval tool — no API keys and no network. It covers the happy path, the iteration-cap path, the structured-output repair retry, and model validation.

## Project layout

```
src/research_agent/
├── config.py          # pydantic-settings configuration
├── models.py          # Plan, Finding, Critique, Report, ResearchState, ...
├── prompts.py         # all agent prompts in one place
├── orchestrator.py    # the plan → research → critique → synthesize loop
├── factory.py         # wire an orchestrator from settings
├── api.py             # FastAPI surface
├── cli.py             # command-line entry point
├── llm/               # provider-agnostic LLM clients + structured-output helper
│   ├── base.py        # LLMClient protocol
│   ├── openai_client.py
│   ├── anthropic_client.py
│   ├── fake.py        # offline deterministic client
│   ├── structured.py  # schema-instructed JSON + validation + retry
│   └── factory.py
├── tools/             # retrieval tools
│   ├── base.py        # RetrievalTool protocol
│   ├── retrieval.py   # RAGRetrievalTool — composes with ai-document-analyst
│   ├── web_search.py  # pluggable web search (stub)
│   ├── stub.py        # offline deterministic retrieval
│   └── factory.py
└── agents/
    ├── planner.py
    ├── researcher.py
    ├── critic.py
    └── synthesizer.py
tests/                 # offline pytest suite
```

## License

MIT
