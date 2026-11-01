import asyncio

from tests.fakes import FakeLLM

from research_agent.agents import Critic, Planner, Researcher, Synthesizer
from research_agent.orchestrator import ResearchOrchestrator
from research_agent.tools.stub import StubRetrievalTool


def _build(llm: FakeLLM, max_iterations: int = 2) -> ResearchOrchestrator:
    return ResearchOrchestrator(
        planner=Planner(llm, max_sub_questions=5),
        researcher=Researcher(llm, StubRetrievalTool()),
        critic=Critic(llm),
        synthesizer=Synthesizer(llm),
        max_iterations=max_iterations,
    )


def test_completes_in_one_round_when_sufficient():
    state = asyncio.run(_build(FakeLLM(sufficient=True)).run("Explain the thing"))
    assert state.report is not None
    assert state.iterations == 1
    assert len(state.findings) == 2
    assert state.report.citations  # stub returns a source per finding


def test_loops_then_stops_at_iteration_cap_when_insufficient():
    state = asyncio.run(_build(FakeLLM(sufficient=False), max_iterations=2).run("Explain the thing"))
    assert state.iterations == 2          # stopped by the cap, not by sufficiency
    assert len(state.findings) == 3       # 2 initial + 1 follow-up
    assert state.report is not None
