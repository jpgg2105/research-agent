from __future__ import annotations

from .agents import Critic, Planner, Researcher, Synthesizer
from .config import Settings
from .llm.factory import build_llm_client
from .orchestrator import ResearchOrchestrator
from .tools.factory import build_retrieval_tool


def build_orchestrator(settings: Settings) -> ResearchOrchestrator:
    """Wire an orchestrator from configuration: one LLM client, one retrieval
    tool, four agents."""
    llm = build_llm_client(settings)
    tool = build_retrieval_tool(settings)
    return ResearchOrchestrator(
        planner=Planner(llm, max_sub_questions=settings.max_sub_questions),
        researcher=Researcher(llm, tool, top_k=settings.top_k),
        critic=Critic(llm),
        synthesizer=Synthesizer(llm),
        max_iterations=settings.max_iterations,
    )
