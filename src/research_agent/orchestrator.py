from __future__ import annotations

import asyncio

from .agents import Critic, Planner, Researcher, Synthesizer
from .models import ResearchState, SubQuestion


class ResearchOrchestrator:
    """Coordinates the agents:

        plan -> research (concurrent) -> critique -> [loop on gaps] -> synthesize

    The research<->critique loop is bounded by ``max_iterations`` so a strict
    critic can never loop forever.
    """

    def __init__(
        self,
        planner: Planner,
        researcher: Researcher,
        critic: Critic,
        synthesizer: Synthesizer,
        max_iterations: int = 2,
    ) -> None:
        self.planner = planner
        self.researcher = researcher
        self.critic = critic
        self.synthesizer = synthesizer
        self.max_iterations = max(1, max_iterations)

    async def run(self, question: str) -> ResearchState:
        state = ResearchState(question=question)

        plan = await self.planner.plan(question)
        state.plan = plan
        state.log("planner", "planned", f"{len(plan.sub_questions)} sub-questions")

        pending: list[SubQuestion] = list(plan.sub_questions)

        while True:
            # Research all currently-pending sub-questions concurrently.
            findings = await asyncio.gather(
                *(self.researcher.research(sq) for sq in pending)
            )
            state.findings.extend(findings)
            for f in findings:
                state.log(
                    "researcher", "answered", f"sq#{f.sub_question_id} (confidence {f.confidence:.2f})"
                )

            state.iterations += 1
            critique = await self.critic.assess(question, state.findings)
            state.critiques.append(critique)
            verdict = "sufficient" if critique.is_sufficient else f"{len(critique.missing)} gap(s)"
            state.log("critic", "assessed", verdict)

            done = (
                critique.is_sufficient
                or not critique.follow_up_questions
                or state.iterations >= self.max_iterations
            )
            if done:
                break

            # Turn the critic's follow-ups into the next batch of sub-questions.
            offset = len(state.findings)
            pending = [
                SubQuestion(
                    id=offset + i + 1,
                    question=q,
                    rationale="Follow-up to close a gap identified by the critic",
                )
                for i, q in enumerate(critique.follow_up_questions[: self.planner.max_sub_questions])
            ]

        report = await self.synthesizer.synthesize(question, state.findings)
        state.report = report
        state.log("synthesizer", "synthesized", f"{len(report.citations)} citations")
        return state
