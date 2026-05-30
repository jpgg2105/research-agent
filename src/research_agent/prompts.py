from __future__ import annotations

from .models import Finding, Source

PLANNER_SYSTEM = (
    "You are the Planner in a multi-agent research pipeline. "
    "Decompose the user's research question into a small set of focused, "
    "non-overlapping sub-questions that together fully cover the main question. "
    "Prefer fewer, higher-leverage sub-questions over many shallow ones. Each "
    "sub-question must be independently answerable by searching a knowledge source. "
    "Do not answer the questions yourself."
)


def planner_user(question: str, max_sub_questions: int, follow_ups: list[str]) -> str:
    text = f"Main question:\n{question}\n\nProduce at most {max_sub_questions} sub-questions."
    if follow_ups:
        joined = "\n".join(f"- {q}" for q in follow_ups)
        text += f"\n\nThese gaps were identified and must be covered:\n{joined}"
    return text


RESEARCHER_SYSTEM = (
    "You are the Researcher in a multi-agent research pipeline. You are given one "
    "sub-question and a list of retrieved source snippets. Answer the sub-question "
    "using ONLY the information in the provided sources. If the sources are "
    "insufficient, say so plainly and lower your confidence. Never invent facts. "
    "Report a calibrated confidence between 0 and 1."
)


def researcher_user(sub_question: str, sources: list[Source]) -> str:
    if sources:
        rendered = "\n\n".join(
            f"[Source {i}] {s.title or 'untitled'}"
            + (f" ({s.locator})" if s.locator else "")
            + f"\n{s.snippet or ''}"
            for i, s in enumerate(sources, start=1)
        )
    else:
        rendered = "(no sources retrieved)"
    return f"Sub-question:\n{sub_question}\n\nRetrieved sources:\n{rendered}"


CRITIC_SYSTEM = (
    "You are the Critic in a multi-agent research pipeline. Review the findings "
    "gathered so far against the main question and decide whether they are "
    "sufficient to write a complete, well-supported answer. Be strict: identify "
    "concrete gaps, unsupported claims, or low-confidence findings. If gaps exist, "
    "propose specific follow-up questions to close them. If the findings are "
    "sufficient, mark them sufficient and propose no follow-ups."
)


def critic_user(question: str, findings: list[Finding]) -> str:
    rendered = "\n\n".join(
        f"Sub-question: {f.sub_question}\nAnswer: {f.answer}\nConfidence: {f.confidence:.2f}"
        for f in findings
    )
    return f"Main question:\n{question}\n\nFindings so far:\n{rendered}"


SYNTHESIZER_SYSTEM = (
    "You are the Synthesizer in a multi-agent research pipeline. Using only the "
    "gathered findings, write a clear, well-structured answer to the main question. "
    "Produce a short executive summary and a longer body. Integrate the findings "
    "into a coherent narrative rather than listing them. Do not introduce facts "
    "that are not present in the findings."
)


def synthesizer_user(question: str, findings: list[Finding]) -> str:
    rendered = "\n\n".join(
        f"Finding {i} (confidence {f.confidence:.2f}):\n{f.answer}"
        for i, f in enumerate(findings, start=1)
    )
    return f"Main question:\n{question}\n\nFindings:\n{rendered}"
