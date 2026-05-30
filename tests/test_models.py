import pytest
from pydantic import ValidationError

from research_agent.models import Finding, ResearchState


def test_confidence_must_be_within_bounds():
    with pytest.raises(ValidationError):
        Finding(sub_question_id=1, sub_question="q", answer="a", confidence=1.5)


def test_state_log_appends_to_trace():
    state = ResearchState(question="q")
    state.log("planner", "planned", "2 sub-questions")
    assert len(state.trace) == 1
    assert state.trace[0].agent == "planner"
    assert state.trace[0].action == "planned"
