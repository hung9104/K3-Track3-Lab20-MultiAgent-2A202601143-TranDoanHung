import pytest
from pydantic import ValidationError

from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
    TokenUsage,
)
from multi_agent_research_lab.core.state import ResearchState


def test_state_records_route_and_trace() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.record_route("researcher")
    state.add_trace_event("route", {"next": "researcher"})
    assert state.iteration == 1
    assert state.route_history == ["researcher"]
    assert state.trace[0].name == "route"


def test_state_accumulates_usage_and_structured_errors() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.add_agent_result(
        AgentResult(
            agent=AgentName.RESEARCHER,
            content="Research completed",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )
    )
    state.add_error(
        "Temporary provider failure",
        agent=AgentName.RESEARCHER,
        retryable=True,
        attempt=1,
    )

    assert state.token_usage.total_tokens == 15
    assert state.errors[0].retryable is True


def test_public_schemas_reject_invalid_boundary_data() -> None:
    with pytest.raises(ValidationError):
        ResearchQuery(query="   ")
    with pytest.raises(ValidationError):
        SourceDocument(title="Source", snippet="Content", url="ftp://example.com")
    with pytest.raises(ValueError):
        ResearchState(request=ResearchQuery(query="Valid query")).record_route("unknown")
