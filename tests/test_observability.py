import json
from pathlib import Path

import pytest

from multi_agent_research_lab.core.schemas import ResearchQuery, TraceStatus
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.logging import REDACTED, redact
from multi_agent_research_lab.observability.tracing import export_trace, trace_span


def test_redact_removes_nested_secrets() -> None:
    value = {
        "OPENAI_API_KEY": "sk-super-secret-value",
        "nested": {"authorization": "Bearer abcdefghijklmnop"},
        "message": "failed with sk-another-secret-value",
    }

    sanitized = redact(value)

    assert sanitized["OPENAI_API_KEY"] == REDACTED
    assert sanitized["nested"]["authorization"] == REDACTED
    assert "sk-" not in sanitized["message"]


def test_trace_span_records_error_status() -> None:
    captured = None
    with (
        pytest.raises(RuntimeError),
        trace_span("failing-operation", {"api_key": "secret"}) as span,
    ):
        captured = span
        raise RuntimeError("provider failed")

    assert captured is not None
    assert captured.status is TraceStatus.ERROR
    assert captured.duration_seconds is not None
    assert captured.attributes["api_key"] == REDACTED


def test_export_trace_writes_redacted_json(tmp_path: Path) -> None:
    output_dir = tmp_path
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.add_trace_event("provider", {"api_key": "sk-super-secret-value"})

    output = export_trace(state, output_dir / "trace.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["run_id"] == state.run_id
    assert payload["events"][0]["payload"]["api_key"] == REDACTED
