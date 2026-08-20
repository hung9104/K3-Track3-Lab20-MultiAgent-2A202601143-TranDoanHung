"""Provider-neutral spans and local JSON trace export."""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.schemas import TraceStatus
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.logging import redact

logger = logging.getLogger(__name__)


@dataclass
class Span:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float | None = None
    status: TraceStatus = TraceStatus.OK
    error: str | None = None


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Span]:
    """Measure a block and emit a redacted structured log record."""

    started = perf_counter()
    span = Span(name=name, attributes=redact(attributes or {}))
    try:
        yield span
    except Exception as exc:
        span.status = TraceStatus.ERROR
        span.error = str(redact(str(exc)))
        raise
    finally:
        span.duration_seconds = perf_counter() - started
        logger.info(
            "span=%s status=%s duration_seconds=%.6f attributes=%s error=%s",
            span.name,
            span.status.value,
            span.duration_seconds,
            span.attributes,
            span.error,
        )


def export_trace(state: ResearchState, path: Path) -> Path:
    """Write a redacted, replay-friendly local trace artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": state.run_id,
        "query": state.request.query,
        "route_history": state.route_history,
        "token_usage": state.token_usage.model_dump(),
        "events": [event.model_dump(mode="json") for event in state.trace],
        "errors": [error.model_dump(mode="json") for error in state.errors],
    }
    path.write_text(json.dumps(redact(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
