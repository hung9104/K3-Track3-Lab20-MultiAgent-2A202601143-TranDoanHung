"""Shared state for the multi-agent workflow.

Students should extend this file when adding new agents, outputs, or evaluation metrics.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    ResearchQuery,
    SourceDocument,
    TokenUsage,
    TraceEvent,
    WorkflowError,
    WorkflowRoute,
)


class ResearchState(BaseModel):
    """Single source of truth passed through the workflow."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request: ResearchQuery
    iteration: int = 0
    route_history: list[str] = Field(default_factory=list)

    sources: list[SourceDocument] = Field(default_factory=list)
    research_notes: str | None = None
    analysis_notes: str | None = None
    final_answer: str | None = None

    agent_results: list[AgentResult] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    trace: list[TraceEvent] = Field(default_factory=list)
    errors: list[WorkflowError] = Field(default_factory=list)

    def record_route(self, route: AgentName | WorkflowRoute | str) -> None:
        route_value = WorkflowRoute(route).value
        self.route_history.append(route_value)
        self.iteration += 1

    def add_trace_event(
        self, name: str, payload: dict[str, Any], agent: AgentName | None = None
    ) -> None:
        self.trace.append(TraceEvent(name=name, agent=agent, payload=payload))

    def add_agent_result(self, result: AgentResult) -> None:
        self.agent_results.append(result)
        self.token_usage.add(result.token_usage)

    def add_error(
        self,
        message: str,
        *,
        agent: AgentName | None = None,
        retryable: bool = False,
        attempt: int = 1,
    ) -> None:
        self.errors.append(
            WorkflowError(
                message=message,
                agent=agent,
                retryable=retryable,
                attempt=attempt,
            )
        )
