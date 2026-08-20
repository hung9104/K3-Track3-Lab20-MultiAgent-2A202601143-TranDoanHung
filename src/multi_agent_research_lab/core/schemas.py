"""Public schemas exchanged between CLI, agents, and evaluators."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictSchema(BaseModel):
    """Base contract for data crossing component boundaries."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AgentName(StrEnum):
    BASELINE = "baseline"
    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CRITIC = "critic"


class WorkflowRoute(StrEnum):
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    DONE = "done"


class ResearchQuery(StrictSchema):
    query: str = Field(..., min_length=5)
    max_sources: int = Field(default=5, ge=1, le=20)
    audience: str = "technical learners"


class AgentResult(StrictSchema):
    agent: AgentName
    content: str = Field(..., min_length=1)
    success: bool = True
    latency_seconds: float | None = Field(default=None, ge=0)
    token_usage: "TokenUsage" = Field(default_factory=lambda: TokenUsage())
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceDocument(StrictSchema):
    title: str = Field(..., min_length=1)
    url: str | None = None
    snippet: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def validate_source_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("http://", "https://")):
            raise ValueError("url must use http or https")
        return value


class TokenUsage(StrictSchema):
    """Provider-neutral token counters used for cost reporting."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens


class TraceEvent(StrictSchema):
    """A serializable event for local or remote tracing backends."""

    name: str = Field(..., min_length=1)
    agent: AgentName | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class WorkflowError(StrictSchema):
    """Structured error retained in state for debugging and fallbacks."""

    message: str = Field(..., min_length=1)
    agent: AgentName | None = None
    retryable: bool = False
    attempt: int = Field(default=1, ge=1)


class BenchmarkMetrics(StrictSchema):
    run_name: str
    latency_seconds: float
    estimated_cost_usd: float | None = None
    quality_score: float | None = Field(default=None, ge=0, le=10)
    citation_coverage: float | None = Field(default=None, ge=0, le=1)
    failure_rate: float | None = Field(default=None, ge=0, le=1)
    notes: str = ""
