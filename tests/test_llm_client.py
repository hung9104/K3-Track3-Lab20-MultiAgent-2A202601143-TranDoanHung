from types import SimpleNamespace

import pytest

from multi_agent_research_lab.cli import run_baseline
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery
from multi_agent_research_lab.services.llm_client import LLMClient


class FakeCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls = 0

    def create(self, **_: object) -> object:
        response = self.responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


def fake_client(responses: list[object]) -> tuple[SimpleNamespace, FakeCompletions]:
    completions = FakeCompletions(responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def completion(content: str = "A grounded answer") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8),
    )


def test_baseline_records_answer_usage_and_trace() -> None:
    provider, _ = fake_client([completion()])
    llm = LLMClient(Settings(max_retries=0), client=provider)

    state = run_baseline(ResearchQuery(query="Explain multi-agent systems"), llm)

    assert state.final_answer == "A grounded answer"
    assert state.token_usage.total_tokens == 20
    assert state.agent_results[0].agent is AgentName.BASELINE
    assert state.trace[0].name == "baseline_completed"


def test_llm_client_retries_then_succeeds() -> None:
    provider, completions = fake_client([RuntimeError("temporary"), completion()])
    llm = LLMClient(
        Settings(max_retries=1, retry_backoff_seconds=0),
        client=provider,
    )

    response = llm.complete("system", "user")

    assert response.content == "A grounded answer"
    assert completions.calls == 2


def test_llm_client_raises_domain_error_after_retry_limit() -> None:
    provider, _ = fake_client([RuntimeError("provider unavailable")])
    llm = LLMClient(Settings(max_retries=0), client=provider)

    with pytest.raises(AgentExecutionError, match="after 1 attempt"):
        llm.complete("system", "user")
