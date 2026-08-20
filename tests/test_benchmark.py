import pytest

from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery, TokenUsage
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    citation_coverage,
    estimate_cost,
    run_benchmark_suite,
)


def successful_runner(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    state.final_answer = "A supported factual sentence [1]. Another unsupported sentence."
    state.add_agent_result(
        AgentResult(
            agent=AgentName.BASELINE,
            content=state.final_answer,
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        )
    )
    return state


def failing_runner(query: str) -> ResearchState:
    del query
    raise RuntimeError("provider unavailable")


def test_citation_coverage_and_cost() -> None:
    state = successful_runner("Explain multi-agent systems")
    assert citation_coverage(state.final_answer) == pytest.approx(0.5)
    assert estimate_cost(state, 1.0, 2.0) == pytest.approx(0.0002)
    assert estimate_cost(state, None, 2.0) is None


def test_benchmark_suite_aggregates_success_and_failure() -> None:
    metrics = run_benchmark_suite(
        {"success": successful_runner, "failure": failing_runner},
        ["Explain multi-agent systems"],
        repeats=2,
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )

    success, failure = metrics
    assert success.sample_count == 2
    assert success.total_tokens == 150
    assert success.failure_rate == 0
    assert success.citation_coverage == pytest.approx(0.5)
    assert failure.failure_rate == 1
    assert "provider unavailable" in failure.notes
