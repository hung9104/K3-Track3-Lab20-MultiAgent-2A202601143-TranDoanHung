"""Repeatable benchmark utilities for single-agent and multi-agent runners."""

import re
from collections.abc import Callable, Mapping, Sequence
from statistics import fmean
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]
QualityScorer = Callable[[str, ResearchState], float]
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def citation_coverage(answer: str | None) -> float | None:
    """Estimate the share of substantive answer sentences carrying a numeric citation."""

    if not answer:
        return None
    answer_body = answer.split("\nSources", maxsplit=1)[0]
    claims = [part.strip() for part in SENTENCE.split(answer_body) if len(part.strip()) >= 20]
    if not claims:
        return None
    cited = sum(bool(re.search(r"\[\d+]", claim)) for claim in claims)
    return cited / len(claims)


def estimate_cost(
    state: ResearchState,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
) -> float | None:
    if input_cost_per_million is None or output_cost_per_million is None:
        return None
    usage = state.token_usage
    return (
        usage.prompt_tokens * input_cost_per_million
        + usage.completion_tokens * output_cost_per_million
    ) / 1_000_000


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    *,
    quality_scorer: QualityScorer | None = None,
    input_cost_per_million: float | None = None,
    output_cost_per_million: float | None = None,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Execute and measure one successful benchmark sample."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    quality = quality_scorer(query, state) if quality_scorer else None
    if quality is None and state.critic_review is not None:
        quality = state.critic_review.quality_score
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        total_tokens=state.token_usage.total_tokens,
        estimated_cost_usd=estimate_cost(
            state, input_cost_per_million, output_cost_per_million
        ),
        quality_score=quality,
        citation_coverage=citation_coverage(state.final_answer),
        failure_rate=0,
    )
    return state, metrics


def run_benchmark_suite(
    runners: Mapping[str, Runner],
    queries: Sequence[str],
    *,
    repeats: int = 3,
    quality_scorer: QualityScorer | None = None,
    input_cost_per_million: float | None = None,
    output_cost_per_million: float | None = None,
) -> list[BenchmarkMetrics]:
    """Run every variant against the same queries and aggregate all attempts."""

    if not queries:
        raise ValueError("queries must not be empty")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    aggregates: list[BenchmarkMetrics] = []
    for name, runner in runners.items():
        samples: list[BenchmarkMetrics] = []
        latencies: list[float] = []
        failures: list[str] = []
        for query in queries:
            for _ in range(repeats):
                started = perf_counter()
                try:
                    _, metrics = run_benchmark(
                        name,
                        query,
                        runner,
                        quality_scorer=quality_scorer,
                        input_cost_per_million=input_cost_per_million,
                        output_cost_per_million=output_cost_per_million,
                    )
                    samples.append(metrics)
                    latencies.append(metrics.latency_seconds)
                except Exception as exc:
                    latencies.append(perf_counter() - started)
                    failures.append(f"{type(exc).__name__}: {exc}")

        attempt_count = len(queries) * repeats
        aggregates.append(
            BenchmarkMetrics(
                run_name=name,
                sample_count=attempt_count,
                latency_seconds=fmean(latencies),
                total_tokens=_mean_optional([sample.total_tokens for sample in samples]),
                estimated_cost_usd=_mean_optional(
                    [sample.estimated_cost_usd for sample in samples]
                ),
                quality_score=_mean_optional([sample.quality_score for sample in samples]),
                citation_coverage=_mean_optional(
                    [sample.citation_coverage for sample in samples]
                ),
                failure_rate=len(failures) / attempt_count,
                notes=_failure_notes(failures),
            )
        )
    return aggregates


def _mean_optional(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return fmean(present) if present else None


def _failure_notes(failures: Sequence[str]) -> str:
    if not failures:
        return "All attempts succeeded"
    unique = list(dict.fromkeys(failures))
    return f"{len(failures)} failed; examples: " + "; ".join(unique[:2])
