"""Markdown benchmark report rendering."""

from datetime import UTC, datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def _display(value: float | None, pattern: str) -> str:
    return "N/A" if value is None else format(value, pattern)


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render aggregate metrics with methodology and explicit limitations."""

    lines = [
        "# Benchmark Report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Results",
        "",
        "| Variant | Attempts | Avg latency (s) | Avg tokens | Avg cost (USD) | "
        "Quality | Citation coverage | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        lines.append(
            f"| {_escape(item.run_name)} | {item.sample_count} | "
            f"{item.latency_seconds:.3f} | {_display(item.total_tokens, '.1f')} | "
            f"{_display(item.estimated_cost_usd, '.6f')} | "
            f"{_display(item.quality_score, '.2f')} | "
            f"{_display_percent(item.citation_coverage)} | "
            f"{_display_percent(item.failure_rate)} | {_escape(item.notes)} |"
        )

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            "- Every variant receives the same query set and repeat count.",
            "- Latency includes the complete runner call and failed attempts.",
            "- Token and cost values are averages over successful attempts.",
            "- Citation coverage is the share of substantive sentences containing `[n]`.",
            "- Quality is `N/A` unless a human or automated rubric scorer is supplied.",
            "- Cost is `N/A` unless current provider input/output prices are configured.",
            "",
            "## Interpretation",
            "",
            _interpret(metrics),
            "",
            "## Failure modes and follow-up",
            "",
            "Inspect failed trace events and representative outputs before choosing a variant. "
            "Better reliability or citation coverage may justify higher latency and cost.",
            "",
        ]
    )
    return "\n".join(lines)


def _display_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _interpret(metrics: list[BenchmarkMetrics]) -> str:
    if not metrics:
        return "No benchmark results were supplied."
    fastest = min(metrics, key=lambda item: item.latency_seconds)
    reliable = min(metrics, key=lambda item: item.failure_rate or 0)
    return (
        f"`{fastest.run_name}` had the lowest average latency. "
        f"`{reliable.run_name}` had the lowest observed failure rate. "
        "Review quality and citation evidence before drawing a final conclusion."
    )
