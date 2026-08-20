"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery, TokenUsage
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark_suite
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import export_trace
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def run_baseline(request: ResearchQuery, llm_client: LLMClient | None = None) -> ResearchState:
    """Run one model call that performs the research task without delegation."""

    client = llm_client or LLMClient()
    state = ResearchState(request=request)
    response = client.complete(
        system_prompt=(
            "You are a careful research assistant. Answer the user's request directly, "
            "distinguish facts from uncertainty, and do not invent sources or citations."
        ),
        user_prompt=(
            f"Research request: {request.query}\n"
            f"Audience: {request.audience}\n"
            f"Use no more than {request.max_sources} sources if sources are available."
        ),
    )
    state.final_answer = response.content
    state.add_agent_result(
        AgentResult(
            agent=AgentName.BASELINE,
            content=response.content,
            latency_seconds=response.latency_seconds,
            token_usage=TokenUsage(
                prompt_tokens=response.input_tokens,
                completion_tokens=response.output_tokens,
            ),
            metadata={"model": client.settings.openai_model, "cost_usd": response.cost_usd},
        )
    )
    state.add_trace_event(
        "baseline_completed",
        {
            "latency_seconds": response.latency_seconds,
            "total_tokens": state.token_usage.total_tokens,
        },
        agent=AgentName.BASELINE,
    )
    return state


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline."""

    _init()
    request = _parse_query(query)
    try:
        state = run_baseline(request)
    except LabError as exc:
        console.print(Panel.fit(str(exc), title="Baseline Error", style="red"))
        raise typer.Exit(code=2) from exc

    result = state.agent_results[-1]
    subtitle = (
        f"{result.latency_seconds:.2f}s | {result.token_usage.total_tokens} tokens"
        if result.latency_seconds is not None
        else f"{result.token_usage.total_tokens} tokens"
    )
    console.print(
        Panel.fit(state.final_answer or "", title="Single-Agent Baseline", subtitle=subtitle)
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    trace_output: Annotated[
        Path | None,
        typer.Option("--trace-output", help="Optional path for a redacted JSON trace"),
    ] = None,
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except LabError as exc:
        console.print(Panel.fit(str(exc), title="Workflow Error", style="red"))
        raise typer.Exit(code=2) from exc
    if trace_output is not None:
        saved_path = export_trace(result, trace_output)
        console.print(f"Trace saved to: {saved_path}")
    console.print(result.model_dump_json(indent=2))


@app.command("benchmark")
def benchmark_command(
    config_path: Annotated[
        Path, typer.Option("--config", help="YAML file containing benchmark.queries")
    ] = Path("configs/lab_default.yaml"),
    repeats: Annotated[int, typer.Option("--repeats", min=1)] = 3,
    output: Annotated[Path, typer.Option("--output", help="Markdown report path")] = Path(
        "reports/benchmark_report.md"
    ),
) -> None:
    """Benchmark baseline and multi-agent variants on the same query set."""

    _init()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        queries = config["benchmark"]["queries"]
        if not isinstance(queries, list) or not queries or not all(
            isinstance(query, str) and query.strip() for query in queries
        ):
            raise ValueError("benchmark.queries must be a non-empty list of strings")
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        console.print(Panel.fit(str(exc), title="Benchmark Config Error", style="red"))
        raise typer.Exit(code=1) from exc

    settings = get_settings()
    workflow = MultiAgentWorkflow(settings=settings)
    metrics = run_benchmark_suite(
        {
            "single-agent": lambda query: run_baseline(ResearchQuery(query=query)),
            "multi-agent": lambda query: workflow.run(
                ResearchState(request=ResearchQuery(query=query))
            ),
        },
        queries,
        repeats=repeats,
        input_cost_per_million=settings.openai_input_cost_per_million,
        output_cost_per_million=settings.openai_output_cost_per_million,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown_report(metrics), encoding="utf-8")
    console.print(f"Benchmark report saved to: {output}")


if __name__ == "__main__":
    app()
