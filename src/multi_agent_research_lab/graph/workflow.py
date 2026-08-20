"""LangGraph orchestration for the research workflow."""

from collections.abc import Callable
from time import sleep
from typing import Any, TypedDict, cast

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, LabError
from multi_agent_research_lab.core.schemas import AgentName, TraceStatus, WorkflowRoute
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.logging import redact
from multi_agent_research_lab.observability.tracing import trace_span


class WorkflowEnvelope(TypedDict):
    state: ResearchState


class MultiAgentWorkflow:
    """Build and execute Supervisor → workers → Supervisor using LangGraph."""

    def __init__(
        self,
        *,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(self.settings)
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self.critic = critic or CriticAgent()

    def _supervisor_node(self, envelope: WorkflowEnvelope) -> WorkflowEnvelope:
        return {"state": self.supervisor.run(envelope["state"])}

    def _worker_node(self, agent: BaseAgent) -> Callable[[WorkflowEnvelope], WorkflowEnvelope]:
        def run(envelope: WorkflowEnvelope) -> WorkflowEnvelope:
            state = envelope["state"]
            for attempt in range(1, self.settings.max_retries + 2):
                try:
                    with trace_span(
                        "agent.run",
                        {"agent": agent.name, "attempt": attempt, "run_id": state.run_id},
                    ) as span:
                        result = agent.run(state)
                    state.add_trace_event(
                        "agent_attempt_completed",
                        {"attempt": attempt},
                        agent=AgentName(agent.name),
                        duration_seconds=span.duration_seconds,
                    )
                    return {"state": result}
                except LabError as exc:
                    safe_error = str(redact(str(exc)))
                    state.add_trace_event(
                        "agent_attempt_failed",
                        {"error": safe_error, "attempt": attempt},
                        agent=AgentName(agent.name),
                        duration_seconds=span.duration_seconds,
                        status=TraceStatus.ERROR,
                    )
                    if attempt > self.settings.max_retries:
                        state.add_error(
                            safe_error,
                            agent=AgentName(agent.name),
                            retryable=False,
                            attempt=attempt,
                        )
                        return {"state": state}
                    sleep(self.settings.retry_backoff_seconds * (2 ** (attempt - 1)))
            return {"state": state}

        return run

    @staticmethod
    def _next_route(envelope: WorkflowEnvelope) -> str:
        return envelope["state"].route_history[-1]

    def build(self) -> Any:
        """Compile and return the LangGraph application."""

        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise LabError('LangGraph is not installed; run: pip install -e ".[llm]"') from exc

        graph = StateGraph(WorkflowEnvelope)
        graph.add_node("supervisor", cast(Any, self._supervisor_node))
        graph.add_node("researcher", cast(Any, self._worker_node(self.researcher)))
        graph.add_node("analyst", cast(Any, self._worker_node(self.analyst)))
        graph.add_node("writer", cast(Any, self._worker_node(self.writer)))
        graph.add_node("critic", cast(Any, self._worker_node(self.critic)))
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._next_route,
            {
                WorkflowRoute.RESEARCHER.value: "researcher",
                WorkflowRoute.ANALYST.value: "analyst",
                WorkflowRoute.WRITER.value: "writer",
                WorkflowRoute.CRITIC.value: "critic",
                WorkflowRoute.DONE.value: END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        graph.add_edge("critic", "supervisor")
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        try:
            with trace_span(
                "workflow.run", {"run_id": state.run_id, "query": state.request.query}
            ) as span:
                result = self.build().invoke(
                    {"state": state},
                    config={"recursion_limit": self.settings.max_iterations * 2 + 4},
                )
                final_state = result["state"]
                if not isinstance(final_state, ResearchState):
                    final_state = ResearchState.model_validate(final_state)
                if final_state.errors:
                    raise AgentExecutionError(final_state.errors[-1].message)
                if not final_state.final_answer:
                    raise AgentExecutionError("Workflow ended without a final answer")
        except LabError:
            raise
        except Exception as exc:
            raise AgentExecutionError(f"Workflow execution failed: {redact(str(exc))}") from exc
        final_state.add_trace_event(
            "workflow_completed",
            {
                "routes": len(final_state.route_history),
                "total_tokens": final_state.token_usage.total_tokens,
            },
            duration_seconds=span.duration_seconds,
        )
        return final_state
