"""LangGraph orchestration for the research workflow."""

from collections.abc import Callable
from typing import Any, TypedDict

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, LabError
from multi_agent_research_lab.core.schemas import AgentName, WorkflowRoute
from multi_agent_research_lab.core.state import ResearchState


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
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(self.settings)
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()

    def _supervisor_node(self, envelope: WorkflowEnvelope) -> WorkflowEnvelope:
        return {"state": self.supervisor.run(envelope["state"])}

    def _worker_node(self, agent: BaseAgent) -> Callable[[WorkflowEnvelope], WorkflowEnvelope]:
        def run(envelope: WorkflowEnvelope) -> WorkflowEnvelope:
            state = envelope["state"]
            for attempt in range(1, self.settings.max_retries + 2):
                try:
                    return {"state": agent.run(state)}
                except LabError as exc:
                    if attempt > self.settings.max_retries:
                        state.add_error(
                            str(exc),
                            agent=AgentName(agent.name),
                            retryable=False,
                            attempt=attempt,
                        )
                        state.add_trace_event(
                            "agent_failed",
                            {"error": str(exc), "attempt": attempt},
                            agent=AgentName(agent.name),
                        )
                        return {"state": state}
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
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("researcher", self._worker_node(self.researcher))
        graph.add_node("analyst", self._worker_node(self.analyst))
        graph.add_node("writer", self._worker_node(self.writer))
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._next_route,
            {
                WorkflowRoute.RESEARCHER.value: "researcher",
                WorkflowRoute.ANALYST.value: "analyst",
                WorkflowRoute.WRITER.value: "writer",
                WorkflowRoute.DONE.value: END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
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
        return final_state
