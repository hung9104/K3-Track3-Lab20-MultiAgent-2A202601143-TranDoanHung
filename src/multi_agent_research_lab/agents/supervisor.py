"""Deterministic supervisor for state-based routing."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import AgentName, WorkflowRoute
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Choose the next worker from validated state, without another LLM call."""

    name = AgentName.SUPERVISOR.value

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def route(self, state: ResearchState) -> WorkflowRoute:
        if state.errors:
            return WorkflowRoute.DONE
        if state.critic_review is not None and (
            state.critic_review.approved or state.revision_count >= self.settings.max_revisions
        ):
            return WorkflowRoute.DONE
        if state.iteration >= self.settings.max_iterations:
            state.add_error("Workflow reached max_iterations", agent=AgentName.SUPERVISOR)
            return WorkflowRoute.DONE
        if not state.sources or not state.research_notes:
            return WorkflowRoute.RESEARCHER
        if not state.analysis_notes:
            return WorkflowRoute.ANALYST
        if not state.final_answer:
            return WorkflowRoute.WRITER
        if state.critic_review is None:
            return WorkflowRoute.CRITIC
        state.revision_count += 1
        return WorkflowRoute.WRITER

    def run(self, state: ResearchState) -> ResearchState:
        route = self.route(state)
        state.record_route(route)
        state.add_trace_event(
            "route_selected",
            {"route": route.value, "iteration": state.iteration},
            agent=AgentName.SUPERVISOR,
        )
        return state
