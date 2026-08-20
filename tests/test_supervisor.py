from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def make_state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_supervisor_routes_from_state_completeness() -> None:
    supervisor = SupervisorAgent(Settings(max_iterations=6))
    state = make_state()

    supervisor.run(state)
    assert state.route_history[-1] == "researcher"

    state.sources = [SourceDocument(title="Source", snippet="Evidence")]
    state.research_notes = "Notes [1]"
    supervisor.run(state)
    assert state.route_history[-1] == "analyst"

    state.analysis_notes = "Analysis [1]"
    supervisor.run(state)
    assert state.route_history[-1] == "writer"

    state.final_answer = "Answer [1]"
    supervisor.run(state)
    assert state.route_history[-1] == "done"


def test_supervisor_stops_at_iteration_limit() -> None:
    supervisor = SupervisorAgent(Settings(max_iterations=1))
    state = make_state()
    state.iteration = 1

    supervisor.run(state)

    assert state.route_history[-1] == "done"
    assert state.errors[-1].message == "Workflow reached max_iterations"
