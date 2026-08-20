from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


class StubResearcher(ResearcherAgent):
    def __init__(self) -> None:
        pass

    def run(self, state: ResearchState) -> ResearchState:
        state.sources = [SourceDocument(title="Source", snippet="Evidence")]
        state.research_notes = "Research [1]"
        return state


class StubAnalyst(AnalystAgent):
    def __init__(self) -> None:
        pass

    def run(self, state: ResearchState) -> ResearchState:
        state.analysis_notes = "Analysis [1]"
        return state


class StubWriter(WriterAgent):
    def __init__(self) -> None:
        pass

    def run(self, state: ResearchState) -> ResearchState:
        state.final_answer = "Final answer [1]"
        return state


class FlakyResearcher(StubResearcher):
    def __init__(self) -> None:
        self.calls = 0

    def run(self, state: ResearchState) -> ResearchState:
        self.calls += 1
        if self.calls == 1:
            raise AgentExecutionError("Temporary search failure")
        return super().run(state)


def test_langgraph_workflow_runs_all_workers_to_completion() -> None:
    settings = Settings(max_iterations=6, max_retries=0)
    workflow = MultiAgentWorkflow(
        researcher=StubResearcher(),
        analyst=StubAnalyst(),
        writer=StubWriter(),
        settings=settings,
    )
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))

    result = workflow.run(state)

    assert result.final_answer == "Final answer [1]"
    assert result.route_history == ["researcher", "analyst", "writer", "done"]
    assert [event.payload["route"] for event in result.trace] == [
        "researcher",
        "analyst",
        "writer",
        "done",
    ]


def test_workflow_retries_a_failed_worker() -> None:
    researcher = FlakyResearcher()
    settings = Settings(max_iterations=6, max_retries=1, retry_backoff_seconds=0)
    workflow = MultiAgentWorkflow(
        researcher=researcher,
        analyst=StubAnalyst(),
        writer=StubWriter(),
        settings=settings,
    )

    result = workflow.run(
        ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    )

    assert result.final_answer == "Final answer [1]"
    assert researcher.calls == 2
