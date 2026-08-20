from collections.abc import Iterator

import pytest

from multi_agent_research_lab.agents import AnalystAgent, ResearcherAgent, WriterAgent
from multi_agent_research_lab.core.errors import AgentExecutionError, ValidationError
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import MockSearchClient


class FakeLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses: Iterator[str] = iter(responses)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        del system_prompt, user_prompt
        return LLMResponse(
            content=next(self._responses),
            input_tokens=10,
            output_tokens=5,
            latency_seconds=0.01,
        )


def source(title: str, url: str) -> SourceDocument:
    return SourceDocument(title=title, url=url, snippet=f"Evidence from {title}")


def test_workers_handoff_state_end_to_end() -> None:
    duplicate_url = "https://example.com/one"
    search = MockSearchClient(
        [
            source("One", duplicate_url),
            source("Duplicate", duplicate_url),
            source("Two", "https://example.com/two"),
        ]
    )
    llm = FakeLLMClient(
        [
            "Claim from the sources [1].",
            "The strongest supported claim is backed by source [1].",
            "Final grounded answer [1].\n\nSources\n[1] One\n[2] Two",
        ]
    )
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))

    ResearcherAgent(search, llm).run(state)
    AnalystAgent(llm).run(state)
    WriterAgent(llm).run(state)

    assert len(state.sources) == 2
    assert state.research_notes
    assert state.analysis_notes
    assert state.final_answer
    assert [result.agent for result in state.agent_results] == [
        AgentName.RESEARCHER,
        AgentName.ANALYST,
        AgentName.WRITER,
    ]
    assert state.token_usage.total_tokens == 45


def test_researcher_rejects_empty_search_results() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    with pytest.raises(AgentExecutionError, match="no usable sources"):
        ResearcherAgent(MockSearchClient([]), FakeLLMClient([])).run(state)


def test_analyst_requires_research_handoff() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    with pytest.raises(ValidationError, match="requires sources"):
        AnalystAgent(FakeLLMClient([])).run(state)


def test_writer_rejects_unknown_citation() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[source("One", "https://example.com/one")],
        research_notes="Evidence [1]",
        analysis_notes="Analysis [1]",
    )
    with pytest.raises(ValidationError, match="unknown source"):
        WriterAgent(FakeLLMClient(["Unsupported citation [2]"])).run(state)
