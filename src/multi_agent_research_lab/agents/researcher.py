"""Researcher agent implementation."""

from time import perf_counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument, TokenUsage
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collect sources and create concise, source-indexed research notes."""

    name = AgentName.RESEARCHER.value

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    @staticmethod
    def _deduplicate(sources: list[SourceDocument]) -> list[SourceDocument]:
        unique: list[SourceDocument] = []
        seen: set[str] = set()
        for source in sources:
            key = (source.url or source.title).casefold()
            if key not in seen:
                seen.add(key)
                unique.append(source)
        return unique

    def run(self, state: ResearchState) -> ResearchState:
        started = perf_counter()
        sources = self._deduplicate(
            self.search_client.search(state.request.query, state.request.max_sources)
        )
        if not sources:
            raise AgentExecutionError("Researcher found no usable sources")
        source_text = "\n\n".join(
            f"[{index}] {source.title}\nURL: {source.url or 'Unavailable'}\n{source.snippet}"
            for index, source in enumerate(sources, start=1)
        )
        response = self.llm_client.complete(
            "You are the researcher. Extract claims only from the supplied sources. "
            "Attach a [source number] to every factual claim and flag conflicting evidence.",
            f"Question: {state.request.query}\n\nSources:\n{source_text}",
        )
        state.sources = sources
        state.research_notes = response.content
        state.add_agent_result(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                latency_seconds=perf_counter() - started,
                token_usage=TokenUsage(
                    prompt_tokens=response.input_tokens,
                    completion_tokens=response.output_tokens,
                ),
                metadata={"source_count": len(sources)},
            )
        )
        state.add_trace_event(
            "research_completed", {"source_count": len(sources)}, agent=AgentName.RESEARCHER
        )
        return state
