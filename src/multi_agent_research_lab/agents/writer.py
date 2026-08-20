"""Writer agent implementation."""

import re
from time import perf_counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, TokenUsage
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

CITATION_PATTERN = re.compile(r"\[(\d+)]")


class WriterAgent(BaseAgent):
    """Produce a final answer grounded in the analyzed sources."""

    name = AgentName.WRITER.value

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.sources or not state.research_notes or not state.analysis_notes:
            raise ValidationError("Writer requires sources, research_notes, and analysis_notes")
        started = perf_counter()
        source_list = "\n".join(
            f"[{index}] {source.title} — {source.url or 'URL unavailable'}"
            for index, source in enumerate(state.sources, start=1)
        )
        response = self.llm_client.complete(
            "You are the writer. Answer clearly for the requested audience using only the "
            "analysis. Cite factual claims with [source number], then include the supplied "
            "Sources list unchanged.",
            f"Question: {state.request.query}\nAudience: {state.request.audience}\n\n"
            f"Analysis:\n{state.analysis_notes}\n\nSources:\n{source_list}",
        )
        citations = [int(value) for value in CITATION_PATTERN.findall(response.content)]
        if not citations:
            raise ValidationError("Writer output must contain at least one [source number]")
        if any(citation < 1 or citation > len(state.sources) for citation in citations):
            raise ValidationError("Writer output contains a citation to an unknown source")
        state.final_answer = response.content
        state.add_agent_result(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                latency_seconds=perf_counter() - started,
                token_usage=TokenUsage(
                    prompt_tokens=response.input_tokens,
                    completion_tokens=response.output_tokens,
                ),
                metadata={"citation_count": len(citations)},
            )
        )
        state.add_trace_event(
            "writing_completed", {"citation_count": len(citations)}, agent=AgentName.WRITER
        )
        return state
