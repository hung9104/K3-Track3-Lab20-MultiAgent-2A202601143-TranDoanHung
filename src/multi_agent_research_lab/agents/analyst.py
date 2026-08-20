"""Analyst agent implementation."""

from time import perf_counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, TokenUsage
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turn research notes into evaluated, structured insights."""

    name = AgentName.ANALYST.value

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.sources or not state.research_notes:
            raise ValidationError("Analyst requires sources and research_notes")
        started = perf_counter()
        response = self.llm_client.complete(
            "You are the analyst. Compare claims, assess source support, identify conflicts and "
            "knowledge gaps. Preserve [source number] references. Do not introduce new facts.",
            f"Question: {state.request.query}\n\nResearch notes:\n{state.research_notes}",
        )
        state.analysis_notes = response.content
        state.add_agent_result(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                latency_seconds=perf_counter() - started,
                token_usage=TokenUsage(
                    prompt_tokens=response.input_tokens,
                    completion_tokens=response.output_tokens,
                ),
            )
        )
        state.add_trace_event("analysis_completed", {}, agent=AgentName.ANALYST)
        return state
