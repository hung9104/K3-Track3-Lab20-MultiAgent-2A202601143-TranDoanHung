"""Fact-checking and quality-review agent."""

import json
import re
from time import perf_counter

from pydantic import ValidationError as PydanticValidationError

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import (
    AgentName,
    AgentResult,
    CriticReview,
    TokenUsage,
)
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

CITATION_PATTERN = re.compile(r"\[(\d+)]")


class CriticAgent(BaseAgent):
    """Review grounding, citation validity, completeness, and clarity."""

    name = AgentName.CRITIC.value

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.final_answer or not state.sources:
            raise ValidationError("Critic requires final_answer and sources")

        started = perf_counter()
        source_text = "\n\n".join(
            f"[{index}] {source.title}\n{source.snippet}"
            for index, source in enumerate(state.sources, start=1)
        )
        response = self.llm_client.complete(
            "You are a strict research editor. Check whether claims are supported by the "
            "numbered sources, citations point to the right evidence, uncertainty is explicit, "
            "and the answer addresses the request. Return JSON only with keys: approved "
            "(boolean), quality_score (0-10), issues (string array), recommendations "
            "(string array). Approve only when quality_score is at least 8.",
            f"Question: {state.request.query}\n\nAnswer:\n{state.final_answer}\n\n"
            f"Sources:\n{source_text}",
        )
        review = self._parse_review(response.content)
        citations = [int(value) for value in CITATION_PATTERN.findall(state.final_answer)]
        if not citations or any(value < 1 or value > len(state.sources) for value in citations):
            review.approved = False
            review.issues.append("Answer has missing or invalid numeric citations")
        if review.quality_score < 8:
            review.approved = False

        state.critic_review = review
        state.add_agent_result(
            AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                success=review.approved,
                latency_seconds=perf_counter() - started,
                token_usage=TokenUsage(
                    prompt_tokens=response.input_tokens,
                    completion_tokens=response.output_tokens,
                ),
                metadata={
                    "approved": review.approved,
                    "quality_score": review.quality_score,
                    "issue_count": len(review.issues),
                },
            )
        )
        state.add_trace_event(
            "critique_completed",
            {
                "approved": review.approved,
                "quality_score": review.quality_score,
                "issue_count": len(review.issues),
            },
            agent=AgentName.CRITIC,
        )
        return state

    @staticmethod
    def _parse_review(content: str) -> CriticReview:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1])
        try:
            return CriticReview.model_validate(json.loads(cleaned))
        except (json.JSONDecodeError, PydanticValidationError) as exc:
            raise ValidationError(f"Critic returned invalid structured output: {exc}") from exc
