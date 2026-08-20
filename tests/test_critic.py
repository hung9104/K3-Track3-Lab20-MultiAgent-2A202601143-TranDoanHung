from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse


class FakeCriticLLM(LLMClient):
    def __init__(self, content: str) -> None:
        self.content = content

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        del system_prompt, user_prompt
        return LLMResponse(content=self.content, input_tokens=20, output_tokens=10)


def review_state(answer: str = "Grounded answer [1].") -> ResearchState:
    return ResearchState(
        request=ResearchQuery(query="Explain multi-agent systems"),
        sources=[SourceDocument(title="Source", snippet="Supporting evidence")],
        research_notes="Research [1]",
        analysis_notes="Analysis [1]",
        final_answer=answer,
    )


def test_critic_records_structured_approval() -> None:
    llm = FakeCriticLLM(
        '{"approved": true, "quality_score": 9, "issues": [], "recommendations": []}'
    )

    state = CriticAgent(llm).run(review_state())

    assert state.critic_review is not None
    assert state.critic_review.approved is True
    assert state.agent_results[-1].agent is AgentName.CRITIC
    assert state.token_usage.total_tokens == 30


def test_critic_overrides_approval_when_citation_is_invalid() -> None:
    llm = FakeCriticLLM(
        '{"approved": true, "quality_score": 9, "issues": [], "recommendations": []}'
    )

    state = CriticAgent(llm).run(review_state("Answer with unknown source [2]."))

    assert state.critic_review is not None
    assert state.critic_review.approved is False
    assert "invalid numeric citations" in state.critic_review.issues[-1]
