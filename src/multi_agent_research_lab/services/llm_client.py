"""Provider boundary for model completions."""

import time
from dataclasses import dataclass
from typing import Any, Protocol

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, LabError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float | None = None


class ChatCompletionsClient(Protocol):
    """Small SDK-shaped protocol that also makes the provider easy to mock."""

    chat: Any


class LLMClient:
    """OpenAI-backed client with centralized timeout and retry behavior."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: ChatCompletionsClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client

    def _get_client(self) -> ChatCompletionsClient:
        if self._client is not None:
            return self._client
        if not self.settings.openai_api_key:
            raise LabError("OPENAI_API_KEY is required to call the LLM provider")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LabError('OpenAI SDK is not installed; run: pip install -e ".[llm]"') from exc

        self._client = OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.timeout_seconds,
            max_retries=0,
        )
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return one completion, retrying transient provider failures."""

        client = self._get_client()
        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(self.settings.max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )
                content = completion.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("provider returned an empty completion")
                usage = getattr(completion, "usage", None)
                return LLMResponse(
                    content=content.strip(),
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_seconds=time.perf_counter() - started,
                )
            except Exception as exc:  # Provider SDKs expose different exception hierarchies.
                last_error = exc
                if attempt < self.settings.max_retries:
                    time.sleep(self.settings.retry_backoff_seconds * (2**attempt))

        raise AgentExecutionError(
            f"LLM completion failed after {self.settings.max_retries + 1} attempt(s): "
            f"{last_error}"
        ) from last_error
