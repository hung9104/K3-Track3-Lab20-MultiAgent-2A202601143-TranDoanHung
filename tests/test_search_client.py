import json
from types import TracebackType
from typing import Self

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.services.search_client import SearchClient


class FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_search_client_normalizes_tavily_results() -> None:
    payload = {
        "results": [
            {
                "title": "Source title",
                "url": "https://example.com/source",
                "content": "Relevant evidence",
                "score": 0.9,
            }
        ]
    }

    def opener(request: object, timeout: int) -> FakeHttpResponse:
        del request, timeout
        return FakeHttpResponse(payload)

    client = SearchClient(Settings(tavily_api_key="test-key"), opener=opener)
    results = client.search("multi-agent systems", max_results=1)

    assert len(results) == 1
    assert results[0].title == "Source title"
    assert results[0].metadata["provider"] == "tavily"
