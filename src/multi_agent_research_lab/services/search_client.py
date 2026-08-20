"""Search provider boundary used by the researcher agent."""

import json
from collections.abc import Callable
from typing import Any
from urllib.request import Request, urlopen

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, LabError
from multi_agent_research_lab.core.schemas import SourceDocument

UrlOpener = Callable[..., Any]


class SearchClient:
    """Minimal Tavily HTTP client with normalized search results."""

    endpoint = "https://api.tavily.com/search"

    def __init__(self, settings: Settings | None = None, opener: UrlOpener = urlopen) -> None:
        self.settings = settings or get_settings()
        self._opener = opener

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        if not self.settings.tavily_api_key:
            raise LabError(
                "TAVILY_API_KEY is required for live search; "
                "inject MockSearchClient for offline runs"
            )
        body = json.dumps(
            {
                "api_key": self.settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            }
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.settings.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise AgentExecutionError(f"Search request failed: {exc}") from exc

        results = payload.get("results", [])
        if not isinstance(results, list):
            raise AgentExecutionError("Search provider returned an invalid results payload")
        documents: list[SourceDocument] = []
        for item in results[:max_results]:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            snippet = item.get("content")
            if isinstance(title, str) and isinstance(snippet, str) and title and snippet:
                documents.append(
                    SourceDocument(
                        title=title,
                        url=item.get("url"),
                        snippet=snippet,
                        metadata={"score": item.get("score"), "provider": "tavily"},
                    )
                )
        return documents


class MockSearchClient(SearchClient):
    """Deterministic offline source for tests, demos, and development."""

    def __init__(self, documents: list[SourceDocument]) -> None:
        self.documents = documents

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        del query
        return self.documents[:max_results]
