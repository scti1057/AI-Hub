import logging
import re
from typing import Any

import requests

from ai_hub.config import (
    BRAVE_SEARCH_API_KEY,
    TAVILY_API_KEY,
    WEB_SEARCH_ENABLED,
    WEB_SEARCH_FETCH_PAGES,
    WEB_SEARCH_FETCH_TOP_N,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_PROVIDER,
    WEB_SEARCH_TIMEOUT_SECONDS,
)
from ai_hub.logging_config import log_event, setup_logging


logger = logging.getLogger(__name__)
setup_logging()

USER_AGENT = "AI-Hub/0.1 (+local research worker)"


class WebSearchError(RuntimeError):
    def __init__(self, message: str, code: str = "web_search_error") -> None:
        super().__init__(message)
        self.code = code


class WebSearchClient:
    def __init__(
        self,
        provider: str = WEB_SEARCH_PROVIDER,
        timeout_seconds: int = WEB_SEARCH_TIMEOUT_SECONDS,
        max_results: int = WEB_SEARCH_MAX_RESULTS,
        fetch_pages: bool = WEB_SEARCH_FETCH_PAGES,
        fetch_top_n: int = WEB_SEARCH_FETCH_TOP_N,
        brave_api_key: str = BRAVE_SEARCH_API_KEY,
        tavily_api_key: str = TAVILY_API_KEY,
        enabled: bool = WEB_SEARCH_ENABLED,
        session: requests.Session | None = None,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.fetch_pages = fetch_pages
        self.fetch_top_n = fetch_top_n
        self.brave_api_key = brave_api_key
        self.tavily_api_key = tavily_api_key
        self.enabled = enabled
        self.session = session or requests.Session()

    def is_available(self) -> bool:
        if not self.enabled:
            return False
        provider = self._resolve_provider()
        if provider == "brave":
            return bool(self.brave_api_key)
        if provider == "tavily":
            return bool(self.tavily_api_key)
        return False

    def search(self, query: str, *, max_results: int | None = None) -> dict:
        normalized_query = " ".join(query.strip().split())
        if not normalized_query:
            raise WebSearchError("Search query is empty.", code="web_search_empty_query")
        if not self.enabled:
            raise WebSearchError("Web search is disabled.", code="web_search_disabled")

        provider = self._resolve_provider()
        result_limit = max(1, min(max_results or self.max_results, 10))
        log_event(logger, "web_search_started", provider=provider, query=normalized_query, max_results=result_limit)

        if provider == "brave":
            payload = self._search_brave(normalized_query, result_limit)
        elif provider == "tavily":
            payload = self._search_tavily(normalized_query, result_limit)
        else:
            raise WebSearchError("No supported web search provider is configured.", code="web_search_provider_missing")

        if self.fetch_pages:
            self._attach_page_previews(payload["results"])

        log_event(
            logger,
            "web_search_finished",
            provider=provider,
            query=normalized_query,
            results=len(payload["results"]),
        )
        return payload

    def _resolve_provider(self) -> str:
        provider = (self.provider or "auto").strip().lower()
        if provider == "auto":
            if self.brave_api_key:
                return "brave"
            if self.tavily_api_key:
                return "tavily"
            return "none"
        return provider

    def _search_brave(self, query: str, max_results: int) -> dict:
        if not self.brave_api_key:
            raise WebSearchError("Brave Search API key is missing.", code="web_search_missing_brave_key")
        response = self.session.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "User-Agent": USER_AGENT,
                "X-Subscription-Token": self.brave_api_key,
            },
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise WebSearchError(
                f"Brave Search request failed: {response.status_code}",
                code="web_search_http_error",
            )
        data = response.json()
        results = []
        for item in (data.get("web") or {}).get("results") or []:
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            results.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "url": url,
                    "snippet": str(item.get("description", "")).strip(),
                    "source": "brave",
                }
            )
        return {"provider": "brave", "query": query, "results": results[:max_results]}

    def _search_tavily(self, query: str, max_results: int) -> dict:
        if not self.tavily_api_key:
            raise WebSearchError("Tavily API key is missing.", code="web_search_missing_tavily_key")
        response = self.session.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_raw_content": False,
            },
            headers={
                "User-Agent": USER_AGENT,
                "Authorization": f"Bearer {self.tavily_api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise WebSearchError(
                f"Tavily Search request failed: {response.status_code}",
                code="web_search_http_error",
            )
        data = response.json()
        results = []
        for item in data.get("results") or []:
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            results.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "url": url,
                    "snippet": str(item.get("content", "")).strip(),
                    "source": "tavily",
                }
            )
        return {"provider": "tavily", "query": query, "results": results[:max_results]}

    def _attach_page_previews(self, results: list[dict]) -> None:
        for item in results[: max(0, self.fetch_top_n)]:
            url = item.get("url", "")
            if not url:
                continue
            try:
                item["page_preview"] = self.fetch_page_preview(url)
            except Exception as exc:
                item["page_preview_error"] = str(exc)

    def fetch_page_preview(self, url: str) -> str:
        response = self.session.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            timeout=self.timeout_seconds,
        )
        if not response.ok:
            raise WebSearchError(
                f"Page fetch failed: {response.status_code}",
                code="web_search_page_fetch_error",
            )
        return _extract_text_preview(response.text)


def _extract_text_preview(html: str, limit: int = 1200) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def format_search_context(search_payload: dict[str, Any], max_results: int = 4) -> str:
    results = search_payload.get("results") or []
    if not results:
        return "No web results."
    lines = [
        f"Provider: {search_payload.get('provider', 'unknown')}",
        f"Query: {search_payload.get('query', '')}",
    ]
    for index, item in enumerate(results[:max_results], start=1):
        lines.append(f"[{index}] {item.get('title', '').strip()}")
        lines.append(f"URL: {item.get('url', '').strip()}")
        snippet = item.get("snippet", "").strip()
        if snippet:
            lines.append(f"Snippet: {snippet[:400]}")
        preview = item.get("page_preview", "").strip()
        if preview:
            lines.append(f"Preview: {preview[:500]}")
    return "\n".join(lines)
