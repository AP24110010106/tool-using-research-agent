from __future__ import annotations

import httpx

from agent.config import settings
from agent.schemas import SearchResponse, SearchResult


TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def search(query: str) -> SearchResponse:
    """
    Search the web using Tavily.

    Search results are discovery candidates only. They do not become
    citable evidence until the selected URL is successfully fetched.
    """

    query = query.strip()

    if not query:
        return SearchResponse(
            success=False,
            query=query,
            error="Search query cannot be empty.",
        )

    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": settings.search_max_results,
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        with httpx.Client(
            timeout=settings.search_timeout_seconds,
        ) as client:
            response = client.post(
                TAVILY_SEARCH_URL,
                json=payload,
            )

        response.raise_for_status()
        data = response.json()

    except httpx.TimeoutException:
        return SearchResponse(
            success=False,
            query=query,
            error="Search request timed out.",
        )

    except httpx.HTTPStatusError as exc:
        return SearchResponse(
            success=False,
            query=query,
            error=(
                "Search API returned HTTP "
                f"{exc.response.status_code}."
            ),
        )

    except httpx.RequestError as exc:
        return SearchResponse(
            success=False,
            query=query,
            error=f"Search network error: {exc}",
        )

    except ValueError:
        return SearchResponse(
            success=False,
            query=query,
            error="Search API returned invalid JSON.",
        )

    raw_results = data.get("results", [])

    results: list[SearchResult] = []
    seen_urls: set[str] = set()

    for item in raw_results:
        url = str(item.get("url", "")).strip()

        if not url or url in seen_urls:
            continue

        try:
            result = SearchResult(
                title=str(
                    item.get("title")
                    or "Untitled result"
                ).strip(),
                url=url,
                snippet=str(
                    item.get("content")
                    or ""
                ).strip(),
            )
        except Exception:
            continue

        results.append(result)
        seen_urls.add(url)

    if not results:
        return SearchResponse(
            success=False,
            query=query,
            error="Search returned no valid results.",
        )

    return SearchResponse(
        success=True,
        query=query,
        results=results,
    )