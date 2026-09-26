from __future__ import annotations

from urllib.parse import urlparse

import httpx
import trafilatura

from agent.config import settings
from agent.schemas import FetchResponse


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; ToolUsingResearchAgent/1.0)"
    )
}


def _valid_web_url(url: str) -> bool:
    try:
        parsed = urlparse(url)

        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.netloc)
        )

    except Exception:
        return False


def fetch(url: str) -> FetchResponse:
    """
    Fetch an HTML webpage and extract its main readable content.

    HTTP, parsing and unsupported-content failures are converted into
    structured observations instead of crashing the agent.
    """

    url = url.strip()

    if not _valid_web_url(url):
        return FetchResponse(
            success=False,
            url=url,
            error="Invalid HTTP/HTTPS URL.",
        )

    try:
        with httpx.Client(
            timeout=settings.fetch_timeout_seconds,
            follow_redirects=True,
            headers=HEADERS,
        ) as client:
            response = client.get(url)

        response.raise_for_status()

    except httpx.TimeoutException:
        return FetchResponse(
            success=False,
            url=url,
            error="Fetch request timed out.",
        )

    except httpx.HTTPStatusError as exc:
        return FetchResponse(
            success=False,
            url=str(exc.response.url),
            error=(
                "Fetch returned HTTP "
                f"{exc.response.status_code}."
            ),
        )

    except httpx.RequestError as exc:
        return FetchResponse(
            success=False,
            url=url,
            error=f"Fetch network error: {exc}",
        )

    final_url = str(response.url)

    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()

    if (
        "text/html" not in content_type
        and "application/xhtml+xml" not in content_type
    ):
        return FetchResponse(
            success=False,
            url=final_url,
            error=(
                "Unsupported content type: "
                f"{content_type or 'unknown'}"
            ),
        )

    try:
        content = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )

    except Exception as exc:
        return FetchResponse(
            success=False,
            url=final_url,
            error=f"Content extraction failed: {exc}",
        )

    if not content or len(content.strip()) < 100:
        return FetchResponse(
            success=False,
            url=final_url,
            error=(
                "No useful readable content could be extracted."
            ),
        )

    content = content.strip()

    if len(content) > settings.max_fetched_chars:
        content = content[: settings.max_fetched_chars]

    title: str | None = None

    try:
        metadata = trafilatura.extract_metadata(
            response.text
        )

        if metadata and metadata.title:
            title = metadata.title.strip()

    except Exception:
        pass

    return FetchResponse(
        success=True,
        url=final_url,
        title=title,
        content=content,
    )