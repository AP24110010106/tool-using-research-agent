import pytest
from pydantic import ValidationError

from agent.schemas import (
    FetchToolInput,
    FinishToolInput,
    SearchToolInput,
)


def test_search_input_accepts_valid_query():
    item = SearchToolInput(
        query="coral bleaching NOAA"
    )

    assert (
        item.query
        == "coral bleaching NOAA"
    )


def test_search_input_rejects_short_query():
    with pytest.raises(
        ValidationError
    ):
        SearchToolInput(
            query=""
        )


def test_fetch_requires_valid_url():
    item = FetchToolInput(
        url="https://example.com"
    )

    assert (
        str(item.url)
        == "https://example.com/"
    )


def test_fetch_rejects_invalid_url():
    with pytest.raises(
        ValidationError
    ):
        FetchToolInput(
            url="not-a-url"
        )


def test_finish_requires_source():
    with pytest.raises(
        ValidationError
    ):
        FinishToolInput(
            summary="Answer [S1]",
            sources=[],
        )