from agent.scratchpad import Scratchpad


def test_duplicate_search_detection():
    pad = Scratchpad()

    pad.record_query("Coral Bleaching NOAA")

    assert pad.has_searched(
        "coral   bleaching   noaa"
    )


def test_duplicate_url_attempt_detection():
    pad = Scratchpad()

    pad.record_url_attempt(
        "https://example.com/"
    )

    assert pad.has_attempted_url(
        "https://example.com"
    )


def test_source_ids_are_sequential():
    pad = Scratchpad()

    first = pad.add_source(
        title="Source One",
        url="https://example.com/one",
        findings=["Finding one"],
    )

    second = pad.add_source(
        title="Source Two",
        url="https://example.com/two",
        findings=["Finding two"],
    )

    assert first == "S1"
    assert second == "S2"


def test_duplicate_source_url_is_merged():
    pad = Scratchpad()

    first = pad.add_source(
        title="Example",
        url="https://example.com",
        findings=["Finding A"],
    )

    second = pad.add_source(
        title="Example",
        url="https://example.com/",
        findings=["Finding B"],
    )

    assert first == second
    assert pad.source_count == 1

    source = pad.get_source("S1")

    assert source is not None

    assert source.findings == [
        "Finding A",
        "Finding B",
    ]


def test_render_contains_compact_evidence():
    pad = Scratchpad()

    pad.add_source(
        title="NOAA Example",
        url="https://example.com",
        findings=[
            "Compact verified finding."
        ],
    )

    rendered = pad.render()

    assert "[S1]" in rendered
    assert "NOAA Example" in rendered
    assert "Compact verified finding." in rendered