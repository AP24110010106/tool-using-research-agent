from agent.scratchpad import Scratchpad
from agent.verifier import EvidenceVerifier


def build_scratchpad():
    pad = Scratchpad()

    pad.add_source(
        title="NOAA",
        url="https://noaa.gov/example",
        findings=[
            "Ocean warming can cause coral bleaching."
        ],
    )

    return pad


def test_valid_citation_structure():
    pad = build_scratchpad()

    verifier = EvidenceVerifier(
        client=None
    )

    result = verifier.validate_citations(
        summary=(
            "Ocean warming can cause "
            "coral bleaching [S1]."
        ),
        declared_sources=[
            "S1"
        ],
        scratchpad=pad,
    )

    assert result.passed is True


def test_unknown_citation_is_rejected():
    pad = build_scratchpad()

    verifier = EvidenceVerifier(
        client=None
    )

    result = verifier.validate_citations(
        summary="Claim [S999].",
        declared_sources=[
            "S999"
        ],
        scratchpad=pad,
    )

    assert result.passed is False


def test_missing_citation_is_rejected():
    pad = build_scratchpad()

    verifier = EvidenceVerifier(
        client=None
    )

    result = verifier.validate_citations(
        summary=(
            "Ocean warming can cause "
            "coral bleaching."
        ),
        declared_sources=[
            "S1"
        ],
        scratchpad=pad,
    )

    assert result.passed is False


def test_declared_but_not_cited_is_rejected():
    pad = build_scratchpad()

    verifier = EvidenceVerifier(
        client=None
    )

    result = verifier.validate_citations(
        summary="Claim [S1].",
        declared_sources=[
            "S1",
            "S2",
        ],
        scratchpad=pad,
    )

    assert result.passed is False


def test_cited_but_not_declared_is_rejected():
    pad = build_scratchpad()

    pad.add_source(
        title="Second Source",
        url="https://example.org",
        findings=[
            "Second finding."
        ],
    )

    verifier = EvidenceVerifier(
        client=None
    )

    result = verifier.validate_citations(
        summary="Claims [S1][S2].",
        declared_sources=[
            "S1"
        ],
        scratchpad=pad,
    )

    assert result.passed is False