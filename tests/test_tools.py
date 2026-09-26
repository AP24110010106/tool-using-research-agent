from agent.tools.fetch import (
    _valid_web_url,
)


def test_http_url_is_valid():
    assert _valid_web_url(
        "http://example.com"
    )


def test_https_url_is_valid():
    assert _valid_web_url(
        "https://example.com/page"
    )


def test_invalid_scheme_is_rejected():
    assert not _valid_web_url(
        "ftp://example.com"
    )


def test_missing_hostname_is_rejected():
    assert not _valid_web_url(
        "https://"
    )


def test_random_text_is_rejected():
    assert not _valid_web_url(
        "hello world"
    )