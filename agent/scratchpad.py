from dataclasses import dataclass, field


@dataclass
class Source:
    """
    A successfully fetched source containing compact verified
    research findings.
    """

    source_id: str
    title: str
    url: str

    findings: list[str] = field(
        default_factory=list
    )


class Scratchpad:
    """
    Bounded research memory.

    The scratchpad stores compact findings and provenance rather than
    raw webpages. This prevents fetched documents from continuously
    expanding the model context.
    """

    def __init__(self) -> None:
        self.sources: dict[str, Source] = {}

        self._url_to_source_id: dict[str, str] = {}

        self._executed_queries: set[str] = set()

        self._attempted_urls: set[str] = set()

    # ========================================================
    # SEARCH HISTORY
    # ========================================================

    def has_searched(
        self,
        query: str,
    ) -> bool:
        return (
            self._normalize_query(query)
            in self._executed_queries
        )

    def record_query(
        self,
        query: str,
    ) -> None:
        self._executed_queries.add(
            self._normalize_query(query)
        )

    # ========================================================
    # FETCH HISTORY
    # ========================================================

    def has_attempted_url(
        self,
        url: str,
    ) -> bool:
        return (
            self._normalize_url(url)
            in self._attempted_urls
        )

    def record_url_attempt(
        self,
        url: str,
    ) -> None:
        self._attempted_urls.add(
            self._normalize_url(url)
        )

    # ========================================================
    # VERIFIED SOURCES
    # ========================================================

    def add_source(
        self,
        title: str,
        url: str,
        findings: list[str],
    ) -> str:
        """
        Add successfully extracted evidence.

        Repeated canonical URLs are merged into their existing
        source instead of receiving another source ID.
        """

        normalized_url = self._normalize_url(url)

        existing_id = self._url_to_source_id.get(
            normalized_url
        )

        if existing_id:
            source = self.sources[existing_id]

            for finding in findings:
                clean = finding.strip()

                if (
                    clean
                    and clean not in source.findings
                ):
                    source.findings.append(clean)

            return existing_id

        source_id = (
            f"S{len(self.sources) + 1}"
        )

        clean_findings = [
            finding.strip()
            for finding in findings
            if finding.strip()
        ]

        source = Source(
            source_id=source_id,
            title=title.strip() or "Untitled source",
            url=url.strip(),
            findings=clean_findings,
        )

        self.sources[source_id] = source

        self._url_to_source_id[
            normalized_url
        ] = source_id

        return source_id

    def source_exists(
        self,
        source_id: str,
    ) -> bool:
        return source_id in self.sources

    def get_source(
        self,
        source_id: str,
    ) -> Source | None:
        return self.sources.get(source_id)

    def valid_source_ids(
        self,
    ) -> set[str]:
        return set(self.sources)

    # ========================================================
    # STATISTICS
    # ========================================================

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def search_count(self) -> int:
        return len(self._executed_queries)

    @property
    def fetch_attempt_count(self) -> int:
        return len(self._attempted_urls)

    # ========================================================
    # MODEL VIEW
    # ========================================================

    def render(self) -> str:
        """
        Render only compact evidence for the model.

        Raw webpage content never enters this representation.
        """

        if not self.sources:
            return (
                "No verified evidence has been collected yet."
            )

        sections: list[str] = []

        for source in self.sources.values():
            findings = "\n".join(
                f"- {finding}"
                for finding in source.findings
            )

            sections.append(
                f"[{source.source_id}]\n"
                f"Title: {source.title}\n"
                f"URL: {source.url}\n"
                f"Findings:\n"
                f"{findings}"
            )

        return "\n\n".join(sections)

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_query(
        query: str,
    ) -> str:
        return " ".join(
            query.casefold().strip().split()
        )

    @staticmethod
    def _normalize_url(
        url: str,
    ) -> str:
        return (
            url.strip()
            .rstrip("/")
            .casefold()
        )