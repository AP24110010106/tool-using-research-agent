from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yaml
from rich.console import Console
from rich.table import Table


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from agent.loop import run_agent


console = Console()

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "eval"
    / "questions.yaml"
)

REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "eval"
    / "reports"
)


def load_questions() -> list[dict]:
    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    return data["questions"]


def hostname(url: str) -> str:
    try:
        return (
            urlparse(url)
            .hostname
            or ""
        ).lower()

    except Exception:
        return ""


def domain_matches(
    url: str,
    expected_domains: list[str],
) -> bool:

    host = hostname(url)

    for domain in expected_domains:
        domain = domain.lower()

        if (
            host == domain
            or host.endswith(
                "." + domain
            )
        ):
            return True

    return False


def check_url(
    url: str,
) -> bool:
    """
    Verify that a cited URL is reachable.

    GET is used instead of HEAD because some legitimate websites
    reject HEAD requests.
    """

    try:
        with httpx.Client(
            timeout=10.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(compatible; ResearchAgentEvaluator/1.0)"
                )
            },
        ) as client:
            response = client.get(url)

        return (
            200
            <= response.status_code
            < 400
        )

    except httpx.HTTPError:
        return False


def evaluate_result(
    question: dict,
    result: dict,
) -> dict:

    source_ids = result.get(
        "source_ids",
        [],
    )

    sources = result.get(
        "sources",
        {},
    )

    answer = (
        result.get("answer")
        or ""
    )

    cited_urls: list[str] = []

    citations_valid = True

    for source_id in source_ids:
        source = sources.get(
            source_id
        )

        if source is None:
            citations_valid = False
            continue

        marker = f"[{source_id}]"

        if marker not in answer:
            citations_valid = False

        cited_urls.append(
            source.url
        )

    if not source_ids:
        citations_valid = False

    authoritative_source = any(
        domain_matches(
            url,
            question[
                "authoritative_domains"
            ],
        )
        for url in cited_urls
    )

    reachable_results = {
        url: check_url(url)
        for url in cited_urls
    }

    urls_reachable = (
        bool(reachable_results)
        and all(
            reachable_results.values()
        )
    )

    self_stopped = (
        result.get("termination")
        == "finish"
    )

    return {
        "id": question["id"],
        "category": question[
            "category"
        ],
        "question": question[
            "question"
        ],
        "success": bool(
            result.get("success")
        ),
        "self_stopped": self_stopped,
        "citations_valid": (
            citations_valid
        ),
        "authoritative_source": (
            authoritative_source
        ),
        "urls_reachable": (
            urls_reachable
        ),
        "iterations": result.get(
            "iterations",
            0,
        ),
        "tool_calls": result.get(
            "tool_calls",
            0,
        ),
        "total_tokens": result.get(
            "total_tokens",
            0,
        ),
        "source_count": len(
            source_ids
        ),
        "termination": result.get(
            "termination",
            "unknown",
        ),
        "cited_urls": cited_urls,
        "url_checks": (
            reachable_results
        ),
        "run_id": result.get(
            "run_id"
        ),
    }


def percentage(
    count: int,
    total: int,
) -> float:

    if total == 0:
        return 0.0

    return round(
        (count / total) * 100,
        1,
    )


def main() -> None:
    questions = load_questions()

    results: list[dict] = []

    console.print(
        "\n[bold cyan]"
        "TOOL-USING RESEARCH AGENT — EVALUATION"
        "[/bold cyan]\n"
    )

    for index, question in enumerate(
        questions,
        start=1,
    ):
        console.rule(
            f"[bold]"
            f"{question['id']} "
            f"({index}/{len(questions)})"
            f"[/bold]"
        )

        console.print(
            question["question"]
        )

        console.print(
            "[dim]Running agent...[/dim]"
        )

        agent_result = run_agent(
            question["question"]
        )

        evaluation = evaluate_result(
            question,
            agent_result,
        )

        results.append(
            evaluation
        )

        status = (
            "[green]PASS[/green]"
            if (
                evaluation["success"]
                and evaluation[
                    "citations_valid"
                ]
                and evaluation[
                    "urls_reachable"
                ]
            )
            else "[red]CHECK[/red]"
        )

        console.print(
            f"\nResult: {status}"
        )

        console.print(
            "Termination:",
            evaluation[
                "termination"
            ],
        )

        console.print(
            "Tokens:",
            f"{evaluation['total_tokens']:,}",
        )

    total = len(results)

    successful = sum(
        item["success"]
        for item in results
    )

    self_stopped = sum(
        item["self_stopped"]
        for item in results
    )

    citation_valid = sum(
        item["citations_valid"]
        for item in results
    )

    authoritative = sum(
        item["authoritative_source"]
        for item in results
    )

    reachable = sum(
        item["urls_reachable"]
        for item in results
    )

    average_iterations = round(
        sum(
            item["iterations"]
            for item in results
        )
        / total,
        2,
    )

    average_tool_calls = round(
        sum(
            item["tool_calls"]
            for item in results
        )
        / total,
        2,
    )

    average_tokens = round(
        sum(
            item["total_tokens"]
            for item in results
        )
        / total,
        2,
    )

    summary = {
        "questions": total,
        "successful_runs": successful,
        "success_rate": percentage(
            successful,
            total,
        ),
        "self_stop_rate": percentage(
            self_stopped,
            total,
        ),
        "valid_citation_rate": percentage(
            citation_valid,
            total,
        ),
        "authoritative_source_rate": percentage(
            authoritative,
            total,
        ),
        "reachable_url_rate": percentage(
            reachable,
            total,
        ),
        "average_iterations": (
            average_iterations
        ),
        "average_tool_calls": (
            average_tool_calls
        ),
        "average_tokens": (
            average_tokens
        ),
    }

    table = Table(
        title="Evaluation Summary",
    )

    table.add_column(
        "Metric"
    )

    table.add_column(
        "Result",
        justify="right",
    )

    table.add_row(
        "Successful runs",
        f"{successful}/{total}",
    )

    table.add_row(
        "Success rate",
        f"{summary['success_rate']}%",
    )

    table.add_row(
        "Self-stop rate",
        f"{summary['self_stop_rate']}%",
    )

    table.add_row(
        "Valid citation rate",
        (
            f"{summary['valid_citation_rate']}%"
        ),
    )

    table.add_row(
        "Authoritative source rate",
        (
            f"{summary['authoritative_source_rate']}%"
        ),
    )

    table.add_row(
        "Reachable URL rate",
        (
            f"{summary['reachable_url_rate']}%"
        ),
    )

    table.add_row(
        "Average iterations",
        str(average_iterations),
    )

    table.add_row(
        "Average research calls",
        str(average_tool_calls),
    )

    table.add_row(
        "Average tokens",
        f"{average_tokens:,.0f}",
    )

    console.print()
    console.print(table)

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    report_path = (
        REPORT_DIRECTORY
        / f"evaluation-{timestamp}.json"
    )

    report = {
        "generated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "summary": summary,
        "results": results,
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    console.print(
        f"\n[green]Evaluation report saved:[/green] "
        f"{report_path}"
    )


if __name__ == "__main__":
    main()