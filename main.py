from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.config import settings
from agent.loop import run_agent


console = Console()


def show_header() -> None:
    title = Text(
        "TOOL-USING RESEARCH AGENT",
        justify="center",
        style="bold cyan",
    )

    subtitle = Text(
        "\nSearch • Read • Verify • Cite • Stop",
        justify="center",
        style="dim",
    )

    title.append_text(subtitle)

    console.print(
        Panel(
            title,
            border_style="cyan",
            padding=(1, 4),
        )
    )


def show_sources(
    result: dict,
) -> None:

    source_ids = result.get(
        "source_ids",
        [],
    )

    sources = result.get(
        "sources",
        {},
    )

    if not source_ids:
        return

    console.print(
        "\n[bold cyan]Sources[/bold cyan]"
    )

    for source_id in source_ids:
        source = sources.get(
            source_id
        )

        if not source:
            continue

        console.print(
            f"\n[bold][{source_id}] "
            f"{source.title}[/bold]"
        )

        console.print(
            f"[dim]{source.url}[/dim]"
        )


def show_statistics(
    result: dict,
) -> None:

    table = Table(
        title="Run Summary",
        show_header=False,
        border_style="cyan",
    )

    table.add_column(
        "Metric",
        style="bold",
    )

    table.add_column(
        "Value",
    )

    success = result.get(
        "success",
        False,
    )

    status = (
        "[green]✓ Completed[/green]"
        if success
        else
        "[red]✗ Incomplete[/red]"
    )

    table.add_row(
        "Status",
        status,
    )

    table.add_row(
        "Termination",
        str(
            result.get(
                "termination",
                "unknown",
            )
        ),
    )

    table.add_row(
        "Iterations",
        (
            f"{result.get('iterations', 0)}"
            f" / {settings.max_iterations}"
        ),
    )

    table.add_row(
        "Research tool calls",
        (
            f"{result.get('tool_calls', 0)}"
            f" / {settings.max_tool_calls}"
        ),
    )

    table.add_row(
        "Verified sources",
        str(
            len(
                result.get(
                    "source_ids",
                    [],
                )
            )
        ),
    )

    table.add_row(
        "Model errors",
        str(
            result.get(
                "model_errors",
                0,
            )
        ),
    )

    table.add_row(
        "Input tokens",
        f"{result.get('input_tokens', 0):,}",
    )

    table.add_row(
        "Output tokens",
        f"{result.get('output_tokens', 0):,}",
    )

    table.add_row(
        "Total tokens",
        (
            f"{result.get('total_tokens', 0):,}"
            f" / {settings.max_total_tokens:,}"
        ),
    )

    table.add_row(
        "Run ID",
        str(
            result.get(
                "run_id",
                "unknown",
            )
        ),
    )

    console.print()
    console.print(table)


def main() -> None:
    show_header()

    console.print(
        "\n[bold]Research question[/bold]"
    )

    question = console.input(
        "[cyan]> [/cyan]"
    ).strip()

    if not question:
        console.print(
            "\n[red]A research question is required.[/red]"
        )

        return

    console.print(
        "\n[dim]Researching with bounded tools and "
        "verified evidence...[/dim]\n"
    )

    result = run_agent(
        question
    )

    if result["success"]:
        console.print(
            Panel(
                result["answer"],
                title="Verified Answer",
                border_style="green",
                padding=(1, 2),
            )
        )

        show_sources(
            result
        )

    else:
        console.print(
            Panel(
                result.get(
                    "error",
                    "Research did not complete.",
                ),
                title="Research Incomplete",
                border_style="red",
            )
        )

    show_statistics(
        result
    )

    console.print(
        f"\n[dim]Execution log: "
        f"{result.get('log_path', 'unavailable')}[/dim]"
    )


if __name__ == "__main__":
    main()