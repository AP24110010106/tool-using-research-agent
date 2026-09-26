from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class RunLogger:
    """
    Append-only structured logger for a single research run.

    Secrets and raw fetched webpage contents should never be logged.
    """

    def __init__(
        self,
        question: str,
        log_directory: str = "runs",
    ) -> None:

        directory = Path(log_directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")

        run_suffix = uuid4().hex[:8]

        self.run_id = (
            f"{timestamp}-{run_suffix}"
        )

        self.path = directory / (
            f"{self.run_id}.jsonl"
        )

        self.log(
            "run_started",
            {
                "question": question,
            },
        )

    def log(
        self,
        event: str,
        data: dict[str, Any] | None = None,
    ) -> None:

        record = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "run_id": self.run_id,
            "event": event,
            "data": data or {},
        }

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    def finish(
        self,
        *,
        success: bool,
        termination: str,
        iterations: int,
        tool_calls: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:

        self.log(
            "run_finished",
            {
                "success": success,
                "termination": termination,
                "iterations": iterations,
                "tool_calls": tool_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": (
                    input_tokens
                    + output_tokens
                ),
            },
        )