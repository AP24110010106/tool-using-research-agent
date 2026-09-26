from __future__ import annotations

import json
from typing import Any

from groq import Groq
from pydantic import ValidationError

from agent.config import settings
from agent.logger import RunLogger
from agent.schemas import (
    FetchToolInput,
    FinishToolInput,
    SearchToolInput,
)
from agent.scratchpad import Scratchpad
from agent.tools.fetch import fetch
from agent.tools.search import search
from agent.verifier import EvidenceVerifier


client = Groq(
    api_key=settings.groq_api_key
)


# ============================================================
# TOOL SCHEMAS
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the web for candidate sources. "
                "Search snippets are discovery information only "
                "and cannot be cited as evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A focused web search query."
                        ),
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": (
                "Fetch and read a candidate webpage. "
                "Only successfully fetched pages can become "
                "verified evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "Complete HTTP or HTTPS URL."
                        ),
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Propose the final cited research answer when "
                "sufficient verified evidence has been collected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "Final answer containing inline "
                            "citations such as [S1]."
                        ),
                    },
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "Exactly the verified source IDs "
                            "cited in the answer."
                        ),
                    },
                },
                "required": [
                    "summary",
                    "sources",
                ],
                "additionalProperties": False,
            },
        },
    },
]


# ============================================================
# AGENT INSTRUCTIONS
# ============================================================

SYSTEM_PROMPT = """
You are an autonomous, evidence-grounded research agent.

Your task is to answer a user's research question by searching,
reading reliable sources, maintaining compact evidence, citing that
evidence, and stopping when sufficient evidence exists.

You operate using:

DECIDE -> ACT -> OBSERVE -> DECIDE


AVAILABLE ACTIONS

search(query)
    Discover candidate webpages.

fetch(url)
    Read a candidate webpage.

finish(summary, sources)
    Propose the final cited answer.


IMPORTANT EVIDENCE RULES

- Search snippets are discovery information only.
- Search snippets are NOT citable evidence.
- A source becomes citable only after fetch succeeds and it appears
  in CURRENT VERIFIED EVIDENCE.
- Never invent a source ID.
- Never invent a URL.
- Never cite information that is absent from verified evidence.
- Material factual claims must be supported by citations.


SOURCE SELECTION

Prefer authoritative and primary sources when reasonably available:

- government agencies
- universities
- peer-reviewed publications
- scientific institutions
- standards organizations
- official documentation
- first-party organizations

Use secondary sources when they materially improve the answer.


STOPPING CONDITION

Research is sufficient when:

1. the user's question can be directly answered;
2. important factual claims are supported by fetched evidence;
3. authoritative evidence has been collected when reasonably
   available;
4. important uncertainty or disagreement has been investigated when
   relevant; and
5. further searching is unlikely to materially improve the answer.

Do not continue searching simply to increase the number of sources.


TOOL RULES

Call exactly one provided tool for each decision.

The only actions are:

search
fetch
finish

Do not invent actions such as:

answer
final
commentary
respond
summarize
complete

If enough evidence exists, use finish.

Do not provide the final answer as ordinary assistant text.


FINISH FORMAT

The summary must contain inline source citations:

[S1]

or:

[S1][S2]

The sources array must contain exactly the source IDs cited in the
summary.
"""


# ============================================================
# TOKEN TRACKER
# ============================================================

class TokenBudget:
    """
    Tracks all LLM usage in the run, including research decisions,
    evidence extraction and final verification.
    """

    def __init__(
        self,
        limit: int,
    ) -> None:
        self.limit = limit
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
        )

    @property
    def remaining(self) -> int:
        return max(
            0,
            self.limit - self.total,
        )

    @property
    def exhausted(self) -> bool:
        return self.total >= self.limit

    def add(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.input_tokens += max(
            0,
            input_tokens,
        )

        self.output_tokens += max(
            0,
            output_tokens,
        )


def _usage(
    response: Any,
) -> tuple[int, int]:

    if not getattr(
        response,
        "usage",
        None,
    ):
        return 0, 0

    return (
        response.usage.prompt_tokens or 0,
        response.usage.completion_tokens or 0,
    )


# ============================================================
# EVIDENCE EXTRACTION
# ============================================================

def extract_findings(
    *,
    question: str,
    title: str,
    url: str,
    content: str,
    budget: TokenBudget,
) -> list[str]:

    if budget.exhausted:
        return []

    prompt = f"""
RESEARCH QUESTION:

{question}


SOURCE:

Title: {title}
URL: {url}


TASK

Extract 3 to 7 concise findings from the source that are directly
relevant to the research question.

RULES

- Use only information contained in SOURCE CONTENT.
- Do not add outside knowledge.
- Preserve important numbers, dates and qualifications.
- Do not speculate.
- Do not create citations.
- Do not mention source IDs.
- Return one finding per line.
- Start every finding with "- ".


SOURCE CONTENT:

{content[:settings.max_source_chars]}
"""

    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract faithful, compact evidence "
                        "from a supplied source. Never introduce "
                        "outside facts."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

    except Exception:
        return []

    input_tokens, output_tokens = _usage(
        response
    )

    budget.add(
        input_tokens,
        output_tokens,
    )

    text = (
        response.choices[0].message.content
        or ""
    )

    findings: list[str] = []

    for line in text.splitlines():
        clean = line.strip()

        if not clean.startswith("-"):
            continue

        finding = clean.lstrip("-").strip()

        if (
            finding
            and finding not in findings
        ):
            findings.append(finding)

    return findings[:7]


# ============================================================
# RESULT HELPER
# ============================================================

def _result(
    *,
    success: bool,
    termination: str,
    scratchpad: Scratchpad,
    budget: TokenBudget,
    iterations: int,
    tool_calls: int,
    model_errors: int,
    logger: RunLogger,
    answer: str | None = None,
    source_ids: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:

    logger.finish(
        success=success,
        termination=termination,
        iterations=iterations,
        tool_calls=tool_calls,
        input_tokens=budget.input_tokens,
        output_tokens=budget.output_tokens,
    )

    return {
        "success": success,
        "termination": termination,
        "answer": answer,
        "source_ids": source_ids or [],
        "sources": scratchpad.sources,
        "error": error,
        "iterations": iterations,
        "tool_calls": tool_calls,
        "model_errors": model_errors,
        "input_tokens": budget.input_tokens,
        "output_tokens": budget.output_tokens,
        "total_tokens": budget.total,
        "token_limit": budget.limit,
        "run_id": logger.run_id,
        "log_path": str(logger.path),
    }


# ============================================================
# MAIN LOOP
# ============================================================

def run_agent(
    question: str,
) -> dict[str, Any]:

    question = question.strip()

    # Logger requires a string, so it can also record bad input.
    logger = RunLogger(
        question=question
    )

    scratchpad = Scratchpad()

    budget = TokenBudget(
        settings.max_total_tokens
    )

    if not question:
        return _result(
            success=False,
            termination="invalid_question",
            scratchpad=scratchpad,
            budget=budget,
            iterations=0,
            tool_calls=0,
            model_errors=0,
            logger=logger,
            error="Research question cannot be empty.",
        )

    verifier = EvidenceVerifier(
        client
    )

    tool_calls_used = 0
    model_errors = 0

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "ORIGINAL RESEARCH QUESTION:\n"
                f"{question}"
            ),
        },
    ]

    # ========================================================
    # BOUNDED AGENT LOOP
    # ========================================================

    for iteration in range(
        1,
        settings.max_iterations + 1,
    ):

        if budget.exhausted:
            return _result(
                success=False,
                termination="token_limit",
                scratchpad=scratchpad,
                budget=budget,
                iterations=iteration - 1,
                tool_calls=tool_calls_used,
                model_errors=model_errors,
                logger=logger,
                error=(
                    "Hard token budget reached before completion."
                ),
            )

        evidence = scratchpad.render()

        decision_prompt = f"""
CURRENT VERIFIED EVIDENCE:

{evidence}


RESEARCH QUESTION:

{question}


RESOURCE STATE:

Iteration:
{iteration}/{settings.max_iterations}

Research tool calls:
{tool_calls_used}/{settings.max_tool_calls}

Token budget remaining:
{budget.remaining}


Choose the SINGLE best next action.

If sufficient verified evidence exists, call finish.
"""

        logger.log(
            "iteration_started",
            {
                "iteration": iteration,
                "source_count": scratchpad.source_count,
                "tool_calls": tool_calls_used,
                "tokens_used": budget.total,
                "tokens_remaining": budget.remaining,
            },
        )

        try:
            response = client.chat.completions.create(
                model=settings.model,
                messages=messages
                + [
                    {
                        "role": "user",
                        "content": decision_prompt,
                    }
                ],
                tools=TOOLS,
                tool_choice="auto",
                parallel_tool_calls=False,
                temperature=0,
            )

        except Exception as exc:
            model_errors += 1

            error_text = str(exc)

            logger.log(
                "model_error",
                {
                    "iteration": iteration,
                    "error": error_text[:1000],
                },
            )

            recoverable = any(
                marker in error_text
                for marker in (
                    "tool_use_failed",
                    "Failed to parse tool call",
                    "not in request.tools",
                )
            )

            if (
                recoverable
                and model_errors
                <= settings.max_model_errors
            ):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "TOOL OBSERVATION:\n"
                            "Your previous action was invalid. "
                            "Call exactly one available action: "
                            "search, fetch, or finish."
                        ),
                    }
                )

                continue

            return _result(
                success=False,
                termination="model_error",
                scratchpad=scratchpad,
                budget=budget,
                iterations=iteration,
                tool_calls=tool_calls_used,
                model_errors=model_errors,
                logger=logger,
                error=error_text,
            )

        input_tokens, output_tokens = _usage(
            response
        )

        budget.add(
            input_tokens,
            output_tokens,
        )

        if budget.exhausted:
            return _result(
                success=False,
                termination="token_limit",
                scratchpad=scratchpad,
                budget=budget,
                iterations=iteration,
                tool_calls=tool_calls_used,
                model_errors=model_errors,
                logger=logger,
                error=(
                    "Hard token budget reached during "
                    "model execution."
                ),
            )

        message = (
            response.choices[0].message
        )

        # ====================================================
        # INVALID PLAIN-TEXT ACTION
        # ====================================================

        if not message.tool_calls:
            model_errors += 1

            logger.log(
                "invalid_model_action",
                {
                    "iteration": iteration,
                    "content": (
                        message.content or ""
                    )[:1000],
                },
            )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "TOOL OBSERVATION:\n"
                        "No valid action was called. Use exactly "
                        "one of search, fetch, or finish."
                    ),
                }
            )

            if (
                model_errors
                > settings.max_model_errors
            ):
                return _result(
                    success=False,
                    termination="model_error_limit",
                    scratchpad=scratchpad,
                    budget=budget,
                    iterations=iteration,
                    tool_calls=tool_calls_used,
                    model_errors=model_errors,
                    logger=logger,
                    error=(
                        "Model repeatedly failed to produce "
                        "a valid action."
                    ),
                )

            continue

        call = message.tool_calls[0]

        tool_name = call.function.name

        # finish is a completion action rather than an external
        # research call. Only search/fetch count toward the research
        # tool ceiling.
        if tool_name in {
            "search",
            "fetch",
        }:
            tool_calls_used += 1

            if (
                tool_calls_used
                > settings.max_tool_calls
            ):
                return _result(
                    success=False,
                    termination="tool_call_limit",
                    scratchpad=scratchpad,
                    budget=budget,
                    iterations=iteration,
                    tool_calls=tool_calls_used,
                    model_errors=model_errors,
                    logger=logger,
                    error=(
                        "Maximum research tool-call limit reached."
                    ),
                )

        try:
            arguments = json.loads(
                call.function.arguments
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            model_errors += 1

            observation = (
                "The action contained malformed JSON. "
                "Use valid structured arguments."
            )

            logger.log(
                "invalid_tool_arguments",
                {
                    "iteration": iteration,
                    "tool": tool_name,
                },
            )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "TOOL OBSERVATION:\n"
                        f"{observation}"
                    ),
                }
            )

            continue

        logger.log(
            "action_selected",
            {
                "iteration": iteration,
                "tool": tool_name,
                "arguments": arguments,
            },
        )

        # ====================================================
        # SEARCH
        # ====================================================

        if tool_name == "search":
            try:
                validated = SearchToolInput(
                    **arguments
                )

            except ValidationError as exc:
                observation = (
                    f"Invalid search input: {exc}"
                )

            else:
                query = validated.query.strip()

                if scratchpad.has_searched(
                    query
                ):
                    observation = (
                        "Equivalent search already executed. "
                        "Choose a materially different query, "
                        "fetch a candidate, or finish."
                    )

                else:
                    scratchpad.record_query(
                        query
                    )

                    result = search(
                        query
                    )

                    if not result.success:
                        observation = (
                            f"Search failed: {result.error}"
                        )

                    else:
                        candidates: list[str] = []

                        for index, item in enumerate(
                            result.results,
                            start=1,
                        ):
                            candidates.append(
                                f"{index}. {item.title}\n"
                                f"URL: {item.url}\n"
                                f"Snippet: {item.snippet}"
                            )

                        observation = (
                            "Search succeeded.\n"
                            "IMPORTANT: These are discovery "
                            "candidates, not citable evidence.\n\n"
                            + "\n\n".join(candidates)
                        )

        # ====================================================
        # FETCH
        # ====================================================

        elif tool_name == "fetch":
            try:
                validated = FetchToolInput(
                    **arguments
                )

            except ValidationError as exc:
                observation = (
                    f"Invalid fetch input: {exc}"
                )

            else:
                url = str(validated.url)

                if scratchpad.has_attempted_url(
                    url
                ):
                    observation = (
                        "This URL has already been attempted. "
                        "Choose another source or finish."
                    )

                else:
                    scratchpad.record_url_attempt(
                        url
                    )

                    result = fetch(
                        url
                    )

                    if not result.success:
                        observation = (
                            f"Fetch failed: {result.error}"
                        )

                    else:
                        findings = extract_findings(
                            question=question,
                            title=(
                                result.title
                                or "Untitled source"
                            ),
                            url=result.url,
                            content=(
                                result.content or ""
                            ),
                            budget=budget,
                        )

                        if budget.exhausted:
                            return _result(
                                success=False,
                                termination="token_limit",
                                scratchpad=scratchpad,
                                budget=budget,
                                iterations=iteration,
                                tool_calls=tool_calls_used,
                                model_errors=model_errors,
                                logger=logger,
                                error=(
                                    "Hard token budget reached "
                                    "during evidence extraction."
                                ),
                            )

                        if not findings:
                            observation = (
                                "The page was fetched but useful "
                                "evidence could not be extracted. "
                                "Choose another source."
                            )

                        else:
                            source_id = (
                                scratchpad.add_source(
                                    title=(
                                        result.title
                                        or "Untitled source"
                                    ),
                                    url=result.url,
                                    findings=findings,
                                )
                            )

                            logger.log(
                                "source_verified",
                                {
                                    "iteration": iteration,
                                    "source_id": source_id,
                                    "title": (
                                        result.title
                                        or "Untitled source"
                                    ),
                                    "url": result.url,
                                    "findings": findings,
                                },
                            )

                            findings_text = "\n".join(
                                f"- {finding}"
                                for finding in findings
                            )

                            observation = (
                                "Fetch succeeded. "
                                "Verified evidence stored as "
                                f"[{source_id}].\n\n"
                                f"{findings_text}"
                            )

        # ====================================================
        # FINISH
        # ====================================================

        elif tool_name == "finish":
            try:
                validated = FinishToolInput(
                    **arguments
                )

            except ValidationError as exc:
                observation = (
                    f"Invalid finish input: {exc}"
                )

            else:
                structural = (
                    verifier.validate_citations(
                        summary=validated.summary,
                        declared_sources=validated.sources,
                        scratchpad=scratchpad,
                    )
                )

                if not structural.passed:
                    observation = (
                        "Finish rejected by citation validation: "
                        f"{structural.reason}"
                    )

                    logger.log(
                        "finish_rejected",
                        {
                            "iteration": iteration,
                            "stage": "citation_validation",
                            "reason": structural.reason,
                        },
                    )

                else:
                    (
                        evidence_check,
                        verifier_input_tokens,
                        verifier_output_tokens,
                    ) = verifier.verify_evidence_support(
                        summary=validated.summary,
                        scratchpad=scratchpad,
                    )

                    budget.add(
                        verifier_input_tokens,
                        verifier_output_tokens,
                    )

                    if budget.exhausted:
                        return _result(
                            success=False,
                            termination="token_limit",
                            scratchpad=scratchpad,
                            budget=budget,
                            iterations=iteration,
                            tool_calls=tool_calls_used,
                            model_errors=model_errors,
                            logger=logger,
                            error=(
                                "Hard token budget reached "
                                "during final verification."
                            ),
                        )

                    if not evidence_check.passed:
                        details = ""

                        if (
                            evidence_check.unsupported_claims
                        ):
                            details = (
                                "\nUnsupported claims:\n- "
                                + "\n- ".join(
                                    evidence_check
                                    .unsupported_claims
                                )
                            )

                        observation = (
                            "Finish rejected by evidence "
                            "verification: "
                            f"{evidence_check.reason}"
                            f"{details}\n"
                            "Rewrite using only verified evidence "
                            "or research further if necessary."
                        )

                        logger.log(
                            "finish_rejected",
                            {
                                "iteration": iteration,
                                "stage": (
                                    "evidence_verification"
                                ),
                                "reason": (
                                    evidence_check.reason
                                ),
                                "unsupported_claims": (
                                    evidence_check
                                    .unsupported_claims
                                ),
                            },
                        )

                    else:
                        logger.log(
                            "finish_accepted",
                            {
                                "iteration": iteration,
                                "source_ids": (
                                    validated.sources
                                ),
                                "verification": (
                                    evidence_check.reason
                                ),
                            },
                        )

                        return _result(
                            success=True,
                            termination="finish",
                            scratchpad=scratchpad,
                            budget=budget,
                            iterations=iteration,
                            tool_calls=tool_calls_used,
                            model_errors=model_errors,
                            logger=logger,
                            answer=validated.summary,
                            source_ids=(
                                validated.sources
                            ),
                        )

        # ====================================================
        # UNKNOWN ACTION
        # ====================================================

        else:
            model_errors += 1

            observation = (
                f"Unknown action '{tool_name}'. "
                "Only search, fetch and finish are valid."
            )

        logger.log(
            "observation",
            {
                "iteration": iteration,
                "tool": tool_name,
                "observation": observation[:3000],
            },
        )

        # Keep observations bounded to avoid unbounded history.
        messages.append(
            {
                "role": "user",
                "content": (
                    "TOOL OBSERVATION:\n"
                    f"{observation[:5000]}"
                ),
            }
        )

    return _result(
        success=False,
        termination="iteration_limit",
        scratchpad=scratchpad,
        budget=budget,
        iterations=settings.max_iterations,
        tool_calls=tool_calls_used,
        model_errors=model_errors,
        logger=logger,
        error=(
            "The agent reached its hard iteration limit before "
            "a verified finish was accepted."
        ),
    )