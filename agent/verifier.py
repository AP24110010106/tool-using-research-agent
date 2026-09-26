from __future__ import annotations

import json
import re

from groq import Groq

from agent.config import settings
from agent.schemas import VerificationResult
from agent.scratchpad import Scratchpad


SOURCE_PATTERN = re.compile(
    r"\[(S\d+)\]"
)


class EvidenceVerifier:
    """
    Validates a proposed final answer before the agent is allowed
    to terminate.

    Layer 1:
        Deterministic citation/provenance checks.

    Layer 2:
        LLM-based evidence support check against only the compact
        verified scratchpad.
    """

    def __init__(
        self,
        client: Groq,
    ) -> None:
        self.client = client

    # ========================================================
    # DETERMINISTIC VALIDATION
    # ========================================================

    def validate_citations(
        self,
        summary: str,
        declared_sources: list[str],
        scratchpad: Scratchpad,
    ) -> VerificationResult:

        summary = summary.strip()

        if not summary:
            return VerificationResult(
                passed=False,
                reason="The proposed summary is empty.",
            )

        cited_sources = list(
            dict.fromkeys(
                SOURCE_PATTERN.findall(summary)
            )
        )

        if not cited_sources:
            return VerificationResult(
                passed=False,
                reason=(
                    "The answer contains no [S#] citations."
                ),
            )

        valid_sources = scratchpad.valid_source_ids()

        unknown = [
            source_id
            for source_id in cited_sources
            if source_id not in valid_sources
        ]

        if unknown:
            return VerificationResult(
                passed=False,
                reason=(
                    "The answer cites unknown source IDs: "
                    + ", ".join(unknown)
                ),
                cited_sources=cited_sources,
            )

        declared_unique = list(
            dict.fromkeys(declared_sources)
        )

        unknown_declared = [
            source_id
            for source_id in declared_unique
            if source_id not in valid_sources
        ]

        if unknown_declared:
            return VerificationResult(
                passed=False,
                reason=(
                    "finish() declared unknown source IDs: "
                    + ", ".join(unknown_declared)
                ),
                cited_sources=cited_sources,
            )

        cited_set = set(cited_sources)
        declared_set = set(declared_unique)

        if cited_set != declared_set:
            missing_from_declaration = (
                cited_set - declared_set
            )

            unused_declarations = (
                declared_set - cited_set
            )

            problems: list[str] = []

            if missing_from_declaration:
                problems.append(
                    "cited but not declared: "
                    + ", ".join(
                        sorted(missing_from_declaration)
                    )
                )

            if unused_declarations:
                problems.append(
                    "declared but not cited: "
                    + ", ".join(
                        sorted(unused_declarations)
                    )
                )

            return VerificationResult(
                passed=False,
                reason=(
                    "Citation/source mismatch ("
                    + "; ".join(problems)
                    + ")."
                ),
                cited_sources=cited_sources,
            )

        return VerificationResult(
            passed=True,
            reason="Citation structure is valid.",
            cited_sources=cited_sources,
        )

    # ========================================================
    # EVIDENCE SUPPORT
    # ========================================================

    def verify_evidence_support(
        self,
        summary: str,
        scratchpad: Scratchpad,
    ) -> tuple[VerificationResult, int, int]:
        """
        Ask a separate bounded verification pass whether material
        factual claims are supported by the collected evidence.

        Returns:
            result,
            prompt_tokens,
            completion_tokens
        """

        evidence = scratchpad.render()

        prompt = f"""
You are an evidence verifier, not a research agent.

Evaluate whether the proposed answer is supported by the VERIFIED
EVIDENCE below.

Do not use outside knowledge.

A claim is supported only if the cited evidence reasonably supports
that claim.

Minor wording differences and reasonable paraphrases are allowed.

Do not reject an answer merely because it is concise.

Reject claims that introduce material facts, causal explanations,
numbers, dates, examples, or conclusions that are not supported by
the supplied evidence.

PROPOSED ANSWER:

{summary}


VERIFIED EVIDENCE:

{evidence}


Return ONLY valid JSON using this exact structure:

{{
  "passed": true,
  "reason": "short explanation",
  "unsupported_claims": []
}}

If unsupported material claims exist:

{{
  "passed": false,
  "reason": "short explanation",
  "unsupported_claims": [
    "unsupported claim"
  ]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=settings.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Check factual support using only the "
                            "provided evidence. Return valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
            )

        except Exception as exc:
            return (
                VerificationResult(
                    passed=False,
                    reason=(
                        "Evidence verifier failed: "
                        f"{str(exc)[:300]}"
                    ),
                ),
                0,
                0,
            )

        prompt_tokens = 0
        completion_tokens = 0

        if response.usage:
            prompt_tokens = (
                response.usage.prompt_tokens or 0
            )

            completion_tokens = (
                response.usage.completion_tokens or 0
            )

        text = (
            response.choices[0].message.content
            or ""
        ).strip()

        # Models occasionally wrap JSON in markdown fences.
        if text.startswith("```"):
            text = re.sub(
                r"^```(?:json)?\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )

            text = re.sub(
                r"\s*```$",
                "",
                text,
            )

        try:
            data = json.loads(text)

        except json.JSONDecodeError:
            return (
                VerificationResult(
                    passed=False,
                    reason=(
                        "Evidence verifier returned invalid JSON."
                    ),
                ),
                prompt_tokens,
                completion_tokens,
            )

        passed = bool(
            data.get("passed", False)
        )

        reason = str(
            data.get(
                "reason",
                "No verifier explanation provided.",
            )
        )

        raw_unsupported = data.get(
            "unsupported_claims",
            [],
        )

        unsupported_claims = (
            [
                str(item)
                for item in raw_unsupported
            ]
            if isinstance(raw_unsupported, list)
            else []
        )

        return (
            VerificationResult(
                passed=passed,
                reason=reason,
                cited_sources=list(
                    dict.fromkeys(
                        SOURCE_PATTERN.findall(summary)
                    )
                ),
                unsupported_claims=unsupported_claims,
            ),
            prompt_tokens,
            completion_tokens,
        )