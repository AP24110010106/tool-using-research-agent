from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


# ============================================================
# SEARCH
# ============================================================

class SearchResult(BaseModel):
    title: str
    url: HttpUrl
    snippet: str = ""


class SearchResponse(BaseModel):
    success: bool
    query: str

    results: list[SearchResult] = Field(
        default_factory=list
    )

    error: str | None = None


class SearchToolInput(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=500,
    )


# ============================================================
# FETCH
# ============================================================

class FetchResponse(BaseModel):
    success: bool
    url: str

    title: str | None = None
    content: str | None = None
    error: str | None = None


class FetchToolInput(BaseModel):
    url: HttpUrl


# ============================================================
# FINISH
# ============================================================

class FinishToolInput(BaseModel):
    summary: str = Field(
        min_length=1,
    )

    sources: list[str] = Field(
        min_length=1,
    )


# ============================================================
# VERIFICATION
# ============================================================

class VerificationResult(BaseModel):
    passed: bool

    reason: str

    cited_sources: list[str] = Field(
        default_factory=list
    )

    unsupported_claims: list[str] = Field(
        default_factory=list
    )


# ============================================================
# RUN RESULT
# ============================================================

class RunResult(BaseModel):
    success: bool

    termination: Literal[
        "finish",
        "invalid_question",
        "iteration_limit",
        "tool_call_limit",
        "token_limit",
        "model_error",
        "model_error_limit",
    ]

    answer: str | None = None

    source_ids: list[str] = Field(
        default_factory=list
    )

    error: str | None = None

    iterations: int = 0
    tool_calls: int = 0
    model_errors: int = 0

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
        )