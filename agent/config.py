import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Central configuration for the research agent.

    Operational limits live outside the model so the model cannot
    override its own safety and resource boundaries.
    """

    groq_api_key: str
    tavily_api_key: str

    model: str = "openai/gpt-oss-120b"

    max_iterations: int = 8
    max_tool_calls: int = 12
    max_model_errors: int = 2

    # Hard LLM token ceiling for one research run.
    max_total_tokens: int = 20_000

    # Maximum source text sent to the evidence extractor.
    max_source_chars: int = 12_000

    search_max_results: int = 5

    search_timeout_seconds: float = 15.0
    fetch_timeout_seconds: float = 20.0

    max_fetched_chars: int = 20_000


def load_settings() -> Settings:
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()

    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add it to your .env file."
        )

    if not tavily_api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is missing. "
            "Add it to your .env file."
        )

    return Settings(
        groq_api_key=groq_api_key,
        tavily_api_key=tavily_api_key,
    )


settings = load_settings()