"""
PRESS benchmark configuration.

Loads settings from environment variables / .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Global configuration for PRESS."""

    # ── API Keys ──────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    together_api_key: str = Field(default="", alias="TOGETHER_API_KEY")

    # ── Models to evaluate ────────────────────────────────────────────────
    models: list[str] = Field(
        default=[
            "gpt-4o",
            "claude-3-5-sonnet-20241022",
            "claude-3-7-sonnet-20250219",
            "gemini-1.5-pro",
            "meta-llama/Llama-3-70b-chat-hf",
        ],
        description="Model identifiers to benchmark",
    )

    # ── Evaluation parameters ─────────────────────────────────────────────
    runs_per_instance: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of times each instance is run (for averaging)",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for model calls",
    )
    max_tokens: int = Field(
        default=512,
        description="Max tokens per model response",
    )
    request_logprobs: bool = Field(
        default=True,
        description="Whether to request logprobs from the API (when supported)",
    )
    top_logprobs: int = Field(
        default=5,
        description="Number of top logprobs to request",
    )
    concurrency: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of concurrent API requests",
    )

    # ── Paths ─────────────────────────────────────────────────────────────
    dataset_path: Path = Field(
        default=ROOT_DIR / "dataset" / "questions",
        description="Directory containing domain JSON files",
    )
    output_dir: Path = Field(
        default=Path("results"),
        description="Directory for evaluation outputs",
    )
    cache_dir: Path = Field(
        default=Path(".press_cache"),
        description="Directory for caching intermediate results",
    )

    # ── Confidence classifier ─────────────────────────────────────────────
    use_logprobs: bool = Field(
        default=True,
        description="Use logprobs for confidence when available",
    )
    classifier_model_path: Optional[str] = Field(
        default=None,
        description="Path to trained confidence classifier (fallback)",
    )

    # ── Answer matching ───────────────────────────────────────────────────
    answer_match_mode: str = Field(
        default="normalized",
        description="How to match answers: exact | normalized | llm",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
