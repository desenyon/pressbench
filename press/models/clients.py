"""
Unified model client interface for the PRESS benchmark.

Each provider adapter normalises responses into a common format
so the evaluation pipeline is provider-agnostic.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# ── Common response wrapper ─────────────────────────────────────────────────


@dataclass
class LLMResponse:
    """Provider-agnostic response from a language model."""

    text: str
    model: str
    logprobs: list[TokenLogprob] | None = None
    finish_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None  # Provider-specific raw response object


@dataclass
class TokenLogprob:
    """Log-probability information for a single token."""

    token: str
    logprob: float
    top_logprobs: dict[str, float] = field(default_factory=dict)


# ── Abstract base ────────────────────────────────────────────────────────────


class ModelClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(self, model_id: str, api_key: str = ""):
        self.model_id = model_id
        self.api_key = api_key

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        logprobs: bool = True,
        top_logprobs: int = 5,
    ) -> LLMResponse:
        """Send a chat completion request and return a normalised response."""
        ...

    @property
    def provider(self) -> str:
        return self.__class__.__name__


# ── OpenAI / GPT ─────────────────────────────────────────────────────────────


class OpenAIClient(ModelClient):
    """Client for OpenAI models (GPT-4o, etc.)."""

    def __init__(self, model_id: str, api_key: str = ""):
        super().__init__(model_id, api_key)
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        logprobs: bool = True,
        top_logprobs: int = 5,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""

        token_logprobs: list[TokenLogprob] | None = None
        if logprobs and choice.logprobs and choice.logprobs.content:
            token_logprobs = [
                TokenLogprob(
                    token=t.token,
                    logprob=t.logprob,
                    top_logprobs={tl.token: tl.logprob for tl in (t.top_logprobs or [])},
                )
                for t in choice.logprobs.content
            ]

        return LLMResponse(
            text=text,
            model=self.model_id,
            logprobs=token_logprobs,
            finish_reason=choice.finish_reason or "",
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
            raw=resp,
        )


# ── Anthropic / Claude ───────────────────────────────────────────────────────


class AnthropicClient(ModelClient):
    """Client for Anthropic models (Claude 3.5/3.7 Sonnet, etc.)."""

    def __init__(self, model_id: str, api_key: str = ""):
        super().__init__(model_id, api_key)
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        logprobs: bool = True,
        top_logprobs: int = 5,
    ) -> LLMResponse:
        # Anthropic uses a system message separately
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                chat_messages.append(m)

        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system_msg:
            kwargs["system"] = system_msg

        resp = await self._client.messages.create(**kwargs)
        text = resp.content[0].text if resp.content else ""

        return LLMResponse(
            text=text,
            model=self.model_id,
            logprobs=None,  # Anthropic doesn't expose logprobs
            finish_reason=resp.stop_reason or "",
            usage={
                "prompt_tokens": resp.usage.input_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.output_tokens if resp.usage else 0,
            },
            raw=resp,
        )


# ── Google / Gemini ──────────────────────────────────────────────────────────


class GeminiClient(ModelClient):
    """Client for Google Gemini models."""

    def __init__(self, model_id: str, api_key: str = ""):
        super().__init__(model_id, api_key)
        try:
            # New google-genai SDK (google-genai package)
            import google.genai as genai  # type: ignore[import]
            self._genai = genai
            self._client = genai.Client(api_key=api_key)
            self._use_new_sdk = True
        except ImportError:
            # Fallback: old google-generativeai SDK
            import google.generativeai as _genai_old  # type: ignore[import]
            _genai_old.configure(api_key=api_key)  # type: ignore[attr-defined]
            self._model = _genai_old.GenerativeModel(model_id)  # type: ignore[attr-defined]
            self._use_new_sdk = False

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        logprobs: bool = True,
        top_logprobs: int = 5,
    ) -> LLMResponse:
        # Build a single-string prompt from the message list
        # (both SDK versions accept a plain string)
        parts: list[str] = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                parts.append(f"[System]: {content}")
            elif role == "user":
                parts.append(f"[User]: {content}")
            elif role == "assistant":
                parts.append(f"[Assistant]: {content}")
        prompt = "\n\n".join(parts)

        loop = asyncio.get_event_loop()

        if self._use_new_sdk:
            genai = self._genai
            client = self._client

            def _call_new() -> str:
                response = client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(  # type: ignore[attr-defined]
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text or ""

            text = await loop.run_in_executor(None, _call_new)
        else:
            generation_config = {"temperature": temperature, "max_output_tokens": max_tokens}

            def _call_old() -> str:
                resp = self._model.generate_content(  # type: ignore[attr-defined, arg-type]
                    prompt, generation_config=generation_config  # type: ignore[arg-type]
                )
                return resp.text if resp.text else ""

            text = await loop.run_in_executor(None, _call_old)

        return LLMResponse(
            text=text,
            model=self.model_id,
            logprobs=None,
            finish_reason="stop",
            raw=None,
        )


# ── Together AI / Llama ──────────────────────────────────────────────────────


class TogetherClient(ModelClient):
    """Client for Together AI hosted models (Llama 3 70B, etc.)."""

    def __init__(self, model_id: str, api_key: str = ""):
        super().__init__(model_id, api_key)
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
    )
    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        logprobs: bool = True,
        top_logprobs: int = 5,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""

        token_logprobs: list[TokenLogprob] | None = None
        if logprobs and choice.logprobs and choice.logprobs.content:
            token_logprobs = [
                TokenLogprob(
                    token=t.token,
                    logprob=t.logprob,
                    top_logprobs={tl.token: tl.logprob for tl in (t.top_logprobs or [])},
                )
                for t in choice.logprobs.content
            ]

        return LLMResponse(
            text=text,
            model=self.model_id,
            logprobs=token_logprobs,
            finish_reason=choice.finish_reason or "",
            usage={
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
            raw=resp,
        )


# ── Factory ──────────────────────────────────────────────────────────────────


def get_client(model_id: str, settings: Optional[Any] = None) -> ModelClient:
    """Return the appropriate ModelClient for a given model identifier.

    Routing logic:
    - "gpt-*" or "o1-*" → OpenAI
    - "claude-*" → Anthropic
    - "gemini-*" → Google
    - everything else → Together AI (Llama, Mixtral, etc.)
    """
    if settings is None:
        from press.config import get_settings

        settings = get_settings()

    model_lower = model_id.lower()

    if model_lower.startswith(("gpt-", "o1-", "o3-")):
        return OpenAIClient(model_id, api_key=settings.openai_api_key)
    elif model_lower.startswith("claude-"):
        return AnthropicClient(model_id, api_key=settings.anthropic_api_key)
    elif model_lower.startswith("gemini-"):
        return GeminiClient(model_id, api_key=settings.google_api_key)
    else:
        # Default to Together for open-source models
        return TogetherClient(model_id, api_key=settings.together_api_key)
