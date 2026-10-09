"""
Unified model client interface for the PRESS benchmark.

Each provider adapter normalises responses into a common format
so the evaluation pipeline is provider-agnostic.
"""

from __future__ import annotations

import inspect
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _is_transient(exc: BaseException) -> bool:
    """Retry transport errors and retryable HTTP statuses, never all exceptions."""
    import httpx

    if isinstance(exc, (TimeoutError, ConnectionError, httpx.TransportError)) or isinstance(
        exc.__cause__, httpx.TransportError
    ):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return isinstance(status, int) and (status in {408, 409, 429} or 500 <= status < 600)


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

    async def aclose(self) -> None:
        """Release SDK connection pools after completion, failure or cancellation."""
        client = getattr(self, "_client", None)
        if client is not None:
            close = getattr(client, "close", None)
            if close:
                result = close()
                if inspect.isawaitable(result):
                    await result

    @property
    def provider(self) -> str:
        return self.__class__.__name__


# ── OpenAI / GPT ─────────────────────────────────────────────────────────────


class OpenAIClient(ModelClient):
    """Client for OpenAI models (GPT-4o, etc.)."""

    def __init__(self, model_id: str, api_key: str = ""):
        super().__init__(model_id, api_key)
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, max_retries=0)

    @retry(
        retry=retry_if_exception(_is_transient),
        reraise=True,
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

        self._client = AsyncAnthropic(api_key=api_key, max_retries=0)

    @retry(
        retry=retry_if_exception(_is_transient),
        reraise=True,
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
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")

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
        from google import genai

        self._client = genai.Client(api_key=api_key)

    async def aclose(self) -> None:
        await self._client.aio.aclose()
        self._client.close()

    @retry(
        retry=retry_if_exception(_is_transient),
        reraise=True,
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
        from google.genai import types

        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents: list[types.ContentUnion] = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
            if m["role"] != "system"
        ]
        response = await self._client.aio.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return LLMResponse(text=response.text or "", model=self.model_id, raw=response)


# ── Together AI / Llama ──────────────────────────────────────────────────────


class TogetherClient(ModelClient):
    """Client for Together AI hosted models (Llama 3 70B, etc.)."""

    def __init__(self, model_id: str, api_key: str = ""):
        super().__init__(model_id, api_key)
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            max_retries=0,
        )

    @retry(
        retry=retry_if_exception(_is_transient),
        reraise=True,
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


def get_client(model_id: str, settings: Any | None = None) -> ModelClient:
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

    provider = provider_for_model(model_id)
    key_name = {
        "openai": "openai_api_key",
        "anthropic": "anthropic_api_key",
        "google": "google_api_key",
        "together": "together_api_key",
    }[provider]
    key = getattr(settings, key_name)
    if not key:
        raise ValueError(f"Missing {key_name.upper()} for {model_id}")
    adapters: dict[str, Callable[..., ModelClient]] = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "google": GeminiClient,
        "together": TogetherClient,
    }
    return adapters[provider](model_id, api_key=key)


def provider_for_model(model_id: str) -> str:
    """One routing rule shared by execution and CLI validation."""
    mid = model_id.lower()
    if mid.startswith("gpt-") or re.match(r"^o[1-9]\d*(?:-|$)", mid):
        return "openai"
    if mid.startswith("claude-"):
        return "anthropic"
    if mid.startswith("gemini-"):
        return "google"
    return "together"
