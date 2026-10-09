from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from press.models.clients import (
    GeminiClient,
    ModelClient,
    OpenAIClient,
    _is_transient,
    provider_for_model,
)


@pytest.mark.parametrize(
    "status, retry",
    [(400, False), (401, False), (403, False), (429, True), (500, True), (503, True)],
)
def test_retry_policy(status, retry):
    error = Exception("provider error")
    error.status_code = status
    assert _is_transient(error) is retry


def test_transport_retry_policy():
    assert _is_transient(httpx.ConnectError("offline"))
    assert not _is_transient(ValueError("invalid response"))


def test_provider_routing():
    assert provider_for_model("o3") == "openai"
    assert provider_for_model("o4-mini") == "openai"
    assert provider_for_model("org/llama") == "together"


async def test_authentication_error_not_retried():
    client = object.__new__(OpenAIClient)
    ModelClient.__init__(client, "gpt-test")
    error = Exception("bad auth")
    error.status_code = 401
    create = AsyncMock(side_effect=error)
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    with pytest.raises(Exception, match="bad auth"):
        await client.generate([])
    assert create.await_count == 1


async def test_gemini_preserves_roles_and_system_instruction():
    client = object.__new__(GeminiClient)
    ModelClient.__init__(client, "gemini-test")
    generate = AsyncMock(return_value=SimpleNamespace(text="Paris"))
    client._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )
    result = await client.generate(
        [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "Paris"},
            {"role": "user", "content": "Are you sure?"},
        ]
    )
    args = generate.call_args.kwargs
    assert [c.role for c in args["contents"]] == ["user", "model", "user"]
    assert args["config"].system_instruction == "instructions"
    assert result.text == "Paris"


async def test_openai_sdk_wire_contract_and_transient_retry(monkeypatch):
    import openai
    from tenacity import wait_none

    calls = []

    def handler(request):
        import json

        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "retry", "type": "rate_limit"}})
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Paris"},
                        "logprobs": {
                            "content": [{"token": "Paris", "logprob": -0.2, "top_logprobs": []}]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

    client = object.__new__(OpenAIClient)
    ModelClient.__init__(client, "gpt-test")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client._client = openai.AsyncOpenAI(api_key="fake", max_retries=0, http_client=http)
    monkeypatch.setattr(OpenAIClient.generate.retry, "wait", wait_none())
    result = await client.generate([{"role": "user", "content": "Capital?"}])
    await client.aclose()
    assert len(calls) == 2
    assert calls[0]["logprobs"] is True
    assert result.text == "Paris"
    assert result.logprobs[0].logprob == -0.2
    assert result.usage["completion_tokens"] == 1
    assert http.is_closed


def test_wrapped_sdk_connection_error_is_transient():
    import openai

    error = openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    error.__cause__ = httpx.ConnectError("connection failed")
    assert _is_transient(error)
