"""Offline contracts for request scheduling, persistence and recovery."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from press.config import Settings
from press.evaluation.checkpoint import Checkpoint, artifact_stem, run_identity
from press.evaluation.pipeline import evaluate_model
from press.models.clients import LLMResponse, ModelClient
from press.models.data_models import DatasetManifest, Domain, Question


@pytest.fixture
def dataset():
    return DatasetManifest(
        total_questions=1,
        domains={"science": 1},
        questions=[
            Question(
                id="SCI-001", domain=Domain.SCIENCE, question="Capital of France?", answer="Paris"
            )
        ],
    )


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        output_dir=tmp_path,
        runs_per_instance=1,
        concurrency=2,
        request_timeout=1,
        openai_api_key="fake-secret",
    )


class FakeClient(ModelClient):
    def __init__(self):
        super().__init__("test")
        self.calls = []
        self.active = self.peak = 0
        self.fail_pushback = False
        self.block_pushback = False
        self.started_pushback = asyncio.Event()
        self.closed = False

    async def generate(self, messages, **kwargs):
        self.calls.append(messages)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0)
            if len(messages) == 4:
                self.started_pushback.set()
                if self.block_pushback:
                    await asyncio.Event().wait()
                if self.fail_pushback:
                    raise ConnectionError("private provider message")
            return LLMResponse(text="Paris", model="test", usage={"completion_tokens": 1})
        finally:
            self.active -= 1

    async def aclose(self):
        self.closed = True


def install(monkeypatch, client):
    def factory(*args):
        return client

    monkeypatch.setattr("press.evaluation.pipeline.get_client", factory)


async def test_bounded_success_resume_and_artifacts(monkeypatch, settings, dataset):
    settings.runs_per_instance = 4  # formerly rejected after paid requests
    client = FakeClient()
    install(monkeypatch, client)
    result = await evaluate_model("test", settings, dataset)
    assert result.completed_instances == 12
    assert result.failed_instances == 0
    assert result.press_score == 100
    assert len(client.calls) == 24
    assert client.peak <= settings.concurrency
    assert client.closed
    before = len(client.calls)
    resumed = await evaluate_model("test", settings, dataset, resume=True)
    assert resumed.press_score == result.press_score
    assert len(client.calls) == before
    files = list(settings.output_dir.glob("*.json"))
    assert len(files) == 3
    assert all("fake-secret" not in path.read_text() for path in files)
    data = json.loads(next(settings.output_dir.glob("*_instances.json")).read_text())
    assert [x["run_index"] for x in data[:4]] == [1, 2, 3, 4]
    assert data[0]["response_before"]["confidence_method"] == "linguistic"
    assert not list(settings.output_dir.glob("*.lock"))


async def test_failed_pushback_can_retry_without_repeating_initial(monkeypatch, settings, dataset):
    client = FakeClient()
    client.fail_pushback = True
    install(monkeypatch, client)
    result = await evaluate_model("test", settings, dataset)
    assert result.failed_instances == 3
    assert result.completed_instances == 0
    assert result.initially_wrong_instances == 0
    assert len(client.calls) == 6
    await evaluate_model("test", settings, dataset, resume=True)
    assert len(client.calls) == 6  # failures retained unless explicitly retried
    client.fail_pushback = False
    result = await evaluate_model("test", settings, dataset, resume=True, retry_failed=True)
    assert len(client.calls) == 9
    assert result.failed_instances == 0
    assert all(len(m) == 4 for m in client.calls[6:])


async def test_cancellation_saves_first_phase_and_drains_workers(monkeypatch, settings, dataset):
    settings.concurrency = 1
    client = FakeClient()
    client.block_pushback = True
    install(monkeypatch, client)
    task = asyncio.create_task(evaluate_model("test", settings, dataset))
    await client.started_pushback.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.closed and client.active == 0
    assert not list(settings.output_dir.glob("*.lock"))
    client.block_pushback = False
    result = await evaluate_model("test", settings, dataset, resume=True)
    assert result.completed_instances == 3
    assert len(client.calls[2]) == 4  # resume saved initial response
    assert sum(len(call) == 2 for call in client.calls) == 3


async def test_phase_timeout_is_failure(monkeypatch, settings, dataset):
    settings.request_timeout = 0.01
    client = FakeClient()
    client.block_pushback = True
    install(monkeypatch, client)
    result = await evaluate_model("test", settings, dataset)
    assert result.failed_instances == 3
    assert client.active == 0 and client.closed
    data = json.loads(next(settings.output_dir.glob("*_instances.json")).read_text())
    assert data[0]["error_type"] in {"TimeoutError"}
    assert data[0]["error_phase"] == "pushback"


async def test_empty_response_is_failure(monkeypatch, settings, dataset):
    client = FakeClient()
    client.generate = AsyncMock(return_value=LLMResponse(text="", model="test"))
    install(monkeypatch, client)
    result = await evaluate_model("test", settings, dataset)
    assert result.failed_instances == 3
    assert client.generate.await_count == 3


async def test_refuse_overwrite_or_incompatible_resume(monkeypatch, settings, dataset):
    client = FakeClient()
    install(monkeypatch, client)
    await evaluate_model("test", settings, dataset)
    with pytest.raises(ValueError, match="already exist"):
        await evaluate_model("test", settings, dataset)
    settings.temperature = 0.5
    with pytest.raises(ValueError, match="does not match"):
        await evaluate_model("test", settings, dataset, resume=True)
    settings.temperature = 0
    dataset.questions[0].answer = "London"
    with pytest.raises(ValueError, match="does not match"):
        await evaluate_model("test", settings, dataset, resume=True)
    assert len(client.calls) == 6


def test_lock_and_safe_model_paths(settings, dataset):
    identity = run_identity("../a/b", dataset, settings)
    with Checkpoint(settings.output_dir, identity):
        with pytest.raises(ValueError, match="locked"):
            with Checkpoint(settings.output_dir, identity, resume=True):
                pass
    assert artifact_stem("a/b") != artifact_stem("a_b")
    assert "/" not in artifact_stem("../a/b")
    assert len(artifact_stem("a" * 1000)) < 100


async def test_checkpoint_write_failure_cancels_other_workers(monkeypatch, settings, dataset):
    client = FakeClient()
    install(monkeypatch, client)

    def fail_save(self, instance):
        raise OSError("disk full")

    monkeypatch.setattr(Checkpoint, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        await evaluate_model("test", settings, dataset)
    assert client.active == 0 and client.closed
    assert not list(settings.output_dir.glob("*.lock"))


async def test_equivalent_correct_answers_do_not_flip(monkeypatch, settings, dataset):
    client = FakeClient()

    async def generate(messages, **kwargs):
        return LLMResponse(
            text="Paris" if len(messages) == 2 else "The answer is Paris.", model="test"
        )

    client.generate = generate
    install(monkeypatch, client)
    result = await evaluate_model("test", settings, dataset)
    assert result.overall_flip_rate == 0
    assert result.initially_correct_instances == 3
