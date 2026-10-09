"""Bounded, resumable two-phase PRESS evaluation."""

from __future__ import annotations

import asyncio
import logging
import platform
from collections.abc import Callable
from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from press.calibration.confidence_classifier import extract_confidence
from press.config import Settings, get_settings
from press.dataset.loader import load_dataset, select_dataset
from press.evaluation.checkpoint import Checkpoint, atomic_json, run_identity
from press.evaluation.prompts import build_initial_messages, build_pushback_messages
from press.models.clients import LLMResponse, ModelClient, get_client
from press.models.data_models import (
    DatasetManifest,
    EvalInstance,
    ModelResponse,
    ModelResult,
    PushbackTier,
    Question,
)
from press.scoring.engine import compute_model_result
from press.utils.answer_matching import (
    answers_equivalent,
    check_answer,
    extract_answer_from_response,
)

logger = logging.getLogger(__name__)


def _build_instance_id(question_id: str, tier: PushbackTier, run: int) -> str:
    return f"{question_id}_T{tier.value}_R{run}"


def _response_to_model_response(
    llm_resp: LLMResponse,
    ground_truth: str,
    answer_match_mode: str,
    use_logprobs: bool,
) -> ModelResponse:
    if not llm_resp.text.strip():
        raise ValueError("Provider returned an empty response")
    extracted = extract_answer_from_response(llm_resp.text)
    logprob = llm_resp.logprobs[0].logprob if use_logprobs and llm_resp.logprobs else None
    conf = extract_confidence(llm_resp.text, logprob=logprob, prefer_logprob=use_logprobs)
    return ModelResponse(
        raw_text=llm_resp.text,
        extracted_answer=extracted,
        confidence=conf.score,
        confidence_method=conf.method,
        logprob=logprob,
        is_correct=check_answer(extracted, ground_truth, mode=answer_match_mode),
        finish_reason=llm_resp.finish_reason,
        usage=llm_resp.usage,
    )


async def evaluate_instance(
    client: ModelClient,
    question: Question,
    tier: PushbackTier,
    run_index: int,
    settings: Settings,
    semaphore: asyncio.Semaphore,
    previous: EvalInstance | None = None,
    save: Callable[[EvalInstance], None] | None = None,
) -> EvalInstance:
    """Checkpoint C1 before calling C2; retain it when C2 fails or is cancelled."""
    instance = (
        previous.model_copy(deep=True)
        if previous
        else EvalInstance(
            instance_id=_build_instance_id(question.id, tier, run_index),
            question_id=question.id,
            domain=question.domain,
            pushback_tier=tier,
            run_index=run_index,
            ground_truth=question.answer,
        )
    )
    instance.status = "pending"
    instance.error_phase = None
    instance.error_type = None

    async def generate(messages: list[dict[str, str]]) -> ModelResponse:
        response = await asyncio.wait_for(
            client.generate(
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                logprobs=settings.request_logprobs,
                top_logprobs=settings.top_logprobs,
            ),
            timeout=settings.request_timeout,
        )
        return _response_to_model_response(
            response,
            question.answer,
            settings.answer_match_mode,
            settings.use_logprobs,
        )

    async with semaphore:
        if instance.response_before is None:
            try:
                instance.response_before = await generate(build_initial_messages(question.question))
            except Exception as exc:
                instance.status = "failed"
                instance.error_phase = "initial"
                instance.error_type = type(exc).__name__
                logger.error(
                    "[%s] Initial request failed (%s)", instance.instance_id, type(exc).__name__
                )
                return instance
            if save is not None:
                save(instance)
        try:
            instance.response_after = await generate(
                build_pushback_messages(
                    question.question,
                    instance.response_before.raw_text,
                    tier,
                )
            )
        except Exception as exc:
            instance.status = "failed"
            instance.error_phase = "pushback"
            instance.error_type = type(exc).__name__
            logger.error("[%s] Pushback failed (%s)", instance.instance_id, type(exc).__name__)
            return instance

        before, after = instance.response_before, instance.response_after
        instance.c1, instance.c2 = before.confidence, after.confidence
        instance.cds = before.confidence - after.confidence
        # Two accepted renderings of the reference answer are not a harmful flip.
        instance.answer_flipped = not (
            (before.is_correct and after.is_correct)
            or answers_equivalent(
                before.extracted_answer, after.extracted_answer, mode=settings.answer_match_mode
            )
        )
        instance.flip_direction = None
        if instance.answer_flipped:
            source = "correct" if before.is_correct else "wrong"
            target = "correct" if after.is_correct else "wrong"
            instance.flip_direction = f"{source}_to_{target}"
        instance.status = "completed"
        return instance


async def evaluate_model(
    model_id: str,
    settings: Settings | None = None,
    dataset: DatasetManifest | None = None,
    output_dir: Path | None = None,
    *,
    resume: bool = False,
    retry_failed: bool = False,
) -> ModelResult:
    """Evaluate one model with a fixed worker pool and durable per-phase state."""
    settings = settings or get_settings()
    dataset = select_dataset(dataset or load_dataset(settings.dataset_path))
    output_dir = Path(output_dir or settings.output_dir)
    if retry_failed and not resume:
        raise ValueError("retry_failed requires resume")
    identity = run_identity(model_id, dataset, settings)
    with Checkpoint(output_dir, identity, resume=resume) as checkpoint:
        saved = checkpoint.load()
        jobs = [
            (q, tier, run)
            for q in dataset.questions
            for tier in PushbackTier
            for run in range(1, settings.runs_per_instance + 1)
        ]
        expected = {_build_instance_id(q.id, tier, run) for q, tier, run in jobs}
        if set(saved) - expected:
            raise ValueError("Checkpoint contains unexpected instance IDs")
        for q, tier, run in jobs:
            record = saved.get(_build_instance_id(q.id, tier, run))
            if record and (
                record.question_id != q.id
                or record.domain != q.domain
                or record.ground_truth != q.answer
                or record.pushback_tier != tier
                or record.run_index != run
            ):
                raise ValueError("Checkpoint instance does not match its question")
        pending_jobs = [
            job
            for job in jobs
            if (record := saved.get(_build_instance_id(job[0].id, job[1], job[2]))) is None
            or (not record.completed and (record.status != "failed" or retry_failed))
        ]
        pending = iter(pending_jobs)
        total = len(jobs)
        manifest = {**identity, "python_version": platform.python_version()}
        atomic_json(output_dir / f"{checkpoint.stem}_manifest.json", manifest)
        client: ModelClient | None = None
        workers: list[asyncio.Task] = []
        semaphore = asyncio.Semaphore(settings.concurrency)
        with Progress(
            TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()
        ) as progress:
            task = progress.add_task(
                f"Evaluating {model_id}", total=total, completed=total - len(pending_jobs)
            )

            async def worker() -> None:
                nonlocal client
                for question, tier, run in pending:
                    if client is None:
                        client = get_client(model_id, settings)
                    key = _build_instance_id(question.id, tier, run)
                    instance = await evaluate_instance(
                        client,
                        question,
                        tier,
                        run,
                        settings,
                        semaphore,
                        previous=saved.get(key),
                        save=checkpoint.save,
                    )
                    checkpoint.save(instance)
                    saved[key] = instance
                    progress.advance(task)

            try:
                workers = [
                    asyncio.create_task(worker()) for _ in range(min(settings.concurrency, total))
                ]
                await asyncio.gather(*workers)
            finally:
                for worker_task in workers:
                    if not worker_task.done():
                        worker_task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                if client is not None:
                    await client.aclose()

        instances = [saved[_build_instance_id(q.id, tier, run)] for q, tier, run in jobs]
        result = compute_model_result(model_id, instances)
        result.run_metadata = manifest
        atomic_json(
            output_dir / f"{checkpoint.stem}_instances.json",
            [inst.model_dump(mode="json") for inst in instances],
        )
        atomic_json(output_dir / f"{checkpoint.stem}_result.json", result.model_dump(mode="json"))
        return result


async def evaluate_all_models(
    model_ids: list[str] | None = None,
    settings: Settings | None = None,
    dataset: DatasetManifest | None = None,
    *,
    resume: bool = False,
    retry_failed: bool = False,
) -> list[ModelResult]:
    """Run models sequentially. Fatal setup/storage errors propagate to the caller."""
    settings = settings or get_settings()
    dataset = dataset or load_dataset(settings.dataset_path)
    results = []
    for model_id in dict.fromkeys(model_ids if model_ids is not None else settings.models):
        results.append(
            await evaluate_model(
                model_id,
                settings=settings,
                dataset=dataset,
                resume=resume,
                retry_failed=retry_failed,
            )
        )
    return results
