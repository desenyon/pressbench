"""
Core evaluation pipeline — orchestrates the end-to-end PRESS benchmark run.

For each model:
  1. Load dataset
  2. For each question × pushback tier × run:
     a. Send initial question → record C1 response
     b. Send pushback follow-up → record C2 response
  3. Extract answers, score confidence, compute CDS
  4. Aggregate and produce ModelResult
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from press.calibration.confidence_classifier import extract_confidence
from press.config import Settings, get_settings
from press.dataset.loader import load_dataset
from press.evaluation.prompts import build_initial_messages, build_pushback_messages
from press.models.clients import LLMResponse, ModelClient, get_client
from press.models.data_models import (
    DatasetManifest,
    Domain,
    EvalInstance,
    ModelResponse,
    ModelResult,
    PushbackTier,
    Question,
)
from press.scoring.engine import compute_model_result
from press.utils.answer_matching import check_answer, extract_answer_from_response

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_instance_id(question_id: str, tier: PushbackTier, run: int) -> str:
    return f"{question_id}_T{tier.value}_R{run}"


def _response_to_model_response(
    llm_resp: LLMResponse,
    ground_truth: str,
    answer_match_mode: str,
    use_logprobs: bool,
) -> ModelResponse:
    """Convert an LLMResponse into a scored ModelResponse."""
    extracted = extract_answer_from_response(llm_resp.text)
    is_correct = check_answer(extracted, ground_truth, mode=answer_match_mode)

    # Determine logprob for the answer token (take the first token's logprob as proxy)
    logprob_value: float | None = None
    if use_logprobs and llm_resp.logprobs and len(llm_resp.logprobs) > 0:
        logprob_value = llm_resp.logprobs[0].logprob

    conf = extract_confidence(
        llm_resp.text,
        logprob=logprob_value,
        prefer_logprob=use_logprobs,
    )

    return ModelResponse(
        raw_text=llm_resp.text,
        extracted_answer=extracted,
        confidence=conf.score,
        logprob=logprob_value,
        is_correct=is_correct,
    )


# ── Single instance evaluator ───────────────────────────────────────────────


async def evaluate_instance(
    client: ModelClient,
    question: Question,
    tier: PushbackTier,
    run_index: int,
    settings: Settings,
    semaphore: asyncio.Semaphore,
) -> EvalInstance:
    """Evaluate a single question × tier × run instance."""
    instance_id = _build_instance_id(question.id, tier, run_index)

    async with semaphore:
        # ── Phase 1: Initial question ────────────────────────────────────
        initial_messages = build_initial_messages(question.question)
        try:
            resp1 = await client.generate(
                messages=initial_messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                logprobs=settings.request_logprobs,
                top_logprobs=settings.top_logprobs,
            )
        except Exception as exc:
            logger.error(f"[{instance_id}] Phase 1 failed: {exc}")
            return EvalInstance(
                instance_id=instance_id,
                question_id=question.id,
                domain=question.domain,
                pushback_tier=tier,
                run_index=run_index,
                ground_truth=question.answer,
            )

        mr1 = _response_to_model_response(
            resp1, question.answer, settings.answer_match_mode, settings.use_logprobs
        )

        # ── Phase 2: Pushback ────────────────────────────────────────────
        pushback_messages = build_pushback_messages(
            question.question, resp1.text, tier
        )
        try:
            resp2 = await client.generate(
                messages=pushback_messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                logprobs=settings.request_logprobs,
                top_logprobs=settings.top_logprobs,
            )
        except Exception as exc:
            logger.error(f"[{instance_id}] Phase 2 failed: {exc}")
            return EvalInstance(
                instance_id=instance_id,
                question_id=question.id,
                domain=question.domain,
                pushback_tier=tier,
                run_index=run_index,
                ground_truth=question.answer,
                response_before=mr1,
            )

        mr2 = _response_to_model_response(
            resp2, question.answer, settings.answer_match_mode, settings.use_logprobs
        )

        # ── Compute per-instance scores ──────────────────────────────────
        c1 = mr1.confidence
        c2 = mr2.confidence
        cds = c1 - c2 if (c1 >= 0 and c2 >= 0) else None

        answer_flipped = (
            mr1.extracted_answer.strip().lower() != mr2.extracted_answer.strip().lower()
        )

        flip_direction: str | None = None
        if answer_flipped:
            if mr1.is_correct and not mr2.is_correct:
                flip_direction = "correct_to_wrong"
            elif not mr1.is_correct and mr2.is_correct:
                flip_direction = "wrong_to_correct"
            elif mr1.is_correct and mr2.is_correct:
                flip_direction = "correct_to_correct"
            else:
                flip_direction = "wrong_to_wrong"

        return EvalInstance(
            instance_id=instance_id,
            question_id=question.id,
            domain=question.domain,
            pushback_tier=tier,
            run_index=run_index,
            ground_truth=question.answer,
            response_before=mr1,
            response_after=mr2,
            c1=c1,
            c2=c2,
            cds=cds,
            answer_flipped=answer_flipped,
            flip_direction=flip_direction,
        )


# ── Full model evaluation ───────────────────────────────────────────────────


async def evaluate_model(
    model_id: str,
    settings: Settings | None = None,
    dataset: DatasetManifest | None = None,
    output_dir: Path | None = None,
) -> ModelResult:
    """Run the full PRESS benchmark on a single model.

    Parameters
    ----------
    model_id:
        The model identifier (e.g. "gpt-4o", "claude-3-5-sonnet-20241022").
    settings:
        PRESS settings. Defaults to loading from env.
    dataset:
        Pre-loaded dataset. Defaults to loading from disk.
    output_dir:
        Where to write raw instance results. Defaults to settings.output_dir.
    """
    if settings is None:
        settings = get_settings()
    if dataset is None:
        dataset = load_dataset(settings.dataset_path)
    if output_dir is None:
        output_dir = settings.output_dir

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = get_client(model_id, settings)
    semaphore = asyncio.Semaphore(settings.concurrency)

    # Build task list: question × tier × run
    tiers = list(PushbackTier)
    tasks: list[asyncio.Task] = []

    logger.info(
        f"Starting PRESS evaluation for {model_id}: "
        f"{len(dataset.questions)} questions × {len(tiers)} tiers × "
        f"{settings.runs_per_instance} runs = "
        f"{len(dataset.questions) * len(tiers) * settings.runs_per_instance} instances"
    )

    for question in dataset.questions:
        for tier in tiers:
            for run in range(1, settings.runs_per_instance + 1):
                task = asyncio.create_task(
                    evaluate_instance(client, question, tier, run, settings, semaphore)
                )
                tasks.append(task)

    total = len(tasks)

    # Execute with progress tracking
    instances: list[EvalInstance] = []
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        progress_task = progress.add_task(f"Evaluating {model_id}", total=total)

        for coro in asyncio.as_completed(tasks):
            instance = await coro
            instances.append(instance)
            progress.advance(progress_task)

    # Save raw instances
    safe_model_name = model_id.replace("/", "_").replace(":", "_")
    raw_path = output_dir / f"{safe_model_name}_instances.json"
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(
            [inst.model_dump() for inst in instances],
            fh,
            indent=2,
            default=str,
        )
    logger.info(f"Saved {len(instances)} raw instances to {raw_path}")

    # Compute aggregate result
    result = compute_model_result(model_id, instances)

    # Save result
    result_path = output_dir / f"{safe_model_name}_result.json"
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(result.model_dump(), fh, indent=2, default=str)
    logger.info(f"Saved model result to {result_path}")

    return result


# ── Multi-model evaluation ───────────────────────────────────────────────────


async def evaluate_all_models(
    model_ids: list[str] | None = None,
    settings: Settings | None = None,
) -> list[ModelResult]:
    """Run the PRESS benchmark on all configured models sequentially.

    Models are evaluated one at a time to avoid API rate limit conflicts.
    """
    if settings is None:
        settings = get_settings()
    if model_ids is None:
        model_ids = settings.models

    dataset = load_dataset(settings.dataset_path)
    results: list[ModelResult] = []

    for model_id in model_ids:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_id}")
        logger.info(f"{'='*60}")
        try:
            result = await evaluate_model(
                model_id, settings=settings, dataset=dataset
            )
            results.append(result)
        except Exception as exc:
            logger.error(f"Failed to evaluate {model_id}: {exc}")

    return results
