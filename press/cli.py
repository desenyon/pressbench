"""
PRESS CLI — command-line interface for the PRESS benchmark.

Usage:
    press run                         # Evaluate all configured models
    press run --model gpt-4o          # Evaluate a single model
    press run --discover              # Fetch & evaluate EVERY model from all providers
    press models list                 # List every model available from all providers
    press dataset validate            # Validate the dataset
    press report <results_dir>        # Generate report from existing results
    press leaderboard <dir>           # Print leaderboard from results
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.console import Console
from rich.logging import RichHandler

from press import __version__
from press.config import Settings
from press.models.data_models import Domain

console = Console()
error_console = Console(stderr=True)

# ── Logging setup ────────────────────────────────────────────────────────────


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


# ── Model discovery ──────────────────────────────────────────────────────────

# Models from each provider we want to skip (embeddings, tts, image, audio, etc.)
_SKIP_PREFIXES = (
    "text-embedding",
    "text-search",
    "text-similarity",
    "text-moderation",
    "dall-e",
    "tts-",
    "whisper-",
    "babbage-",
    "davinci-",
    "ada-",
    "curie-",
)

_SKIP_SUFFIXES = ("-embed", "-embedding", "-base", "-instruct-v1")

_SKIP_SUBSTRINGS = (
    "image-generation",
    "image-gen",
    "-image",
    "native-audio",
    "preview-tts",
    "-tts",
    "embedding",
    "robotics",
    "computer-use",
    "vision",
    "audio",
)


def _is_chat_model(model_id: str) -> bool:
    """Return True if the model looks like a chat/completion model."""
    mid = model_id.lower()
    for p in _SKIP_PREFIXES:
        if mid.startswith(p):
            return False
    for s in _SKIP_SUFFIXES:
        if mid.endswith(s):
            return False
    for sub in _SKIP_SUBSTRINGS:
        if sub in mid:
            return False
    # Must look like something useful for chat
    chat_hints = (
        "gpt-",
        "o1",
        "o3",
        "claude-",
        "gemini-",
        "llama",
        "mistral",
        "mixtral",
        "qwen",
        "phi-",
        "falcon",
        "vicuna",
        "alpaca",
        "chat",
        "instruct",
        "command",
        "deepseek",
        "solar",
        "yi-",
    )
    return any(h in mid for h in chat_hints)


async def _fetch_openai_models(api_key: str) -> list[str]:
    """Fetch all chat-capable models from OpenAI."""
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        pages = await client.models.list()
        return sorted(m.id for m in pages.data if _is_chat_model(m.id))
    except Exception as exc:
        error_console.print(f"[yellow]  OpenAI model fetch failed: {exc}[/]")
        return []


async def _fetch_anthropic_models(api_key: str) -> list[str]:
    """Fetch all available models from Anthropic."""
    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.models.list()
        return sorted(m.id for m in response.data)
    except Exception as exc:
        error_console.print(f"[yellow]  Anthropic model fetch failed: {exc}[/]")
        return []


async def _fetch_google_models(api_key: str) -> list[str]:
    """Fetch all generative models from Google."""
    try:
        import google.genai as genai  # type: ignore[import]

        client = genai.Client(api_key=api_key)
        result = []
        for m in client.models.list():
            mid = m.name.replace("models/", "") if hasattr(m, "name") and m.name else str(m)
            if _is_chat_model(mid):
                result.append(mid)
        return sorted(result)
    except Exception as exc:
        error_console.print(f"[yellow]  Google model fetch failed: {exc}[/]")
        return []


async def _fetch_together_models(api_key: str) -> list[str]:
    """Fetch all chat models from Together AI."""
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
        )
        pages = await client.models.list()
        return sorted(m.id for m in pages.data if _is_chat_model(m.id))
    except Exception as exc:
        error_console.print(f"[yellow]  Together AI model fetch failed: {exc}[/]")
        return []


async def _discover_all_models(settings: Settings) -> dict[str, list[str]]:
    """Return a mapping of provider → available model IDs."""
    tasks = {}
    if settings.openai_api_key:
        tasks["OpenAI"] = _fetch_openai_models(settings.openai_api_key)
    if settings.anthropic_api_key:
        tasks["Anthropic"] = _fetch_anthropic_models(settings.anthropic_api_key)
    if settings.google_api_key:
        tasks["Google"] = _fetch_google_models(settings.google_api_key)
    if settings.together_api_key:
        tasks["Together AI"] = _fetch_together_models(settings.together_api_key)

    results: dict[str, list[str]] = {}
    for provider, coro in tasks.items():
        results[provider] = await coro
    return results


# ── Main group ───────────────────────────────────────────────────────────────


@click.group()
@click.version_option(version=__version__, prog_name="PRESS")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """PRESS — Pushback Resistance & Epistemic Stability Score benchmark."""
    load_dotenv()
    _setup_logging(verbose)


# ── Run benchmark ────────────────────────────────────────────────────────────


@main.command()
@click.option("--model", "-m", multiple=True, help="Model ID(s); default: MODELS from settings")
@click.option("--discover", "-d", is_flag=True, help="Discover and evaluate available models")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--runs", "-r", type=click.IntRange(1, 10), default=None)
@click.option("--concurrency", "-c", type=click.IntRange(1, 50), default=None)
@click.option("--temperature", "-t", type=click.FloatRange(0, 2), default=None)
@click.option(
    "--timeout",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
    help="Deadline in seconds for each phase, including retries",
)
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
)
@click.option("--domain", "domains", multiple=True, type=click.Choice([d.value for d in Domain]))
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Use first N selected IDs")
@click.option("--resume", is_flag=True, help="Continue compatible checkpoints")
@click.option("--retry-failed", is_flag=True, help="With --resume, retry recorded failures")
@click.option("--dry-run", is_flag=True, help="Print local plan without API calls or writes")
def run(
    model: tuple[str, ...],
    discover: bool,
    output: Path | None,
    runs: int | None,
    concurrency: int | None,
    temperature: float | None,
    timeout: float | None,
    dataset_path: Path | None,
    domains: tuple[str, ...],
    limit: int | None,
    resume: bool,
    retry_failed: bool,
    dry_run: bool,
) -> None:
    """Run independently repeated question/pushback conversations."""
    from press.config import get_settings
    from press.dataset.loader import load_dataset, select_dataset
    from press.evaluation.pipeline import evaluate_all_models
    from press.models.clients import provider_for_model

    if retry_failed and not resume:
        raise click.UsageError("--retry-failed requires --resume")
    if discover and (model or dry_run):
        raise click.UsageError("--discover cannot be combined with --model or --dry-run")
    try:
        settings = get_settings()
        for name, value in {
            "output_dir": output,
            "runs_per_instance": runs,
            "concurrency": concurrency,
            "temperature": temperature,
            "request_timeout": timeout,
            "dataset_path": dataset_path,
        }.items():
            if value is not None:
                setattr(settings, name, value)
        selected = select_dataset(load_dataset(settings.dataset_path), domains, limit)
    except (ValueError, OSError, ValidationError) as exc:
        raise click.ClickException(f"Invalid configuration or dataset: {exc}") from exc

    model_ids = list(dict.fromkeys(model or settings.models))
    if discover:
        provider_map = asyncio.run(_discover_all_models(settings))
        model_ids = list(dict.fromkeys(mid for mids in provider_map.values() for mid in mids))
    if not model_ids:
        raise click.ClickException("No models selected or discovered.")
    instances = len(selected.questions) * 3 * settings.runs_per_instance
    plan = {
        "models": model_ids,
        "questions": len(selected.questions),
        "runs_per_instance": settings.runs_per_instance,
        "instances_per_model": instances,
        "max_requests_before_retries": instances * 2 * len(model_ids),
        "concurrency": settings.concurrency,
        "temperature": settings.temperature,
        "timeout_seconds": settings.request_timeout,
        "output": str(settings.output_dir),
    }
    if dry_run:
        click.echo(json.dumps(plan, indent=2))
        return
    console.print(f"PRESS v{__version__}: {instances} instances/model, {len(model_ids)} model(s)")
    console.print(f"Up to {plan['max_requests_before_retries']} requests before retries.")
    for mid in model_ids:
        provider = provider_for_model(mid)
        key_name = {
            "google": "google_api_key",
            "together": "together_api_key",
            "anthropic": "anthropic_api_key",
            "openai": "openai_api_key",
        }[provider]
        if not getattr(settings, key_name):
            raise click.ClickException(f"Missing {key_name.upper()} for {mid}")
    try:
        results = asyncio.run(
            evaluate_all_models(
                model_ids=model_ids,
                settings=settings,
                dataset=selected,
                resume=resume,
                retry_failed=retry_failed,
            )
        )
    except (ValueError, OSError, sqlite3.DatabaseError) as exc:
        raise click.ClickException(str(exc)) from exc

    from press.reporting.html_report import generate_html_report
    from press.reporting.visualize import (
        generate_all_charts,
        print_leaderboard,
        print_model_result,
        save_leaderboard,
    )

    for result in results:
        print_model_result(result)
    if len(results) > 1:
        print_leaderboard(results)
    generate_all_charts(results, settings.output_dir / "charts")
    save_leaderboard(results, settings.output_dir)
    path = generate_html_report(results, settings.output_dir)
    console.print(f"Report saved to {path}")
    if any(r.failed_instances for r in results):
        raise click.ClickException(
            "Run has failed instances; partial artifacts saved. Use --resume --retry-failed."
        )


# ── Dataset commands ─────────────────────────────────────────────────────────


@main.group()
def dataset() -> None:
    """Dataset management commands."""
    pass


@dataset.command()
@click.option("--path", "-p", default=None, help="Custom dataset directory")
def validate(path: str | None) -> None:
    """Validate the PRESS dataset."""
    from press.dataset.loader import load_dataset, validate_dataset

    console.print("[bold cyan]Validating PRESS dataset...[/]\n")

    try:
        manifest = load_dataset(path)
    except (ValueError, OSError) as e:
        console.print(f"[red]Error: {e}[/]")
        sys.exit(1)

    console.print(f"Total questions: {manifest.total_questions}")
    for domain, count in manifest.domains.items():
        console.print(f"  {domain}: {count}")

    issues = validate_dataset(manifest)
    if issues:
        console.print(f"\n[yellow]Found {len(issues)} issue(s):[/]")
        for issue in issues:
            console.print(f"  [yellow]⚠ {issue}[/]")
        raise click.ClickException("Dataset validation failed.")
    else:
        console.print("\n[bold green]✓ Dataset is valid![/]")


@dataset.command()
@click.option("--path", "-p", default=None, help="Custom dataset directory")
def stats(path: str | None) -> None:
    """Print dataset statistics."""
    from press.dataset.loader import load_dataset

    manifest = load_dataset(path)

    console.print("\n[bold cyan]PRESS Dataset Statistics[/]\n")
    console.print(f"Total questions: {manifest.total_questions}")
    console.print(f"Total instances (× 3 tiers × 3 runs): {manifest.total_questions * 9}")

    for domain, count in manifest.domains.items():
        bar = "█" * (count // 2)
        console.print(f"  {domain:<15} {count:>3}  {bar}")

    # Difficulty distribution
    difficulties = {"easy": 0, "medium": 0, "hard": 0}
    for q in manifest.questions:
        difficulties[q.difficulty] = difficulties.get(q.difficulty, 0) + 1

    console.print("\nDifficulty distribution:")
    for diff, count in difficulties.items():
        bar = "█" * (count // 3)
        console.print(f"  {diff:<10} {count:>3}  {bar}")


# ── Report generation from existing results ──────────────────────────────────


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
def report(results_dir: str) -> None:
    """Generate visualizations and HTML report from existing results."""
    from press.models.data_models import ModelResult
    from press.reporting.html_report import generate_html_report
    from press.reporting.visualize import (
        generate_all_charts,
        print_leaderboard,
        save_leaderboard,
    )

    results_path = Path(results_dir)
    result_files = list(results_path.glob("*_result.json"))

    if not result_files:
        console.print(f"[red]No *_result.json files found in {results_dir}[/]")
        sys.exit(1)

    results: list[ModelResult] = []
    for rf in result_files:
        with open(rf) as fh:
            data = json.load(fh)
        results.append(ModelResult(**data))
        console.print(f"Loaded: {rf.name}")

    print_leaderboard(results)

    chart_dir = results_path / "charts"
    generate_all_charts(results, chart_dir)
    save_leaderboard(results, results_path)
    report_path = generate_html_report(results, results_path)
    console.print(f"\n[bold green]Report saved to {report_path}[/]")


# ── Leaderboard ──────────────────────────────────────────────────────────────


@main.command()
@click.argument("results_dir", type=click.Path(exists=True))
def leaderboard(results_dir: str) -> None:
    """Display the PRESS leaderboard from existing results."""
    from press.models.data_models import ModelResult
    from press.reporting.visualize import print_leaderboard

    results_path = Path(results_dir)
    result_files = list(results_path.glob("*_result.json"))

    if not result_files:
        console.print(f"[red]No *_result.json files found in {results_dir}[/]")
        sys.exit(1)

    results: list[ModelResult] = []
    for rf in result_files:
        with open(rf) as fh:
            data = json.load(fh)
        results.append(ModelResult(**data))

    print_leaderboard(results)


# ── Models discovery commands ─────────────────────────────────────────────────


@main.group()
def models() -> None:
    """Commands for listing and discovering models from providers."""
    pass


@models.command("list")
@click.option(
    "--provider",
    "-p",
    default=None,
    type=click.Choice(["openai", "anthropic", "google", "together"], case_sensitive=False),
    help="Only list models from this provider",
)
@click.option("--json-out", is_flag=True, help="Output as JSON")
def models_list(provider: str | None, json_out: bool) -> None:
    """List every model available from all configured providers.

    Only providers with a configured API key are queried. Models are
    filtered to chat/completion models (embeddings, TTS, image models, etc.
    are excluded).
    """
    from press.config import get_settings

    settings = get_settings()

    console.print("\n[bold cyan]Discovering available models…[/]\n") if not json_out else None

    async def _run() -> dict[str, list[str]]:
        provider_map: dict[str, list[str]] = {}
        fetchers = {
            "openai": (settings.openai_api_key, _fetch_openai_models),
            "anthropic": (settings.anthropic_api_key, _fetch_anthropic_models),
            "google": (settings.google_api_key, _fetch_google_models),
            "together": (settings.together_api_key, _fetch_together_models),
        }
        for prov, (key, fn) in fetchers.items():
            if provider and prov != provider:
                continue
            if not key:
                error_console.print(f"[dim]  {prov}: no API key configured — skipped[/]")
                continue
            provider_map[prov] = await fn(key)
        return provider_map

    provider_map = asyncio.run(_run())

    if json_out:
        import json as _json

        click.echo(_json.dumps(provider_map, indent=2))
        return

    total = 0
    for prov, mids in provider_map.items():
        console.print(f"[bold]{prov}[/] ({len(mids)} models)")
        for mid in mids:
            console.print(f"  • {mid}")
        total += len(mids)

    if not provider_map:
        console.print("[red]No providers with configured API keys found.[/]")
        sys.exit(1)

    console.print(f"\n[bold green]Total: {total} models across {len(provider_map)} provider(s)[/]")
    console.print("\n[dim]Run [bold]press run --discover[/] to benchmark all of them.[/]")


if __name__ == "__main__":
    main()
