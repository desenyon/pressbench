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
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from press import __version__
from press.config import Settings

console = Console()

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
    "text-embedding", "text-search", "text-similarity", "text-moderation",
    "dall-e", "tts-", "whisper-", "babbage-", "davinci-", "ada-", "curie-",
)

_SKIP_SUFFIXES = ("-embed", "-embedding", "-base", "-instruct-v1")

_SKIP_SUBSTRINGS = (
    "image-generation", "image-gen", "-image", "native-audio", "preview-tts",
    "-tts", "embedding", "robotics", "computer-use", "vision", "audio",
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
        "gpt-", "o1", "o3", "claude-", "gemini-", "llama", "mistral",
        "mixtral", "qwen", "phi-", "falcon", "vicuna", "alpaca", "chat",
        "instruct", "command", "deepseek", "solar", "yi-",
    )
    return any(h in mid for h in chat_hints)


async def _fetch_openai_models(api_key: str) -> list[str]:
    """Fetch all chat-capable models from OpenAI."""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        pages = await client.models.list()
        return sorted(
            m.id for m in pages.data if _is_chat_model(m.id)
        )
    except Exception as exc:
        console.print(f"[yellow]  OpenAI model fetch failed: {exc}[/]")
        return []


async def _fetch_anthropic_models(api_key: str) -> list[str]:
    """Fetch all available models from Anthropic."""
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.models.list()
        return sorted(m.id for m in response.data)
    except Exception as exc:
        console.print(f"[yellow]  Anthropic model fetch failed: {exc}[/]")
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
        console.print(f"[yellow]  Google model fetch failed: {exc}[/]")
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
        return sorted(
            m.id for m in pages.data if _is_chat_model(m.id)
        )
    except Exception as exc:
        console.print(f"[yellow]  Together AI model fetch failed: {exc}[/]")
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
@click.option("--model", "-m", multiple=True, help="Model ID(s) to evaluate (default: all configured)")
@click.option("--discover", "-d", is_flag=True, help="Fetch every available model from all providers and run them all")
@click.option("--output", "-o", default="results", help="Output directory")
@click.option("--runs", "-r", default=3, type=int, help="Runs per instance (default: 3)")
@click.option("--concurrency", "-c", default=5, type=int, help="Max concurrent API requests")
@click.option("--temperature", "-t", default=0.0, type=float, help="Sampling temperature")
def run(
    model: tuple[str, ...],
    discover: bool,
    output: str,
    runs: int,
    concurrency: int,
    temperature: float,
) -> None:
    """Run the PRESS benchmark evaluation."""
    from press.config import get_settings
    from press.evaluation.pipeline import evaluate_all_models
    from press.reporting.html_report import generate_html_report
    from press.reporting.visualize import (
        generate_all_charts,
        print_leaderboard,
        print_model_result,
        save_leaderboard,
    )

    settings = get_settings()
    settings.output_dir = Path(output)
    settings.runs_per_instance = runs
    settings.concurrency = concurrency
    settings.temperature = temperature

    console.print(f"\n[bold cyan]PRESS Benchmark v{__version__}[/]")
    console.print(f"Output: {settings.output_dir}")
    console.print(f"Runs per instance: {settings.runs_per_instance}")
    console.print(f"Temperature: {settings.temperature}")

    if discover:
        console.print("\n[bold]Discovering models from all providers…[/]")
        provider_map = asyncio.run(_discover_all_models(settings))
        model_ids: list[str] = []
        for provider, mids in provider_map.items():
            console.print(f"  [cyan]{provider}[/]: {len(mids)} model(s)")
            model_ids.extend(mids)
        if not model_ids:
            console.print("[red]No models discovered. Check your API keys.[/]")
            sys.exit(1)
        # Deduplicate while preserving order
        seen: set[str] = set()
        model_ids = [m for m in model_ids if not (m in seen or seen.add(m))]  # type: ignore[func-returns-value]
    else:
        model_ids = list(model) if model else settings.models

    console.print(f"\nModels ([bold]{len(model_ids)}[/]): {', '.join(model_ids)}\n")

    # Validate API keys
    for mid in model_ids:
        mid_lower = mid.lower()
        if mid_lower.startswith("gpt-") and not settings.openai_api_key:
            console.print(f"[red]Missing OPENAI_API_KEY for {mid}[/]")
            sys.exit(1)
        elif mid_lower.startswith("claude-") and not settings.anthropic_api_key:
            console.print(f"[red]Missing ANTHROPIC_API_KEY for {mid}[/]")
            sys.exit(1)
        elif mid_lower.startswith("gemini-") and not settings.google_api_key:
            console.print(f"[red]Missing GOOGLE_API_KEY for {mid}[/]")
            sys.exit(1)
        elif not any(mid_lower.startswith(p) for p in ("gpt-", "claude-", "gemini-", "o1-", "o3-")):
            if not settings.together_api_key:
                console.print(f"[red]Missing TOGETHER_API_KEY for {mid}[/]")
                sys.exit(1)

    # Run evaluation
    results = asyncio.run(
        evaluate_all_models(model_ids=model_ids, settings=settings)
    )

    if not results:
        console.print("[red]No models were successfully evaluated.[/]")
        sys.exit(1)

    # Print results
    for r in results:
        print_model_result(r)

    if len(results) > 1:
        print_leaderboard(results)

    # Generate charts and report
    chart_dir = settings.output_dir / "charts"
    generate_all_charts(results, chart_dir)
    save_leaderboard(results, settings.output_dir)
    report_path = generate_html_report(results, settings.output_dir, runs)
    console.print(f"\n[bold green]Report saved to {report_path}[/]")


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
    except FileNotFoundError as e:
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
    else:
        console.print("\n[bold green]✓ Dataset is valid![/]")


@dataset.command()
@click.option("--path", "-p", default=None, help="Custom dataset directory")
def stats(path: str | None) -> None:
    """Print dataset statistics."""
    from press.dataset.loader import load_dataset

    manifest = load_dataset(path)

    console.print(f"\n[bold cyan]PRESS Dataset Statistics[/]\n")
    console.print(f"Total questions: {manifest.total_questions}")
    console.print(f"Total instances (× 3 tiers × 3 runs): {manifest.total_questions * 9}")

    for domain, count in manifest.domains.items():
        bar = "█" * (count // 2)
        console.print(f"  {domain:<15} {count:>3}  {bar}")

    # Difficulty distribution
    difficulties = {"easy": 0, "medium": 0, "hard": 0}
    for q in manifest.questions:
        difficulties[q.difficulty] = difficulties.get(q.difficulty, 0) + 1

    console.print(f"\nDifficulty distribution:")
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
        with open(rf, "r") as fh:
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
        with open(rf, "r") as fh:
            data = json.load(fh)
        results.append(ModelResult(**data))

    print_leaderboard(results)


# ── Models discovery commands ─────────────────────────────────────────────────


@main.group()
def models() -> None:
    """Commands for listing and discovering models from providers."""
    pass


@models.command("list")
@click.option("--provider", "-p", default=None,
              type=click.Choice(["openai", "anthropic", "google", "together"], case_sensitive=False),
              help="Only list models from this provider")
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
                console.print(f"[dim]  {prov}: no API key configured — skipped[/]")
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
