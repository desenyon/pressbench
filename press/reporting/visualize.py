"""
Visualization and reporting — generate charts, tables, and the leaderboard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from rich.console import Console
from rich.table import Table

from press.models.data_models import (
    Domain,
    LeaderboardEntry,
    ModelResult,
    PushbackTier,
)

logger = logging.getLogger(__name__)
console = Console()

# ── Color palette ────────────────────────────────────────────────────────────

MODEL_COLORS = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # gray
]

TIER_COLORS = {
    PushbackTier.SOFT: "#55A868",
    PushbackTier.MEDIUM: "#DD8452",
    PushbackTier.HARD: "#C44E52",
}

TIER_LABELS = {
    PushbackTier.SOFT: "Tier 1 (Soft)",
    PushbackTier.MEDIUM: "Tier 2 (Medium)",
    PushbackTier.HARD: "Tier 3 (Hard)",
}


# ── Console output ───────────────────────────────────────────────────────────


def print_model_result(result: ModelResult) -> None:
    """Pretty-print a single model's PRESS results to the console."""
    console.print(f"\n[bold cyan]{'═' * 60}[/]")
    console.print(f"[bold white]  PRESS Results: {result.model_name}[/]")
    console.print(f"[bold cyan]{'═' * 60}[/]\n")

    # Primary metrics table
    table = Table(title="Primary Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", min_width=30)
    table.add_column("Value", justify="right", style="bold white", min_width=12)

    table.add_row("PRESS Score (0–100)", f"{result.press_score:.2f}")
    table.add_row("Mean CDS", f"{result.overall_mean_cds:.4f}")
    table.add_row("Median CDS", f"{result.overall_median_cds:.4f}")
    table.add_row("Std CDS", f"{result.overall_std_cds:.4f}")
    table.add_row("Flip Rate", f"{result.overall_flip_rate:.2%}")
    table.add_row("Correct → Wrong Flip Rate", f"{result.overall_correct_flip_rate:.2%}")
    table.add_row("Wrong → Correct Flip Rate", f"{result.overall_incorrect_flip_rate:.2%}")
    table.add_row("Initially Correct Instances", str(result.initially_correct_instances))
    table.add_row("Excluded Instances", str(result.excluded_instances))
    table.add_row("Total Instances", str(result.total_instances))
    console.print(table)

    # By tier
    if result.by_tier:
        tier_table = Table(title="\nCDS by Pushback Tier", show_header=True, header_style="bold magenta")
        tier_table.add_column("Tier", style="cyan")
        tier_table.add_column("Mean CDS", justify="right")
        tier_table.add_column("Median CDS", justify="right")
        tier_table.add_column("Flip Rate", justify="right")
        tier_table.add_column("N", justify="right")

        for tr in result.by_tier:
            tier_table.add_row(
                TIER_LABELS.get(tr.tier, str(tr.tier)),
                f"{tr.mean_cds:.4f}",
                f"{tr.median_cds:.4f}",
                f"{tr.flip_rate:.2%}",
                str(tr.n_instances),
            )
        console.print(tier_table)

    # By domain
    if result.by_domain:
        domain_table = Table(title="\nCDS by Domain", show_header=True, header_style="bold magenta")
        domain_table.add_column("Domain", style="cyan")
        domain_table.add_column("Mean CDS", justify="right")
        domain_table.add_column("Median CDS", justify="right")
        domain_table.add_column("Flip Rate", justify="right")
        domain_table.add_column("N", justify="right")

        for dr in result.by_domain:
            domain_table.add_row(
                dr.domain.value.replace("_", " ").title(),
                f"{dr.mean_cds:.4f}",
                f"{dr.median_cds:.4f}",
                f"{dr.flip_rate:.2%}",
                str(dr.n_instances),
            )
        console.print(domain_table)


def print_leaderboard(results: Sequence[ModelResult]) -> None:
    """Print a comparative leaderboard across all evaluated models."""
    entries = sorted(results, key=lambda r: r.press_score, reverse=True)

    console.print(f"\n[bold yellow]{'═' * 70}[/]")
    console.print("[bold white]  PRESS Benchmark Leaderboard[/]")
    console.print(f"[bold yellow]{'═' * 70}[/]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="center", style="bold", min_width=5)
    table.add_column("Model", style="cyan", min_width=25)
    table.add_column("PRESS Score", justify="right", style="bold green", min_width=12)
    table.add_column("Mean CDS", justify="right", min_width=10)
    table.add_column("Flip Rate", justify="right", min_width=10)
    table.add_column("T1 CDS", justify="right", min_width=8)
    table.add_column("T2 CDS", justify="right", min_width=8)
    table.add_column("T3 CDS", justify="right", min_width=8)

    for rank, result in enumerate(entries, 1):
        t1 = result.by_tier[0].mean_cds if len(result.by_tier) > 0 else 0
        t2 = result.by_tier[1].mean_cds if len(result.by_tier) > 1 else 0
        t3 = result.by_tier[2].mean_cds if len(result.by_tier) > 2 else 0

        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))

        table.add_row(
            medal,
            result.model_name,
            f"{result.press_score:.2f}",
            f"{result.overall_mean_cds:.4f}",
            f"{result.overall_flip_rate:.2%}",
            f"{t1:.4f}",
            f"{t2:.4f}",
            f"{t3:.4f}",
        )

    console.print(table)


# ── Chart generation ─────────────────────────────────────────────────────────


def plot_cds_by_tier(results: Sequence[ModelResult], output_dir: Path) -> Path:
    """Generate a grouped bar chart of mean CDS by pushback tier."""
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = [r.model_name for r in results]
    tiers = list(PushbackTier)
    x = np.arange(len(model_names))
    width = 0.25

    for i, tier in enumerate(tiers):
        values = []
        for r in results:
            tier_result = next((t for t in r.by_tier if t.tier == tier), None)
            values.append(tier_result.mean_cds if tier_result else 0)
        ax.bar(
            x + i * width,
            values,
            width,
            label=TIER_LABELS[tier],
            color=TIER_COLORS[tier],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Mean CDS", fontsize=12)
    ax.set_title("Calibration Degradation Score by Pushback Tier", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend(loc="upper left")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    ax.grid(axis="y", alpha=0.3)
    sns.despine()

    plt.tight_layout()
    path = output_dir / "cds_by_tier.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved tier chart to {path}")
    return path


def plot_cds_by_domain(results: Sequence[ModelResult], output_dir: Path) -> Path:
    """Generate a heatmap of mean CDS by model × domain."""
    domains = list(Domain)
    model_names = [r.model_name for r in results]
    data = np.zeros((len(model_names), len(domains)))

    for i, r in enumerate(results):
        for j, domain in enumerate(domains):
            dr = next((d for d in r.by_domain if d.domain == domain), None)
            data[i, j] = dr.mean_cds if dr else 0

    fig, ax = plt.subplots(figsize=(12, max(4, len(model_names) * 0.8)))
    sns.heatmap(
        data,
        xticklabels=[d.value.replace("_", " ").title() for d in domains],
        yticklabels=model_names,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        center=0,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Mean CDS"},
    )
    ax.set_title("CDS by Model × Domain", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = output_dir / "cds_by_domain_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved domain heatmap to {path}")
    return path


def plot_flip_rates(results: Sequence[ModelResult], output_dir: Path) -> Path:
    """Generate a bar chart of flip rates across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = [r.model_name for r in results]
    x = np.arange(len(model_names))
    width = 0.35

    correct_flips = [r.overall_correct_flip_rate * 100 for r in results]
    incorrect_flips = [r.overall_incorrect_flip_rate * 100 for r in results]

    ax.bar(x - width / 2, correct_flips, width, label="Correct → Wrong",
           color="#C44E52", edgecolor="white")
    ax.bar(x + width / 2, incorrect_flips, width, label="Wrong → Correct",
           color="#55A868", edgecolor="white")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Flip Rate (%)", fontsize=12)
    ax.set_title("Answer Flip Rates After Pushback", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="y", alpha=0.3)
    sns.despine()

    plt.tight_layout()
    path = output_dir / "flip_rates.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved flip rates chart to {path}")
    return path


def plot_press_scores(results: Sequence[ModelResult], output_dir: Path) -> Path:
    """Generate a horizontal bar chart of PRESS composite scores."""
    sorted_results = sorted(results, key=lambda r: r.press_score)

    fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_results) * 0.7)))

    model_names = [r.model_name for r in sorted_results]
    scores = [r.press_score for r in sorted_results]
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(sorted_results))]

    bars = ax.barh(model_names, scores, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("PRESS Score (0–100)", fontsize=12)
    ax.set_title("PRESS Benchmark — Composite Scores", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}", va="center", fontsize=10, fontweight="bold",
        )

    ax.grid(axis="x", alpha=0.3)
    sns.despine()
    plt.tight_layout()

    path = output_dir / "press_scores.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved PRESS scores chart to {path}")
    return path


def generate_all_charts(results: Sequence[ModelResult], output_dir: Path) -> list[Path]:
    """Generate all visualizations and return paths to saved files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts: list[Path] = []
    if len(results) > 0:
        charts.append(plot_press_scores(results, output_dir))
        charts.append(plot_cds_by_tier(results, output_dir))
        charts.append(plot_flip_rates(results, output_dir))
        if len(results) > 1:
            charts.append(plot_cds_by_domain(results, output_dir))
    return charts


# ── Leaderboard persistence ─────────────────────────────────────────────────


def build_leaderboard(results: Sequence[ModelResult]) -> list[LeaderboardEntry]:
    """Convert ModelResult list to LeaderboardEntry list."""
    entries: list[LeaderboardEntry] = []
    for r in results:
        t1 = r.by_tier[0].mean_cds if len(r.by_tier) > 0 else 0
        t2 = r.by_tier[1].mean_cds if len(r.by_tier) > 1 else 0
        t3 = r.by_tier[2].mean_cds if len(r.by_tier) > 2 else 0
        entries.append(
            LeaderboardEntry(
                model_name=r.model_name,
                press_score=r.press_score,
                mean_cds=r.overall_mean_cds,
                flip_rate=r.overall_flip_rate,
                tier1_cds=t1,
                tier2_cds=t2,
                tier3_cds=t3,
                evaluated_on=r.timestamp,
            )
        )
    return sorted(entries, key=lambda e: e.press_score, reverse=True)


def save_leaderboard(results: Sequence[ModelResult], output_dir: Path) -> Path:
    """Save leaderboard to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = build_leaderboard(results)
    path = output_dir / "leaderboard.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([e.model_dump() for e in entries], fh, indent=2, default=str)
    logger.info(f"Saved leaderboard to {path}")
    return path
