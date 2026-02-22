"""
Dataset loader — reads question JSON files and produces a DatasetManifest.
"""

from __future__ import annotations

import json
from pathlib import Path

from press.models.data_models import DatasetManifest, Domain, Question


def load_domain_file(path: Path) -> list[Question]:
    """Load questions from a single domain JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [Question(**item) for item in raw]


def load_dataset(dataset_dir: Path | str | None = None) -> DatasetManifest:
    """Load the full PRESS dataset from a directory of per-domain JSON files.

    Parameters
    ----------
    dataset_dir:
        Path containing ``science.json``, ``history.json``, etc.
        Defaults to the built-in ``press/dataset/questions`` directory.
    """
    if dataset_dir is None:
        dataset_dir = Path(__file__).resolve().parent / "questions"
    else:
        dataset_dir = Path(dataset_dir)

    domain_file_map = {
        Domain.SCIENCE: "science.json",
        Domain.HISTORY: "history.json",
        Domain.MATHEMATICS: "mathematics.json",
        Domain.GEOGRAPHY: "geography.json",
        Domain.LAW_POLICY: "law_policy.json",
        Domain.TECHNOLOGY: "technology.json",
    }

    all_questions: list[Question] = []
    domain_counts: dict[str, int] = {}

    for domain, filename in domain_file_map.items():
        filepath = dataset_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Missing dataset file for domain {domain.value}: {filepath}"
            )
        questions = load_domain_file(filepath)
        all_questions.extend(questions)
        domain_counts[domain.value] = len(questions)

    manifest = DatasetManifest(
        total_questions=len(all_questions),
        domains=domain_counts,
        questions=all_questions,
    )
    return manifest


def validate_dataset(manifest: DatasetManifest) -> list[str]:
    """Return a list of validation warnings/errors (empty = all good)."""
    issues: list[str] = []

    # Check total count
    if manifest.total_questions < 500:
        issues.append(
            f"Dataset has {manifest.total_questions} questions, target is 500."
        )

    # Check for duplicate IDs
    ids = [q.id for q in manifest.questions]
    if len(ids) != len(set(ids)):
        dupes = [qid for qid in ids if ids.count(qid) > 1]
        issues.append(f"Duplicate question IDs found: {set(dupes)}")

    # Check domain distribution
    for domain in Domain:
        count = manifest.domains.get(domain.value, 0)
        if count < 80:
            issues.append(
                f"Domain {domain.value} has only {count} questions (target: ~83)."
            )

    # Check that all questions have non-empty answers
    for q in manifest.questions:
        if not q.answer.strip():
            issues.append(f"Question {q.id} has an empty answer.")

    return issues
