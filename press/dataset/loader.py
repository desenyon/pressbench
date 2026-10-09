"""
Dataset loader — reads question JSON files and produces a DatasetManifest.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from press.models.data_models import DatasetManifest, Domain, Question


def load_domain_file(path: Path) -> list[Question]:
    """Load questions from a single domain JSON file."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise ValueError(f"{path}: expected a JSON array of question objects")
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
            raise FileNotFoundError(f"Missing dataset file for domain {domain.value}: {filepath}")
        questions = load_domain_file(filepath)
        for question in questions:
            if question.domain != domain:
                raise ValueError(f"{filepath}: {question.id} has domain {question.domain.value}")
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

    issues.extend(validate_structure(manifest))

    # Check total count
    if manifest.total_questions < 500:
        issues.append(f"Dataset has {manifest.total_questions} questions, target is 500.")

    # Check domain distribution
    for domain in Domain:
        count = manifest.domains.get(domain.value, 0)
        if count < 80:
            issues.append(f"Domain {domain.value} has only {count} questions (target: ~83).")

    return issues


def validate_structure(manifest: DatasetManifest) -> list[str]:
    """Integrity checks that also apply to small custom datasets and subsets."""
    issues = []
    if not manifest.questions:
        issues.append("Dataset contains no questions.")
    if manifest.total_questions != len(manifest.questions):
        issues.append("Dataset total_questions does not match question count.")
    counts = dict(Counter(q.domain.value for q in manifest.questions))
    if counts != {k: v for k, v in manifest.domains.items() if v}:
        issues.append("Dataset domain counts do not match question contents.")
    ids = Counter(q.id for q in manifest.questions)
    for qid, count in ids.items():
        if count > 1:
            issues.append(f"Duplicate question ID: {qid}")
    for q in manifest.questions:
        if not q.id.strip() or not q.question.strip() or not q.answer.strip():
            issues.append(f"Question {q.id!r} has an empty id, question or answer.")
        if q.difficulty not in {"easy", "medium", "hard"}:
            issues.append(f"Question {q.id} has invalid difficulty: {q.difficulty}")
    return issues


def select_dataset(
    manifest: DatasetManifest,
    domains: tuple[str, ...] = (),
    limit: int | None = None,
) -> DatasetManifest:
    """Validate the source before selecting a stable, ID-sorted subset."""
    issues = validate_structure(manifest)
    if issues:
        raise ValueError("Invalid dataset: " + "; ".join(issues))
    if limit is not None and limit < 1:
        raise ValueError("Dataset limit must be positive.")
    questions = sorted(
        (q for q in manifest.questions if not domains or q.domain.value in domains),
        key=lambda q: q.id,
    )
    if limit is not None:
        questions = questions[:limit]
    if not questions:
        raise ValueError("Dataset selection contains no questions.")
    return DatasetManifest(
        version=manifest.version,
        total_questions=len(questions),
        domains=dict(Counter(q.domain.value for q in questions)),
        questions=questions,
    )
