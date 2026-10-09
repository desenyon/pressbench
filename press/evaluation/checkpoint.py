"""Durable phase checkpoints and atomic exports; no provider dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from press import __version__
from press.config import Settings
from press.evaluation.prompts import SYSTEM_PROMPT
from press.models.data_models import PUSHBACK_SCRIPTS, DatasetManifest, EvalInstance


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_stem(model_id: str) -> str:
    """Readable, bounded, collision-resistant filenames, including for namespaced IDs."""
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_id)[:80].strip("_") or "model"
    return f"{name}-{_digest(model_id)[:12]}"


def run_identity(model_id: str, dataset: DatasetManifest, settings: Settings) -> dict:
    """Only explicitly allowlisted reproducibility fields; never serialize API keys."""
    return {
        "schema_version": 2,
        "press_version": __version__,
        "model_id": model_id,
        "dataset_version": dataset.version,
        "dataset_sha256": _digest([q.model_dump(mode="json") for q in dataset.questions]),
        "question_ids": [q.id for q in dataset.questions],
        "question_count": len(dataset.questions),
        "prompt_sha256": _digest({"system": SYSTEM_PROMPT, "pushback": PUSHBACK_SCRIPTS}),
        "settings": {
            key: getattr(settings, key)
            for key in (
                "runs_per_instance",
                "temperature",
                "max_tokens",
                "request_logprobs",
                "top_logprobs",
                "use_logprobs",
                "answer_match_mode",
            )
        },
    }


def atomic_json(path: Path, value: Any) -> None:
    """Replace a complete JSON document, keeping the previous export on write failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Checkpoint:
    """Single writer SQLite store. Every save commits one phase independently."""

    def __init__(self, output_dir: Path, identity: dict, *, resume: bool = False):
        self.output_dir = output_dir
        self.identity = identity
        self.stem = artifact_stem(identity["model_id"])
        self.path = output_dir / f"{self.stem}_checkpoint.sqlite3"
        self.lock = self.path.with_suffix(".lock")
        self.connection: sqlite3.Connection | None = None
        self.resume = resume

    def __enter__(self) -> Checkpoint:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ValueError(
                f"Run is locked: {self.lock}. If its process has stopped, remove this lock."
            ) from exc
        os.close(fd)
        try:
            exists = self.path.exists()
            exports = [
                self.output_dir / f"{self.stem}_{suffix}.json"
                for suffix in ("instances", "result", "manifest")
            ]
            if not self.resume and (exists or any(p.exists() for p in exports)):
                raise ValueError(
                    "Run artifacts already exist; use --resume or a new output directory."
                )
            if self.resume and not exists:
                raise ValueError("No checkpoint to resume; omit --resume to start a new run.")
            self.connection = sqlite3.connect(self.path)
            if exists:
                row = self.connection.execute("SELECT identity FROM metadata").fetchone()
                if row is None or json.loads(row[0]) != self.identity:
                    raise ValueError(
                        "Checkpoint does not match model, dataset, prompts or settings."
                    )
            else:
                with self.connection:
                    self.connection.execute("CREATE TABLE metadata (identity TEXT NOT NULL)")
                    self.connection.execute(
                        "INSERT INTO metadata VALUES (?)", (json.dumps(self.identity),)
                    )
                    self.connection.execute(
                        "CREATE TABLE instances (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
                    )
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args: Any) -> None:
        if self.connection is not None:
            self.connection.close()
        self.lock.unlink(missing_ok=True)

    def load(self) -> dict[str, EvalInstance]:
        assert self.connection is not None
        records = {}
        for instance_id, payload in self.connection.execute("SELECT id, payload FROM instances"):
            instance = EvalInstance.model_validate_json(payload)
            if instance_id != instance.instance_id:
                raise ValueError("Checkpoint instance identity is corrupt.")
            if instance.status == "completed" and not instance.completed:
                raise ValueError("Checkpoint completed instance is missing scores or responses.")
            before = instance.response_before
            if before is not None and (before.confidence < 0 or before.is_correct is None):
                raise ValueError("Checkpoint contains an unscored initial response.")
            records[instance_id] = instance
        return records

    def save(self, instance: EvalInstance) -> None:
        assert self.connection is not None
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO instances VALUES (?, ?)",
                (instance.instance_id, instance.model_dump_json()),
            )
