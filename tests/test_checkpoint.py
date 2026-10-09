import json
import sqlite3

import pytest

from press.evaluation.checkpoint import Checkpoint, atomic_json


def test_atomic_export_keeps_previous_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "data.json"
    atomic_json(path, {"old": True})

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr("press.evaluation.checkpoint.os.replace", fail)
    with pytest.raises(OSError, match="disk full"):
        atomic_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]


def test_corrupt_database_fails_without_overwrite(tmp_path):
    identity = {"model_id": "test"}
    store = Checkpoint(tmp_path, identity)
    store.path.write_bytes(b"corrupt database")
    with pytest.raises(sqlite3.DatabaseError), Checkpoint(tmp_path, identity, resume=True):
        pass
    assert store.path.read_bytes() == b"corrupt database"
    assert not store.lock.exists()


def test_missing_checkpoint_does_not_create_database(tmp_path):
    store = Checkpoint(tmp_path, {"model_id": "test"}, resume=True)
    with pytest.raises(ValueError, match="No checkpoint"), store:
        pass
    assert not store.path.exists()
    assert not store.lock.exists()
