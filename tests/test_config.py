import pytest
from pydantic import ValidationError

from press.config import Settings
from press.dataset.loader import load_dataset, select_dataset, validate_structure


def test_default_dataset_is_installed_package_data():
    settings = Settings(_env_file=None)
    assert load_dataset(settings.dataset_path).total_questions == 500


@pytest.mark.parametrize(
    "key,value",
    [
        ("concurrency", 0),
        ("runs_per_instance", 11),
        ("temperature", float("nan")),
        ("max_tokens", 0),
        ("request_timeout", float("inf")),
        ("answer_match_mode", "llm"),
    ],
)
def test_configuration_rejects_invalid_assignment(key, value):
    settings = Settings(_env_file=None)
    with pytest.raises(ValidationError):
        setattr(settings, key, value)


def test_subset_is_deterministic_and_valid():
    dataset = load_dataset()
    selected = select_dataset(dataset, ("science",), 3)
    assert selected.total_questions == 3
    assert selected.domains == {"science": 3}
    assert [q.id for q in selected.questions] == ["SCI-001", "SCI-002", "SCI-003"]
    assert validate_structure(selected) == []
    assert len(dataset.questions) == 500


def test_duplicates_rejected_before_filtering():
    dataset = load_dataset()
    dataset.questions[-1].id = dataset.questions[-2].id
    with pytest.raises(ValueError, match="Duplicate"):
        select_dataset(dataset, ("science",), 1)
