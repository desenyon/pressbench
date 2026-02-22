"""
Tests for dataset loading and validation.
"""

import pytest
from press.dataset.loader import load_dataset, validate_dataset
from press.models.data_models import Domain


def test_load_dataset():
    """Test that the dataset loads successfully."""
    manifest = load_dataset()
    assert manifest.total_questions >= 500, (
        f"Expected at least 500 questions, got {manifest.total_questions}"
    )
    assert len(manifest.questions) == manifest.total_questions


def test_all_domains_present():
    """Test that all 6 domains are represented."""
    manifest = load_dataset()
    for domain in Domain:
        assert domain.value in manifest.domains, f"Missing domain: {domain.value}"
        count = manifest.domains[domain.value]
        assert count >= 80, f"Domain {domain.value} has only {count} questions"


def test_no_duplicate_ids():
    """Test that all question IDs are unique."""
    manifest = load_dataset()
    ids = [q.id for q in manifest.questions]
    assert len(ids) == len(set(ids)), "Duplicate question IDs found"


def test_all_questions_have_answers():
    """Test that every question has a non-empty answer."""
    manifest = load_dataset()
    for q in manifest.questions:
        assert q.answer.strip(), f"Question {q.id} has an empty answer"


def test_validate_dataset():
    """Test the validation function."""
    manifest = load_dataset()
    issues = validate_dataset(manifest)
    # Allow warnings but should be minimal
    for issue in issues:
        print(f"  WARNING: {issue}")
