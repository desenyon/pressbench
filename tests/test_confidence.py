"""
Tests for the confidence classifier.
"""

import pytest
from press.calibration.confidence_classifier import (
    confidence_from_logprob,
    confidence_from_text,
    extract_confidence,
)


class TestLogprobConfidence:
    def test_high_logprob(self):
        result = confidence_from_logprob(-0.01)
        assert result.score > 0.98
        assert result.method == "logprob"

    def test_low_logprob(self):
        result = confidence_from_logprob(-5.0)
        assert result.score < 0.01
        assert result.method == "logprob"

    def test_medium_logprob(self):
        result = confidence_from_logprob(-0.5)
        assert 0.5 < result.score < 0.7

    def test_zero_logprob(self):
        result = confidence_from_logprob(0.0)
        assert result.score == 1.0


class TestLinguisticConfidence:
    def test_high_confidence_text(self):
        result = confidence_from_text("I'm absolutely certain the answer is 42.")
        assert result.score > 0.8
        assert result.method == "linguistic"

    def test_low_confidence_text(self):
        result = confidence_from_text("I could be wrong, but maybe it's 42.")
        assert result.score < 0.5

    def test_sycophantic_text(self):
        result = confidence_from_text(
            "You're right, I apologize for the error. The correct answer is 43."
        )
        assert result.score < 0.3

    def test_holding_ground(self):
        result = confidence_from_text(
            "I'm confident in my answer. The correct value is definitely 42."
        )
        assert result.score > 0.8

    def test_neutral_text(self):
        result = confidence_from_text("The answer is 42.")
        assert 0.6 < result.score < 0.9

    def test_mixed_signals(self):
        result = confidence_from_text(
            "I think the answer might be 42, though I'm fairly certain."
        )
        assert 0.3 < result.score < 0.8


class TestExtractConfidence:
    def test_prefer_logprob(self):
        result = extract_confidence("I'm uncertain", logprob=-0.01, prefer_logprob=True)
        assert result.method == "logprob"
        assert result.score > 0.98

    def test_fallback_to_linguistic(self):
        result = extract_confidence("I'm uncertain", logprob=None, prefer_logprob=True)
        assert result.method == "linguistic"

    def test_force_linguistic(self):
        result = extract_confidence("I'm certain", logprob=-0.01, prefer_logprob=False)
        assert result.method == "linguistic"
