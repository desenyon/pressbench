"""
Tests for answer matching.
"""

import pytest

from press.utils.answer_matching import (
    check_answer,
    exact_match,
    extract_answer_from_response,
    normalized_match,
)


class TestExactMatch:
    def test_identical(self):
        assert exact_match("Paris", "Paris")

    def test_case_insensitive(self):
        assert exact_match("paris", "Paris")

    def test_no_match(self):
        assert not exact_match("London", "Paris")


class TestNormalizedMatch:
    def test_articles_stripped(self):
        assert normalized_match("the Nile", "Nile")

    def test_punctuation_stripped(self):
        assert normalized_match("H2O.", "H2O")

    def test_contained_answer(self):
        assert normalized_match(
            "The answer is definitely Au",
            "Au",
        )

    def test_no_match(self):
        assert not normalized_match("Silver", "Gold")


class TestExtractAnswer:
    def test_answer_is_pattern(self):
        text = "The answer is 42. This is well established."
        assert "42" in extract_answer_from_response(text)

    def test_short_response(self):
        text = "Paris"
        assert extract_answer_from_response(text) == "Paris"

    def test_sentence_extraction(self):
        text = "Mercury is the closest planet to the Sun. It has a very thin atmosphere."
        answer = extract_answer_from_response(text)
        assert "Mercury" in answer


class TestCheckAnswer:
    def test_exact_mode(self):
        assert check_answer("Paris", "Paris", mode="exact")
        assert check_answer("paris", "PARIS", mode="exact") is not False  # case insensitive

    def test_normalized_mode(self):
        assert check_answer("the Nile River", "Nile", mode="normalized")

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            check_answer("x", "y", mode="invalid")


@pytest.mark.parametrize(
    "answer, truth",
    [
        ("Australia", "Au"),
        ("18420", "42"),
        ("York", "New York"),
        ("314", "3.14"),
        ("-1", "1"),
        ("1.5", "1"),
    ],
)
def test_partial_substrings_are_not_correct(answer, truth):
    assert not normalized_match(answer, truth)


def test_decimal_answer_extraction():
    assert extract_answer_from_response("The answer is 3.14. This is an approximation.") == "3.14"


def test_unimplemented_llm_judge_is_not_silent_fallback():
    with pytest.raises(ValueError):
        check_answer("a", "b", mode="llm")
