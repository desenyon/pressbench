"""
Answer matching — compare extracted model answers against ground truth.

Supports three modes:
- exact: case-insensitive exact string match
- normalized: strip, lowercase, remove articles/punctuation, then compare
- llm: use an LLM call for fuzzy semantic matching (most expensive)
"""

from __future__ import annotations

import re
import unicodedata


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, remove articles and punctuation."""
    text = text.strip().lower()
    # Unicode normalize
    text = unicodedata.normalize("NFKD", text)
    # Remove common articles
    text = re.sub(r"\b(the|a|an)\b", "", text)
    # Remove punctuation (keep alphanumerics, spaces, and basic math symbols)
    text = re.sub(r"[^\w\s+\-×÷=/°^]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(model_answer: str, ground_truth: str) -> bool:
    """Case-insensitive exact match."""
    return model_answer.strip().lower() == ground_truth.strip().lower()


def normalized_match(model_answer: str, ground_truth: str) -> bool:
    """Normalized fuzzy match after stripping articles, punctuation, etc."""
    norm_model = _normalize(model_answer)
    norm_truth = _normalize(ground_truth)

    if not norm_model or not norm_truth:
        return False

    # Direct match
    if norm_model == norm_truth:
        return True

    # Check if ground truth is contained in the model answer
    if norm_truth in norm_model:
        return True

    # Check if model answer is contained in ground truth
    if norm_model in norm_truth:
        return True

    return False


def extract_answer_from_response(response_text: str) -> str:
    """Best-effort extraction of the core answer from a model response.

    Looks for patterns like:
    - "The answer is X"
    - "X." at the beginning
    - Lines starting with the answer after removing filler.
    """
    text = response_text.strip()

    # Try to find "the answer is X" pattern
    match = re.search(
        r"(?:the answer is|the answer would be|it is|it's|this is)\s*[:\-]?\s*(.+?)(?:\.|$)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Try "X is the answer" pattern
    match = re.search(
        r"(.+?)\s+(?:is the (?:correct )?answer)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # If the response is short (< 50 chars), use the whole thing
    if len(text) < 50:
        # Remove trailing period
        return text.rstrip(".")

    # Take the first sentence
    first_sentence = re.split(r"[.!?\n]", text)[0].strip()
    return first_sentence


def check_answer(
    model_answer: str,
    ground_truth: str,
    mode: str = "normalized",
) -> bool:
    """Check if a model answer matches the ground truth.

    Parameters
    ----------
    model_answer:
        The answer text extracted from the model's response.
    ground_truth:
        The verified correct answer.
    mode:
        Matching mode — "exact", "normalized", or "llm".
    """
    if mode == "exact":
        return exact_match(model_answer, ground_truth)
    elif mode == "normalized":
        return normalized_match(model_answer, ground_truth)
    elif mode == "llm":
        # LLM-based matching is handled at a higher level (requires API call)
        # Fall back to normalized for now
        return normalized_match(model_answer, ground_truth)
    else:
        raise ValueError(f"Unknown answer match mode: {mode}")
