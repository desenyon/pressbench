"""
Tests for prompts module.
"""

from press.evaluation.prompts import (
    SYSTEM_PROMPT,
    build_initial_messages,
    build_pushback_messages,
)
from press.models.data_models import PUSHBACK_SCRIPTS, PushbackTier


class TestPrompts:
    def test_initial_messages_structure(self):
        messages = build_initial_messages("What is 2+2?")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is 2+2?"

    def test_pushback_messages_structure(self):
        messages = build_pushback_messages(
            "What is 2+2?",
            "The answer is 4.",
            PushbackTier.SOFT,
        )
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"

    def test_pushback_tiers(self):
        for tier in PushbackTier:
            messages = build_pushback_messages("Q?", "A.", tier)
            assert messages[3]["content"] == PUSHBACK_SCRIPTS[tier]

    def test_system_prompt_present(self):
        messages = build_initial_messages("Q?")
        assert SYSTEM_PROMPT in messages[0]["content"]
