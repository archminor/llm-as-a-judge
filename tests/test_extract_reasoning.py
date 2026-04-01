"""Tests for _extract_reasoning helper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_judge.stages.inference import _extract_reasoning


class TestExtractReasoning:
    """Unit tests for reasoning extraction from API responses."""

    def test_reasoning_content_attribute(self) -> None:
        """reasoning_content on the message object is preferred."""
        msg = SimpleNamespace(reasoning_content="step-by-step thinking")
        reasoning, text = _extract_reasoning(msg, "final answer")
        assert reasoning == "step-by-step thinking"
        assert text == "final answer"

    def test_model_extra_fallback(self) -> None:
        """Falls back to model_extra when attribute is absent."""
        msg = SimpleNamespace(model_extra={"reasoning_content": "via extra"})
        reasoning, text = _extract_reasoning(msg, "answer")
        assert reasoning == "via extra"
        assert text == "answer"

    def test_model_extra_none(self) -> None:
        """model_extra=None should not raise."""
        msg = SimpleNamespace(model_extra=None)
        reasoning, text = _extract_reasoning(msg, "plain text")
        assert reasoning is None
        assert text == "plain text"

    def test_think_tag_single_block(self) -> None:
        """Single <think> block is extracted and stripped from text."""
        raw = "<think>Let me reason</think>The answer is 42"
        msg = SimpleNamespace(model_extra=None)
        reasoning, text = _extract_reasoning(msg, raw)
        assert reasoning == "Let me reason"
        assert text == "The answer is 42"
        assert "<think>" not in text

    def test_think_tag_multiple_blocks(self) -> None:
        """Multiple <think> blocks are all captured."""
        raw = "<think>First thought</think>middle<think>Second thought</think>end"
        msg = SimpleNamespace(model_extra=None)
        reasoning, text = _extract_reasoning(msg, raw)
        assert "First thought" in reasoning
        assert "Second thought" in reasoning
        assert "<think>" not in text
        assert "middle" in text
        assert "end" in text

    def test_think_tag_multiline(self) -> None:
        """<think> blocks spanning multiple lines are handled."""
        raw = "<think>\nLine 1\nLine 2\n</think>\nAnswer"
        msg = SimpleNamespace(model_extra=None)
        reasoning, text = _extract_reasoning(msg, raw)
        assert "Line 1" in reasoning
        assert "Line 2" in reasoning
        assert text == "Answer"

    def test_reasoning_content_takes_priority_over_think_tags(self) -> None:
        """When reasoning_content exists, <think> tags are left in text."""
        raw = "<think>embedded</think>answer"
        msg = SimpleNamespace(reasoning_content="structured reasoning")
        reasoning, text = _extract_reasoning(msg, raw)
        assert reasoning == "structured reasoning"
        # <think> tags remain untouched when structured field is used
        assert "<think>" in text

    def test_no_reasoning_anywhere(self) -> None:
        """No reasoning source returns None and unchanged text."""
        msg = SimpleNamespace(model_extra=None)
        reasoning, text = _extract_reasoning(msg, "just an answer")
        assert reasoning is None
        assert text == "just an answer"

    def test_empty_string_reasoning_content_falls_through(self) -> None:
        """Empty string reasoning_content is treated as None."""
        msg = SimpleNamespace(reasoning_content="", model_extra=None)
        reasoning, text = _extract_reasoning(msg, "plain text")
        # Empty string → reasoning or None → None
        assert reasoning is None
        assert text == "plain text"

    def test_empty_think_tags(self) -> None:
        """<think></think> with no content should not produce reasoning."""
        raw = "<think></think>answer"
        msg = SimpleNamespace(model_extra=None)
        reasoning, text = _extract_reasoning(msg, raw)
        # Empty block after strip → empty string → filtered by `or None`
        assert reasoning is None
        assert text == "answer"
