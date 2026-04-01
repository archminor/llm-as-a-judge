"""Tests for multiturn (input.messages) prompt generation."""

from __future__ import annotations

from llm_judge.models import Testcase
from llm_judge.prompts import build_inference_prompt


def _make_messages_testcase() -> Testcase:
    return Testcase(
        testcase_id="mt-001",
        task_type="report_qa",
        input={
            "messages": [
                {"role": "system", "content": "あなたはアナリストです。"},
                {"role": "user", "content": "質問1"},
                {"role": "assistant", "content": "回答1"},
                {"role": "user", "content": "質問2"},
            ]
        },
    )


def _make_legacy_testcase() -> Testcase:
    return Testcase(
        testcase_id="legacy-001",
        task_type="report_qa",
        input={"question": "テスト質問", "context": "テストコンテキスト"},
    )


class TestMultiturnPrompt:
    def test_messages_format_returns_messages_directly(self) -> None:
        tc = _make_messages_testcase()
        result = build_inference_prompt(tc)
        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "あなたはアナリストです。"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "質問1"
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "回答1"
        assert result[3]["role"] == "user"
        assert result[3]["content"] == "質問2"

    def test_messages_result_is_deep_copied(self) -> None:
        tc = _make_messages_testcase()
        result = build_inference_prompt(tc)
        # Mutate result
        result[0]["content"] = "MUTATED"
        result.append({"role": "user", "content": "extra"})
        # Original should be unchanged
        assert tc.input["messages"][0]["content"] == "あなたはアナリストです。"
        assert len(tc.input["messages"]) == 4

    def test_legacy_format_unchanged(self) -> None:
        tc = _make_legacy_testcase()
        result = build_inference_prompt(tc)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
