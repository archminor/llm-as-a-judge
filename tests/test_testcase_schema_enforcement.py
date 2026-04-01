"""Tests for testcase_loader schema enforcement (Task A)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from llm_judge.testcase_loader import (
    TestcaseLoadError,
    load_testcase_map,
    load_testcases,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.fixture
def valid_testcase() -> dict:
    return {
        "testcase_id": "tc-001",
        "task_type": "preprocessing",
        "input": {"text": "hello"},
    }


@pytest.fixture
def valid_testcase_full() -> dict:
    return {
        "testcase_id": "tc-002",
        "task_type": "report_generation",
        "input": {"text": "hello"},
        "metadata": {"difficulty": 3, "input_length_bucket": "M", "tags": ["foo"]},
        "constraints": {
            "required_points": ["point1"],
            "output_format": {"type": "json"},
        },
    }


class TestLoadTestcases:
    def test_valid_load(self, valid_testcase: dict, tmp_path: Path) -> None:
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [valid_testcase])
        result = load_testcases(path)
        assert len(result) == 1
        assert result[0].testcase_id == "tc-001"

    def test_valid_full(self, valid_testcase_full: dict, tmp_path: Path) -> None:
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [valid_testcase_full])
        result = load_testcases(path)
        assert len(result) == 1
        assert result[0].metadata.difficulty == 3

    def test_any_task_type_accepted(self, tmp_path: Path) -> None:
        """task_type is a free string, any non-empty value is accepted."""
        tc = {
            "testcase_id": "tc-free",
            "task_type": "custom_task_type",
            "input": {"text": "hello"},
        }
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [tc])
        result = load_testcases(path)
        assert len(result) == 1
        assert result[0].task_type == "custom_task_type"

    def test_extra_key_rejected(self, tmp_path: Path) -> None:
        bad = {
            "testcase_id": "tc-extra",
            "task_type": "preprocessing",
            "input": {"text": "hello"},
            "extra_field": "should_fail",
        }
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [bad])
        with pytest.raises(TestcaseLoadError) as exc_info:
            load_testcases(path)
        assert "tc-extra" in str(exc_info.value)

    def test_missing_required_field(self, tmp_path: Path) -> None:
        bad = {"testcase_id": "tc-missing"}  # missing task_type, input
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [bad])
        with pytest.raises(TestcaseLoadError) as exc_info:
            load_testcases(path)
        assert "tc-missing" in str(exc_info.value)

    def test_multiple_errors_collected(self, valid_testcase: dict, tmp_path: Path) -> None:
        bad1 = {"testcase_id": "bad1", "task_type": "ok", "input": {}, "extra_field": "x"}
        bad2 = {"testcase_id": "bad2"}
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [valid_testcase, bad1, bad2])
        with pytest.raises(TestcaseLoadError) as exc_info:
            load_testcases(path)
        err = exc_info.value
        assert len(err.errors) >= 2
        assert "line 2" in str(err)
        assert "line 3" in str(err)

    def test_line_numbers_correct(self, valid_testcase: dict, tmp_path: Path) -> None:
        bad = {"testcase_id": "bad-line3", "task_type": "ok", "input": {}, "extra_field": "x"}
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [valid_testcase, valid_testcase, bad])
        with pytest.raises(TestcaseLoadError) as exc_info:
            load_testcases(path)
        assert "line 3" in str(exc_info.value)
        assert "bad-line3" in str(exc_info.value)


class TestMessagesFormat:
    def test_messages_format_valid(self, tmp_path: Path) -> None:
        tc = {
            "testcase_id": "mt-001",
            "task_type": "report_qa",
            "input": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ]
            },
        }
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [tc])
        result = load_testcases(path)
        assert len(result) == 1
        assert result[0].has_messages is True

    def test_messages_empty_rejected(self, tmp_path: Path) -> None:
        tc = {
            "testcase_id": "mt-empty",
            "task_type": "report_qa",
            "input": {"messages": []},
        }
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [tc])
        with pytest.raises(TestcaseLoadError):
            load_testcases(path)

    def test_messages_invalid_role_rejected(self, tmp_path: Path) -> None:
        tc = {
            "testcase_id": "mt-bad-role",
            "task_type": "report_qa",
            "input": {
                "messages": [
                    {"role": "tool", "content": "not allowed"},
                ]
            },
        }
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [tc])
        with pytest.raises(TestcaseLoadError):
            load_testcases(path)

    def test_messages_mixed_with_legacy_keys_rejected(self, tmp_path: Path) -> None:
        tc = {
            "testcase_id": "mt-mixed",
            "task_type": "report_qa",
            "input": {
                "messages": [{"role": "user", "content": "hi"}],
                "question": "extra key",
            },
        }
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [tc])
        with pytest.raises(TestcaseLoadError):
            load_testcases(path)


class TestGenerationParams:
    def test_generation_params_max_tokens(self, tmp_path: Path) -> None:
        tc = {
            "testcase_id": "gp-001",
            "task_type": "report_generation",
            "input": {"text": "hello"},
            "generation_params": {"max_tokens": 512},
        }
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [tc])
        result = load_testcases(path)
        assert len(result) == 1
        assert result[0].generation_params == {"max_tokens": 512}

    def test_generation_params_empty(self, tmp_path: Path) -> None:
        tc = {
            "testcase_id": "gp-002",
            "task_type": "preprocessing",
            "input": {"text": "hello"},
            "generation_params": {},
        }
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [tc])
        result = load_testcases(path)
        assert len(result) == 1
        assert result[0].generation_params == {}

    def test_generation_params_arbitrary_keys_accepted(self, tmp_path: Path) -> None:
        """generation_params now accepts any keys (free-form dict)."""
        tc = {
            "testcase_id": "gp-free",
            "task_type": "preprocessing",
            "input": {"text": "hello"},
            "generation_params": {"temperature": 0.5, "top_p": 0.9},
        }
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [tc])
        result = load_testcases(path)
        assert len(result) == 1
        assert result[0].generation_params == {"temperature": 0.5, "top_p": 0.9}


class TestLoadTestcaseMap:
    def test_returns_dict(self, valid_testcase: dict, tmp_path: Path) -> None:
        path = tmp_path / "tc.jsonl"
        _write_jsonl(path, [valid_testcase])
        result = load_testcase_map(path)
        assert isinstance(result, dict)
        assert "tc-001" in result
        assert result["tc-001"].task_type == "preprocessing"
