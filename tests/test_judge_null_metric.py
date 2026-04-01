"""Tests for Judge per_metric normalization, null handling, and drop warnings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_judge.llm_client import CompletionResult
from llm_judge.models import (
    InferenceRecord,
    ModelInfo,
    OutputInfo,
    RunConfig,
    StatusInfo,
    Testcase,
)
from llm_judge.prompts import build_absolute_judge_prompt, build_pairwise_judge_prompt
from llm_judge.stages.judge import _normalize_per_metric


# ── _normalize_per_metric ────────────────────────────────


class TestNormalizePerMetric:
    def test_int_values_kept(self):
        normalized, dropped = _normalize_per_metric({"accuracy": 5, "clarity": 3})
        assert normalized == {"accuracy": 5, "clarity": 3}
        assert dropped == []

    def test_float_values_rounded(self):
        normalized, dropped = _normalize_per_metric({"accuracy": 4.6, "clarity": 2.3})
        assert normalized == {"accuracy": 5, "clarity": 2}
        assert dropped == []

    def test_none_values_dropped(self):
        normalized, dropped = _normalize_per_metric({"accuracy": 5, "citation_quality": None})
        assert normalized == {"accuracy": 5}
        assert len(dropped) == 1
        assert "citation_quality" in dropped[0]
        assert "null" in dropped[0]

    def test_string_values_dropped(self):
        normalized, dropped = _normalize_per_metric({"accuracy": 5, "citation_quality": "N/A"})
        assert normalized == {"accuracy": 5}
        assert len(dropped) == 1
        assert "citation_quality" in dropped[0]
        assert "non-numeric" in dropped[0]

    def test_dict_values_dropped_when_not_allowed(self):
        normalized, dropped = _normalize_per_metric(
            {"accuracy": {"A": 5, "B": 3}}, allow_dict_values=False,
        )
        assert normalized == {}
        assert len(dropped) == 1
        assert "absolute mode" in dropped[0]

    def test_dict_values_averaged_when_allowed(self):
        normalized, dropped = _normalize_per_metric(
            {"accuracy": {"A": 5, "B": 3}}, allow_dict_values=True,
        )
        assert normalized == {"accuracy": 4}
        assert dropped == []

    def test_dict_with_non_numeric_values_dropped(self):
        normalized, dropped = _normalize_per_metric(
            {"accuracy": {"A": "good", "B": "bad"}}, allow_dict_values=True,
        )
        assert normalized == {}
        assert len(dropped) == 1
        assert "no numeric" in dropped[0]

    def test_empty_dict(self):
        normalized, dropped = _normalize_per_metric({})
        assert normalized == {}
        assert dropped == []

    def test_all_none(self):
        normalized, dropped = _normalize_per_metric({"a": None, "b": None})
        assert normalized == {}
        assert len(dropped) == 2

    def test_mixed_types(self):
        normalized, dropped = _normalize_per_metric({
            "accuracy": 5,
            "clarity": 3.7,
            "citation_quality": None,
            "relevance": "N/A",
            "coherence": [1, 2],
        })
        assert normalized == {"accuracy": 5, "clarity": 4}
        assert len(dropped) == 3


# ── _judge_absolute with null per_metric ─────────────────


def _make_testcase() -> Testcase:
    return Testcase(
        testcase_id="tc-001",
        task_type="report_generation",
        input={"raw_text": "test input"},
    )


def _make_inference() -> InferenceRecord:
    return InferenceRecord(
        run_id="test-run",
        testcase_id="tc-001",
        candidate_id="c1",
        model=ModelInfo(vendor="openai", model_id="gpt-4o"),
        output=OutputInfo(text="test output"),
        status=StatusInfo(ok=True),
    )


def _make_config(mode: str = "absolute") -> RunConfig:
    candidates = [
        {"candidate_id": "c1", "vendor": "openai", "model_id": "gpt-4o"},
    ]
    if mode in ("pairwise", "hybrid"):
        candidates.append(
            {"candidate_id": "c2", "vendor": "openai", "model_id": "gpt-4o"},
        )
    return RunConfig.model_validate({
        "run_id": "test-run",
        "dataset": {"testcases_path": "data/testcases.jsonl"},
        "candidates": candidates,
        "judges": [{
            "judge_id": "j1",
            "vendor": "openai",
            "model_id": "gpt-4o",
            "rubric_version": "v1",
        }],
        "protocol": {
            "evaluation_mode": mode,
            "metrics": ["accuracy", "citation_quality"],
            "aggregation": {"method": "mean"},
        },
    })


class TestAbsoluteJudgeNullMetric:
    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_null_metric_does_not_crash(self, mock_completion):
        from llm_judge.stages.judge import _judge_absolute

        mock_completion.return_value = CompletionResult(
            text=json.dumps({
                "per_metric": {"accuracy": 5, "citation_quality": None},
                "overall_score": 4,
                "rationale": "Good",
                "critical_issue": False,
            }),
        )

        cfg = _make_config("absolute")
        tc = _make_testcase()
        inf = _make_inference()

        rec = _judge_absolute(
            cfg=cfg, tc=tc, inf=inf, idx=0,
            judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        # Should not crash; citation_quality should be dropped
        assert rec.scores.per_metric["accuracy"] == 5
        assert "citation_quality" not in rec.scores.per_metric
        assert rec.overall_rationale == "Good"

    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_string_metric_does_not_crash(self, mock_completion):
        from llm_judge.stages.judge import _judge_absolute

        mock_completion.return_value = CompletionResult(
            text=json.dumps({
                "per_metric": {"accuracy": 5, "citation_quality": "N/A"},
                "overall_score": 4,
                "rationale": "Good",
                "critical_issue": False,
            }),
        )

        cfg = _make_config("absolute")
        tc = _make_testcase()
        inf = _make_inference()

        rec = _judge_absolute(
            cfg=cfg, tc=tc, inf=inf, idx=0,
            judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.scores.per_metric == {"accuracy": 5}

    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_normal_metric_preserved(self, mock_completion):
        from llm_judge.stages.judge import _judge_absolute

        mock_completion.return_value = CompletionResult(
            text=json.dumps({
                "per_metric": {"accuracy": 5, "citation_quality": 3},
                "overall_score": 4,
                "rationale": "Good",
                "critical_issue": False,
            }),
        )

        cfg = _make_config("absolute")
        tc = _make_testcase()
        inf = _make_inference()

        rec = _judge_absolute(
            cfg=cfg, tc=tc, inf=inf, idx=0,
            judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.scores.per_metric == {"accuracy": 5, "citation_quality": 3}

    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_path_object_inference_ref_is_serialized_to_string(self, mock_completion):
        from llm_judge.stages.judge import _judge_absolute

        mock_completion.return_value = CompletionResult(
            text=json.dumps({
                "per_metric": {"accuracy": 5, "citation_quality": 3},
                "overall_score": 4,
                "rationale": "Good",
                "critical_issue": False,
            }),
        )

        cfg = _make_config("absolute")
        tc = _make_testcase()
        inf = _make_inference()

        rec = _judge_absolute(
            cfg=cfg, tc=tc, inf=inf, idx=0,
            judge_ref=cfg.judges[0], inf_path=Path("data/inf.jsonl"),
            client=MagicMock(),
        )

        assert rec.targets[0].inference_ref.path == "data/inf.jsonl"


# ── _judge_pairwise with null per_metric ─────────────────


class TestPairwiseJudgeNullMetric:
    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_null_metric_does_not_crash(self, mock_completion):
        from llm_judge.stages.judge import _judge_pairwise

        mock_completion.return_value = CompletionResult(
            text=json.dumps({
                "per_metric": {"accuracy": 5, "citation_quality": None},
                "overall_winner": "A",
                "critical_issue_a": False,
                "critical_issue_b": False,
                "rationale": "A is better",
            }),
        )

        cfg = _make_config("pairwise")
        tc = _make_testcase()
        inf_a = _make_inference()
        inf_b = InferenceRecord(
            run_id="test-run",
            testcase_id="tc-001",
            candidate_id="c2",
            model=ModelInfo(vendor="openai", model_id="gpt-4o"),
            output=OutputInfo(text="test output 2"),
            status=StatusInfo(ok=True),
        )

        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.scores.per_metric["accuracy"] == 5
        assert "citation_quality" not in rec.scores.per_metric

    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_dict_metric_still_works(self, mock_completion):
        from llm_judge.stages.judge import _judge_pairwise

        mock_completion.return_value = CompletionResult(
            text=json.dumps({
                "per_metric": {"accuracy": {"A": 5, "B": 3}},
                "overall_winner": "A",
                "critical_issue_a": False,
                "critical_issue_b": False,
                "rationale": "A is better",
            }),
        )

        cfg = _make_config("pairwise")
        tc = _make_testcase()
        inf_a = _make_inference()
        inf_b = InferenceRecord(
            run_id="test-run",
            testcase_id="tc-001",
            candidate_id="c2",
            model=ModelInfo(vendor="openai", model_id="gpt-4o"),
            output=OutputInfo(text="test output 2"),
            status=StatusInfo(ok=True),
        )

        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.scores.per_metric["accuracy"] == 4  # avg(5, 3)

    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_path_object_inference_refs_are_serialized_to_strings(self, mock_completion):
        from llm_judge.stages.judge import _judge_pairwise

        mock_completion.return_value = CompletionResult(
            text=json.dumps({
                "per_metric": {"accuracy": {"A": 5, "B": 3}},
                "overall_winner": "A",
                "critical_issue_a": False,
                "critical_issue_b": False,
                "rationale": "A is better",
            }),
        )

        cfg = _make_config("pairwise")
        tc = _make_testcase()
        inf_a = _make_inference()
        inf_b = InferenceRecord(
            run_id="test-run",
            testcase_id="tc-001",
            candidate_id="c2",
            model=ModelInfo(vendor="openai", model_id="gpt-4o"),
            output=OutputInfo(text="test output 2"),
            status=StatusInfo(ok=True),
        )

        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path=Path("data/inf.jsonl"),
            client=MagicMock(),
        )

        assert rec.targets[0].inference_ref.path == "data/inf.jsonl"
        assert rec.targets[1].inference_ref.path == "data/inf.jsonl"


# ── Prompt null prohibition ──────────────────────────────


class TestPromptNullProhibition:
    def test_absolute_prompt_prohibits_null(self):
        tc = _make_testcase()
        messages = build_absolute_judge_prompt(
            testcase=tc,
            output_text="test",
            candidate_label="c1",
            metrics=["accuracy", "citation_quality"],
            rubric_version="v1",
        )
        system_msg = messages[0]["content"]
        assert "null" in system_msg.lower() or "do not use null" in system_msg.lower()
        assert "omit" in system_msg.lower()

    def test_pairwise_prompt_prohibits_null(self):
        tc = _make_testcase()
        messages = build_pairwise_judge_prompt(
            testcase=tc,
            output_a="output A",
            output_b="output B",
            label_a="A",
            label_b="B",
            metrics=["accuracy", "citation_quality"],
            rubric_version="v1",
        )
        system_msg = messages[0]["content"]
        assert "null" in system_msg.lower() or "do not use null" in system_msg.lower()
        assert "omit" in system_msg.lower()


# ── Warning log on metric drop ───────────────────────────


class TestMetricDropWarning:
    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_absolute_null_metric_emits_warning(self, mock_completion, caplog):
        from llm_judge.stages.judge import _judge_absolute

        mock_completion.return_value = CompletionResult(
            text=json.dumps({
                "per_metric": {"accuracy": 5, "citation_quality": None},
                "overall_score": 4,
                "rationale": "Good",
                "critical_issue": False,
            }),
        )

        cfg = _make_config("absolute")
        tc = _make_testcase()
        inf = _make_inference()

        with caplog.at_level(logging.WARNING, logger="llm_judge.stages.judge"):
            _judge_absolute(
                cfg=cfg, tc=tc, inf=inf, idx=0,
                judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
                client=MagicMock(),
            )

        assert any("citation_quality" in msg for msg in caplog.messages)
        assert any("absolute" in msg for msg in caplog.messages)

    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_pairwise_null_metric_emits_warning(self, mock_completion, caplog):
        from llm_judge.stages.judge import _judge_pairwise

        mock_completion.return_value = CompletionResult(
            text=json.dumps({
                "per_metric": {"accuracy": 5, "citation_quality": None},
                "overall_winner": "A",
                "critical_issue_a": False,
                "critical_issue_b": False,
                "rationale": "A is better",
            }),
        )

        cfg = _make_config("pairwise")
        tc = _make_testcase()
        inf_a = _make_inference()
        inf_b = InferenceRecord(
            run_id="test-run",
            testcase_id="tc-001",
            candidate_id="c2",
            model=ModelInfo(vendor="openai", model_id="gpt-4o"),
            output=OutputInfo(text="test output 2"),
            status=StatusInfo(ok=True),
        )

        with caplog.at_level(logging.WARNING, logger="llm_judge.stages.judge"):
            _judge_pairwise(
                cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
                idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
                presented_swap=False, inf_path="data/inf.jsonl",
                client=MagicMock(),
            )

        assert any("citation_quality" in msg for msg in caplog.messages)
        assert any("pairwise" in msg for msg in caplog.messages)

    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_absolute_normal_metric_no_warning(self, mock_completion, caplog):
        from llm_judge.stages.judge import _judge_absolute

        mock_completion.return_value = CompletionResult(
            text=json.dumps({
                "per_metric": {"accuracy": 5, "citation_quality": 3},
                "overall_score": 4,
                "rationale": "Good",
                "critical_issue": False,
            }),
        )

        cfg = _make_config("absolute")
        tc = _make_testcase()
        inf = _make_inference()

        with caplog.at_level(logging.WARNING, logger="llm_judge.stages.judge"):
            _judge_absolute(
                cfg=cfg, tc=tc, inf=inf, idx=0,
                judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
                client=MagicMock(),
            )

        assert not any("Dropped" in msg for msg in caplog.messages)

    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_pairwise_dict_metric_no_warning(self, mock_completion, caplog):
        from llm_judge.stages.judge import _judge_pairwise

        mock_completion.return_value = CompletionResult(
            text=json.dumps({
                "per_metric": {"accuracy": {"A": 5, "B": 3}},
                "overall_winner": "A",
                "critical_issue_a": False,
                "critical_issue_b": False,
                "rationale": "A is better",
            }),
        )

        cfg = _make_config("pairwise")
        tc = _make_testcase()
        inf_a = _make_inference()
        inf_b = InferenceRecord(
            run_id="test-run",
            testcase_id="tc-001",
            candidate_id="c2",
            model=ModelInfo(vendor="openai", model_id="gpt-4o"),
            output=OutputInfo(text="test output 2"),
            status=StatusInfo(ok=True),
        )

        with caplog.at_level(logging.WARNING, logger="llm_judge.stages.judge"):
            _judge_pairwise(
                cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
                idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
                presented_swap=False, inf_path="data/inf.jsonl",
                client=MagicMock(),
            )

        assert not any("Dropped" in msg for msg in caplog.messages)

    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_absolute_dict_metric_emits_warning(self, mock_completion, caplog):
        from llm_judge.stages.judge import _judge_absolute

        mock_completion.return_value = CompletionResult(
            text=json.dumps({
                "per_metric": {"accuracy": {"A": 5, "B": 3}},
                "overall_score": 4,
                "rationale": "Good",
                "critical_issue": False,
            }),
        )

        cfg = _make_config("absolute")
        tc = _make_testcase()
        inf = _make_inference()

        with caplog.at_level(logging.WARNING, logger="llm_judge.stages.judge"):
            _judge_absolute(
                cfg=cfg, tc=tc, inf=inf, idx=0,
                judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
                client=MagicMock(),
            )

        assert any("absolute mode" in msg for msg in caplog.messages)
