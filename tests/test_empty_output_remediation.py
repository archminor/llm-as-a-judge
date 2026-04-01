"""Tests for report_qa empty-output remediation (plan 2026-03-13).

Covers:
- Judge deterministic failure short-circuit (absolute & pairwise)
- Autocheck empty-output detection for free-text cases
- Inference token-budget retry
- Compare report annotation for deterministic failures
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from llm_judge.llm_client import CompletionResult
from llm_judge.models import (
    AutoCheckRecord,
    Checks,
    FormatCompliance,
    InferenceRecord,
    JudgementRecord,
    ModelInfo,
    OutputInfo,
    RunConfig,
    Scores,
    StatusInfo,
    Testcase,
)


# ── Helpers ──────────────────────────────────────────────


def _make_testcase(**overrides) -> Testcase:
    defaults = {
        "testcase_id": "uc3-report-qa-001",
        "task_type": "report_qa",
        "input": {"raw_text": "test input"},
    }
    defaults.update(overrides)
    return Testcase(**defaults)


def _make_inference(
    candidate_id: str = "c1",
    ok: bool = True,
    text: str = "some output",
    error_type: str | None = None,
    error_message: str | None = None,
) -> InferenceRecord:
    status = StatusInfo(ok=ok, error_type=error_type, error_message=error_message)
    return InferenceRecord(
        run_id="test-run",
        testcase_id="uc3-report-qa-001",
        candidate_id=candidate_id,
        model=ModelInfo(vendor="openai", model_id="gpt-4o"),
        output=OutputInfo(text=text),
        status=status,
    )


def _make_config(mode: str = "hybrid") -> RunConfig:
    return RunConfig.model_validate({
        "run_id": "test-run",
        "dataset": {"testcases_path": "data/testcases.jsonl"},
        "candidates": [
            {"candidate_id": "c1", "vendor": "openai", "model_id": "gpt-4o"},
            {"candidate_id": "c2", "vendor": "openai", "model_id": "gpt-4o"},
        ],
        "judges": [{
            "judge_id": "j1",
            "vendor": "openai",
            "model_id": "gpt-4o",
            "rubric_version": "v1",
        }],
        "protocol": {
            "evaluation_mode": mode,
            "metrics": ["accuracy", "reasoning", "citation_quality"],
            "aggregation": {"method": "mean"},
        },
    })


# ── Judge: absolute short-circuit ────────────────────────


class TestAbsoluteFailedInference:
    def test_failed_inference_is_scored_as_critical_one(self):
        from llm_judge.stages.judge import _judge_absolute

        cfg = _make_config("absolute")
        tc = _make_testcase()
        inf = _make_inference(ok=False, text="", error_type="BadRequestError",
                              error_message="max_completion_tokens is too large")

        rec = _judge_absolute(
            cfg=cfg, tc=tc, inf=inf, idx=0,
            judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.scores.overall_score == 1.0
        assert rec.critical_issue is True
        assert inf.candidate_id in rec.critical_issue_candidates
        assert all(v == 1 for v in rec.scores.per_metric.values())
        assert rec.overall_rationale.startswith("Deterministic inference failure:")

    def test_empty_output_is_scored_as_critical_one(self):
        from llm_judge.stages.judge import _judge_absolute

        cfg = _make_config("absolute")
        tc = _make_testcase()
        inf = _make_inference(ok=True, text="")

        rec = _judge_absolute(
            cfg=cfg, tc=tc, inf=inf, idx=0,
            judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.scores.overall_score == 1.0
        assert rec.critical_issue is True
        assert "Empty output" in rec.overall_rationale

    def test_whitespace_only_output_is_scored_as_critical_one(self):
        from llm_judge.stages.judge import _judge_absolute

        cfg = _make_config("absolute")
        tc = _make_testcase()
        inf = _make_inference(ok=True, text="   \n  ")

        rec = _judge_absolute(
            cfg=cfg, tc=tc, inf=inf, idx=0,
            judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.scores.overall_score == 1.0
        assert rec.critical_issue is True

    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_short_circuit_does_not_call_llm_judge(self, mock_completion):
        from llm_judge.stages.judge import _judge_absolute

        cfg = _make_config("absolute")
        tc = _make_testcase()
        inf = _make_inference(ok=False, text="")

        _judge_absolute(
            cfg=cfg, tc=tc, inf=inf, idx=0,
            judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        mock_completion.assert_not_called()

    def test_applicable_metrics_used_for_report_qa(self):
        from llm_judge.stages.judge import _judge_absolute

        cfg = _make_config("absolute")
        tc = _make_testcase(task_type="report_qa")
        inf = _make_inference(ok=False, text="")

        rec = _judge_absolute(
            cfg=cfg, tc=tc, inf=inf, idx=0,
            judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        # report_qa gets accuracy, reasoning, citation_quality
        assert "accuracy" in rec.scores.per_metric
        assert "reasoning" in rec.scores.per_metric
        assert "citation_quality" in rec.scores.per_metric

    def test_format_compliance_included_in_short_circuit(self):
        """Failed inference short-circuit: FC=1 via judge_metrics (no schema gate needed)."""
        from llm_judge.stages.judge import _judge_absolute

        cfg = _make_config("absolute")
        cfg.protocol.metrics = ["accuracy", "format_compliance"]
        tc = Testcase(
            testcase_id="uc2-001",
            task_type="preprocessing",
            input={"raw_text": "test"},
            constraints={
                "output_format": {
                    "type": "json",
                    "json_schema_ref": "schemas/uc2_output.json",
                },
            },
        )
        inf = _make_inference(ok=False, text="")

        rec = _judge_absolute(
            cfg=cfg, tc=tc, inf=inf, idx=0,
            judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert "format_compliance" in rec.scores.per_metric
        assert rec.scores.per_metric["format_compliance"] == 1

    def test_failure_rationale_is_distinct_from_judge_error(self):
        from llm_judge.stages.judge import _judge_absolute

        cfg = _make_config("absolute")
        tc = _make_testcase()
        inf = _make_inference(ok=False, text="")

        rec = _judge_absolute(
            cfg=cfg, tc=tc, inf=inf, idx=0,
            judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.overall_rationale.startswith("Deterministic inference failure:")
        assert not rec.overall_rationale.startswith("Judge error:")


# ── Judge: pairwise short-circuit ────────────────────────


class TestPairwiseFailedInference:
    def test_failed_vs_successful_picks_successful_candidate(self):
        from llm_judge.stages.judge import _judge_pairwise

        cfg = _make_config("pairwise")
        tc = _make_testcase()
        inf_a = _make_inference(candidate_id="c1", ok=False, text="")
        inf_b = _make_inference(candidate_id="c2", ok=True, text="good answer")

        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.scores.overall_winner == "c2"
        assert rec.critical_issue is True
        assert "c1" in rec.critical_issue_candidates
        assert "c2" not in rec.critical_issue_candidates
        assert all(v == 5 for v in rec.scores.per_metric.values())

    def test_successful_vs_failed_picks_successful_candidate(self):
        from llm_judge.stages.judge import _judge_pairwise

        cfg = _make_config("pairwise")
        tc = _make_testcase()
        inf_a = _make_inference(candidate_id="c1", ok=True, text="good answer")
        inf_b = _make_inference(candidate_id="c2", ok=False, text="")

        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.scores.overall_winner == "c1"
        assert "c2" in rec.critical_issue_candidates

    def test_both_failed_returns_tie_and_marks_both_critical(self):
        from llm_judge.stages.judge import _judge_pairwise

        cfg = _make_config("pairwise")
        tc = _make_testcase()
        inf_a = _make_inference(candidate_id="c1", ok=False, text="")
        inf_b = _make_inference(candidate_id="c2", ok=False, text="")

        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.scores.overall_winner == "tie"
        assert rec.critical_issue is True
        assert "c1" in rec.critical_issue_candidates
        assert "c2" in rec.critical_issue_candidates
        assert all(v == 1 for v in rec.scores.per_metric.values())

    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_short_circuit_does_not_call_llm_judge(self, mock_completion):
        from llm_judge.stages.judge import _judge_pairwise

        cfg = _make_config("pairwise")
        tc = _make_testcase()
        inf_a = _make_inference(candidate_id="c1", ok=False, text="")
        inf_b = _make_inference(candidate_id="c2", ok=True, text="answer")

        _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        mock_completion.assert_not_called()

    def test_pairwise_one_fail_uses_max_metric_scores(self):
        from llm_judge.stages.judge import _judge_pairwise

        cfg = _make_config("pairwise")
        tc = _make_testcase()
        inf_a = _make_inference(candidate_id="c1", ok=False, text="")
        inf_b = _make_inference(candidate_id="c2", ok=True, text="answer")

        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        # When one side fails, per_metric should all be 5 (clear winner)
        assert all(v == 5 for v in rec.scores.per_metric.values())

    def test_pairwise_format_compliance_included_in_short_circuit(self):
        """Pairwise one-fail short-circuit: FC=5 (relative score, winner side)."""
        from llm_judge.stages.judge import _judge_pairwise

        cfg = _make_config("pairwise")
        cfg.protocol.metrics = ["accuracy", "format_compliance"]
        tc = Testcase(
            testcase_id="uc2-001",
            task_type="preprocessing",
            input={"raw_text": "test"},
            constraints={
                "output_format": {
                    "type": "json",
                    "json_schema_ref": "schemas/uc2_output.json",
                },
            },
        )
        inf_a = _make_inference(candidate_id="c1", ok=False, text="")
        inf_b = _make_inference(candidate_id="c2", ok=True, text="answer")

        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert "format_compliance" in rec.scores.per_metric
        # Pairwise one-fail: all metrics are 5 (clear winner), FC included
        assert rec.scores.per_metric["format_compliance"] == 5

    def test_pairwise_failure_rationale_is_distinct_from_judge_error(self):
        from llm_judge.stages.judge import _judge_pairwise

        cfg = _make_config("pairwise")
        tc = _make_testcase()
        inf_a = _make_inference(candidate_id="c1", ok=False, text="")
        inf_b = _make_inference(candidate_id="c2", ok=True, text="answer")

        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.overall_rationale.startswith("Deterministic inference failure comparison:")
        assert not rec.overall_rationale.startswith("Judge error:")


# ── Autocheck: empty output / failed inference ───────────


class TestAutocheckEmptyOutput:
    def test_empty_output_fails_without_output_format_constraint(self):
        from llm_judge.stages.autocheck import _check_format_compliance

        inf = _make_inference(ok=True, text="")
        tc = _make_testcase()  # no output_format constraint

        result = _check_format_compliance(inf, tc, schema_result=None)

        assert result.passed is False
        assert "Empty output" in result.details

    def test_non_empty_free_text_without_output_format_still_passes(self):
        from llm_judge.stages.autocheck import _check_format_compliance

        inf = _make_inference(ok=True, text="This is a valid answer")
        tc = _make_testcase()  # no output_format constraint

        result = _check_format_compliance(inf, tc, schema_result=None)

        assert result.passed is True

    def test_failed_inference_short_circuits_schema_checks(self):
        from llm_judge.stages.autocheck import _run_checks

        inf = _make_inference(ok=False, text="", error_type="BadRequestError",
                              error_message="token budget exceeded")
        tc = _make_testcase()

        checks = _run_checks(inf, tc)

        assert checks.format_compliance is not None
        assert checks.format_compliance.passed is False
        assert "Inference failed" in checks.format_compliance.details
        assert checks.json_schema_validation is None

    def test_whitespace_only_output_fails_format_check(self):
        from llm_judge.stages.autocheck import _check_format_compliance

        inf = _make_inference(ok=True, text="   \n\t  ")
        tc = _make_testcase()

        result = _check_format_compliance(inf, tc, schema_result=None)

        assert result.passed is False
        assert "Empty output" in result.details


# ── Inference: token budget retry ────────────────────────


class TestTokenBudgetRetry:
    def test_is_token_budget_error_detects_max_completion_tokens(self):
        from llm_judge.stages.inference import _is_token_budget_error

        from openai import BadRequestError

        exc = BadRequestError(
            message="max_completion_tokens is too large: 8192. This model supports at most 4096.",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )
        assert _is_token_budget_error(exc) is True

    def test_is_token_budget_error_detects_context_length(self):
        from llm_judge.stages.inference import _is_token_budget_error

        from openai import BadRequestError

        exc = BadRequestError(
            message="This model's maximum context length is 128000 tokens. However, you requested 8192 tokens in the completion.",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )
        assert _is_token_budget_error(exc) is True

    def test_is_token_budget_error_ignores_unrelated_400(self):
        from llm_judge.stages.inference import _is_token_budget_error

        from openai import BadRequestError

        exc = BadRequestError(
            message="Invalid model specified",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )
        assert _is_token_budget_error(exc) is False

    def test_extract_reduced_budget_from_at_most_pattern(self):
        from llm_judge.stages.inference import _extract_reduced_budget

        from openai import BadRequestError

        exc = BadRequestError(
            message="max_completion_tokens is too large: 8192. at most 2048 completion tokens",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )
        reduced = _extract_reduced_budget(exc, 8192)
        assert reduced == 2048

    def test_extract_reduced_budget_returns_none_when_explicit_budget_too_small(self):
        from llm_judge.stages.inference import _extract_reduced_budget

        from openai import BadRequestError

        exc = BadRequestError(
            message="at most 32 tokens remaining",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )
        reduced = _extract_reduced_budget(exc, 8192)
        # Server explicitly says 32 tokens remaining, which is below threshold.
        # Should NOT fall back to halving — return None to prevent useless retry.
        assert reduced is None

    def test_extract_reduced_budget_halves_as_fallback(self):
        from llm_judge.stages.inference import _extract_reduced_budget

        from openai import BadRequestError

        exc = BadRequestError(
            message="some unknown token error format",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )
        reduced = _extract_reduced_budget(exc, 4096)
        assert reduced == 2048

    def test_extract_reduced_budget_from_tsuzumi2_request_has_input_tokens(self):
        """Production tsuzumi2 error: 'your request has N input tokens'."""
        from llm_judge.stages.inference import _extract_reduced_budget

        from openai import BadRequestError

        exc = BadRequestError(
            message=(
                "This model's maximum context length is 131072 tokens. "
                "However, your request has 128231 input tokens and 8192 "
                "max_completion_tokens. Please reduce the length."
            ),
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )
        reduced = _extract_reduced_budget(exc, 8192)
        # 131072 - 128231 = 2841
        assert reduced == 2841

    def test_extract_reduced_budget_tsuzumi2_too_small_returns_none(self):
        """When tsuzumi2 remaining budget is below threshold, return None."""
        from llm_judge.stages.inference import _extract_reduced_budget

        from openai import BadRequestError

        exc = BadRequestError(
            message=(
                "This model's maximum context length is 131072 tokens. "
                "However, your request has 131050 input tokens and 8192 "
                "max_completion_tokens. Please reduce the length."
            ),
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )
        reduced = _extract_reduced_budget(exc, 8192)
        # 131072 - 131050 = 22, below threshold
        assert reduced is None

    def test_no_retry_when_bad_request_is_not_token_budget_related(self):
        """Non-token-budget 400 errors should propagate without retry."""
        from llm_judge.stages.inference import _is_token_budget_error

        from openai import BadRequestError

        exc = BadRequestError(
            message="content_filter: The response was filtered",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )
        assert _is_token_budget_error(exc) is False


# ── Compare: report annotation ───────────────────────────


class TestCompareReportAnnotation:
    def test_report_qa_failed_candidate_no_longer_appears_as_five_point_zero(self):
        """Deterministic failure scores (1) should bring down mean_score."""
        from llm_judge.stages.compare import _compute_aggregate

        # Create judgements: c1 fails (score 1), c2 succeeds (score 5)
        judgements = [
            JudgementRecord(
                run_id="r1", testcase_id="tc1",
                judge={"judge_id": "j1", "vendor": "openai", "model_id": "m1", "rubric_version": "v1"},
                mode="absolute",
                targets=[{"candidate_id": "c1", "inference_ref": {"path": "p", "line_index": 0}}],
                scores=Scores(per_metric={"accuracy": 1}, overall_score=1.0),
                critical_issue=True,
                critical_issue_candidates=["c1"],
                overall_rationale="Deterministic inference failure: Empty output",
            ),
            JudgementRecord(
                run_id="r1", testcase_id="tc1",
                judge={"judge_id": "j1", "vendor": "openai", "model_id": "m1", "rubric_version": "v1"},
                mode="absolute",
                targets=[{"candidate_id": "c2", "inference_ref": {"path": "p", "line_index": 1}}],
                scores=Scores(per_metric={"accuracy": 5}, overall_score=5.0),
                critical_issue=False,
                critical_issue_candidates=[],
                overall_rationale="Good answer",
            ),
        ]

        agg = _compute_aggregate(judgements, ["c1", "c2"], [])

        assert agg.mean_score["accuracy"]["c1"] == 1.0
        assert agg.mean_score["accuracy"]["c2"] == 5.0
        assert agg.critical_issue_count.get("c1", 0) == 1

    def test_empty_output_autocheck_surfaces_in_notable_failures(self):
        from llm_judge.stages.compare import _compute_aggregate

        autochecks = [
            AutoCheckRecord(
                run_id="r1",
                testcase_id="uc3-report-qa-001",
                candidate_id="c1",
                checks=Checks(
                    format_compliance=FormatCompliance(passed=False, details="Empty output"),
                ),
            ),
        ]

        agg = _compute_aggregate([], ["c1", "c2"], autochecks)

        assert len(agg.notable_failures) == 1
        assert agg.notable_failures[0].candidate_id == "c1"
        assert "Empty output" in agg.notable_failures[0].reason

    def test_markdown_report_mentions_deterministic_failures(self):
        from llm_judge.stages.compare import _write_markdown_report
        from llm_judge.models import (
            AggregateBlock,
            ComparisonReport,
            JudgeAgreement,
            ReportDataset,
            ReportSummary,
            Results,
        )
        from pathlib import Path
        import tempfile

        report = ComparisonReport(
            run_id="test-run",
            dataset=ReportDataset(dataset_version="v1", testcase_count=10),
            candidates=[],
            judges=[],
            protocol={},
            summary=ReportSummary(
                total_judgements=10, valid_judgements=8, excluded_judgements=2,
            ),
            results=Results(
                overall=AggregateBlock(),
                by_task={},
                by_bucket={},
                judge_agreement=JudgeAgreement(),
            ),
        )

        det_judgement = JudgementRecord(
            run_id="r1", testcase_id="tc1",
            judge={"judge_id": "j1", "vendor": "openai", "model_id": "m1", "rubric_version": "v1"},
            mode="absolute",
            targets=[{"candidate_id": "c1", "inference_ref": {"path": "p", "line_index": 0}}],
            scores=Scores(per_metric={"accuracy": 1}, overall_score=1.0),
            critical_issue=True,
            critical_issue_candidates=["c1"],
            overall_rationale="Deterministic inference failure: Empty output",
        )

        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "report.md"
            _write_markdown_report(report, md_path, judgements=[det_judgement])
            content = md_path.read_text()

        assert "WARNING" in content
        assert "deterministic scoring" in content
        assert "**1**" in content
        assert "working side wins" in content or "wins" in content
        assert "tie" in content
