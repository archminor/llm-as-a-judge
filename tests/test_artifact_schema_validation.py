"""Tests for artifact schema validation (Task D)."""

from __future__ import annotations

import pytest

from llm_judge.artifact_validation import (
    ArtifactValidationError,
    validate_artifacts,
    validate_single_artifact,
)
from llm_judge.models import (
    AutoCheckRecord,
    Checks,
    FormatCompliance,
    InferenceRecord,
    JudgementRecord,
    ModelInfo,
    OutputInfo,
    StatusInfo,
    UsageInfo,
)


class TestInferenceRecordValidation:
    def test_valid_record(self) -> None:
        record = InferenceRecord(
            run_id="run-1",
            testcase_id="tc-001",
            candidate_id="cand-1",
            model=ModelInfo(vendor="openai", model_id="gpt-4"),
            output=OutputInfo(text="hello"),
            status=StatusInfo(ok=True),
        )
        # Should not raise
        validate_artifacts("inference-record", [record])

    def test_invalid_record_dict(self) -> None:
        bad = {
            "run_id": "run-1",
            # missing testcase_id, candidate_id, model, output
        }
        with pytest.raises(ArtifactValidationError) as exc_info:
            validate_artifacts("inference-record", [bad])
        assert "inference-record" in str(exc_info.value)
        assert len(exc_info.value.errors) > 0


    def test_valid_record_with_reasoning(self) -> None:
        record = InferenceRecord(
            run_id="run-1",
            testcase_id="tc-001",
            candidate_id="cand-1",
            model=ModelInfo(vendor="openai", model_id="o3"),
            output=OutputInfo(text="hello", reasoning="Let me think step by step..."),
            usage=UsageInfo(input_tokens=10, output_tokens=50, reasoning_tokens=30),
            status=StatusInfo(ok=True),
        )
        validate_artifacts("inference-record", [record])

    def test_valid_record_without_reasoning(self) -> None:
        """reasoning=None should be excluded by exclude_none and still pass."""
        record = InferenceRecord(
            run_id="run-1",
            testcase_id="tc-001",
            candidate_id="cand-1",
            model=ModelInfo(vendor="openai", model_id="gpt-4"),
            output=OutputInfo(text="hello"),
            usage=UsageInfo(input_tokens=10, output_tokens=50),
            status=StatusInfo(ok=True),
        )
        validate_artifacts("inference-record", [record])


class TestAutoCheckRecordValidation:
    def test_valid_record(self) -> None:
        record = AutoCheckRecord(
            run_id="run-1",
            testcase_id="tc-001",
            candidate_id="cand-1",
            checks=Checks(
                format_compliance=FormatCompliance(passed=True, details="OK")
            ),
        )
        validate_artifacts("autocheck-record", [record])

    def test_invalid_extra_field(self) -> None:
        bad = {
            "run_id": "run-1",
            "testcase_id": "tc-001",
            "candidate_id": "cand-1",
            "checks": {"format_compliance": {"passed": True}},
            "extra_field": "should_fail",
        }
        with pytest.raises(ArtifactValidationError):
            validate_artifacts("autocheck-record", [bad])


class TestJudgementRecordValidation:
    def test_valid_record(self) -> None:
        record = JudgementRecord(
            run_id="run-1",
            testcase_id="tc-001",
            judge={
                "judge_id": "j1",
                "vendor": "openai",
                "model_id": "gpt-4",
                "rubric_version": "v1",
            },
            mode="absolute",
            targets=[{
                "candidate_id": "cand-1",
                "inference_ref": {"path": "test.jsonl", "line_index": 0},
            }],
            scores={"per_metric": {"accuracy": 3}},
        )
        validate_artifacts("judgement-record", [record])


class TestComparisonReportValidation:
    def test_valid_minimal_report(self) -> None:
        report = {
            "run_id": "run-1",
            "dataset": {"testcase_count": 1},
            "candidates": [
                {"candidate_id": "c1", "vendor": "openai", "model_id": "gpt-4"}
            ],
            "judges": [
                {"judge_id": "j1", "vendor": "openai", "model_id": "gpt-4", "rubric_version": "v1"}
            ],
            "protocol": {},
            "results": {
                "overall": {},
            },
        }
        validate_single_artifact("comparison-report", report)

    def test_invalid_report(self) -> None:
        bad = {"run_id": "run-1"}  # missing required fields
        with pytest.raises(ArtifactValidationError) as exc_info:
            validate_single_artifact("comparison-report", bad)
        assert "comparison-report" in exc_info.value.artifact_type


class TestErrorCapping:
    def test_max_errors_capped(self) -> None:
        """Errors should be capped at _MAX_ERRORS."""
        bad_records = [{"run_id": f"r-{i}"} for i in range(50)]
        with pytest.raises(ArtifactValidationError) as exc_info:
            validate_artifacts("inference-record", bad_records)
        assert len(exc_info.value.errors) <= 20


class TestUnknownArtifactType:
    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown artifact type"):
            validate_artifacts("nonexistent-type", [{}])
