"""Tests for task-aware metric filtering in the judge stage.

_TASK_RESTRICTED_METRICS is now an empty dict, so _filter_llm_metrics
never removes any metrics based on task_type.
"""

from __future__ import annotations

import json

from llm_judge.models import Constraints, OutputFormat, Testcase as JudgeTestcase
from llm_judge.stages.judge import _filter_llm_metrics, _get_schema_gate_result


def _make_testcase(
    task_type: str,
    structured: bool = False,
) -> JudgeTestcase:
    output_format = None
    if structured:
        output_format = OutputFormat(
            type="json",
            json_schema_ref="schemas/testcase.schema.json",
        )
    constraints = Constraints(output_format=output_format) if output_format else None
    return JudgeTestcase(
        testcase_id=f"{task_type}-001",
        task_type=task_type,
        input={"raw_text": "dummy"},
        constraints=constraints,
    )


def test_preprocessing_keeps_all_metrics():
    """With empty _TASK_RESTRICTED_METRICS, no metrics are filtered."""
    tc = _make_testcase("preprocessing")
    metrics = [
        "accuracy",
        "reasoning",
        "citation_quality",
        "format_compliance",
    ]

    filtered = _filter_llm_metrics(metrics, tc)

    assert filtered == metrics


def test_report_qa_keeps_all_metrics():
    tc = _make_testcase("report_qa")
    metrics = [
        "accuracy",
        "reasoning",
        "citation_quality",
        "format_compliance",
    ]

    filtered = _filter_llm_metrics(metrics, tc)

    assert filtered == metrics


def test_report_generation_keeps_all_metrics():
    tc = _make_testcase("report_generation")
    metrics = [
        "accuracy",
        "reasoning",
        "citation_quality",
        "format_compliance",
    ]

    filtered = _filter_llm_metrics(metrics, tc)

    assert filtered == metrics


def test_custom_task_type_keeps_all_metrics():
    """Any task_type keeps all metrics since _TASK_RESTRICTED_METRICS is empty."""
    tc = _make_testcase("custom_task")
    metrics = ["accuracy", "reasoning", "format_compliance"]

    filtered = _filter_llm_metrics(metrics, tc)

    assert filtered == metrics


def test_structured_output_keeps_format_compliance_in_llm_metrics():
    """_filter_llm_metrics does not remove format_compliance."""
    tc = _make_testcase("report_qa", structured=True)
    metrics = [
        "accuracy",
        "reasoning",
        "citation_quality",
        "format_compliance",
    ]

    filtered = _filter_llm_metrics(metrics, tc)

    assert "format_compliance" in filtered


def test_schema_gate_returns_none_when_fc_not_in_metrics():
    tc = _make_testcase("preprocessing", structured=True)
    result = _get_schema_gate_result(tc, '{"valid": "json"}', ["accuracy"])
    assert result is None


def test_schema_gate_returns_none_when_no_schema():
    tc = _make_testcase("preprocessing", structured=False)
    result = _get_schema_gate_result(tc, "some text", ["format_compliance", "accuracy"])
    assert result is None


def test_schema_gate_returns_one_on_schema_failure():
    tc = _make_testcase("report_qa", structured=True)
    # Invalid JSON will fail schema validation
    result = _get_schema_gate_result(tc, "not json at all", ["format_compliance", "accuracy"])
    assert result == 1


def test_schema_gate_returns_none_on_schema_pass():
    """Schema passed -> None (delegate to Judge)."""
    tc = _make_testcase("report_qa", structured=True)
    # Construct data that matches testcase.schema.json
    valid_output = json.dumps({
        "testcase_id": "tc-001",
        "task_type": "test",
        "input": {"text": "hello"},
    })
    result = _get_schema_gate_result(tc, valid_output, ["format_compliance", "accuracy"])
    assert result is None


# -- _pairwise_schema_gate tests --


def _make_inference_record(candidate_id: str, text: str):
    from llm_judge.models import InferenceRecord, ModelInfo, OutputInfo
    return InferenceRecord(
        run_id="test",
        testcase_id="tc-001",
        candidate_id=candidate_id,
        model=ModelInfo(vendor="openai", model_id="gpt-4o"),
        output=OutputInfo(text=text),
    )


def test_pairwise_gate_returns_none_when_fc_not_in_metrics():
    from llm_judge.stages.judge import _pairwise_schema_gate
    tc = _make_testcase("report_qa", structured=True)
    inf_a = _make_inference_record("c1", "text")
    inf_b = _make_inference_record("c2", "text")
    result = _pairwise_schema_gate(tc, inf_a, inf_b, ["accuracy"])
    assert result is None


def test_pairwise_gate_returns_none_when_no_schema():
    from llm_judge.stages.judge import _pairwise_schema_gate
    tc = _make_testcase("report_qa", structured=False)
    inf_a = _make_inference_record("c1", "text")
    inf_b = _make_inference_record("c2", "text")
    result = _pairwise_schema_gate(tc, inf_a, inf_b, ["format_compliance"])
    assert result is None


def test_pairwise_gate_returns_none_when_both_pass():
    from llm_judge.stages.judge import _pairwise_schema_gate
    tc = _make_testcase("report_qa", structured=True)
    valid = json.dumps({
        "testcase_id": "tc-001",
        "task_type": "test",
        "input": {"text": "hello"},
    })
    inf_a = _make_inference_record("c1", valid)
    inf_b = _make_inference_record("c2", valid)
    result = _pairwise_schema_gate(tc, inf_a, inf_b, ["format_compliance"])
    assert result is None


def test_pairwise_gate_a_pass_b_fail():
    from llm_judge.stages.judge import _pairwise_schema_gate
    tc = _make_testcase("report_qa", structured=True)
    valid = json.dumps({
        "testcase_id": "tc-001",
        "task_type": "test",
        "input": {"text": "hello"},
    })
    inf_a = _make_inference_record("c1", valid)
    inf_b = _make_inference_record("c2", "not json")
    result = _pairwise_schema_gate(tc, inf_a, inf_b, ["format_compliance"])
    assert result is not None
    assert result.per_metric_value.score_a == 5
    assert result.per_metric_value.score_b == 1
    assert result.winner_override == "c1"


def test_pairwise_gate_both_fail():
    from llm_judge.stages.judge import _pairwise_schema_gate
    tc = _make_testcase("report_qa", structured=True)
    inf_a = _make_inference_record("c1", "bad")
    inf_b = _make_inference_record("c2", "bad")
    result = _pairwise_schema_gate(tc, inf_a, inf_b, ["format_compliance"])
    assert result is not None
    assert result.per_metric_value.score_a == 1
    assert result.per_metric_value.score_b == 1
    assert result.winner_override is None


# -- Integration: schema gate FC override in judge result --


def test_schema_gate_fc_override_absolute():
    """Normal path: schema failed -> FC=1 injected into Judge result."""
    from unittest.mock import MagicMock, patch
    from llm_judge.stages.judge import _judge_absolute
    from llm_judge.models import RunConfig, Testcase

    cfg = RunConfig.model_validate({
        "run_id": "test",
        "dataset": {"testcases_path": "data/testcases.jsonl"},
        "candidates": [{"candidate_id": "c1", "vendor": "openai", "model_id": "gpt-4o"}],
        "judges": [{"judge_id": "j1", "vendor": "openai", "model_id": "gpt-4o", "rubric_version": "v1"}],
        "protocol": {
            "evaluation_mode": "absolute",
            "metrics": ["accuracy", "format_compliance"],
            "aggregation": {"method": "mean"},
        },
    })
    tc = Testcase(
        testcase_id="tc-001",
        task_type="report_qa",
        input={"raw_text": "test"},
        constraints={
            "output_format": {
                "type": "json",
                "json_schema_ref": "schemas/testcase.schema.json",
            },
        },
    )
    # Output is valid JSON but does NOT match schema (missing required fields)
    inf = _make_inference_record("c1", '{"answer": "hello"}')

    mock_result = '{"per_metric": {"accuracy": {"rationale": "ok", "score": 3}}, "overall_score": 3.0}'
    with patch("llm_judge.stages.judge.judge_chat_completion") as mock_judge:
        mock_judge.return_value = MagicMock(text=mock_result)
        rec = _judge_absolute(
            cfg=cfg, tc=tc, inf=inf, idx=0,
            judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

    # Schema failed -> FC=1 programmatically inserted
    assert rec.scores.per_metric["format_compliance"] == 1
    # accuracy comes from Judge
    assert "accuracy" in rec.scores.per_metric


# -- Regression: schema gate + presented_swap --


def test_pairwise_schema_gate_swap_aligns_scores_to_presented_labels():
    """When presented_swap=True, schema gate score_a/score_b must be swapped
    so that layer-1 gate checks the correct side.

    Scenario: inf_a (c1) passes schema, inf_b (c2) fails schema.
    With presented_swap=True, label A = c2, label B = c1.
    score_a (label A) should be 1 (c2 failed), score_b (label B) should be 5 (c1 passed).
    """
    from unittest.mock import MagicMock, patch
    from llm_judge.stages.judge import _judge_pairwise
    from llm_judge.models import RunConfig, Testcase

    cfg = RunConfig.model_validate({
        "run_id": "test",
        "dataset": {"testcases_path": "data/testcases.jsonl"},
        "candidates": [
            {"candidate_id": "c1", "vendor": "openai", "model_id": "gpt-4o"},
            {"candidate_id": "c2", "vendor": "openai", "model_id": "gpt-4o"},
        ],
        "judges": [{"judge_id": "j1", "vendor": "openai", "model_id": "gpt-4o", "rubric_version": "v1"}],
        "protocol": {
            "evaluation_mode": "pairwise",
            "metrics": ["accuracy", "format_compliance"],
            "blinding": {"enabled": True},
            "aggregation": {"method": "mean"},
        },
    })
    tc = Testcase(
        testcase_id="tc-001",
        task_type="report_qa",
        input={"raw_text": "test"},
        constraints={
            "output_format": {
                "type": "json",
                "json_schema_ref": "schemas/testcase.schema.json",
            },
        },
    )
    valid = json.dumps({
        "testcase_id": "tc-001",
        "task_type": "test",
        "input": {"text": "hello"},
    })
    inf_a = _make_inference_record("c1", valid)   # passes schema
    inf_b = _make_inference_record("c2", "bad")   # fails schema

    mock_result = json.dumps({
        "per_metric": {
            "accuracy": {"rationale": "ok", "score_a": 3, "score_b": 3},
        },
        "overall_winner": "tie",
    })
    with patch("llm_judge.stages.judge.judge_chat_completion") as mock_judge:
        mock_judge.return_value = MagicMock(text=mock_result)
        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1,
            judge_ref=cfg.judges[0],
            presented_swap=True,  # label A = c2, label B = c1
            inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

    fc = rec.scores.per_metric["format_compliance"]
    # After swap: label A = c2 (failed) -> score_a=1, label B = c1 (passed) -> score_b=5
    assert fc.score_a == 1, f"label A (c2, schema failed) should be 1, got {fc.score_a}"
    assert fc.score_b == 5, f"label B (c1, schema passed) should be 5, got {fc.score_b}"
    # Winner should be c1 (the one that passed)
    assert rec.scores.overall_winner == "c1"
