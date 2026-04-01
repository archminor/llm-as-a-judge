"""E2E tests: full pipeline flow for metrics and consistency modes.

Each test runs inference → autocheck → judge → compare with mocked LLM calls,
verifying that output files are produced and contain correct data.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from llm_judge.llm_client import CompletionResult
from llm_judge.models import InferenceRecord
from llm_judge.utils import read_jsonl


# ── helpers ───────────────────────────────────────────────────────────────────


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    testcases_path = tmp_path / "testcases.jsonl"
    _write_jsonl(testcases_path, [
        {
            "testcase_id": "tc-001",
            "task_type": "report_generation",
            "input": {"raw_text": "E2E test input"},
        },
    ])

    cfg = {
        "run_id": "e2e-test",
        "dataset": {"testcases_path": str(testcases_path)},
        "candidates": [
            {
                "candidate_id": "cand-a",
                "vendor": "openai",
                "model_id": "gpt-4o",
                "generation_params": {"temperature": 0, "max_tokens": 64},
            },
            {
                "candidate_id": "cand-b",
                "vendor": "openai",
                "model_id": "gpt-4o",
                "generation_params": {"temperature": 0, "max_tokens": 64},
            },
        ],
        "judges": [
            {
                "judge_id": "judge-1",
                "vendor": "openai",
                "model_id": "gpt-4o",
                "rubric_version": "v1",
            },
        ],
        "protocol": {
            "evaluation_mode": "pairwise",
            "metrics": ["accuracy"],
            "aggregation": {"method": "mean"},
            "blinding": {"enabled": True, "random_seed": 42},
            "repeats": {"inference_repeats": 1, "judge_repeats": 1},
            "parallelism": {
                "inference_max_concurrency": 1,
                "judge_max_concurrency": 1,
                "consistency_max_concurrency": 1,
            },
        },
    }
    if overrides:
        for key, val in overrides.items():
            if isinstance(val, dict) and key in cfg:
                cfg[key].update(val)
            else:
                cfg[key] = val

    config_path = tmp_path / "run-config.yaml"
    config_path.write_text(yaml.dump(cfg, allow_unicode=True))
    return config_path


def _make_mock_openai_response(content: str = "mock output") -> MagicMock:
    mock = MagicMock()
    mock.choices = [MagicMock()]
    msg = mock.choices[0].message
    msg.content = content
    msg.reasoning_content = None
    msg.model_extra = {}
    mock.choices[0].finish_reason = "stop"
    usage = mock.usage
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    usage.completion_tokens_details = None
    return mock


def _mock_judge_response_pairwise() -> CompletionResult:
    return CompletionResult(
        text=json.dumps({
            "per_metric": {"accuracy": 5},
            "overall_winner": "A",
            "critical_issue_a": False,
            "critical_issue_b": False,
            "rationale": "A is better",
        }),
    )


def _mock_consistency_response() -> CompletionResult:
    return CompletionResult(
        text=json.dumps({
            "overall": 4.0,
            "rationale": "Outputs are mostly consistent",
        }),
    )


# ── Stage runners with mocks ────────────────────────────────────────────────


def _run_inference_mocked(config_path: str, output_path: str) -> Path:
    """Run inference stage with mocked LLM calls."""
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "llm_judge.stages.inference.get_thread_local_openai_client",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "llm_judge.stages.inference.chat_completion",
                return_value=_make_mock_openai_response("mock LLM output"),
            )
        )
        stack.enter_context(patch("llm_judge.stages.inference.validate_artifacts"))

        from llm_judge.stages.inference import run_inference

        return run_inference(config_path, output_path)


def _run_autocheck(config_path: str, inference_path: str, output_path: str) -> Path:
    """Run autocheck stage (no LLM calls needed)."""
    from llm_judge.stages.autocheck import run_autocheck

    return run_autocheck(config_path, inference_path, output_path)


def _run_judge_metrics_mocked(
    config_path: str, inference_path: str, output_path: str,
) -> Path:
    """Run judge stage (metrics mode) with mocked LLM calls."""
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "llm_judge.stages.judge.get_thread_local_openai_client",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "llm_judge.stages.judge.judge_chat_completion",
                return_value=_mock_judge_response_pairwise(),
            )
        )
        stack.enter_context(patch("llm_judge.stages.judge.validate_artifacts"))

        from llm_judge.stages.judge import run_judge

        return run_judge(config_path, inference_path, output_path)


def _run_judge_consistency_mocked(
    config_path: str, inference_path: str, output_path: str,
) -> Path:
    """Run judge stage (consistency mode) with mocked LLM calls."""
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "llm_judge.stages.consistency.get_thread_local_openai_client",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "llm_judge.stages.consistency.judge_chat_completion",
                return_value=_mock_consistency_response(),
            )
        )
        stack.enter_context(
            patch("llm_judge.artifact_validation.validate_artifacts")
        )

        from llm_judge.stages.consistency import run_consistency

        return run_consistency(config_path, inference_path, output_path)


def _run_compare(
    config_path: str,
    judgements_path: str | None = None,
    consistency_path: str | None = None,
    output_path: str | None = None,
) -> Path:
    """Run compare stage."""
    from llm_judge.stages.compare import run_compare

    return run_compare(
        config_path,
        judgements_path,
        consistency_path=consistency_path,
        output_path=output_path,
    )


# ── E2E: metrics mode ───────────────────────────────────────────────────────


class TestMetricsModePipeline:
    """inference_repeats=1 → inference → autocheck → judge (metrics) → compare."""

    def test_full_pipeline_produces_all_outputs(self, tmp_path):
        config_path = str(_write_config(tmp_path))

        inf_out = _run_inference_mocked(
            config_path, str(tmp_path / "inference.jsonl"),
        )
        ac_out = _run_autocheck(
            config_path, str(inf_out), str(tmp_path / "autocheck.jsonl"),
        )
        jdg_out = _run_judge_metrics_mocked(
            config_path, str(inf_out), str(tmp_path / "judgements.jsonl"),
        )
        cmp_out = _run_compare(
            config_path,
            judgements_path=str(jdg_out),
            output_path=str(tmp_path / "report.json"),
        )

        assert inf_out.exists()
        assert ac_out.exists()
        assert jdg_out.exists()
        assert cmp_out.exists()
        assert cmp_out.with_suffix(".md").exists()

    def test_inference_records_match_candidates(self, tmp_path):
        config_path = str(_write_config(tmp_path))

        inf_out = _run_inference_mocked(
            config_path, str(tmp_path / "inference.jsonl"),
        )

        records = read_jsonl(inf_out)
        candidate_ids = {r["candidate_id"] for r in records}
        assert candidate_ids == {"cand-a", "cand-b"}
        assert all(r["run_id"] == "e2e-test" for r in records)

    def test_judgements_contain_scores(self, tmp_path):
        config_path = str(_write_config(tmp_path))

        inf_out = _run_inference_mocked(
            config_path, str(tmp_path / "inference.jsonl"),
        )
        jdg_out = _run_judge_metrics_mocked(
            config_path, str(inf_out), str(tmp_path / "judgements.jsonl"),
        )

        records = read_jsonl(jdg_out)
        assert len(records) > 0
        for rec in records:
            assert "scores" in rec
            assert "accuracy" in rec["scores"]["per_metric"]

    def test_report_contains_candidates(self, tmp_path):
        config_path = str(_write_config(tmp_path))

        inf_out = _run_inference_mocked(
            config_path, str(tmp_path / "inference.jsonl"),
        )
        _run_autocheck(
            config_path, str(inf_out), str(tmp_path / "autocheck.jsonl"),
        )
        jdg_out = _run_judge_metrics_mocked(
            config_path, str(inf_out), str(tmp_path / "judgements.jsonl"),
        )
        cmp_out = _run_compare(
            config_path,
            judgements_path=str(jdg_out),
            output_path=str(tmp_path / "report.json"),
        )

        with open(cmp_out) as f:
            report = json.load(f)

        assert report["run_id"] == "e2e-test"
        cids = {c["candidate_id"] for c in report["candidates"]}
        assert cids == {"cand-a", "cand-b"}


# ── E2E: consistency mode ───────────────────────────────────────────────────


class TestConsistencyModePipeline:
    """inference_repeats=2 → inference → autocheck → judge (consistency) → compare."""

    def test_full_pipeline_produces_all_outputs(self, tmp_path):
        config_path = str(_write_config(tmp_path, overrides={
            "protocol": {
                "evaluation_mode": "pairwise",
                "metrics": ["accuracy"],
                "aggregation": {"method": "mean"},
                "blinding": {"enabled": True, "random_seed": 42},
                "repeats": {"inference_repeats": 2, "judge_repeats": 1},
                "parallelism": {
                    "inference_max_concurrency": 1,
                    "judge_max_concurrency": 1,
                    "consistency_max_concurrency": 1,
                },
            },
        }))

        inf_out = _run_inference_mocked(
            config_path, str(tmp_path / "inference.jsonl"),
        )
        ac_out = _run_autocheck(
            config_path, str(inf_out), str(tmp_path / "autocheck.jsonl"),
        )
        con_out = _run_judge_consistency_mocked(
            config_path, str(inf_out), str(tmp_path / "consistency.jsonl"),
        )
        cmp_out = _run_compare(
            config_path,
            consistency_path=str(con_out),
            output_path=str(tmp_path / "report.json"),
        )

        assert inf_out.exists()
        assert ac_out.exists()
        assert con_out.exists()
        assert cmp_out.exists()
        assert cmp_out.with_suffix(".md").exists()

    def test_inference_repeats_produce_multiple_records(self, tmp_path):
        config_path = str(_write_config(tmp_path, overrides={
            "protocol": {
                "evaluation_mode": "pairwise",
                "metrics": ["accuracy"],
                "aggregation": {"method": "mean"},
                "blinding": {"enabled": True, "random_seed": 42},
                "repeats": {"inference_repeats": 2, "judge_repeats": 1},
                "parallelism": {
                    "inference_max_concurrency": 1,
                    "judge_max_concurrency": 1,
                    "consistency_max_concurrency": 1,
                },
            },
        }))

        inf_out = _run_inference_mocked(
            config_path, str(tmp_path / "inference.jsonl"),
        )

        records = read_jsonl(inf_out)
        # 1 testcase × 2 candidates × 2 repeats = 4
        assert len(records) == 4

    def test_consistency_records_have_scores(self, tmp_path):
        config_path = str(_write_config(tmp_path, overrides={
            "protocol": {
                "evaluation_mode": "pairwise",
                "metrics": ["accuracy"],
                "aggregation": {"method": "mean"},
                "blinding": {"enabled": True, "random_seed": 42},
                "repeats": {"inference_repeats": 2, "judge_repeats": 1},
                "parallelism": {
                    "inference_max_concurrency": 1,
                    "judge_max_concurrency": 1,
                    "consistency_max_concurrency": 1,
                },
            },
        }))

        inf_out = _run_inference_mocked(
            config_path, str(tmp_path / "inference.jsonl"),
        )
        con_out = _run_judge_consistency_mocked(
            config_path, str(inf_out), str(tmp_path / "consistency.jsonl"),
        )

        records = read_jsonl(con_out)
        assert len(records) > 0
        for rec in records:
            assert rec["scores"]["overall"] == 4.0
            assert rec["repeat_count"] == 2

    def test_compare_without_judgements(self, tmp_path):
        """compare should succeed with only consistency data (no judgements)."""
        config_path = str(_write_config(tmp_path, overrides={
            "protocol": {
                "evaluation_mode": "pairwise",
                "metrics": ["accuracy"],
                "aggregation": {"method": "mean"},
                "blinding": {"enabled": True, "random_seed": 42},
                "repeats": {"inference_repeats": 2, "judge_repeats": 1},
                "parallelism": {
                    "inference_max_concurrency": 1,
                    "judge_max_concurrency": 1,
                    "consistency_max_concurrency": 1,
                },
            },
        }))

        inf_out = _run_inference_mocked(
            config_path, str(tmp_path / "inference.jsonl"),
        )
        con_out = _run_judge_consistency_mocked(
            config_path, str(inf_out), str(tmp_path / "consistency.jsonl"),
        )
        cmp_out = _run_compare(
            config_path,
            consistency_path=str(con_out),
            output_path=str(tmp_path / "report.json"),
        )

        with open(cmp_out) as f:
            report = json.load(f)

        assert report["run_id"] == "e2e-test"
        # inference_consistency should be populated
        consistency_data = report["results"]["overall"].get("inference_consistency", {})
        assert len(consistency_data) > 0


# ── E2E: CLI judge command auto-routing ──────────────────────────────────────


class TestJudgeCommandRouting:
    """Verify that `judge` CLI command auto-selects mode based on inference_repeats."""

    def test_metrics_mode_selected_when_repeats_1(self, tmp_path):
        config_path = str(_write_config(tmp_path))

        inf_out = _run_inference_mocked(
            config_path, str(tmp_path / "inference.jsonl"),
        )

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "llm_judge.stages.judge.get_thread_local_openai_client",
                    return_value=MagicMock(),
                )
            )
            stack.enter_context(
                patch(
                    "llm_judge.stages.judge.judge_chat_completion",
                    return_value=_mock_judge_response_pairwise(),
                )
            )
            stack.enter_context(
                patch("llm_judge.stages.judge.validate_artifacts")
            )

            from llm_judge.cli import app
            from typer.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(app, [
                "judge",
                "--config", config_path,
                "--inference", str(inf_out),
                "--output", str(tmp_path / "judge-out.jsonl"),
            ])

        assert result.exit_code == 0
        assert "Judge (metrics)" in result.output

    def test_consistency_mode_selected_when_repeats_2(self, tmp_path):
        config_path = str(_write_config(tmp_path, overrides={
            "protocol": {
                "evaluation_mode": "pairwise",
                "metrics": ["accuracy"],
                "aggregation": {"method": "mean"},
                "blinding": {"enabled": True, "random_seed": 42},
                "repeats": {"inference_repeats": 2, "judge_repeats": 1},
                "parallelism": {
                    "inference_max_concurrency": 1,
                    "judge_max_concurrency": 1,
                    "consistency_max_concurrency": 1,
                },
            },
        }))

        inf_out = _run_inference_mocked(
            config_path, str(tmp_path / "inference.jsonl"),
        )

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "llm_judge.stages.consistency.get_thread_local_openai_client",
                    return_value=MagicMock(),
                )
            )
            stack.enter_context(
                patch(
                    "llm_judge.stages.consistency.judge_chat_completion",
                    return_value=_mock_consistency_response(),
                )
            )
            stack.enter_context(
                patch("llm_judge.artifact_validation.validate_artifacts")
            )

            from llm_judge.cli import app
            from typer.testing import CliRunner

            runner = CliRunner()
            result = runner.invoke(app, [
                "judge",
                "--config", config_path,
                "--inference", str(inf_out),
                "--output", str(tmp_path / "consistency-out.jsonl"),
            ])

        assert result.exit_code == 0
        assert "Judge (consistency)" in result.output
