from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from llm_judge.stages.autocheck import run_autocheck
from llm_judge.utils import (
    build_stage_output_path,
    derive_dataset_layer,
    find_latest_stage_output_path,
    resolve_stage_input_path,
    sanitize_path_component,
    strip_run_id_timestamp,
)


def test_derive_dataset_layer_real_core() -> None:
    assert derive_dataset_layer("data/eval/real-core/testcases.jsonl") == "real-core"


def test_derive_dataset_layer_synthetic_metric() -> None:
    assert derive_dataset_layer("data/eval/synthetic-metric/testcases.jsonl") == "synthetic-metric"


def test_derive_dataset_layer_synthetic_adversarial() -> None:
    assert derive_dataset_layer("data/eval/synthetic-adversarial/testcases.jsonl") == "synthetic-adversarial"


def test_derive_dataset_layer_fallback() -> None:
    assert derive_dataset_layer("data/testcases.jsonl") == "data"


def test_find_latest_stage_output_path_with_dataset_layer(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    d = results_dir / "real-20260313-101500"
    d.mkdir(parents=True)
    target = d / "inference.jsonl"
    target.write_text("", encoding="utf-8")

    resolved = find_latest_stage_output_path(
        "inference",
        "any-run-id",
        ".jsonl",
        results_dir=results_dir,
        dataset_layer="real",
    )
    assert resolved == target


def test_strip_run_id_timestamp_removes_trailing_date() -> None:
    assert strip_run_id_timestamp("synthetic-metric-main-20260310") == "synthetic-metric-main"
    assert strip_run_id_timestamp("synthetic-metric-main-20260310-101530") == "synthetic-metric-main"
    assert strip_run_id_timestamp("local-openai-4o-51") == "local-openai-4o-51"


def test_build_stage_output_path_uses_execution_start_datetime() -> None:
    started_at = datetime(2026, 3, 13, 10, 15, 30, tzinfo=timezone.utc)
    path = build_stage_output_path(
        "comparison-report",
        "synthetic-adversarial-main-20260310",
        ".json",
        started_at=started_at,
    )
    assert path == Path("data/comparison-report-synthetic-adversarial-main-20260313-101530.json")


def test_sanitize_path_component_removes_glob_and_path_chars() -> None:
    assert sanitize_path_component("../bad*[name]?") == "bad_name"
    assert sanitize_path_component("///") == "run"


def test_build_stage_output_path_sanitizes_run_id() -> None:
    started_at = datetime(2026, 3, 13, 10, 15, 30, tzinfo=timezone.utc)
    path = build_stage_output_path(
        "inference",
        "../bad*[name]?-20260310",
        ".jsonl",
        started_at=started_at,
    )
    assert path == Path("data/inference-bad_name-20260313-101530.jsonl")


def test_find_latest_stage_output_path_prefers_newest_timestamped_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    legacy = data_dir / "inference-demo-run-20260310.jsonl"
    older = data_dir / "inference-demo-run-20260313-090000.jsonl"
    newer = data_dir / "inference-demo-run-20260313-101500.jsonl"
    for path in (legacy, older, newer):
        path.write_text("", encoding="utf-8")

    resolved = find_latest_stage_output_path(
        "inference",
        "demo-run-20260310",
        ".jsonl",
        data_dir=data_dir,
    )

    assert resolved == newer


def test_find_latest_stage_output_path_does_not_expand_glob_chars(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    matching = data_dir / "inference-demo_a_b-20260313-101500.jsonl"
    unrelated = data_dir / "inference-demo-a-20260313-111500.jsonl"
    matching.write_text("", encoding="utf-8")
    unrelated.write_text("", encoding="utf-8")

    resolved = find_latest_stage_output_path(
        "inference",
        "demo[a]b-20260310",
        ".jsonl",
        data_dir=data_dir,
    )

    assert resolved == matching


def test_resolve_stage_input_path_raises_when_missing(tmp_path: Path) -> None:
    try:
        resolve_stage_input_path(
            "judgements",
            "missing-run-20260310",
            ".jsonl",
            data_dir=tmp_path / "data",
        )
    except FileNotFoundError as exc:
        assert "missing-run-20260310" in str(exc)
    else:
        raise AssertionError("FileNotFoundError was not raised")


def test_resolve_stage_input_path_does_not_escape_data_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside = tmp_path / "judgements-secret.jsonl"
    outside.write_text("{}", encoding="utf-8")

    try:
        resolve_stage_input_path(
            "judgements",
            "../../secret",
            ".jsonl",
            data_dir=data_dir,
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("FileNotFoundError was not raised")


def test_run_autocheck_uses_latest_timestamped_inference_and_started_at_for_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    testcase_path = tmp_path / "testcases.jsonl"
    testcase_path.write_text(
        json.dumps(
            {
                "testcase_id": "tc-001",
                "task_type": "preprocessing",
                "input": {"text": "hello"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "run-config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "demo-run-20260310",
                "dataset": {"testcases_path": str(testcase_path)},
                "candidates": [
                    {
                        "candidate_id": "cand-1",
                        "vendor": "openai",
                        "model_id": "gpt-4o",
                    }
                ],
                "judges": [
                    {
                        "judge_id": "judge-1",
                        "vendor": "openai",
                        "model_id": "gpt-4o",
                        "rubric_version": "v1",
                    }
                ],
                "protocol": {
                    "evaluation_mode": "absolute",
                    "aggregation": {"method": "mean"},
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    older_inference = data_dir / "inference-demo-run-20260313-090000.jsonl"
    newer_inference = data_dir / "inference-demo-run-20260313-101500.jsonl"
    older_inference.write_text(
        json.dumps(
            {
                "run_id": "demo-run-20260310",
                "testcase_id": "tc-001",
                "candidate_id": "cand-1",
                "model": {"vendor": "openai", "model_id": "gpt-4o"},
                "output": {"text": "stale"},
                "status": {"ok": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    newer_inference.write_text(
        json.dumps(
            {
                "run_id": "demo-run-20260310",
                "testcase_id": "tc-001",
                "candidate_id": "cand-1",
                "model": {"vendor": "openai", "model_id": "gpt-4o"},
                "output": {"text": "fresh"},
                "status": {"ok": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    started_at = datetime(2026, 3, 13, 12, 0, 0, tzinfo=timezone.utc)
    out = run_autocheck(str(config_path), started_at=started_at)

    assert out == Path("data/autocheck-demo-run-20260313-120000.jsonl")
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["testcase_id"] == "tc-001"
    assert record["candidate_id"] == "cand-1"
