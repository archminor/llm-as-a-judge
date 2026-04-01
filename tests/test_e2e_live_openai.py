"""Live E2E tests using real OpenAI API calls.

Requires OPENAI_API_KEY in .env. Skipped if not set.
These tests are slow (real API calls) — run explicitly:
    uv run python -m pytest tests/test_e2e_live_openai.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from dotenv import load_dotenv

from llm_judge.utils import read_jsonl

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

ALL_TESTCASES = [
    {
        "testcase_id": "live-qa-001",
        "task_type": "qa",
        "input": {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
            ],
        },
    },
    {
        "testcase_id": "live-summarization-001",
        "task_type": "summarization",
        "input": {"text": "The quick brown fox jumps over the lazy dog. This sentence is a pangram."},
        "constraints": {"required_points": ["mention the fox"]},
    },
    {
        "testcase_id": "live-analysis-001",
        "task_type": "analysis",
        "input": {
            "messages": [
                {"role": "system", "content": "You are a business analyst."},
                {"role": "user", "content": "What are the pros and cons of remote work?"},
            ],
        },
    },
]

N_TESTCASES = len(ALL_TESTCASES)


# ── helpers ───────────────────────────────────────────────────────────────────


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_config(tmp_path: Path, inference_repeats: int = 1) -> Path:
    testcases_path = tmp_path / "testcases.jsonl"
    _write_jsonl(testcases_path, ALL_TESTCASES)

    cfg = {
        "run_id": "live-e2e",
        "dataset": {"testcases_path": str(testcases_path)},
        "candidates": [
            {
                "candidate_id": "openai-4o-mini",
                "vendor": "openai",
                "model_id": "gpt-4o-mini",
                "generation_params": {"temperature": 0, "max_tokens": 128},
            },
        ],
        "judges": [
            {
                "judge_id": "openai-judge",
                "vendor": "openai",
                "model_id": "gpt-4o-mini",
                "rubric_version": "v1",
            },
        ],
        "protocol": {
            "evaluation_mode": "absolute",
            "metrics": ["accuracy", "completeness", "relevance", "format_compliance"],
            "aggregation": {"method": "mean"},
            "repeats": {
                "inference_repeats": inference_repeats,
                "judge_repeats": 1,
            },
        },
    }

    config_path = tmp_path / "run-config.yaml"
    config_path.write_text(yaml.dump(cfg, allow_unicode=True))
    return config_path


# ── metrics mode (inference_repeats=1) ───────────────────────────────────────


class TestLiveMetricsMode:
    """qa / summarization / analysis × metrics mode."""

    def test_full_pipeline(self, tmp_path):
        config_path = str(_write_config(tmp_path, inference_repeats=1))

        from llm_judge.stages.inference import run_inference

        inf_out = run_inference(config_path, str(tmp_path / "inference.jsonl"))
        assert inf_out.exists()

        inf_records = read_jsonl(inf_out)
        assert len(inf_records) == N_TESTCASES
        for rec in inf_records:
            assert rec["status"]["ok"] is True, (
                f"{rec['testcase_id']} failed: {rec['status']}"
            )
            assert len(rec["output"]["text"]) > 0

        from llm_judge.stages.autocheck import run_autocheck

        ac_out = run_autocheck(
            config_path, str(inf_out), str(tmp_path / "autocheck.jsonl"),
        )
        assert ac_out.exists()

        from llm_judge.stages.judge import run_judge

        jdg_out = run_judge(
            config_path, str(inf_out), str(tmp_path / "judgements.jsonl"),
        )
        assert jdg_out.exists()

        jdg_records = read_jsonl(jdg_out)
        assert len(jdg_records) == N_TESTCASES
        for rec in jdg_records:
            per_metric = rec["scores"]["per_metric"]
            for metric in ["accuracy", "completeness", "relevance"]:
                assert metric in per_metric, f"{rec['testcase_id']} missing {metric}"

        from llm_judge.stages.compare import run_compare

        cmp_out = run_compare(
            config_path,
            str(jdg_out),
            output_path=str(tmp_path / "report.json"),
        )
        assert cmp_out.exists()
        assert cmp_out.with_suffix(".md").exists()

        with open(cmp_out) as f:
            report = json.load(f)
        assert report["run_id"] == "live-e2e"
        assert report["summary"]["valid_judgements"] == N_TESTCASES


# ── consistency mode (inference_repeats=2) ───────────────────────────────────


class TestLiveConsistencyMode:
    """qa / summarization / analysis × consistency mode."""

    def test_full_pipeline(self, tmp_path):
        config_path = str(_write_config(tmp_path, inference_repeats=2))

        from llm_judge.stages.inference import run_inference

        inf_out = run_inference(config_path, str(tmp_path / "inference.jsonl"))
        assert inf_out.exists()

        inf_records = read_jsonl(inf_out)
        # 3 testcases × 1 candidate × 2 repeats = 6
        assert len(inf_records) == N_TESTCASES * 2
        ok_records = [r for r in inf_records if r["status"]["ok"]]
        assert len(ok_records) == N_TESTCASES * 2

        from llm_judge.stages.autocheck import run_autocheck

        ac_out = run_autocheck(
            config_path, str(inf_out), str(tmp_path / "autocheck.jsonl"),
        )
        assert ac_out.exists()

        from llm_judge.stages.consistency import run_consistency

        con_out = run_consistency(
            config_path, str(inf_out), str(tmp_path / "consistency.jsonl"),
        )
        assert con_out.exists()

        con_records = read_jsonl(con_out)
        # 3 testcases × 1 candidate × 1 judge = 3
        assert len(con_records) == N_TESTCASES
        for rec in con_records:
            assert rec["status"]["ok"] is True, (
                f"{rec['testcase_id']} consistency failed: {rec['status']}"
            )
            assert rec["repeat_count"] == 2
            assert 1.0 <= rec["scores"]["overall"] <= 5.0

        from llm_judge.stages.compare import run_compare

        cmp_out = run_compare(
            config_path,
            consistency_path=str(con_out),
            output_path=str(tmp_path / "report.json"),
        )
        assert cmp_out.exists()

        with open(cmp_out) as f:
            report = json.load(f)
        assert report["run_id"] == "live-e2e"
        consistency_data = report["results"]["overall"].get("inference_consistency", {})
        assert "openai-4o-mini" in consistency_data
