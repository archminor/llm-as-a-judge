"""Tests for candidate count validation by evaluation_mode."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_judge.config import load_run_config
from llm_judge.models import JudgeRef, ModelRef, Parallelism, RunConfig


def _candidate(cid: str) -> dict:
    return {
        "candidate_id": cid,
        "vendor": "openai",
        "model_id": "gpt-4o",
    }


def _base_config(mode: str, num_candidates: int) -> dict:
    candidates = [_candidate(f"c{i}") for i in range(num_candidates)]
    return {
        "run_id": "test-run",
        "dataset": {"testcases_path": "data/testcases.jsonl"},
        "candidates": candidates,
        "judges": [
            {
                "judge_id": "j1",
                "vendor": "openai",
                "model_id": "gpt-4o",
                "rubric_version": "v1",
            }
        ],
        "protocol": {
            "evaluation_mode": mode,
            "aggregation": {"method": "mean"},
        },
    }


# ── Pydantic (model_validate) tests ──


class TestRunConfigPydanticValidation:
    def test_absolute_1_candidate_ok(self):
        cfg = _base_config("absolute", 1)
        rc = RunConfig.model_validate(cfg)
        assert len(rc.candidates) == 1

    def test_absolute_2_candidates_ok(self):
        cfg = _base_config("absolute", 2)
        rc = RunConfig.model_validate(cfg)
        assert len(rc.candidates) == 2

    def test_absolute_0_candidates_fails(self):
        cfg = _base_config("absolute", 0)
        with pytest.raises(Exception):
            RunConfig.model_validate(cfg)

    def test_pairwise_1_candidate_fails(self):
        cfg = _base_config("pairwise", 1)
        with pytest.raises(Exception, match="at least 2 candidates"):
            RunConfig.model_validate(cfg)

    def test_pairwise_2_candidates_ok(self):
        cfg = _base_config("pairwise", 2)
        rc = RunConfig.model_validate(cfg)
        assert len(rc.candidates) == 2

    def test_hybrid_1_candidate_fails(self):
        cfg = _base_config("hybrid", 1)
        with pytest.raises(Exception, match="at least 2 candidates"):
            RunConfig.model_validate(cfg)

    def test_hybrid_2_candidates_ok(self):
        cfg = _base_config("hybrid", 2)
        rc = RunConfig.model_validate(cfg)
        assert len(rc.candidates) == 2


# ── load_run_config (YAML + JSON Schema + Pydantic) tests ──


class TestLoadRunConfigValidation:
    def _write_yaml(self, tmp_path: Path, data: dict) -> Path:
        p = tmp_path / "run-config.yaml"
        p.write_text(yaml.dump(data, allow_unicode=True))
        return p

    def test_absolute_1_candidate_loads(self, tmp_path):
        cfg = _base_config("absolute", 1)
        path = self._write_yaml(tmp_path, cfg)
        rc = load_run_config(path)
        assert len(rc.candidates) == 1

    def test_pairwise_1_candidate_rejected(self, tmp_path):
        cfg = _base_config("pairwise", 1)
        path = self._write_yaml(tmp_path, cfg)
        with pytest.raises(Exception):
            load_run_config(path)

    def test_hybrid_1_candidate_rejected(self, tmp_path):
        cfg = _base_config("hybrid", 1)
        path = self._write_yaml(tmp_path, cfg)
        with pytest.raises(Exception):
            load_run_config(path)

    def test_pairwise_2_candidates_loads(self, tmp_path):
        cfg = _base_config("pairwise", 2)
        path = self._write_yaml(tmp_path, cfg)
        rc = load_run_config(path)
        assert len(rc.candidates) == 2


class TestParallelismValidation:
    def test_parallelism_defaults_to_1_when_omitted(self):
        cfg = _base_config("absolute", 1)
        rc = RunConfig.model_validate(cfg)
        assert rc.protocol.parallelism.inference_max_concurrency == 1
        assert rc.protocol.parallelism.judge_max_concurrency == 1
        assert rc.protocol.parallelism.consistency_max_concurrency == 1

    def test_parallelism_model_defaults(self):
        p = Parallelism()
        assert p.inference_max_concurrency == 1
        assert p.judge_max_concurrency == 1
        assert p.consistency_max_concurrency == 1

    def test_parallelism_accepts_high_value(self):
        cfg = _base_config("absolute", 1)
        cfg["protocol"]["parallelism"] = {"inference_max_concurrency": 100}
        rc = RunConfig.model_validate(cfg)
        assert rc.protocol.parallelism.inference_max_concurrency == 100

    def test_parallelism_rejects_zero(self):
        with pytest.raises(Exception):
            Parallelism(inference_max_concurrency=0)

    def test_parallelism_rejects_negative(self):
        with pytest.raises(Exception):
            Parallelism(inference_max_concurrency=-1)

    def test_parallelism_rejects_zero_via_run_config(self):
        cfg = _base_config("absolute", 1)
        cfg["protocol"]["parallelism"] = {"inference_max_concurrency": 0}
        with pytest.raises(Exception):
            RunConfig.model_validate(cfg)

    def test_existing_config_loads_without_parallelism(self, tmp_path):
        cfg = _base_config("absolute", 1)
        path = tmp_path / "run-config.yaml"
        path.write_text(yaml.dump(cfg, allow_unicode=True))
        rc = load_run_config(path)
        assert rc.protocol.parallelism.inference_max_concurrency == 1

    def test_schema_and_pydantic_agree_on_parallelism(self, tmp_path):
        cfg = _base_config("absolute", 1)
        cfg["protocol"]["parallelism"] = {
            "inference_max_concurrency": 4,
            "judge_max_concurrency": 2,
            "consistency_max_concurrency": 3,
        }
        path = tmp_path / "run-config.yaml"
        path.write_text(yaml.dump(cfg, allow_unicode=True))
        rc = load_run_config(path)
        assert rc.protocol.parallelism.inference_max_concurrency == 4
        assert rc.protocol.parallelism.judge_max_concurrency == 2
        assert rc.protocol.parallelism.consistency_max_concurrency == 3


class TestGenerationParamsValidation:
    def test_model_ref_rejects_both_token_keys(self):
        with pytest.raises(Exception, match="max_tokens and max_completion_tokens"):
            ModelRef.model_validate(
                {
                    "candidate_id": "cand-1",
                    "vendor": "openai",
                    "model_id": "gpt-5-mini",
                    "generation_params": {
                        "max_tokens": 1024,
                        "max_completion_tokens": 2048,
                    },
                }
            )

    def test_judge_ref_rejects_both_token_keys(self):
        with pytest.raises(Exception, match="max_tokens and max_completion_tokens"):
            JudgeRef.model_validate(
                {
                    "judge_id": "judge-1",
                    "vendor": "openai",
                    "model_id": "gpt-5-mini",
                    "rubric_version": "v1",
                    "generation_params": {
                        "max_tokens": 1024,
                        "max_completion_tokens": 2048,
                    },
                }
            )
