"""Tests for prompt_hash recording in inference (Task E)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_judge.models import (
    Aggregation,
    DatasetConfig,
    InferenceRecord,
    ModelRef,
    Protocol,
    Repeats,
    RunConfig,
    Testcase,
)
from llm_judge.stages.inference import _call_model
from llm_judge.utils import content_hash


def _make_cfg() -> RunConfig:
    return RunConfig(
        run_id="test-run",
        dataset=DatasetConfig(testcases_path="test.jsonl"),
        candidates=[
            ModelRef(candidate_id="c0", vendor="openai", model_id="gpt-4o"),
        ],
        judges=[],
        protocol=Protocol(
            evaluation_mode="absolute",
            aggregation=Aggregation(method="mean"),
        ),
    )


def _make_tc(json_schema_ref: str | None = None) -> Testcase:
    constraints = None
    if json_schema_ref is not None:
        from llm_judge.models import Constraints, OutputFormat

        constraints = Constraints(
            output_format=OutputFormat(type="json", json_schema_ref=json_schema_ref)
        )
    return Testcase(
        testcase_id="tc-001",
        task_type="preprocessing",
        input={"text": "hello"},
        constraints=constraints,
    )


def _make_candidate(vendor: str = "tsuzumi2") -> ModelRef:
    return ModelRef(
        candidate_id="cand-1",
        vendor=vendor,
        model_id="test-model",
        prompt_version="v1",
    )


class TestPromptHashSuccess:
    def test_prompt_hash_set_on_success(self) -> None:
        """prompt_hash is set in successful InferenceRecord."""
        cfg = _make_cfg()
        tc = _make_tc()
        candidate = _make_candidate()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "hello"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]

        with patch("llm_judge.stages.inference.chat_completion", return_value=mock_response):
            record = _call_model(
                cfg=cfg, tc=tc, candidate=candidate,
                client=MagicMock(), messages=messages, gen_params={},
            )

        assert record.status.ok is True
        assert record.prompt is not None
        assert record.prompt.prompt_hash is not None
        assert record.prompt.prompt_hash == content_hash(str(messages))

    def test_prompt_hash_stable(self) -> None:
        """Same messages → same prompt_hash."""
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
        h1 = content_hash(str(messages))
        h2 = content_hash(str(messages))
        assert h1 == h2


class TestPromptHashFailure:
    def test_prompt_hash_set_on_failure(self) -> None:
        """prompt_hash is set even when inference fails."""
        cfg = _make_cfg()
        tc = _make_tc()
        candidate = _make_candidate()
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]

        with patch(
            "llm_judge.stages.inference.chat_completion",
            side_effect=RuntimeError("connection failed"),
        ):
            record = _call_model(
                cfg=cfg, tc=tc, candidate=candidate,
                client=MagicMock(), messages=messages, gen_params={},
            )

        assert record.status.ok is False
        assert record.prompt is not None
        assert record.prompt.prompt_hash is not None
        assert record.prompt.prompt_version == "v1"


class TestPromptHashStructuredOutput:
    def test_prompt_hash_differs_with_structured_output(self, tmp_path) -> None:
        """When structured output changes messages, prompt_hash should differ."""
        import json

        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps({
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "additionalProperties": False,
        }))

        cfg = _make_cfg()
        tc = _make_tc(json_schema_ref=str(schema_path))
        candidate = _make_candidate(vendor="openai")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "ok"}'
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        messages = [{"role": "system", "content": "original system"}, {"role": "user", "content": "usr"}]
        original_hash = content_hash(str(messages))

        with patch("llm_judge.stages.inference.chat_completion", return_value=mock_response):
            record = _call_model(
                cfg=cfg, tc=tc, candidate=candidate,
                client=MagicMock(), messages=messages, gen_params={},
            )

        assert record.status.ok is True
        assert record.prompt.prompt_hash is not None
        # Structured output overrides system message, so hash should differ
        assert record.prompt.prompt_hash != original_hash
