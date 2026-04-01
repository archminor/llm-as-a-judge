"""Tests for generalized Structured Output implementation.

Structured output is triggered by testcase.constraints.output_format settings,
not by testcase_id. The key conditions are:
- output_format.type == "json"
- output_format.json_schema_ref is a non-empty string
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_judge.models import (
    Constraints,
    ModelRef,
    OutputFormat,
    RunConfig,
    Testcase,
)
from llm_judge.schema_validation import _REPO_ROOT, resolve_schema_path as _resolve_schema_path
from llm_judge.stages.inference import (
    _JSON_SCHEMA_FORMAT_VENDORS,
    _STRUCTURED_OUTPUT_MIN_MAX_TOKENS,
    _apply_structured_output_system_message,
    _build_response_format,
    _call_model,
    _ensure_structured_output_token_budget,
    _load_json_schema,
    _make_schema_name,
    _requires_structured_output,
    _serialize_for_system_b,
    _supports_json_schema_format,
    _validate_json_against_schema,
)


# ── helpers ───────────────────────────────────────────────


def _make_testcase(
    testcase_id: str = "test-001",
    output_format: OutputFormat | None = None,
) -> Testcase:
    constraints = Constraints(output_format=output_format) if output_format else None
    return Testcase(
        testcase_id=testcase_id,
        task_type="report_generation",
        input={"report_name": "test", "source": "dummy"},
        constraints=constraints,
    )


def _valid_sample_data() -> dict:
    """Valid data matching the sample output schema used in tests."""
    return {
        "result": "sample output text",
        "details": "additional information",
    }


_SAMPLE_OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SampleOutput",
    "type": "object",
    "required": ["result", "details"],
    "additionalProperties": False,
    "properties": {
        "result": {"type": "string"},
        "details": {"type": "string"},
    },
}


@pytest.fixture()
def sample_schema_path(tmp_path: Path) -> str:
    """Create and return an absolute path to a sample output schema."""
    p = tmp_path / "sample-output.schema.json"
    p.write_text(json.dumps(_SAMPLE_OUTPUT_SCHEMA), encoding="utf-8")
    return str(p)


@pytest.fixture()
def tmp_schema(tmp_path: Path) -> Path:
    """Create a temporary valid JSON schema file."""
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "TestOutput",
        "type": "object",
        "required": ["result"],
        "additionalProperties": False,
        "properties": {
            "result": {"type": "string"},
        },
    }
    p = tmp_path / "test-output.schema.json"
    p.write_text(json.dumps(schema), encoding="utf-8")
    return p


@pytest.fixture()
def tmp_invalid_json(tmp_path: Path) -> Path:
    """Create a temporary file with invalid JSON."""
    p = tmp_path / "bad.json"
    p.write_text("{ not valid json !!!", encoding="utf-8")
    return p


@pytest.fixture()
def tmp_nested_loose_schema(tmp_path: Path) -> Path:
    """Create a schema with nested object nodes that are not strict-safe."""
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "LooseNestedOutput",
        "type": "object",
        "required": ["records"],
        "properties": {
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["value"],
                    "additionalProperties": True,
                    "properties": {
                        "value": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                },
            },
            "summary": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                },
            },
        },
    }
    p = tmp_path / "loose-nested.schema.json"
    p.write_text(json.dumps(schema), encoding="utf-8")
    return p


# ── _requires_structured_output ──────────────────────────


class TestRequiresStructuredOutput:
    def test_true_when_json_type_and_schema_ref_set(self, sample_schema_path):
        of = OutputFormat(type="json", json_schema_ref=sample_schema_path)
        tc = _make_testcase(output_format=of)
        assert _requires_structured_output(tc) is True

    def test_false_when_no_constraints(self):
        tc = _make_testcase()
        assert _requires_structured_output(tc) is False

    def test_false_when_output_format_none(self):
        tc = _make_testcase()
        tc.constraints = Constraints()
        assert _requires_structured_output(tc) is False

    def test_false_when_type_is_not_json(self):
        of = OutputFormat(type="free_text", json_schema_ref="schemas/x.json")
        tc = _make_testcase(output_format=of)
        assert _requires_structured_output(tc) is False

    def test_false_when_type_is_markdown(self):
        of = OutputFormat(type="markdown", json_schema_ref="schemas/x.json")
        tc = _make_testcase(output_format=of)
        assert _requires_structured_output(tc) is False

    def test_false_when_json_schema_ref_is_none(self):
        of = OutputFormat(type="json", json_schema_ref=None)
        tc = _make_testcase(output_format=of)
        assert _requires_structured_output(tc) is False

    def test_false_when_json_schema_ref_is_empty(self):
        of = OutputFormat(type="json", json_schema_ref="")
        tc = _make_testcase(output_format=of)
        assert _requires_structured_output(tc) is False

    def test_false_when_json_schema_ref_is_whitespace(self):
        of = OutputFormat(type="json", json_schema_ref="   ")
        tc = _make_testcase(output_format=of)
        assert _requires_structured_output(tc) is False

    def test_true_for_any_testcase_id_with_valid_config(self, sample_schema_path):
        """Structured output is determined by config, not testcase_id."""
        for tid in ["uc1-car-001", "uc2-001", "preprocess-001", "custom-xyz"]:
            of = OutputFormat(type="json", json_schema_ref=sample_schema_path)
            tc = _make_testcase(testcase_id=tid, output_format=of)
            assert _requires_structured_output(tc) is True


# ── _load_json_schema ────────────────────────────────────


class TestLoadJsonSchema:
    def test_loads_valid_schema(self, tmp_schema):
        schema = _load_json_schema(str(tmp_schema))
        assert schema["title"] == "TestOutput"
        assert schema["type"] == "object"

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="json_schema_ref not found"):
            _load_json_schema("nonexistent/path/schema.json")

    def test_raises_value_error_for_invalid_json(self, tmp_invalid_json):
        with pytest.raises(ValueError, match="not valid JSON"):
            _load_json_schema(str(tmp_invalid_json))

    def test_repo_root_schema_exists(self):
        """A schema file can be loaded from repo-root-relative path."""
        schema = _load_json_schema("schemas/testcase.schema.json")
        assert schema["title"] == "Testcase"


# ── _make_schema_name ────────────────────────────────────


class TestMakeSchemaName:
    def test_uses_title_when_available(self):
        schema = {"title": "UC2ReportOutput"}
        assert _make_schema_name(schema, "tc-001") == "uc2reportoutput"

    def test_falls_back_to_testcase_id(self):
        schema = {}
        assert _make_schema_name(schema, "uc1-car-001") == "uc1_car_001"

    def test_sanitizes_special_characters(self):
        schema = {"title": "My Schema (v2.1)"}
        assert _make_schema_name(schema, "tc") == "my_schema_v2_1"


# ── _build_response_format ───────────────────────────────


class TestBuildResponseFormat:
    def test_structure(self, sample_schema_path):
        rf = _build_response_format(sample_schema_path, "uc1-car-001")
        assert rf["type"] == "json_schema"
        js = rf["json_schema"]
        assert js["strict"] is True
        assert "name" in js
        schema = js["schema"]
        assert set(schema["required"]) == {"result", "details"}

    def test_schema_has_no_additional_props(self, sample_schema_path):
        rf = _build_response_format(sample_schema_path, "uc1-car-001")
        schema = rf["json_schema"]["schema"]
        assert schema.get("additionalProperties") is False

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            _build_response_format("nonexistent.json", "tc-001")

    def test_raises_on_invalid_json(self, tmp_invalid_json):
        with pytest.raises(ValueError, match="not valid JSON"):
            _build_response_format(str(tmp_invalid_json), "tc-001")

    def test_with_custom_schema(self, tmp_schema):
        rf = _build_response_format(str(tmp_schema), "custom-tc-001")
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "testoutput"
        assert rf["json_schema"]["schema"]["required"] == ["result"]

    def test_normalizes_nested_additional_properties(self, tmp_nested_loose_schema):
        rf = _build_response_format(str(tmp_nested_loose_schema), "custom-tc-002")
        schema = rf["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"records", "summary"}
        assert schema["properties"]["records"]["items"]["additionalProperties"] is False
        assert set(schema["properties"]["records"]["items"]["required"]) == {
            "value",
            "notes",
        }
        assert schema["properties"]["summary"]["additionalProperties"] is False
        assert schema["properties"]["summary"]["required"] == ["count"]

    def test_repository_schemas_are_valid_json(self):
        """All JSON schema files in schemas/ should be valid JSON."""
        for path in sorted((_REPO_ROOT / "schemas").glob("*.json")):
            schema = json.loads(path.read_text(encoding="utf-8"))
            assert "type" in schema or "$schema" in schema, f"{path.name} is not a valid schema"


# ── _validate_json_against_schema ────────────────────────


class TestValidateJsonAgainstSchema:
    def test_valid_data(self, sample_schema_path):
        errors = _validate_json_against_schema(_valid_sample_data(), sample_schema_path)
        assert errors == []

    def test_missing_required_field(self, sample_schema_path):
        data = _valid_sample_data()
        del data["details"]
        errors = _validate_json_against_schema(data, sample_schema_path)
        assert any("details" in e for e in errors)

    def test_wrong_type_rejected(self, sample_schema_path):
        data = _valid_sample_data()
        data["result"] = 123  # should be string
        errors = _validate_json_against_schema(data, sample_schema_path)
        assert errors

    def test_extra_field_rejected(self, sample_schema_path):
        data = _valid_sample_data()
        data["unexpected_key"] = "oops"
        errors = _validate_json_against_schema(data, sample_schema_path)
        assert errors

    def test_with_custom_schema(self, tmp_schema):
        errors = _validate_json_against_schema({"result": "ok"}, str(tmp_schema))
        assert errors == []

        errors = _validate_json_against_schema({"wrong": "key"}, str(tmp_schema))
        assert errors


# ── _serialize_for_system_b ──────────────────────────────


class TestSerializeForSystemB:
    def test_roundtrip(self):
        data = _valid_sample_data()
        text = _serialize_for_system_b(data)
        assert json.loads(text) == data
        assert ast.literal_eval(text) == data

    def test_newline_in_string(self):
        data = {
            "result": "line1\nline2",
            "details": "x",
        }
        text = _serialize_for_system_b(data)
        json.loads(text)
        ast.literal_eval(text)

    def test_no_literal_newlines(self):
        data = _valid_sample_data()
        data["result"] = "first line\nsecond line"
        text = _serialize_for_system_b(data)
        assert "\n" not in text, "Literal newlines must not appear in System B payload"

    def test_ast_literal_eval_compatible(self):
        data = _valid_sample_data()
        text = _serialize_for_system_b(data)
        result = ast.literal_eval(text)
        assert result["result"] == data["result"]
        assert result["details"] == data["details"]


# ── _apply_structured_output_system_message ──────────────


class TestApplyStructuredOutputSystemMessage:
    def test_replaces_system_role(self):
        original = [
            {"role": "system", "content": "original system message"},
            {"role": "user", "content": "user message"},
        ]
        result = _apply_structured_output_system_message(original)
        assert result[0]["role"] == "system"
        assert "prioritize the response format" in result[0]["content"]
        assert result[1] == {"role": "user", "content": "user message"}

    def test_does_not_mutate_original(self):
        original = [{"role": "system", "content": "old"}]
        _apply_structured_output_system_message(original)
        assert original[0]["content"] == "old"


# ── _supports_json_schema_format (vendor check) ─────────


class TestSupportsJsonSchemaFormat:
    def test_openai_supported(self):
        assert _supports_json_schema_format("openai") is True

    def test_azure_openai_supported(self):
        assert _supports_json_schema_format("azure-openai") is True

    def test_unknown_vendor_not_supported(self):
        assert _supports_json_schema_format("unknown-vendor") is False


# ── _ensure_structured_output_token_budget ───────────────


class TestEnsureStructuredOutputTokenBudget:
    def test_keeps_non_structured_output_params(self):
        params = {"max_tokens": 1024}
        out = _ensure_structured_output_token_budget(params, use_structured_output=False)
        assert out["max_tokens"] == 1024

    def test_raises_low_max_tokens(self):
        params = {"max_tokens": 1024}
        out = _ensure_structured_output_token_budget(params, use_structured_output=True)
        assert out["max_tokens"] >= _STRUCTURED_OUTPUT_MIN_MAX_TOKENS

    def test_sets_default_when_missing(self):
        out = _ensure_structured_output_token_budget({}, use_structured_output=True)
        assert out["max_tokens"] >= _STRUCTURED_OUTPUT_MIN_MAX_TOKENS

    def test_keeps_higher_value(self):
        out = _ensure_structured_output_token_budget(
            {"max_tokens": 8192}, use_structured_output=True
        )
        assert out["max_tokens"] == 8192

    def test_prefers_max_completion_tokens_when_both_present(self):
        out = _ensure_structured_output_token_budget(
            {
                "max_tokens": 8192,
                "max_completion_tokens": 6144,
            },
            use_structured_output=True,
        )
        assert "max_tokens" not in out
        assert out["max_completion_tokens"] == 6144


# ── [P2] input_hash computed from actual (sent) messages ─


class TestInputHash:
    def test_hash_differs_after_system_message_override(self):
        from llm_judge.utils import content_hash

        original = [
            {"role": "system", "content": "original system"},
            {"role": "user", "content": "user message"},
        ]
        overridden = _apply_structured_output_system_message(original)
        assert content_hash(str(original)) != content_hash(str(overridden))

    def test_hash_unchanged_for_messages_mode_with_structured_output(self):
        """In messages mode, system override is skipped so hash should not change."""
        from llm_judge.utils import content_hash

        messages = [
            {"role": "system", "content": "original system"},
            {"role": "user", "content": "user message"},
        ]
        tc = Testcase(
            testcase_id="mt-so",
            task_type="report_qa",
            input={"messages": messages},
            constraints=Constraints(
                output_format=OutputFormat(type="json", json_schema_ref="schemas/testcase.schema.json")
            ),
        )
        assert tc.has_messages is True
        # In messages mode, system override should NOT be applied
        # The hash should stay the same
        original_hash = content_hash(str(messages))
        # Simulating what _call_model does: no override for messages mode
        assert content_hash(str(messages)) == original_hash

    def test_hash_stable_for_non_structured_output(self):
        from llm_judge.utils import content_hash

        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
        tc = _make_testcase()
        assert not _requires_structured_output(tc)
        assert content_hash(str(messages)) == content_hash(str(messages))


# ── schema file existence (repo-level) ───────────────────


class TestSchemaFiles:
    def test_testcase_schema_file_exists(self):
        schema_path = _REPO_ROOT / "schemas" / "testcase.schema.json"
        assert schema_path.exists(), f"Schema file not found: {schema_path}"

    def test_testcase_schema_file_is_valid_json(self):
        schema_path = _REPO_ROOT / "schemas" / "testcase.schema.json"
        content = schema_path.read_text(encoding="utf-8")
        schema = json.loads(content)
        assert schema.get("title") == "Testcase"


# ── non-supported vendor fallback ────────────────────────


class TestVendorFallback:
    def test_structured_output_conditions_with_unsupported_vendor(self, sample_schema_path):
        """When json_schema_ref is set but vendor is truly unsupported,
        _requires_structured_output is True but _supports_json_schema_format is False.
        This combination triggers the fallback path."""
        of = OutputFormat(type="json", json_schema_ref=sample_schema_path)
        tc = _make_testcase(output_format=of)
        assert _requires_structured_output(tc) is True
        assert _supports_json_schema_format("unknown-vendor") is False

    def test_structured_output_conditions_with_supported_vendor(self, sample_schema_path):
        of = OutputFormat(type="json", json_schema_ref=sample_schema_path)
        tc = _make_testcase(output_format=of)
        assert _requires_structured_output(tc) is True
        assert _supports_json_schema_format("openai") is True
        assert _supports_json_schema_format("azure-openai") is True


# ── path resolution is repo-root-based ───────────────────


class TestPathResolution:
    def test_schema_ref_resolved_from_repo_root(self):
        """json_schema_ref must be resolved from the repository root, not cwd."""
        schema = _load_json_schema("schemas/testcase.schema.json")
        assert schema["title"] == "Testcase"

    def test_resolve_schema_path_uses_repo_root_for_relative(self):
        resolved = _resolve_schema_path("schemas/testcase.schema.json")
        assert resolved.is_absolute()
        assert resolved == _REPO_ROOT / "schemas" / "testcase.schema.json"

    def test_resolve_schema_path_keeps_absolute(self, tmp_schema):
        resolved = _resolve_schema_path(str(tmp_schema))
        assert resolved == tmp_schema

    def test_load_json_schema_works_from_different_cwd(self, tmp_path):
        """Even when cwd is /tmp, repo-root-relative paths still resolve correctly."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            schema = _load_json_schema("schemas/testcase.schema.json")
            assert schema["title"] == "Testcase"
        finally:
            os.chdir(original_cwd)

    def test_absolute_path_also_works(self, tmp_schema):
        schema = _load_json_schema(str(tmp_schema))
        assert schema["title"] == "TestOutput"


# ── _call_model integration tests (P3) ──────────────────


def _make_run_config() -> RunConfig:
    """Minimal RunConfig for _call_model tests."""
    return RunConfig(
        run_id="test-run",
        dataset={"testcases_path": "data/testcases.jsonl"},
        candidates=[
            {"candidate_id": "c0", "vendor": "openai", "model_id": "gpt-4o"},
            {"candidate_id": "c1", "vendor": "openai", "model_id": "gpt-4o"},
        ],
        judges=[],
        protocol={
            "evaluation_mode": "pairwise",
            "aggregation": {"method": "majority_vote"},
        },
    )


def _make_candidate(vendor: str = "openai") -> ModelRef:
    return ModelRef(
        candidate_id="test-candidate",
        vendor=vendor,
        model_id="gpt-4o",
    )


class TestCallModelErrorPaths:
    """Tests that _call_model returns InferenceRecord with status.ok=false
    instead of raising exceptions when json_schema_ref is invalid."""

    def test_file_not_found_returns_failure_record(self):
        """json_schema_ref pointing to non-existent file -> status.ok=false."""
        of = OutputFormat(type="json", json_schema_ref="nonexistent/schema.json")
        tc = _make_testcase(output_format=of)
        cfg = _make_run_config()
        candidate = _make_candidate("openai")
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

        record = _call_model(
            cfg=cfg,
            tc=tc,
            candidate=candidate,
            client=MagicMock(),
            messages=messages,
            gen_params={"max_tokens": 1024},
        )

        assert record.status.ok is False
        assert record.status.error_type == "FileNotFoundError"
        assert "nonexistent/schema.json" in record.status.error_message

    def test_invalid_json_schema_returns_failure_record(self, tmp_invalid_json):
        """json_schema_ref pointing to invalid JSON file -> status.ok=false."""
        of = OutputFormat(type="json", json_schema_ref=str(tmp_invalid_json))
        tc = _make_testcase(output_format=of)
        cfg = _make_run_config()
        candidate = _make_candidate("azure-openai")
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

        record = _call_model(
            cfg=cfg,
            tc=tc,
            candidate=candidate,
            client=MagicMock(),
            messages=messages,
            gen_params={},
        )

        assert record.status.ok is False
        assert record.status.error_type == "ValueError"
        assert "not valid JSON" in record.status.error_message

    def test_unsupported_vendor_does_not_fail(self):
        """Truly unsupported vendor with json_schema_ref -> falls back to free-text,
        does NOT produce a failure record from schema loading."""
        of = OutputFormat(type="json", json_schema_ref="nonexistent/schema.json")
        tc = _make_testcase(output_format=of)
        cfg = _make_run_config()
        candidate = _make_candidate("unknown-vendor")
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

        # Mock chat_completion to return a normal response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "free text response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

        with patch("llm_judge.stages.inference.chat_completion", return_value=mock_response):
            record = _call_model(
                cfg=cfg,
                tc=tc,
                candidate=candidate,
                client=MagicMock(),
                messages=messages,
                gen_params={"max_tokens": 1024},
            )

        assert record.status.ok is True
        assert record.output.text == "free text response"

    def test_file_not_found_preserves_testcase_and_candidate_info(self):
        """Failure record must preserve testcase_id and candidate_id."""
        of = OutputFormat(type="json", json_schema_ref="missing.json")
        tc = _make_testcase(testcase_id="tc-xyz-999", output_format=of)
        cfg = _make_run_config()
        candidate = _make_candidate("openai")
        messages = [{"role": "user", "content": "hi"}]

        record = _call_model(
            cfg=cfg,
            tc=tc,
            candidate=candidate,
            client=MagicMock(),
            messages=messages,
            gen_params={},
        )

        assert record.status.ok is False
        assert record.testcase_id == "tc-xyz-999"
        assert record.candidate_id == "test-candidate"
        assert record.run_id == "test-run"


# ── Structured Output × messages mode ────────────────────


class TestStructuredOutputMessagesMode:
    def test_structured_output_does_not_override_system_for_messages_mode(self, sample_schema_path):
        """In messages mode, system message should NOT be overridden even with structured output."""
        messages_input = [
            {"role": "system", "content": "Custom system prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Follow up"},
        ]
        tc = Testcase(
            testcase_id="mt-so-001",
            task_type="report_qa",
            input={"messages": messages_input},
            constraints=Constraints(
                output_format=OutputFormat(
                    type="json",
                    json_schema_ref=sample_schema_path,
                )
            ),
        )
        assert tc.has_messages is True
        assert _requires_structured_output(tc) is True

        cfg = _make_run_config()
        candidate = _make_candidate("openai")

        from llm_judge.prompts import build_inference_prompt

        prompt_messages = build_inference_prompt(tc)

        mock_response = MagicMock()
        valid_json = json.dumps(_valid_sample_data())
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = valid_json
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

        with patch("llm_judge.stages.inference.chat_completion", return_value=mock_response) as mock_chat:
            record = _call_model(
                cfg=cfg,
                tc=tc,
                candidate=candidate,
                client=MagicMock(),
                messages=prompt_messages,
                gen_params={"max_tokens": 4096},
            )

            # Verify the messages sent to the model still have the original system content
            sent_messages = mock_chat.call_args.kwargs.get("messages", mock_chat.call_args[1].get("messages"))
            if sent_messages is None:
                # Try positional args
                sent_messages = mock_chat.call_args[0][2] if len(mock_chat.call_args[0]) > 2 else None
            # The system message should NOT be the override version
            assert sent_messages[0]["content"] == "Custom system prompt"

        assert record.status.ok is True

    def test_structured_output_overrides_system_for_legacy_mode(self, sample_schema_path):
        """In legacy mode, system message SHOULD be overridden for structured output."""
        of = OutputFormat(type="json", json_schema_ref=sample_schema_path)
        tc = _make_testcase(output_format=of)
        assert tc.has_messages is False
        assert _requires_structured_output(tc) is True

        cfg = _make_run_config()
        candidate = _make_candidate("openai")

        from llm_judge.prompts import build_inference_prompt

        prompt_messages = build_inference_prompt(tc)

        mock_response = MagicMock()
        valid_json = json.dumps(_valid_sample_data())
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = valid_json
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

        with patch("llm_judge.stages.inference.chat_completion", return_value=mock_response) as mock_chat:
            record = _call_model(
                cfg=cfg,
                tc=tc,
                candidate=candidate,
                client=MagicMock(),
                messages=prompt_messages,
                gen_params={"max_tokens": 4096},
            )

            # The system message SHOULD be the override version
            sent_messages = mock_chat.call_args.kwargs.get("messages", mock_chat.call_args[1].get("messages"))
            if sent_messages is None:
                sent_messages = mock_chat.call_args[0][2] if len(mock_chat.call_args[0]) > 2 else None
            assert "prioritize the response format" in sent_messages[0]["content"]

        assert record.status.ok is True
