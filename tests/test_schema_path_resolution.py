"""Tests for json_schema_ref path resolution (Task B)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_judge.schema_validation import _REPO_ROOT, resolve_schema_path


class TestResolveSchemaPath:
    def test_relative_resolves_to_repo_root(self) -> None:
        result = resolve_schema_path("schemas/test.json")
        assert result == _REPO_ROOT / "schemas" / "test.json"
        assert result.is_absolute()

    def test_absolute_path_unchanged(self, tmp_path: Path) -> None:
        abs_path = str(tmp_path / "schema.json")
        result = resolve_schema_path(abs_path)
        assert result == Path(abs_path)

    def test_cwd_change_does_not_affect_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = resolve_schema_path("schemas/test.json")
        # Should still resolve against repo root, not cwd
        assert result == _REPO_ROOT / "schemas" / "test.json"

    def test_schema_validation_uses_resolve(self) -> None:
        """Verify schema_validation.validate_output_against_testcase_schema
        uses resolve_schema_path (relative path resolves correctly)."""
        from llm_judge.models import Constraints, OutputFormat, Testcase
        from llm_judge.schema_validation import validate_output_against_testcase_schema

        tc = Testcase(
            testcase_id="tc-test",
            task_type="preprocessing",
            input={"text": "hello"},
            constraints=Constraints(
                output_format=OutputFormat(
                    type="json",
                    json_schema_ref="schemas/testcase.schema.json",
                )
            ),
        )
        # Should find the schema relative to repo root, not cwd
        result = validate_output_against_testcase_schema(
            tc,
            '{"testcase_id": "t1", "task_type": "preprocessing", "input": {}}',
        )
        assert result is not None
        assert result.passed is True

    def test_prompts_uses_resolve(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Verify prompts.build_inference_prompt uses resolve_schema_path."""
        import json

        from llm_judge.models import Constraints, OutputFormat, Testcase

        # Use a schema file at the repo-root relative path
        schema_ref = "schemas/testcase.schema.json"

        tc = Testcase(
            testcase_id="tc-prompt-test",
            task_type="preprocessing",
            input={"text": "hello"},
            constraints=Constraints(
                output_format=OutputFormat(type="json", json_schema_ref=schema_ref)
            ),
        )

        # Change cwd to tmp_path -- should still resolve
        monkeypatch.chdir(tmp_path)
        from llm_judge.prompts import build_inference_prompt

        messages = build_inference_prompt(tc)
        system_msg = messages[0]["content"]
        # Should have loaded the schema content, not just the ref string
        assert "JSON schema" in system_msg
