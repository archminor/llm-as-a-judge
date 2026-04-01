"""Testcase loader with JSON Schema enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from llm_judge.models import Testcase
from llm_judge.utils import read_jsonl

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TESTCASE_SCHEMA_PATH = _REPO_ROOT / "schemas" / "testcase.schema.json"


class TestcaseLoadError(Exception):
    """Raised when one or more testcase records fail JSON Schema validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(
            f"{len(errors)} testcase validation error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def _load_schema() -> dict:
    return json.loads(_TESTCASE_SCHEMA_PATH.read_text(encoding="utf-8"))


def load_testcases(path: str | Path) -> list[Testcase]:
    """Load testcases from a JSONL file with JSON Schema validation."""
    schema = _load_schema()
    validator = jsonschema.Draft202012Validator(schema)

    raw_records = read_jsonl(path)
    errors: list[str] = []
    testcases: list[Testcase] = []

    for idx, record in enumerate(raw_records, start=1):
        tc_id = record.get("testcase_id", "<unknown>")
        record_errors = list(validator.iter_errors(record))
        if record_errors:
            for err in record_errors[:5]:
                errors.append(
                    f"line {idx} (testcase_id={tc_id}): "
                    f"{err.json_path}: {err.message}"
                )
        else:
            testcases.append(Testcase.model_validate(record))

    if errors:
        raise TestcaseLoadError(errors)

    return testcases


def load_testcase_map(path: str | Path) -> dict[str, Testcase]:
    """Load testcases and return a dict keyed by testcase_id."""
    return {tc.testcase_id: tc for tc in load_testcases(path)}
