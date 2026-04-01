"""Artifact schema validation for output records before writing."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_DIR = _REPO_ROOT / "schemas"

_ARTIFACT_SCHEMA_MAP: dict[str, str] = {
    "inference-record": "inference-record.schema.json",
    "autocheck-record": "autocheck-record.schema.json",
    "judgement-record": "judgement-record.schema.json",
    "consistency-record": "consistency-record.schema.json",
    "comparison-report": "comparison-report.schema.json",
}

_MAX_ERRORS = 20


class ArtifactValidationError(Exception):
    """Raised when artifact records fail JSON Schema validation."""

    def __init__(self, artifact_type: str, errors: list[str]) -> None:
        self.artifact_type = artifact_type
        self.errors = errors
        super().__init__(
            f"Artifact validation failed for '{artifact_type}' "
            f"({len(errors)} error(s)):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def _load_artifact_schema(artifact_type: str) -> dict:
    filename = _ARTIFACT_SCHEMA_MAP.get(artifact_type)
    if filename is None:
        raise ValueError(
            f"Unknown artifact type: '{artifact_type}'. "
            f"Valid types: {', '.join(sorted(_ARTIFACT_SCHEMA_MAP))}"
        )
    schema_path = _SCHEMA_DIR / filename
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _record_to_dict(record: BaseModel | dict) -> dict:
    if isinstance(record, BaseModel):
        raw = record.model_dump_json(exclude_none=True, by_alias=True)
        return json.loads(raw)
    return record


def validate_artifacts(
    artifact_type: str,
    records: list[BaseModel | dict],
) -> None:
    """Validate a list of records against the artifact JSON Schema."""
    schema = _load_artifact_schema(artifact_type)
    validator = jsonschema.Draft202012Validator(schema)
    errors: list[str] = []

    for idx, record in enumerate(records):
        data = _record_to_dict(record)
        record_errors = list(validator.iter_errors(data))
        for err in record_errors:
            errors.append(
                f"record[{idx}]: {err.json_path}: {err.message}"
            )
            if len(errors) >= _MAX_ERRORS:
                break
        if len(errors) >= _MAX_ERRORS:
            break

    if errors:
        raise ArtifactValidationError(artifact_type, errors)


def validate_single_artifact(
    artifact_type: str,
    record: BaseModel | dict,
) -> None:
    """Validate a single record against the artifact JSON Schema."""
    validate_artifacts(artifact_type, [record])
