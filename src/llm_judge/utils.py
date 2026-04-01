"""JSONL I/O, hashing, and statistics helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dicts."""
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any] | BaseModel]) -> None:
    """Write records (dicts or Pydantic models) to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            if isinstance(record, BaseModel):
                f.write(record.model_dump_json(exclude_none=True, by_alias=True) + "\n")
            else:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: dict[str, Any] | BaseModel) -> None:
    """Write a single JSON object to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, BaseModel):
        text = data.model_dump_json(indent=2, exclude_none=True, by_alias=True)
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False)
    with open(path, "w") as f:
        f.write(text + "\n")


def content_hash(text: str) -> str:
    """Return a short SHA-256 hex digest of text."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def mean(values: list[float | int]) -> float:
    """Compute arithmetic mean. Returns 0.0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def variance(values: list[float | int]) -> float:
    """Compute population variance. Returns 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


_RUN_ID_TIMESTAMP_SUFFIX_RE = re.compile(
    r"^(?P<base>.+?)[-_](?:\d{8})(?:[-_T]?\d{6})?$"
)
_SAFE_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
_TIMESTAMP_GLOB = ("[0-9]" * 8) + "-" + ("[0-9]" * 6)

_FENCED_RE = re.compile(r"^```(?:json|JSON)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def strip_fenced_json(text: str) -> str:
    """Remove ```json ... ``` fencing if present."""
    stripped = text.strip()
    m = _FENCED_RE.match(stripped)
    if m:
        return m.group(1).strip()
    return stripped


def strip_run_id_timestamp(run_id: str) -> str:
    """Drop a trailing YYYYMMDD or YYYYMMDD-HHMMSS segment from run_id."""
    match = _RUN_ID_TIMESTAMP_SUFFIX_RE.fullmatch(run_id)
    return match.group("base") if match else run_id


def sanitize_path_component(text: str, *, default: str = "run") -> str:
    """Return a filename-safe single path component."""
    sanitized = _SAFE_PATH_COMPONENT_RE.sub("_", text).strip("._-")
    return sanitized or default


def derive_dataset_layer(testcases_path: str) -> str:
    """Derive a short label from a testcases file path for output directory naming.

    Uses the parent directory name of the testcases file, sanitized.
    Falls back to 'result'.
    """
    p = Path(testcases_path)
    label = p.parent.name if p.parent.name else "result"
    return sanitize_path_component(label, default="result")


def build_run_dir(
    label: str,
    *,
    started_at: datetime | None = None,
    base_dir: str | Path = "results",
) -> Path:
    """Create and return a run output directory.

    Directory pattern: ``{base_dir}/{label}-{YYYYMMDD-HHMMSS}/``
    """
    timestamp = (started_at or datetime.now().astimezone()).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / f"{label}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_stage_output_path(
    stage_name: str,
    run_id: str,
    suffix: str,
    *,
    started_at: datetime | None = None,
    data_dir: str | Path = "data",
    run_dir: str | Path | None = None,
) -> Path:
    """Build a default output path using the execution start timestamp."""
    if run_dir is not None:
        return Path(run_dir) / f"{stage_name}{suffix}"
    timestamp = (started_at or datetime.now().astimezone()).strftime("%Y%m%d-%H%M%S")
    run_base = sanitize_path_component(strip_run_id_timestamp(run_id))
    return Path(data_dir) / f"{stage_name}-{run_base}-{timestamp}{suffix}"


def find_latest_stage_output_path(
    stage_name: str,
    run_id: str,
    suffix: str,
    *,
    data_dir: str | Path = "data",
    run_dir: str | Path | None = None,
    results_dir: str | Path = "results",
    dataset_layer: str | None = None,
) -> Path | None:
    """Find the newest default output file for a stage."""
    run_base = sanitize_path_component(strip_run_id_timestamp(run_id))

    if run_dir is not None:
        candidate = Path(run_dir) / f"{stage_name}{suffix}"
        if candidate.is_file():
            return candidate

    res_root = Path(results_dir)
    if res_root.is_dir():
        if dataset_layer:
            glob_pattern = f"{dataset_layer}-{_TIMESTAMP_GLOB}/{stage_name}{suffix}"
        else:
            glob_pattern = f"*-{run_base}-{_TIMESTAMP_GLOB}/{stage_name}{suffix}"
        candidates = sorted(
            p for p in res_root.glob(glob_pattern)
            if p.is_file()
        )
        if candidates:
            return candidates[-1]

    root = Path(data_dir)
    candidates = sorted(
        p for p in root.glob(f"{stage_name}-{run_base}-{_TIMESTAMP_GLOB}{suffix}")
        if p.is_file()
    )
    if candidates:
        return candidates[-1]

    legacy_run_id = sanitize_path_component(run_id)
    legacy = root / f"{stage_name}-{legacy_run_id}{suffix}"
    if legacy.is_file():
        return legacy
    return None


def progress_log_enabled() -> bool:
    """Return True if per-call progress logs should be printed."""
    v = os.getenv("LLM_JUDGE_PROGRESS_LOG", "1").strip().lower()
    return v not in {"0", "false", "no", "off"}


def progress_log(message: str) -> None:
    """Print a progress line with flush so long calls remain visible."""
    if progress_log_enabled():
        print(message, flush=True)


def resolve_stage_input_path(
    stage_name: str,
    run_id: str,
    suffix: str,
    *,
    data_dir: str | Path = "data",
    run_dir: str | Path | None = None,
    dataset_layer: str | None = None,
) -> Path:
    """Resolve the default input path for a stage or raise if missing."""
    path = find_latest_stage_output_path(
        stage_name,
        run_id,
        suffix,
        data_dir=data_dir,
        run_dir=run_dir,
        dataset_layer=dataset_layer,
    )
    if path is None:
        raise FileNotFoundError(
            f"No default {stage_name} output found for run_id={run_id!r} under {Path(data_dir)}"
        )
    return path
