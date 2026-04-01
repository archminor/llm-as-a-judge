"""RunConfig loader and environment variable resolution."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from llm_judge.models import RunConfig

load_dotenv()


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a run-config YAML file.

    Validates against the JSON Schema first, then against the Pydantic model.
    """
    import json

    import jsonschema

    with open(path) as f:
        raw = yaml.safe_load(f)

    schema_path = (
        Path(__file__).resolve().parents[2]
        / "schemas"
        / "run-config.schema.json"
    )
    if schema_path.exists():
        with open(schema_path) as sf:
            schema = json.load(sf)
        jsonschema.validate(instance=raw, schema=schema)

    return RunConfig.model_validate(raw)


def resolve_vendor_env(vendor: str) -> tuple[str, str]:
    """Return (api_key, endpoint) for a vendor by environment variable convention.

    Convention: {VENDOR}_API_KEY, {VENDOR}_ENDPOINT where vendor is upper-cased
    and hyphens replaced with underscores.
    """
    prefix = vendor.upper().replace("-", "_")
    api_key = os.environ.get(f"{prefix}_API_KEY", "")
    endpoint = os.environ.get(f"{prefix}_ENDPOINT", "")
    return api_key, endpoint
