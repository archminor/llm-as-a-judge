"""Prompt builders for candidate inference and LLM-as-a-Judge."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any
from warnings import warn

from llm_judge.models import Constraints, Testcase
from llm_judge.schema_validation import resolve_schema_path

# ── Rubric loading ────────────────────────────────────────

RUBRIC_DIRS = [
    Path("rubrics"),
]

_RUBRIC_CACHE: dict[str, str | None] = {}
_RUBRIC_PARTS_CACHE: dict[str, tuple[str, dict[str, str]]] = {}
_MISSING_RUBRICS: set[str] = set()


def load_rubric(version: str) -> str:
    """Load a rubric by version name."""
    if version in _RUBRIC_CACHE:
        return _RUBRIC_CACHE[version] or ""

    for d in RUBRIC_DIRS:
        path = d / f"{version}.md"
        if path.exists():
            content = path.read_text()
            _RUBRIC_CACHE[version] = content
            return content

    _RUBRIC_CACHE[version] = None
    if version not in _MISSING_RUBRICS:
        _MISSING_RUBRICS.add(version)
        warn(
            f"Rubric '{version}.md' not found in {', '.join(str(d) for d in RUBRIC_DIRS)}",
            RuntimeWarning,
        )
    return ""


_METRIC_HEADING_RE = re.compile(r'^### \d+\.\d+\s+(\w+)\s*—', re.MULTILINE)


def load_rubric_parts(version: str) -> tuple[str, dict[str, str]]:
    """Load rubric and split into common part and per-metric sections.

    Returns:
        (common_text, metric_sections)
        - common_text: section before the "## N. Evaluation Metrics" heading
        - metric_sections: {"accuracy": "### 4.1 accuracy — ...", ...}
    """
    if version in _RUBRIC_PARTS_CACHE:
        return _RUBRIC_PARTS_CACHE[version]

    raw = load_rubric(version)

    split_match = re.search(r'^## \d+\. ', raw, flags=re.MULTILINE)
    if not split_match:
        raise ValueError(
            f"Rubric '{version}' has no '## N. ' heading to split on. "
            f"Expected format: '## 4. Evaluation Metrics'"
        )

    # Find the first numbered heading that contains metric sections (### N.N)
    # Walk from the first ## N. to find where metrics start
    metrics_section_match = re.search(r'^## .*', raw, flags=re.MULTILINE)
    # Actually split at the last common section before per-metric sections
    # Strategy: common = everything before first ### N.N metric heading
    first_metric_match = _METRIC_HEADING_RE.search(raw)
    if not first_metric_match:
        return (raw, {})

    # Find the ## heading that precedes the first metric
    preceding_h2 = None
    for m in re.finditer(r'^## .*', raw, flags=re.MULTILINE):
        if m.start() < first_metric_match.start():
            preceding_h2 = m
    if preceding_h2 is None:
        common = ""
        metrics_pool = raw
    else:
        common = raw[:preceding_h2.start()].rstrip()
        metrics_pool = raw[preceding_h2.start():]

    metric_sections: dict[str, str] = {}
    parts = re.split(r'(?=^### \d+\.\d+ )', metrics_pool, flags=re.MULTILINE)
    for part in parts:
        m = _METRIC_HEADING_RE.match(part)
        if m:
            metric_sections[m.group(1)] = part.strip()

    result = (common, metric_sections)
    _RUBRIC_PARTS_CACHE[version] = result
    return result


_LAYER_DECL_RE = re.compile(r'^>\s*Layers:\s*3-layer', re.MULTILINE)


def rubric_supports_layers(version: str) -> bool:
    """Check if the rubric declares layered (multi-call) evaluation.

    Detects ``> Layers: 3-layer ...`` in the rubric front matter.
    """
    raw = load_rubric(version)
    return bool(_LAYER_DECL_RE.search(raw))


def _join_metric_rubrics(
    metrics: list[str],
    metric_sections: dict[str, str],
    rubric_version: str,
) -> str:
    """Join requested metric rubric sections, raising on missing metrics."""
    missing = [m for m in metrics if m not in metric_sections]
    if missing:
        available = sorted(metric_sections.keys())
        raise ValueError(
            f"Rubric '{rubric_version}' is missing sections for metrics: {missing}. "
            f"Available: {available}. "
            f"Check that rubric '### N.N metric_id —' headings match config metrics."
        )
    return "\n\n".join(metric_sections[m] for m in metrics)


# ── Candidate inference prompt ────────────────────────────


def build_inference_prompt(testcase: Testcase) -> list[dict[str, str]]:
    """Build chat messages for candidate model inference."""
    if testcase.has_messages:
        return copy.deepcopy(testcase.input["messages"])

    constraints = testcase.constraints or Constraints()
    system_parts = ["You are a helpful assistant that produces high-quality outputs."]

    if constraints.output_format:
        fmt = constraints.output_format
        if fmt.type == "json":
            system_parts.append("Respond in JSON format.")
            if fmt.json_schema_ref:
                schema_path = resolve_schema_path(fmt.json_schema_ref)
                if schema_path.exists():
                    schema_content = schema_path.read_text().strip()
                    system_parts.append(
                        f"Your output must conform to this JSON schema:\n```json\n{schema_content}\n```"
                    )
                else:
                    system_parts.append(f"JSON schema: {fmt.json_schema_ref}")
        elif fmt.type == "markdown":
            system_parts.append("Respond in Markdown format.")

    if constraints.required_points:
        points = "\n".join(f"- {p}" for p in constraints.required_points)
        system_parts.append(f"## Required Points\n{points}")

    if constraints.forbidden_points:
        points = "\n".join(f"- {p}" for p in constraints.forbidden_points)
        system_parts.append(f"## Forbidden Points\n{points}")

    system_msg = "\n\n".join(system_parts)
    user_msg = _format_input(testcase.input)

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _format_input(input_data: dict[str, Any]) -> str:
    """Format testcase input dict into a user message string."""
    parts = []
    for key, value in input_data.items():
        if isinstance(value, str):
            parts.append(f"## {key}\n{value}")
        else:
            parts.append(f"## {key}\n{json.dumps(value, ensure_ascii=False)}")
    return "\n\n".join(parts)


# ── Constraints formatting for judge prompts ─────────────


def _format_constraints_for_judge(constraints: Constraints | None) -> str:
    """Format constraints as a section for judge prompts."""
    if constraints is None:
        return ""

    parts: list[str] = []

    if constraints.required_points:
        items = "\n".join(f"  - {p}" for p in constraints.required_points)
        parts.append(f"- required_points:\n{items}")

    if constraints.forbidden_points:
        items = "\n".join(f"  - {p}" for p in constraints.forbidden_points)
        parts.append(f"- forbidden_points:\n{items}")

    if constraints.output_format:
        fmt = constraints.output_format
        fmt_parts: list[str] = []
        if fmt.type:
            fmt_parts.append(f"  - type: {fmt.type}")
        if fmt.json_schema_ref:
            fmt_parts.append(f"  - json_schema_ref: {fmt.json_schema_ref}")
        if fmt_parts:
            parts.append("- output_format:\n" + "\n".join(fmt_parts))

    if not parts:
        return ""

    return "\n\n## Constraints\n" + "\n".join(parts)


# ── Judge prompts ─────────────────────────────────────────


def build_pairwise_judge_prompt(
    testcase: Testcase,
    output_a: str,
    output_b: str,
    label_a: str,
    label_b: str,
    metrics: list[str],
    rubric_version: str,
) -> list[dict[str, str]]:
    """Build prompt for pairwise judge evaluation."""
    common, metric_sections = load_rubric_parts(rubric_version)

    system_msg = common.replace("{mode}", "pairwise")

    relevant_rubrics = _join_metric_rubrics(metrics, metric_sections, rubric_version)

    constraints_section = _format_constraints_for_judge(testcase.constraints)

    user_msg = f"""## Evaluation Metric Rubrics

{relevant_rubrics}

## Task Info
- task_type: {testcase.task_type}
- testcase_id: {testcase.testcase_id}

## Input
{_format_input(testcase.input)}
{constraints_section}
## Answer {label_a}
{output_a}

## Answer {label_b}
{output_b}
"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_absolute_judge_prompt(
    testcase: Testcase,
    output_text: str,
    candidate_label: str,
    metrics: list[str],
    rubric_version: str,
) -> list[dict[str, str]]:
    """Build prompt for absolute (single-answer) judge evaluation."""
    common, metric_sections = load_rubric_parts(rubric_version)

    system_msg = common.replace("{mode}", "absolute")

    relevant_rubrics = _join_metric_rubrics(metrics, metric_sections, rubric_version)

    constraints_section = _format_constraints_for_judge(testcase.constraints)

    user_msg = f"""## Evaluation Metric Rubrics

{relevant_rubrics}

## Task Info
- task_type: {testcase.task_type}
- testcase_id: {testcase.testcase_id}

## Input
{_format_input(testcase.input)}
{constraints_section}
## Answer
{output_text}
"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ── Layered judge prompts ─────────────────────────────────


_LAYER_SYSTEM_PROMPTS = {
    "layer1": (
        "You are an evaluator specializing in output format compliance.\n"
        "Do not evaluate content correctness or expression quality."
    ),
    "layer2": (
        "You are an evaluator specializing in output content quality.\n"
        "Do not evaluate text readability or writing style."
    ),
    "layer3": (
        "You are an evaluator specializing in text expression quality.\n"
        "Do not evaluate whether content is correct or incorrect.\n"
        "Evaluate only readability, clarity, and conciseness of the text."
    ),
}

_LAYER_METRICS = {
    "layer1": {"format_compliance", "harmlessness"},
    "layer2": {"accuracy", "faithfulness", "completeness", "relevance", "reasoning", "citation_quality"},
    "layer3": {"expression_quality"},
}


def build_layered_judge_prompts(
    testcase: Testcase,
    metrics: list[str],
    rubric_version: str,
    mode: str,
    *,
    output_a: str | None = None,
    output_b: str | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
    output_text: str | None = None,
    candidate_label: str | None = None,
) -> list[tuple[str, list[str], list[dict[str, str]]]]:
    """Build per-layer judge prompts.

    Returns a list of (layer_name, layer_metrics, messages) tuples.
    Only layers that have at least one requested metric are included.
    """
    common, metric_sections = load_rubric_parts(rubric_version)

    constraints_section = _format_constraints_for_judge(testcase.constraints)

    layers: list[tuple[str, list[str], list[dict[str, str]]]] = []

    for layer_name in ("layer1", "layer2", "layer3"):
        layer_metric_set = _LAYER_METRICS[layer_name]
        layer_metrics = [m for m in metrics if m in layer_metric_set]
        if not layer_metrics:
            continue

        layer_system = _LAYER_SYSTEM_PROMPTS[layer_name]
        base_system = common.replace("{mode}", mode)
        system_msg = f"{base_system}\n\n## Layer Evaluation Guidance\n{layer_system}"

        relevant_rubrics = _join_metric_rubrics(layer_metrics, metric_sections, rubric_version)

        if mode == "pairwise":
            user_msg = f"""## Evaluation Metric Rubrics

{relevant_rubrics}

## Task Info
- task_type: {testcase.task_type}
- testcase_id: {testcase.testcase_id}

## Input
{_format_input(testcase.input)}
{constraints_section}
## Answer {label_a}
{output_a}

## Answer {label_b}
{output_b}
"""
        else:
            user_msg = f"""## Evaluation Metric Rubrics

{relevant_rubrics}

## Task Info
- task_type: {testcase.task_type}
- testcase_id: {testcase.testcase_id}

## Input
{_format_input(testcase.input)}
{constraints_section}
## Answer
{output_text}
"""

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        layers.append((layer_name, layer_metrics, messages))

    return layers


def build_consistency_judge_prompt(
    testcase: Testcase,
    outputs: list[str],
    rubric_version: str,
) -> list[dict[str, str]]:
    """Build prompt to evaluate consistency among N repeated inference outputs."""
    rubric = load_rubric(rubric_version)

    system_msg = f"""You are a fair Judge evaluating LLM output quality.
Evaluate how consistently the candidate model answered the same prompt across all outputs.

## Consistency Score Criteria
- **5 (Very consistent)**: All outputs convey the same core claims/conclusions/information without contradiction.
- **3 (Mostly consistent)**: Mostly the same content, but with notable variation in details or minor contradictions.
- **1 (Inconsistent)**: Core claims or facts contradict across outputs, or conclusions differ significantly.

## Rubric (reference)
{rubric}

## Output Format
Respond only in the following JSON object format (no explanation text):
Do not use array format ([...]). Return a single object ({{...}}).
{{
  "overall": <score 1-5 (integer or decimal)>,
  "rationale": "reasoning"
}}
"""

    outputs_str = "\n\n".join(
        f"## Output {i + 1}\n{text}" for i, text in enumerate(outputs)
    )

    user_msg = f"""## Task Info
- task_type: {testcase.task_type}
- testcase_id: {testcase.testcase_id}

## Input (shared)
{_format_input(testcase.input)}

## Subject: Repeated outputs for the same prompt ({len(outputs)} outputs)

{outputs_str}
"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
