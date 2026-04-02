"""Stage 1: Candidate model inference."""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import jsonschema
from openai import BadRequestError
from rich.progress import Progress

from llm_judge.artifact_validation import validate_artifacts
from dotenv import load_dotenv
from llm_judge.config import load_run_config
from llm_judge.llm_client import chat_completion, gemini_inference_completion, get_thread_local_openai_client
from llm_judge.models import (
    InferenceRecord,
    ModelInfo,
    ModelRef,
    OutputFormat,
    OutputInfo,
    PromptInfo,
    RunConfig,
    StatusInfo,
    Testcase,
    TimingInfo,
    UsageInfo,
)
from llm_judge.parallelism import PlannedTask, run_bounded_parallel
from llm_judge.prompts import build_inference_prompt
from llm_judge.schema_validation import resolve_schema_path
from llm_judge.testcase_loader import load_testcases
from llm_judge.utils import (
    build_stage_output_path,
    content_hash,
    progress_log,
    write_jsonl,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceTaskPayload:
    """Per-task data passed from the planner to each worker."""

    testcase: Testcase
    candidate: ModelRef
    repeat_idx: int
    messages: tuple  # immutable so workers safely share the original
    planned_response_format: dict | None  # pre-built by planner; None = no SO
    gen_params: dict


# ── Structured Output configuration ──────────────────────

_STRUCTURED_OUTPUT_MIN_MAX_TOKENS = 4096

# Vendors that support schema-constrained structured output
_JSON_SCHEMA_FORMAT_VENDORS = frozenset({"openai", "azure-openai", "gemini"})


def _requires_structured_output(tc: Testcase) -> bool:
    """Return True if the testcase requires structured output via json_schema_ref.

    Conditions:
    - ``output_format.type == "json"``
    - ``output_format.json_schema_ref`` is a non-empty string
    """
    of = _get_output_format(tc)
    if of is None:
        return False
    if of.type != "json":
        return False
    ref = of.json_schema_ref
    return ref is not None and ref.strip() != ""


def _get_output_format(tc: Testcase) -> OutputFormat | None:
    """Extract output_format from testcase constraints, or None."""
    if tc.constraints is None:
        return None
    return tc.constraints.output_format


def _supports_json_schema_format(vendor: str, candidate_flag: bool = False) -> bool:
    """Return True if the vendor supports schema-constrained structured output.

    Returns True if the vendor is in the known list, or if the candidate
    config explicitly declares ``structured_output: true``.
    """
    return vendor in _JSON_SCHEMA_FORMAT_VENDORS or candidate_flag


def _load_json_schema(json_schema_ref: str) -> dict:
    """Load and parse a JSON schema from a repository-root-relative path.

    Raises FileNotFoundError if the file does not exist.
    Raises ValueError if the file is not valid JSON.
    """
    path = resolve_schema_path(json_schema_ref)
    if not path.exists():
        raise FileNotFoundError(
            f"json_schema_ref not found: {json_schema_ref}"
        )
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"json_schema_ref is not valid JSON ({json_schema_ref}): {exc}"
        ) from exc


def _make_schema_name(schema: dict, testcase_id: str) -> str:
    """Generate a safe schema name for the response_format payload.

    Uses the schema ``title`` if available, otherwise derives from testcase_id.
    The name is lowercased, non-alphanumeric characters replaced with ``_``.
    """
    raw = schema.get("title") or testcase_id
    return re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()


def _normalize_json_schema_for_response_format(node: Any) -> Any:
    """Normalize schemas so strict response_format works on Azure/OpenAI.

    Azure rejects strict schemas when nested object nodes omit
    ``additionalProperties: false`` and ``required`` coverage for every
    property key. Normalize the payload we send so one schema authoring
    mistake does not take down inference.
    """
    if isinstance(node, dict):
        normalized = {
            key: _normalize_json_schema_for_response_format(value)
            for key, value in node.items()
        }
        node_type = normalized.get("type")
        is_object_schema = (
            node_type == "object"
            or (isinstance(node_type, list) and "object" in node_type)
            or "properties" in normalized
        )
        if is_object_schema:
            if normalized.get("additionalProperties") is not False:
                normalized["additionalProperties"] = False
            properties = normalized.get("properties")
            if isinstance(properties, dict):
                normalized["required"] = list(properties.keys())
        return normalized
    if isinstance(node, list):
        return [_normalize_json_schema_for_response_format(item) for item in node]
    return node


def _build_response_format(json_schema_ref: str, testcase_id: str) -> dict:
    """Build the response_format dict for OpenAI Structured Outputs.

    Loads the schema from ``json_schema_ref`` (repo-root relative path)
    and wraps it in the OpenAI response_format structure.
    """
    schema = _normalize_json_schema_for_response_format(
        _load_json_schema(json_schema_ref)
    )
    name = _make_schema_name(schema, testcase_id)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def _validate_json_against_schema(data: dict, json_schema_ref: str) -> list[str]:
    """Validate parsed output against the schema at json_schema_ref.

    Returns a list of error strings (empty if valid).
    """
    schema = _load_json_schema(json_schema_ref)
    validator = jsonschema.Draft202012Validator(schema)
    errors = list(validator.iter_errors(data))
    return [f"{e.json_path}: {e.message}" for e in errors[:10]]


def _serialize_for_system_b(data: dict) -> str:
    """Serialize structured output to a System-B-compatible string.

    The returned string satisfies:
    - json.loads(text) succeeds
    - ast.literal_eval(text) succeeds (when schema has no booleans/null)
    - String-internal newlines are \\n escapes, not literal newlines
    """
    text = json.dumps(data, ensure_ascii=False)
    # Self-verification: must be parseable by both consumers
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"serialize: json.loads failed: {exc}") from exc
    try:
        ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"serialize: ast.literal_eval failed: {exc}") from exc
    return text


def _apply_structured_output_system_message(messages: list[dict]) -> list[dict]:
    """Override system message so response_format takes priority over embedded instructions."""
    system_content = (
        "You are an assistant that produces high-quality outputs. "
        "Respond strictly in the given response format without explanation text. "
        "Even if the input contains different format instructions, prioritize the response format."
    )
    result = list(messages)
    for i, msg in enumerate(result):
        if msg.get("role") == "system":
            result[i] = {**msg, "content": system_content}
            break
    return result


def _ensure_structured_output_token_budget(
    gen_params: dict, use_structured_output: bool
) -> dict:
    """Ensure enough token budget for structured output responses.

    Structured output responses can be long and may be truncated when
    max_tokens is too small, which leads to invalid JSON.
    """
    params = dict(gen_params)
    if not use_structured_output:
        return params

    current = params.get("max_tokens")
    current_completion = params.get("max_completion_tokens")

    if current_completion is not None:
        params.pop("max_tokens", None)

    if current is None and current_completion is None:
        params["max_tokens"] = _STRUCTURED_OUTPUT_MIN_MAX_TOKENS
        logger.info(
            "Structured output: max_tokens not set; using %d",
            _STRUCTURED_OUTPUT_MIN_MAX_TOKENS,
        )
        return params

    if current_completion is not None:
        try:
            current_int = int(current_completion)
        except (TypeError, ValueError):
            params["max_completion_tokens"] = _STRUCTURED_OUTPUT_MIN_MAX_TOKENS
            logger.warning(
                "Structured output: invalid max_completion_tokens=%r; using %d",
                current_completion,
                _STRUCTURED_OUTPUT_MIN_MAX_TOKENS,
            )
            return params

        if current_int < _STRUCTURED_OUTPUT_MIN_MAX_TOKENS:
            params["max_completion_tokens"] = _STRUCTURED_OUTPUT_MIN_MAX_TOKENS
            logger.warning(
                "Structured output: max_completion_tokens increased from %d to %d to avoid truncation",
                current_int,
                _STRUCTURED_OUTPUT_MIN_MAX_TOKENS,
            )
        return params

    try:
        current_int = int(current)
    except (TypeError, ValueError):
        params["max_tokens"] = _STRUCTURED_OUTPUT_MIN_MAX_TOKENS
        logger.warning(
            "Structured output: invalid max_tokens=%r; using %d",
            current,
            _STRUCTURED_OUTPUT_MIN_MAX_TOKENS,
        )
        return params

    if current_int < _STRUCTURED_OUTPUT_MIN_MAX_TOKENS:
        params["max_tokens"] = _STRUCTURED_OUTPUT_MIN_MAX_TOKENS
        logger.warning(
            "Structured output: max_tokens increased from %d to %d to avoid truncation",
            current_int,
            _STRUCTURED_OUTPUT_MIN_MAX_TOKENS,
        )

    return params


_TOKEN_BUDGET_ERROR_PATTERNS = [
    re.compile(r"max_completion_tokens?\s+is\s+too\s+large", re.IGNORECASE),
    re.compile(r"maximum\s+context\s+length\s+is\s+(\d+)\s+tokens", re.IGNORECASE),
    re.compile(r"resulting\s+in\s+(\d+)\s+tokens.*reduce", re.IGNORECASE),
]

_MIN_USEFUL_COMPLETION_TOKENS = 64


def _is_token_budget_error(exc: BadRequestError) -> bool:
    """Return True if the 400 error is caused by token budget exceeded."""
    msg = str(exc).lower()
    return any(p.search(msg) for p in _TOKEN_BUDGET_ERROR_PATTERNS)


def _extract_reduced_budget(exc: BadRequestError, original_budget: int) -> int | None:
    """Try to extract a safe reduced budget from the error message.

    Returns None if the remaining budget is too small to be useful.
    When a concrete remaining budget is extracted from the error message,
    it is authoritative — if it is below the threshold, None is returned
    immediately without falling back to halving.
    """
    msg = str(exc)
    # Pattern: "at most N completion tokens" or "leaving at most N tokens"
    m = re.search(r"at\s+most\s+(\d+)\s+(?:completion\s+)?tokens", msg, re.IGNORECASE)
    if m:
        remaining = int(m.group(1))
        if remaining >= _MIN_USEFUL_COMPLETION_TOKENS:
            return remaining
        return None  # server explicitly says budget is too small

    # Pattern: "maximum context length is X ... your messages/prompt/request resulted in/has Y tokens"
    m_ctx = re.search(r"maximum\s+context\s+length\s+is\s+(\d+)", msg, re.IGNORECASE)
    m_prompt = re.search(
        r"(?:messages|prompt|request)\s+(?:resulted\s+in|has)\s+(\d+)\s+(?:input\s+)?tokens",
        msg, re.IGNORECASE,
    )
    if m_ctx and m_prompt:
        ctx_limit = int(m_ctx.group(1))
        prompt_tokens = int(m_prompt.group(1))
        remaining = ctx_limit - prompt_tokens
        if remaining >= _MIN_USEFUL_COMPLETION_TOKENS:
            return remaining
        return None  # computed remaining budget is too small

    # Fallback: halve the original budget (no explicit budget in error message)
    halved = original_budget // 2
    if halved >= _MIN_USEFUL_COMPLETION_TOKENS:
        return halved

    return None


def run_inference(
    config_path: str,
    output_path: str | None = None,
    *,
    started_at: datetime | None = None,
    run_dir: Path | None = None,
) -> Path:
    """Run inference for all testcase x candidate combinations."""
    # Load env to ensure .env is read
    load_dotenv()

    cfg = load_run_config(config_path)
    testcases = load_testcases(cfg.dataset.testcases_path)

    out = Path(output_path) if output_path else build_stage_output_path(
        "inference",
        cfg.run_id,
        ".jsonl",
        started_at=started_at,
        run_dir=run_dir,
    )

    total = len(testcases) * len(cfg.candidates) * cfg.protocol.repeats.inference_repeats
    max_workers = cfg.protocol.parallelism.inference_max_concurrency

    # ── Planner ──────────────────────────────────────────────────
    # Pre-compute response_format once per testcase (not per candidate/repeat).
    tc_response_format: dict[str, dict | None] = {}
    for tc in testcases:
        if _requires_structured_output(tc):
            json_schema_ref = tc.constraints.output_format.json_schema_ref  # type: ignore[union-attr]
            tc_response_format[tc.testcase_id] = _build_response_format(
                json_schema_ref, tc.testcase_id
            )
        else:
            tc_response_format[tc.testcase_id] = None

    tasks: list[PlannedTask[InferenceTaskPayload]] = []
    task_index = 0
    for tc in testcases:
        messages = tuple(build_inference_prompt(tc))
        requires_so = _requires_structured_output(tc)
        for candidate in cfg.candidates:
            gen_params = dict(candidate.generation_params)
            tc_max = tc.generation_params.get("max_tokens")
            if tc_max is not None:
                # Preserve the token key convention used by the candidate:
                # if the candidate uses max_completion_tokens, override that
                # key instead of forcing max_tokens (which some models reject).
                use_completion_key = "max_completion_tokens" in candidate.generation_params
                gen_params.pop("max_tokens", None)
                gen_params.pop("max_completion_tokens", None)
                if use_completion_key:
                    gen_params["max_completion_tokens"] = tc_max
                else:
                    gen_params["max_tokens"] = tc_max
            use_so = requires_so and _supports_json_schema_format(candidate.vendor, candidate.structured_output)
            planned_rf = tc_response_format[tc.testcase_id] if use_so else None
            for repeat_idx in range(cfg.protocol.repeats.inference_repeats):
                tasks.append(PlannedTask(
                    task_index=task_index,
                    payload=InferenceTaskPayload(
                        testcase=tc,
                        candidate=candidate,
                        repeat_idx=repeat_idx,
                        messages=messages,
                        planned_response_format=planned_rf,
                        gen_params=gen_params,
                    ),
                ))
                task_index += 1

    # ── Worker ───────────────────────────────────────────────────
    with Progress() as progress:
        bar_task = progress.add_task("Inference", total=total)

        def worker_fn(payload: InferenceTaskPayload) -> InferenceRecord:
            client = get_thread_local_openai_client(
                payload.candidate.vendor, payload.candidate.endpoint
            )
            progress_log(
                "[inference] start testcase={} candidate={} repeat={}/{}".format(
                    payload.testcase.testcase_id,
                    payload.candidate.candidate_id,
                    payload.repeat_idx + 1,
                    cfg.protocol.repeats.inference_repeats,
                )
            )
            record = _call_model(
                cfg=cfg,
                tc=payload.testcase,
                candidate=payload.candidate,
                client=client,
                messages=list(payload.messages),
                gen_params=payload.gen_params,
                planned_response_format=payload.planned_response_format,
            )
            latency_ms = record.timing.latency_ms if record.timing else 0
            status = "ok" if record.status.ok else f"error:{record.status.error_type}"
            progress_log(
                "[inference] done status={} latency_ms={:.1f} testcase={} candidate={}".format(
                    status,
                    latency_ms,
                    payload.testcase.testcase_id,
                    payload.candidate.candidate_id,
                )
            )
            progress.advance(bar_task)
            return record

        records = run_bounded_parallel(tasks, worker_fn, max_workers=max_workers)

    validate_artifacts("inference-record", records)
    write_jsonl(out, records)
    return out


def _extract_reasoning(
    msg: object,
    raw_text: str,
) -> tuple[str | None, str]:
    """Extract reasoning from an API response message.

    Checks (in order):
    1. ``reasoning_content`` attribute on the message object
    2. ``reasoning_content`` in ``model_extra`` (vLLM / non-standard providers)
    3. ``<think>…</think>`` tags embedded in the content text

    Returns ``(reasoning_text, cleaned_raw_text)``.  *cleaned_raw_text* has
    ``<think>`` blocks stripped when they were the source of reasoning.
    """
    # Prefer structured reasoning_content field (accept str only)
    _rc = getattr(msg, "reasoning_content", None)
    reasoning: str | None = _rc if isinstance(_rc, str) else None
    if reasoning is None:
        _rc2 = (getattr(msg, "model_extra", None) or {}).get("reasoning_content")
        reasoning = _rc2 if isinstance(_rc2, str) else None

    # Fall back to <think> tags in content
    if reasoning is None and "<think>" in raw_text:
        blocks = re.findall(r"<think>(.*?)</think>", raw_text, re.DOTALL)
        if blocks:
            reasoning = "\n\n".join(b.strip() for b in blocks)
            raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    return reasoning or None, raw_text


def _call_model(
    cfg: RunConfig,
    tc: Testcase,
    candidate,
    client,
    messages: list[dict[str, str]],
    gen_params: dict,
    planned_response_format: dict | None = None,
) -> InferenceRecord:
    """Call a single model and return an InferenceRecord."""
    started_at = datetime.now(timezone.utc)
    t0 = time.monotonic()

    requires_so = _requires_structured_output(tc)
    json_schema_ref = (
        tc.constraints.output_format.json_schema_ref
        if requires_so and tc.constraints and tc.constraints.output_format
        else None
    )

    # [P1] Only apply response_format=json_schema on vendors that support it.
    use_structured_output = requires_so and _supports_json_schema_format(candidate.vendor, candidate.structured_output)
    extra_kwargs: dict = {}
    actual_messages = messages
    actual_input_hash = content_hash(str(messages))
    prompt_hash = content_hash(str(messages))

    try:
        if use_structured_output:
            if planned_response_format is not None:
                extra_kwargs["response_format"] = planned_response_format
            else:
                extra_kwargs["response_format"] = _build_response_format(
                    json_schema_ref, tc.testcase_id
                )
            if not tc.has_messages:
                actual_messages = _apply_structured_output_system_message(messages)
                # [P2] Recompute hash from the messages actually sent.
                actual_input_hash = content_hash(str(actual_messages))
                prompt_hash = content_hash(str(actual_messages))
        elif requires_so:
            logger.warning(
                "Structured output skipped for %s/%s: vendor '%s' does not support "
                "json_schema response_format; json_schema_ref='%s'; "
                "falling back to free-text inference.",
                tc.testcase_id,
                candidate.candidate_id,
                candidate.vendor,
                json_schema_ref,
            )

        call_gen_params = _ensure_structured_output_token_budget(gen_params, use_structured_output)
        logger.debug(
            "call_gen_params for %s/%s: %s",
            tc.testcase_id, candidate.candidate_id, call_gen_params,
        )

        # Gemini native path: use responseSchema for structured output
        if candidate.vendor == "gemini":
            gemini_schema = None
            if use_structured_output and json_schema_ref:
                gemini_schema = _load_json_schema(json_schema_ref)
            result = gemini_inference_completion(
                model=candidate.model_id,
                messages=actual_messages,
                endpoint=candidate.endpoint,
                generation_params=call_gen_params,
                response_schema=gemini_schema,
            )
            t1 = time.monotonic()
            raw_text = result.text
            reasoning_text = None
            finish_reason = result.finish_reason
            usage = None
            input_tokens = result.input_tokens or 0
            output_tokens = result.output_tokens or 0
        else:
            # OpenAI / Azure OpenAI / custom path
            try:
                response = chat_completion(
                    client=client,
                    model=candidate.model_id,
                    messages=actual_messages,
                    **call_gen_params,
                    **extra_kwargs,
                )
            except BadRequestError as budget_exc:
                if not _is_token_budget_error(budget_exc):
                    raise
                original_budget = (
                    call_gen_params.get("max_completion_tokens")
                    or call_gen_params.get("max_tokens")
                    or 0
                )
                reduced = _extract_reduced_budget(budget_exc, int(original_budget))
                if reduced is None:
                    logger.warning(
                        "Token budget exceeded for %s/%s but remaining budget too small; not retrying: %s",
                        tc.testcase_id, candidate.candidate_id, budget_exc,
                    )
                    raise
                logger.warning(
                    "Token budget exceeded for %s/%s; retrying with max_completion_tokens=%d (was %s): %s",
                    tc.testcase_id, candidate.candidate_id, reduced, original_budget, budget_exc,
                )
                retry_params = dict(call_gen_params)
                retry_params.pop("max_tokens", None)
                retry_params["max_completion_tokens"] = reduced
                call_gen_params = retry_params
                response = chat_completion(
                    client=client,
                    model=candidate.model_id,
                    messages=actual_messages,
                    **retry_params,
                    **extra_kwargs,
                )
            t1 = time.monotonic()
            raw_text = response.choices[0].message.content or ""
            reasoning_text, raw_text = _extract_reasoning(
                response.choices[0].message, raw_text,
            )
            finish_reason = response.choices[0].finish_reason
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

        # Detect format from constraints
        fmt = None
        if tc.constraints and tc.constraints.output_format:
            fmt = tc.constraints.output_format.type

        if use_structured_output:
            # Parse, validate, and re-serialize for System B
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                token_budget = call_gen_params.get("max_tokens") or call_gen_params.get("max_completion_tokens")
                detail = (
                    f"Structured output response is not valid JSON for "
                    f"{tc.testcase_id} (schema_ref={json_schema_ref}): {exc} "
                    f"(finish_reason={finish_reason}, max_tokens={token_budget})"
                )
                if finish_reason == "length":
                    detail += " [response likely truncated; increase max_tokens]"
                raise ValueError(detail) from exc

            validation_errors = _validate_json_against_schema(parsed, json_schema_ref)
            if validation_errors:
                err_str = "; ".join(validation_errors)
                logger.error(
                    "Schema validation failed for %s/%s (schema_ref=%s): %s",
                    tc.testcase_id,
                    candidate.candidate_id,
                    json_schema_ref,
                    err_str,
                )
                raise ValueError(
                    f"Schema validation failed for {tc.testcase_id} "
                    f"(schema_ref={json_schema_ref}): {err_str}"
                )

            system_b_text = _serialize_for_system_b(parsed)
            output = OutputInfo(text=system_b_text, reasoning=reasoning_text, format="json", json_data=parsed)
        else:
            output = OutputInfo(text=raw_text, reasoning=reasoning_text, format=fmt)

            # Try to parse JSON if expected
            if fmt == "json":
                from llm_judge.utils import strip_fenced_json
                try:
                    output.json_data = json.loads(strip_fenced_json(raw_text))
                except json.JSONDecodeError:
                    pass

        return InferenceRecord(
            run_id=cfg.run_id,
            testcase_id=tc.testcase_id,
            candidate_id=candidate.candidate_id,
            model=ModelInfo(
                vendor=candidate.vendor,
                model_id=candidate.model_id,
                endpoint=candidate.endpoint,
            ),
            prompt=PromptInfo(prompt_version=candidate.prompt_version, prompt_hash=prompt_hash),
            generation_params=call_gen_params or None,
            input_hash=actual_input_hash,
            output=output,
            usage=UsageInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=getattr(
                    getattr(usage, "completion_tokens_details", None),
                    "reasoning_tokens",
                    None,
                ) if usage else None,
            ),
            timing=TimingInfo(
                started_at=started_at.isoformat(),
                ended_at=datetime.now(timezone.utc).isoformat(),
                latency_ms=round((t1 - t0) * 1000, 1),
            ),
            status=StatusInfo(ok=True),
        )
    except Exception as e:
        t1 = time.monotonic()
        logger.error(
            "Inference failed for %s/%s: %s: %s",
            tc.testcase_id,
            candidate.candidate_id,
            type(e).__name__,
            e,
        )
        return InferenceRecord(
            run_id=cfg.run_id,
            testcase_id=tc.testcase_id,
            candidate_id=candidate.candidate_id,
            model=ModelInfo(
                vendor=candidate.vendor,
                model_id=candidate.model_id,
                endpoint=candidate.endpoint,
            ),
            output=OutputInfo(text=""),
            prompt=PromptInfo(prompt_version=candidate.prompt_version, prompt_hash=prompt_hash),
            input_hash=actual_input_hash,
            timing=TimingInfo(
                started_at=started_at.isoformat(),
                ended_at=datetime.now(timezone.utc).isoformat(),
                latency_ms=round((t1 - t0) * 1000, 1),
            ),
            status=StatusInfo(
                ok=False,
                error_type=type(e).__name__,
                error_message=str(e),
            ),
        )
