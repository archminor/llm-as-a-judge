"""Stage 3b: Inference consistency evaluation (LLM-as-a-Judge).

For each (testcase_id, candidate_id) group that has >= 2 inference records,
a Judge model is asked to score how consistently the candidate answered the
same prompt across all repeated outputs (1 = incoherent/contradicting,
5 = perfectly consistent).

This stage is only meaningful when inference_repeats >= 2.
Results are written to data/consistency-<run_id>.jsonl.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.progress import Progress

if TYPE_CHECKING:
    from openai import OpenAI

from dotenv import load_dotenv
from llm_judge.config import load_run_config
from llm_judge.llm_client import get_thread_local_openai_client, judge_chat_completion
from llm_judge.models import (
    ConsistencyRecord,
    ConsistencyScores,
    InferenceRecord,
    JudgeRef,
    StatusInfo,
    Testcase,
)
from llm_judge.parallelism import PlannedTask, run_bounded_parallel
from llm_judge.prompts import build_consistency_judge_prompt
from llm_judge.testcase_loader import load_testcase_map
from llm_judge.utils import (
    build_stage_output_path,
    derive_dataset_layer,
    progress_log,
    read_jsonl,
    resolve_stage_input_path,
    strip_fenced_json,
    write_jsonl,
)

logger = logging.getLogger(__name__)

MAX_CONSISTENCY_RETRY = 3

# ── Task payload ─────────────────────────────────────────


@dataclass
class ConsistencyTaskPayload:
    testcase_id: str
    candidate_id: str
    inf_list: list[InferenceRecord]
    judge_ref: JudgeRef
    tc: Testcase


def run_consistency(
    config_path: str,
    inference_path: str | None = None,
    output_path: str | None = None,
    *,
    started_at: datetime | None = None,
    run_dir: Path | None = None,
) -> Path:
    """Evaluate consistency of repeated inference outputs via LLM-as-a-Judge."""
    load_dotenv()

    cfg = load_run_config(config_path)
    layer = derive_dataset_layer(cfg.dataset.testcases_path)
    inf_path = (
        Path(inference_path)
        if inference_path
        else resolve_stage_input_path("inference", cfg.run_id, ".jsonl", run_dir=run_dir, dataset_layer=layer)
    )
    raw_inferences = read_jsonl(inf_path)
    inferences = [InferenceRecord.model_validate(r) for r in raw_inferences]

    tc_map = load_testcase_map(cfg.dataset.testcases_path)

    # Group inferences by (testcase_id, candidate_id)
    groups: dict[tuple[str, str], list[InferenceRecord]] = defaultdict(list)
    for inf in inferences:
        if inf.status.ok and inf.output.text:
            groups[(inf.testcase_id, inf.candidate_id)].append(inf)

    # Only evaluate groups with >= 2 repeats
    eligible = {k: v for k, v in groups.items() if len(v) >= 2}

    if not eligible:
        import warnings
        warnings.warn(
            "No groups with inference_repeats >= 2 found. "
            "Set inference_repeats >= 2 in run config to enable consistency evaluation.",
            RuntimeWarning,
        )

    out = Path(output_path) if output_path else build_stage_output_path(
        "consistency",
        cfg.run_id,
        ".jsonl",
        started_at=started_at,
        run_dir=run_dir,
    )

    # ── Planner ──────────────────────────────────────────────
    tasks: list[PlannedTask[ConsistencyTaskPayload]] = []
    task_index = 0

    for (tc_id, candidate_id), inf_list in sorted(eligible.items()):
        tc = tc_map.get(tc_id)
        if not tc:
            continue
        for judge_ref in cfg.judges:
            tasks.append(PlannedTask(
                task_index=task_index,
                payload=ConsistencyTaskPayload(
                    testcase_id=tc_id,
                    candidate_id=candidate_id,
                    inf_list=inf_list,
                    judge_ref=judge_ref,
                    tc=tc,
                ),
            ))
            task_index += 1

    total = len(tasks)

    # ── Worker ───────────────────────────────────────────────
    with Progress() as progress:
        bar_task = progress.add_task("Consistency", total=total)

        def worker_fn(payload: ConsistencyTaskPayload) -> ConsistencyRecord:
            client = get_thread_local_openai_client(
                payload.judge_ref.vendor, payload.judge_ref.endpoint
            )
            outputs = [inf.output.text for inf in payload.inf_list]
            progress_log(
                "[consistency] start testcase={} candidate={} judge={} repeats={}".format(
                    payload.testcase_id,
                    payload.candidate_id,
                    payload.judge_ref.judge_id,
                    len(outputs),
                )
            )
            messages = build_consistency_judge_prompt(
                testcase=payload.tc,
                outputs=outputs,
                rubric_version=payload.judge_ref.rubric_version,
            )
            rec = _call_consistency_judge(
                cfg_run_id=cfg.run_id,
                tc_id=payload.testcase_id,
                candidate_id=payload.candidate_id,
                judge_ref=payload.judge_ref,
                messages=messages,
                repeat_count=len(outputs),
                client=client,
            )
            status = "ok" if rec.status.ok else f"error:{rec.status.error_type}"
            progress_log(
                "[consistency] done testcase={} candidate={} judge={} status={}".format(
                    payload.testcase_id,
                    payload.candidate_id,
                    payload.judge_ref.judge_id,
                    status,
                )
            )
            progress.advance(bar_task)
            return rec

        records = run_bounded_parallel(
            tasks,
            worker_fn,
            max_workers=cfg.protocol.parallelism.consistency_max_concurrency,
        )

    from llm_judge.artifact_validation import validate_artifacts

    validate_artifacts("consistency-record", records)
    write_jsonl(out, records)
    return out


def _call_consistency_judge(
    cfg_run_id: str,
    tc_id: str,
    candidate_id: str,
    judge_ref: JudgeRef,
    messages: list[dict[str, str]],
    repeat_count: int,
    client: OpenAI,
) -> ConsistencyRecord:
    _task_ctx = (
        "tc=%s cand=%s judge=%s" % (tc_id, candidate_id, judge_ref.judge_id)
    )
    try:
        for attempt in range(MAX_CONSISTENCY_RETRY):
            completion = judge_chat_completion(
                vendor=judge_ref.vendor,
                model=judge_ref.model_id,
                messages=messages,
                endpoint=judge_ref.endpoint,
                generation_params=judge_ref.generation_params or None,
                client=client,
            )
            result_text = completion.text or "{}"

            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                try:
                    result = json.loads(strip_fenced_json(result_text))
                except (json.JSONDecodeError, ValueError):
                    logger.warning(
                        "[%s] JSON parse failed (attempt %d/%d): %.120s",
                        _task_ctx, attempt + 1, MAX_CONSISTENCY_RETRY,
                        result_text,
                    )
                    continue

            if isinstance(result, list):
                if not result:
                    logger.warning(
                        "[%s] Judge returned empty list (attempt %d/%d)",
                        _task_ctx, attempt + 1, MAX_CONSISTENCY_RETRY,
                    )
                    continue
                logger.warning(
                    "[%s] Judge returned list instead of dict; using first element",
                    _task_ctx,
                )
                result = result[0]

            if isinstance(result, dict) and result.get("overall") is not None:
                try:
                    overall = float(result["overall"])
                except (ValueError, TypeError):
                    logger.warning(
                        "[%s] float conversion failed for overall=%r (attempt %d/%d)",
                        _task_ctx, result["overall"],
                        attempt + 1, MAX_CONSISTENCY_RETRY,
                    )
                    continue
                return ConsistencyRecord(
                    run_id=cfg_run_id,
                    testcase_id=tc_id,
                    candidate_id=candidate_id,
                    judge_id=judge_ref.judge_id,
                    repeat_count=repeat_count,
                    scores=ConsistencyScores(
                        overall=overall,
                        rationale=result.get("rationale"),
                    ),
                    status=StatusInfo(ok=True),
                )

            logger.warning(
                "[%s] overall missing (attempt %d/%d): %s",
                _task_ctx, attempt + 1, MAX_CONSISTENCY_RETRY,
                type(result).__name__,
            )
        else:
            raise ValueError(
                "Max retries exceeded: could not obtain valid overall score"
            )

    except Exception as e:
        return ConsistencyRecord(
            run_id=cfg_run_id,
            testcase_id=tc_id,
            candidate_id=candidate_id,
            judge_id=judge_ref.judge_id,
            repeat_count=repeat_count,
            scores=ConsistencyScores(),
            status=StatusInfo(
                ok=False,
                error_type=type(e).__name__,
                error_message=str(e),
            ),
        )
