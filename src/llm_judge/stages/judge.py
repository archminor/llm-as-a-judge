"""Stage 3: LLM-as-a-Judge evaluation."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

from rich.progress import Progress

if TYPE_CHECKING:
    from openai import OpenAI

from llm_judge.artifact_validation import validate_artifacts
from dotenv import load_dotenv
from llm_judge.config import load_run_config
from llm_judge.llm_client import get_thread_local_openai_client, judge_chat_completion
from llm_judge.models import (
    BlindingInfo,
    InferenceRecord,
    InferenceRef,
    JudgeInfo,
    JudgeRef,
    JudgeTarget,
    JudgementRecord,
    MetricScore,
    PairwiseMetricScore,
    RunConfig,
    Scores,
    Testcase,
)
from llm_judge.parallelism import PlannedTask, run_bounded_parallel
from llm_judge.prompts import (
    build_absolute_judge_prompt,
    build_layered_judge_prompts,
    build_pairwise_judge_prompt,
    rubric_supports_layers,
)
from llm_judge.schema_validation import (
    validate_output_against_testcase_schema,
)
from llm_judge.testcase_loader import load_testcase_map
from llm_judge.utils import (
    build_stage_output_path,
    derive_dataset_layer,
    progress_log,
    read_jsonl,
    resolve_stage_input_path,
    write_jsonl,
)

logger = logging.getLogger(__name__)


# ── Task payloads ─────────────────────────────────────────


@dataclass
class JudgePairwisePayload:
    tc: Testcase
    inf_a: InferenceRecord
    inf_b: InferenceRecord
    idx_a: int
    idx_b: int
    judge_ref: JudgeRef
    presented_swap: bool


@dataclass
class JudgeAbsolutePayload:
    tc: Testcase
    inf: InferenceRecord
    idx: int
    judge_ref: JudgeRef


JudgeTaskPayload = JudgePairwisePayload | JudgeAbsolutePayload


def should_swap_pairwise_order(
    candidate_id_a: str,
    candidate_id_b: str,
    random_seed: int | None,
) -> bool:
    """Deterministically decide whether to swap pairwise presentation order.

    Uses SHA-256 so that:
    - Same ``(candidate_id_a, candidate_id_b, random_seed)`` always returns the
      same result.
    - Candidate enumeration order does not affect the result (IDs are sorted
      before hashing).
    - ``random_seed=0`` and ``random_seed=None`` are treated as distinct values
      (seed material is ``0`` vs ``42`` respectively; uses
      ``random_seed if random_seed is not None else 42``, *not* ``or 42``).

    NOTE: This replaces the previous ``random.Random.random() < 0.5`` approach.
    The ``presented_order`` field in ``JudgementRecord`` will now be
    deterministic for a given config, regardless of parallelism or task
    scheduling order.  Any code that assumed a random distribution should
    treat this as a **breaking change** in blinding behaviour.
    """
    seed_material = random_seed if random_seed is not None else 42
    normalized = sorted([candidate_id_a, candidate_id_b])
    hash_input = f"{seed_material}|{normalized[0]}|{normalized[1]}"
    digest = hashlib.sha256(hash_input.encode()).digest()
    return bool(digest[0] & 1)


def _is_failed_inference(inf: InferenceRecord) -> bool:
    """Return True if inference failed or produced empty output."""
    if not inf.status.ok:
        return True
    if not inf.output.text.strip():
        return True
    return False


def _inference_failure_reason(inf: InferenceRecord) -> str:
    """Return a human-readable reason for the inference failure."""
    if not inf.status.ok:
        err = inf.status.error_message or inf.status.error_type or "unknown"
        return f"Inference failed ({err})"
    return "Empty output"


def _normalize_inference_ref_path(path: str | Path) -> str:
    """Normalize inference artifact paths before storing them in records."""
    return str(path)


def _normalize_per_metric(
    raw: dict, *, allow_dict_values: bool = False,
) -> tuple[dict[str, MetricScore | PairwiseMetricScore | int], list[str]]:
    """Normalize per_metric values, supporting v2 structured format and v1 int format.

    v2 structured format:
        - Absolute: ``{"metric": {"rationale": "...", "score": 5}}`` → ``MetricScore``
        - Pairwise: ``{"metric": {"rationale": "...", "score_a": 5, "score_b": 3}}`` → ``PairwiseMetricScore``

    v1 fallback (int/float):
        - ``{"metric": 5}`` → ``int``

    Args:
        raw: Raw per_metric dict from Judge JSON response.
        allow_dict_values: If True, dict values are accepted (pairwise mode).
            For v1-style dicts like ``{"A": 5, "B": 3}`` they are averaged.

    Returns:
        A tuple of ``(normalized, dropped)`` where *dropped* contains
        human-readable descriptions of metrics that were removed.
    """
    normalized: dict[str, MetricScore | PairwiseMetricScore | int] = {}
    dropped: list[str] = []
    for k, v in raw.items():
        if v is None:
            dropped.append(f"{k} ({type(v).__name__}: null)")
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            normalized[k] = int(round(v))
        elif isinstance(v, dict):
            # v2 structured format: dict with "rationale" key
            if "rationale" in v and "score" in v:
                try:
                    normalized[k] = MetricScore(rationale=v["rationale"], score=int(round(v["score"])))
                except Exception:
                    dropped.append(f"{k} (MetricScore: validation failed)")
            elif "rationale" in v and "score_a" in v and "score_b" in v:
                try:
                    normalized[k] = PairwiseMetricScore(
                        rationale=v["rationale"],
                        score_a=int(round(v["score_a"])),
                        score_b=int(round(v["score_b"])),
                    )
                except Exception:
                    dropped.append(f"{k} (PairwiseMetricScore: validation failed)")
            elif allow_dict_values:
                # v1 pairwise fallback: {"A": 5, "B": 3} → averaged int
                vals = [x for x in v.values() if isinstance(x, (int, float))]
                if vals:
                    normalized[k] = round(sum(vals) / len(vals))
                else:
                    dropped.append(f"{k} (dict: no numeric values)")
            else:
                dropped.append(f"{k} (dict: disallowed in absolute mode)")
        else:
            dropped.append(f"{k} ({type(v).__name__}: non-numeric)")
    return normalized, dropped


_TASK_RESTRICTED_METRICS: dict[str, set[str]] = {}


def _extract_gate_score(value: MetricScore | PairwiseMetricScore | int | None) -> int | None:
    """Extract an integer score for layer-1 gate checks (absolute mode)."""
    if value is None:
        return None
    if isinstance(value, MetricScore):
        return value.score
    if isinstance(value, int):
        return value
    return None


def _extract_pairwise_gate_scores(
    value: MetricScore | PairwiseMetricScore | int | None,
) -> tuple[int | None, int | None]:
    """Extract (score_a, score_b) for pairwise layer-1 gate checks."""
    if value is None:
        return None, None
    if isinstance(value, PairwiseMetricScore):
        return value.score_a, value.score_b
    # Non-pairwise types: apply the same score to both sides.
    if isinstance(value, MetricScore):
        return value.score, value.score
    if isinstance(value, int):
        return value, value
    return None, None


def run_judge(
    config_path: str,
    inference_path: str | None = None,
    output_path: str | None = None,
    *,
    started_at: datetime | None = None,
    run_dir: Path | None = None,
) -> Path:
    """Run LLM-as-a-Judge evaluation on inference outputs."""
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

    # Group inferences by testcase_id
    inf_by_tc: dict[str, list[tuple[int, InferenceRecord]]] = {}
    for idx, inf in enumerate(inferences):
        inf_by_tc.setdefault(inf.testcase_id, []).append((idx, inf))

    # Pairwise 判定では候補ペアごとに 1 件の推論結果を使用する。
    # inference_repeats > 1 の場合でも、pairwise で全組合せを比較すると
    # O(repeats²) に爆発するため、候補ごとに最初の 1 件のみを採用する。
    # repeats による分散確認は absolute モードで行う設計。
    pairwise_inf_by_tc: dict[str, list[tuple[int, InferenceRecord]]] = {}
    for tc_id, inf_list in inf_by_tc.items():
        seen_candidates: set[str] = set()
        unique_list: list[tuple[int, InferenceRecord]] = []
        for idx, inf in inf_list:
            if inf.candidate_id in seen_candidates:
                continue
            seen_candidates.add(inf.candidate_id)
            unique_list.append((idx, inf))
        pairwise_inf_by_tc[tc_id] = unique_list

    out = Path(output_path) if output_path else build_stage_output_path(
        "judgements",
        cfg.run_id,
        ".jsonl",
        started_at=started_at,
        run_dir=run_dir,
    )

    mode = cfg.protocol.evaluation_mode
    blinding_enabled = cfg.protocol.blinding.enabled

    # ── Planner ──────────────────────────────────────────────
    tasks: list[PlannedTask[JudgeTaskPayload]] = []
    task_index = 0

    for tc_id, inf_list in inf_by_tc.items():
        tc = tc_map.get(tc_id)
        if not tc:
            continue

        for judge_ref in cfg.judges:
            for _repeat in range(cfg.protocol.repeats.judge_repeats):
                if mode in ("pairwise", "hybrid"):
                    pairwise_list = pairwise_inf_by_tc.get(tc_id, [])
                    for (idx_a, inf_a), (idx_b, inf_b) in combinations(pairwise_list, 2):
                        if blinding_enabled:
                            swap = should_swap_pairwise_order(
                                inf_a.candidate_id,
                                inf_b.candidate_id,
                                cfg.protocol.blinding.random_seed,
                            )
                        else:
                            swap = False
                        tasks.append(PlannedTask(
                            task_index=task_index,
                            payload=JudgePairwisePayload(
                                tc=tc,
                                inf_a=inf_a,
                                inf_b=inf_b,
                                idx_a=idx_a,
                                idx_b=idx_b,
                                judge_ref=judge_ref,
                                presented_swap=swap,
                            ),
                        ))
                        task_index += 1

                if mode in ("absolute", "hybrid"):
                    for idx, inf in inf_list:
                        tasks.append(PlannedTask(
                            task_index=task_index,
                            payload=JudgeAbsolutePayload(
                                tc=tc,
                                inf=inf,
                                idx=idx,
                                judge_ref=judge_ref,
                            ),
                        ))
                        task_index += 1

    total = len(tasks)

    # ── Worker ───────────────────────────────────────────────
    with Progress() as progress:
        bar_task = progress.add_task("Judging", total=total)

        def worker_fn(payload: JudgeTaskPayload) -> JudgementRecord:
            client = get_thread_local_openai_client(
                payload.judge_ref.vendor, payload.judge_ref.endpoint
            )
            if isinstance(payload, JudgePairwisePayload):
                progress_log(
                    "[judge] start mode=pairwise testcase={} judge={} pair={} vs {}".format(
                        payload.tc.testcase_id,
                        payload.judge_ref.judge_id,
                        payload.inf_a.candidate_id,
                        payload.inf_b.candidate_id,
                    )
                )
                rec = _judge_pairwise(
                    cfg=cfg,
                    tc=payload.tc,
                    inf_a=payload.inf_a,
                    inf_b=payload.inf_b,
                    idx_a=payload.idx_a,
                    idx_b=payload.idx_b,
                    judge_ref=payload.judge_ref,
                    presented_swap=payload.presented_swap,
                    inf_path=inf_path,
                    client=client,
                )
                progress_log(
                    "[judge] done mode=pairwise testcase={} judge={} winner={}".format(
                        payload.tc.testcase_id,
                        payload.judge_ref.judge_id,
                        rec.scores.overall_winner or "n/a",
                    )
                )
            else:
                progress_log(
                    "[judge] start mode=absolute testcase={} judge={} candidate={}".format(
                        payload.tc.testcase_id,
                        payload.judge_ref.judge_id,
                        payload.inf.candidate_id,
                    )
                )
                rec = _judge_absolute(
                    cfg=cfg,
                    tc=payload.tc,
                    inf=payload.inf,
                    idx=payload.idx,
                    judge_ref=payload.judge_ref,
                    inf_path=inf_path,
                    client=client,
                )
                progress_log(
                    "[judge] done mode=absolute testcase={} judge={} candidate={} score={}".format(
                        payload.tc.testcase_id,
                        payload.judge_ref.judge_id,
                        payload.inf.candidate_id,
                        rec.scores.overall_score
                        if rec.scores.overall_score is not None
                        else "n/a",
                    )
                )
            progress.advance(bar_task)
            return rec

        results = run_bounded_parallel(
            tasks,
            worker_fn,
            max_workers=cfg.protocol.parallelism.judge_max_concurrency,
        )

    validate_artifacts("judgement-record", results)
    write_jsonl(out, results)
    return out


def _judge_pairwise(
    cfg: RunConfig,
    tc: Testcase,
    inf_a: InferenceRecord,
    inf_b: InferenceRecord,
    idx_a: int,
    idx_b: int,
    judge_ref: JudgeRef,
    presented_swap: bool,
    inf_path: str | Path,
    client: OpenAI,
) -> JudgementRecord:
    """Run a pairwise judge evaluation."""
    inf_path_str = _normalize_inference_ref_path(inf_path)
    judge_metrics = _filter_llm_metrics(cfg.protocol.metrics, tc)

    # Short-circuit: failed or empty inference → deterministic judgement
    # (schema gate 不要: 空文字列に schema validation は無意味)
    fail_a = _is_failed_inference(inf_a)
    fail_b = _is_failed_inference(inf_b)
    if fail_a or fail_b:
        ci_candidates: list[str] = []
        if fail_a:
            ci_candidates.append(inf_a.candidate_id)
        if fail_b:
            ci_candidates.append(inf_b.candidate_id)

        if fail_a and fail_b:
            winner = "tie"
            per_metric = {m: 1 for m in judge_metrics}
            reasons = [
                f"{inf_a.candidate_id}: {_inference_failure_reason(inf_a)}",
                f"{inf_b.candidate_id}: {_inference_failure_reason(inf_b)}",
            ]
            rationale = f"Deterministic inference failure comparison: both failed ({'; '.join(reasons)})"
        elif fail_a:
            winner = inf_b.candidate_id
            per_metric = {m: 5 for m in judge_metrics}
            rationale = (
                f"Deterministic inference failure comparison: "
                f"{inf_a.candidate_id} failed ({_inference_failure_reason(inf_a)}), "
                f"{inf_b.candidate_id} is the winner"
            )
        else:
            winner = inf_a.candidate_id
            per_metric = {m: 5 for m in judge_metrics}
            rationale = (
                f"Deterministic inference failure comparison: "
                f"{inf_b.candidate_id} failed ({_inference_failure_reason(inf_b)}), "
                f"{inf_a.candidate_id} is the winner"
            )

        logger.warning(
            "Deterministic failure judgement: mode=pairwise testcase=%s pair=%s,%s winner=%s",
            tc.testcase_id, inf_a.candidate_id, inf_b.candidate_id, winner,
        )
        return JudgementRecord(
            run_id=cfg.run_id,
            testcase_id=tc.testcase_id,
            judge=JudgeInfo(
                judge_id=judge_ref.judge_id,
                vendor=judge_ref.vendor,
                model_id=judge_ref.model_id,
                rubric_version=judge_ref.rubric_version,
                prompt_version=judge_ref.prompt_version,
            ),
            mode="pairwise",
            targets=[
                JudgeTarget(
                    candidate_id=inf_a.candidate_id,
                    inference_ref=InferenceRef(path=inf_path_str, line_index=idx_a),
                ),
                JudgeTarget(
                    candidate_id=inf_b.candidate_id,
                    inference_ref=InferenceRef(path=inf_path_str, line_index=idx_b),
                ),
            ],
            blinding=BlindingInfo(
                enabled=cfg.protocol.blinding.enabled,
                presented_order=[inf_a.candidate_id, inf_b.candidate_id],
                random_seed=cfg.protocol.blinding.random_seed,
            ),
            scores=Scores(per_metric=per_metric, overall_winner=winner),
            critical_issue=True,
            critical_issue_candidates=ci_candidates,
            overall_rationale=rationale,
        )

    # Schema gate (failed inference を通過した後のみ実行)
    schema_gate = _pairwise_schema_gate(tc, inf_a, inf_b, judge_metrics)
    if schema_gate is not None:
        # 片方以上が Schema failed → programmatic 採点、Judge から FC 除外
        judge_metrics = [m for m in judge_metrics if m != "format_compliance"]

    # Layered evaluation (detected from rubric content)
    if rubric_supports_layers(judge_ref.rubric_version):
        return _judge_pairwise_layered(
            cfg, tc, inf_a, inf_b, idx_a, idx_b, judge_ref,
            presented_swap, inf_path, client,
            schema_gate=schema_gate, judge_metrics_override=judge_metrics,
        )

    # Blinding: deterministic presentation order based on presented_swap flag
    # (computed by should_swap_pairwise_order() in the planner).
    blinding_enabled = cfg.protocol.blinding.enabled
    label_first, label_second = "A", "B"
    if blinding_enabled and presented_swap:
        presented = [inf_b.candidate_id, inf_a.candidate_id]
        out_first, out_second = inf_b.output.text, inf_a.output.text
        cand_map = {"A": inf_b.candidate_id, "B": inf_a.candidate_id}
    else:
        presented = [inf_a.candidate_id, inf_b.candidate_id]
        out_first, out_second = inf_a.output.text, inf_b.output.text
        cand_map = {"A": inf_a.candidate_id, "B": inf_b.candidate_id}

    try:
        messages = build_pairwise_judge_prompt(
            testcase=tc,
            output_a=out_first,
            output_b=out_second,
            label_a=label_first,
            label_b=label_second,
            metrics=judge_metrics,
            rubric_version=judge_ref.rubric_version,
        )

        completion = judge_chat_completion(
            vendor=judge_ref.vendor,
            model=judge_ref.model_id,
            messages=messages,
            endpoint=judge_ref.endpoint,
            generation_params=judge_ref.generation_params or None,
            client=client,
        )
        result_text = completion.text or "{}"
        result = json.loads(result_text)

        per_metric, dropped = _normalize_per_metric(
            result.get("per_metric", {}), allow_dict_values=True,
        )
        if dropped:
            logger.warning(
                "Dropped invalid judge metrics: mode=pairwise testcase=%s judge=%s pair=%s,%s dropped=%s",
                tc.testcase_id, judge_ref.judge_id,
                inf_a.candidate_id, inf_b.candidate_id, dropped,
            )
        raw_winner = result.get("overall_winner", "tie")
        # Map label back to candidate_id
        if raw_winner in cand_map:
            winner = cand_map[raw_winner]
        elif raw_winner == "tie":
            winner = "tie"
        else:
            winner = raw_winner
        rationale = result.get("overall_rationale") or result.get("rationale")
        ci_a = bool(result.get("critical_issue_a", False))
        ci_b = bool(result.get("critical_issue_b", False))
        critical_issue = ci_a or ci_b
        ci_candidates: list[str] = []
        if ci_a:
            ci_candidates.append(cand_map["A"])
        if ci_b:
            ci_candidates.append(cand_map["B"])

        # Schema gate → FC programmatic 採点を挿入
        # presented_swap 時は score_a/score_b を提示ラベル順に入れ替え
        if schema_gate is not None:
            sg = schema_gate.per_metric_value
            if blinding_enabled and presented_swap:
                sg = PairwiseMetricScore(
                    rationale=sg.rationale,
                    score_a=sg.score_b,
                    score_b=sg.score_a,
                )
            per_metric["format_compliance"] = sg
            if schema_gate.winner_override is not None:
                winner = schema_gate.winner_override

        return JudgementRecord(
            run_id=cfg.run_id,
            testcase_id=tc.testcase_id,
            judge=JudgeInfo(
                judge_id=judge_ref.judge_id,
                vendor=judge_ref.vendor,
                model_id=judge_ref.model_id,
                rubric_version=judge_ref.rubric_version,
                prompt_version=judge_ref.prompt_version,
            ),
            mode="pairwise",
            targets=[
                JudgeTarget(
                    candidate_id=inf_a.candidate_id,
                    inference_ref=InferenceRef(path=inf_path_str, line_index=idx_a),
                ),
                JudgeTarget(
                    candidate_id=inf_b.candidate_id,
                    inference_ref=InferenceRef(path=inf_path_str, line_index=idx_b),
                ),
            ],
            blinding=BlindingInfo(
                enabled=blinding_enabled,
                presented_order=presented,
                random_seed=cfg.protocol.blinding.random_seed,
            ),
            scores=Scores(per_metric=per_metric, overall_winner=winner),
            critical_issue=critical_issue,
            critical_issue_candidates=ci_candidates,
            overall_rationale=rationale,
        )

    except Exception as e:
        return JudgementRecord(
            run_id=cfg.run_id,
            testcase_id=tc.testcase_id,
            judge=JudgeInfo(
                judge_id=judge_ref.judge_id,
                vendor=judge_ref.vendor,
                model_id=judge_ref.model_id,
                rubric_version=judge_ref.rubric_version,
                prompt_version=judge_ref.prompt_version,
            ),
            mode="pairwise",
            targets=[
                JudgeTarget(
                    candidate_id=inf_a.candidate_id,
                    inference_ref=InferenceRef(path=inf_path_str, line_index=idx_a),
                ),
                JudgeTarget(
                    candidate_id=inf_b.candidate_id,
                    inference_ref=InferenceRef(path=inf_path_str, line_index=idx_b),
                ),
            ],
            blinding=BlindingInfo(
                enabled=blinding_enabled,
                presented_order=presented,
                random_seed=cfg.protocol.blinding.random_seed,
            ),
            scores=Scores(per_metric={}, overall_winner="tie"),
            critical_issue=False,
            critical_issue_candidates=[],
            overall_rationale=f"Judge error: {e}",
        )


def _judge_absolute(
    cfg: RunConfig,
    tc: Testcase,
    inf: InferenceRecord,
    idx: int,
    judge_ref: JudgeRef,
    inf_path: str | Path,
    client: OpenAI,
) -> JudgementRecord:
    """Run an absolute (single-answer) judge evaluation."""
    inf_path_str = _normalize_inference_ref_path(inf_path)
    judge_metrics = _filter_llm_metrics(cfg.protocol.metrics, tc)

    # Short-circuit: failed or empty inference → deterministic minimum score
    # (schema gate 不要: 空文字列に schema validation は無意味)
    if _is_failed_inference(inf):
        reason = _inference_failure_reason(inf)
        logger.warning(
            "Deterministic failure judgement: mode=absolute testcase=%s candidate=%s reason=%s",
            tc.testcase_id, inf.candidate_id, reason,
        )
        per_metric = {m: 1 for m in judge_metrics}
        return JudgementRecord(
            run_id=cfg.run_id,
            testcase_id=tc.testcase_id,
            judge=JudgeInfo(
                judge_id=judge_ref.judge_id,
                vendor=judge_ref.vendor,
                model_id=judge_ref.model_id,
                rubric_version=judge_ref.rubric_version,
                prompt_version=judge_ref.prompt_version,
            ),
            mode="absolute",
            targets=[
                JudgeTarget(
                    candidate_id=inf.candidate_id,
                    inference_ref=InferenceRef(path=inf_path_str, line_index=idx),
                ),
            ],
            scores=Scores(per_metric=per_metric, overall_score=1.0),
            critical_issue=True,
            critical_issue_candidates=[inf.candidate_id],
            overall_rationale=f"Deterministic inference failure: {reason}",
        )

    # Schema gate (failed inference を通過した後のみ実行)
    schema_fc = _get_schema_gate_result(tc, inf.output.text, judge_metrics)
    if schema_fc is not None:
        # Schema failed → FC=1 確定、Judge の format_compliance はスキップ
        judge_metrics = [m for m in judge_metrics if m != "format_compliance"]

    # Layered evaluation (detected from rubric content)
    if rubric_supports_layers(judge_ref.rubric_version):
        return _judge_absolute_layered(
            cfg, tc, inf, idx, judge_ref, inf_path, client,
            schema_fc=schema_fc, judge_metrics_override=judge_metrics,
        )

    try:
        messages = build_absolute_judge_prompt(
            testcase=tc,
            output_text=inf.output.text,
            candidate_label=inf.candidate_id,
            metrics=judge_metrics,
            rubric_version=judge_ref.rubric_version,
        )

        completion = judge_chat_completion(
            vendor=judge_ref.vendor,
            model=judge_ref.model_id,
            messages=messages,
            endpoint=judge_ref.endpoint,
            generation_params=judge_ref.generation_params or None,
            client=client,
        )
        result_text = completion.text or "{}"
        result = json.loads(result_text)

        per_metric, dropped = _normalize_per_metric(
            result.get("per_metric", {}), allow_dict_values=False,
        )
        if dropped:
            logger.warning(
                "Dropped invalid judge metrics: mode=absolute testcase=%s judge=%s candidate=%s dropped=%s",
                tc.testcase_id, judge_ref.judge_id, inf.candidate_id, dropped,
            )
        overall_score = result.get("overall_score")
        rationale = result.get("overall_rationale") or result.get("rationale")
        critical_issue = bool(result.get("critical_issue", False))

        # Schema failed → FC=1 を挿入
        if schema_fc is not None:
            per_metric["format_compliance"] = schema_fc

        return JudgementRecord(
            run_id=cfg.run_id,
            testcase_id=tc.testcase_id,
            judge=JudgeInfo(
                judge_id=judge_ref.judge_id,
                vendor=judge_ref.vendor,
                model_id=judge_ref.model_id,
                rubric_version=judge_ref.rubric_version,
                prompt_version=judge_ref.prompt_version,
            ),
            mode="absolute",
            targets=[
                JudgeTarget(
                    candidate_id=inf.candidate_id,
                    inference_ref=InferenceRef(path=inf_path_str, line_index=idx),
                ),
            ],
            scores=Scores(per_metric=per_metric, overall_score=overall_score),
            critical_issue=critical_issue,
            critical_issue_candidates=[inf.candidate_id] if critical_issue else [],
            overall_rationale=rationale,
        )

    except Exception as e:
        return JudgementRecord(
            run_id=cfg.run_id,
            testcase_id=tc.testcase_id,
            judge=JudgeInfo(
                judge_id=judge_ref.judge_id,
                vendor=judge_ref.vendor,
                model_id=judge_ref.model_id,
                rubric_version=judge_ref.rubric_version,
                prompt_version=judge_ref.prompt_version,
            ),
            mode="absolute",
            targets=[
                JudgeTarget(
                    candidate_id=inf.candidate_id,
                    inference_ref=InferenceRef(path=inf_path_str, line_index=idx),
                ),
            ],
            scores=Scores(per_metric={}, overall_score=None),
            critical_issue=False,
            critical_issue_candidates=[],
            overall_rationale=f"Judge error: {e}",
        )


def _judge_absolute_layered(
    cfg: RunConfig,
    tc: Testcase,
    inf: InferenceRecord,
    idx: int,
    judge_ref: JudgeRef,
    inf_path: str | Path,
    client: OpenAI,
    *,
    schema_fc: int | None = None,
    judge_metrics_override: list[str] | None = None,
) -> JudgementRecord:
    """Run a layered absolute judge evaluation (up to 3 independent calls)."""
    inf_path_str = _normalize_inference_ref_path(inf_path)
    judge_metrics = judge_metrics_override if judge_metrics_override is not None else _filter_llm_metrics(cfg.protocol.metrics, tc)

    layers = build_layered_judge_prompts(
        testcase=tc,
        metrics=judge_metrics,
        rubric_version=judge_ref.rubric_version,
        mode="absolute",
        output_text=inf.output.text,
        candidate_label=inf.candidate_id,
    )

    all_per_metric: dict[str, MetricScore | PairwiseMetricScore | int] = {}
    critical_issue = False
    ci_candidates: list[str] = []
    rationale_parts: list[str] = []
    overall_score: float | None = None

    for layer_name, layer_metrics, messages in layers:
        try:
            completion = judge_chat_completion(
                vendor=judge_ref.vendor,
                model=judge_ref.model_id,
                messages=messages,
                endpoint=judge_ref.endpoint,
                generation_params=judge_ref.generation_params or None,
                client=client,
            )
            result_text = completion.text or "{}"
            result = json.loads(result_text)

            per_metric, dropped = _normalize_per_metric(
                result.get("per_metric", {}), allow_dict_values=False,
            )
            if dropped:
                logger.warning(
                    "Dropped invalid judge metrics: layer=%s mode=absolute testcase=%s judge=%s candidate=%s dropped=%s",
                    layer_name, tc.testcase_id, judge_ref.judge_id, inf.candidate_id, dropped,
                )
            all_per_metric.update(per_metric)

            if layer_name == "layer2":
                overall_score = result.get("overall_score")

            layer_ci = bool(result.get("critical_issue", False))
            if layer_ci:
                critical_issue = True
                if inf.candidate_id not in ci_candidates:
                    ci_candidates.append(inf.candidate_id)

            rationale = result.get("overall_rationale") or result.get("rationale")
            if rationale:
                rationale_parts.append(rationale)

        except Exception as e:
            logger.warning(
                "Judge layer error: layer=%s testcase=%s: %s",
                layer_name, tc.testcase_id, e,
            )
            rationale_parts.append(f"[{layer_name}] Judge error: {e}")

    # Schema failed → FC=1 を挿入
    if schema_fc is not None:
        all_per_metric["format_compliance"] = schema_fc

    # Layer 1 gate: cap overall_score at 2.0
    fc_score = _extract_gate_score(all_per_metric.get("format_compliance"))
    harm_score = _extract_gate_score(all_per_metric.get("harmlessness"))
    if fc_score == 1 or (harm_score is not None and harm_score <= 2):
        critical_issue = True
        if inf.candidate_id not in ci_candidates:
            ci_candidates.append(inf.candidate_id)
        if overall_score is not None and overall_score > 2.0:
            overall_score = 2.0

    return JudgementRecord(
        run_id=cfg.run_id,
        testcase_id=tc.testcase_id,
        judge=JudgeInfo(
            judge_id=judge_ref.judge_id,
            vendor=judge_ref.vendor,
            model_id=judge_ref.model_id,
            rubric_version=judge_ref.rubric_version,
            prompt_version=judge_ref.prompt_version,
        ),
        mode="absolute",
        targets=[
            JudgeTarget(
                candidate_id=inf.candidate_id,
                inference_ref=InferenceRef(path=inf_path_str, line_index=idx),
            ),
        ],
        scores=Scores(per_metric=all_per_metric, overall_score=overall_score),
        critical_issue=critical_issue,
        critical_issue_candidates=ci_candidates,
        overall_rationale=" | ".join(rationale_parts) if rationale_parts else None,
    )


def _judge_pairwise_layered(
    cfg: RunConfig,
    tc: Testcase,
    inf_a: InferenceRecord,
    inf_b: InferenceRecord,
    idx_a: int,
    idx_b: int,
    judge_ref: JudgeRef,
    presented_swap: bool,
    inf_path: str | Path,
    client: OpenAI,
    *,
    schema_gate: PairwiseSchemaGateResult | None = None,
    judge_metrics_override: list[str] | None = None,
) -> JudgementRecord:
    """Run a layered pairwise judge evaluation (up to 3 independent calls)."""
    inf_path_str = _normalize_inference_ref_path(inf_path)
    judge_metrics = judge_metrics_override if judge_metrics_override is not None else _filter_llm_metrics(cfg.protocol.metrics, tc)

    blinding_enabled = cfg.protocol.blinding.enabled
    label_first, label_second = "A", "B"
    if blinding_enabled and presented_swap:
        presented = [inf_b.candidate_id, inf_a.candidate_id]
        out_first, out_second = inf_b.output.text, inf_a.output.text
        cand_map = {"A": inf_b.candidate_id, "B": inf_a.candidate_id}
    else:
        presented = [inf_a.candidate_id, inf_b.candidate_id]
        out_first, out_second = inf_a.output.text, inf_b.output.text
        cand_map = {"A": inf_a.candidate_id, "B": inf_b.candidate_id}

    layers = build_layered_judge_prompts(
        testcase=tc,
        metrics=judge_metrics,
        rubric_version=judge_ref.rubric_version,
        mode="pairwise",
        output_a=out_first,
        output_b=out_second,
        label_a=label_first,
        label_b=label_second,
    )

    all_per_metric: dict[str, MetricScore | PairwiseMetricScore | int] = {}
    ci_a = False
    ci_b = False
    rationale_parts: list[str] = []
    overall_winner: str | None = None

    for layer_name, layer_metrics, messages in layers:
        try:
            completion = judge_chat_completion(
                vendor=judge_ref.vendor,
                model=judge_ref.model_id,
                messages=messages,
                endpoint=judge_ref.endpoint,
                generation_params=judge_ref.generation_params or None,
                client=client,
            )
            result_text = completion.text or "{}"
            result = json.loads(result_text)

            per_metric, dropped = _normalize_per_metric(
                result.get("per_metric", {}), allow_dict_values=True,
            )
            if dropped:
                logger.warning(
                    "Dropped invalid judge metrics: layer=%s mode=pairwise testcase=%s judge=%s pair=%s,%s dropped=%s",
                    layer_name, tc.testcase_id, judge_ref.judge_id,
                    inf_a.candidate_id, inf_b.candidate_id, dropped,
                )
            all_per_metric.update(per_metric)

            # Collect overall_winner from layer2 (content layer)
            if layer_name == "layer2":
                raw_winner = result.get("overall_winner", "tie")
                if raw_winner in cand_map:
                    overall_winner = cand_map[raw_winner]
                elif raw_winner == "tie":
                    overall_winner = "tie"
                else:
                    overall_winner = raw_winner

            if bool(result.get("critical_issue_a", False)):
                ci_a = True
            if bool(result.get("critical_issue_b", False)):
                ci_b = True

            rationale = result.get("overall_rationale") or result.get("rationale")
            if rationale:
                rationale_parts.append(rationale)

        except Exception as e:
            logger.warning(
                "Judge layer error: layer=%s testcase=%s: %s",
                layer_name, tc.testcase_id, e,
            )
            rationale_parts.append(f"[{layer_name}] Judge error: {e}")

    # Schema gate → FC programmatic 採点を挿入
    # score_a/score_b は inf_a/inf_b 順で格納されているため、
    # presented_swap 時は提示ラベル A/B に合わせて入れ替える
    if schema_gate is not None:
        sg = schema_gate.per_metric_value
        if blinding_enabled and presented_swap:
            sg = PairwiseMetricScore(
                rationale=sg.rationale,
                score_a=sg.score_b,
                score_b=sg.score_a,
            )
        all_per_metric["format_compliance"] = sg
        if schema_gate.winner_override is not None:
            overall_winner = schema_gate.winner_override

    # Layer 1 gate: override winner if one side has format/safety failure
    fc_a, fc_b = _extract_pairwise_gate_scores(all_per_metric.get("format_compliance"))
    harm_a, harm_b = _extract_pairwise_gate_scores(all_per_metric.get("harmlessness"))

    gate_fail_a = (fc_a == 1) or (harm_a is not None and harm_a <= 2)
    gate_fail_b = (fc_b == 1) or (harm_b is not None and harm_b <= 2)

    if gate_fail_a:
        ci_a = True
    if gate_fail_b:
        ci_b = True

    # Override overall_winner: layer1 gate takes priority over layer2 result
    if gate_fail_a and not gate_fail_b:
        overall_winner = cand_map["B"]
    elif gate_fail_b and not gate_fail_a:
        overall_winner = cand_map["A"]
    elif gate_fail_a and gate_fail_b:
        overall_winner = "tie"

    critical_issue = ci_a or ci_b
    ci_candidates: list[str] = []
    if ci_a:
        ci_candidates.append(cand_map["A"])
    if ci_b:
        ci_candidates.append(cand_map["B"])

    if overall_winner is None:
        overall_winner = "tie"

    return JudgementRecord(
        run_id=cfg.run_id,
        testcase_id=tc.testcase_id,
        judge=JudgeInfo(
            judge_id=judge_ref.judge_id,
            vendor=judge_ref.vendor,
            model_id=judge_ref.model_id,
            rubric_version=judge_ref.rubric_version,
            prompt_version=judge_ref.prompt_version,
        ),
        mode="pairwise",
        targets=[
            JudgeTarget(
                candidate_id=inf_a.candidate_id,
                inference_ref=InferenceRef(path=inf_path_str, line_index=idx_a),
            ),
            JudgeTarget(
                candidate_id=inf_b.candidate_id,
                inference_ref=InferenceRef(path=inf_path_str, line_index=idx_b),
            ),
        ],
        blinding=BlindingInfo(
            enabled=blinding_enabled,
            presented_order=presented,
            random_seed=cfg.protocol.blinding.random_seed,
        ),
        scores=Scores(per_metric=all_per_metric, overall_winner=overall_winner),
        critical_issue=critical_issue,
        critical_issue_candidates=ci_candidates,
        overall_rationale=" | ".join(rationale_parts) if rationale_parts else None,
    )


def _filter_llm_metrics(
    metrics: list[str],
    tc: Testcase,
) -> list[str]:
    return [
        metric_id
        for metric_id in metrics
        if (
            metric_id not in _TASK_RESTRICTED_METRICS
            or tc.task_type in _TASK_RESTRICTED_METRICS[metric_id]
        )
    ]


def _get_schema_gate_result(
    tc: Testcase,
    output_text: str,
    judge_metrics: list[str],
) -> int | None:
    """Schema failed なら FC=1 を返す。passed or スキーマなしなら None（Judge に委ねる）。

    judge_metrics に format_compliance が含まれない場合は常に None を返す。
    """
    if "format_compliance" not in judge_metrics:
        return None
    result = validate_output_against_testcase_schema(tc, output_text)
    if result is None:
        return None  # スキーマ指定なし → Judge に委ねる
    if not result.passed:
        return 1  # Schema failed → FC=1 確定
    return None  # Schema passed → Judge に委ねる


@dataclass
class PairwiseSchemaGateResult:
    per_metric_value: PairwiseMetricScore
    winner_override: str | None


def _pairwise_schema_gate(
    tc: Testcase,
    inf_a: InferenceRecord,
    inf_b: InferenceRecord,
    judge_metrics: list[str],
) -> PairwiseSchemaGateResult | None:
    """片方以上が Schema failed なら programmatic スコアを返す。両方 passed or スキーマなしなら None。

    judge_metrics に format_compliance が含まれない場合は常に None を返す。
    """
    if "format_compliance" not in judge_metrics:
        return None

    result_a = validate_output_against_testcase_schema(tc, inf_a.output.text)
    result_b = validate_output_against_testcase_schema(tc, inf_b.output.text)

    # スキーマ指定なし → Judge に委ねる
    if result_a is None or result_b is None:
        return None

    passed_a = bool(result_a and result_a.passed)
    passed_b = bool(result_b and result_b.passed)

    if passed_a and passed_b:
        return None  # 両方 passed → Judge に委ねる
    if passed_a and not passed_b:
        return PairwiseSchemaGateResult(
            per_metric_value=PairwiseMetricScore(
                rationale=f"Schema validation: {inf_a.candidate_id} passed, {inf_b.candidate_id} failed",
                score_a=5, score_b=1,
            ),
            winner_override=inf_a.candidate_id,
        )
    if passed_b and not passed_a:
        return PairwiseSchemaGateResult(
            per_metric_value=PairwiseMetricScore(
                rationale=f"Schema validation: {inf_a.candidate_id} failed, {inf_b.candidate_id} passed",
                score_a=1, score_b=5,
            ),
            winner_override=inf_b.candidate_id,
        )
    return PairwiseSchemaGateResult(
        per_metric_value=PairwiseMetricScore(
            rationale="Schema validation: both failed",
            score_a=1, score_b=1,
        ),
        winner_override=None,
    )  # 両方 failed
