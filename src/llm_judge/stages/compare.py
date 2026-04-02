"""Stage 4: Aggregation and comparison report generation."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from llm_judge.artifact_validation import validate_single_artifact
from llm_judge.config import load_run_config
from llm_judge.models import (
    AggregateBlock,
    AggregationMethod,
    AutoCheckRecord,
    CandidateInfo,
    ComparisonReport,
    ConsistencyRecord,
    JudgeAgreement,
    JudgeSummary,
    JudgementRecord,
    MetricScore,
    NotableFailure,
    PairwiseMetricScore,
    ReportDataset,
    ReportSummary,
    Results,
    Testcase,
)


def _extract_score(value: MetricScore | PairwiseMetricScore | int) -> float | None:
    if isinstance(value, MetricScore):
        return float(value.score)
    if isinstance(value, PairwiseMetricScore):
        return (value.score_a + value.score_b) / 2.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


from llm_judge.testcase_loader import load_testcases as _load_testcases
from llm_judge.utils import (
    build_stage_output_path,
    derive_dataset_layer,
    find_latest_stage_output_path,
    mean,
    read_jsonl,
    write_json,
    write_jsonl,
)


def run_compare(
    config_path: str,
    judgements_path: str | None = None,
    autocheck_path: str | None = None,
    inference_path: str | None = None,
    output_path: str | None = None,
    consistency_path: str | None = None,
    *,
    started_at: datetime | None = None,
    run_dir: Path | None = None,
) -> Path:
    """Aggregate judgements and produce comparison report."""
    cfg = load_run_config(config_path)
    layer = derive_dataset_layer(cfg.dataset.testcases_path)

    jdg_path = (
        Path(judgements_path)
        if judgements_path
        else find_latest_stage_output_path("judgements", cfg.run_id, ".jsonl", run_dir=run_dir, dataset_layer=layer)
    )
    judgements: list[JudgementRecord] = []
    if jdg_path and jdg_path.exists():
        judgements = [JudgementRecord.model_validate(r) for r in read_jsonl(jdg_path)]

    ack_path = (
        Path(autocheck_path)
        if autocheck_path
        else find_latest_stage_output_path("autocheck", cfg.run_id, ".jsonl", run_dir=run_dir, dataset_layer=layer)
    )
    autochecks: list[AutoCheckRecord] = []
    if ack_path and ack_path.exists():
        autochecks = [
            AutoCheckRecord.model_validate(r) for r in read_jsonl(ack_path)
        ]

    con_path = (
        Path(consistency_path)
        if consistency_path
        else find_latest_stage_output_path("consistency", cfg.run_id, ".jsonl", run_dir=run_dir, dataset_layer=layer)
    )
    consistencies: list[ConsistencyRecord] = []
    if con_path and con_path.exists():
        consistencies = [
            ConsistencyRecord.model_validate(r) for r in read_jsonl(con_path)
        ]

    testcases = _load_testcases(cfg.dataset.testcases_path)
    tc_map = {tc.testcase_id: tc for tc in testcases}

    candidate_ids = [c.candidate_id for c in cfg.candidates]

    agg_method = cfg.protocol.aggregation.method

    overall = _compute_aggregate(
        judgements, candidate_ids, autochecks, consistencies,
        method=agg_method,
    )

    by_task = _compute_by_group(
        judgements, candidate_ids, autochecks, tc_map, consistencies, group_by="task_type",
        method=agg_method,
    )
    by_bucket = _compute_by_group(
        judgements, candidate_ids, autochecks, tc_map, consistencies, group_by="bucket",
        method=agg_method,
    )

    agreement = _compute_judge_agreement(judgements)
    summary = _compute_summary(judgements)

    report = ComparisonReport(
        run_id=cfg.run_id,
        dataset=ReportDataset(
            dataset_version=cfg.dataset.dataset_version,
            testcase_count=len(testcases),
        ),
        candidates=[
            CandidateInfo(
                candidate_id=c.candidate_id,
                vendor=c.vendor,
                model_id=c.model_id,
            )
            for c in cfg.candidates
        ],
        judges=[
            JudgeSummary(
                judge_id=j.judge_id,
                vendor=j.vendor,
                model_id=j.model_id,
                rubric_version=j.rubric_version,
            )
            for j in cfg.judges
        ],
        protocol=cfg.protocol.model_dump(),
        summary=summary,
        results=Results(
            overall=overall,
            by_task=by_task,
            by_bucket=by_bucket,
            judge_agreement=agreement,
        ),
    )

    json_out = Path(output_path) if output_path else build_stage_output_path(
        "comparison-report",
        cfg.run_id,
        ".json",
        started_at=started_at,
        run_dir=run_dir,
    )
    validate_single_artifact("comparison-report", report)
    write_json(json_out, report)

    md_out = json_out.with_suffix(".md")
    _write_markdown_report(report, md_out, judgements=judgements)

    return json_out


def _mode_value(values: list[float | int]) -> float:
    from collections import Counter

    if not values:
        return 0.0
    counts = Counter(values)
    max_count = max(counts.values())
    modes = [v for v, c in counts.items() if c == max_count]
    return float(min(modes))


def _aggregate_scores(
    method: AggregationMethod,
    metric_scores: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, float]]:
    if method == "mean":
        return {
            m: {cid: round(mean(scores), 2) for cid, scores in by_cid.items()}
            for m, by_cid in metric_scores.items()
        }
    if method == "worst_case":
        return {
            m: {cid: round(min(scores), 2) for cid, scores in by_cid.items()}
            for m, by_cid in metric_scores.items()
        }
    if method == "majority_vote":
        return {
            m: {cid: round(mean(scores), 2) for cid, scores in by_cid.items()}
            for m, by_cid in metric_scores.items()
        }
    raise ValueError(f"Unknown aggregation method: '{method}'")


def _compute_overall_score_aggregate(
    method: AggregationMethod,
    candidate_ids: list[str],
    judge_overall_scores: dict[str, list[float]] | None = None,
) -> dict[str, float]:
    if not judge_overall_scores:
        return {}

    result: dict[str, float] = {}
    for cid in candidate_ids:
        scores = judge_overall_scores.get(cid, [])
        if scores:
            if method == "worst_case":
                result[cid] = round(min(scores), 2)
            else:
                result[cid] = round(mean(scores), 2)
    return result


def _count_pairwise_majority(
    judgements: list[JudgementRecord],
) -> dict[str, int]:
    groups: dict[str, list[str]] = defaultdict(list)
    for jdg in judgements:
        if jdg.mode != "pairwise" or not jdg.scores.overall_winner:
            continue
        target_key = "|".join(sorted(t.candidate_id for t in jdg.targets))
        key = f"{jdg.testcase_id}:{target_key}:{jdg.judge.judge_id}"
        groups[key].append(jdg.scores.overall_winner)

    from collections import Counter

    counts: dict[str, int] = defaultdict(int)
    for winners in groups.values():
        c = Counter(winners)
        max_count = max(c.values())
        modes = [w for w, cnt in c.items() if cnt == max_count]
        majority_winner = "tie" if len(modes) > 1 else modes[0]
        counts[majority_winner] += 1

    return dict(counts)


def _aggregate_majority_vote_pairwise(
    judgements: list[JudgementRecord],
    candidate_ids: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    counts = _count_pairwise_majority(judgements)
    total = sum(counts.values())
    if total == 0:
        return {}, {}

    groups: dict[str, list[str]] = defaultdict(list)
    group_targets: dict[str, set[str]] = {}
    for jdg in judgements:
        if jdg.mode != "pairwise" or not jdg.scores.overall_winner:
            continue
        target_key = "|".join(sorted(t.candidate_id for t in jdg.targets))
        key = f"{jdg.testcase_id}:{target_key}:{jdg.judge.judge_id}"
        groups[key].append(jdg.scores.overall_winner)
        if key not in group_targets:
            group_targets[key] = {t.candidate_id for t in jdg.targets}

    from collections import Counter

    win_counts: dict[str, int] = defaultdict(int)
    loss_counts: dict[str, int] = defaultdict(int)
    for key, winners in groups.items():
        c = Counter(winners)
        max_count = max(c.values())
        modes = [w for w, cnt in c.items() if cnt == max_count]
        majority_winner = "tie" if len(modes) > 1 else modes[0]
        if majority_winner != "tie":
            win_counts[majority_winner] += 1
            for cid in group_targets[key]:
                if cid != majority_winner:
                    loss_counts[cid] += 1

    win_rate: dict[str, float] = {}
    loss_rate: dict[str, float] = {}
    for cid in candidate_ids:
        win_rate[cid] = round(win_counts[cid] / total, 4)
        loss_rate[cid] = round(loss_counts[cid] / total, 4)

    return win_rate, loss_rate


def _reduce_absolute_scores_by_majority(
    judgements: list[JudgementRecord],
) -> dict[str, dict[str, list[float]]]:
    raw: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for jdg in judgements:
        if jdg.mode == "pairwise":
            continue
        for t in jdg.targets:
            cid = t.candidate_id
            for metric_id, value in jdg.scores.per_metric.items():
                score = _extract_score(value)
                if score is not None:
                    key = (jdg.testcase_id, cid, jdg.judge.judge_id, metric_id)
                    raw[key].append(score)

    reduced: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (tc_id, cid, judge_id, metric_id), scores in raw.items():
        representative = _mode_value(scores)
        reduced[metric_id][cid].append(representative)

    return reduced


def _compute_aggregate(
    judgements: list[JudgementRecord],
    candidate_ids: list[str],
    autochecks: list[AutoCheckRecord],
    consistencies: list[ConsistencyRecord] | None = None,
    method: AggregationMethod = "mean",
) -> AggregateBlock:
    import math

    from llm_judge.utils import variance as _variance

    win_counts: dict[str, int] = defaultdict(int)
    loss_counts: dict[str, int] = defaultdict(int)
    tie_count = 0
    pairwise_total = 0

    for jdg in judgements:
        if jdg.mode == "pairwise" and jdg.scores.overall_winner:
            pairwise_total += 1
            winner = jdg.scores.overall_winner
            if winner == "tie":
                tie_count += 1
            else:
                win_counts[winner] += 1
                for t in jdg.targets:
                    if t.candidate_id != winner:
                        loss_counts[t.candidate_id] += 1

    win_rate: dict[str, float] = {}
    loss_rate: dict[str, float] = {}
    if pairwise_total > 0:
        for cid in candidate_ids:
            win_rate[cid] = round(win_counts[cid] / pairwise_total, 4)
            loss_rate[cid] = round(loss_counts[cid] / pairwise_total, 4)
        win_rate["tie"] = round(tie_count / pairwise_total, 4)

    metric_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    judge_overall_scores: dict[str, list[float]] = defaultdict(list)
    for jdg in judgements:
        if jdg.mode == "pairwise":
            continue
        for t in jdg.targets:
            cid = t.candidate_id
            if jdg.scores.overall_score is not None:
                judge_overall_scores[cid].append(jdg.scores.overall_score)
            for metric_id, value in jdg.scores.per_metric.items():
                score = _extract_score(value)
                if score is not None:
                    metric_scores[metric_id][cid].append(score)

    if method == "majority_vote":
        metric_scores = _reduce_absolute_scores_by_majority(judgements)

    mean_score = _aggregate_scores(method, metric_scores)

    overall_score_aggregate = _compute_overall_score_aggregate(
        method, candidate_ids,
        judge_overall_scores=dict(judge_overall_scores) if judge_overall_scores else None,
    )

    if method == "majority_vote" and pairwise_total > 0:
        win_rate, loss_rate = _aggregate_majority_vote_pairwise(
            judgements, candidate_ids,
        )
        pairwise_total_adjusted = sum(
            v for k, v in _count_pairwise_majority(judgements).items()
        )
        if pairwise_total_adjusted > 0:
            tie_count_mv = _count_pairwise_majority(judgements).get("tie", 0)
            win_rate["tie"] = round(
                tie_count_mv / pairwise_total_adjusted, 4
            )

    confidence_intervals: dict[str, dict[str, dict[str, float]]] = {}
    for m, by_cid in metric_scores.items():
        ci_by_cid: dict[str, dict[str, float]] = {}
        for cid, scores in by_cid.items():
            n = len(scores)
            if n >= 2:
                avg = mean(scores)
                var = _variance(scores)
                se = math.sqrt(var / n)
                ci_by_cid[cid] = {
                    "mean": round(avg, 4),
                    "lower": round(avg - 1.96 * se, 4),
                    "upper": round(avg + 1.96 * se, 4),
                    "n": n,
                }
        if ci_by_cid:
            confidence_intervals[m] = ci_by_cid

    ci_counts: dict[str, int] = defaultdict(int)
    for jdg in judgements:
        for cid in jdg.critical_issue_candidates:
            ci_counts[cid] += 1
    critical_issue_count = dict(ci_counts) if ci_counts else {}

    failures: list[NotableFailure] = []
    for ac in autochecks:
        if ac.checks.format_compliance and not ac.checks.format_compliance.passed:
            failures.append(
                NotableFailure(
                    testcase_id=ac.testcase_id,
                    candidate_id=ac.candidate_id,
                    reason=f"Format check failed: {ac.checks.format_compliance.details}",
                )
            )
        if (
            ac.checks.json_schema_validation
            and not ac.checks.json_schema_validation.passed
        ):
            failures.append(
                NotableFailure(
                    testcase_id=ac.testcase_id,
                    candidate_id=ac.candidate_id,
                    reason=f"Schema validation failed: {'; '.join(ac.checks.json_schema_validation.errors[:3])}",
                )
            )

    inference_consistency: dict[str, float] = {}
    if consistencies:
        con_scores: dict[str, list[float]] = defaultdict(list)
        for rec in consistencies:
            if rec.status.ok and rec.scores.overall is not None:
                con_scores[rec.candidate_id].append(rec.scores.overall)
        for cid, scores in con_scores.items():
            if scores:
                inference_consistency[cid] = round(mean(scores), 2)

    return AggregateBlock(
        win_rate=win_rate,
        loss_rate=loss_rate,
        mean_score=mean_score,
        overall_score_aggregate=overall_score_aggregate,
        confidence_intervals=confidence_intervals,
        critical_issue_count=critical_issue_count,
        notable_failures=failures,
        inference_consistency=inference_consistency,
    )


def _compute_by_group(
    judgements: list[JudgementRecord],
    candidate_ids: list[str],
    autochecks: list[AutoCheckRecord],
    tc_map: dict[str, Testcase],
    consistencies: list[ConsistencyRecord] | None = None,
    group_by: str = "task_type",
    method: AggregationMethod = "mean",
) -> dict[str, AggregateBlock]:
    groups: dict[str, list[JudgementRecord]] = defaultdict(list)
    ac_groups: dict[str, list[AutoCheckRecord]] = defaultdict(list)
    con_groups: dict[str, list[ConsistencyRecord]] = defaultdict(list)

    for jdg in judgements:
        tc = tc_map.get(jdg.testcase_id)
        if not tc:
            continue
        if group_by == "task_type":
            key = tc.task_type
        elif group_by == "bucket":
            key = (tc.metadata.input_length_bucket if tc.metadata else None) or "unknown"
        else:
            key = "all"
        groups[key].append(jdg)

    for ac in autochecks:
        tc = tc_map.get(ac.testcase_id)
        if not tc:
            continue
        if group_by == "task_type":
            key = tc.task_type
        elif group_by == "bucket":
            key = (tc.metadata.input_length_bucket if tc.metadata else None) or "unknown"
        else:
            key = "all"
        ac_groups[key].append(ac)

    for con in (consistencies or []):
        tc = tc_map.get(con.testcase_id)
        if not tc:
            continue
        if group_by == "task_type":
            key = tc.task_type
        elif group_by == "bucket":
            key = (tc.metadata.input_length_bucket if tc.metadata else None) or "unknown"
        else:
            key = "all"
        con_groups[key].append(con)

    result = {}
    for key, jdgs in groups.items():
        result[key] = _compute_aggregate(
            jdgs, candidate_ids, ac_groups.get(key, []), con_groups.get(key, []),
            method=method,
        )

    return result


def _compute_judge_agreement(judgements: list[JudgementRecord]) -> JudgeAgreement:
    groups: dict[str, list[str]] = defaultdict(list)

    for jdg in judgements:
        if jdg.mode != "pairwise":
            continue
        target_key = "|".join(sorted(t.candidate_id for t in jdg.targets))
        key = f"{jdg.testcase_id}:{target_key}"
        groups[key].append(jdg.scores.overall_winner or "tie")

    if not groups:
        return JudgeAgreement(notes="No pairwise judgements found")

    agreements = 0
    total_pairs = 0

    for winners in groups.values():
        if len(winners) < 2:
            continue
        for i in range(len(winners)):
            for j in range(i + 1, len(winners)):
                total_pairs += 1
                if winners[i] == winners[j]:
                    agreements += 1

    if total_pairs == 0:
        return JudgeAgreement(notes="Insufficient data for agreement calculation")

    rate = round(agreements / total_pairs, 4)
    return JudgeAgreement(
        pairwise_agreement_rate=rate,
        notes=f"Based on {total_pairs} judge pairs across {len(groups)} cases",
    )


def _compute_summary(judgements: list[JudgementRecord]) -> ReportSummary:
    total = len(judgements)
    valid = sum(1 for j in judgements if j.scores.per_metric)
    excluded = total - valid

    stability = _compute_repeat_stability(judgements)

    return ReportSummary(
        total_judgements=total,
        valid_judgements=valid,
        excluded_judgements=excluded,
        repeat_stability=stability,
    )


def _compute_repeat_stability(judgements: list[JudgementRecord]) -> float | None:
    groups: dict[str, list[str]] = defaultdict(list)

    for jdg in judgements:
        if jdg.mode != "pairwise":
            continue
        target_key = "|".join(sorted(t.candidate_id for t in jdg.targets))
        key = f"{jdg.testcase_id}:{jdg.judge.judge_id}:{target_key}"
        groups[key].append(jdg.scores.overall_winner or "tie")

    repeat_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if not repeat_groups:
        return None

    agreements = 0
    total_pairs = 0
    for winners in repeat_groups.values():
        for i in range(len(winners)):
            for j in range(i + 1, len(winners)):
                total_pairs += 1
                if winners[i] == winners[j]:
                    agreements += 1

    if total_pairs == 0:
        return None

    return round(agreements / total_pairs, 4)


def _write_markdown_report(
    report: ComparisonReport,
    path: Path,
    judgements: list[JudgementRecord] | None = None,
) -> None:
    """Write a human-readable markdown report."""
    lines: list[str] = []
    lines.append(f"# Comparison Report: {report.run_id}")
    lines.append("")
    lines.append(f"- Dataset version: {report.dataset.dataset_version}")
    lines.append(f"- Testcase count: {report.dataset.testcase_count}")
    lines.append(
        f"- Candidates: {', '.join(c.candidate_id for c in report.candidates)}"
    )
    lines.append(f"- Judges: {', '.join(j.judge_id for j in report.judges)}")
    lines.append("")

    det_failure_count = 0
    if judgements:
        det_failure_count = sum(
            1 for j in judgements
            if j.overall_rationale and j.overall_rationale.startswith("Deterministic inference failure")
        )
    if det_failure_count > 0:
        lines.append("> [!WARNING]")
        lines.append(
            f"> This report contains **{det_failure_count}** judgements with deterministic "
            f"scoring due to inference failure or empty output."
        )
        lines.append(
            "> Absolute: overall_score=1.0 (minimum). Pairwise: working side wins, both-fail → tie."
        )
        lines.append("")

    s = report.summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total judgements: {s.total_judgements}")
    lines.append(f"- Valid judgements: {s.valid_judgements}")
    lines.append(f"- Excluded judgements: {s.excluded_judgements}")
    if s.repeat_stability is not None:
        lines.append(f"- Repeat stability: {s.repeat_stability:.1%}")
    lines.append("")

    lines.append("## Overall Results")
    lines.append("")

    if report.results.overall.win_rate:
        lines.append("### Win Rate / Loss Rate")
        lines.append(
            "Pairwise comparison results: which candidate produces better answers head-to-head."
        )
        lines.append("")
        lines.append("| Candidate | Win | Loss | Tie |")
        lines.append("|-----------|-----|------|-----|")
        tie_rate_val = report.results.overall.win_rate.get("tie", 0.0)
        for cid in [c.candidate_id for c in report.candidates]:
            w = report.results.overall.win_rate.get(cid, 0.0)
            l = report.results.overall.loss_rate.get(cid, 0.0)
            lines.append(f"| {cid} | {w:.1%} | {l:.1%} | {tie_rate_val:.1%} |")
        lines.append("")

    if report.results.overall.mean_score:
        cids = [c.candidate_id for c in report.candidates]
        lines.append("### Mean Scores by Metric")
        lines.append("Per-metric scores averaged across all test cases (1-5 scale).")
        lines.append("")
        header = "| Metric | " + " | ".join(cids) + " |"
        sep = "|--------" + "|------" * len(cids) + "|"
        lines.append(header)
        lines.append(sep)
        for metric, by_cid in sorted(report.results.overall.mean_score.items()):
            vals = " | ".join(f"{by_cid.get(c, 0.0):.2f}" for c in cids)
            lines.append(f"| {metric} | {vals} |")
        lines.append("")

    if report.results.overall.confidence_intervals:
        cids = [c.candidate_id for c in report.candidates]
        lines.append("### Confidence Intervals (95%)")
        lines.append("")
        header = "| Metric | " + " | ".join(cids) + " |"
        sep = "|--------" + "|------" * len(cids) + "|"
        lines.append(header)
        lines.append(sep)
        for metric, by_cid in sorted(report.results.overall.confidence_intervals.items()):
            vals = []
            for c in cids:
                ci = by_cid.get(c)
                if ci:
                    vals.append(f"{ci['mean']:.2f} [{ci['lower']:.2f}, {ci['upper']:.2f}]")
                else:
                    vals.append("—")
            lines.append(f"| {metric} | {' | '.join(vals)} |")
        lines.append("")

    if report.results.overall.overall_score_aggregate:
        cids = [c.candidate_id for c in report.candidates]
        lines.append("### Overall Score Aggregate")
        lines.append("Aggregate of Judge-provided overall_score values.")
        lines.append("")
        for cid in cids:
            score = report.results.overall.overall_score_aggregate.get(cid, 0.0)
            lines.append(f"- {cid}: {score:.2f}")
        lines.append("")

    if report.results.by_task:
        lines.append("## Results by Task Type")
        lines.append("")
        for task_type, agg in report.results.by_task.items():
            lines.append(f"### {task_type}")
            if agg.win_rate:
                for cid, rate in agg.win_rate.items():
                    lines.append(f"- {cid}: {rate:.1%}")
            if agg.mean_score:
                if agg.win_rate:
                    lines.append("")
                cids = [c.candidate_id for c in report.candidates]
                header = "| Metric | " + " | ".join(cids) + " |"
                sep = "|--------" + "|------" * len(cids) + "|"
                lines.append(header)
                lines.append(sep)
                for metric, by_cid in sorted(agg.mean_score.items()):
                    vals = " | ".join(f"{by_cid.get(c, 0.0):.2f}" for c in cids)
                    lines.append(f"| {metric} | {vals} |")
            lines.append("")

    if report.results.by_bucket:
        lines.append("## Results by Input Length Bucket")
        lines.append("")
        for bucket, agg in report.results.by_bucket.items():
            lines.append(f"### Bucket: {bucket}")
            if agg.win_rate:
                for cid, rate in agg.win_rate.items():
                    lines.append(f"- {cid}: {rate:.1%}")
            if agg.mean_score:
                if agg.win_rate:
                    lines.append("")
                cids = [c.candidate_id for c in report.candidates]
                header = "| Metric | " + " | ".join(cids) + " |"
                sep = "|--------" + "|------" * len(cids) + "|"
                lines.append(header)
                lines.append(sep)
                for metric, by_cid in sorted(agg.mean_score.items()):
                    vals = " | ".join(f"{by_cid.get(c, 0.0):.2f}" for c in cids)
                    lines.append(f"| {metric} | {vals} |")
            lines.append("")

    if report.results.judge_agreement.pairwise_agreement_rate is not None:
        lines.append("## Judge Agreement")
        lines.append(
            f"- Pairwise agreement rate: {report.results.judge_agreement.pairwise_agreement_rate:.1%}"
        )
        if report.results.judge_agreement.notes:
            lines.append(f"- {report.results.judge_agreement.notes}")
        lines.append("")

    if report.results.overall.inference_consistency:
        cids = [c.candidate_id for c in report.candidates]
        lines.append("## Inference Consistency")
        lines.append(
            "LLM-as-a-Judge consistency score (1-5) for repeated inference outputs. "
            "5 = very consistent, 1 = inconsistent."
        )
        lines.append("")
        lines.append("| Candidate | Consistency Score (1-5) |")
        lines.append("|-----------|------------------------|")
        for cid in cids:
            score = report.results.overall.inference_consistency.get(cid)
            score_str = f"{score:.2f}" if score is not None else "—"
            lines.append(f"| {cid} | {score_str} |")
        lines.append("")

    if report.results.overall.critical_issue_count:
        lines.append("## Critical Issues")
        lines.append("")
        lines.append("| Candidate | Count |")
        lines.append("|-----------|-------|")
        for cid, count in sorted(report.results.overall.critical_issue_count.items()):
            lines.append(f"| {cid} | {count} |")
        lines.append("")

    failures = report.results.overall.notable_failures
    if failures:
        lines.append("## Notable Failures")
        lines.append("Format/schema violations detected by autocheck.")
        lines.append("")
        for f in failures:
            lines.append(f"- **{f.testcase_id}** ({f.candidate_id}): {f.reason}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
