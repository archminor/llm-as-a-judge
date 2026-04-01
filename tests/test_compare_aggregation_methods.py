"""Tests for aggregation methods in compare.py (Task C)."""

from __future__ import annotations

import pytest

from llm_judge.models import (
    AggregateBlock,
    InferenceRef,
    JudgeInfo,
    JudgeTarget,
    JudgementRecord,
    Scores,
)
from llm_judge.stages.compare import (
    _aggregate_majority_vote_pairwise,
    _aggregate_scores,
    _compute_aggregate,
    _compute_weighted_overall,
    _mode_value,
    _reduce_absolute_scores_by_majority,
)


def _make_absolute_judgement(
    testcase_id: str,
    candidate_id: str,
    per_metric: dict[str, int],
    judge_id: str = "j1",
    overall_score: float | None = None,
) -> JudgementRecord:
    return JudgementRecord(
        run_id="run-1",
        testcase_id=testcase_id,
        judge=JudgeInfo(
            judge_id=judge_id, vendor="openai", model_id="gpt-4", rubric_version="v1"
        ),
        mode="absolute",
        targets=[
            JudgeTarget(
                candidate_id=candidate_id,
                inference_ref=InferenceRef(path="test.jsonl", line_index=0),
            )
        ],
        scores=Scores(per_metric=per_metric, overall_score=overall_score),
    )


def _make_pairwise_judgement(
    testcase_id: str,
    cand_a: str,
    cand_b: str,
    winner: str,
    per_metric: dict[str, int] | None = None,
    judge_id: str = "j1",
) -> JudgementRecord:
    return JudgementRecord(
        run_id="run-1",
        testcase_id=testcase_id,
        judge=JudgeInfo(
            judge_id=judge_id, vendor="openai", model_id="gpt-4", rubric_version="v1"
        ),
        mode="pairwise",
        targets=[
            JudgeTarget(
                candidate_id=cand_a,
                inference_ref=InferenceRef(path="test.jsonl", line_index=0),
            ),
            JudgeTarget(
                candidate_id=cand_b,
                inference_ref=InferenceRef(path="test.jsonl", line_index=1),
            ),
        ],
        scores=Scores(per_metric=per_metric or {}, overall_winner=winner),
    )


class TestModeValue:
    def test_single_value(self) -> None:
        assert _mode_value([3]) == 3.0

    def test_clear_mode(self) -> None:
        assert _mode_value([1, 3, 3, 5]) == 3.0

    def test_tie_returns_min(self) -> None:
        # 1 and 3 each appear once → min = 1
        assert _mode_value([1, 3]) == 1.0

    def test_empty(self) -> None:
        assert _mode_value([]) == 0.0

    def test_all_same(self) -> None:
        assert _mode_value([5, 5, 5]) == 5.0


class TestAggregateScoresMean:
    def test_mean(self) -> None:
        scores = {"accuracy": {"c1": [1, 3, 5]}}
        result = _aggregate_scores("mean", scores)
        assert result["accuracy"]["c1"] == 3.0


class TestAggregateScoresWorstCase:
    def test_worst_case(self) -> None:
        scores = {"accuracy": {"c1": [1, 3, 5]}}
        result = _aggregate_scores("worst_case", scores)
        assert result["accuracy"]["c1"] == 1.0


class TestAggregateScoresMajorityVote:
    def test_majority_vote_mean_of_reduced(self) -> None:
        """After reduction, _aggregate_scores takes mean of pre-reduced values."""
        # Two groups already reduced to their modes: 5 and 1
        scores = {"accuracy": {"c1": [5, 1]}}
        result = _aggregate_scores("majority_vote", scores)
        assert result["accuracy"]["c1"] == 3.0

    def test_majority_vote_single_group(self) -> None:
        scores = {"accuracy": {"c1": [3]}}
        result = _aggregate_scores("majority_vote", scores)
        assert result["accuracy"]["c1"] == 3.0


class TestAggregateScoresCustom:
    def test_custom_uses_mean(self) -> None:
        scores = {"accuracy": {"c1": [2, 4]}}
        result = _aggregate_scores("custom", scores)
        assert result["accuracy"]["c1"] == 3.0


class TestUnknownMethodRaises:
    def test_unknown_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            _aggregate_scores("nonexistent", {})


class TestComputeWeightedOverall:
    def test_custom_requires_weights(self) -> None:
        with pytest.raises(ValueError, match="requires non-empty weights"):
            _compute_weighted_overall("custom", {}, {}, ["c1"])

    def test_custom_with_weights(self) -> None:
        scores = {
            "accuracy": {"c1": [4.0, 4.0]},
            "fluency": {"c1": [2.0, 2.0]},
        }
        weights = {"accuracy": 0.7, "fluency": 0.3}
        result = _compute_weighted_overall("custom", weights, scores, ["c1"])
        # weighted = (4.0 * 0.7 + 2.0 * 0.3) / (0.7 + 0.3) = 3.4
        assert abs(result["c1"] - 3.4) < 0.01

    def test_mean_with_weights(self) -> None:
        scores = {"accuracy": {"c1": [3.0, 3.0]}}
        weights = {"accuracy": 1.0}
        result = _compute_weighted_overall("mean", weights, scores, ["c1"])
        assert result["c1"] == 3.0

    def test_no_weights_returns_empty(self) -> None:
        scores = {"accuracy": {"c1": [3.0]}}
        result = _compute_weighted_overall("mean", {}, scores, ["c1"])
        assert result == {}


class TestMajorityVotePairwise:
    def test_majority_winner(self) -> None:
        judgements = [
            _make_pairwise_judgement("tc1", "A", "B", "A", judge_id="j1"),
            _make_pairwise_judgement("tc1", "A", "B", "A", judge_id="j1"),
            _make_pairwise_judgement("tc1", "A", "B", "B", judge_id="j1"),
        ]
        win_rate, loss_rate = _aggregate_majority_vote_pairwise(judgements, ["A", "B"])
        # A wins the majority for tc1:A|B:j1 group → A gets win
        assert win_rate["A"] > 0
        assert loss_rate["B"] > 0

    def test_tie_on_even_split(self) -> None:
        judgements = [
            _make_pairwise_judgement("tc1", "A", "B", "A", judge_id="j1"),
            _make_pairwise_judgement("tc1", "A", "B", "B", judge_id="j1"),
        ]
        win_rate, loss_rate = _aggregate_majority_vote_pairwise(judgements, ["A", "B"])
        # Even split → tie, so neither A nor B wins
        assert win_rate["A"] == 0.0
        assert win_rate["B"] == 0.0


class TestReduceAbsoluteScoresByMajority:
    def test_single_group(self) -> None:
        judgements = [
            _make_absolute_judgement("tc1", "c1", {"accuracy": 3}),
            _make_absolute_judgement("tc1", "c1", {"accuracy": 3}),
            _make_absolute_judgement("tc1", "c1", {"accuracy": 5}),
        ]
        reduced = _reduce_absolute_scores_by_majority(judgements)
        # Single group (tc1, c1, j1, accuracy) → mode = 3
        assert reduced["accuracy"]["c1"] == [3.0]

    def test_multi_testcase(self) -> None:
        judgements = [
            # tc1: repeats [5, 5, 1] → mode = 5
            _make_absolute_judgement("tc1", "c1", {"accuracy": 5}),
            _make_absolute_judgement("tc1", "c1", {"accuracy": 5}),
            _make_absolute_judgement("tc1", "c1", {"accuracy": 1}),
            # tc2: repeats [1, 1, 5] → mode = 1
            _make_absolute_judgement("tc2", "c1", {"accuracy": 1}),
            _make_absolute_judgement("tc2", "c1", {"accuracy": 1}),
            _make_absolute_judgement("tc2", "c1", {"accuracy": 5}),
        ]
        reduced = _reduce_absolute_scores_by_majority(judgements)
        assert sorted(reduced["accuracy"]["c1"]) == [1.0, 5.0]

    def test_ignores_pairwise(self) -> None:
        judgements = [
            _make_pairwise_judgement("tc1", "A", "B", "A", per_metric={"accuracy": 3}),
        ]
        reduced = _reduce_absolute_scores_by_majority(judgements)
        assert len(reduced) == 0


class TestComputeAggregateIntegration:
    def test_mean_method(self) -> None:
        judgements = [
            _make_absolute_judgement("tc1", "c1", {"accuracy": 3}),
            _make_absolute_judgement("tc1", "c1", {"accuracy": 5}),
        ]
        result = _compute_aggregate(judgements, ["c1"], [], method="mean")
        assert result.mean_score["accuracy"]["c1"] == 4.0

    def test_worst_case_method(self) -> None:
        judgements = [
            _make_absolute_judgement("tc1", "c1", {"accuracy": 3}),
            _make_absolute_judgement("tc1", "c1", {"accuracy": 5}),
        ]
        result = _compute_aggregate(judgements, ["c1"], [], method="worst_case")
        assert result.mean_score["accuracy"]["c1"] == 3.0

    def test_majority_vote_absolute(self) -> None:
        """Single testcase: 3 repeats [3, 3, 5] → mode=3, mean_score=3.0."""
        judgements = [
            _make_absolute_judgement("tc1", "c1", {"accuracy": 3}),
            _make_absolute_judgement("tc1", "c1", {"accuracy": 3}),
            _make_absolute_judgement("tc1", "c1", {"accuracy": 5}),
        ]
        result = _compute_aggregate(judgements, ["c1"], [], method="majority_vote")
        assert result.mean_score["accuracy"]["c1"] == 3.0

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            _compute_aggregate([], ["c1"], [], method="nonexistent")


class TestMajorityVoteAbsoluteMultiTestcase:
    def test_multi_testcase_grouping(self) -> None:
        """tc1:[5,5,1]→mode=5, tc2:[1,1,5]→mode=1, mean([5,1])=3.0"""
        judgements = [
            _make_absolute_judgement("tc1", "c1", {"accuracy": 5}),
            _make_absolute_judgement("tc1", "c1", {"accuracy": 5}),
            _make_absolute_judgement("tc1", "c1", {"accuracy": 1}),
            _make_absolute_judgement("tc2", "c1", {"accuracy": 1}),
            _make_absolute_judgement("tc2", "c1", {"accuracy": 1}),
            _make_absolute_judgement("tc2", "c1", {"accuracy": 5}),
        ]
        result = _compute_aggregate(judgements, ["c1"], [], method="majority_vote")
        assert result.mean_score["accuracy"]["c1"] == 3.0

    def test_weighted_overall_uses_reduced_scores(self) -> None:
        """weighted_overall must use reduced (post-majority) data."""
        judgements = [
            # tc1: accuracy [5,5,1]→5, fluency [3,3,1]→3
            _make_absolute_judgement("tc1", "c1", {"accuracy": 5, "fluency": 3}),
            _make_absolute_judgement("tc1", "c1", {"accuracy": 5, "fluency": 3}),
            _make_absolute_judgement("tc1", "c1", {"accuracy": 1, "fluency": 1}),
            # tc2: accuracy [1,1,5]→1, fluency [1,1,3]→1
            _make_absolute_judgement("tc2", "c1", {"accuracy": 1, "fluency": 1}),
            _make_absolute_judgement("tc2", "c1", {"accuracy": 1, "fluency": 1}),
            _make_absolute_judgement("tc2", "c1", {"accuracy": 5, "fluency": 3}),
        ]
        weights = {"accuracy": 0.6, "fluency": 0.4}
        result = _compute_aggregate(
            judgements, ["c1"], [], method="majority_vote", weights=weights,
        )
        # reduced accuracy: [5, 1] → mean = 3.0
        # reduced fluency:  [3, 1] → mean = 2.0
        # weighted = (3.0 * 0.6 + 2.0 * 0.4) / (0.6 + 0.4) = 2.6
        assert abs(result.weighted_overall["c1"] - 2.6) < 0.01

    def test_confidence_interval_uses_reduced_scores(self) -> None:
        """n in CI must be count of reduced groups, not raw repeat count."""
        judgements = [
            # tc1: [3,3,5] → mode=3
            _make_absolute_judgement("tc1", "c1", {"accuracy": 3}),
            _make_absolute_judgement("tc1", "c1", {"accuracy": 3}),
            _make_absolute_judgement("tc1", "c1", {"accuracy": 5}),
            # tc2: [5,5,3] → mode=5
            _make_absolute_judgement("tc2", "c1", {"accuracy": 5}),
            _make_absolute_judgement("tc2", "c1", {"accuracy": 5}),
            _make_absolute_judgement("tc2", "c1", {"accuracy": 3}),
        ]
        result = _compute_aggregate(judgements, ["c1"], [], method="majority_vote")
        ci = result.confidence_intervals["accuracy"]["c1"]
        # n should be 2 (two testcase groups), not 6 (raw repeats)
        assert ci["n"] == 2
        # mean of reduced [3, 5] = 4.0
        assert abs(ci["mean"] - 4.0) < 0.01
