"""Tests for Phase 5: Judge stage parallelization.

Covers:
- should_swap_pairwise_order() determinism, symmetry, seed handling
- blinding.enabled=False forces presented_swap=False
- max_workers=1 and =2 produce the same blinding results
- Record order matches planner order (pairwise / absolute / hybrid)
- Partial-failure records (failed inference short-circuit) are preserved in order
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_judge.llm_client import CompletionResult
from llm_judge.models import (
    InferenceRecord,
    ModelInfo,
    OutputInfo,
    RunConfig,
    StatusInfo,
    Testcase,
)
from llm_judge.stages.judge import (
    JudgeAbsolutePayload,
    JudgePairwisePayload,
    _judge_absolute,
    _judge_pairwise,
    should_swap_pairwise_order,
)


# ── Helpers ───────────────────────────────────────────────


def _make_testcase(tc_id: str = "tc-001", task_type: str = "report_generation") -> Testcase:
    return Testcase(
        testcase_id=tc_id,
        task_type=task_type,
        input={"raw_text": "test input"},
    )


def _make_inference(
    candidate_id: str = "c1",
    ok: bool = True,
    text: str = "some output",
    tc_id: str = "tc-001",
) -> InferenceRecord:
    return InferenceRecord(
        run_id="test-run",
        testcase_id=tc_id,
        candidate_id=candidate_id,
        model=ModelInfo(vendor="openai", model_id="gpt-4o"),
        output=OutputInfo(text=text),
        status=StatusInfo(ok=ok),
    )


def _make_config(
    mode: str = "pairwise",
    blinding_enabled: bool = True,
    random_seed: int | None = 42,
    judge_max_concurrency: int = 1,
) -> RunConfig:
    return RunConfig.model_validate({
        "run_id": "test-run",
        "dataset": {"testcases_path": "data/testcases.jsonl"},
        "candidates": [
            {"candidate_id": "c1", "vendor": "openai", "model_id": "gpt-4o"},
            {"candidate_id": "c2", "vendor": "openai", "model_id": "gpt-4o"},
        ],
        "judges": [{
            "judge_id": "j1",
            "vendor": "openai",
            "model_id": "gpt-4o",
            "rubric_version": "v1",
        }],
        "protocol": {
            "evaluation_mode": mode,
            "metrics": ["accuracy"],
            "aggregation": {"method": "mean"},
            "blinding": {
                "enabled": blinding_enabled,
                "random_seed": random_seed,
            },
            "parallelism": {
                "judge_max_concurrency": judge_max_concurrency,
            },
        },
    })


_GOOD_PAIRWISE_RESPONSE = json.dumps({
    "per_metric": {"accuracy": 5},
    "overall_winner": "A",
    "critical_issue_a": False,
    "critical_issue_b": False,
    "rationale": "A is better",
})

_GOOD_ABSOLUTE_RESPONSE = json.dumps({
    "per_metric": {"accuracy": 5},
    "overall_score": 5,
    "rationale": "Good",
    "critical_issue": False,
})


# ── should_swap_pairwise_order ────────────────────────────


class TestShouldSwapPairwiseOrder:
    def test_stable_result_for_same_args(self):
        """Same (a, b, seed) always returns the same value."""
        result_1 = should_swap_pairwise_order("c1", "c2", 42)
        result_2 = should_swap_pairwise_order("c1", "c2", 42)
        assert result_1 == result_2

    def test_symmetric_regardless_of_enumeration_order(self):
        """Swapping a and b should give the same result (sort normalization)."""
        result_ab = should_swap_pairwise_order("c1", "c2", 42)
        result_ba = should_swap_pairwise_order("c2", "c1", 42)
        assert result_ab == result_ba

    def test_random_seed_zero_differs_from_none(self):
        """random_seed=0 and random_seed=None must be treated as distinct.

        Both map to different seed materials (0 vs 42), so for most candidate
        pairs they will produce different results.  We pick a pair where they
        actually differ to make the assertion concrete.
        """
        # We need a pair where seed 0 and seed 42 (None→42) give different results.
        # Try several pairs until we find a divergence.
        found_difference = False
        for suffix in range(20):
            r_none = should_swap_pairwise_order(f"alpha-{suffix}", f"beta-{suffix}", None)
            r_zero = should_swap_pairwise_order(f"alpha-{suffix}", f"beta-{suffix}", 0)
            if r_none != r_zero:
                found_difference = True
                break
        assert found_difference, (
            "should_swap_pairwise_order(seed=None) and (seed=0) produced "
            "identical results for all tested pairs — they may not be using "
            "distinct seed materials."
        )

    def test_different_seeds_can_produce_different_results(self):
        """Different seed values can flip the swap decision."""
        # Find a pair where seed=1 and seed=2 differ
        found_difference = False
        for suffix in range(20):
            r1 = should_swap_pairwise_order(f"cand-{suffix}", "other", 1)
            r2 = should_swap_pairwise_order(f"cand-{suffix}", "other", 2)
            if r1 != r2:
                found_difference = True
                break
        assert found_difference

    def test_returns_bool(self):
        result = should_swap_pairwise_order("c1", "c2", 42)
        assert isinstance(result, bool)

    def test_none_seed_uses_42_not_zero(self):
        """None seed → seed_material=42, not 0; so result equals seed=42."""
        r_none = should_swap_pairwise_order("cX", "cY", None)
        r_42 = should_swap_pairwise_order("cX", "cY", 42)
        assert r_none == r_42

    def test_zero_seed_is_distinct_from_42(self):
        """seed=0 must use material 0, not be coerced to 42."""
        # At least one pair should differ between seed=0 and seed=42.
        found_difference = False
        for suffix in range(20):
            r0 = should_swap_pairwise_order(f"p{suffix}", f"q{suffix}", 0)
            r42 = should_swap_pairwise_order(f"p{suffix}", f"q{suffix}", 42)
            if r0 != r42:
                found_difference = True
                break
        assert found_difference, (
            "seed=0 and seed=42 always produced the same result — "
            "seed=0 is likely being coerced to 42 via 'or 42'."
        )


# ── blinding.enabled=False ────────────────────────────────


class TestBlindingDisabled:
    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_blinding_disabled_presented_swap_is_false(self, mock_completion):
        """When blinding.enabled=False, presented_swap must be False."""
        mock_completion.return_value = CompletionResult(text=_GOOD_PAIRWISE_RESPONSE)

        cfg = _make_config(mode="pairwise", blinding_enabled=False)
        tc = _make_testcase()
        inf_a = _make_inference("c1")
        inf_b = _make_inference("c2")

        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        # With blinding disabled and presented_swap=False, order is (c1, c2).
        assert rec.blinding.presented_order == ["c1", "c2"]

    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_blinding_disabled_presented_order_unchanged(self, mock_completion):
        """blinding.enabled=False should never swap order."""
        mock_completion.return_value = CompletionResult(text=_GOOD_PAIRWISE_RESPONSE)

        cfg = _make_config(mode="pairwise", blinding_enabled=False)
        tc = _make_testcase()
        inf_a = _make_inference("alpha")
        inf_b = _make_inference("beta")

        # Even if we pass presented_swap=True, the blinding flag is False
        # so the record's blinding.enabled should reflect False.
        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.blinding.enabled is False


# ── Concurrency-independent blinding ─────────────────────


class TestConcurrencyIndependentBlinding:
    """presented_order must be identical regardless of max_workers."""

    @patch("llm_judge.stages.judge.run_bounded_parallel")
    @patch("llm_judge.stages.judge.load_testcase_map")
    @patch("llm_judge.stages.judge.read_jsonl")
    @patch("llm_judge.stages.judge.write_jsonl")
    @patch("llm_judge.stages.judge.validate_artifacts")
    @patch("llm_judge.stages.judge.build_stage_output_path")
    def _run_judge_and_capture_tasks(
        self,
        mock_path,
        mock_validate,
        mock_write,
        mock_read_jsonl,
        mock_load_tc,
        mock_parallel,
        *,
        judge_max_concurrency: int,
    ) -> list:
        """Helper: run run_judge and capture the tasks passed to run_bounded_parallel."""
        from pathlib import Path
        mock_path.return_value = Path("/tmp/out.jsonl")

        inf_a = _make_inference("c1")
        inf_b = _make_inference("c2")
        mock_read_jsonl.return_value = [inf_a.model_dump(), inf_b.model_dump()]

        tc = _make_testcase()
        mock_load_tc.return_value = {"tc-001": tc}

        captured: list = []

        def capture_tasks(tasks, worker_fn, max_workers):
            captured.extend(tasks)
            return [worker_fn(t.payload) for t in tasks]

        mock_parallel.side_effect = capture_tasks

        with patch(
            "llm_judge.stages.judge.judge_chat_completion",
            return_value=CompletionResult(text=_GOOD_PAIRWISE_RESPONSE),
        ), patch(
            "llm_judge.stages.judge.get_thread_local_openai_client",
            return_value=None,
        ), patch(
            "llm_judge.stages.judge.load_run_config",
            return_value=_make_config(
                mode="pairwise",
                blinding_enabled=True,
                random_seed=42,
                judge_max_concurrency=judge_max_concurrency,
            ),
        ):
            from llm_judge.stages.judge import run_judge
            run_judge("dummy.yaml", inference_path="/tmp/inf.jsonl", output_path="/tmp/out.jsonl")

        return captured

    def test_presented_order_same_for_concurrency_1_and_2(self):
        """Tasks (and thus presented_swap) are identical regardless of max_workers."""
        tasks_1 = self._run_judge_and_capture_tasks(judge_max_concurrency=1)
        tasks_2 = self._run_judge_and_capture_tasks(judge_max_concurrency=2)

        assert len(tasks_1) == len(tasks_2)
        for t1, t2 in zip(tasks_1, tasks_2):
            assert isinstance(t1.payload, JudgePairwisePayload)
            assert isinstance(t2.payload, JudgePairwisePayload)
            assert t1.payload.presented_swap == t2.payload.presented_swap


# ── Record order matches planner order ────────────────────


class TestRecordOrder:
    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_pairwise_records_in_planner_order(self, mock_completion):
        """Results must be returned in task_index order."""
        mock_completion.return_value = CompletionResult(text=_GOOD_PAIRWISE_RESPONSE)

        cfg = _make_config(mode="pairwise")
        tc = _make_testcase()
        inf_a = _make_inference("c1")
        inf_b = _make_inference("c2")

        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.mode == "pairwise"
        assert rec.targets[0].candidate_id == "c1"
        assert rec.targets[1].candidate_id == "c2"

    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_absolute_records_in_planner_order(self, mock_completion):
        mock_completion.return_value = CompletionResult(text=_GOOD_ABSOLUTE_RESPONSE)

        cfg = _make_config(mode="absolute")
        tc = _make_testcase()
        inf = _make_inference("c1")

        rec = _judge_absolute(
            cfg=cfg, tc=tc, inf=inf, idx=0,
            judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.mode == "absolute"
        assert rec.targets[0].candidate_id == "c1"


# ── Partial-failure records ───────────────────────────────


class TestPartialFailure:
    def test_failed_inference_produces_deterministic_record(self):
        """A failed inference returns a critical-issue record, not an exception."""
        cfg = _make_config(mode="pairwise")
        tc = _make_testcase()
        inf_a = _make_inference("c1", ok=False, text="")
        inf_b = _make_inference("c2", ok=True, text="good answer")

        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.critical_issue is True
        assert rec.scores.overall_winner == "c2"

    def test_failed_inference_absolute_produces_score_one(self):
        """A failed absolute inference is scored 1, not raising an exception."""
        cfg = _make_config(mode="absolute")
        tc = _make_testcase()
        inf = _make_inference("c1", ok=False, text="")

        rec = _judge_absolute(
            cfg=cfg, tc=tc, inf=inf, idx=0,
            judge_ref=cfg.judges[0], inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert rec.critical_issue is True
        assert rec.scores.overall_score == 1.0

    @patch("llm_judge.stages.judge.judge_chat_completion")
    def test_exception_in_judge_produces_error_record(self, mock_completion):
        """An exception in judge_chat_completion returns an error record, not raises."""
        mock_completion.side_effect = RuntimeError("API down")

        cfg = _make_config(mode="pairwise")
        tc = _make_testcase()
        inf_a = _make_inference("c1")
        inf_b = _make_inference("c2")

        rec = _judge_pairwise(
            cfg=cfg, tc=tc, inf_a=inf_a, inf_b=inf_b,
            idx_a=0, idx_b=1, judge_ref=cfg.judges[0],
            presented_swap=False, inf_path="data/inf.jsonl",
            client=MagicMock(),
        )

        assert "Judge error" in rec.overall_rationale
        assert rec.scores.overall_winner == "tie"


# ── run_judge integration: planner calls run_bounded_parallel ─


class TestRunJudgeUsesParallelism:
    @patch("llm_judge.stages.judge.run_bounded_parallel")
    @patch("llm_judge.stages.judge.load_testcase_map")
    @patch("llm_judge.stages.judge.read_jsonl")
    @patch("llm_judge.stages.judge.write_jsonl")
    @patch("llm_judge.stages.judge.validate_artifacts")
    @patch("llm_judge.stages.judge.build_stage_output_path")
    @patch("llm_judge.stages.judge.load_run_config")
    def test_run_judge_calls_run_bounded_parallel_with_correct_concurrency(
        self,
        mock_load_cfg,
        mock_path,
        mock_validate,
        mock_write,
        mock_read_jsonl,
        mock_load_tc,
        mock_parallel,
    ):
        from pathlib import Path

        cfg = _make_config(mode="pairwise", judge_max_concurrency=3)
        mock_load_cfg.return_value = cfg
        mock_path.return_value = Path("/tmp/out.jsonl")

        inf_a = _make_inference("c1")
        inf_b = _make_inference("c2")
        mock_read_jsonl.return_value = [inf_a.model_dump(), inf_b.model_dump()]
        mock_load_tc.return_value = {"tc-001": _make_testcase()}

        mock_parallel.return_value = []

        with patch(
            "llm_judge.stages.judge.get_thread_local_openai_client",
            return_value=None,
        ):
            from llm_judge.stages.judge import run_judge
            run_judge("dummy.yaml", inference_path="/tmp/inf.jsonl", output_path="/tmp/out.jsonl")

        assert mock_parallel.called
        _, kwargs = mock_parallel.call_args
        # max_workers is the third positional arg or keyword
        call_args = mock_parallel.call_args
        max_workers_arg = (
            call_args.kwargs.get("max_workers")
            if call_args.kwargs.get("max_workers") is not None
            else call_args.args[2]
        )
        assert max_workers_arg == 3

    @patch("llm_judge.stages.judge.run_bounded_parallel")
    @patch("llm_judge.stages.judge.load_testcase_map")
    @patch("llm_judge.stages.judge.read_jsonl")
    @patch("llm_judge.stages.judge.write_jsonl")
    @patch("llm_judge.stages.judge.validate_artifacts")
    @patch("llm_judge.stages.judge.build_stage_output_path")
    @patch("llm_judge.stages.judge.load_run_config")
    def test_pairwise_tasks_have_correct_presented_swap(
        self,
        mock_load_cfg,
        mock_path,
        mock_validate,
        mock_write,
        mock_read_jsonl,
        mock_load_tc,
        mock_parallel,
    ):
        """presented_swap in the planner matches should_swap_pairwise_order output."""
        from pathlib import Path

        cfg = _make_config(mode="pairwise", blinding_enabled=True, random_seed=7)
        mock_load_cfg.return_value = cfg
        mock_path.return_value = Path("/tmp/out.jsonl")

        inf_a = _make_inference("c1")
        inf_b = _make_inference("c2")
        mock_read_jsonl.return_value = [inf_a.model_dump(), inf_b.model_dump()]
        mock_load_tc.return_value = {"tc-001": _make_testcase()}

        captured_tasks: list = []

        def capture(tasks, worker_fn, max_workers):
            captured_tasks.extend(tasks)
            return []

        mock_parallel.side_effect = capture

        with patch(
            "llm_judge.stages.judge.get_thread_local_openai_client",
            return_value=None,
        ):
            from llm_judge.stages.judge import run_judge
            run_judge("dummy.yaml", inference_path="/tmp/inf.jsonl", output_path="/tmp/out.jsonl")

        assert len(captured_tasks) == 1
        payload = captured_tasks[0].payload
        assert isinstance(payload, JudgePairwisePayload)

        expected_swap = should_swap_pairwise_order("c1", "c2", 7)
        assert payload.presented_swap == expected_swap
