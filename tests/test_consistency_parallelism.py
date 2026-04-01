"""Tests for Phase 6: Consistency stage parallelization.

Covers:
- consistency_max_concurrency=1, =2, =3 produce the same record count
- consistency_max_concurrency=1 and =2 produce the same output order (group dict order × judge order)
- Partial failure: success + failure record counts and order preserved
- All tasks fail: failure record list is non-empty
- Groups with < 2 inference records are excluded (existing precondition)
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_judge.llm_client import CompletionResult
from llm_judge.models import (
    ConsistencyRecord,
    InferenceRecord,
    ModelInfo,
    OutputInfo,
    RunConfig,
    StatusInfo,
    Testcase,
)
from llm_judge.stages.consistency import (
    ConsistencyTaskPayload,
    MAX_CONSISTENCY_RETRY,
    run_consistency,
)


# ── Helpers ───────────────────────────────────────────────


def _make_testcase(tc_id: str = "tc-001") -> Testcase:
    return Testcase(
        testcase_id=tc_id,
        task_type="report_generation",
        input={"raw_text": "test input"},
    )


def _make_inference(
    tc_id: str = "tc-001",
    candidate_id: str = "c1",
    text: str = "output text",
    ok: bool = True,
) -> InferenceRecord:
    return InferenceRecord(
        run_id="test-run",
        testcase_id=tc_id,
        candidate_id=candidate_id,
        model=ModelInfo(vendor="openai", model_id="gpt-4o"),
        output=OutputInfo(text=text),
        status=StatusInfo(ok=ok),
    )


def _make_config(consistency_max_concurrency: int = 1) -> RunConfig:
    return RunConfig.model_validate({
        "run_id": "test-run",
        "dataset": {"testcases_path": "data/testcases.jsonl"},
        "candidates": [
            {"candidate_id": "c1", "vendor": "openai", "model_id": "gpt-4o"},
            {"candidate_id": "c2", "vendor": "openai", "model_id": "gpt-4o"},
        ],
        "judges": [
            {
                "judge_id": "j1",
                "vendor": "openai",
                "model_id": "gpt-4o",
                "rubric_version": "v1",
            },
            {
                "judge_id": "j2",
                "vendor": "openai",
                "model_id": "gpt-4o",
                "rubric_version": "v1",
            },
        ],
        "protocol": {
            "evaluation_mode": "absolute",
            "aggregation": {"method": "mean"},
            "parallelism": {
                "consistency_max_concurrency": consistency_max_concurrency,
            },
        },
    })


_GOOD_CONSISTENCY_RESPONSE = json.dumps({
    "overall": 4.5,
    "rationale": "Consistent answers",
})

_LIST_CONSISTENCY_RESPONSE = json.dumps([
    {"overall": 3.0, "rationale": "List response"},
])

_EMPTY_LIST_RESPONSE = json.dumps([])

_FENCED_CONSISTENCY_RESPONSE = (
    '```json\n{"overall": 4.5, "rationale": "Consistent answers"}\n```'
)


def _run_with_mocks(
    inferences: list[InferenceRecord],
    testcases: dict[str, Testcase],
    cfg: RunConfig,
    output_path: str,
    completion_side_effect=None,
) -> list[ConsistencyRecord]:
    """Run run_consistency with all I/O mocked; return records passed to write_jsonl."""
    captured: list[ConsistencyRecord] = []

    if completion_side_effect is None:
        mock_completion = MagicMock(
            return_value=CompletionResult(text=_GOOD_CONSISTENCY_RESPONSE)
        )
    else:
        mock_completion = MagicMock(side_effect=completion_side_effect)

    def fake_write_jsonl(path, records):
        captured.extend(records)

    with ExitStack() as stack:
        stack.enter_context(
            patch("llm_judge.stages.consistency.load_run_config", return_value=cfg)
        )
        stack.enter_context(
            patch(
                "llm_judge.stages.consistency.read_jsonl",
                return_value=[inf.model_dump() for inf in inferences],
            )
        )
        stack.enter_context(
            patch(
                "llm_judge.stages.consistency.load_testcase_map",
                return_value=testcases,
            )
        )
        stack.enter_context(
            patch(
                "llm_judge.stages.consistency.build_consistency_judge_prompt",
                return_value=[{"role": "user", "content": "judge consistency"}],
            )
        )
        stack.enter_context(
            patch(
                "llm_judge.stages.consistency.get_thread_local_openai_client",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "llm_judge.stages.consistency.judge_chat_completion",
                mock_completion,
            )
        )
        stack.enter_context(
            patch(
                "llm_judge.stages.consistency.write_jsonl",
                side_effect=fake_write_jsonl,
            )
        )

        run_consistency(
            "config.yaml",
            inference_path="/tmp/inf.jsonl",
            output_path=output_path,
        )

    return captured


# ── Test: record count matches across concurrency levels ─


class TestRecordCount:
    def test_count_matches_for_concurrency_1_2_3(self, tmp_path):
        """Record count is identical for consistency_max_concurrency=1, 2, 3."""
        inferences = [
            _make_inference("tc-001", "c1", f"output-{i}")
            for i in range(3)
        ] + [
            _make_inference("tc-001", "c2", f"output-{i}")
            for i in range(3)
        ] + [
            _make_inference("tc-002", "c1", f"output-{i}")
            for i in range(2)
        ]
        testcases = {
            "tc-001": _make_testcase("tc-001"),
            "tc-002": _make_testcase("tc-002"),
        }

        # 3 eligible groups: (tc-001,c1), (tc-001,c2), (tc-002,c1)
        # × 2 judges = 6 records
        expected_count = 3 * 2

        for concurrency in (1, 2, 3):
            cfg = _make_config(consistency_max_concurrency=concurrency)
            records = _run_with_mocks(
                inferences, testcases, cfg, str(tmp_path / f"out-{concurrency}.jsonl")
            )
            assert len(records) == expected_count, (
                f"concurrency={concurrency}: expected {expected_count}, got {len(records)}"
            )


# ── Test: output order matches between concurrency=1 and =2 ─


class TestOutputOrder:
    def test_order_matches_between_concurrency_1_and_2(self, tmp_path):
        """Output order (sorted group × judge) is identical for concurrency=1 and =2."""
        inferences = [
            _make_inference("tc-002", "c1", f"out-{i}") for i in range(2)
        ] + [
            _make_inference("tc-001", "c2", f"out-{i}") for i in range(2)
        ] + [
            _make_inference("tc-001", "c1", f"out-{i}") for i in range(2)
        ]
        testcases = {
            "tc-001": _make_testcase("tc-001"),
            "tc-002": _make_testcase("tc-002"),
        }

        cfg_1 = _make_config(consistency_max_concurrency=1)
        cfg_2 = _make_config(consistency_max_concurrency=2)

        records_1 = _run_with_mocks(
            inferences, testcases, cfg_1, str(tmp_path / "seq.jsonl")
        )
        records_2 = _run_with_mocks(
            inferences, testcases, cfg_2, str(tmp_path / "par.jsonl")
        )

        assert len(records_1) == len(records_2)
        for r1, r2 in zip(records_1, records_2):
            assert r1.testcase_id == r2.testcase_id
            assert r1.candidate_id == r2.candidate_id
            assert r1.judge_id == r2.judge_id

    def test_order_is_sorted_group_then_judge(self, tmp_path):
        """Records are ordered by sorted (testcase_id, candidate_id) then judge."""
        # Provide inferences in non-sorted order to verify sorting
        inferences = [
            _make_inference("tc-002", "c1", f"out-{i}") for i in range(2)
        ] + [
            _make_inference("tc-001", "c1", f"out-{i}") for i in range(2)
        ]
        testcases = {
            "tc-001": _make_testcase("tc-001"),
            "tc-002": _make_testcase("tc-002"),
        }

        cfg = _make_config(consistency_max_concurrency=1)
        records = _run_with_mocks(
            inferences, testcases, cfg, str(tmp_path / "out.jsonl")
        )

        # 2 groups × 2 judges = 4 records
        assert len(records) == 4
        # Expected order: (tc-001,c1,j1), (tc-001,c1,j2), (tc-002,c1,j1), (tc-002,c1,j2)
        assert records[0].testcase_id == "tc-001"
        assert records[0].judge_id == "j1"
        assert records[1].testcase_id == "tc-001"
        assert records[1].judge_id == "j2"
        assert records[2].testcase_id == "tc-002"
        assert records[2].judge_id == "j1"
        assert records[3].testcase_id == "tc-002"
        assert records[3].judge_id == "j2"


# ── Test: partial failure ─────────────────────────────────


class TestPartialFailure:
    def test_success_and_failure_counts_and_order(self, tmp_path):
        """When some judge calls fail, both success and failure records are
        returned in planner order."""
        inferences = [
            _make_inference("tc-001", "c1", f"out-{i}") for i in range(2)
        ]
        testcases = {"tc-001": _make_testcase("tc-001")}

        call_counter = 0

        def side_effect(*args, **kwargs):
            nonlocal call_counter
            idx = call_counter
            call_counter += 1
            if idx == 0:
                return CompletionResult(text=_GOOD_CONSISTENCY_RESPONSE)
            raise RuntimeError("API down")

        cfg = _make_config(consistency_max_concurrency=1)
        records = _run_with_mocks(
            inferences, testcases, cfg, str(tmp_path / "out.jsonl"),
            completion_side_effect=side_effect,
        )

        # 1 group × 2 judges = 2 records
        assert len(records) == 2
        ok_records = [r for r in records if r.status.ok]
        fail_records = [r for r in records if not r.status.ok]
        assert len(ok_records) == 1
        assert len(fail_records) == 1

        # Order: j1 (success), j2 (failure)
        assert records[0].judge_id == "j1"
        assert records[0].status.ok is True
        assert records[1].judge_id == "j2"
        assert records[1].status.ok is False
        assert records[1].status.error_type == "RuntimeError"


# ── Test: all tasks fail ──────────────────────────────────


class TestAllFail:
    def test_all_failure_returns_nonempty_list(self, tmp_path):
        """When all judge calls fail, a non-empty failure record list is returned."""
        inferences = [
            _make_inference("tc-001", "c1", f"out-{i}") for i in range(2)
        ] + [
            _make_inference("tc-001", "c2", f"out-{i}") for i in range(2)
        ]
        testcases = {"tc-001": _make_testcase("tc-001")}

        cfg = _make_config(consistency_max_concurrency=2)
        records = _run_with_mocks(
            inferences, testcases, cfg, str(tmp_path / "out.jsonl"),
            completion_side_effect=RuntimeError("total failure"),
        )

        # 2 groups × 2 judges = 4 records
        assert len(records) == 4
        assert all(not r.status.ok for r in records)
        assert all(r.status.error_type == "RuntimeError" for r in records)


# ── Test: groups with < 2 repeats excluded ────────────────


class TestSingleRepeatExcluded:
    def test_single_repeat_group_excluded(self, tmp_path):
        """Groups with only 1 inference record are excluded from consistency evaluation."""
        inferences = [
            # tc-001/c1: 2 repeats → eligible
            _make_inference("tc-001", "c1", "out-0"),
            _make_inference("tc-001", "c1", "out-1"),
            # tc-001/c2: 1 repeat → excluded
            _make_inference("tc-001", "c2", "single"),
        ]
        testcases = {"tc-001": _make_testcase("tc-001")}

        cfg = _make_config(consistency_max_concurrency=1)
        records = _run_with_mocks(
            inferences, testcases, cfg, str(tmp_path / "out.jsonl")
        )

        # Only (tc-001, c1) is eligible → 1 group × 2 judges = 2 records
        assert len(records) == 2
        assert all(r.candidate_id == "c1" for r in records)

    def test_failed_inference_not_counted(self, tmp_path):
        """Failed inferences are not counted toward the 2-repeat threshold."""
        inferences = [
            _make_inference("tc-001", "c1", "out-0", ok=True),
            _make_inference("tc-001", "c1", "", ok=False),
        ]
        testcases = {"tc-001": _make_testcase("tc-001")}

        cfg = _make_config(consistency_max_concurrency=1)
        records = _run_with_mocks(
            inferences, testcases, cfg, str(tmp_path / "out.jsonl")
        )

        # Only 1 ok inference → not eligible → 0 records
        assert len(records) == 0

    def test_no_eligible_groups_warns(self, tmp_path):
        """When no groups meet the >= 2 threshold, a RuntimeWarning is issued."""
        inferences = [
            _make_inference("tc-001", "c1", "single"),
        ]
        testcases = {"tc-001": _make_testcase("tc-001")}

        cfg = _make_config(consistency_max_concurrency=1)

        with pytest.warns(RuntimeWarning, match="inference_repeats >= 2"):
            _run_with_mocks(
                inferences, testcases, cfg, str(tmp_path / "out.jsonl")
            )


# ── Test: list response handling (方針 A) ─────────────────


class TestListResponseHandling:
    def test_list_response_uses_first_element(self, tmp_path):
        """When judge returns a JSON list, first element is used and score is correct."""
        inferences = [
            _make_inference("tc-001", "c1", f"out-{i}") for i in range(2)
        ]
        testcases = {"tc-001": _make_testcase("tc-001")}

        cfg = _make_config(consistency_max_concurrency=1)
        records = _run_with_mocks(
            inferences, testcases, cfg, str(tmp_path / "out.jsonl"),
            completion_side_effect=lambda *a, **kw: CompletionResult(
                text=_LIST_CONSISTENCY_RESPONSE,
            ),
        )

        assert len(records) == 2
        for r in records:
            assert r.status.ok is True
            assert r.scores.overall == 3.0
            assert r.scores.rationale == "List response"

    def test_empty_list_response_returns_failure(self, tmp_path):
        """When judge returns an empty JSON list on all retries, status is ok=False."""
        inferences = [
            _make_inference("tc-001", "c1", f"out-{i}") for i in range(2)
        ]
        testcases = {"tc-001": _make_testcase("tc-001")}

        cfg = _make_config(consistency_max_concurrency=1)
        records = _run_with_mocks(
            inferences, testcases, cfg, str(tmp_path / "out.jsonl"),
            completion_side_effect=lambda *a, **kw: CompletionResult(
                text=_EMPTY_LIST_RESPONSE,
            ),
        )

        assert len(records) == 2
        for r in records:
            assert r.status.ok is False
            assert r.status.error_type == "ValueError"


# ── Test: retry mechanism (方針 B) ────────────────────────


class TestRetryMechanism:
    def test_retry_recovers_after_json_parse_failure(self, tmp_path):
        """Retry succeeds when first attempt returns invalid JSON but second succeeds."""
        inferences = [
            _make_inference("tc-001", "c1", f"out-{i}") for i in range(2)
        ]
        testcases = {"tc-001": _make_testcase("tc-001")}

        call_counter = 0

        def side_effect(*args, **kwargs):
            nonlocal call_counter
            call_counter += 1
            if call_counter % 2 == 1:
                return CompletionResult(text="not valid json {{{")
            return CompletionResult(text=_GOOD_CONSISTENCY_RESPONSE)

        cfg = _make_config(consistency_max_concurrency=1)
        records = _run_with_mocks(
            inferences, testcases, cfg, str(tmp_path / "out.jsonl"),
            completion_side_effect=side_effect,
        )

        assert len(records) == 2
        for r in records:
            assert r.status.ok is True
            assert r.scores.overall == 4.5

    def test_max_retries_exceeded_returns_failure(self, tmp_path):
        """When all retries return invalid JSON, status is ok=False."""
        inferences = [
            _make_inference("tc-001", "c1", f"out-{i}") for i in range(2)
        ]
        testcases = {"tc-001": _make_testcase("tc-001")}

        cfg = _make_config(consistency_max_concurrency=1)
        records = _run_with_mocks(
            inferences, testcases, cfg, str(tmp_path / "out.jsonl"),
            completion_side_effect=lambda *a, **kw: CompletionResult(
                text="not json",
            ),
        )

        assert len(records) == 2
        for r in records:
            assert r.status.ok is False
            assert r.status.error_type == "ValueError"

    def test_retry_recovers_after_float_conversion_failure(self, tmp_path):
        """Retry succeeds when first attempt has non-numeric overall but second succeeds."""
        inferences = [
            _make_inference("tc-001", "c1", f"out-{i}") for i in range(2)
        ]
        testcases = {"tc-001": _make_testcase("tc-001")}

        call_counter = 0

        def side_effect(*args, **kwargs):
            nonlocal call_counter
            call_counter += 1
            if call_counter % 2 == 1:
                return CompletionResult(
                    text=json.dumps({"overall": "not_a_number", "rationale": "bad"}),
                )
            return CompletionResult(text=_GOOD_CONSISTENCY_RESPONSE)

        cfg = _make_config(consistency_max_concurrency=1)
        records = _run_with_mocks(
            inferences, testcases, cfg, str(tmp_path / "out.jsonl"),
            completion_side_effect=side_effect,
        )

        assert len(records) == 2
        for r in records:
            assert r.status.ok is True
            assert r.scores.overall == 4.5

    def test_float_conversion_failure_exhausts_retries(self, tmp_path):
        """When all retries return non-numeric overall, status is ok=False."""
        inferences = [
            _make_inference("tc-001", "c1", f"out-{i}") for i in range(2)
        ]
        testcases = {"tc-001": _make_testcase("tc-001")}

        cfg = _make_config(consistency_max_concurrency=1)
        records = _run_with_mocks(
            inferences, testcases, cfg, str(tmp_path / "out.jsonl"),
            completion_side_effect=lambda *a, **kw: CompletionResult(
                text=json.dumps({"overall": "NaN_string", "rationale": "bad"}),
            ),
        )

        assert len(records) == 2
        for r in records:
            assert r.status.ok is False
            assert r.status.error_type == "ValueError"

    def test_fenced_json_parsed_without_retry(self, tmp_path):
        """When judge wraps response in ```json fences, it is parsed without needing a retry."""
        inferences = [
            _make_inference("tc-001", "c1", f"out-{i}") for i in range(2)
        ]
        testcases = {"tc-001": _make_testcase("tc-001")}

        call_counts = 0

        def side_effect(*args, **kwargs):
            nonlocal call_counts
            call_counts += 1
            return CompletionResult(text=_FENCED_CONSISTENCY_RESPONSE)

        cfg = _make_config(consistency_max_concurrency=1)
        records = _run_with_mocks(
            inferences, testcases, cfg, str(tmp_path / "out.jsonl"),
            completion_side_effect=side_effect,
        )

        assert len(records) == 2
        for r in records:
            assert r.status.ok is True
            assert r.scores.overall == 4.5
        # Each record should need exactly 1 API call (no retry)
        assert call_counts == 2
