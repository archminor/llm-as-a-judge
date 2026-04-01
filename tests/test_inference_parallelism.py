"""Tests for run_inference() parallelism via run_bounded_parallel."""

from __future__ import annotations

import threading
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_judge.models import (
    Constraints,
    ModelRef,
    OutputFormat,
    RunConfig,
    Testcase,
)
from llm_judge.stages.inference import run_inference


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_testcase(tc_id: str, has_so: bool = False) -> Testcase:
    constraints = (
        Constraints(
            output_format=OutputFormat(
                type="json",
                json_schema_ref="schemas/testcase.schema.json",
            )
        )
        if has_so
        else None
    )
    return Testcase(
        testcase_id=tc_id,
        task_type="report_generation",
        input={"report_name": "test", "source": "dummy"},
        constraints=constraints,
    )


def _make_run_config(
    n_candidates: int = 2,
    n_repeats: int = 1,
    max_concurrency: int = 1,
) -> RunConfig:
    candidates = [
        ModelRef(candidate_id=f"cand-{i}", vendor="openai", model_id="gpt-4o")
        for i in range(n_candidates)
    ]
    return RunConfig(
        run_id="test-run",
        dataset={"testcases_path": "data/testcases.jsonl"},
        candidates=candidates,
        judges=[],
        protocol={
            "evaluation_mode": "absolute" if n_candidates == 1 else "pairwise",
            "aggregation": {"method": "mean"},
            "repeats": {"inference_repeats": n_repeats},
            "parallelism": {"inference_max_concurrency": max_concurrency},
        },
    )


def _make_mock_response(content: str = "response") -> MagicMock:
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock.choices[0].finish_reason = "stop"
    mock.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
    return mock


def _run_with_mocks(
    testcases: list[Testcase],
    cfg: RunConfig,
    output_path: str,
    chat_side_effect=None,
) -> list:
    """Run run_inference with all I/O mocked; return the records passed to write_jsonl."""
    captured: list = []

    if chat_side_effect is None:
        mock_chat = MagicMock(return_value=_make_mock_response())
    else:
        mock_chat = MagicMock(side_effect=chat_side_effect)

    def fake_write_jsonl(path, records):
        captured.extend(records)

    with ExitStack() as stack:
        stack.enter_context(patch("llm_judge.stages.inference.load_dotenv"))
        stack.enter_context(
            patch("llm_judge.stages.inference.load_run_config", return_value=cfg)
        )
        stack.enter_context(
            patch("llm_judge.stages.inference.load_testcases", return_value=testcases)
        )
        stack.enter_context(
            patch(
                "llm_judge.stages.inference.build_inference_prompt",
                return_value=[{"role": "user", "content": "test"}],
            )
        )
        stack.enter_context(
            patch(
                "llm_judge.stages.inference.get_thread_local_openai_client",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch("llm_judge.stages.inference.chat_completion", mock_chat)
        )
        stack.enter_context(patch("llm_judge.stages.inference.validate_artifacts"))
        stack.enter_context(
            patch("llm_judge.stages.inference.write_jsonl", side_effect=fake_write_jsonl)
        )

        run_inference("config.yaml", output_path)

    return captured


# ── test 1 & 2: count and order match for concurrency=1 vs concurrency=2 ─────


class TestCountAndOrder:
    def test_count_matches_between_concurrency_1_and_2(self, tmp_path):
        testcases = [_make_testcase(f"tc-{i}") for i in range(3)]
        n_candidates = 2
        n_repeats = 2
        expected = len(testcases) * n_candidates * n_repeats

        records_seq = _run_with_mocks(
            testcases,
            _make_run_config(n_candidates=n_candidates, n_repeats=n_repeats, max_concurrency=1),
            str(tmp_path / "seq.jsonl"),
        )
        records_par = _run_with_mocks(
            testcases,
            _make_run_config(n_candidates=n_candidates, n_repeats=n_repeats, max_concurrency=2),
            str(tmp_path / "par.jsonl"),
        )

        assert len(records_seq) == expected
        assert len(records_par) == expected

    def test_output_order_matches_between_concurrency_1_and_2(self, tmp_path):
        """Output order must equal planner order (testcase × candidate × repeat)
        regardless of concurrency level."""
        testcases = [_make_testcase(f"tc-{i}") for i in range(3)]
        cfg_seq = _make_run_config(n_candidates=2, n_repeats=2, max_concurrency=1)
        cfg_par = _make_run_config(n_candidates=2, n_repeats=2, max_concurrency=2)

        records_seq = _run_with_mocks(
            testcases, cfg_seq, str(tmp_path / "seq.jsonl")
        )
        records_par = _run_with_mocks(
            testcases, cfg_par, str(tmp_path / "par.jsonl")
        )

        assert len(records_seq) == len(records_par)
        for r_seq, r_par in zip(records_seq, records_par):
            assert r_seq.testcase_id == r_par.testcase_id
            assert r_seq.candidate_id == r_par.candidate_id

    def test_order_is_testcase_candidate_repeat(self, tmp_path):
        """Records appear in (testcase × candidate × repeat_idx) order."""
        testcases = [_make_testcase(f"tc-{i}") for i in range(2)]
        cfg = _make_run_config(n_candidates=2, n_repeats=2, max_concurrency=2)

        records = _run_with_mocks(testcases, cfg, str(tmp_path / "out.jsonl"))

        expected_order = [
            (tc.testcase_id, f"cand-{c}", r)
            for tc in testcases
            for c in range(2)
            for r in range(2)
        ]
        actual_order = [
            (rec.testcase_id, rec.candidate_id, None) for rec in records
        ]
        # Verify testcase and candidate ordering (repeat_idx is not recorded in the output)
        for idx, (tc_id, cand_id, _) in enumerate(expected_order):
            assert records[idx].testcase_id == tc_id
            assert records[idx].candidate_id == cand_id


# ── test 3: _build_response_format called once per SO testcase ───────────────


class TestBuildResponseFormatCallCount:
    def test_called_once_per_testcase_not_per_candidate_repeat(self, tmp_path):
        """_build_response_format must be called once per SO testcase (not per task)."""
        so_testcases = [_make_testcase(f"so-tc-{i}", has_so=True) for i in range(2)]
        plain_testcases = [_make_testcase(f"plain-tc-{i}") for i in range(2)]
        testcases = so_testcases + plain_testcases

        # 2 candidates × 3 repeats = 6 tasks per testcase → would be 12 calls without caching
        cfg = _make_run_config(n_candidates=2, n_repeats=3, max_concurrency=2)

        with ExitStack() as stack:
            stack.enter_context(patch("llm_judge.stages.inference.load_dotenv"))
            stack.enter_context(
                patch("llm_judge.stages.inference.load_run_config", return_value=cfg)
            )
            stack.enter_context(
                patch("llm_judge.stages.inference.load_testcases", return_value=testcases)
            )
            stack.enter_context(
                patch(
                    "llm_judge.stages.inference.build_inference_prompt",
                    return_value=[{"role": "user", "content": "test"}],
                )
            )
            stack.enter_context(
                patch(
                    "llm_judge.stages.inference.get_thread_local_openai_client",
                    return_value=MagicMock(),
                )
            )
            stack.enter_context(
                patch(
                    "llm_judge.stages.inference.chat_completion",
                    return_value=_make_mock_response(),
                )
            )
            stack.enter_context(patch("llm_judge.stages.inference.validate_artifacts"))
            stack.enter_context(patch("llm_judge.stages.inference.write_jsonl"))

            mock_build_rf = stack.enter_context(
                patch(
                    "llm_judge.stages.inference._build_response_format",
                    return_value={
                        "type": "json_schema",
                        "json_schema": {"name": "test", "strict": True, "schema": {}},
                    },
                )
            )

            run_inference("config.yaml", str(tmp_path / "out.jsonl"))

        # Must be called exactly once per SO testcase, not per (testcase × candidate × repeat)
        assert mock_build_rf.call_count == len(so_testcases)


# ── test 4: partial failure ───────────────────────────────────────────────────


class TestPartialFailure:
    def test_success_and_failure_records_both_returned_in_order(self, tmp_path):
        """When some API calls fail, both success and failure records are returned
        in planner order."""
        testcases = [_make_testcase(f"tc-{i}") for i in range(3)]
        cfg = _make_run_config(n_candidates=1, n_repeats=1, max_concurrency=2)

        # Fail exactly one call.  With max_concurrency=2 the actual call
        # order is non-deterministic, so we identify the target task by
        # call_counter but only assert that *exactly one* failure exists
        # (not *which* testcase_id it maps to).
        call_counter = 0
        lock = threading.Lock()

        def chat_side_effect(*args, **kwargs):
            nonlocal call_counter
            with lock:
                idx = call_counter
                call_counter += 1
            if idx == 1:
                raise ValueError("simulated API error")
            return _make_mock_response(f"response-{idx}")

        records = _run_with_mocks(
            testcases, cfg, str(tmp_path / "out.jsonl"), chat_side_effect=chat_side_effect
        )

        assert len(records) == 3

        # Verify planner ordering is maintained
        for i, rec in enumerate(records):
            assert rec.testcase_id == f"tc-{i}"

        # Exactly one failure (which testcase hit it depends on scheduling)
        ok_records = [r for r in records if r.status.ok]
        fail_records = [r for r in records if not r.status.ok]
        assert len(ok_records) == 2
        assert len(fail_records) == 1
        assert fail_records[0].testcase_id in {"tc-0", "tc-1", "tc-2"}
        assert fail_records[0].status.error_type == "ValueError"

    def test_failure_record_preserves_identifiers(self, tmp_path):
        """Failure records must carry correct run_id, testcase_id, candidate_id."""
        testcases = [_make_testcase("failing-tc")]
        cfg = _make_run_config(n_candidates=1, n_repeats=1, max_concurrency=1)

        records = _run_with_mocks(
            testcases,
            cfg,
            str(tmp_path / "out.jsonl"),
            chat_side_effect=ValueError("boom"),
        )

        assert len(records) == 1
        rec = records[0]
        assert rec.status.ok is False
        assert rec.run_id == "test-run"
        assert rec.testcase_id == "failing-tc"
        assert rec.candidate_id == "cand-0"


# ── test 5: all tasks fail ────────────────────────────────────────────────────


class TestAllFail:
    def test_all_failure_returns_nonempty_list(self, tmp_path):
        """When all API calls fail, the result is a list of failure records (not empty)."""
        testcases = [_make_testcase(f"tc-{i}") for i in range(3)]
        cfg = _make_run_config(n_candidates=2, n_repeats=1, max_concurrency=2)
        expected_count = len(testcases) * 2  # 2 candidates

        records = _run_with_mocks(
            testcases,
            cfg,
            str(tmp_path / "out.jsonl"),
            chat_side_effect=RuntimeError("total failure"),
        )

        assert len(records) == expected_count
        assert all(not r.status.ok for r in records)

    def test_all_failure_order_maintained(self, tmp_path):
        """Even when all tasks fail, output order equals planner order."""
        testcases = [_make_testcase(f"tc-{i}") for i in range(3)]
        cfg = _make_run_config(n_candidates=1, n_repeats=1, max_concurrency=3)

        records = _run_with_mocks(
            testcases,
            cfg,
            str(tmp_path / "out.jsonl"),
            chat_side_effect=RuntimeError("boom"),
        )

        assert len(records) == 3
        for i, rec in enumerate(records):
            assert rec.testcase_id == f"tc-{i}"
