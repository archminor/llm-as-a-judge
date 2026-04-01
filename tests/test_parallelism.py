"""Tests for run_bounded_parallel."""

from __future__ import annotations

import threading
import time

import pytest

from llm_judge.parallelism import PlannedTask, run_bounded_parallel


def make_tasks(payloads: list) -> list[PlannedTask]:
    return [PlannedTask(task_index=i, payload=p) for i, p in enumerate(payloads)]


# ── sequential fast path ──────────────────────────────────────────────────────


def test_max_workers_1_returns_in_order():
    tasks = make_tasks([3, 1, 2])
    results = run_bounded_parallel(tasks, lambda x: x * 10, max_workers=1)
    assert results == [30, 10, 20]


def test_max_workers_1_empty():
    assert run_bounded_parallel([], lambda x: x, max_workers=1) == []


# ── parallel path ─────────────────────────────────────────────────────────────


def test_max_workers_gt1_order_independent_of_completion():
    """Even when slow tasks finish last, results are task_index ordered."""
    # task 0 sleeps longer than task 1 → task 1 finishes first
    delays = [0.05, 0.0, 0.0]
    tasks = make_tasks(delays)
    results = run_bounded_parallel(tasks, lambda d: d, max_workers=2)
    assert results == delays


def test_max_workers_gt1_same_result_as_sequential():
    tasks = make_tasks(list(range(10)))
    seq = run_bounded_parallel(tasks, lambda x: x ** 2, max_workers=1)
    par = run_bounded_parallel(tasks, lambda x: x ** 2, max_workers=3)
    assert seq == par


def test_empty_with_max_workers_gt1():
    assert run_bounded_parallel([], lambda x: x, max_workers=4) == []


# ── error handling ────────────────────────────────────────────────────────────


def test_worker_exception_raises_runtime_error():
    def bad_worker(x):
        if x == 1:
            raise ValueError("boom")
        return x

    tasks = make_tasks([0, 1, 2])
    with pytest.raises(RuntimeError, match="task_index=1"):
        run_bounded_parallel(tasks, bad_worker, max_workers=1)


def test_worker_exception_parallel_raises_runtime_error():
    def bad_worker(x):
        raise RuntimeError("worker error")

    tasks = make_tasks([0, 1, 2])
    with pytest.raises(RuntimeError):
        run_bounded_parallel(tasks, bad_worker, max_workers=2)


def test_keyboard_interrupt_reraises():
    def interruptible(x):
        raise KeyboardInterrupt

    tasks = make_tasks([0])
    with pytest.raises(KeyboardInterrupt):
        run_bounded_parallel(tasks, interruptible, max_workers=1)


def test_keyboard_interrupt_reraises_parallel():
    barrier = threading.Barrier(2, timeout=5)

    def interruptible(x):
        if x == 0:
            barrier.wait()
            raise KeyboardInterrupt
        barrier.wait()
        return x

    tasks = make_tasks([0, 1])
    with pytest.raises(KeyboardInterrupt):
        run_bounded_parallel(tasks, interruptible, max_workers=2)


# ── bounded submit (not all-at-once) ─────────────────────────────────────────


def test_bounded_submit_max_concurrency():
    """Verify that at most max_workers tasks run concurrently."""
    max_workers = 2
    concurrent_count = 0
    max_concurrent_seen = 0
    lock = threading.Lock()

    def worker(x):
        nonlocal concurrent_count, max_concurrent_seen
        with lock:
            concurrent_count += 1
            if concurrent_count > max_concurrent_seen:
                max_concurrent_seen = concurrent_count
        time.sleep(0.02)
        with lock:
            concurrent_count -= 1
        return x

    tasks = make_tasks(list(range(8)))
    results = run_bounded_parallel(tasks, worker, max_workers=max_workers)

    assert results == list(range(8))
    assert max_concurrent_seen <= max_workers
