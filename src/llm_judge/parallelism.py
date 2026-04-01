"""Bounded parallel executor for evaluation stages."""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class PlannedTask(Generic[T]):
    task_index: int
    payload: T


def run_bounded_parallel(
    tasks: list[PlannedTask[T]],
    worker_fn: Callable[[T], R],
    max_workers: int,
) -> list[R]:
    """Run tasks with bounded parallelism and return results in task_index order.

    Args:
        tasks: Planner-ordered list of tasks with task_index starting at 0.
        worker_fn: Function to execute per task payload.
        max_workers: Maximum number of in-flight futures at once.

    Returns:
        Results in task_index order.

    Raises:
        RuntimeError: If any worker raises an uncaught exception.
        KeyboardInterrupt / SystemExit: Re-raised after cleanup.
    """
    if not tasks:
        return []

    if max_workers == 1:
        results: list[R | None] = [None] * len(tasks)
        for task in tasks:
            try:
                results[task.task_index] = worker_fn(task.payload)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                logger.error(
                    "Stage abort: task_index=%d raised %s: %s",
                    task.task_index,
                    type(exc).__name__,
                    exc,
                )
                raise RuntimeError(
                    f"Worker failed at task_index={task.task_index}: {type(exc).__name__}: {exc}"
                ) from exc
        return results  # type: ignore[return-value]

    results = [None] * len(tasks)
    future_to_index: dict[Future[R], int] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = iter(tasks)
        in_flight: list[Future[R]] = []

        for task in pending:
            future = executor.submit(worker_fn, task.payload)
            future_to_index[future] = task.task_index
            in_flight.append(future)
            if len(in_flight) >= max_workers:
                break

        try:
            while in_flight:
                done_future = next(as_completed(in_flight))
                in_flight.remove(done_future)
                idx = future_to_index.pop(done_future)

                try:
                    results[idx] = done_future.result()
                except BaseException as exc:
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    logger.error(
                        "Stage abort: task_index=%d raised %s: %s",
                        idx,
                        type(exc).__name__,
                        exc,
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(
                        f"Worker failed at task_index={idx}: {type(exc).__name__}: {exc}"
                    ) from exc

                try:
                    next_task = next(pending)
                    future = executor.submit(worker_fn, next_task.payload)
                    future_to_index[future] = next_task.task_index
                    in_flight.append(future)
                except StopIteration:
                    pass

        except (KeyboardInterrupt, SystemExit):
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    return results  # type: ignore[return-value]
