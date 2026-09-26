"""Shared spawn-process-pool policy for CPU-bound jobs."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def create_spawn_process_pool(max_workers: int) -> ProcessPoolExecutor:
    """Create the bounded spawn pool used inside non-daemon optimizer workers."""
    return ProcessPoolExecutor(
        max_workers=max(1, int(max_workers)),
        mp_context=multiprocessing.get_context("spawn"),
    )
