"""Re-read Library's workflow (core.workflows.reading_refresh): the texts read, the library's owner writes them back a
batch of tracks a job; one batch's write failing says the run failed, not finished. A stand-in owner runs each job's
callback at once; no process is started."""
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from core.library import jobs  # noqa: E402
from core.workers.reading_refresh import BATCH_TRACKS, ReadingRefreshRequest  # noqa: E402
from core.workflows.reading_refresh import LibraryReadingRefresh  # noqa: E402

APP = QApplication.instance() or QApplication([])


class _Library:
    """Each write's callback at once; the batch starting at ``fail_at`` raises."""

    def __init__(self, fail_at=None):
        self.fail_at = fail_at

    def write(self, job, *args, done=None, failed=None):
        if job is jobs.apply_readings and self.fail_at is not None and args[0][0] == f"t{self.fail_at}":
            if failed is not None:
                failed(OSError("disk full"))
            return
        if done is not None:
            done(1 if job is jobs.apply_readings else None)


class ReadingRefreshTests(unittest.TestCase):
    def run_writes(self, library):
        flow = LibraryReadingRefresh(cfg_getter=lambda: SimpleNamespace(), library=library)
        finished, failed = [], []
        flow.finished.connect(lambda request, result: finished.append(result))
        flow.failed.connect(lambda request, message: failed.append(message))
        flow._busy = True
        flow._uids = [f"t{k}" for k in range(2 * BATCH_TRACKS)]
        flow._active = ReadingRefreshRequest(texts=(), languages=(), han_run=0, tracks=2 * BATCH_TRACKS)
        flow._job_finished({})
        return flow, finished, failed

    def test_every_batch_written_finishes(self):
        flow, finished, failed = self.run_writes(_Library())
        self.assertEqual((len(finished), failed), (1, []))
        self.assertEqual(finished[0].changed_fields, 2)
        self.assertFalse(flow._busy)

    def test_a_batch_not_written_fails_the_run(self):
        flow, finished, failed = self.run_writes(_Library(fail_at=BATCH_TRACKS))
        self.assertEqual((finished, len(failed)), ([], 1))
        self.assertIn("disk full", failed[0])
        self.assertFalse(flow._busy)


if __name__ == "__main__":
    unittest.main()
