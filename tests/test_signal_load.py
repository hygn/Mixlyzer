import os
import tempfile
import threading
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore  # noqa: E402

from core.playback.player import _Latest  # noqa: E402
from core.workers.rekordbox_sync import RekordboxSyncRequest, _RekordboxSyncEngine  # noqa: E402


class _Source(QtCore.QObject):
    ready = QtCore.Signal()


class _Sink(QtCore.QObject):
    def __init__(self, latest):
        super().__init__()
        self.latest = latest
        self.got = []

    @QtCore.Slot()
    def relay(self):
        self.got.append(self.latest.take())


class LatestReportTests(unittest.TestCase):
    def setUp(self):
        self.app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    def test_reports_do_not_pile_up(self):
        source = _Source()
        latest = _Latest(source.ready)
        sink = _Sink(latest)
        source.ready.connect(sink.relay, QtCore.Qt.QueuedConnection)
        # a thread publishing while the receiving thread is busy: one notification, the newest value
        worker = threading.Thread(target=lambda: [latest.publish(i) for i in range(1000)])
        worker.start()
        worker.join()
        self.app.processEvents()
        self.assertEqual(sink.got, [999])
        # taken: the next value notifies again
        latest.publish(1000)
        self.app.processEvents()
        self.assertEqual(sink.got, [999, 1000])


class RekordboxProgressTests(unittest.TestCase):
    def test_track_progress_by_percent(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            engine = _RekordboxSyncEngine(
                RekordboxSyncRequest(library_dir=Path(tmp), xml_path=Path(tmp) / "x.xml", mode="library"),
                lambda value, text: calls.append((value, text)), None)
            for index in range(1, 10_001):
                engine._track_progress(index, 10_000, f"Syncing tracks {index}/10000")
            self.assertLessEqual(len(calls), 87)   # the 86 percents 5..90 and the last track's count
            self.assertEqual(calls[-1], (90, "Syncing tracks 10000/10000"))


if __name__ == "__main__":
    unittest.main()
