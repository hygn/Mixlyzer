import os
import subprocess
import sys
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

from core.workflows.event_bus import EventBus  # noqa: E402
from core.track.model import DataModel  # noqa: E402
from views.timeline import TimelineCoordinator  # noqa: E402
from views import waveform  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
STYLE = waveform._WaveStyle(bands=((20.0, 200.0), (200.0, 2000.0), (2000.0, 20000.0)), filter_order=4,
                            frame_ms=2.0, waveform_style="rgb", band_ref=None)


class _Pool:
    """Records the chunks started instead of rendering them."""

    def __init__(self, started=None):
        self.started = [] if started is None else started

    def start(self, job):
        self.started.append(job.chunk_index)

    def clear(self):
        pass


def specs(chunks):
    return [waveform._ChunkSpec(chunk_index=i, pcm=None, sample_rate=1, style=STYLE, height=16) for i in chunks]


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.worker = waveform._WaveformRenderWorker()
        self.worker.pool = _Pool()
        self.worker.extra_pool = _Pool()
        self.both = _Pool(self.worker.pool.started)   # the extra pool's starts also in the order of all
        self.worker.extra_pool.start = lambda job: (self.both.start(job), self.extra.append(job.chunk_index))
        self.extra = []

    def spawn(self, times):
        for _ in range(times):
            self.worker._spawn_next()
        return self.worker.pool.started

    def finish(self, chunk):
        self.worker._on_job_finished(0, waveform._WaveRenderResult(image=None, chunk_index=chunk, start_sec=0.0,
                                                                   end_sec=1.0, style=STYLE))

    def test_the_playhead_first_then_ahead_before_behind(self):
        self.worker.enqueue(specs(range(21)), 10)
        self.assertEqual(self.spawn(4), [10, 11, 9, 12])

    def test_a_request_at_the_playhead_overtakes_older_ones(self):
        self.worker.enqueue(specs(range(10)), 2)
        self.spawn(1)
        self.worker.enqueue(specs(range(50, 60)), 55)                # a seek: the old requests still pending
        self.assertEqual(self.spawn(2), [2, 55, 56])
        self.worker.set_playhead(5)                                  # the playhead alone moving reorders them
        self.assertEqual(self.spawn(1)[-1], 5)

    def test_slots_bound_the_jobs_running_and_refill_at_once(self):
        self.worker.enqueue(specs(range(10)), 0)
        self.assertEqual(len(self.spawn(6)), waveform.RENDER_BATCH_SIZE)
        self.finish(0)
        self.assertEqual(self.spawn(1)[-1], 4)                       # no wait for the whole batch

    def test_rendered_and_forgotten(self):
        self.worker.enqueue(specs([3]), 3)
        self.spawn(1)
        self.finish(3)
        self.worker.enqueue(specs([3, 4]), 3)                        # rendered: not again
        self.worker.forget([(STYLE, 4)])                             # dropped while pending
        self.assertEqual(self.spawn(3), [3])
        self.worker.forget([(STYLE, 3)])                             # the image dropped: asked again, rendered
        self.worker.enqueue(specs([3]), 3)
        self.assertEqual(self.spawn(1), [3, 3])


    def test_resuming_starts_its_chunk_at_once_and_drops_the_scrub_requests(self):
        self.worker.enqueue(specs(range(20)), 5)                     # requests made while scrubbing
        self.spawn(4)                                                # every slot busy
        self.worker.resume(specs(range(100, 110)), 104)              # let go at chunk 104
        self.assertEqual(self.extra, [104])                          # started now, on a thread of its own
        self.assertEqual(self.spawn(1)[-1], 104)                     # the slots still full: nothing more
        self.finish(5)
        self.assertEqual(self.spawn(1)[-1], 104)                     # the extra thread counts: 4 still run
        self.finish(104)
        self.assertEqual(self.spawn(1)[-1], 105)                     # then the view there, not the scrub's
        self.assertFalse({k[1] for k in self.worker._pending} & set(range(20)))

    def test_resuming_with_a_free_slot_uses_it(self):
        self.worker.resume(specs(range(100, 110)), 104)
        self.assertEqual((self.worker.pool.started, self.extra), ([104], []))


class BandFilterTests(unittest.TestCase):
    def test_the_cached_design_filters(self):
        # a filter failure is swallowed (the audio returned as it is): every band alike, the waveform white
        from scipy.signal import butter, sosfiltfilt
        y = np.random.default_rng(0).standard_normal(48000)
        for _ in range(2):   # designed, then from the cache
            got = waveform._band_filter(y, 48000, 20.0, 200.0, 4)
            np.testing.assert_allclose(got, sosfiltfilt(butter(4, [20 / 24000, 200 / 24000], btype="band",
                                                               output="sos"), y))


class EvictionTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.model = DataModel()
        self.model.set_decoded_duration(300.0)
        self.tl = TimelineCoordinator(self.bus)
        self.view = waveform.WaveformView(self.bus, self.model, self.tl)
        self.view.duration = 300.0
        self.view.attach(pg.PlotItem())
        with mock.patch.object(waveform._WaveStyle, "from_config", return_value=STYLE):
            self.view._allocate_canvas(force=True)
        self.layer = self.view._layer
        self.forgotten = []
        self.view._sig_forget.connect(self.forgotten.extend)

    def tearDown(self):
        self.view.detach()

    def render(self, chunks, shown):
        for i in chunks:
            self.layer.items[i] = self.view._make_chunk_item()
            self.layer.spans[i] = (float(i), float(i + 1))
            self.layer.shown[i] = shown

    def test_far_chunks_dropped_but_the_recently_shown(self):
        self.render(range(0, 10), shown=1)       # shown long ago
        self.render(range(10, 20), shown=5)      # shown a moment ago
        self.render(range(20, 30), shown=0)      # rendered, never shown
        self.render(range(190, 211), shown=3)    # around the playhead
        self.layer.submitted.update({40, 41})    # requested, not rendered yet
        with mock.patch.object(waveform, "RENDER_RECENT_KEEP_SEC", 15.0):
            self.view._evict_far_chunks(200)
        kept = {i for i, item in enumerate(self.layer.items) if item is not None}
        self.assertTrue(set(range(190, 211)) <= kept)
        self.assertTrue(set(range(10, 20)) <= kept)
        self.assertEqual(len(kept & set(range(0, 10))), 5)           # the cap: 15 chunks of the far ones
        self.assertFalse(kept & set(range(20, 30)))                  # never shown: not recently used
        self.assertFalse(self.layer.submitted & {40, 41})
        self.assertEqual({i for _, i in self.forgotten}, (set(range(30)) - kept) | {40, 41})

    def test_shown_chunks_are_stamped(self):
        self.render(range(0, 300), shown=0)
        self.view._set_rect(force=True)
        self.assertGreater(self.layer.shown[0], 0)                   # in the view at the start
        self.assertEqual(self.layer.shown[250], 0)

    def test_scrub_end_resumes_there(self):
        self.model.predecoded_pcm, self.model.predecoded_rate = np.zeros((300 * 100, 1), np.float32), 100
        self.render(range(0, 20), shown=4)
        self.layer.submitted.update({30, 31})                        # asked for while scrubbing
        sent = []
        self.view._sig_resume.connect(lambda specs_, chunk: sent.append(([s.chunk_index for s in specs_], chunk)))
        with mock.patch.object(waveform._WaveStyle, "from_config", return_value=STYLE):
            self.bus.sig_scrub_end.emit(150.5)
        chunks, chunk = sent[-1]
        self.assertEqual(chunk, 150)
        self.assertIn(150, chunks)                                   # the view at 150 s, its chunk with it
        self.assertFalse(self.layer.submitted & {30, 31})            # the worker drops them: asked again later
        self.assertTrue(all(self.layer.items[i] is not None for i in range(20)))   # kept: shown recently

    def test_a_jump_of_the_playhead_resumes_there(self):
        """The playhead jumped (a seek, a CALL, a JumpCUE: past the next chunk) is played on from as a scrub let go;
        playing on into the next chunk, or dragged, only followed."""
        self.model.predecoded_pcm, self.model.predecoded_rate = np.zeros((300 * 100, 1), np.float32), 100
        resumed, followed = [], []
        self.view._sig_resume.connect(lambda _specs, chunk: resumed.append(chunk))
        self.view._sig_playhead.connect(followed.append)

        def at(t):
            self.tl.current_time = t
            with mock.patch.object(waveform._WaveStyle, "from_config", return_value=STYLE):
                self.view._follow_playhead()
        at(10.2)
        at(11.1)                       # played on into the next chunk
        self.assertEqual((resumed, followed), ([], [10, 11]))
        at(150.5)                      # jumped
        self.assertEqual(resumed, [150])
        at(151.2)
        self.assertEqual(followed[-1], 151)
        self.bus.sig_scrub_begin.emit()
        at(40.0)                       # dragged there: followed, resumed when let go
        self.assertEqual((resumed, followed[-1]), ([150], 40))
        with mock.patch.object(waveform._WaveStyle, "from_config", return_value=STYLE):
            self.bus.sig_scrub_end.emit(40.0)
        self.assertEqual(resumed, [150, 40])

    def test_a_drag_or_a_travel_of_the_playhead_resumes_where_it_stops(self):
        """The main plot dragged is a scrub as the overview's; the playhead jumping frame after frame (a controller's
        jog search) is rendered once where it stops, not at every frame."""
        self.model.predecoded_pcm, self.model.predecoded_rate = np.zeros((300 * 100, 1), np.float32), 100
        resumed = []
        self.view._sig_resume.connect(lambda _specs, chunk: resumed.append(chunk))

        def at(t):
            self.tl.current_time = t
            with mock.patch.object(waveform._WaveStyle, "from_config", return_value=STYLE):
                self.view._follow_playhead()
        at(10.2)
        self.bus.sig_scratch_begin.emit(0.01)
        for t in (20.0, 35.0, 50.0):    # flicked: seconds a frame
            at(t)
        self.assertEqual(resumed, [])
        with mock.patch.object(waveform._WaveStyle, "from_config", return_value=STYLE):
            self.bus.sig_scratch_end.emit(50.0)
        self.assertEqual(resumed, [50])
        for t in (60.0, 70.0, 80.0, 90.0):   # searched
            at(t)
        self.assertEqual(resumed, [50, 60])     # its first jump, as a seek's
        at(90.4)                                # stopped there
        self.assertEqual(resumed, [50, 60, 90])
        at(91.1)
        self.assertEqual(resumed, [50, 60, 90])


# The app's exit: the views held by the window's pane, freed when main()'s frame unwinds after app.exec()
# (the window, its pane and their list of views with it), the render thread still running unless stopped.
QUIT_SCRIPT = """
import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets


def main():
    app = QtWidgets.QApplication([])
    from core.workflows.event_bus import EventBus
    from core.track.model import DataModel
    from views.timeline import TimelineCoordinator
    from views.waveform import WaveformView
    window = QtWidgets.QMainWindow()
    pane = QtWidgets.QWidget(window)
    bus = EventBus()
    pane.plugins = [WaveformView(bus, DataModel(), TimelineCoordinator(bus))]
    pane.plugins[0].attach(pg.PlotItem())
    QtCore.QTimer.singleShot(0, app.quit)
    sys.exit(app.exec())


main()
"""


class QuitTests(unittest.TestCase):
    def test_quitting_stops_the_render_thread(self):
        env = dict(os.environ, QT_FORCE_STDERR_LOGGING="1")   # Qt's messages on stderr (Windows: not by default)
        run = subprocess.run([sys.executable, "-c", QUIT_SCRIPT], capture_output=True, text=True, timeout=120,
                             cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), env=env)
        self.assertNotIn("QThread: Destroyed", run.stderr)
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()
