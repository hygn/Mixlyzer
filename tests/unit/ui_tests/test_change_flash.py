"""Where the marks just changed, drawn (views.change_flash): only that stretch of the time axis blinks, on and off, in its
own marks: a key's strip darkened, a phrase's at its full colour, a beatgrid's lines thickened, a CUE's marker; nothing
else is drawn and nothing is left behind; the same in the overview's own strips."""
import math
import unittest
from unittest import mock

import numpy as np

from unit.ui_tests.qt_env import APP, drain, isolated

from PySide6 import QtGui
from PySide6.QtTest import QTest
import pyqtgraph as pg

from analyzer.music.marks import CuePoint, JumpCue, JumpCueGroup, Phrase
from core.analysis_library.cue_points import build_cue_points_np
from core.analysis_library.stored import StoredAnalysis
from core.marks.phrases import UNASSIGNED, phrase_color
from core.settings.config import default_cfg
from core.track.changes import BEATGRID, CUE, KEY, PHRASE, Change
from core.track.model import DataModel
from core.workflows.event_bus import EventBus
from views.JumpCUE import JumpCUEView
from views.base import KEY_LANE, PHRASE_LANE, WAVEFORM_LANE
from views.beatgrid import BeatgridView
from views.change_flash import (BLINK_MS, BLINKS, HOLD_MS, JUMP_CUE_Y, LIT_DARKEN, LIT_UNASSIGNED, Blinker, ChangeFlashView,
                                LitBands, band_rect, phrase_lit_segments)
from views.overview import OverviewWidget
from views.timeline import TimelineCoordinator

_ISOLATION = None
GONE_MS = BLINK_MS * 2 * BLINKS + 150


def setUpModule():
    global _ISOLATION
    _ISOLATION = isolated()


def tearDownModule():
    _ISOLATION.close()


def model_of(duration: float = 64.0) -> DataModel:
    model = DataModel()
    model.load(StoredAnalysis.from_arrays({
        "beats_time_sec": np.arange(0.0, duration, 0.5),
        "tempo_segments": np.asarray([[0.0, duration, 120.0, 0.0, 4.0]]),
        "key_segments": np.asarray([[0.0, 0.0, 0.0, 30.0], [9.0, 1.0, 30.0, duration]]),
        "cue_points_np": build_cue_points_np([CuePoint(time_sec=8.0)]),
        "duration_sec": duration}), {"uid": "u"}, duration)
    return model


def rects_of(bands) -> list:
    return [(r.x(), r.y(), r.width(), r.height()) for r in bands._rects]


class BlinkerTests(unittest.TestCase):
    def test_on_and_off_a_few_times(self):
        told, done = [], []
        host = pg.QtCore.QObject()
        blinker = Blinker(host, told.append, 10, 60, done=lambda: done.append(True))
        blinker.run(2)
        QTest.qWait(120)
        self.assertEqual(told, [True, False, True, False])            # on, off, on, off; the hold's end adds nothing
        self.assertEqual(done, [True])

    def test_changes_in_a_row_stay_on_until_they_stop(self):
        told, done = [], []
        host = pg.QtCore.QObject()
        blinker = Blinker(host, told.append, 10, 500, done=lambda: done.append(True))
        blinker.run(2)
        QTest.qWait(120)                                              # its blink is over
        self.assertEqual(told, [True, False, True, False])
        done.clear()
        told.clear()
        blinker.run(2)                                                # one more, soon after
        QTest.qWait(60)
        blinker.run(2)
        QTest.qWait(60)
        self.assertEqual(told, [True, True])                          # on, no blink
        self.assertEqual(done, [])
        QTest.qWait(700)
        self.assertEqual(told, [True, True, False])                   # off once they stopped
        self.assertEqual(done, [True])
        told.clear()
        blinker.run(2)                                                # long after: a blink again
        QTest.qWait(120)
        self.assertEqual(told, [True, False, True, False])

    def test_a_change_during_a_blink_stays_on(self):
        told = []
        host = pg.QtCore.QObject()
        blinker = Blinker(host, told.append, 10, 500)
        blinker.run(2)
        blinker._toggle()                                             # in its off
        self.assertEqual(told, [True, False])
        blinker.run(2)
        QTest.qWait(100)                                              # the blink's own turns would have come
        self.assertEqual(told, [True, False, True])
        blinker.stop()

    def test_stopped_nothing_follows(self):
        told, done = [], []
        host = pg.QtCore.QObject()
        blinker = Blinker(host, told.append, 10, 30, done=lambda: done.append(True))
        blinker.run(2)
        blinker.run(2)
        blinker.stop()
        QTest.qWait(80)
        self.assertEqual(told, [True, True, False])
        self.assertEqual(done, [])

    def test_a_changes_blinks_are_all_there_is(self):
        self.assertEqual(BLINKS, 1)
        self.assertLess(BLINK_MS * 2 * BLINKS, HOLD_MS)               # a blink is over before the hold
        self.assertLessEqual(BLINKS * math.ceil(1000 / HOLD_MS), 3)   # blinks a hold apart: at most three in any second


class ChangeFlashViewTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.tl = TimelineCoordinator(self.bus)
        self.tl.set_pixel_sec(0.01)
        self.model = model_of()
        self.view = ChangeFlashView(self.bus, self.model, self.tl)
        self.view.attach(pg.PlotItem())

    def tearDown(self):
        self.view.detach()
        drain()

    def test_a_change_blinks_over_its_stretch_and_is_gone_on_its_own(self):
        self.model.set_key_segments(np.asarray([[0.0, 0.0, 0.0, 20.0], [9.0, 1.0, 20.0, 64.0]]))
        (x, y, w, h), = rects_of(self.view._band)
        self.assertEqual((x, w), (0.0, 64.0))                                # the boundary moved: both its segments
        self.assertAlmostEqual(y, KEY_LANE[0])
        self.assertAlmostEqual(h, KEY_LANE[1] - KEY_LANE[0])
        self.assertTrue(self.view._band._lit)                                # on at once
        self.assertTrue(self.view._beats[0].path().isEmpty())                # nothing else
        self.assertEqual(len(self.view._marker.getData()[0]), 0)
        QTest.qWait(GONE_MS)
        self.assertEqual(rects_of(self.view._band), [])
        self.assertFalse(self.view._band._lit)

    def test_a_grid_nudged_again_and_again_stays_on_at_the_latest_then_nothing_left(self):
        lines = self.view._beats[0]
        for shift in (0.01, 0.02, 0.03):
            self.model.set_beatgrid(np.array(self.model.beats) + 0.01, self.model.tempo_segments)
            self.assertTrue(lines.isVisible())
            self.assertAlmostEqual(lines.path().elementAt(0).x, shift)    # the beats where they are now
            QTest.qWait(BLINK_MS + 20)                                    # a blink's off, had it blinked
        self.assertTrue(lines.isVisible())
        QTest.qWait(HOLD_MS + 150)
        self.assertFalse(lines.isVisible())
        self.assertTrue(lines.path().isEmpty())

    def test_a_keys_stretch_dark_while_lit_under_the_beat_lines(self):
        self.assertLess(self.view._band.zValue(), self.view._beats[0].zValue())
        image = QtGui.QImage(4, 4, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(200, 100, 50))
        bands = LitBands()
        bands.set_rects([(0.0, 0.0, 4.0, 4.0)])
        painter = QtGui.QPainter(image)
        bands.paint(painter)                                # not lit: nothing
        self.assertEqual(image.pixelColor(1, 1).red(), 200)
        bands.set_lit(True)
        bands.paint(painter)
        painter.end()
        self.assertAlmostEqual(image.pixelColor(1, 1).red(), 200 * (1.0 - LIT_DARKEN), delta=2)
        self.assertAlmostEqual(image.pixelColor(1, 1).blue(), 50 * (1.0 - LIT_DARKEN), delta=2)

    def test_a_phrases_stretch_at_its_full_colour(self):
        rows = [Phrase(0.0, 16.0, "INTRO"), Phrase(16.0, 48.0, "VERSE"), Phrase(48.0, 64.0, UNASSIGNED)]
        self.assertEqual(phrase_lit_segments(rows, 10.0, 20.0, 0.1),         # more than one: each its own colour
                         [(10.0, 16.0, phrase_color("INTRO")), (16.0, 20.0, phrase_color("VERSE"))])
        self.assertEqual(phrase_lit_segments(rows, 50.0, 60.0, 0.1)[0][2], LIT_UNASSIGNED)
        (start, end, _colour), = phrase_lit_segments(rows, 20.0, 20.0, 0.1)  # too short: still shown
        self.assertAlmostEqual(end - start, 0.1)
        self.model.set_phrases(rows)
        self.view._draw([Change(PHRASE, 16.0, 48.0)])
        self.assertEqual(rects_of(self.view._band), [])
        strip = self.view._phrase_strip
        self.assertEqual([s[:2] for s in strip._segments], [(16.0, 48.0)])
        self.assertAlmostEqual(strip._y0, PHRASE_LANE[0])
        self.assertAlmostEqual(strip.opacity(), 1.0)                         # not the strip's own faint look

    def test_a_beatgrids_own_lines_not_the_waveform(self):
        xs = lambda item: sorted({item.path().elementAt(i).x for i in range(item.path().elementCount())})  # noqa: E731
        self.view._draw([Change(BEATGRID, 10.0, 12.0)])
        self.assertEqual(rects_of(self.view._band), [])
        self.assertEqual(xs(self.view._beats[0]), [10.0, 10.5, 11.0, 11.5])  # the downbeat among them, as drawn
        self.assertEqual(xs(self.view._downbeats[0]), [10.0])
        self.view._set_lit(True)
        self.assertTrue(all(item.isVisible() for item in self.view._beats))
        self.view._set_lit(False)
        self.assertFalse(any(item.isVisible() for item in self.view._beats))
        self.view._draw([Change(BEATGRID, 10.1, 10.2)])                      # no line in it: the stretch itself
        (x, y, w, _h), = rects_of(self.view._band)
        self.assertAlmostEqual(y, WAVEFORM_LANE[0])
        self.assertAlmostEqual(x + w / 2.0, 10.15)

    def test_a_shift_thickens_the_beats_where_they_are_now(self):
        self.model.set_beatgrid(np.array(self.model.beats) + 0.25, self.model.tempo_segments)
        path = self.view._beats[0].path()
        self.assertEqual(path.elementAt(0).x, 0.25)
        self.assertEqual(rects_of(self.view._band), [])

    def test_a_cue_change_its_marker_an_undo_too_and_a_new_track_stops_it(self):
        self.model.set_cue_points([*self.model.cue_points, CuePoint(time_sec=20.0)])
        self.assertEqual(list(self.view._marker.getData()[0]), [20.0])
        self.assertTrue(self.view._marker.isVisible())
        QTest.qWait(GONE_MS)
        self.assertFalse(self.view._marker.isVisible())
        self.model.undo()                                                    # soon after: on, until a new track
        self.assertEqual(list(self.view._marker.getData()[0]), [20.0])
        self.assertTrue(self.view._marker.isVisible())
        self.model.load(StoredAnalysis(), {}, 0.0)
        self.assertEqual(len(self.view._marker.getData()[0]), 0)
        self.assertFalse(self.view._marker.isVisible())

    def test_off_in_the_settings_nothing_blinks(self):
        self.assertTrue(default_cfg().viewconfig.highlight_changes)
        settings = default_cfg()
        settings.viewconfig.highlight_changes = False
        with mock.patch("views.change_flash.current_settings", return_value=settings):
            self.model.set_key_segments(np.asarray([[0.0, 0.0, 0.0, 20.0], [9.0, 1.0, 20.0, 64.0]]))
        self.assertEqual(rects_of(self.view._band), [])
        self.assertFalse(self.view._band._lit)

    def test_a_jumpcue_blinks_as_a_jumpcue_with_its_letter(self):
        self.model.set_jump_cues([JumpCueGroup((JumpCue("A", "A", 20.0, 20.0, 20.0),), 0)])
        (point, marker), = self.view._jump_markers
        self.assertEqual((point, marker._label), (20.0, "A"))
        self.assertTrue(marker.isVisible())
        self.assertEqual(marker.pos().x(), self.tl.left_edge() + 20.0)
        self.assertEqual(marker.pos().y(), 1.0 - JumpCUEView.TOP_PADDING - JumpCUEView.BLOCK_HEIGHT)
        self.assertEqual(marker.pos().y(), JUMP_CUE_Y)
        self.assertGreater(marker.zValue(), 110)                             # over the real ones
        self.assertEqual(len(self.view._marker.getData()[0]), 0)             # no CUE point's triangle
        QTest.qWait(GONE_MS)
        self.assertEqual(self.view._jump_markers, [])
        self.assertIsNone(marker.scene())
        self.model.set_jump_cues([])                                         # taken out: where it was, no letter
        (point, marker), = self.view._jump_markers
        self.assertEqual((point, marker._label), (20.0, ""))
        self.model.set_cue_points([*self.model.cue_points, CuePoint(time_sec=30.0)])
        self.assertEqual(self.view._jump_markers, [])                        # a CUE point: its triangle alone
        self.assertEqual(list(self.view._marker.getData()[0]), [30.0])

    def test_cues_changed_a_marker_at_each_end(self):
        self.view._draw([Change(CUE, 40.0, 50.0)])
        self.assertEqual(rects_of(self.view._band), [])
        self.assertEqual(list(self.view._marker.getData()[0]), [40.0, 50.0])

    def test_moves_with_the_view(self):
        self.view._draw([Change(KEY, 10.0, 20.0)])
        self.bus.sig_time_changed.emit(30.0)
        left = self.tl.left_edge()
        self.assertEqual(self.view._band.pos().x(), left)
        self.assertEqual(self.view._marker.pos().x(), left)
        self.assertEqual([item.pos().x() for item in self.view._beats], [left - 0.01, left, left + 0.01])

    def test_a_very_short_stretch_still_shows(self):
        x, _y, w, _h = band_rect(10.0, 10.001, 0.0, 1.0, 0.04)
        self.assertEqual(w, 0.04)
        self.assertAlmostEqual(x + w / 2.0, 10.0005)


class BeatgridPixelTests(unittest.TestCase):
    """Rendered, measured: lit lines lie on the pixels of the lines they thicken, wherever the beats fall."""

    def test_on_the_base_lines_pixels_at_every_position(self):
        with mock.patch("core.settings.store.current", return_value=default_cfg()):
            pg.setConfigOptions(antialias=False, useOpenGL=False)      # as the app's main view (ui.pane)
            bus = EventBus()
            tl = TimelineCoordinator(bus)
            model = DataModel()
            widget = pg.PlotWidget()
            widget.resize(1200, 260)
            plot = widget.getPlotItem()
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            plot.setYRange(0.0, 1.0, padding=0.05)
            base, lit = BeatgridView(bus, model, tl), ChangeFlashView(bus, model, tl)
            base.attach(plot)
            lit.attach(plot)
            beats = np.arange(0.0, 64.0, 60.0 / 171.3)                 # the beats on every fraction of a pixel
            model.load(StoredAnalysis.from_arrays({"beats_time_sec": beats, "duration_sec": 64.0,
                                                   "tempo_segments": np.asarray([[0.0, 64.0, 171.3, 0.0, 4.0]])}),
                       {"uid": "u"}, 64.0)
            widget.show()
            APP.processEvents()
            bus.sig_window_changed.emit(12.0)
            plot.setXRange(0.0, 12.0, padding=0)
            tl.set_pixel_sec(12.0 / plot.getViewBox().width())
            base.render_initial()

            def columns() -> np.ndarray:
                """The columns lit on a row between the downbeats' caps, the view's edges left out (a line may be
                cut there)."""
                APP.processEvents()
                image = widget.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB32)
                pixels = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), 4)
                row = pixels[image.height() // 2, :, :3].sum(axis=1) > 60
                return np.flatnonzero(row[3:-3]) + 3

            try:
                for t in np.linspace(20.0, 24.0, 12) + 0.00137:
                    bus.sig_time_changed.emit(float(t))
                    lit._set_lit(False)
                    lit._draw(())
                    off = columns()
                    lit._draw([Change(BEATGRID, 0.0, 64.0)])
                    lit._set_lit(True)
                    on = set(columns().tolist())
                    self.assertGreater(off.size, 20)
                    expected = {c + d for c in off.tolist() for d in (-1, 0, 1)}
                    inner = range(3, widget.grab().width() - 3)
                    self.assertEqual(on, {c for c in expected if c in inner})
            finally:
                lit.detach()
                base.detach()
                widget.close()
                widget.deleteLater()
                drain()


class OverviewTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.model = model_of()
        self.overview = OverviewWidget(self.bus, tl=TimelineCoordinator(self.bus), model=self.model)
        self.overview.duration = 64.0

    def tearDown(self):
        self.overview._change_blinker.stop()
        self.overview.close()
        self.overview.deleteLater()
        drain()

    def test_a_change_only_in_its_own_strip(self):
        o = self.overview
        marks = o.change_flash
        self.model.set_key_segments(np.asarray([[0.0, 0.0, 0.0, 20.0], [9.0, 1.0, 20.0, 64.0]]))
        (x, y, w, h), = rects_of(marks.strips)
        self.assertEqual((x, w), (0.0, 64.0))
        self.assertEqual((y, y + h), (o._key_bottom, o._key_bottom + o.KEY_H))
        self.assertEqual(rects_of(marks.bpm), [])
        self.assertTrue(marks.strips._lit)
        self.model.set_phrases([Phrase(0.0, 16.0, "INTRO"), Phrase(16.0, 64.0, "VERSE")])
        self.model.set_phrases([Phrase(0.0, 16.0, "INTRO"), Phrase(16.0, 64.0, "CHORUS")])  # its own colour, at full strength
        self.assertEqual(rects_of(marks.strips), [])
        self.assertEqual([s[:2] for s in marks.phrases._segments], [(16.0, 64.0)])
        self.assertEqual((marks.phrases._y0, marks.phrases._h), (o._wave_bottom, o._phrase_band_h))
        self.model.set_beatgrid(np.array(self.model.beats) + 0.25, self.model.tempo_segments)   # the BPM strip: black, so filled
        self.assertEqual(rects_of(marks.strips), [])
        (_x, y, _w, h), = rects_of(marks.bpm)
        self.assertEqual((y, y + h), (o._segment_bottom, 0.0))

    def test_a_few_blinks_then_nothing_lit(self):
        marks = self.overview.change_flash
        self.model.set_key_segments(np.asarray([[0.0, 0.0, 0.0, 20.0], [9.0, 1.0, 20.0, 64.0]]))
        self.assertTrue(marks.strips._lit)
        QTest.qWait(GONE_MS)
        self.assertFalse(marks.strips._lit)
        self.assertFalse(marks.phrases.isVisible())
        self.assertFalse(marks.marker.isVisible())

    def test_changes_in_a_row_stay_on_here_too(self):
        marks = self.overview.change_flash
        for boundary in (20.0, 20.5, 21.0):
            self.model.set_key_segments(np.asarray([[0.0, 0.0, 0.0, boundary], [9.0, 1.0, boundary, 64.0]]))
            self.assertTrue(marks.strips._lit)
            QTest.qWait(BLINK_MS + 20)
        self.assertTrue(marks.strips._lit)
        QTest.qWait(HOLD_MS + 150)
        self.assertFalse(marks.strips._lit)

    def test_off_in_the_settings_nothing_blinks(self):
        marks = self.overview.change_flash
        settings = default_cfg()
        settings.viewconfig.highlight_changes = False
        with mock.patch("views.overview.current_settings", return_value=settings):
            self.model.set_key_segments(np.asarray([[0.0, 0.0, 0.0, 20.0], [9.0, 1.0, 20.0, 64.0]]))
        self.assertEqual(rects_of(marks.strips), [])
        self.assertFalse(marks.strips._lit)

    def test_a_jumpcue_its_own_square_under_its_letter(self):
        marks = self.overview.change_flash
        self.overview.p.setXRange(0.0, 64.0, padding=0)                      # a cue off the view is not drawn
        self.model.set_jump_cues([JumpCueGroup((JumpCue("A", "A", 15.0, 15.0, 15.0),), 0)])
        self.assertEqual(list(marks.jump_cue.getData()[0]), [15.0])
        self.assertEqual(list(marks.jump_cue.getData()[1]), [self.overview._cue_label_y])
        self.assertTrue(marks.jump_cue.isVisible())
        self.assertEqual(len(marks.marker.getData()[0]), 0)                  # no CUE point's triangle
        marker, text = self.overview.jump_items[0]
        self.assertLess(marker.zValue(), marks.jump_cue.zValue())
        self.assertLess(marks.jump_cue.zValue(), text.zValue())

    def test_a_cue_its_marker_and_a_short_stretch_wide_enough_to_see(self):
        marks = self.overview.change_flash
        self.model.set_cue_points([*self.model.cue_points, CuePoint(time_sec=15.0)])
        self.assertEqual(list(marks.marker.getData()[0]), [15.0])
        self.assertEqual(len(marks.jump_cue.getData()[0]), 0)
        self.assertEqual(rects_of(marks.strips) + rects_of(marks.bpm), [])
        self.model.set_key_segments(np.asarray([[0.0, 0.0, 0.0, 30.0], [9.0, 1.0, 30.0, 30.01], [9.0, 1.0, 30.01, 64.0]]))
        self.model.set_key_segments(np.asarray([[0.0, 0.0, 0.0, 30.0], [5.0, 1.0, 30.0, 30.01], [9.0, 1.0, 30.01, 64.0]]))
        (x, _y, w, _h), = rects_of(marks.strips)
        self.assertGreater(w, 0.01)
        self.assertAlmostEqual(x + w / 2.0, 30.005)


if __name__ == "__main__":
    unittest.main()
