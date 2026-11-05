"""The toolbar's KEY block, right of COLOR FX: the shift's reading, the knob's scale (a request in whole cents), the
semitone buttons and keys (requests), what the deck reports shown, and the key sent to the views moved by the nearest
semitone of the fraction."""
import unittest

from unit.ui_tests.qt_env import APP  # noqa: F401  (the application, offscreen)

from PySide6 import QtCore, QtGui, QtWidgets

from audioengine import controls
from core.workflows.event_bus import EventBus
from ui import pane
from ui.deck import deck_controls, deck_switches
from ui.deck.deck_switches import DeckSwitches, key_shift_text


def _report(key_shift: float = 0.0) -> controls.DeckReport:
    return controls.DeckReport(False, 1.0, 1.0, 0.0, 0.0, 0, 100, 0, mix=[0.0] * len(controls.MIX_KEYS), key_shift=key_shift)


def _mouse(kind, y: float) -> QtGui.QMouseEvent:
    point = QtCore.QPointF(10.0, y)
    return QtGui.QMouseEvent(kind, point, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier)


class _Case(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.s = DeckSwitches(self.bus)
        self.requests: list[float] = []
        self.actions: list[tuple] = []
        self.bus.sig_key_shift_requested.connect(self.requests.append)
        self.bus.sig_deck_action.connect(lambda code, pressed: self.actions.append((code, pressed)))


class ReadingTests(unittest.TestCase):
    def test_always_two_decimals(self):
        for shift, text in ((0.0, "±0.00"), (2.0, "+2.00"), (-3.0, "−3.00"), (1.25, "+1.25"), (-5.88, "−5.88"), (0.05, "+0.05"),
                            (0.004, "±0.00"), (2.0000001, "+2.00")):
            self.assertEqual(key_shift_text(shift), text, shift)


class PlaceTests(_Case):
    def test_the_block_right_of_color_fx(self):
        row = self.s.layout()
        titles = []
        for k in range(row.count()):
            frame = row.itemAt(k).widget()
            titles.append(frame.findChildren(QtWidgets.QLabel)[0].text())
        self.assertEqual(titles, ["JOG MODE", "EQ CURVE", "CH FADER", "COLOR FX", "KEY", "PHONES"])

    def test_the_values_text_asks_for_no_layout(self):
        requests = []

        class Watch(QtCore.QObject):
            def eventFilter(self, _watched, event):
                if event.type() == QtCore.QEvent.LayoutRequest:
                    requests.append(event)
                return False

        v = self.s.key_value
        self.s.show()
        QtWidgets.QApplication.processEvents()
        watch = Watch()
        parent = v.parentWidget()
        parent.installEventFilter(watch)
        requests.clear()
        for k in range(20):
            v.show_text(key_shift_text(k / 100.0))
            QtWidgets.QApplication.processEvents()
        parent.removeEventFilter(watch)
        self.s.close()
        self.assertEqual(requests, [])
        widest = v.fontMetrics().horizontalAdvance(key_shift_text(0.12 - controls.KEY_SHIFT_MAX))
        self.assertGreaterEqual(v.minimumWidth(), widest)


class WideWindowTests(_Case):
    """A full screen's row: every frame filled by what it holds, each control about as wide as the others, the row filled
    end to end, a block of knobs alone as wide as it is; nothing cut in a narrow one."""

    def _laid_out(self, width: int) -> None:
        self.s.resize(width, self.s.sizeHint().height())
        self.s.show()
        QtWidgets.QApplication.processEvents()

    def tearDown(self):
        self.s.close()
        self.s.deleteLater()

    def test_every_frame_is_filled_by_what_it_holds(self):
        for width in (1200, 1800, 2400):
            self._laid_out(width)
            row = self.s.layout()
            for k in range(row.count()):
                frame = row.itemAt(k).widget()
                inside = frame.childrenRect()
                self.assertLessEqual(inside.left(), 8, (width, k))
                self.assertGreaterEqual(inside.right(), frame.width() - 8, (width, k))

    def test_the_row_is_filled_end_to_end(self):
        self._laid_out(2400)
        row = self.s.layout()
        frames = [row.itemAt(k).widget() for k in range(row.count())]
        self.assertEqual(frames[0].x(), 0)
        self.assertEqual(frames[-1].geometry().right(), self.s.width() - 1)
        for left, right in zip(frames, frames[1:]):
            self.assertEqual(right.x() - left.geometry().right() - 1, row.spacing(), "the frames side by side")

    def test_each_control_about_as_wide_as_the_others(self):
        self._laid_out(2400)
        widths = [b.width() for b in (self.s.vinyl, self.s.cdj, *self.s.eq, *self.s.fader, self.s.cue)]
        self.assertLess(max(widths) / min(widths), 1.4, widths)
        self.assertLess(self.s.color_type.width(), 2.2 * min(widths), "the combo box no giant")

    def test_the_keys_steps_stay_narrow_the_value_takes_the_room_up_to_its_most(self):
        v = self.s.key_value
        for width in (1300, 1800, 2400, 3200):
            self._laid_out(width)
            self.assertEqual(self.s.key_down.width(), deck_switches.KEY_STEP_W, width)
            self.assertEqual(self.s.key_up.width(), deck_switches.KEY_STEP_W, width)
        self.assertLessEqual(v.width(), v.width_with_unit + deck_switches.KEY_VALUE_SPARE, "the three stop widening")

    def test_the_value_shows_its_unit_once_it_has_the_room(self):
        v = self.s.key_value
        self._laid_out(self.s.minimumSizeHint().width())
        self.assertEqual(v.reading(), "±0.00", "narrow: the number alone")
        self._laid_out(2400)
        self.assertEqual(v.reading(), "±0.00 SEMITONE")
        self.assertGreaterEqual(v.width(), deck_controls.text_room(v, v.reading()), "and it fits")

    def test_a_block_of_knobs_alone_keeps_its_width(self):
        holder = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(holder)
        switches = DeckSwitches(self.bus)
        levels = deck_switches.MasterLevels(self.bus)
        deck_controls.fill(row, [switches, levels])
        holder.resize(2400, 40)
        holder.show()
        QtWidgets.QApplication.processEvents()
        self.assertEqual(levels.width(), levels.sizeHint().width())
        self.assertGreater(switches.width(), 2000)
        holder.close()

    def test_the_keys_buttons_stay_joined(self):
        for width in (1100, 1600, 2400):
            self._laid_out(width)
            self.assertEqual(self.s.key_down.geometry().right() + 1, self.s.key_value.x(), width)
            self.assertEqual(self.s.key_value.geometry().right() + 1, self.s.key_up.x(), width)

    def test_no_text_cut_however_narrow(self):
        for width in (self.s.minimumSizeHint().width(), self.s.sizeHint().width(), 1400):
            self._laid_out(width)
            for button in (self.s.vinyl, self.s.cdj, *self.s.eq, self.s.key_down, self.s.key_up, self.s.cue):
                self.assertGreaterEqual(button.width(), deck_controls.text_room(button, button.text()), (width, button.text()))
            self.assertGreaterEqual(self.s.key_value.width(),
                                    deck_controls.text_room(self.s.key_value, key_shift_text(0.12 - controls.KEY_SHIFT_MAX)), width)


class KnobTests(_Case):
    def test_the_wheel_a_cent(self):
        event = QtGui.QWheelEvent(QtCore.QPointF(5, 5), QtCore.QPointF(5, 5), QtCore.QPoint(0, 120), QtCore.QPoint(0, 120),
                                  QtCore.Qt.NoButton, QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)
        self.s.key_knob.wheelEvent(event)
        self.s.key_knob.wheelEvent(event)
        self.assertEqual(self.requests, [0.01, 0.02])

    def test_a_drag_2_cents_a_pixel_through_0_and_the_whole_range_in_reach(self):
        knob = self.s.key_knob
        knob.mousePressEvent(_mouse(QtCore.QEvent.MouseButtonPress, 100.0))
        knob.mouseMoveEvent(_mouse(QtCore.QEvent.MouseMove, 90.0))      # 10 px up
        self.assertEqual(self.requests[-1], 0.2)
        knob.mouseMoveEvent(_mouse(QtCore.QEvent.MouseMove, 115.0))     # 15 px down of where it began
        self.assertEqual(self.requests[-1], -0.3)
        knob.mouseMoveEvent(_mouse(QtCore.QEvent.MouseMove, 100.0 - 300.0))
        self.assertEqual(self.requests[-1], float(controls.KEY_SHIFT_MAX))
        knob.mouseMoveEvent(_mouse(QtCore.QEvent.MouseMove, 100.0 + 400.0))
        self.assertEqual(self.requests[-1], float(-controls.KEY_SHIFT_MAX))
        knob.mouseReleaseEvent(_mouse(QtCore.QEvent.MouseButtonRelease, 0.0))

    def test_a_double_click_the_tracks_own_key(self):
        self.s.key_knob.set_value(0.75)
        self.s.key_knob.mouseDoubleClickEvent(_mouse(QtCore.QEvent.MouseButtonDblClick, 5.0))
        self.assertEqual(self.requests, [0.0])


class RequestTests(_Case):
    def test_the_buttons_are_requests(self):
        self.s.key_down.click()
        self.s.key_up.click()
        self.s.key_value.click()
        self.assertEqual(self.actions, [(controls.ACT_KEY_DOWN, True), (controls.ACT_KEY_UP, True),
                                        (controls.ACT_KEY_RESET, True)])
        self.assertEqual(self.s.key_value.shown, "±0.00", "shown as the deck reports it, not before")

    def test_the_keys(self):
        handlers = self.s.shortcut_handlers()
        for action, code in (("playback.key_down", controls.ACT_KEY_DOWN), ("playback.key_up", controls.ACT_KEY_UP),
                             ("playback.key_reset", controls.ACT_KEY_RESET),
                             ("playback.key_fine_down", controls.ACT_KEY_FINE_DOWN),
                             ("playback.key_fine_up", controls.ACT_KEY_FINE_UP),
                             ("playback.key_set.+3", controls.ACT_KEY_SET + controls.KEY_SHIFT_MAX + 3),
                             ("playback.key_set.-6", controls.ACT_KEY_SET)):
            self.actions.clear()
            handlers[action]()
            self.assertEqual(self.actions, [(code, True)], action)

    def test_off_on_the_python_engine_the_keys_too(self):
        handlers = self.s.shortcut_handlers()
        self.s.set_available(False)
        handlers["playback.key_up"]()
        self.s.key_up.click()
        self.assertEqual(self.actions, [])


class ReportTests(_Case):
    def test_the_report_shown_lit_while_shifted(self):
        self.s.set_report(_report(1.25))
        self.assertEqual(self.s.key_value.shown, "+1.25")
        self.assertTrue(self.s.key_value.property("lit"))
        self.assertAlmostEqual(self.s.key_knob.value(), 0.5 + 1.25 / 12.0)
        self.s.set_report(_report(0.0))
        self.assertEqual(self.s.key_value.shown, "±0.00")
        self.assertFalse(self.s.key_value.property("lit"))
        self.assertAlmostEqual(self.s.key_knob.value(), 0.5)
        self.assertEqual(self.requests, [], "a report asks for nothing")

    def test_the_hand_on_the_knob_the_reading_follows_the_deck(self):
        knob = self.s.key_knob
        knob.mousePressEvent(_mouse(QtCore.QEvent.MouseButtonPress, 100.0))
        knob.mouseMoveEvent(_mouse(QtCore.QEvent.MouseMove, 50.0))      # +1
        held = knob.value()
        self.s.set_report(_report(0.5))     # the deck not there yet
        self.assertEqual(self.s.key_value.shown, "+0.50")
        self.assertEqual(knob.value(), held, "the hand's position")
        knob.mouseReleaseEvent(_mouse(QtCore.QEvent.MouseButtonRelease, 50.0))


class _Played(QtWidgets.QWidget):
    """MainPane's send_played and what it reads."""

    send_played = pane.MainPane.send_played

    def __init__(self, master_tempo: bool = True) -> None:
        super().__init__()
        self.bus = EventBus()
        self._key_shift, self._tempo_factor, self._played = 0.0, 1.0, None
        self._master_tempo = master_tempo
        self.played: list[tuple] = []
        self.bus.sig_played_changed.connect(lambda speed, semitones: self.played.append((speed, semitones)))

    def _timestretch_enabled(self) -> bool:
        return self._master_tempo


class PlayedTests(unittest.TestCase):
    def test_a_fraction_moves_the_key_by_the_nearest_semitone(self):
        c = _Played()
        for shift, semitones in ((0.25, 0), (0.75, 1), (1.25, 1), (-1.75, -2)):
            c._key_shift = shift
            c.send_played(force=True)
            self.assertEqual(c.played[-1], (1.0, semitones), shift)

    def test_mt_off_the_tempos_pitch_and_the_key_rounded_together(self):
        c = _Played(master_tempo=False)
        c._tempo_factor = 2 ** (0.6 / 12)       # 0.6 semitones up by the tempo, 0.6 by the key: 1.2, not 2
        c._key_shift = 0.6
        c.send_played(force=True)
        self.assertEqual(c.played[-1][1], 1)


if __name__ == "__main__":
    unittest.main()
