"""The deck's and its mixer channel's switches, the toolbar's CH 1 row, as a DJM's and a CDJ's are: the jog wheel's
mode (VINYL: its top holds the record, CDJ: a bend only), the EQ (ISO: an isolator, a band down to a kill; EQ:
-26 dB at the bottom), the channel fader's curve (each drawn: the app's default, steep: up fast, slow: most of the
rise at the top), Sound Color FX (the effect, the channel's COLOR knob, PARAMETER) and the key shift (a knob,
continuous; − / its value / +, a semitone at once); ``MasterLevels`` the MASTER row's level. A click or a turn sends a
request (sig_jog_mode_requested, sig_mixer_requested, sig_knob_requested, sig_key_shift_requested, sig_deck_action);
what is lit and where a knob points is what the deck reports (DeckReport: its lamps' VINYL bit, color, key_shift, the
mixer's values), so a controller's switch or knob shows too. The native engine's (the Python engine has none of them):
disabled there. The controls' looks: ui/deck/deck_controls.py."""
from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

from audioengine import controls
from core.workflows.event_bus import EventBus
from system.i18n import tr
from ui.deck.deck_controls import KNOB, PADDING, Segment, block, blocks_row, label, segments, text_room
from ui.widgets import tooltips
from ui.widgets.knob import Knob
from ui.theme import current_theme

GLYPH_W, GLYPH_H = 26, 14
COLOR_KNOB = 4   # the deck's channel knob for its COLOR (player.set_knob)
KEY_STEP_W = 26  # the key's − and +: they stay this wide, the value between them takes the room
KEY_VALUE_SPARE = 14    # the value at its widest: its reading with the unit and this much more
KEY_UNIT = " SEMITONE"


def key_shift_text(semitones: float) -> str:
    """The key shift as its value shows it: ±0.00, +2.00, −1.25 (U+2212 minus, as wide as "+")."""
    s = round(float(semitones), 2)
    return f"{s:+.2f}".replace("-", "−") if s else "±0.00"


class _KeyValue(Segment):
    """The shift's reading on its button between − and +, with its unit once the button has the room for it. Its text
    changes as the knob turns, so it is painted, never set: a button's setText asks for a layout, which climbs to the
    window, unless its size is fixed."""

    def __init__(self) -> None:
        super().__init__()
        self.shown = key_shift_text(0.0)
        widest = key_shift_text(0.12 - controls.KEY_SHIFT_MAX)
        self.setMinimumWidth(text_room(self, widest))
        self.width_with_unit = text_room(self, widest + KEY_UNIT)

    def show_text(self, text: str) -> None:
        if text != self.shown:
            self.shown = text
            self.update()

    def reading(self) -> str:
        return self.shown + KEY_UNIT if self.width() >= self.width_with_unit else self.shown

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        option = QtWidgets.QStyleOptionToolButton()
        self.initStyleOption(option)
        option.text = self.reading()
        painter = QtWidgets.QStylePainter(self)
        painter.drawComplexControl(QtWidgets.QStyle.CC_ToolButton, option)


def _curve_icon(curve: int, color: str) -> QtGui.QIcon:
    """The channel fader's curve ``curve`` (FADER_CURVES' place) drawn: its gain over its travel."""
    ratio = 2.0
    pix = QtGui.QPixmap(int(GLYPH_W * ratio), int(GLYPH_H * ratio))
    pix.setDevicePixelRatio(ratio)
    pix.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setPen(QtGui.QPen(QtGui.QColor(color), 1.6, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
    path = QtGui.QPainterPath()
    for k in range(41):
        x = k / 40
        if curve == 1:
            y = 1 - (1 - x) ** 3
        elif curve == 2:
            y = x ** 3
        else:   # the app's default (audioengine/volume.py; src/mixer.rs fader_gain)
            v = x ** 0.4
            y = 1.0 if v > 0.99 else -math.log(1 - v) / math.log(100)
        point = QtCore.QPointF(2 + x * (GLYPH_W - 4), GLYPH_H - 2 - y * (GLYPH_H - 4))
        path.moveTo(point) if k == 0 else path.lineTo(point)
    p.drawPath(path)
    p.end()
    return QtGui.QIcon(pix)


class DeckSwitches(QtWidgets.QWidget):
    """The CH 1 row's switches (see the module). ``set_report`` shows the deck's report; ``set_available`` on the
    native engine."""

    def __init__(self, bus: EventBus, parent=None) -> None:
        super().__init__(parent)
        self.bus = bus
        self._report = None
        self._key_shift = 0.0

        self.vinyl, self.cdj = Segment("VINYL"), Segment("CDJ")
        self.vinyl.setToolTip(tr("JOG MODE VINYL: touching the jog wheel's top holds the record (scratch)."))
        self.cdj.setToolTip(tr("JOG MODE CDJ: the jog wheel bends the tempo only, its top's touch ignored."))
        self.vinyl.clicked.connect(lambda: self.bus.sig_jog_mode_requested.emit(True))
        self.cdj.clicked.connect(lambda: self.bus.sig_jog_mode_requested.emit(False))

        self.eq = [Segment("ISO"), Segment("EQ")]
        self.eq[0].setToolTip(tr("EQ CURVE ISO: an isolator, each band from a kill to +6 dB."))
        self.eq[1].setToolTip(tr("EQ CURVE EQ: an equalizer, each band from -26 dB to +6 dB."))
        for k, b in enumerate(self.eq):
            b.clicked.connect(lambda _c=False, k=k: self.bus.sig_mixer_requested.emit("eq_mode", float(k)))

        self.fader = []
        tips = (tr("CH FADER curve: the app's (slow at first, full at the top)."),
                tr("CH FADER curve: steep (up fast)."),
                tr("CH FADER curve: slow (most of the rise at the top)."))
        for k, tip in enumerate(tips):
            b = Segment()
            b.setIcon(_curve_icon(k, current_theme().text))
            b.setIconSize(QtCore.QSize(GLYPH_W, GLYPH_H))
            b.setMinimumWidth(GLYPH_W + 2 * PADDING + 2)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, k=k: self.bus.sig_mixer_requested.emit("fader_curve", float(k)))
            self.fader.append(b)

        # Sound Color FX: the effect, the channel's COLOR knob, PARAMETER
        self.color_type = QtWidgets.QComboBox()
        for code, name in enumerate(controls.COLOR_TYPES):
            self.color_type.addItem(name, code)
        self.color_type.addItem("OFF", controls.COLOR_OFF)
        self.color_type.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.color_type.setToolTip(tr("Sound Color FX: the effect the COLOR knob turns on (OFF: none)."))
        self.color_type.activated.connect(lambda _i: self._mix("color_type", self.color_type.currentData()))
        self.color = Knob(KNOB)
        self.color.setToolTip(tr("COLOR: the effect on the deck's channel, its two flavours left and right of the "
                                 "middle (off). Double-click: off."))
        self.color.moved.connect(lambda v: self.bus.sig_knob_requested.emit(COLOR_KNOB, v))
        self.param = Knob(KNOB)
        self.param.setToolTip(tr("PARAMETER: the effect's quantity."))
        self.param.moved.connect(lambda v: self._mix("color_param", v))

        # the key shift: the knob continuous, 2 cents a pixel dragged (the whole range 600 px), the wheel a cent
        range_cents = 200.0 * controls.KEY_SHIFT_MAX
        self.key_knob = Knob(KNOB, drag_px=range_cents / 2.0, wheel_step=1.0 / range_cents)
        self.key_knob.setToolTip(tr("Key shift, continuous (the tempo kept). The wheel moves it a cent; a double-click: "
                                    "the track's own key."))
        self.key_knob.moved.connect(self._on_key_knob)
        # − / the shift (lit while shifted, a click back to the track's own key) / +: a semitone at once
        self.key_down, self.key_value, self.key_up = Segment("−"), _KeyValue(), Segment("+")
        self.key_down.setToolTip(tr("Key a semitone down (the tempo kept)"))
        self.key_value.setToolTip(tr("Key shift in semitones (±{max}); click: the track's own key")
                                  .format(max=controls.KEY_SHIFT_MAX))
        self.key_up.setToolTip(tr("Key a semitone up (the tempo kept)"))
        self.key_down.clicked.connect(lambda: self._key_action(controls.ACT_KEY_DOWN))
        self.key_value.clicked.connect(lambda: self._key_action(controls.ACT_KEY_RESET))
        self.key_up.clicked.connect(lambda: self._key_action(controls.ACT_KEY_UP))
        key_buttons = segments([self.key_down, self.key_value, self.key_up])
        for step in (self.key_down, self.key_up):
            step.setFixedWidth(KEY_STEP_W)
        # one control in the row's sharing, and only so wide: a long − value + is no look
        key_buttons.setProperty("controls", 1)
        key_buttons.setMaximumWidth(2 * KEY_STEP_W + self.key_value.width_with_unit + KEY_VALUE_SPARE)

        key_block = block("KEY", [self.key_knob, key_buttons])
        # its frame stops with them, so it is never left empty
        key_block.setMaximumWidth(key_block.sizeHint().width() - key_buttons.sizeHint().width() + key_buttons.maximumWidth())
        blocks_row(self, [
            block("JOG MODE", [segments([self.vinyl, self.cdj])]),
            block("EQ CURVE", [segments(self.eq)]),
            block("CH FADER", [segments(self.fader)]),
            block("COLOR FX", [self.color_type, 6, label("COLOR"), self.color, 6, label("PARAMETER"), self.param]),
            key_block,
        ])

    def _mix(self, key: str, value) -> None:
        if value is not None:
            self.bus.sig_mixer_requested.emit(key, float(value))

    def _on_key_knob(self, position: float) -> None:
        """The key knob turned: the shift in whole cents (a request: it shows what the deck then reports)."""
        self.bus.sig_key_shift_requested.emit(round((position - 0.5) * 2.0 * controls.KEY_SHIFT_MAX, 2))

    def _key_action(self, code: int) -> None:
        if self.isEnabled():
            self.bus.sig_deck_action.emit(int(code), True)

    def step_color_type(self, by: int) -> None:
        """The next / previous Sound Color FX, OFF among them."""
        codes = list(range(len(controls.COLOR_TYPES))) + [controls.COLOR_OFF]
        report = self._report
        now = int(report.mix_value("color_type", 5.0)) if report is not None and report.mix else 5
        at = codes.index(now) if now in codes else 0
        self._mix("color_type", codes[(at + by) % len(codes)])

    def shortcut_handlers(self) -> dict:
        # each button's tooltip names its keys (ui.widgets.tooltips)
        for button, action in ((self.key_down, "playback.key_down"), (self.key_up, "playback.key_up"),
                               (self.key_value, "playback.key_reset")):
            tooltips.tag(button, action)
        return {"fx.color_next": lambda: self.step_color_type(1),
                "fx.color_previous": lambda: self.step_color_type(-1),
                **{action: (lambda code=code: self._key_action(code))
                   for action, code in (("playback.key_down", controls.ACT_KEY_DOWN),
                                        ("playback.key_up", controls.ACT_KEY_UP),
                                        ("playback.key_reset", controls.ACT_KEY_RESET),
                                        ("playback.key_fine_down", controls.ACT_KEY_FINE_DOWN),
                                        ("playback.key_fine_up", controls.ACT_KEY_FINE_UP),
                                        *((f"playback.key_set.{s:+d}", controls.ACT_KEY_SET + controls.KEY_SHIFT_MAX + s)
                                          for s in range(-controls.KEY_SHIFT_MAX, controls.KEY_SHIFT_MAX + 1) if s))}}

    def set_available(self, available: bool) -> None:
        self.setEnabled(available)
        if not available:
            self.setToolTip(tr("Available with the native audio engine only."))

    def set_report(self, report) -> None:
        """The deck's report (DeckReport): the jog's mode, the mixer's EQ, Sound Color FX and fader curve, the key
        shift."""
        if not report.mix:
            return
        self._report = report
        if report.key_shift != self._key_shift:
            self._key_shift = report.key_shift
            self.key_value.show_text(key_shift_text(report.key_shift))
            self.key_value.set_lit(bool(report.key_shift))
            self.key_knob.set_value(0.5 + report.key_shift / (2.0 * controls.KEY_SHIFT_MAX))
        if not self.color_type.view().isVisible():   # open: the hand is on it
            k = self.color_type.findData(int(report.mix_value("color_type")))
            if k >= 0 and k != self.color_type.currentIndex():
                self.color_type.setCurrentIndex(k)
        self.color.set_value(report.color)
        self.param.set_value(report.mix_value("color_param", 0.5))
        self.vinyl.set_lit(report.vinyl)
        self.cdj.set_lit(not report.vinyl)
        eq = int(report.mix_value("eq_mode"))
        for k, b in enumerate(self.eq):
            b.set_lit(k == eq)
        curve = int(report.mix_value("fader_curve"))
        for k, b in enumerate(self.fader):
            b.set_lit(k == curve)


class MasterLevels(QtWidgets.QWidget):
    """The MASTER row's level: the master's knob, its 0 dB at controls.LEVEL_UNITY."""

    def __init__(self, bus: EventBus, parent=None) -> None:
        super().__init__(parent)
        self.bus = bus
        self.master = Knob(KNOB, neutral=controls.LEVEL_UNITY)
        self.master.moved.connect(lambda v: self.bus.sig_mixer_requested.emit("master_level", v))
        blocks_row(self, [block("LEVEL", [label("MASTER"), self.master])])
        self._show_level(self.master, tr("Master level"), controls.LEVEL_UNITY)

    @staticmethod
    def _show_level(knob: Knob, title: str, value: float) -> None:
        knob.set_value(value)
        db = controls.level_knob_to_db(knob.value())
        text = tr("-inf dB") if db <= controls.LEVEL_DB_MIN else f"{db:+.1f} dB"
        knob.setToolTip(tr("{name}: {level} (double-click: 0 dB)").format(name=title, level=text))

    def set_available(self, available: bool) -> None:
        self.setEnabled(available)

    def set_report(self, report) -> None:
        if not report.mix:
            return
        value = report.mix_value("master_level", controls.LEVEL_UNITY)
        if abs(value - self.master.value()) > 1e-6:
            self._show_level(self.master, tr("Master level"), value)
